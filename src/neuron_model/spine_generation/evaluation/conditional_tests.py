"""Statistical checks of a CONDITIONAL spine generator against real spines (units: the metrics of stage 8, nm / nm^2 / ...).

Two kinds of samples with different dependence structure:

- REAL spines of one neuron are dependent (shared segmentation, cell size, local conditions) - the NEURON is the unit of
  inference, as in ``morphology_by_condition.py``: tests between neuron groups permute neurons; the noise floor of a
  "real vs real" comparison is built from two DISJOINT random halves of the real neurons;
- GENERATED spines are independent draws of one model: ordinary two-sample / bootstrap procedures apply.

Tests (each function documents its null hypothesis, statistic, reference distribution and unit):

A. :func:`type_effect_real`, :func:`type_effect_generated`, :func:`concordance` - is the DIFFERENCE between two types
   (e.g. inhibitory / excitatory) reproduced? Per metric: the real log ratio of pooled medians with a NEURON bootstrap
   interval and the neuron-level permutation test; the generated log ratio with an iid bootstrap interval and a
   two-sample permutation test; the status of each metric (reproduced / consistent / underpowered / ...).
B. :class:`SameTypeTester` - is a generated set of type v indistinguishable from real spines of type v? Statistic W1/sd
   per metric and the energy distance of the standardised log vector; reference = the same statistic between real
   samples drawn from two disjoint halves of the real neurons (the noise floor with the neuron as the unit); TOST
   equivalence of the median with a stated margin; Spearman-correlation structure.
C. :func:`fit_condition_classifier`, :func:`apply_condition_classifier` - a classifier of the condition trained on
   real spines (neuron-grouped cross-validation): does the generated set of condition v look like v to it?
   (+ :func:`rogan_gladen` - the effective share of the condition in an unconditional set).
D. :func:`real_vs_generated_auc` - a classifier two-sample test, neuron-grouped, reference = real half vs real half.
E. :func:`dose_response` - the generated effect as a function of the guidance weight.
F. :func:`validity_table` - rates of meshes, bases found, valid meshes, complete metric vectors.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from ...spine_morphometrics import SCALAR_METRICS
from .morphology_by_condition import bh_qvalues, cliffs_delta, compare_neuron_groups, neuron_medians

# the metrics computed WITHOUT the attachment region of the generated mesh: unaffected by the bias of the base finder
BASE_INDEPENDENT = ("Volume", "ConvexHullVolume", "ConvexHullRatio")
Z_ALPHA = 1.959964          # two-sided 5 %
Z_POWER = 0.841621          # power 80 %
TOST_MARGIN_SD = 0.2        # equivalence margin: 0.2 pooled sd of the log metric (a "small" standardised effect)


def _norm_cdf(x):
    from scipy.stats import norm

    return norm.cdf(x)


def _ci(values: np.ndarray, level: float = 0.95) -> Tuple[float, float]:
    lo, hi = np.nanpercentile(values, [50 * (1 - level), 100 - 50 * (1 - level)])
    return float(lo), float(hi)


def wilson(k: int, n: int, z: float = Z_ALPHA) -> Tuple[float, float, float]:
    """(share, lower, upper) of a binomial proportion (Wilson score interval)."""
    if n <= 0:
        return float("nan"), float("nan"), float("nan")
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return p, max(0.0, centre - half), min(1.0, centre + half)


def parse_weight(label: str) -> float:
    """Guidance weight from a set label (``g1.25`` -> 1.25); NaN if the label has no number."""
    m = re.search(r"(\d+(?:\.\d+)?)", str(label))
    return float(m.group(1)) if m else float("nan")


# ---------------------------------------------------------------------------
# real spines grouped by neuron
# ---------------------------------------------------------------------------


class RealPool:
    """Real spines of one population (rows = spines with positive finite metrics), indexed by neuron for fast draws.

    Neurons with fewer than ``min_spines`` spines are left out (as in the neuron-level tests). ``draw`` picks a neuron
    (``weighting="equal"``: every neuron equally likely - balanced per-neuron draws, no neuron can dominate the sample;
    ``"spines"``: proportional to its number of spines = a uniform draw from the pooled spines) and then a spine of it.
    """

    def __init__(self, frame: pd.DataFrame, metrics: Sequence[str], *, min_spines: int = 20, neuron_col: str = "neuron_id"):
        self.metrics = list(metrics)
        vals = frame[self.metrics].to_numpy(float)
        ok = np.isfinite(vals).all(axis=1) & (vals > 0).all(axis=1)
        self.n_dropped = int((~ok).sum())
        f = frame.loc[ok, [neuron_col]].reset_index(drop=True)
        vals = vals[ok]
        codes, uniques = pd.factorize(f[neuron_col])
        counts = np.bincount(codes)
        keep = counts[codes] >= min_spines
        codes, vals = codes[keep], vals[keep]
        codes, uniques = pd.factorize(codes)               # recode after the filter
        order = np.argsort(codes, kind="stable")
        self.values = vals[order]
        self.nid = codes[order]
        self.n_neurons = int(len(uniques))
        self.counts = np.bincount(self.nid, minlength=self.n_neurons)
        self.offsets = np.concatenate([[0], np.cumsum(self.counts)[:-1]])
        self.n_spines = int(len(self.values))
        self.sd = self.values.std(axis=0, ddof=1) if self.n_spines > 1 else np.ones(len(self.metrics))
        logs = np.log(self.values)
        self.log_mean = logs.mean(axis=0)
        self.log_sd = logs.std(axis=0, ddof=1) if self.n_spines > 1 else np.ones(len(self.metrics))

    def draw(self, n: int, rng: np.random.Generator, neurons: Optional[np.ndarray] = None, weighting: str = "equal") -> Tuple[np.ndarray, np.ndarray]:
        """``(rows [n, M], neuron index [n])``."""
        cand = np.arange(self.n_neurons) if neurons is None else np.asarray(neurons)
        if weighting == "equal":
            nid = cand[rng.integers(0, len(cand), n)]
        elif weighting == "spines":
            c = self.counts[cand]
            nid = cand[rng.choice(len(cand), size=n, p=c / c.sum())]
        else:
            raise ValueError(f"weighting must be equal or spines, got {weighting!r}")
        pos = self.offsets[nid] + (rng.random(n) * self.counts[nid]).astype(np.int64)
        return self.values[pos], nid

    def halves(self, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
        """Two DISJOINT random halves of the neurons."""
        perm = rng.permutation(self.n_neurons)
        h = self.n_neurons // 2
        return perm[:h], perm[h:2 * h]

    def standardise(self, rows: np.ndarray) -> np.ndarray:
        return (np.log(rows) - self.log_mean) / self.log_sd


def complete_rows(gen: pd.DataFrame, metrics: Sequence[str]) -> np.ndarray:
    """Generated samples with all metrics finite and positive (the multivariate checks need whole vectors)."""
    x = gen[list(metrics)].to_numpy(float)
    return x[np.isfinite(x).all(axis=1) & (x > 0).all(axis=1)]


# ---------------------------------------------------------------------------
# A. difference between two types: real (neuron = unit) and generated (iid)
# ---------------------------------------------------------------------------


def cluster_median_bootstrap(values: np.ndarray, nid: np.ndarray, n_boot: int, rng: np.random.Generator) -> Tuple[float, np.ndarray]:
    """Pooled median of the spines and ``n_boot`` replicates in which NEURONS are resampled with replacement
    (all spines of a drawn neuron come along): the interval of the pooled median under neuron-level dependence."""
    ok = np.isfinite(values)
    v, g = values[ok], nid[ok]
    _, g = np.unique(g, return_inverse=True)
    k = int(g.max()) + 1
    order = np.argsort(v, kind="stable")
    vs, gs = v[order], g[order]
    out = np.empty(n_boot)
    for i in range(n_boot):
        counts = np.bincount(rng.integers(0, k, k), minlength=k)
        cs = np.cumsum(counts[gs])
        out[i] = vs[min(int(np.searchsorted(cs, cs[-1] / 2.0)), len(vs) - 1)]
    return float(np.median(v)), out


def iid_median_bootstrap(values: np.ndarray, n_boot: int, rng: np.random.Generator, chunk: int = 500) -> np.ndarray:
    v = values[np.isfinite(values)]
    out = np.empty(n_boot)
    for s in range(0, n_boot, chunk):
        m = min(chunk, n_boot - s)
        out[s:s + m] = np.median(v[rng.integers(0, len(v), (m, len(v)))], axis=1)
    return out


def type_effect_real(frame: pd.DataFrame, block: str, a: str, b: str, metrics: Sequence[str], *, min_spines: int = 20,
                     n_permutations: int = 10000, n_boot: int = 1000, seed: int = 0) -> Tuple[pd.DataFrame, Dict[str, np.ndarray]]:
    """Real difference of type ``a`` vs ``b`` (``frame``: spines with ``neuron_id``, ``block`` column, metrics).

    H0: the neuron-level medians of the metric have the same distribution in both types. Statistic: difference of the
    medians of the neuron medians; reference: labels permuted between neurons (``morphology_by_condition``). q = BH over
    the metrics. Effect size: log ratio ``a / b`` of the POOLED spine medians (the quantity comparable with a generated
    sample) with the 95 % interval of a neuron bootstrap (neurons resampled within each type), and Cliff's delta of the
    neuron medians. Returns the table and the bootstrap replicates of the log ratio per metric."""
    sub = frame[frame[block].isin([a, b])]
    info = sub.groupby("neuron_id")[[block]].first()
    med = neuron_medians(sub, metrics, ["neuron_id"], min_spines).merge(info, on="neuron_id")
    table = pd.DataFrame(compare_neuron_groups(med, block, a, b, metrics, n_permutations=n_permutations, seed=seed))
    if "p" not in table.columns:
        table["p"] = np.nan
    table["q_bh"] = bh_qvalues(table["p"].to_numpy(float))
    table["significant"] = table["q_bh"] < 0.05
    rng = np.random.default_rng(seed + 7)
    keep = sub.merge(med[["neuron_id"]], on="neuron_id")
    boots: Dict[str, np.ndarray] = {}
    rows = []
    for m in metrics:
        lr = np.nan
        lo = hi = np.nan
        pa = pb = np.nan
        parts = []
        for t in (a, b):
            g = keep[keep[block] == t]
            codes = pd.factorize(g["neuron_id"])[0]
            parts.append(cluster_median_bootstrap(g[m].to_numpy(float), codes, n_boot, rng) if len(g) else (np.nan, np.full(n_boot, np.nan)))
        (pa, ba), (pb, bb) = parts
        if np.isfinite(pa) and np.isfinite(pb) and pa > 0 and pb > 0:
            with np.errstate(invalid="ignore", divide="ignore"):
                rep = np.log(ba) - np.log(bb)
            lr = float(np.log(pa / pb))
            lo, hi = _ci(rep)
            boots[m] = rep
        rows.append({"metric": m, "pooled_median_a": pa, "pooled_median_b": pb, "log_ratio": lr, "lr_lo": lo, "lr_hi": hi})
    table = table.merge(pd.DataFrame(rows), on="metric", how="left")
    return table, boots


def _perm_median_diff_p(x: np.ndarray, y: np.ndarray, n_permutations: int, rng: np.random.Generator, chunk: int = 2000) -> float:
    """Two-sided permutation p of |median(x) - median(y)| (labels permuted; the samples are independent draws)."""
    z = np.concatenate([x, y])
    n = len(x)
    obs = abs(np.median(x) - np.median(y))
    exceed = 0
    done = 0
    while done < n_permutations:
        m = min(chunk, n_permutations - done)
        idx = np.argsort(rng.random((m, len(z))), axis=1)
        zs = z[idx]
        null = np.abs(np.median(zs[:, :n], axis=1) - np.median(zs[:, n:], axis=1))
        exceed += int((null >= obs - 1e-15 * max(1.0, abs(obs))).sum())
        done += m
    return (1 + exceed) / (1 + n_permutations)


def type_effect_generated(ga: pd.DataFrame, gb: pd.DataFrame, metrics: Sequence[str], *, n_permutations: int = 10000, n_boot: int = 1000,
                          seed: int = 0) -> Tuple[pd.DataFrame, Dict[str, np.ndarray]]:
    """Generated set ``ga`` (condition a) vs ``gb`` (condition b); independent samples.

    H0: both sets come from one distribution. Statistic: difference of medians; reference: labels permuted between the
    generated samples (``n_permutations``); q = BH over the metrics; effect: log ratio of medians with a 95 % bootstrap
    interval, Cliff's delta. ``se`` = bootstrap sd of the log ratio, ``mde`` = minimal detectable |log ratio| at
    power 80 % and two-sided alpha 5 % (uncorrected: (z_alpha + z_power) * se) - it tells whether a non-significant
    generated difference is informative."""
    rng = np.random.default_rng(seed)
    rows, boots = [], {}
    for m in metrics:
        x, y = ga[m].to_numpy(float), gb[m].to_numpy(float)
        x, y = x[np.isfinite(x) & (x > 0)], y[np.isfinite(y) & (y > 0)]
        row = {"metric": m, "n_a": int(len(x)), "n_b": int(len(y))}
        if len(x) >= 10 and len(y) >= 10:
            ba, bb = iid_median_bootstrap(x, n_boot, rng), iid_median_bootstrap(y, n_boot, rng)
            rep = np.log(ba) - np.log(bb)
            lo, hi = _ci(rep)
            se = float(np.std(rep, ddof=1))
            row.update({"median_a": float(np.median(x)), "median_b": float(np.median(y)), "log_ratio": float(np.log(np.median(x) / np.median(y))),
                        "lr_lo": lo, "lr_hi": hi, "se": se, "mde": (Z_ALPHA + Z_POWER) * se, "cliffs_delta": cliffs_delta(x[:2000], y[:2000]),
                        "p": _perm_median_diff_p(x, y, n_permutations, rng)})
            boots[m] = rep
        rows.append(row)
    table = pd.DataFrame(rows)
    if "p" not in table.columns:
        table["p"] = np.nan
    table["q_bh"] = bh_qvalues(table["p"].to_numpy(float))
    table["significant"] = table["q_bh"] < 0.05
    return table, boots


def _sign(x: float) -> int:
    return int(np.sign(x))


def metric_status(real_sig: bool, gen_sig: bool, lr_r, lo_r, hi_r, lr_g, lo_g, hi_g, mde) -> str:
    """Status of one metric: ``reproduced`` / ``consistent`` (concordant); ``underpowered`` (the generated sample is too
    small to see the real effect: |real log ratio| < minimal detectable effect); without a real effect ``consistent`` =
    the generated log ratio lies inside the real interval, or is not significant and its interval overlaps the real
    one (the generated noise is not punished); the rest is discordant:
    ``wrong_sign``, ``amplitude_over`` / ``amplitude_under``, ``not_reproduced``, ``spurious_effect``, ``outside_real_ci``."""
    if not all(np.isfinite(v) for v in (lr_r, lo_r, hi_r, lr_g, lo_g, hi_g)):
        return "no_data"
    overlap = lo_g <= hi_r and lo_r <= hi_g
    if real_sig:
        weak = np.isfinite(mde) and abs(lr_r) < mde
        if _sign(lr_g) != _sign(lr_r):
            return "wrong_sign" if gen_sig else ("underpowered" if weak else "not_reproduced")
        if gen_sig:
            if overlap:
                return "reproduced"
            return "amplitude_over" if abs(lr_g) > abs(lr_r) else "amplitude_under"
        return "underpowered" if weak else "not_reproduced"
    if lo_r <= lr_g <= hi_r or (not gen_sig and overlap):
        return "consistent"
    return "spurious_effect" if gen_sig else "outside_real_ci"


CONCORDANT = ("reproduced", "consistent")


def concordance(real: pd.DataFrame, rboots: Dict[str, np.ndarray], gen: pd.DataFrame, gboots: Dict[str, np.ndarray]) -> pd.DataFrame:
    """Per metric: significance in both, sign agreement, status (:func:`metric_status`) and the difference of the log
    ratios ``gen - real`` with the 95 % interval of independent bootstrap replicates (0 inside = effect reproduced)."""
    r = real.set_index("metric")
    g = gen.set_index("metric")
    rows = []
    for m in r.index:
        if m not in g.index:
            continue
        rr, gg = r.loc[m], g.loc[m]
        real_sig, gen_sig = bool(rr.get("significant")), bool(gg.get("significant"))
        row = {"metric": m, "real_sig": real_sig, "gen_sig": gen_sig, "real_q": rr.get("q_bh"), "gen_q": gg.get("q_bh"),
               "lr_real": rr.get("log_ratio"), "lo_real": rr.get("lr_lo"), "hi_real": rr.get("lr_hi"),
               "lr_gen": gg.get("log_ratio"), "lo_gen": gg.get("lr_lo"), "hi_gen": gg.get("lr_hi"), "mde": gg.get("mde"),
               "same_sign": bool(_sign(rr.get("log_ratio", np.nan)) == _sign(gg.get("log_ratio", np.nan))) if np.isfinite(rr.get("log_ratio", np.nan)) and np.isfinite(gg.get("log_ratio", np.nan)) else None}
        row["status"] = metric_status(real_sig, gen_sig, row["lr_real"], row["lo_real"], row["hi_real"], row["lr_gen"], row["lo_gen"], row["hi_gen"], row["mde"])
        if m in rboots and m in gboots:
            k = min(len(rboots[m]), len(gboots[m]))
            d = gboots[m][:k] - rboots[m][:k]
            row["delta"] = float(row["lr_gen"] - row["lr_real"])
            row["delta_lo"], row["delta_hi"] = _ci(d)
        row["concordant"] = row["status"] in CONCORDANT
        row["underpowered"] = row["status"] == "underpowered"
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# B. same type: generated vs real, noise floor = real half vs real half (neuron = unit)
# ---------------------------------------------------------------------------


def energy_distance(x: np.ndarray, y: np.ndarray, xx: Optional[float] = None) -> float:
    """Energy distance 2E|X-Y| - E|X-X'| - E|Y-Y'| (U-statistic for the within terms); ``xx`` = precomputed E|X-X'|."""
    from scipy.spatial.distance import cdist

    nx, ny = len(x), len(y)
    dxy = cdist(x, y).mean()
    if xx is None:
        xx = cdist(x, x).sum() / (nx * (nx - 1))
    yy = cdist(y, y).sum() / (ny * (ny - 1))
    return float(2 * dxy - xx - yy)


def _within(x: np.ndarray) -> float:
    from scipy.spatial.distance import cdist

    n = len(x)
    return float(cdist(x, x).sum() / (n * (n - 1)))


def spearman_matrix(rows: np.ndarray) -> np.ndarray:
    from scipy.stats import rankdata

    return np.corrcoef(rankdata(rows, axis=0), rowvar=False)


def corr_rms(ca: np.ndarray, cb: np.ndarray) -> float:
    iu = np.triu_indices(ca.shape[0], 1)
    return float(np.sqrt(np.mean((ca[iu] - cb[iu]) ** 2)))


def _p_upper(obs: float, ref: np.ndarray) -> float:
    ref = ref[np.isfinite(ref)]
    return (1 + int((ref >= obs).sum())) / (1 + len(ref)) if np.isfinite(obs) and len(ref) else float("nan")


class SameTypeTester:
    """Test B. ``pools``: {name: RealPool}. Reference distributions are cached per (pool, sample size).

    Null hypothesis: the generated samples are exchangeable with real spines of the pool. Statistic: W1/sd of each metric
    (sd = spine-level sd of the real pool; W1 between two samples of the same size n) and the energy distance of the
    standardised log vector (complete vectors). Reference: ``n_reference`` repetitions of the same statistic between a
    sample of n spines of one real half and n spines of the other, the halves being DISJOINT random halves of the
    neurons and each draw balanced over neurons (``weighting``) - the exchangeability of two independent real samples
    with the neuron as the unit (noise floor). Observed: generated sample vs real samples of the same size drawn from
    ALL neurons of the pool (mean over ``n_observed_draws`` draws). p = (1 + #{reference >= observed}) / (1 + n_reference),
    BH-corrected over the metrics. The reference is slightly conservative: each reference sample carries the neuron
    sampling noise of half of the neurons, the generated one carries none of the evaluated neurons.

    Equivalence (TOST on the median, log scale): d = log median(generated) - log median(real); standard error = sd of
    the reference differences (conservative); margin = ``margin_sd`` pooled sd of the log metric (default 0.2: a
    pre-specified, interpretable "negligible" effect). Equivalent only if p_TOST (BH over metrics) < 0.05 AND the
    distribution test does not reject. The 95th percentile of the |reference difference| (``delta_noise``) is NOT used as
    a TOST margin (it is about two standard errors wide, a TOST with such a margin can hardly ever pass); it is reported
    with the flag ``within_noise`` (|d| <= delta_noise: the median is as close to the real data as a second real
    sample). With a few hundred generated samples and strong neuron styles the 0.2 sd margin may be unprovable
    (``indistinguishable_equivalence_not_shown``): that is a statement about power, not about the generator.
    """

    def __init__(self, pools: Dict[str, RealPool], metrics: Sequence[str], *, n_reference: int = 1000, n_observed_draws: int = 30,
                 weighting: str = "equal", seed: int = 0, margin_sd: float = TOST_MARGIN_SD):
        self.pools, self.metrics = pools, list(metrics)
        self.margin_sd = margin_sd
        self.n_reference, self.n_observed_draws, self.weighting, self.seed = n_reference, n_observed_draws, weighting, seed
        self._uni: Dict[Tuple[str, int], Dict[str, np.ndarray]] = {}
        self._mv: Dict[Tuple[str, int], Dict[str, np.ndarray]] = {}
        self._names = list(pools)

    def _rng(self, pool: str, n: int, tag: int) -> np.random.Generator:
        return np.random.default_rng([self.seed, self._names.index(pool), int(n), tag])

    def univariate_reference(self, pool: str, n: int) -> Dict[str, np.ndarray]:
        key = (pool, n)
        if key not in self._uni:
            p = self.pools[pool]
            rng = self._rng(pool, n, 1)
            w1 = np.empty((self.n_reference, len(self.metrics)))
            dmed = np.empty_like(w1)
            for r in range(self.n_reference):
                ha, hb = p.halves(rng)
                a, _ = p.draw(n, rng, ha, self.weighting)
                b, _ = p.draw(n, rng, hb, self.weighting)
                w1[r] = np.abs(np.sort(a, axis=0) - np.sort(b, axis=0)).mean(axis=0) / p.sd
                dmed[r] = np.log(np.median(a, axis=0)) - np.log(np.median(b, axis=0))
            self._uni[key] = {"w1": w1, "dmed": dmed}
        return self._uni[key]

    def multivariate_reference(self, pool: str, n: int) -> Dict[str, np.ndarray]:
        key = (pool, n)
        if key not in self._mv:
            p = self.pools[pool]
            rng = self._rng(pool, n, 2)
            ed = np.empty(self.n_reference)
            cr = np.empty(self.n_reference)
            for r in range(self.n_reference):
                ha, hb = p.halves(rng)
                a, _ = p.draw(n, rng, ha, self.weighting)
                b, _ = p.draw(n, rng, hb, self.weighting)
                za, zb = p.standardise(a), p.standardise(b)
                ed[r] = energy_distance(za, zb)
                cr[r] = corr_rms(spearman_matrix(za), spearman_matrix(zb))
            self._mv[key] = {"ed": ed, "corr": cr}
        return self._mv[key]

    def test(self, gen: pd.DataFrame, pool: str, *, keep_reference: bool = False) -> Dict[str, Any]:
        p = self.pools[pool]
        rng = self._rng(pool, 0, 3)
        rows, refs = [], {}
        for j, m in enumerate(self.metrics):
            x = gen[m].to_numpy(float)
            x = x[np.isfinite(x) & (x > 0)]
            row = {"metric": m, "n_gen": int(len(x)), "base_independent": m in BASE_INDEPENDENT}
            if len(x) >= 20:
                ref = self.univariate_reference(pool, len(x))
                xs = np.sort(x)
                w1_obs, lm_real = [], []
                for _ in range(self.n_observed_draws):
                    r, _ = p.draw(len(x), rng, None, self.weighting)
                    w1_obs.append(np.abs(xs - np.sort(r[:, j])).mean() / p.sd[j])
                    lm_real.append(np.log(np.median(r[:, j])))
                w1_obs, lm_real = float(np.mean(w1_obs)), float(np.mean(lm_real))
                d = float(np.log(np.median(x)) - lm_real)
                sd_d = float(np.std(ref["dmed"][:, j], ddof=1))
                delta = self.margin_sd * float(p.log_sd[j])
                delta_noise = float(np.percentile(np.abs(ref["dmed"][:, j]), 95))
                row.update({"median_gen": float(np.median(x)), "median_real": float(np.exp(lm_real)), "log_ratio": d, "w1_sd": w1_obs,
                            "ref_q50": float(np.percentile(ref["w1"][:, j], 50)), "ref_q95": float(np.percentile(ref["w1"][:, j], 95)),
                            "p": _p_upper(w1_obs, ref["w1"][:, j]), "ref_sd_diff": sd_d, "delta": delta, "delta_noise": delta_noise,
                            "p_tost": float(_norm_cdf((abs(d) - delta) / sd_d)) if sd_d > 0 else float("nan"), "within_noise": bool(abs(d) <= delta_noise)})
                if keep_reference:
                    refs[m] = ref["w1"][:, j]
            rows.append(row)
        table = pd.DataFrame(rows)
        for col in ("p", "p_tost"):
            if col not in table.columns:
                table[col] = np.nan
        table["q_bh"] = bh_qvalues(table["p"].to_numpy(float))
        table["q_tost"] = bh_qvalues(table["p_tost"].to_numpy(float))
        table["verdict"] = [_verdict_b(q, qt) for q, qt in zip(table["q_bh"], table["q_tost"])]
        mv: Dict[str, Any] = {"n_complete": 0}
        z = complete_rows(gen, self.metrics)
        mv["n_complete"] = int(len(z))
        if len(z) >= 30:
            zg = p.standardise(z)
            xx = _within(zg)
            ref = self.multivariate_reference(pool, len(z))
            eds, crs = [], []
            cg = spearman_matrix(zg)
            for _ in range(min(self.n_observed_draws, 20)):
                r, _ = p.draw(len(z), rng, None, self.weighting)
                zr = p.standardise(r)
                eds.append(energy_distance(zg, zr, xx))
                crs.append(corr_rms(cg, spearman_matrix(zr)))
            mv.update({"ed": float(np.mean(eds)), "ed_q50": float(np.percentile(ref["ed"], 50)), "ed_q95": float(np.percentile(ref["ed"], 95)), "p_ed": _p_upper(float(np.mean(eds)), ref["ed"]),
                       "corr_rms": float(np.mean(crs)), "corr_q50": float(np.percentile(ref["corr"], 50)), "corr_q95": float(np.percentile(ref["corr"], 95)),
                       "p_corr": _p_upper(float(np.mean(crs)), ref["corr"])})
            if keep_reference:
                refs["_ed"], refs["_corr"] = ref["ed"], ref["corr"]
                refs["_corr_gen"] = cg
                refs["_corr_real"] = np.mean([spearman_matrix(p.standardise(p.draw(len(z), rng, None, self.weighting)[0])) for _ in range(10)], axis=0)
        return {"pool": pool, "n_real_neurons": p.n_neurons, "n_real_spines": p.n_spines, "table": table, "mv": mv, "reference": refs}


def _verdict_b(q: float, q_tost: float) -> str:
    if not np.isfinite(q):
        return "no_data"
    equiv = np.isfinite(q_tost) and q_tost < 0.05
    if q < 0.05:
        return "median_equivalent_distribution_differs" if equiv else "differs"
    return "equivalent" if equiv else "indistinguishable_equivalence_not_shown"


# ---------------------------------------------------------------------------
# C. classifiers of the condition
# ---------------------------------------------------------------------------


def _log_features(x: np.ndarray) -> np.ndarray:
    return np.log(np.asarray(x, float))


def _make_model(kind: str = "logistic", seed: int = 0):
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    if kind == "logistic":
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    if kind == "gbm":
        return HistGradientBoostingClassifier(random_state=seed, max_iter=150)
    raise ValueError(f"unknown classifier {kind!r}")


def _balanced_weights(y: np.ndarray, base: Optional[np.ndarray] = None) -> np.ndarray:
    """Sample weights that give both classes the same total weight (sum = number of rows: the strength of the
    regularisation does not change); ``base`` = per-row weights before balancing (bootstrap multiplicities)."""
    base = np.ones(len(y)) if base is None else np.asarray(base, float)
    w = base.copy()
    for c in (0, 1):
        total = base[y == c].sum()
        w[y == c] = base[y == c] / total if total > 0 else 0.0
    return w * len(y) / 2.0


def _fit(model, x: np.ndarray, y: np.ndarray, base: Optional[np.ndarray] = None):
    from sklearn.base import clone
    from sklearn.pipeline import Pipeline

    m = clone(model)
    w = _balanced_weights(y, base)
    if isinstance(m, Pipeline):
        return m.fit(x, y, **{f"{m.steps[-1][0]}__sample_weight": w})
    return m.fit(x, y, sample_weight=w)


def _oof_scores(model, x: np.ndarray, y: np.ndarray, groups: np.ndarray, n_splits: int, seed: int) -> np.ndarray:
    from sklearn.model_selection import StratifiedGroupKFold

    out = np.full(len(y), np.nan)
    for tr, te in StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed).split(x, y, groups):
        out[te] = _fit(model, x[tr], y[tr]).predict_proba(x[te])[:, 1]
    return out


@dataclass
class ConditionClassifier:
    """A classifier of ``a`` (label 1) against ``b`` (label 0) trained on real spines (classes balanced by weights).

    ``auc`` / ``tpr`` / ``fpr`` come from out-of-fold scores (neuron-grouped cross-validation; tpr / fpr = share of real
    ``a`` / ``b`` spines predicted ``a`` at the threshold 0.5); the intervals from a bootstrap over NEURONS WITH
    REFITTING (the model is retrained on a resample of the neurons and evaluated on the neurons not drawn): they include
    the variation of the classifier itself, which is what a generated set of fresh neurons meets."""

    model: Any
    features: Tuple[str, ...]
    a: str
    b: str
    auc: float
    auc_ci: Tuple[float, float]
    tpr: float
    tpr_ci: Tuple[float, float]
    fpr: float
    fpr_ci: Tuple[float, float]
    boot_tpr: np.ndarray
    boot_fpr: np.ndarray
    n_neurons: Tuple[int, int]
    n_spines: Tuple[int, int]


def fit_condition_classifier(frame: pd.DataFrame, block: str, a: str, b: str, features: Sequence[str], *, kind: str = "logistic", n_splits: int = 5,
                             max_spines_per_neuron: int = 200, n_boot: int = 200, seed: int = 0) -> ConditionClassifier:
    """H0 (AUC): the metrics carry no information on the condition (AUC = 0.5). Evaluation by neuron-grouped, class-stratified
    cross-validation (a neuron is never in the training and the test fold together); at most ``max_spines_per_neuron``
    random spines per neuron so that large neurons do not dominate; intervals by the neuron bootstrap with refitting
    (neurons resampled within each class, model evaluated on the out-of-bag neurons)."""
    from sklearn.metrics import roc_auc_score

    rng = np.random.default_rng(seed)
    sub = frame[frame[block].isin([a, b])]
    x = sub[list(features)].to_numpy(float)
    ok = np.isfinite(x).all(axis=1) & (x > 0).all(axis=1)
    sub, x = sub[ok].reset_index(drop=True), x[ok]
    keep = []
    for _, idx in sub.groupby("neuron_id").indices.items():
        keep.append(idx if len(idx) <= max_spines_per_neuron else rng.choice(idx, max_spines_per_neuron, replace=False))
    keep = np.sort(np.concatenate(keep))
    sub, x = sub.iloc[keep].reset_index(drop=True), x[keep]
    y = (sub[block] == a).to_numpy().astype(int)
    nid, _ = pd.factorize(sub["neuron_id"])
    xf = _log_features(x)
    model = _make_model(kind, seed)
    scores = _oof_scores(model, xf, y, nid, n_splits, seed)
    pred = scores > 0.5
    k = int(nid.max()) + 1
    cls = np.zeros(k, dtype=int)
    cls[nid] = y
    members = [np.where(cls == c)[0] for c in (0, 1)]
    aucs, tprs, fprs = [], [], []
    for _ in range(n_boot):
        counts = np.zeros(k)
        for mem in members:
            counts += np.bincount(rng.choice(mem, len(mem)), minlength=k)
        inbag, oob = counts[nid] > 0, counts[nid] == 0
        if len(np.unique(y[oob])) < 2:
            continue
        m = _fit(model, xf[inbag], y[inbag], counts[nid][inbag])
        s = m.predict_proba(xf[oob])[:, 1]
        aucs.append(roc_auc_score(y[oob], s))
        tprs.append(float((s[y[oob] == 1] > 0.5).mean()))
        fprs.append(float((s[y[oob] == 0] > 0.5).mean()))
    final = _fit(model, xf, y)
    tpr, fpr = float(pred[y == 1].mean()), float(pred[y == 0].mean())
    return ConditionClassifier(final, tuple(features), a, b, float(roc_auc_score(y, scores)), _ci(np.array(aucs)), tpr, _ci(np.array(tprs)), fpr, _ci(np.array(fprs)),
                               np.array(tprs), np.array(fprs), (int(cls.size - cls.sum()), int(cls.sum())), (int((y == 0).sum()), int((y == 1).sum())))


def apply_condition_classifier(clf: ConditionClassifier, gen: pd.DataFrame) -> np.ndarray:
    """Scores P(condition a) of the complete generated vectors of ``gen`` (rows with a missing metric are skipped)."""
    x = gen[list(clf.features)].to_numpy(float)
    ok = np.isfinite(x).all(axis=1) & (x > 0).all(axis=1)
    return clf.model.predict_proba(_log_features(x[ok]))[:, 1] if ok.any() else np.empty(0)


def mann_whitney_auc(pos: np.ndarray, neg: np.ndarray, n_boot: int = 1000, seed: int = 0) -> Tuple[float, Tuple[float, float]]:
    """AUC of ``pos`` scores above ``neg`` scores with a bootstrap interval (independent samples)."""
    from scipy.stats import rankdata

    def auc(p, n):
        r = rankdata(np.concatenate([p, n]))
        return (r[:len(p)].sum() - len(p) * (len(p) + 1) / 2) / (len(p) * len(n))

    rng = np.random.default_rng(seed)
    point = float(auc(pos, neg))
    boots = [auc(pos[rng.integers(0, len(pos), len(pos))], neg[rng.integers(0, len(neg), len(neg))]) for _ in range(n_boot)]
    return point, _ci(np.array(boots))


def rogan_gladen(frac_pred: float, n_gen: int, tpr_boot: np.ndarray, fpr_boot: np.ndarray, seed: int = 0, *, tpr: Optional[float] = None,
                 fpr: Optional[float] = None) -> Tuple[float, Tuple[float, float]]:
    """Effective share of condition ``a`` in a set: pi = (q - FPR) / (TPR - FPR) from the share q predicted ``a``, with
    the real TPR / FPR; interval = bootstrap over the real neurons (TPR, FPR) + binomial noise of q; clipped to [0, 1]."""
    rng = np.random.default_rng(seed)
    q = rng.binomial(n_gen, frac_pred, len(tpr_boot)) / n_gen
    with np.errstate(invalid="ignore", divide="ignore"):
        pi = np.clip((q - fpr_boot) / (tpr_boot - fpr_boot), 0, 1)
    tpr = float(np.mean(tpr_boot)) if tpr is None else tpr
    fpr = float(np.mean(fpr_boot)) if fpr is None else fpr
    point = float(np.clip((frac_pred - fpr) / (tpr - fpr), 0, 1))
    return point, _ci(pi)


# ---------------------------------------------------------------------------
# D. real-vs-generated classifier two-sample test
# ---------------------------------------------------------------------------


def _auc_grouped(x0: np.ndarray, g0: np.ndarray, x1: np.ndarray, g1: np.ndarray, kind: str, n_splits: int, seed: int) -> float:
    from sklearn.metrics import roc_auc_score

    x = np.vstack([_log_features(x0), _log_features(x1)])
    y = np.r_[np.zeros(len(x0), int), np.ones(len(x1), int)]
    groups = np.r_[g0, g1]
    scores = _oof_scores(_make_model(kind, seed), x, y, groups, n_splits, seed)
    return float(roc_auc_score(y, scores))


def real_vs_generated_auc(pool: RealPool, gen_rows: np.ndarray, *, n_reference: int = 100, kind: str = "logistic", n_splits: int = 5,
                          weighting: str = "equal", seed: int = 0) -> Dict[str, Any]:
    """H0: generated vectors are exchangeable with real ones (AUC of the real-vs-generated classifier = the AUC of two
    independent real samples). Statistic: out-of-fold AUC (real = n spines drawn from all neurons, grouped by neuron;
    each generated sample is its own group). Reference: the same procedure for a sample of one real neuron half against
    one of the other half. p = share of reference AUC >= observed."""
    rng = np.random.default_rng(seed)
    n = len(gen_rows)
    real, nid = pool.draw(n, rng, None, weighting)
    obs = _auc_grouped(real, nid, gen_rows, pool.n_neurons + np.arange(n), kind, n_splits, seed)
    ref = np.empty(n_reference)
    for r in range(n_reference):
        ha, hb = pool.halves(rng)
        a, na = pool.draw(n, rng, ha, weighting)
        b, nb = pool.draw(n, rng, hb, weighting)
        ref[r] = _auc_grouped(a, na, b, nb, kind, n_splits, seed + r + 1)
    return {"model": kind, "auc": obs, "ref": ref, "ref_q50": float(np.percentile(ref, 50)), "ref_q95": float(np.percentile(ref, 95)),
            "p": _p_upper(obs, ref), "n": int(n)}


# ---------------------------------------------------------------------------
# E. dose-response; F. validity
# ---------------------------------------------------------------------------


def amplitude_slope(lr_gen: np.ndarray, lr_real: np.ndarray) -> float:
    """Slope through the origin of the generated log ratios on the real ones over metrics: 1 = the real amplitude,
    < 1 = weaker, > 1 = stronger, < 0 = the opposite direction."""
    d = float((lr_real ** 2).sum())
    return float((lr_gen * lr_real).sum() / d) if d > 0 else float("nan")


def dose_response(effects: Dict[float, Tuple[pd.DataFrame, Dict[str, np.ndarray]]], real: pd.DataFrame, rboots: Dict[str, np.ndarray], *,
                  gen_ed: Optional[Dict[float, float]] = None, real_ed: Optional[Tuple[float, float, float]] = None) -> pd.DataFrame:
    """Per guidance weight w: amplitude slope beta over the metrics with a significant real effect (all metrics if none)
    with a 95 % interval from the bootstrap replicates of both log ratios; the generated energy distance between the two
    conditions against the real one (``real_ed`` = (median, q25, q75)); ``closest`` marks the w nearest to the real
    multivariate effect."""
    r = real.set_index("metric")
    sig = [m for m in r.index if bool(r.loc[m, "significant"]) and m in rboots]
    use = sig or [m for m in r.index if m in rboots]
    rows = []
    for w, (gen, gboots) in sorted(effects.items()):
        g = gen.set_index("metric")
        ms = [m for m in use if m in g.index and m in gboots and np.isfinite(g.loc[m, "log_ratio"])]
        row = {"w": w, "n_metrics": len(ms), "metrics_used": "significant" if sig else "all"}
        if ms:
            lg = np.array([g.loc[m, "log_ratio"] for m in ms])
            lr = np.array([r.loc[m, "log_ratio"] for m in ms])
            k = min(min(len(gboots[m]) for m in ms), min(len(rboots[m]) for m in ms))
            bg = np.array([gboots[m][:k] for m in ms])
            br = np.array([rboots[m][:k] for m in ms])
            betas = (bg * br).sum(0) / (br ** 2).sum(0)
            row.update({"beta": amplitude_slope(lg, lr), "beta_lo": _ci(betas)[0], "beta_hi": _ci(betas)[1]})
        if gen_ed and w in gen_ed:
            row["ed_gen"] = gen_ed[w]
            if real_ed:
                row["ed_real"] = real_ed[0]
        rows.append(row)
    table = pd.DataFrame(rows)
    if len(table) and "ed_gen" in table.columns and real_ed:
        table["closest"] = False
        table.loc[(table["ed_gen"] - real_ed[0]).abs().idxmin(), "closest"] = True
    return table


def validity_table(gen_sets: Dict[Tuple[str, str], pd.DataFrame], metrics: Sequence[str]) -> pd.DataFrame:
    """Per set: rates (Wilson 95 % interval) of meshes extracted, bases found, valid meshes and complete metric vectors."""
    rows = []
    for (value, label), g in gen_sets.items():
        n = len(g)
        row = {"value": value, "label": label, "n": n}
        for name, col in (("mesh", g["mesh_present"] if "mesh_present" in g else None), ("base", g["attachment_found"] if "attachment_found" in g else None),
                          ("valid", g["valid"] if "valid" in g else None)):
            if col is not None:
                k = int(col.astype(bool).sum())
                row[f"{name}_n"] = k
                row[f"{name}_rate"], row[f"{name}_lo"], row[f"{name}_hi"] = wilson(k, n)
        k = int(len(complete_rows(g, metrics)))
        row["complete_n"] = k
        row["complete_rate"], row["complete_lo"], row["complete_hi"] = wilson(k, n)
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# orchestration
# ---------------------------------------------------------------------------


def pair_order(block: str, values: Sequence[str]) -> Optional[Tuple[str, str]]:
    """The pair (a, b) of condition values whose difference is tested: inhibitory first for the binary cell type."""
    values = sorted(set(values))
    if block == "cell_type_binary" and {"inhibitory", "excitatory"} <= set(values):
        return "inhibitory", "excitatory"
    return (values[0], values[1]) if len(values) >= 2 else None


def run_conditional_tests(real: pd.DataFrame, gen_sets: Dict[Tuple[str, str], pd.DataFrame], block: str, *, metrics: Optional[Sequence[str]] = None,
                          min_spines: int = 20, n_permutations: int = 10000, n_boot: int = 1000, n_reference: int = 1000, n_reference_classifier: int = 100,
                          weighting: str = "equal", only_valid: bool = False, classifiers: Sequence[str] = ("logistic",), seed: int = 0,
                          tost_margin_sd: float = TOST_MARGIN_SD, log=print) -> Dict[str, Any]:
    """All tests A-F for generated sets ``{(value, label): frame}`` (value ``unconditional`` for a set without a condition)
    against ``real`` (spines with ``neuron_id``, ``block`` column and the metrics). Returns a dict of tables / arrays."""
    metrics = list(metrics or [m for m in SCALAR_METRICS if m in real.columns])
    metrics = [m for m in metrics if all(m in g.columns and np.isfinite(g[m].to_numpy(float)).any() for g in gen_sets.values())]
    out: Dict[str, Any] = {"metrics": metrics, "block": block, "settings": {
        "min_spines": min_spines, "n_permutations": n_permutations, "n_boot": n_boot, "n_reference": n_reference, "n_reference_classifier": n_reference_classifier,
        "weighting": weighting, "only_valid": only_valid, "seed": seed, "tost_margin_sd": tost_margin_sd}}
    out["validity"] = validity_table(gen_sets, metrics)
    if only_valid:
        gen_sets = {k: (g[g["valid"].astype(bool)] if "valid" in g else g) for k, g in gen_sets.items()}
    known = real[real[block].notna()]
    values = sorted({v for v, _ in gen_sets if v != "unconditional"})
    pair = pair_order(block, values + (["inhibitory", "excitatory"] if block == "cell_type_binary" and values else []))
    pool_names = list(dict.fromkeys(values + (list(pair) if pair else []) + ["mixture"]))
    pools: Dict[str, RealPool] = {}
    for name in pool_names:
        sub = known if name == "mixture" else known[known[block] == name]
        pool = RealPool(sub, metrics, min_spines=min_spines)
        if pool.n_neurons >= 6:
            pools[name] = pool
        log(f"[cond-tests] real pool {name}: {pool.n_spines} spines of {pool.n_neurons} neurons ({pool.n_dropped} rows dropped: missing / non-positive metrics)"
            + ("" if name in pools else " - too few neurons, pool skipped"))
    out["pools"] = {n: {"n_spines": p.n_spines, "n_neurons": p.n_neurons, "n_dropped": p.n_dropped} for n, p in pools.items()}
    keys = list(gen_sets)

    # ---- A: difference between the two types
    out["pair"] = list(pair) if pair else None
    labels = sorted({lab for (v, lab) in keys if pair and v in pair and (pair[0], lab) in gen_sets and (pair[1], lab) in gen_sets})
    effects: Dict[str, Any] = {}
    if pair and labels:
        a, b = pair
        log(f"[cond-tests] A: real {a} / {b}: neuron-level permutation tests + neuron bootstrap")
        real_tbl, rboots = type_effect_real(known, block, a, b, metrics, min_spines=min_spines, n_permutations=n_permutations, n_boot=n_boot, seed=seed)
        out["real_effect"] = real_tbl
        gen_effects: Dict[float, Tuple[pd.DataFrame, Dict[str, np.ndarray]]] = {}
        for i, lab in enumerate(labels):
            log(f"[cond-tests] A: generated {a} / {b} @ {lab}")
            gtbl, gboots = type_effect_generated(gen_sets[(a, lab)], gen_sets[(b, lab)], metrics, n_permutations=n_permutations, n_boot=n_boot, seed=seed + 100 + i)
            conc = concordance(real_tbl, rboots, gtbl, gboots)
            effects[lab] = {"generated": gtbl, "concordance": conc, "n_concordant": int(conc["concordant"].sum()), "n_underpowered": int(conc["underpowered"].sum()),
                            "n_discordant": int((~conc["concordant"] & ~conc["underpowered"]).sum()), "n_metrics": int(len(conc)),
                            "n_real_significant": int(conc["real_sig"].sum()), "n_reproduced": int((conc["status"] == "reproduced").sum())}
            gen_effects[parse_weight(lab)] = (gtbl, gboots)
        out["effects"] = effects

        # ---- E: dose-response over the guidance weights (labels with a number)
        weights = [w for w in gen_effects if np.isfinite(w)]
        if weights:
            std_pool = pools.get("mixture") or next(iter(pools.values()))
            rng = np.random.default_rng(seed + 500)
            gen_ed = {}
            for lab in labels:
                za, zb = complete_rows(gen_sets[(a, lab)], metrics), complete_rows(gen_sets[(b, lab)], metrics)
                if len(za) >= 30 and len(zb) >= 30:
                    za = std_pool.standardise(za[rng.choice(len(za), min(len(za), 500), replace=False)])
                    zb = std_pool.standardise(zb[rng.choice(len(zb), min(len(zb), 500), replace=False)])
                    gen_ed[parse_weight(lab)] = energy_distance(za, zb)
            real_ed = None
            if a in pools and b in pools:
                n_ed = 500
                eds = [energy_distance(std_pool.standardise(pools[a].draw(n_ed, rng, None, weighting)[0]), std_pool.standardise(pools[b].draw(n_ed, rng, None, weighting)[0]))
                       for _ in range(min(100, n_reference))]
                real_ed = (float(np.median(eds)), float(np.percentile(eds, 25)), float(np.percentile(eds, 75)))
            out["dose"] = dose_response({w: gen_effects[w] for w in weights}, real_tbl, rboots, gen_ed=gen_ed, real_ed=real_ed)
            out["real_ed"] = real_ed

    # ---- B: same type (+ control against the other pools)
    tester = SameTypeTester(pools, metrics, n_reference=n_reference, weighting=weighting, seed=seed, margin_sd=tost_margin_sd)
    same: Dict[str, Any] = {}
    for (value, lab) in keys:
        own = value if value in pools else "mixture"
        if own not in pools:
            continue
        gen = gen_sets[(value, lab)]
        log(f"[cond-tests] B: {value}@{lab} against real {own} (+ control against {[p for p in pools if p != own]})")
        by_pool = {p: tester.test(gen, p, keep_reference=(p == own)) for p in pools}
        same[f"{value}@{lab}"] = {"value": value, "label": lab, "own": own, "by_pool": by_pool}
    out["same_type"] = same

    # ---- C: classifiers of the condition
    clf_out: Dict[str, Any] = {}
    if pair and pair[0] in pools and pair[1] in pools:
        a, b = pair
        feature_sets = {"all": metrics, "base_independent": [m for m in BASE_INDEPENDENT if m in metrics]}
        for kind in classifiers:
            for fname, feats in feature_sets.items():
                if len(feats) < 2:
                    continue
                log(f"[cond-tests] C: condition classifier ({kind}, features: {fname})")
                clf = fit_condition_classifier(known, block, a, b, feats, kind=kind, n_boot=min(n_boot, 200), seed=seed)
                entry = {"auc": clf.auc, "auc_ci": clf.auc_ci, "tpr": clf.tpr, "tpr_ci": clf.tpr_ci, "fpr": clf.fpr, "fpr_ci": clf.fpr_ci,
                         "n_neurons": clf.n_neurons, "n_spines": clf.n_spines, "features": feats, "sets": {}, "gen_auc": {}}
                scores_by_key = {}
                for (value, lab), g in gen_sets.items():
                    s = apply_condition_classifier(clf, g)
                    scores_by_key[(value, lab)] = s
                    k = int((s > 0.5).sum())
                    frac, lo, hi = wilson(k, len(s))
                    row = {"value": value, "label": lab, "n": int(len(s)), "n_total": int(len(g)), "frac_pred_a": frac, "lo": lo, "hi": hi}
                    if value == a:
                        row["expected"], row["expected_ci"] = clf.tpr, clf.tpr_ci
                    elif value == b:
                        row["expected"], row["expected_ci"] = clf.fpr, clf.fpr_ci
                    elif len(s) > 0:
                        pi, pi_ci = rogan_gladen(frac, len(s), clf.boot_tpr, clf.boot_fpr, seed, tpr=clf.tpr, fpr=clf.fpr)
                        row["effective_share_a"], row["effective_share_a_ci"] = pi, pi_ci
                    if "expected_ci" in row:
                        row["within_expected"] = bool(lo <= row["expected_ci"][1] and row["expected_ci"][0] <= hi)
                    entry["sets"][f"{value}@{lab}"] = row
                for lab in labels:
                    sa, sb = scores_by_key[(a, lab)], scores_by_key[(b, lab)]
                    if len(sa) >= 10 and len(sb) >= 10:
                        auc, ci = mann_whitney_auc(sa, sb, seed=seed)
                        entry["gen_auc"][lab] = {"auc": auc, "ci": ci, "overlaps_real": bool(ci[0] <= clf.auc_ci[1] and clf.auc_ci[0] <= ci[1])}
                clf_out[f"{kind}/{fname}"] = entry
        known_n = known.groupby("neuron_id")[block].first().value_counts().to_dict()
        out["class_balance"] = {"neurons": {str(k): int(v) for k, v in known_n.items()},
                                "spines": {str(k): int(v) for k, v in known[block].value_counts().items()}}
    out["classifier"] = clf_out

    # ---- D: real-vs-generated discriminability
    disc: Dict[str, Any] = {}
    for (value, lab), g in gen_sets.items():
        own = value if value in pools else "mixture"
        z = complete_rows(g, metrics)
        if own not in pools or len(z) < 50:
            continue
        for kind in classifiers:
            log(f"[cond-tests] D: real vs generated classifier ({kind}) {value}@{lab} against {own}")
            disc.setdefault(f"{value}@{lab}", {})[kind] = real_vs_generated_auc(pools[own], z, n_reference=n_reference_classifier, kind=kind, weighting=weighting, seed=seed)
    out["discriminability"] = disc
    return out


def to_jsonable(obj: Any, digits: int = 6) -> Any:
    """Tables and arrays -> plain JSON data (NaN -> None)."""
    if isinstance(obj, pd.DataFrame):
        return [to_jsonable(r, digits) for r in obj.to_dict("records")]
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v, digits) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v, digits) for v in obj]
    if isinstance(obj, np.ndarray):
        return to_jsonable(obj.tolist(), digits)
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        f = float(obj)
        return float(f"{f:.{digits}g}") if math.isfinite(f) else None
    return obj
