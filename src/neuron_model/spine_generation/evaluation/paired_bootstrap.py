"""Paired comparison of two generators against the same real spines, with the real NEURON as the resampling unit.

Why: ``evaluate_generators.py`` reports mean +- sd over generation seeds; that spread holds the real reference fixed and
says nothing about how much the conclusion "A is closer to the real spines than v3" depends on which real neurons
happen to be in the test part. Here one bootstrap replicate = real neurons drawn with replacement (cluster bootstrap)
+ every generated set drawn with replacement within its generation seed; ALL models are compared with the SAME real
replicate, so the difference of two models is paired and the shared real noise cancels.

Statistics per replicate, on log metrics standardised by the real (full test part) sd:
  * ``w1_<metric>``  - 1-D Wasserstein-1 distance generated vs real (in real sd);
  * ``w1_mean``      - its mean over the metrics;
  * ``energy``       - multivariate energy distance;
  * ``auc``          - logistic real-vs-generated classifier, k-fold cross-validated ROC AUC (0.5 = indistinguishable).
For a pair (a, b): difference a - b, percentile 95 % interval and the two-sided bootstrap p (2 * min share of
replicates on either side of 0). Lower = closer to real for every statistic.
"""

from __future__ import annotations

from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

METRICS = ("OpenAngle", "CVD", "AverageDistance", "LengthVolumeRatio", "LengthAreaRatio", "Length", "JunctionArea",
           "Area", "Volume", "ConvexHullVolume", "ConvexHullRatio")


def neuron_of(spine_key: str) -> str:
    """``minnie65/<neuron>/<limb>/<branch>/<spine>`` -> ``minnie65/<neuron>``."""
    parts = str(spine_key).split("/")
    return "/".join(parts[:2])


def log_complete(frame: pd.DataFrame, metrics: Sequence[str]) -> Tuple[np.ndarray, np.ndarray]:
    """log of the metrics of rows where all are finite and > 0; returns (values, row mask)."""
    x = frame[list(metrics)].to_numpy(dtype=float)
    ok = np.all(np.isfinite(x) & (x > 0), axis=1)
    return np.log(x[ok]), ok


def w1_1d(a: np.ndarray, b: np.ndarray, n_q: int = 512) -> float:
    """W1 between two 1-D samples via quantile functions on a common grid."""
    q = (np.arange(n_q) + 0.5) / n_q
    return float(np.mean(np.abs(np.quantile(a, q) - np.quantile(b, q))))


def energy_distance(x: np.ndarray, y: np.ndarray) -> float:
    def mean_dist(a, b):
        d2 = (a * a).sum(1)[:, None] + (b * b).sum(1)[None, :] - 2.0 * a @ b.T
        return float(np.sqrt(np.maximum(d2, 0.0)).mean())

    return 2.0 * mean_dist(x, y) - mean_dist(x, x) - mean_dist(y, y)


def cv_auc(real: np.ndarray, gen: np.ndarray, rng: np.random.Generator, k: int = 5) -> float:
    """k-fold logistic regression real (0) vs generated (1), pooled out-of-fold ROC AUC."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score

    x = np.vstack([real, gen])
    y = np.r_[np.zeros(len(real)), np.ones(len(gen))]
    fold = rng.permutation(len(x)) % k
    score = np.empty(len(x))
    for f in range(k):
        tr, te = fold != f, fold == f
        clf = LogisticRegression(max_iter=1000).fit(x[tr], y[tr])
        score[te] = clf.decision_function(x[te])
    return float(roc_auc_score(y, score))


def replicate_stats(real: np.ndarray, gen: np.ndarray, metrics: Sequence[str], rng: np.random.Generator,
                    *, auc: bool = True) -> Dict[str, float]:
    out = {f"w1_{m}": w1_1d(real[:, j], gen[:, j]) for j, m in enumerate(metrics)}
    out["w1_mean"] = float(np.mean([out[f"w1_{m}"] for m in metrics]))
    out["energy"] = energy_distance(real, gen)
    if auc:
        out["auc"] = cv_auc(real, gen, rng)
    return out


def paired_bootstrap(real: pd.DataFrame, generated: Mapping[str, pd.DataFrame], *, metrics: Sequence[str] = METRICS,
                     n_boot: int = 500, n_real: int = 3000, seed: int = 0, auc: bool = True,
                     key_col: str = "spine_key", seed_col: str = "generation_seed",
                     log=print) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """``real``: one row per real spine (``key_col`` + metrics); ``generated``: model -> rows (metrics + ``seed_col``).

    Returns (point table: model x statistic on the full samples, replicate table: replicate x model x statistic).
    ``n_real``: real spines drawn (from the resampled neurons, pooled) per replicate - keeps the energy distance and
    the classifier cheap; the point estimate uses the same size."""
    rng = np.random.default_rng(seed)
    rx, ok = log_complete(real, metrics)
    neurons = real.loc[ok, key_col].map(neuron_of).to_numpy()
    mu, sd = rx.mean(0), rx.std(0)
    rz = (rx - mu) / sd
    groups = pd.Series(np.arange(len(rz))).groupby(neurons).apply(lambda s: s.to_numpy()).to_dict()
    names = list(groups)
    gens = {}
    for model, frame in generated.items():
        gx, gok = log_complete(frame, metrics)
        seeds = frame.loc[gok, seed_col].to_numpy() if seed_col in frame.columns else np.zeros(len(gx))
        gens[model] = ((gx - mu) / sd, seeds)
        log(f"[paired-bootstrap] {model}: {len(gx)} complete samples of {len(frame)} (seeds {sorted(set(seeds.tolist()))})")
    log(f"[paired-bootstrap] real: {len(rz)} spines, {len(names)} neurons; {n_boot} replicates, {n_real} real spines each")

    def draw_gen(z, seeds):
        idx = np.concatenate([rng.choice(np.flatnonzero(seeds == s), (seeds == s).sum(), replace=True) for s in np.unique(seeds)])
        return z[idx]

    point_real = rz[rng.choice(len(rz), min(n_real, len(rz)), replace=False)]
    points = [{"model": m, **replicate_stats(point_real, z, metrics, rng, auc=auc)} for m, (z, _) in gens.items()]
    rows = []
    for b in range(n_boot):
        pick = rng.choice(len(names), len(names), replace=True)
        pool = np.concatenate([groups[names[i]] for i in pick])
        r = rz[rng.choice(pool, min(n_real, len(pool)), replace=True)]
        for m, (z, seeds) in gens.items():
            rows.append({"replicate": b, "model": m, **replicate_stats(r, draw_gen(z, seeds), metrics, rng, auc=auc)})
        if (b + 1) % 50 == 0:
            log(f"[paired-bootstrap] {b + 1}/{n_boot}")
    return pd.DataFrame(points), pd.DataFrame(rows)


def pair_differences(points: pd.DataFrame, reps: pd.DataFrame, pairs: Sequence[Tuple[str, str]]) -> pd.DataFrame:
    """a - b per statistic: point estimate, 95 % percentile interval, two-sided bootstrap p, share of replicates a < b."""
    stats = [c for c in reps.columns if c not in ("replicate", "model")]
    wide = {m: g.set_index("replicate")[stats] for m, g in reps.groupby("model")}
    pt = points.set_index("model")
    out = []
    for a, b in pairs:
        d = wide[a] - wide[b]
        for s in stats:
            v = d[s].dropna().to_numpy()
            lo, hi = np.percentile(v, [2.5, 97.5])
            p = min(1.0, 2.0 * min((v <= 0).mean(), (v >= 0).mean()))
            out.append({"a": a, "b": b, "statistic": s, "a_value": float(pt.loc[a, s]), "b_value": float(pt.loc[b, s]),
                        "diff": float(pt.loc[a, s] - pt.loc[b, s]), "lo": float(lo), "hi": float(hi), "p": float(p),
                        "share_a_better": float((v < 0).mean()), "n_boot": int(len(v))})
    return pd.DataFrame(out)
