"""Density-ratio (discriminator) rejection sampling on spine morphometrics.

Idea (Azadi et al., 2019, "Discriminator Rejection Sampling"): with a probabilistic classifier D(x) = P(real | x) trained
on real vs generated samples with equal class weights, the density ratio is r(x) = p_real(x) / p_gen(x) = D / (1 - D).
Accepting a generated sample with probability min(1, r(x) / M) turns the generated distribution into
p_gen(x) * min(1, r / M) / Z, which equals p_real where r <= M. M is a high quantile of r on the generated fitting set
(the exact maximum is dominated by a few samples and would reject almost everything).

Here x = standardised log morphometrics of a reconstructed generated mesh. The correction is a declared post-processing
step of the generator: it does not change the generator, only which of its samples are kept, and it cannot create
shapes the generator never produces. It must be fitted on TRAINING real spines and on a generation seed that is not
used for evaluation; evaluation uses another seed and held-out neurons.

Samples with incomplete metrics (no attachment found) cannot be scored; ``score`` returns NaN for them and the caller
decides (``accept`` keeps them with the mean acceptance rate, so the selection does not silently drop failures).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from .paired_bootstrap import METRICS


def log_features(frame: pd.DataFrame, metrics: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
    x = frame[list(metrics)].to_numpy(dtype=float)
    ok = np.all(np.isfinite(x) & (x > 0), axis=1)
    out = np.full(x.shape, np.nan)
    out[ok] = np.log(x[ok])
    return out, ok


@dataclass
class DensityRatioSelector:
    metrics: Sequence[str] = METRICS
    model: str = "gbm"  # "gbm" (histogram gradient boosting) or "logistic" (quadratic features)
    quantile: float = 0.95  # M = this quantile of r on the generated fitting set
    clip: float = 1e-3
    seed: int = 0
    mu_: Optional[np.ndarray] = field(default=None, init=False)
    sd_: Optional[np.ndarray] = field(default=None, init=False)
    clf_: object = field(default=None, init=False)
    M_: float = field(default=float("nan"), init=False)
    mean_accept_: float = field(default=float("nan"), init=False)

    def _make(self):
        if self.model == "logistic":
            from sklearn.linear_model import LogisticRegression
            from sklearn.pipeline import make_pipeline
            from sklearn.preprocessing import PolynomialFeatures

            return make_pipeline(PolynomialFeatures(2, include_bias=False), LogisticRegression(C=1.0, max_iter=5000))
        from sklearn.ensemble import HistGradientBoostingClassifier

        return HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=300, l2_regularization=1.0,
                                              early_stopping=True, validation_fraction=0.2, random_state=self.seed)

    def fit(self, real: pd.DataFrame, generated: pd.DataFrame) -> "DensityRatioSelector":
        from sklearn.calibration import CalibratedClassifierCV

        xr, okr = log_features(real, self.metrics)
        xg, okg = log_features(generated, self.metrics)
        xr, xg = xr[okr], xg[okg]
        self.mu_, self.sd_ = xr.mean(0), xr.std(0)
        x = np.vstack([(xr - self.mu_) / self.sd_, (xg - self.mu_) / self.sd_])
        y = np.r_[np.ones(len(xr)), np.zeros(len(xg))]
        w = np.r_[np.full(len(xr), 0.5 / len(xr)), np.full(len(xg), 0.5 / len(xg))] * len(y)
        self.clf_ = CalibratedClassifierCV(self._make(), method="isotonic", cv=5)
        self.clf_.fit(x, y, sample_weight=w)
        r = self._ratio((xg - self.mu_) / self.sd_)
        self.M_ = float(np.quantile(r, self.quantile))
        self.mean_accept_ = float(np.minimum(1.0, r / self.M_).mean())
        return self

    def _ratio(self, z: np.ndarray) -> np.ndarray:
        p = np.clip(self.clf_.predict_proba(z)[:, 1], self.clip, 1 - self.clip)
        return p / (1 - p)

    def score(self, generated: pd.DataFrame) -> np.ndarray:
        """Acceptance probability per row (NaN where metrics are incomplete)."""
        x, ok = log_features(generated, self.metrics)
        out = np.full(len(generated), np.nan)
        if ok.any():
            out[ok] = np.minimum(1.0, self._ratio((x[ok] - self.mu_) / self.sd_) / self.M_)
        return out

    def accept(self, generated: pd.DataFrame, rng: np.random.Generator) -> np.ndarray:
        a = self.score(generated)
        a = np.where(np.isfinite(a), a, self.mean_accept_)
        return rng.random(len(a)) < a


def per_metric_shift(real: pd.DataFrame, sets: dict, *, metrics: Sequence[str] = METRICS, n_boot: int = 1000,
                     seed: int = 0, key_col: str = "spine_key") -> pd.DataFrame:
    """For every set and metric: d = ln(median generated / median real), 95 % interval (real resampled by neuron,
    generated resampled by sample), BH-adjusted two-sided bootstrap p, W1 of the logs in real sd, and n."""
    from .paired_bootstrap import neuron_of, w1_1d

    rng = np.random.default_rng(seed)
    xr, okr = log_features(real, metrics)
    xr = xr[okr]
    neurons = real.loc[okr, key_col].map(neuron_of).to_numpy()
    groups = [np.flatnonzero(neurons == n) for n in pd.unique(neurons)]
    sd = xr.std(0)
    rows = []
    for name, frame in sets.items():
        xg, okg = log_features(frame, metrics)
        xg = xg[okg]
        boots = np.empty((n_boot, len(metrics)))
        for b in range(n_boot):
            pick = np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])
            gi = rng.integers(0, len(xg), len(xg))
            boots[b] = np.median(xg[gi], 0) - np.median(xr[pick], 0)
        d = np.median(xg, 0) - np.median(xr, 0)
        for j, m in enumerate(metrics):
            v = boots[:, j]
            p = min(1.0, 2 * min((v <= 0).mean(), (v >= 0).mean()))
            rows.append({"set": name, "metric": m, "n": int(len(xg)), "d": float(d[j]), "lo": float(np.percentile(v, 2.5)),
                         "hi": float(np.percentile(v, 97.5)), "p": p, "w1_sd": w1_1d(xg[:, j], xr[:, j]) / sd[j]})
    out = pd.DataFrame(rows)
    out["q"] = np.nan
    for name, g in out.groupby("set"):
        ps = g["p"].to_numpy()
        order = np.argsort(ps)
        q = np.empty_like(ps)
        q[order] = np.minimum.accumulate((ps[order] * len(ps) / np.arange(1, len(ps) + 1))[::-1])[::-1]
        out.loc[g.index, "q"] = np.minimum(q, 1.0)
    out["significant"] = out["q"] < 0.05
    return out
