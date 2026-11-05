"""Which morphometric features let the real-vs-generated classifier tell the sets apart (evaluation/classifier.py)?

The classifier of the generator report (logistic regression on the 10 joint morphometrics, balanced classes, 5-fold CV) gives
one number. Part of that number may not be about the SHAPE at all:

- surface ``Area`` (and ``LengthAreaRatio``) grows with surface roughness: the staircase of the segmented real meshes versus
  the smooth Poisson / Marching-Cubes surfaces of the generated ones;
- the 7 junction metrics depend on the attachment region, which is the stage-1 truth for real spines and the heuristic
  ``DownFacingCapFinder`` for generated ones (known bias, finder_bias.json).

This module re-runs the same classifier on feature subsets: each feature alone, all but one, and named groups (no area, no
attachment, attachment-free only); plus the standardised coefficients of the full model and the standardised mean difference
of every feature. Every score is averaged over ``n_repeats`` random balanced subsamples / fold assignments (the spread is
reported), so differences between subsets are not fold noise.
"""

from __future__ import annotations

from typing import Dict, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from .distribution_metrics import JOINT_FEATURES

AREA_FEATURES = ("Area", "LengthAreaRatio")
# metrics computed with the attachment region (spine_morphometrics.JUNCTION_METRICS) that are in the joint vector
ATTACHMENT_FEATURES = ("OpenAngle", "CVD", "AverageDistance", "LengthVolumeRatio", "LengthAreaRatio", "Length", "Area")
ATTACHMENT_FREE_FEATURES = ("Volume", "ConvexHullVolume", "ConvexHullRatio")


def default_groups(features: Sequence[str] = JOINT_FEATURES) -> Dict[str, List[str]]:
    f = list(features)
    return {
        "all": f,
        "no_area": [x for x in f if x not in AREA_FEATURES],
        "no_attachment": [x for x in f if x not in ATTACHMENT_FEATURES],
        "attachment_only": [x for x in f if x in ATTACHMENT_FEATURES],
        "no_area_no_attachment": [x for x in f if x not in AREA_FEATURES and x not in ATTACHMENT_FEATURES],
        "area_only": [x for x in f if x in AREA_FEATURES],
    }


def _complete(frame: pd.DataFrame, features: Sequence[str]) -> np.ndarray:
    x = frame[list(features)].to_numpy(dtype=float)
    return x[np.isfinite(x).all(axis=1)]


def _estimator(model: str, seed: int):
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    if model == "logistic":
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    if model == "gbm":
        return HistGradientBoostingClassifier(random_state=seed)
    raise ValueError(f"model must be logistic or gbm, got {model!r}")


def repeated_cv(real_x: np.ndarray, gen_x: np.ndarray, *, model: str = "logistic", n_splits: int = 5, n_repeats: int = 10,
                seed: int = 0) -> Dict[str, float]:
    """Balanced subsample (n = smaller class) + stratified k-fold, repeated; mean and SD over repeats of accuracy and ROC AUC."""
    from sklearn.model_selection import StratifiedKFold, cross_validate

    n = min(len(real_x), len(gen_x))
    if n < n_splits:
        raise ValueError(f"need at least {n_splits} complete samples per class, got {n}")
    acc, auc = [], []
    for r in range(n_repeats):
        rng = np.random.default_rng(seed + r)
        x = np.vstack([real_x[rng.choice(len(real_x), n, replace=False)], gen_x[rng.choice(len(gen_x), n, replace=False)]])
        y = np.r_[np.zeros(n, dtype=int), np.ones(n, dtype=int)]
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed + r)
        s = cross_validate(_estimator(model, seed + r), x, y, cv=cv, scoring=("accuracy", "roc_auc"))
        acc.append(s["test_accuracy"].mean())
        auc.append(s["test_roc_auc"].mean())
    return {"accuracy": float(np.mean(acc)), "accuracy_sd": float(np.std(acc, ddof=1)) if n_repeats > 1 else 0.0,
            "roc_auc": float(np.mean(auc)), "roc_auc_sd": float(np.std(auc, ddof=1)) if n_repeats > 1 else 0.0, "n_per_class": int(n)}


def coefficients(real_x: np.ndarray, gen_x: np.ndarray, features: Sequence[str], *, n_repeats: int = 10, seed: int = 0) -> pd.DataFrame:
    """Logistic coefficients on standardised features (sign: + = larger in the GENERATED set), mean and SD over balanced subsamples.
    Correlated features share / swap weight, so read them together with the drop-one table."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    n = min(len(real_x), len(gen_x))
    coefs = []
    for r in range(n_repeats):
        rng = np.random.default_rng(seed + 1000 + r)
        x = np.vstack([real_x[rng.choice(len(real_x), n, replace=False)], gen_x[rng.choice(len(gen_x), n, replace=False)]])
        y = np.r_[np.zeros(n, dtype=int), np.ones(n, dtype=int)]
        coefs.append(LogisticRegression(max_iter=2000).fit(StandardScaler().fit_transform(x), y).coef_[0])
    c = np.asarray(coefs)
    return pd.DataFrame({"feature": list(features), "coef_mean": c.mean(axis=0), "coef_sd": c.std(axis=0, ddof=1) if n_repeats > 1 else 0.0})


def standardized_differences(real: pd.DataFrame, gen: pd.DataFrame, features: Sequence[str]) -> pd.DataFrame:
    """(median_gen - median_real) / SD_real and Cohen's d per feature: which features are shifted, independent of any classifier."""
    rows = []
    for f in features:
        a, b = real[f].to_numpy(float), gen[f].to_numpy(float)
        a, b = a[np.isfinite(a)], b[np.isfinite(b)]
        sd = float(np.std(a, ddof=1)) if len(a) > 1 else float("nan")
        pooled = float(np.sqrt((np.var(a, ddof=1) + np.var(b, ddof=1)) / 2)) if len(a) > 1 and len(b) > 1 else float("nan")
        rows.append({"feature": f, "median_real": float(np.median(a)) if len(a) else float("nan"),
                     "median_generated": float(np.median(b)) if len(b) else float("nan"),
                     "median_shift_sd": (float(np.median(b) - np.median(a)) / sd) if sd and np.isfinite(sd) else float("nan"),
                     "cohens_d": (float(np.mean(b) - np.mean(a)) / pooled) if pooled and np.isfinite(pooled) else float("nan")})
    return pd.DataFrame(rows)


def classifier_ablation(real: pd.DataFrame, gen: pd.DataFrame, *, features: Sequence[str] = JOINT_FEATURES,
                        groups: Optional[Mapping[str, Sequence[str]]] = None, model: str = "logistic", n_repeats: int = 10,
                        seed: int = 0) -> Dict[str, pd.DataFrame]:
    """Tables ``subsets`` (group / single feature / all-but-one -> accuracy, AUC, drop vs all), ``coefficients``, ``shifts``.

    Rows with any missing feature are dropped once, for ALL subsets (same samples everywhere, otherwise a subset could look
    better only because it keeps other spines)."""
    features = [f for f in features if f in real.columns and f in gen.columns]
    groups = dict(groups or default_groups(features))
    rx, gx = _complete(real, features), _complete(gen, features)
    col = {f: i for i, f in enumerate(features)}

    def score(subset: Sequence[str]) -> Dict[str, float]:
        idx = [col[f] for f in subset]
        return repeated_cv(rx[:, idx], gx[:, idx], model=model, n_repeats=n_repeats, seed=seed)

    rows = []
    full = score(features)
    for name, subset in groups.items():
        subset = [f for f in subset if f in col]
        if subset:
            rows.append({"kind": "group", "name": name, "features": ",".join(subset), "n_features": len(subset), **score(subset)})
    for f in features:
        rows.append({"kind": "single", "name": f, "features": f, "n_features": 1, **score([f])})
    for f in features:
        rest = [x for x in features if x != f]
        rows.append({"kind": "drop_one", "name": f"without {f}", "features": ",".join(rest), "n_features": len(rest), **score(rest)})
    subsets = pd.DataFrame(rows)
    subsets["auc_minus_all"] = subsets["roc_auc"] - full["roc_auc"]
    subsets["accuracy_minus_all"] = subsets["accuracy"] - full["accuracy"]
    return {"subsets": subsets,
            "coefficients": coefficients(rx, gx, features, n_repeats=n_repeats, seed=seed),
            "shifts": standardized_differences(real, gen, features)}
