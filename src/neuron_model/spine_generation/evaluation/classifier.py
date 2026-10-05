"""Real-vs-generated classifier on morphometric features (spec: extra diagnostic).

Accuracy ~ 0.5: generated and real are hard to tell apart in this feature space;
high accuracy: a systematic shift. Classes are balanced by subsampling the larger one
(so chance level is exactly 0.5), and accuracy / ROC AUC are cross-validated.
"""

from __future__ import annotations

from typing import Dict

import numpy as np


def real_vs_generated_classifier(
    real_x: np.ndarray,
    gen_x: np.ndarray,
    *,
    model: str = "logistic",
    n_splits: int = 5,
    seed: int = 0,
) -> Dict[str, float]:
    """``model``: ``logistic`` (linear shift) or ``gbm`` (HistGradientBoosting, non-linear)."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_validate
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    rng = np.random.default_rng(seed)
    n = min(len(real_x), len(gen_x))
    if n < n_splits:
        raise ValueError(f"need at least {n_splits} samples per class, got {n}")
    r = np.asarray(real_x, float)[rng.choice(len(real_x), n, replace=False)]
    g = np.asarray(gen_x, float)[rng.choice(len(gen_x), n, replace=False)]
    x = np.vstack([r, g])
    y = np.r_[np.zeros(n, dtype=int), np.ones(n, dtype=int)]
    if model == "logistic":
        estimator = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    elif model == "gbm":
        estimator = HistGradientBoostingClassifier(random_state=seed)
    else:
        raise ValueError(f"model must be logistic or gbm, got {model!r}")
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    scores = cross_validate(estimator, x, y, cv=cv, scoring=("accuracy", "roc_auc"))
    return {
        "accuracy": float(scores["test_accuracy"].mean()),
        "accuracy_std": float(scores["test_accuracy"].std()),
        "roc_auc": float(scores["test_roc_auc"].mean()),
        "n_per_class": int(n),
        "model": model,
    }
