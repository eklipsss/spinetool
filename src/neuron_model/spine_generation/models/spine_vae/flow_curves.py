"""Training curves of the latent-prior flow: NLL per latent dimension, train subset and validation, one estimator."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

BAND = 2.0  # +-2 standard errors over codes


def history_frame(info: Mapping[str, Any]) -> pd.DataFrame:
    return pd.DataFrame(info["history"])


def write_flow_curves(out_dir: Path, info: Mapping[str, Any], gaussian_val_nll: float | None = None) -> None:
    """``flow_history.csv`` and ``flow_curves.png`` in ``out_dir``. Epoch 0 of the x axis is the initial flow (= the fitted
    Gaussian); epoch k >= 1 is the state after k epochs. The best state (early stopping) is marked."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from ....reports import html_kit as hk

    out_dir.mkdir(parents=True, exist_ok=True)
    frame = history_frame(info)
    frame.to_csv(out_dir / "flow_history.csv", index=False)
    p = hk.PALETTE
    init = info.get("init", {})
    x = np.concatenate([[0], frame["epoch"].to_numpy() + 1])
    best_epoch = int(info["best_epoch"]) + 1
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    ax = axes[0]
    for key, se_key, color, label in (("train_eval_nll_per_dim", "train_eval_se", p["series_1"], "train (фиксированное подмножество)"),
                                      ("val_nll_per_dim", "val_se", p["series_2"], "validation")):
        if key not in frame:
            continue
        y = np.concatenate([[init.get(key, np.nan)], frame[key].to_numpy()])
        se = np.concatenate([[np.nan], frame[se_key].to_numpy()])
        ax.plot(x, y, color=color, label=label)
        ax.fill_between(x, y - BAND * se, y + BAND * se, color=color, alpha=0.18, linewidth=0)
    ax.axvline(best_epoch, color=p["series_2"], linestyle=":", linewidth=1.2, label=f"лучшая по validation (эпоха {best_epoch})")
    ax.set_title("Отрицательное логарифмическое правдоподобие кодов на размерность (±2 SE)", fontsize=9, color=p["ink"])
    ax.set_xlabel("эпоха (0 = исходная гауссиана)", fontsize=9)
    ax.grid(color=p["grid"])
    ax.legend(fontsize=7, frameon=False)
    ax = axes[1]
    if "val_nll_per_dim" in frame and "train_eval_nll_per_dim" in frame:
        gap = frame["val_nll_per_dim"].to_numpy() - frame["train_eval_nll_per_dim"].to_numpy()
        se = np.sqrt(frame["val_se"].to_numpy() ** 2 + frame["train_eval_se"].to_numpy() ** 2)  # independent subsets
        ax.plot(frame["epoch"] + 1, gap, color=p["series_3"])
        ax.fill_between(frame["epoch"] + 1, gap - BAND * se, gap + BAND * se, color=p["series_3"], alpha=0.18, linewidth=0)
        ax.axhline(0, color=p["border"], linewidth=1)
        ax.axvline(best_epoch, color=p["series_2"], linestyle=":", linewidth=1.2)
    ax.set_title("Разрыв validation − train (±2 SE)", fontsize=9, color=p["ink"])
    ax.set_xlabel("эпоха", fontsize=9)
    ax.grid(color=p["grid"])
    fig.tight_layout()
    fig.savefig(out_dir / "flow_curves.png", dpi=140, facecolor="white")
    plt.close(fig)
