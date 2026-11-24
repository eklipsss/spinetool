"""Figures for the predefense of 26.11.2026 (docs/neuron-model/sprints/26.11-predefense-slides.md, list R4, R7, R8).

Tables only, runs on the mac:

    python scripts/neuron_model/make_predefense_figures.py \
        --valfloor legacy/output_win/2026-11-20-valfloor --dose legacy/output_win/AI/2026-11-21_conditional-report-v4-type-dose \
        --cloud-size legacy/output_win/2026-11-22_cloud-size --out docs/neuron-model/sprints/26.11-predefense-figures

Figures (PNG, 200 dpi, palette of html-materials-style-guide.md: real data grey, models pink / green / violet):
  r4_t0_cv_versions.png       T0 versions: number of CV folds (of 5) with a significant deviation, per functional
  r7_type1_error.png          false rejections under equality: naive per-spine KS vs the neuron-level test
  r8a_cloud_size_steps.png    point-cloud size / real along the fine-tuning of A, and per model
  r8b_binary_type_effects.png inhibitory vs excitatory (mouse): real, floor, v3, v4, per metric (%)
  r8c_detailed_type_dose.png  23P vs 6P-CT: real, floor, v4 at guidance 1.5 and 3, per metric (%)
The CV counts of R4 and the error rates of R7 are numbers from the run registry (no table on the mac), see CV_COUNTS / TYPE1.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

PINK, GREEN, VIOLET, GREY, INK = "#d6418f", "#3f8a2a", "#8a6bd1", "#8e9196", "#1d1d1f"
LIGHT_GREY, MUTED = "#c9cbd0", "#4a4d55"

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
    "font.family": "DejaVu Sans", "font.size": 11, "text.color": INK, "axes.labelcolor": INK,
    "axes.edgecolor": LIGHT_GREY, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": "#ececef", "grid.linewidth": 0.8,
    "axes.axisbelow": True, "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "legend.frameon": False, "axes.unicode_minus": False,
})

METRIC_RU = {
    "OpenAngle": "угол раскрытия", "CVD": "CVD", "AverageDistance": "среднее расстояние", "LengthVolumeRatio": "длина / объём",
    "LengthAreaRatio": "длина / площадь", "Length": "длина", "JunctionArea": "площадь основания", "Area": "площадь",
    "Volume": "объём", "ConvexHullVolume": "объём оболочки", "ConvexHullRatio": "оболочка / объём",
}

# R4: CV by neurons, 5 folds; folds with a significant deviation (registry rows X, Y, AA, AH)
CV_COUNTS = {
    "functionals": ["изгиб κ_rms", "извилистость", "автокорреляция\nизгиба (1 мкм)"],
    "versions": [
        ("полувариограммы", [5, 5, 5], LIGHT_GREY),
        ("полувариограммы +\nпоправка по симуляции", [5, 5, 5], GREY),
        ("квантили функционалов", [1, 3, 5], GREEN),
        ("квантили + масштаб ℓ(r),\nмышь + человек", [0, 1, 0], PINK),
    ],
}
# R7: share of false rejections when generated and real come from the same population (simulation, registry 2026-11-04)
TYPE1 = [("KS по шипикам\n(наивный)", 55.0, LIGHT_GREY), ("тест с нейроном\nкак единицей", 0.5, PINK)]


def save(fig, out: Path, name: str) -> None:
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / name, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"-> {out / name}")


def fig_r4(out: Path) -> None:
    funcs, versions = CV_COUNTS["functionals"], CV_COUNTS["versions"]
    fig, ax = plt.subplots(figsize=(8.6, 4.0))
    x = np.arange(len(funcs))
    w = 0.8 / len(versions)
    for k, (label, vals, color) in enumerate(versions):
        pos = x - 0.4 + w * (k + 0.5)
        bars = ax.bar(pos, vals, w * 0.9, color=color, label=label)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.08, str(v), ha="center", va="bottom", fontsize=9, color=MUTED)
    ax.set_xticks(x, funcs)
    ax.set_ylim(0, 5.9)
    ax.set_yticks(range(6))
    ax.set_ylabel("блоков со значимым отклонением из 5")
    ax.set_title("Модуль T: кросс-валидация по нейронам для версий подгонки")
    ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0), fontsize=9)
    save(fig, out, "r4_t0_cv_versions.png")


def fig_r7(out: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    labels = [t[0] for t in TYPE1]
    vals = [t[1] for t in TYPE1]
    bars = ax.bar(labels, vals, 0.55, color=[t[2] for t in TYPE1])
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 1.2, f"{v:g} %".replace(".", ","), ha="center", fontsize=10, color=INK)
    ax.axhline(5, color=MUTED, lw=1, ls="--")
    ax.text(1.45, 6.2, "номинал 5 %", ha="right", fontsize=9, color=MUTED)
    ax.set_ylim(0, 65)
    ax.set_ylabel("ложные отклонения при равенстве, %")
    ax.set_title("Зависимость шипиков внутри нейрона")
    save(fig, out, "r7_type1_error.png")


def fig_r8a(cloud_dir: Path, out: Path) -> None:
    s = pd.read_csv(cloud_dir / "summary.csv")
    measures = [("hull_volume_nm3", "объём оболочки облака"), ("radial_rms_nm", "расстояние точек от оси"), ("length_nm", "длина")]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.0), gridspec_kw={"width_ratios": [1.25, 1]})
    steps = []
    for st in (0, 1000, 3000, 8000, 16000):
        if (s["set"] == f"Acurves-step{st}").any():
            steps.append((st, f"Acurves-step{st}"))
    colors = [PINK, GREEN, VIOLET]
    for (m, lab), c in zip(measures, colors):
        xs, ys, lo, hi = [], [], [], []
        for st, name in steps:
            r = s[(s["set"] == name) & (s["measure"] == m)]
            if len(r):
                xs.append(st), ys.append(r["ratio_to_ref"].iloc[0]), lo.append(r["ratio_lo"].iloc[0]), hi.append(r["ratio_hi"].iloc[0])
        if not xs:
            continue
        a1.errorbar(xs, ys, yerr=[np.subtract(ys, lo), np.subtract(hi, ys)], color=c, marker="o", ms=6, lw=2, capsize=3, label=lab)
    for name, mark in (("A-s1", "s"), ("A-s2", "D")):
        r = s[(s["set"] == name) & (s["measure"] == "hull_volume_nm3")]
        if len(r) and not any(st == 16000 for st, _ in steps):
            a1.plot([16000], [r["ratio_to_ref"].iloc[0]], mark, color=PINK, ms=7, mfc="white", mew=2)
    if not any(st == 16000 for st, _ in steps):
        a1.text(16000, 1.32, "модель A,\nшаг 16 000", ha="center", fontsize=8.5, color=MUTED)
    a1.axhline(1, color=GREY, lw=1.2)
    a1.set_xlabel("шаг дообучения")
    a1.set_ylabel("медиана / медиана реальных")
    a1.set_title("Размер облака по ходу дообучения A")
    a1.legend(fontsize=9, loc="upper right")
    a1.set_xticks([st for st, _ in steps] + ([16000] if not any(st == 16000 for st, _ in steps) else []))
    models = [("A-s1", "A"), ("v3-s1", "v3"), ("v4-6P-CT-g1.5", "v4, 6P-CT"), ("v4-23P-g1.5", "v4, 23P")]
    ys, lo, hi, names = [], [], [], []
    for key, lab in models:
        r = s[(s["set"] == key) & (s["measure"] == "hull_volume_nm3")]
        if len(r):
            ys.append(r["ratio_to_ref"].iloc[0]), lo.append(r["ratio_lo"].iloc[0]), hi.append(r["ratio_hi"].iloc[0]), names.append(lab)
    y = np.arange(len(names))
    a2.errorbar(ys, y, xerr=[np.subtract(ys, lo), np.subtract(hi, ys)], fmt="o", color=PINK, ms=7, capsize=3, lw=2)
    r = s[(s["set"] == "real") & (s["measure"] == "hull_volume_nm3")]
    if len(r):
        a2.axvspan(r["ratio_lo"].iloc[0], r["ratio_hi"].iloc[0], color="#eceef1", zorder=0, label="разброс реальных")
    a2.axvline(1, color=GREY, lw=1.2)
    for yi, v in zip(y, ys):
        a2.text(hi[list(ys).index(v)] + 0.02, yi, f"×{v:.2f}".replace(".", ","), va="center", fontsize=9.5, color=MUTED)
    a2.set_yticks(y, names)
    a2.set_xlabel("объём оболочки облака / реальный")
    a2.set_title("Объём облака по моделям")
    a2.set_xlim(0.8, max(hi) + 0.2)
    a2.legend(fontsize=9, loc="upper left", bbox_to_anchor=(0.0, -0.18))
    save(fig, out, "r8a_cloud_size_steps.png")


def effects_plot(ax, sets, *, title: str, only_sig: bool = True) -> None:
    """sets: list of (label, concordance frame, color, is_reference)."""
    base = sets[0][1].set_index("metric")
    metrics = [m for m in base.index if (not only_sig or bool(base.loc[m, "real_sig"]))]
    metrics = sorted(metrics, key=lambda m: base.loc[m, "lr_real"])
    y = np.arange(len(metrics))
    pct = lambda v: 100 * np.expm1(np.asarray(v, float))  # noqa: E731
    real = pct(base.loc[metrics, "lr_real"])
    rlo, rhi = pct(base.loc[metrics, "lo_real"]), pct(base.loc[metrics, "hi_real"])
    ax.barh(y, real, 0.7, color="#eceef1", edgecolor=LIGHT_GREY, label="реальные (интервал по нейронам)")
    ax.errorbar(real, y, xerr=[real - rlo, rhi - real], fmt="none", ecolor=GREY, capsize=3, lw=1.2)
    n = len(sets)
    for k, (label, frame, color, _) in enumerate(sets):
        f = frame.set_index("metric")
        dy = (k - (n - 1) / 2) * 0.17
        ax.plot(pct(f.loc[metrics, "lr_gen"]), y + dy, "o", color=color, ms=6.5, label=label, mec="white", mew=0.6)
    ax.axvline(0, color=MUTED, lw=1)
    ax.set_yticks(y, [METRIC_RU.get(m, m) for m in metrics])
    ax.set_xlabel("различие групп, % (отношение медиан − 1)")
    ax.set_title(title)
    ax.legend(fontsize=9, loc="lower right")


def fig_r8b(valfloor: Path, out: Path) -> None:
    d = next(valfloor.glob("*conditional-report-v4-mouse-valfloor"))
    sets = [("пол (реальные через цепочку)", pd.read_csv(d / "concordance_real-pw8val.csv"), GREY, True),
            ("v3", pd.read_csv(d / "concordance_v3-pw8.csv"), GREEN, False),
            ("v4", pd.read_csv(d / "concordance_v4-pw8.csv"), PINK, False)]
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    effects_plot(ax, sets, title="Тормозные против возбуждающих (мышь): значимые различия")
    save(fig, out, "r8b_binary_type_effects.png")


def fig_r8c(dose: Path, out: Path) -> None:
    sets = [("пол (реальные через цепочку)", pd.read_csv(dose / "concordance_real-pw8val.csv"), GREY, True),
            ("v4, усиление 1,5", pd.read_csv(dose / "concordance_v4g1.5-pw8.csv"), VIOLET, False),
            ("v4, усиление 3", pd.read_csv(dose / "concordance_v4g3-pw8.csv"), PINK, False)]
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    effects_plot(ax, sets, title="23P против 6P-CT (мышь): подробный тип не передаётся")
    save(fig, out, "r8c_detailed_type_dose.png")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--valfloor", type=Path, required=True)
    ap.add_argument("--dose", type=Path, required=True)
    ap.add_argument("--cloud-size", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    fig_r4(args.out)
    fig_r7(args.out)
    fig_r8a(args.cloud_size, args.out)
    fig_r8b(args.valfloor, args.out)
    fig_r8c(args.dose, args.out)


if __name__ == "__main__":
    main()
