"""Figures for the sprint presentation (docs/neuron-model/sprints/12.10-25.10/07-figures-for-slides.md).

    # schemes (A) and numbers taken from the sprint report (B): no data needed, seconds
    python scripts/neuron_model/make_sprint_figures.py --static --out docs/neuron-model/sprints/12.10-25.10/figures

    # plots from run results (C); every block is independent and skipped when its input is missing
    python scripts/neuron_model/make_sprint_figures.py --out runs/analysis/neuron-model/sprint_figures \
        --training "MoGen A=runs/training/neuron-model/<mogen-ft run>" "MoGen B=runs/training/neuron-model/<mogen-scratch run>" \
                   "VAE=runs/training/neuron-model/<vae run>" \
        --evaluation runs/analysis/neuron-model/<generator-evaluation run> --kind postprocessed \
        --t0-run runs/training/neuron-model/<id>_t0 \
        --assembled-eval <tp-dir>/assembly_<date>

Colours follow docs/neuron-model/html-materials-style-guide.md (series pink / green / violet, real data grey, white
background). Only reads files; matplotlib on CPU.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle

KIND_RU = {"raw": "сырые сетки", "postprocessed": "после постобработки"}
PINK, GREEN, VIOLET, GREY, INK = "#d6418f", "#3f8a2a", "#8a6bd1", "#8e9196", "#1d1d1f"
SOFT_PINK, SOFT_GREEN, SOFT_GREY = "#fde0ef", "#e6f5d0", "#eceef1"
BRAND_PINK, BRAND_GREEN = "#fd89c3", "#7fbc41"
SERIES = [PINK, GREEN, VIOLET, INK]
# names in the evaluation tables -> names on the slides (and the fixed colour of each model)
MODEL_LABEL = {"base": "MoGen A0", "A": "MoGen A", "B": "MoGen B", "VAE": "VAE"}
MODEL_ORDER = ["base", "A", "B", "VAE"]
MODEL_COLOR = {"base": INK, "A": PINK, "B": GREEN, "VAE": VIOLET}


def model_label(name: str) -> str:
    return MODEL_LABEL.get(name, name)


def order_models(names: Sequence[str]) -> List[str]:
    known = [n for n in MODEL_ORDER if n in names]
    return known + [n for n in names if n not in known]


def model_color(name: str, fallback: int = 0) -> str:
    return MODEL_COLOR.get(name, SERIES[fallback % len(SERIES)])

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
    "font.family": "DejaVu Sans", "font.size": 11, "text.color": INK, "axes.labelcolor": INK,
    "axes.edgecolor": "#c9cbd0", "xtick.color": "#4a4d55", "ytick.color": "#4a4d55",
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": "#ececef", "grid.linewidth": 0.8,
    "axes.axisbelow": True, "axes.titlesize": 14, "figure.titlesize": 15, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "legend.frameon": False, "axes.unicode_minus": False,
})


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def save(fig, out: Path, name: str) -> None:
    out.mkdir(parents=True, exist_ok=True)
    path = out / name
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[figures] {path}")


def box(ax, x, y, w, h, text, *, fc=SOFT_GREY, ec="#c9cbd0", fs=10.5, bold=False, color=INK):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.018", fc=fc, ec=ec, lw=1.2, mutation_aspect=1))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, fontweight="bold" if bold else "normal", color=color, wrap=True)


def arrow(ax, p0, p1, *, color="#6b6e76", lw=1.6, style="-|>", rad=0.0):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle=style, mutation_scale=14, lw=lw, color=color, connectionstyle=f"arc3,rad={rad}"))


def blank_axes(fig, rect=(0, 0, 1, 1)):
    ax = fig.add_axes(rect)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.grid(False)
    return ax


def grouped_bars(ax, labels, series: Mapping[str, Sequence[float]], colors, *, horizontal=False, width=0.8, fmt="{:.2f}", annotate=True):
    n = len(series)
    x = np.arange(len(labels))
    w = width / n
    for i, (name, values) in enumerate(series.items()):
        pos = x - width / 2 + w * (i + 0.5)
        values = np.asarray(values, dtype=float)
        if horizontal:
            bars = ax.barh(pos, values, height=w * 0.92, color=colors[i % len(colors)], label=name)
        else:
            bars = ax.bar(pos, values, width=w * 0.92, color=colors[i % len(colors)], label=name)
        if annotate:
            for b, v in zip(bars, values):
                if not np.isfinite(v):
                    continue
                if horizontal:
                    ax.text(b.get_width(), b.get_y() + b.get_height() / 2, " " + fmt.format(v), va="center", fontsize=8.5, color="#4a4d55")
                else:
                    ax.text(b.get_x() + b.get_width() / 2, b.get_height(), fmt.format(v), ha="center", va="bottom", fontsize=8.5, color="#4a4d55")
    if horizontal:
        ax.set_yticks(x, labels)
        ax.invert_yaxis()
        ax.grid(axis="y", visible=False)
    else:
        ax.set_xticks(x, labels)
        ax.grid(axis="x", visible=False)


# ---------------------------------------------------------------------------
# A: schemes
# ---------------------------------------------------------------------------


def fig_a1_pipeline(out: Path) -> None:
    fig = plt.figure(figsize=(12, 5.6))
    ax = blank_axes(fig)
    ax.set_xlim(-0.02, 1.02)
    ax.text(0.0, 0.99, "Конвейер: от сеток до собранной ветви", fontsize=13.5, fontweight="bold", va="top")
    box(ax, 0.00, 0.38, 0.14, 0.24, "Данные Minnie65\nсетки ветвей,\nскелеты, шипики", fc=SOFT_GREY)
    box(ax, 0.19, 0.38, 0.14, 0.24, "Предобработка\nось, радиус,\nизгиб, положения", fc=SOFT_GREEN, ec=GREEN, bold=True)
    arrow(ax, (0.14, 0.50), (0.19, 0.50))
    ys = {"T": 0.68, "P": 0.38, "S": 0.08}
    texts = {"T": "T: ствол\nизгиб, радиус\nпо длине", "P": "P: размещение\nчисло шипиков и\nих положения", "S": "S: шипик\nформа каждого\nшипика"}
    for key, y in ys.items():
        box(ax, 0.40, y, 0.16, 0.20, texts[key], fc=SOFT_PINK, ec=PINK, bold=True)
        arrow(ax, (0.33, 0.50), (0.40, y + 0.10))
    arrow(ax, (0.48, 0.68), (0.48, 0.58), color=PINK)
    arrow(ax, (0.48, 0.38), (0.48, 0.28), color=PINK)
    ax.text(0.465, 0.63, "ствол задаёт\nположения", fontsize=8.5, color="#4a4d55", va="center", ha="right")
    ax.text(0.465, 0.33, "положение задаёт\nместо шипика", fontsize=8.5, color="#4a4d55", va="center", ha="right")
    box(ax, 0.64, 0.38, 0.14, 0.24, "Сборка\nбулево\nобъединение", fc=SOFT_GREEN, ec=GREEN, bold=True)
    for y in ys.values():
        arrow(ax, (0.56, y + 0.10), (0.636, 0.50))
    box(ax, 0.84, 0.38, 0.16, 0.24, "Оценка\nпарное сравнение\nс реальными ветвями", fc=SOFT_GREY)
    arrow(ax, (0.78, 0.50), (0.84, 0.50))
    box(ax, 0.40, 0.915, 0.16, 0.07, "Условия C и длина L", fc="white", ec=GREY, fs=10)
    arrow(ax, (0.48, 0.915), (0.48, 0.885), color=GREY)
    ax.text(0.0, 0.01, "Условия C: вид животного, тип клетки, компартмент. Каждая часть оценивается отдельно и в сборе.", fontsize=9.5, color="#4a4d55")
    save(fig, out, "a1_pipeline_tps.png")


def _wavy(n=200, seed=1):
    rng = np.random.default_rng(seed)
    x = np.linspace(0, 10, n)
    y = 0.6 * np.sin(x * 0.9) + 0.25 * np.sin(x * 2.3 + 1.0)
    return x, y, rng


def fig_a2_chain(out: Path) -> None:
    fig = plt.figure(figsize=(14, 3.6))
    ax = blank_axes(fig)
    titles = ["1. Сетка ветви\nсо шипиками", "2. Сечения\nперпендикулярно оси", "3. Центральная линия\nпо центрам окружностей", "4. Узлы и рамка\nшаг 250–500 нм", "5. Описание ствола\nв узлах"]
    xs = [0.0, 0.205, 0.41, 0.615, 0.82]
    w = 0.175
    for t, x0 in zip(titles, xs):
        ax.text(x0 + w / 2, 0.97, t, ha="center", va="top", fontsize=10.5, fontweight="bold")
    for x0 in xs[:-1]:
        arrow(ax, (x0 + w + 0.003, 0.5), (x0 + 0.205 - 0.003, 0.5))
    X, Y, rng = _wavy()
    def panel(i):
        a = fig.add_axes((xs[i], 0.08, w, 0.62))
        a.set_axis_off(); a.grid(False)
        return a
    # 1 mesh with spines
    a = panel(0)
    a.fill_between(X, Y - 0.55, Y + 0.55, color=SOFT_GREY, ec="#c9cbd0")
    for xs_ in (1.2, 2.5, 4.1, 5.0, 6.6, 7.9, 9.0):
        k = np.argmin(np.abs(X - xs_)); side = 1 if int(xs_ * 3) % 2 else -1
        a.plot([xs_, xs_ + 0.12 * side], [Y[k] + side * 0.55, Y[k] + side * 1.15], color=GREY, lw=2.2)
        a.add_patch(Circle((xs_ + 0.12 * side, Y[k] + side * 1.22), 0.13, color=GREY))
    a.set_xlim(0, 10); a.set_ylim(-2, 2)
    # 2 sections
    a = panel(1)
    a.fill_between(X, Y - 0.55, Y + 0.55, color=SOFT_GREY, ec="#c9cbd0")
    for xs_ in np.linspace(0.6, 9.4, 12):
        k = np.argmin(np.abs(X - xs_)); slope = np.gradient(Y, X)[k]
        n = np.array([-slope, 1.0]); n /= np.linalg.norm(n)
        a.plot([xs_ - 0.55 * n[0], xs_ + 0.55 * n[0]], [Y[k] - 0.55 * n[1], Y[k] + 0.55 * n[1]], color=PINK, lw=1.4)
        a.plot([xs_], [Y[k]], "o", color=GREEN, ms=3.5)
    a.set_xlim(0, 10); a.set_ylim(-2, 2)
    # 3 centerline
    a = panel(2)
    a.fill_between(X, Y - 0.55, Y + 0.55, color=SOFT_GREY, ec="#c9cbd0")
    a.plot(X, Y, color=PINK, lw=2.4)
    a.set_xlim(0, 10); a.set_ylim(-2, 2)
    # 4 nodes + frame
    a = panel(3)
    a.plot(X, Y, color=PINK, lw=1.2, alpha=0.5)
    idx = np.linspace(8, len(X) - 9, 12).astype(int)
    slope = np.gradient(Y, X)
    for k in idx:
        t = np.array([1, slope[k]]); t /= np.linalg.norm(t); n = np.array([-t[1], t[0]])
        a.plot([X[k]], [Y[k]], "o", color=PINK, ms=5)
        a.annotate("", xy=(X[k] + 0.55 * t[0], Y[k] + 0.55 * t[1]), xytext=(X[k], Y[k]), arrowprops=dict(arrowstyle="-|>", color=GREEN, lw=1.2))
        a.annotate("", xy=(X[k] + 0.45 * n[0], Y[k] + 0.45 * n[1]), xytext=(X[k], Y[k]), arrowprops=dict(arrowstyle="-|>", color=VIOLET, lw=1.0))
    a.set_xlim(0, 10); a.set_ylim(-2, 2)
    # 5 three profiles
    for j, (name, col, f) in enumerate([("κ₁", PINK, np.gradient(np.gradient(Y, X), X)), ("κ₂", GREEN, 0.5 * np.cos(X * 1.7) * 0.4), ("log r", VIOLET, 0.12 * np.sin(X * 0.5) - 0.05 * X / 10)]):
        a = fig.add_axes((xs[4] + 0.02, 0.50 - 0.20 * j + 0.0, w - 0.03, 0.17))
        a.plot(X, f, color=col, lw=1.8); a.set_axis_off(); a.grid(False)
        a.text(-0.4, np.mean(f), name, fontsize=10, ha="right", va="center", color=col, fontweight="bold")
    ax.text(0.5, 0.0, "Положение и ориентация ветви в мозге в описание не входят: модель изучает форму, а не координаты", ha="center", fontsize=9.5, color="#4a4d55")
    save(fig, out, "a2_preprocessing_chain.png")


def fig_a3_cross_section(out: Path) -> None:
    from src.neuron_model import branch_geometry as bg

    rng = np.random.default_rng(3)
    ang = rng.uniform(0, 2 * np.pi, 150)
    r_true = 1.0
    pts = np.c_[np.cos(ang), np.sin(ang)] * (r_true + rng.normal(0, 0.03, len(ang)))[:, None]
    # unsegmented spine base on the right: points further out
    base_ang = rng.uniform(-0.45, 0.45, 38)
    base_r = rng.uniform(1.05, 1.9, 38)
    base = np.c_[np.cos(base_ang), np.sin(base_ang)] * base_r[:, None]
    allp = np.vstack([pts[np.abs(((ang + np.pi) % (2 * np.pi)) - np.pi) > 0.45], base])
    c_naive, r_naive = bg.fit_circle_2d(allp)
    c_rob, r_rob = bg.robust_fit_circle_2d(allp)
    fig, ax = plt.subplots(figsize=(6.4, 6.0))
    ax.set_aspect("equal"); ax.grid(False)
    ax.add_patch(Circle((0, 0), r_true, fill=False, ec="#c9cbd0", lw=1.0, ls=":"))
    ax.scatter(*allp[: len(allp) - len(base)].T, s=14, color=GREY, label="вершины ствола в срезе")
    ax.scatter(*base.T, s=14, color=VIOLET, label="основание шипика, не отделённое сегментацией")
    t = np.linspace(0, 2 * np.pi, 200)
    ax.plot(c_naive[0] + r_naive * np.cos(t), c_naive[1] + r_naive * np.sin(t), color=GREY, ls="--", lw=1.6, label=f"подгонка по всем точкам (r = {r_naive:.2f})")
    ax.plot(c_rob[0] + r_rob * np.cos(t), c_rob[1] + r_rob * np.sin(t), color=PINK, lw=2.2, label=f"с отбрасыванием выбросов (r = {r_rob:.2f})")
    ax.plot(*c_rob, "o", color=PINK, ms=8); ax.text(c_rob[0] + 0.08, c_rob[1] - 0.22, "центр = точка оси", color=PINK, fontsize=10, ha="left")
    sk = np.array([np.cos(np.radians(215)), np.sin(np.radians(215))]) * 1.0
    ax.plot(*sk, "s", color=GREEN, ms=9); ax.text(sk[0] - 0.08, sk[1] - 0.12, "точка скелета\nиз данных", color=GREEN, fontsize=10, ha="right", va="top")
    ax.annotate("", xy=sk, xytext=c_rob, arrowprops=dict(arrowstyle="<->", color=GREEN, lw=1.2))
    mid = (sk + c_rob) / 2; ax.text(mid[0] - 0.12, mid[1] + 0.10, "≈ r", color=GREEN, fontsize=11, fontweight="bold")
    ax.set_xlim(-2.0, 2.2); ax.set_ylim(-1.6, 1.6)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("Сечение ствола: скелет лежит на мембране, ось находится по окружности")
    ax.legend(loc="lower right", fontsize=8.2, bbox_to_anchor=(1.0, -0.26), ncol=1)
    save(fig, out, "a3_cross_section_fit.png")


def _simulate_paths(kind: str, n_paths: int, length_um=80.0, step=0.25, seed=0, kappa_rms=0.25):
    """2-D toy of the heading: 'ou' = OU bending (heading is its integral), 'tilt' = restoring tilt driven by a smooth OU rate."""
    rng = np.random.default_rng(seed)
    m = int(length_um / step)
    ell, l_g, L_d = 1.0, 3.4, 3.0
    paths, tort = [], []
    for _ in range(n_paths):
        if kind == "ou":
            phi = math.exp(-step / ell)
            k = np.zeros(m); k[0] = rng.normal(0, kappa_rms)
            for j in range(m - 1):
                k[j + 1] = phi * k[j] + rng.normal(0, kappa_rms * math.sqrt(1 - phi**2))
            theta = np.cumsum(k) * step
        else:
            phi = math.exp(-step / l_g); a = math.exp(-step / L_d)
            g = rng.normal(0, 1.0); tilt = np.zeros(m)
            for j in range(m - 1):
                tilt[j + 1] = a * tilt[j] + step * g
                g = phi * g + rng.normal(0, math.sqrt(1 - phi**2))
            theta = np.arctan(tilt)
            k_rms = np.sqrt(np.mean(np.diff(theta) ** 2)) / step
            theta = theta * (kappa_rms / max(k_rms, 1e-12))  # match the bending rms of the OU model
        xy = np.c_[np.cumsum(np.cos(theta)) * step, np.cumsum(np.sin(theta)) * step]
        paths.append(xy)
        tort.append(((m - 1) * step) / max(float(np.linalg.norm(xy[-1] - xy[0])), 1e-9))
    return paths, np.array(tort)


def fig_a4_tortuosity(out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.8))
    allxy = []
    sets = []
    for kind, title, col in [("ou", "Изгиб — процесс О.–У.: направление блуждает", VIOLET), ("tilt", "Наклон с возвратом к оси: направление удержано", PINK)]:
        paths, tort = _simulate_paths(kind, 9, seed=4 if kind == "ou" else 5)
        paths = [xy - xy[0] for xy in paths]
        allxy += paths
        sets.append((kind, title, col, paths, tort))
    xs = np.concatenate([p[:, 0] for p in allxy]); ys = np.concatenate([p[:, 1] for p in allxy])
    xl, yl = (xs.min() - 3, xs.max() + 3), (ys.min() - 3, ys.max() + 3)
    for ax, (kind, title, col, paths, tort) in zip(axes, sets):
        for xy in paths:
            ax.plot(xy[:, 0], xy[:, 1], color=col, lw=1.4, alpha=0.85)
        ax.plot([0], [0], "o", color=INK, ms=5)
        ax.set_xlim(*xl); ax.set_ylim(*yl); ax.set_aspect("equal"); ax.grid(False)
        ax.set_title(title, fontsize=13)
        ax.text(0.02, 0.03, f"медиана извилистости на 80 мкм: {np.median(tort):.2f}", transform=ax.transAxes, fontsize=10, color=INK)
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("Одинаковый среднеквадратичный изгиб, разное направление (упрощённая двумерная иллюстрация, девять ветвей по 80 мкм из одной точки)", fontsize=12.5, x=0.01, ha="left")
    save(fig, out, "a4_tortuosity_mechanism.png")


def fig_a5_assembly(out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.9), gridspec_kw={"width_ratios": [1, 1.25]})
    ax = axes[0]; ax.set_aspect("equal"); ax.grid(False)
    R = 1.0
    ax.add_patch(Circle((0, 0), R, fc=SOFT_GREY, ec="#9a9da5", lw=1.4))
    ax.text(0, -0.05, "ствол", ha="center", color="#4a4d55")
    delta = 0.22
    top = R
    ax.plot([-0.22, 0.22], [top - delta, top - delta], color=PINK, lw=3)
    ax.text(0.36, top - delta - 0.12, "основание\n(крышка)", color=PINK, fontsize=9.5, va="top")
    ax.fill([-0.22, 0.22, 0.14, 0.12, 0.30, 0.0, -0.30, -0.12, -0.14], [top - delta, top - delta, top + 0.35, top + 0.65, top + 1.0, top + 1.18, top + 1.0, top + 0.65, top + 0.35], color=BRAND_PINK, ec=PINK, lw=1.4, alpha=0.9, zorder=3)
    ax.annotate("", xy=(-0.45, top - delta), xytext=(-0.45, top), arrowprops=dict(arrowstyle="<->", color=INK, lw=1.3))
    ax.plot([-0.5, -0.1], [top, top], color="#9a9da5", lw=0.8, ls=":")
    ax.text(-0.52, top - delta / 2, "δ", ha="right", va="center", fontsize=13, fontweight="bold")
    ax.set_xlim(-1.6, 1.6); ax.set_ylim(-1.2, 2.5); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("Шипик погружается основанием в ствол")
    ax = axes[1]
    w = 0.8
    d = np.linspace(0, 2 * R, 600)
    xs_ = np.linspace(-w, w, 41)
    frac = np.array([np.mean(xs_**2 + (R - dd) ** 2 < R**2) for dd in d])
    ax.plot(d, frac, color=PINK, lw=2.6, zorder=3)
    ok = frac >= 0.999
    lo_, hi_ = d[ok][0], d[ok][-1]
    ax.axvspan(0, lo_, color="#f4f4f6"); ax.axvspan(lo_, hi_, color=SOFT_GREEN); ax.axvspan(hi_, 2 * R, color=SOFT_PINK)
    ax.text(lo_ / 2, 0.45, "слишком\nмелко:\nоснование\nснаружи", ha="center", fontsize=8.8, color="#4a4d55")
    ax.text((lo_ + hi_) / 2, 0.5, "основание целиком\nвнутри ствола", ha="center", fontsize=10, color=GREEN, fontweight="bold")
    ax.text((hi_ + 2 * R) / 2 + 0.02, 0.45, "слишком глубоко:\nвыходит с другой\nстороны", ha="center", fontsize=8.8, color=PINK)
    ax.axvline(R, color=GREY, ls=":", lw=1.2); ax.text(R + 0.02, 1.03, "глубина = радиус", color="#4a4d55", fontsize=9)
    ax.set_xlabel("глубина погружения δ / радиус ствола"); ax.set_ylabel("доля вершин основания внутри ствола")
    ax.set_title("Функция немонотонна: простой перебор «глубже, пока внутри» не работает", fontsize=13)
    ax.set_ylim(-0.02, 1.12); ax.set_xlim(0, 2 * R)
    save(fig, out, "a5_assembly_immersion.png")


def fig_a6_flow_matching(out: Path) -> None:
    rng = np.random.default_rng(7)
    n = 350
    # data: a spine-like 2D shape: round head on top of a thin neck
    th = rng.uniform(0, 2 * np.pi, n // 2)
    head = np.c_[0.55 * np.cos(th), 1.3 + 0.55 * np.sin(th)]
    neck = np.c_[rng.normal(0, 0.06, n // 2), rng.uniform(0, 0.8, n // 2)]
    x1 = np.vstack([head, neck])
    x0 = rng.normal(0, 0.7, (len(x1), 2)) + np.array([0.0, 0.7])
    x1 = x1[rng.permutation(len(x1))]
    ts = [0.0, 0.25, 0.5, 0.75, 1.0]
    fig, axes = plt.subplots(1, 5, figsize=(14, 3.5), sharex=True, sharey=True)
    for ax, t in zip(axes, ts):
        xt = (1 - t) * x0 + t * x1
        if t == 0.0:
            for k in range(0, len(x0), 14):
                ax.plot([x0[k, 0], x1[k, 0]], [x0[k, 1], x1[k, 1]], color="#e3e4e8", lw=0.8, zorder=1)
        ax.scatter(xt[:, 0], xt[:, 1], s=7, color=PINK if t > 0 else GREY, zorder=2)
        ax.set_aspect("equal"); ax.grid(False); ax.set_xticks([]); ax.set_yticks([])
        ax.set_title({0.0: "t = 0: шум", 1.0: "t = 1: форма шипика"}.get(t, f"t = {t:g}"), fontsize=12.5)
    fig.suptitle("Flow Matching на плоскости: каждая точка движется по прямой от шума к форме; сеть учится угадывать эту скорость", fontsize=13, x=0.01, ha="left", y=0.98)
    save(fig, out, "a6_flow_matching_toy.png")


def fig_a7_spine_models(out: Path) -> None:
    fig = plt.figure(figsize=(14, 6.2))
    ax = blank_axes(fig)
    ax.set_xlim(-0.02, 1.02)
    ax.text(0.0, 0.99, "Три модели генерации шипиков и вопросы сравнения", fontsize=14, fontweight="bold", va="top")
    rows = [
        (0.66, "A, B   MoGen: облако точек", PINK, ["шум\n8192 точки", "сеть v_θ(x, t)\nPointInfinity", "облако точек\nшипика", "нормали\nPCA + знаки", "Screened\nPoisson"], "A: предобучение + дообучение\nB: та же архитектура с нуля"),
        (0.36, "C   VAE: неявная поверхность", GREEN, ["облако точек\n+ нормали", "энкодер\nPointNeXt", "код z\n128 чисел", "декодер SIREN\nf(q, z) = расстояние", "Marching\nCubes"], "C: расстояние до поверхности\nвместо облака точек"),
    ]
    for y, label, col, steps, q in rows:
        ax.text(0.0, y + 0.20, label, fontsize=12, fontweight="bold", color=col)
        xs = [0.0 + 0.13 * i for i in range(len(steps))]
        for i, (x, st) in enumerate(zip(xs, steps)):
            box(ax, x, y, 0.105, 0.15, st, fc=SOFT_PINK if col == PINK else SOFT_GREEN, ec=col, fs=8.8)
            if i:
                arrow(ax, (xs[i - 1] + 0.108, y + 0.075), (x - 0.003, y + 0.075), color=col)
        arrow(ax, (xs[-1] + 0.108, y + 0.075), (0.665, y + 0.075), color=col)
        box(ax, 0.67, y + 0.01, 0.085, 0.13, "сетка", fc="white", ec=col, bold=True)
        ax.text(0.78, y + 0.075, q, fontsize=9.6, va="center", color="#4a4d55")
    box(ax, 0.0, 0.04, 0.99, 0.20, "", fc="white", ec="#c9cbd0")
    ax.text(0.012, 0.225, "Что отвечает сравнение", fontsize=10.5, fontweight="bold", va="top")
    ax.text(0.012, 0.185, "A против B: помогает ли предобучение.      C против MoGen: лучше ли неявное представление поверхности.\n"
            "MoGen без дообучения (базовая линия): сколько даёт само дообучение.\nОценка — одним протоколом: валидность, морфометрия, разнообразие, запоминание.", fontsize=9.8, va="top", color="#4a4d55", linespacing=1.6)
    save(fig, out, "a7_spine_models_scheme.png")


# ---------------------------------------------------------------------------
# B: numbers from the sprint report
# ---------------------------------------------------------------------------

VALIDITY_200 = {  # 200 samples, seed 1, same reconstruction (generate_mogen.py summaries)
    "MoGen без дообучения": {"valid_raw": 11.0, "valid_post": 27.0, "watertight": 69.0, "single_post": 56.5, "genus0": 33.0, "posvol_post": 69.0},
    "A: дообучение (шаг 16 000)": {"valid_raw": 50.5, "valid_post": 79.0, "watertight": 95.5, "single_post": 84.0, "genus0": 85.0, "posvol_post": 95.5},
    "B: с нуля (шаг 60 000)": {"valid_raw": 52.0, "valid_post": 83.0, "watertight": 97.0, "single_post": 86.0, "genus0": 74.5, "posvol_post": 97.0},
    "C: VAE (шаг 30 000)": {"valid_raw": 80.5, "valid_post": 81.5, "watertight": 100.0, "single_post": 81.5, "genus0": 100.0, "posvol_post": 100.0},
}


def fig_b1_smoothing(out: Path) -> None:
    classes = ["r < 150 нм", "150–250 нм", "r ≥ 250 нм"]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3))
    grouped_bars(axes[0], classes, {"окно 1 мкм": [0.32, 0.58, 0.96], "адаптивное (k = 2)": [0.84, 0.91, 1.00]}, [GREY, PINK])
    axes[0].set_title("Доля линии внутри сетки"); axes[0].set_ylim(0, 1.15); axes[0].legend(loc="lower right")
    grouped_bars(axes[1], classes, {"окно 1 мкм": [0.48, 0.42, 0.26], "адаптивное (k = 2)": [0.28, 0.29, 0.21]}, [GREY, PINK])
    axes[1].set_title("Ошибка круглой трубки / радиус (меньше — лучше)"); axes[1].set_ylim(0, 0.6)
    fig.suptitle("Адаптивное сглаживание центральной линии: 120 ветвей, 10 нейронов", fontsize=13, x=0.01, ha="left")
    save(fig, out, "b1_smoothing_adaptive.png")


def fig_b2_t0(out: Path) -> None:
    classes = ["< 20 мкм", "20–50 мкм", "≥ 50 мкм"]
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.5), gridspec_kw={"width_ratios": [1.5, 1]})
    series = {"реальные ветви": [1.02, 1.04, 1.09], "изгиб — процесс О.–У.": [1.21, 2.13, 2.71], "возврат к оси, общая амплитуда": [1.16, 1.23, 1.31], "возврат к оси, амплитуда ветви": [1.02, 1.09, 1.15]}
    grouped_bars(axes[0], classes, series, [GREY, VIOLET, GREEN, PINK], fmt="{:.2f}")
    axes[0].set_title("Извилистость по классам длины (медианы, без отбраковки)"); axes[0].set_ylim(0.9, 3.0); axes[0].legend(loc="upper left", fontsize=9)
    ax = axes[1]
    labels = ["О.–У.", "общая\nамплитуда", "амплитуда\nветви"]
    vals = [1.48, 1.56, 0.52]
    bars = ax.bar(labels, vals, color=[VIOLET, GREEN, PINK], width=0.6)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.2f}", ha="center", va="bottom", fontsize=10)
    ax.axhline(0.31, color=INK, ls="--", lw=1.2); ax.text(-0.45, 0.34, "шумовой пол 0,31", ha="left", fontsize=9.5)
    ax.set_title("Расхождение с реальностью (W₁/σ)"); ax.set_ylim(0, 1.8); ax.grid(axis="x", visible=False)
    save(fig, out, "b2_t0_versions.png")


def fig_b3_validity(out: Path, data: Optional[Mapping[str, Mapping[str, float]]] = None) -> None:
    data = data or VALIDITY_200
    keys = [("valid_raw", "валидные, сырые"), ("valid_post", "валидные, после постобработки"), ("watertight", "замкнутые"),
            ("single_post", "одна компонента (после постобр.)"), ("genus0", "род 0 (нет сквозных отверстий)"), ("posvol_post", "положительный объём (после постобр.)")]
    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    series = {name: [d.get(k, np.nan) for k, _ in keys] for name, d in data.items()}
    grouped_bars(ax, [l for _, l in keys], series, [GREY, PINK, GREEN, VIOLET], horizontal=True, fmt="{:g} %")
    ax.set_xlim(0, 112); ax.set_xlabel("доля сеток, % (200 шипиков, одно зерно)")
    ax.set_title("Сгенерированные сетки: валидность и топология"); ax.legend(loc="upper center", bbox_to_anchor=(0.45, -0.13), ncol=2, fontsize=9)
    save(fig, out, "b3_spine_validity.png")


def fig_b4_vae(out: Path) -> None:
    labels = ["регуляризация\nна шаге 30 000", "с первого шага,\nпостоянная скорость", "с первого шага +\nкосинусное затухание"]
    cols = [GREY, VIOLET, PINK]
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.3))
    for ax, vals, title, fmt in [(axes[0], [4, 74, 90], "Валидные сырые сетки, %", "{:.0f}"), (axes[1], [255, 45, 31], "Расстояние Чамфера L₁, нм (меньше — лучше)", "{:.0f}"), (axes[2], [0.35, 0.69, 0.84], "F-мера при пороге 50 нм", "{:.2f}")]:
        bars = ax.bar(range(3), vals, color=cols, width=0.62)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v, fmt.format(v), ha="center", va="bottom", fontsize=10)
        ax.set_xticks(range(3), labels, fontsize=8.6); ax.set_title(title, fontsize=12.5); ax.grid(axis="x", visible=False)
        ax.set_ylim(0, max(vals) * 1.18)
    fig.suptitle("VAE: три рецепта обучения (50 проверочных шипиков, шаг 30 000)", fontsize=13, x=0.01, ha="left")
    save(fig, out, "b4_vae_recipes.png")


def fig_b5_assembly(out: Path) -> None:
    # 20 branches generated by T0 + P0 with spines of the fine-tuned MoGen (assembly 2026-10-22), against the real source branches of the same requests
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.2))
    grouped_bars(axes[0], ["все\nветви", "< 20\nмкм", "20–50\nмкм", "≥ 50\nмкм"], {"реальные ветви": [1.00, 0.70, 1.13, 1.03], "собранные ветви": [0.52, 0.22, 0.45, 0.65]}, [GREY, PINK])
    axes[0].set_title("Плотность шипиков, на мкм"); axes[0].set_ylim(0, 1.45); axes[0].legend(loc="upper right", fontsize=9)
    grouped_bars(axes[1], ["разброс радиуса\n(коэфф. вариации)"], {"реальные ветви": [0.126], "собранные ветви": [0.074]}, [GREY, PINK], width=0.6, fmt="{:.3f}")
    axes[1].set_title("Разброс радиуса вдоль ветви"); axes[1].set_ylim(0, 0.16)
    grouped_bars(axes[2], ["расстояние до ближайшего\nшипика, нм (медиана)"], {"реальные ветви": [791], "собранные ветви": [854]}, [GREY, PINK], width=0.6, fmt="{:.0f}")
    axes[2].set_title("Расстояние до ближайшего соседа"); axes[2].set_ylim(0, 1000)
    fig.suptitle("Собранные ветви (шипики MoGen после дообучения) против реальных ветвей тех же запросов\n"
                 "20 ветвей, один проверочный нейрон. Положения вдоль ветви: W₁/σ = 0,05 при шумовом поле 0,13", fontsize=12, x=0.01, y=0.99, ha="left")
    fig.subplots_adjust(top=0.78, wspace=0.28)
    save(fig, out, "b5_assembly_vs_real.png")


def fig_b6_audit(out: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.2))
    ax = axes[0]
    bars = ax.bar(["круглая\nтрубка"], [0.67], color=PINK, width=0.45)
    ax.text(0, 0.67, "0,67", ha="center", va="bottom", fontsize=11)
    ax.axhline(0.25, color=GREEN, lw=2); ax.text(-0.22, 0.27, "критерий 0,25 r", color=GREEN, fontsize=9.5, ha="left")
    ax.set_ylim(0, 0.85); ax.set_title("Ошибка трубки / r (медиана)"); ax.grid(axis="x", visible=False)
    ax = axes[1]
    ax.bar(["ближе 1,75 r\nк линии", "дальше 1,75 r\n(выросты)"], [78, 22], color=[GREY, VIOLET], width=0.55)
    for i, v in enumerate([78, 22]):
        ax.text(i, v, f"{v} %", ha="center", va="bottom", fontsize=11)
    ax.set_ylim(0, 95); ax.set_title("Доля площади ствола"); ax.grid(axis="x", visible=False)
    ax = axes[2]
    grouped_bars(ax, ["«ядро» ствола"], {"круглое сечение": [0.27], "эллипс": [0.25]}, [PINK, GREEN], width=0.6)
    ax.set_ylim(0, 0.34); ax.set_title("Ошибка на «ядре», доля радиуса")
    fig.suptitle("Аудит обратной реконструкции: ошибку определяют выросты сегментации, эллипс почти не помогает", fontsize=13, x=0.01, ha="left")
    save(fig, out, "b6_audit_decomposition.png")


def _tilt_components(length_um=40.0, step=0.25, seed=11):
    """One transverse component of the T0 tilt (toy parameters of the order of the fitted ones, um):
    q_slow' = -q_slow / L_d + g, g = OU(sd sigma_g, memory l_g); w = Gaussian-kernel GP; eps = white noise of the node."""
    rng = np.random.default_rng(seed)
    m = int(length_um / step) + 1
    s = np.arange(m) * step
    L_d, l_g, sigma_g = 13.0, 1.35, 0.06
    a, phi = math.exp(-step / L_d), math.exp(-step / l_g)
    q_slow, g = np.zeros(m), np.zeros(m)
    g[0] = rng.normal(0, sigma_g)
    for j in range(m - 1):
        q_slow[j + 1] = a * q_slow[j] + step * g[j]
        g[j + 1] = phi * g[j] + rng.normal(0, sigma_g * math.sqrt(1 - phi**2))
    sd_w, l_w = 0.12, 2.0
    cov = sd_w**2 * np.exp(-((s[:, None] - s[None, :]) / l_w) ** 2) + 1e-10 * np.eye(m)
    w = np.linalg.cholesky(cov) @ rng.normal(size=m)
    eps = rng.normal(0, 0.012, m)
    return s, q_slow, w, eps


def _draw_tilt_geometry(ax, formula_box: bool = True, title: str = "а) Наклон: отклонение касательной от оси ветви") -> None:
    """Planar toy branch: chord axis, tangent at one node, its parts along / across the axis, angle alpha."""
    # uses a short, strongly bent toy branch so that the tangent and the chord are visible at equal aspect;
    # the tilt is measured against the chord exactly as in direction.tilt_series
    xa = np.linspace(0, 12, 400)
    xy = np.c_[xa, 1.1 * np.sin(2 * np.pi * xa / 10) + 0.35 * np.sin(2 * np.pi * xa / 3.3) + 0.12 * xa]
    d = np.gradient(xy, axis=0)
    t = d / np.linalg.norm(d, axis=1)[:, None]
    chord = xy[-1] - xy[0]
    ax_dir = chord / np.linalg.norm(chord)
    u = np.array([-ax_dir[1], ax_dir[0]])
    q_meas = (t @ u) / (t @ ax_dir)

    ax.plot(xy[:, 0], xy[:, 1], color=GREY, lw=7, solid_capstyle="round", alpha=0.35)
    ax.plot(xy[:, 0], xy[:, 1], color=INK, lw=1.6)
    ax.plot([xy[0, 0], xy[-1, 0]], [xy[0, 1], xy[-1, 1]], ls="--", color=GREEN, lw=1.8)
    ax.text(xy[-1, 0], xy[-1, 1] + 0.35, "хорда = ось ветви $\\mathbf{a}$", color=GREEN, ha="right", va="bottom", fontsize=11)
    j = 300
    p = xy[j]
    L = 2.6
    tj = t[j]
    along, across = (tj @ ax_dir) * L, (tj @ u) * L
    end_along = p + ax_dir * along
    ax.plot([p[0], end_along[0]], [p[1], end_along[1]], color=GREEN, lw=2.0)
    ax.plot([end_along[0], end_along[0] + u[0] * across], [end_along[1], end_along[1] + u[1] * across], color=VIOLET, lw=2.4)
    arrow(ax, p, p + tj * L, color=PINK, lw=2.6)
    ang0, ang1 = math.degrees(math.atan2(ax_dir[1], ax_dir[0])), math.degrees(math.atan2(tj[1], tj[0]))
    from matplotlib.patches import Arc
    ax.add_patch(Arc(p, 1.6, 1.6, theta1=min(ang0, ang1), theta2=max(ang0, ang1), color=INK, lw=1.2))
    mid = math.radians((ang0 + ang1) / 2)
    ax.text(p[0] + 1.0 * math.cos(mid), p[1] + 1.0 * math.sin(mid), "$\\alpha$", fontsize=12, ha="left", va="center")
    ax.scatter([p[0]], [p[1]], s=40, color=PINK, zorder=5)
    tip = p + tj * L
    ax.text(p[0] - 0.2, p[1] + 0.5, "касательная $\\mathbf{t}(s)$", color=PINK, fontsize=11, ha="right", va="bottom")
    ax.text(end_along[0] + u[0] * across / 2 + 0.2, end_along[1] + u[1] * across / 2, "$\\mathbf{t}\\cdot\\mathbf{u}$ (вбок)",
            color=VIOLET, fontsize=10.5, va="center")
    ax.text(p[0] + ax_dir[0] * along / 2 + 0.25, p[1] + ax_dir[1] * along / 2 - 0.3, "$\\mathbf{t}\\cdot\\mathbf{a}$ (вдоль оси)",
            color=GREEN, fontsize=10.5, ha="left", va="top")
    ax.set_title(title)
    if formula_box:
        ax.text(0.02, 0.04, "$q=\\dfrac{\\mathbf{t}\\cdot\\mathbf{u}}{\\mathbf{t}\\cdot\\mathbf{a}}=\\tan\\alpha$\n"
                "в 3D две поперечные оси $\\mathbf{u}_1,\\mathbf{u}_2$:\n$\\mathbf{q}=(q_1,q_2)$, вектор в $\\mathbb{R}^2$",
                transform=ax.transAxes, va="bottom", fontsize=11, bbox=dict(fc="white", ec="#c9cbd0", boxstyle="round,pad=0.4"))
    ax.set_aspect("equal")
    ax.set_ylim(-2.4, 3.6)
    ax.set_xlabel("мкм"); ax.set_ylabel("мкм")



def fig_a8_tilt(out: Path) -> None:
    """What the tilt q(s) is, how it is obtained from the bending (fit) and back (generation), its three components."""
    s, q_slow, w, eps = _tilt_components()
    q = q_slow + w + eps
    fig = plt.figure(figsize=(15, 9.6))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.05, 1.0], width_ratios=[1.0, 1.0], hspace=0.3, wspace=0.12)

    # (a) geometry of the tilt
    _draw_tilt_geometry(fig.add_subplot(gs[0, 0]))

    # (b) data <-> tilt
    axb = fig.add_subplot(gs[0, 1]); axb.axis("off"); axb.grid(False)
    axb.set_xlim(0, 1); axb.set_ylim(0, 1)
    axb.set_title("б) Как наклон получается из данных и обратно")
    bw, bh, y0 = 0.3, 0.24, 0.52
    xs = [0.0, 0.35, 0.7]
    labels = ["изгиб в узлах\n$\\kappa_1(s_j),\\ \\kappa_2(s_j)$\n(предобработка)", "касательные\n$\\mathbf{t}_j$\n(интегрирование\nрамки)", "наклон\n$\\mathbf{q}_j=(q_1,q_2)_j$\n(проекция\nна хорду)"]
    fcs = [SOFT_GREY, SOFT_GREY, SOFT_PINK]
    for x0, lab, fc in zip(xs, labels, fcs):
        box(axb, x0, y0, bw, bh, lab, fc=fc, fs=9.6)
    for x0 in xs[:-1]:
        arrow(axb, (x0 + bw + 0.004, y0 + bh * 0.68), (x0 + 0.35 - 0.004, y0 + bh * 0.68), color=GREEN, lw=2)
        arrow(axb, (x0 + 0.35 - 0.004, y0 + bh * 0.3), (x0 + bw + 0.004, y0 + bh * 0.3), color=PINK, lw=2)
    axb.text(0.5, y0 + bh + 0.07, "подгонка: реальная ветвь $\\rightarrow$ наклон", color=GREEN, ha="center", fontsize=11, fontweight="bold")
    axb.text(0.5, y0 - 0.07, "генерация: наклон $\\rightarrow$ $\\mathbf{t}=(\\mathbf{q},1)/\\|(\\mathbf{q},1)\\|$ $\\rightarrow$ $\\kappa_1,\\kappa_2$",
             color=PINK, ha="center", fontsize=11, fontweight="bold")
    axb.text(0.0, 0.3, "Предобработка хранит изгиб $\\kappa_1,\\kappa_2$ в рамке, минимизирующей вращение.\n"
             "Модель описывает наклон $\\mathbf{q}$, потому что у него есть «ноль» — ось ветви,\n"
             "и к нему можно возвращать (ограниченная извилистость). Изгиб $\\approx d\\mathbf{q}/ds$:\n"
             "это скорость, наклон — положение. Остальная часть T0 и сборка видят только $\\kappa$.",
             fontsize=10.2, va="top", color="#4a4d55")

    # (c) components
    axc = fig.add_subplot(gs[1, :])
    axc.plot(s, q_slow, color=PINK, lw=2.2, label="$q_{\\mathrm{slow}}$: медленный поворот, возврат к оси за $L_d$")
    axc.plot(s, w, color=GREEN, lw=1.8, label="$w$: гладкие извивы, масштаб $\\ell_w(r)\\approx2$ мкм")
    axc.plot(s, eps, color=VIOLET, lw=0, marker="o", ms=2.6, label="$\\varepsilon$: шум узла")
    axc.plot(s, q, color=INK, lw=1.4, ls="--", label="сумма $q=q_{\\mathrm{slow}}+w+\\varepsilon$ (до множителя $m_b$)")
    axc.axhline(0, color="#9a9ca3", lw=1)
    axc.set_xlabel("длина дуги $s$, мкм"); axc.set_ylabel("одна компонента наклона $q_1$")
    axc.set_title("в) Из чего складывается наклон (одна поперечная компонента, параметры порядка подогнанных)")
    axc.legend(loc="upper right", ncol=2, fontsize=10)
    lo, hi = min(q.min(), q_slow.min(), w.min()), max(q.max(), q_slow.max(), w.max())
    axc.set_ylim(lo - 0.05, hi + 0.18)
    save(fig, out, "r10_tilt_explained.png")


def fig_a8a_tilt_geometry(out: Path) -> None:
    """Panel (a) of r10 alone, without the formula box."""
    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    _draw_tilt_geometry(ax, formula_box=False, title="Наклон: отклонение касательной от оси ветви")
    save(fig, out, "r10a_tilt_geometry.png")


def _toy_branch_3d(n=600):
    """3-D toy branch whose chord is the z axis (ends at x = y = 0), arc-length tangents, um."""
    z = np.linspace(0, 10, n)
    xyz = np.c_[0.9 * np.sin(2 * np.pi * z / 10) + 0.25 * np.sin(2 * np.pi * z / 2.7) * np.sin(np.pi * z / 10),
                0.8 * np.sin(np.pi * z / 10) * np.cos(2 * np.pi * z / 7), z]
    d = np.gradient(xyz, axis=0)
    t = d / np.linalg.norm(d, axis=1)[:, None]
    s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(xyz, axis=0), axis=1))]
    return xyz, t, s


def _style_3d(ax, lim=None):
    ax.grid(False)
    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        pane.set_facecolor("white"); pane.set_edgecolor("#e3e4e8")
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    if lim is not None:
        ax.set_xlim(*lim[0]); ax.set_ylim(*lim[1]); ax.set_zlim(*lim[2])
    try:
        ax.set_box_aspect([lim[0][1] - lim[0][0], lim[1][1] - lim[1][0], lim[2][1] - lim[2][0]] if lim else None)
    except Exception:
        pass


def _q3(ax, p, v, color, lw=2.2, ratio=0.12):
    ax.quiver(*p, *v, color=color, lw=lw, arrow_length_ratio=ratio)


def fig_a9_rmf_bending_3d(out: Path) -> None:
    """RMF along a 3-D centre line and how kappa1, kappa2 come out of it (dt/ds = kappa1 e1 + kappa2 e2)."""
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    from src.neuron_model import branch_geometry as bg

    xyz, t, s = _toy_branch_3d()
    e1_0 = bg.section_basis(t[0])[0]
    T, E1, E2 = bg.rotation_minimizing_frames(xyz, t, e1_0)
    fig = plt.figure(figsize=(15, 7.2))
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    ax.plot(*xyz.T, color=INK, lw=1.8)
    L = 0.9
    for j in np.linspace(20, len(xyz) - 40, 7).astype(int):
        p = xyz[j]
        _q3(ax, p, T[j] * L, PINK); _q3(ax, p, E1[j] * L * 0.8, GREEN); _q3(ax, p, E2[j] * L * 0.8, VIOLET)
        ax.scatter(*p, color=INK, s=12)
    ax.text(*(xyz[20] + T[20] * L + [0.1, 0, 0.2]), "$\\mathbf{t}$", color=PINK, fontsize=13)
    ax.text(*(xyz[20] + E1[20] * L * 0.8 + [0.1, 0, 0]), "$\\mathbf{e}_1$", color=GREEN, fontsize=13)
    ax.text(*(xyz[20] + E2[20] * L * 0.8 + [0.1, 0, 0]), "$\\mathbf{e}_2$", color=VIOLET, fontsize=13)
    _style_3d(ax, [(-1.6, 1.6), (-1.6, 1.6), (-0.3, 10.5)])
    ax.view_init(elev=14, azim=-58)
    ax.set_title("а) Рамка, минимизирующая вращение, вдоль центральной линии", loc="left")

    # (b) one step: tangent turn decomposed on the frame of node j
    axb = fig.add_subplot(1, 2, 2, projection="3d")
    j, k = 250, 45
    tj, tk, e1, e2 = T[j], T[j + k], E1[j], E2[j]
    ds = s[j + k] - s[j]
    dt = tk - tj
    c1, c2 = dt @ e1, dt @ e2
    o = np.zeros(3)
    _q3(axb, o, tj, PINK, lw=2.6, ratio=0.08)
    _q3(axb, o, tk, "#e891c0", lw=2.6, ratio=0.08)
    _q3(axb, o, e1 * 0.6, GREEN, lw=1.6); _q3(axb, o, e2 * 0.6, VIOLET, lw=1.6)
    a1 = tj + c1 * e1
    axb.plot(*np.c_[tj, a1], color=GREEN, lw=3)
    axb.plot(*np.c_[a1, a1 + c2 * e2], color=VIOLET, lw=3)
    axb.plot(*np.c_[a1 + c2 * e2, tk], color="#9a9ca3", lw=1, ls=":")
    axb.text(*(tj * 1.05 - e1 * 0.18), "$\\mathbf{t}(s)$", color=PINK, fontsize=13)
    axb.text(*(tk * 0.55 + e1 * 0.12), "$\\mathbf{t}(s+\\Delta s)$", color="#c45a95", fontsize=13)
    axb.text(*(e1 * 0.66), "$\\mathbf{e}_1(s)$", color=GREEN, fontsize=12)
    axb.text(*(e2 * 0.66), "$\\mathbf{e}_2(s)$", color=VIOLET, fontsize=12)
    axb.text(*(tj + c1 * e1 / 2 + 0.04), "$\\kappa_1\\Delta s$", color=GREEN, fontsize=13)
    axb.text(*(a1 + c2 * e2 / 2 + e1 * 0.08), "$\\kappa_2\\Delta s$", color=VIOLET, fontsize=13)
    pts = np.vstack([o, tj, tk, e1 * 0.6, e2 * 0.6])
    lo, hi = pts.min(0) - 0.15, pts.max(0) + 0.15
    _style_3d(axb, list(zip(lo, hi)))
    v = np.cross(e1, e2)
    axb.view_init(elev=22, azim=-35)
    axb.set_title("б) Изгиб: насколько касательная повернула к $\\mathbf{e}_1$ и к $\\mathbf{e}_2$", loc="left")
    fig.text(0.52, 0.04,
             "$\\dfrac{d\\mathbf{t}}{ds}=\\kappa_1\\,\\mathbf{e}_1+\\kappa_2\\,\\mathbf{e}_2$,   "
             "$\\kappa_1=\\dfrac{d\\mathbf{t}}{ds}\\cdot\\mathbf{e}_1$,   $\\kappa_2=\\dfrac{d\\mathbf{t}}{ds}\\cdot\\mathbf{e}_2$\n"
             "Рамка не крутится вокруг $\\mathbf{t}$, поэтому $\\kappa_1,\\kappa_2$ — только изгиб, без кручения.\n"
             "Касательные сдвинуты в общее начало; шаг $\\Delta s$ увеличен для наглядности (в данных 250–500 нм).",
             fontsize=11.5, va="bottom", color=INK)
    fig.subplots_adjust(left=0.0, right=1.0, top=0.92, bottom=0.12, wspace=0.0)
    save(fig, out, "r11_rmf_bending_3d.png")


def fig_a10_tilt_3d(out: Path) -> None:
    """The tilt vector q = (q1, q2) in 3-D: fixed axes (a, u1, u2) of the branch, and q as the point where the
    tangent ray crosses the plane at unit distance along a."""
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    from src.neuron_model import branch_geometry as bg

    xyz, t, s = _toy_branch_3d()
    a = np.array([0.0, 0.0, 1.0])
    u1, u2 = np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])
    fig = plt.figure(figsize=(15, 7.2))
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    ax.plot(*xyz.T, color=INK, lw=1.8)
    ax.plot([0, 0], [0, 0], [0, 10], ls="--", color=GREEN, lw=1.6)
    ax.text(0.1, 0, 10.2, "ось ветви $\\mathbf{a}$ (хорда)", color=GREEN, fontsize=12)
    for jj in (60, 540):
        pp = xyz[jj]
        _q3(ax, pp, u1 * 0.7, "#7fa86f", lw=1.4); _q3(ax, pp, u2 * 0.7, "#ad97de", lw=1.4)
    qn = np.hypot(t[:, 0], t[:, 1]) / t[:, 2]
    j = int(np.argmax(qn[100:-100])) + 100
    p, tj = xyz[j], t[j]
    L = 2.2
    _q3(ax, p, tj * L, PINK, lw=2.6, ratio=0.08)
    c = L * np.array([tj @ a, tj @ u1, tj @ u2])
    p1 = p + c[0] * a; p2 = p1 + c[1] * u1; p3 = p2 + c[2] * u2
    ax.plot(*np.c_[p, p1], color=GREEN, lw=2.4)
    ax.plot(*np.c_[p1, p2], color="#2f6b1f", lw=2.4)
    ax.plot(*np.c_[p2, p3], color=VIOLET, lw=2.4)
    ax.text(*(p + tj * L + [0.1, 0, 0.15]), "$\\mathbf{t}(s)$", color=PINK, fontsize=13)
    ax.text(*(xyz[60] + u1 * 0.78), "$\\mathbf{u}_1$", color="#2f6b1f", fontsize=12)
    ax.text(*(xyz[60] + u2 * 0.78), "$\\mathbf{u}_2$", color=VIOLET, fontsize=12)
    ax.text(*(xyz[540] + u1 * 0.78), "$\\mathbf{u}_1$", color="#2f6b1f", fontsize=12)
    ax.text(*(xyz[540] + u2 * 0.78), "$\\mathbf{u}_2$", color=VIOLET, fontsize=12)
    ax.text(*(p1 + [-1.1, 0, -0.9]), "$\\mathbf{t}\\cdot\\mathbf{a}$", color=GREEN, fontsize=12)
    _style_3d(ax, [(-1.6, 1.6), (-1.6, 1.6), (-0.3, 10.8)])
    ax.view_init(elev=14, azim=-58)
    ax.set_title("а) Оси $\\mathbf{a},\\mathbf{u}_1,\\mathbf{u}_2$ одни на всю ветвь", loc="left")

    # (b) q as a point of the plane a = 1
    axb = fig.add_subplot(1, 2, 2, projection="3d")
    q1, q2 = (tj @ u1) / (tj @ a), (tj @ u2) / (tj @ a)
    lim = max(abs(q1), abs(q2)) + 0.25
    gx, gy = np.meshgrid([-lim, lim], [-lim, lim])
    axb.plot_surface(gx, gy, np.ones_like(gx), color=SOFT_GREY, alpha=0.45, edgecolor="#c9cbd0")
    o = np.zeros(3)
    hit = np.array([q1, q2, 1.0])
    axb.plot([0, 0], [0, 0], [0, 1.3], ls="--", color=GREEN, lw=1.6)
    _q3(axb, o, tj * 0.75, PINK, lw=2.8, ratio=0.12)
    axb.plot(*np.c_[tj * 0.75, hit], color=PINK, lw=1.4, ls=":")
    axb.plot([0, q1], [0, 0], [1, 1], color="#2f6b1f", lw=3)
    axb.plot([q1, q1], [0, q2], [1, 1], color=VIOLET, lw=3)
    axb.plot([0, q1], [0, q2], [1, 1], color=INK, lw=1.6)
    axb.scatter(*hit, color=PINK, s=45)
    axb.scatter(0, 0, 1, color=GREEN, s=30)
    sx, sy = np.sign(q1 or 1), np.sign(q2 or 1)
    axb.text(q1 * 0.3, -0.13 * sy, 1.0, "$q_1$", color="#2f6b1f", fontsize=15, ha="center", va="center")
    axb.text(q1 + 0.3 * sx, q2 * 0.6, 1.0, "$q_2$", color=VIOLET, fontsize=15, ha="center", va="center")
    # the fixed transverse axes of the branch, drawn at the foot of a on the plane and at the origin
    ua = 0.55 * lim
    for z0, lw, alpha in ((1.0, 2.0, 1.0), (0.0, 1.4, 0.6)):
        axb.quiver(0, 0, z0, ua, 0, 0, color="#2f6b1f", lw=lw, arrow_length_ratio=0.15, alpha=alpha)
        axb.quiver(0, 0, z0, 0, ua, 0, color=VIOLET, lw=lw, arrow_length_ratio=0.15, alpha=alpha)
        axb.text(ua * 1.1, 0, z0, "$\\mathbf{u}_1$", color="#2f6b1f", fontsize=13, alpha=alpha)
        axb.text(0, ua * 1.12, z0, "$\\mathbf{u}_2$", color=VIOLET, fontsize=13, alpha=alpha)
    axb.text(*(hit + [0.04, 0.04, 0.06]), "$(q_1,q_2,1)$", color=PINK, fontsize=12)
    axb.text(q1 / 2 - 0.05, q2 / 2 + 0.05, 1.03, "$\\mathbf{q}$", color=INK, fontsize=15)
    axb.text(0.03, 0, 0.12, "$\\mathbf{a}$", color=GREEN, fontsize=14)
    axb.text(*(tj * 0.4 + [0.05, 0.0, 0.0]), "$\\mathbf{t}$", color=PINK, fontsize=14)
    axb.text(-lim, lim, 1.02, "плоскость на расстоянии 1 вдоль $\\mathbf{a}$", color="#4a4d55", fontsize=10.5)
    _style_3d(axb, [(-lim, lim), (-lim, lim), (0, 1.35)])
    axb.view_init(elev=24, azim=-60)
    axb.set_title("б) Наклон $\\mathbf{q}$: где луч касательной пересекает плоскость", loc="left")
    fig.text(0.52, 0.04,
             "$\\mathbf{q}=(q_1,q_2)=\\left(\\dfrac{\\mathbf{t}\\cdot\\mathbf{u}_1}{\\mathbf{t}\\cdot\\mathbf{a}},\\ "
             "\\dfrac{\\mathbf{t}\\cdot\\mathbf{u}_2}{\\mathbf{t}\\cdot\\mathbf{a}}\\right)$,   "
             "$\\|\\mathbf{q}\\|=\\tan\\theta$ ($\\theta$ — полный угол между $\\mathbf{t}$ и $\\mathbf{a}$),\n"
             "направление $\\mathbf{q}$ — куда отклонилась касательная. Обратно: $\\mathbf{t}=(q_1,q_2,1)/\\|(q_1,q_2,1)\\|$.",
             fontsize=11.5, va="bottom", color=INK)
    fig.subplots_adjust(left=0.0, right=1.0, top=0.92, bottom=0.12, wspace=0.0)
    save(fig, out, "r12_tilt_3d.png")


def fig_a11_tilt_steps(out: Path) -> None:
    """Three steps: (1) tangent t and the branch axes a, u1, u2; (2) the tilt component q1 = tan alpha1 = t.u1 / t.a
    in the plane (a, u1); (3) the same for u2."""
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    a, u1, u2 = np.array([0.0, 0, 1]), np.array([1.0, 0, 0]), np.array([0.0, 1, 0])
    t = np.array([0.5, 0.32, 1.0]); t /= np.linalg.norm(t)
    ta, tu1, tu2 = t @ a, t @ u1, t @ u2
    o = np.zeros(3)
    U1C, U2C = "#2f6b1f", VIOLET
    lim = [(-0.15, 0.75), (-0.15, 0.75), (0, 1.15)]
    fig = plt.figure(figsize=(16, 6.2))

    def axes_common(ax, faint=False):
        al = 0.35 if faint else 1.0
        ax.quiver(*o, *(a * 1.05), color=GREEN, lw=2, arrow_length_ratio=0.08, alpha=al)
        ax.quiver(*o, *(u1 * 0.6), color=U1C, lw=2, arrow_length_ratio=0.12, alpha=al)
        ax.quiver(*o, *(u2 * 0.6), color=U2C, lw=2, arrow_length_ratio=0.12, alpha=al)
        ax.text(0.03, 0, 1.08, "$\\mathbf{a}$", color=GREEN, fontsize=14, alpha=al)
        ax.text(0.64, 0, 0, "$\\mathbf{u}_1$", color=U1C, fontsize=14, alpha=al)
        ax.text(0, 0.64, 0, "$\\mathbf{u}_2$", color=U2C, fontsize=14, alpha=al)

    def finish(ax, title, azim=-50):
        _style_3d(ax, lim)
        ax.view_init(elev=18, azim=azim)
        ax.set_title(title, loc="left")

    # (1)
    ax = fig.add_subplot(1, 3, 1, projection="3d")
    axes_common(ax)
    ax.quiver(*o, *t, color=PINK, lw=2.8, arrow_length_ratio=0.1)
    ax.text(*(t + [0.02, 0.02, 0.04]), "$\\mathbf{t}$", color=PINK, fontsize=15)
    ax.scatter(*o, color=INK, s=18)
    ax.text(0.03, -0.12, -0.05, "точка ствола", color="#4a4d55", fontsize=10)
    finish(ax, "1) Касательная $\\mathbf{t}$ и оси ветви $\\mathbf{a},\\mathbf{u}_1,\\mathbf{u}_2$")

    def component(idx, u, tu, col, name, azim):
        ax = fig.add_subplot(1, 3, idx, projection="3d")
        ax.computed_zorder = False
        axes_common(ax, faint=False)
        ax.quiver(*o, *t, color=PINK, lw=1.6, arrow_length_ratio=0.1, alpha=0.3)
        ax.text(*(t + [0.04, 0.04, 0.0]), "$\\mathbf{t}$", color=PINK, fontsize=13, alpha=0.45)
        # the plane (a, u)
        pl = [o, u * 0.75, u * 0.75 + a * 1.1, a * 1.1]
        ax.add_collection3d(Poly3DCollection([pl], facecolor=SOFT_GREY, edgecolor="#c9cbd0", alpha=0.6, zorder=0))
        proj = ta * a + tu * u
        ax.plot(*np.c_[t, proj], color="#9a9ca3", lw=1, ls=":")
        ax.plot(*np.c_[o, ta * a], color=GREEN, lw=4)
        ax.plot(*np.c_[ta * a, proj], color=col, lw=4)
        ax.quiver(*o, *proj, color=PINK, lw=2.6, arrow_length_ratio=0.1)
        ang = np.arctan2(tu, ta)
        r = 0.3
        th = np.linspace(0, ang, 30)
        arc = np.outer(np.cos(th), a * r) + np.outer(np.sin(th), u * r)
        ax.plot(*arc.T, color=INK, lw=1.4)
        mid = 0.5 * ang
        ax.text(*((np.cos(mid) * a + np.sin(mid) * u) * (r + 0.05)), f"$\\alpha_{name}$", color=INK, fontsize=14)
        ax.text(*(ta * a * 0.5 - u * 0.17), "$\\mathbf{t}\\cdot\\mathbf{a}$", color=GREEN, fontsize=13)
        ax.text(*(ta * a + tu * u * 0.3 + a * 0.05), f"$\\mathbf{{t}}\\cdot\\mathbf{{u}}_{name}$", color=col, fontsize=13)
        ax.text(*(proj + a * 0.07 - u * 0.25), f"проекция $\\mathbf{{t}}$\nна плоскость $(\\mathbf{{a}},\\mathbf{{u}}_{name})$", color=PINK, fontsize=10)
        ax.text2D(0.02, 0.02, f"$q_{name}=\\tan\\alpha_{name}=\\dfrac{{\\mathbf{{t}}\\cdot\\mathbf{{u}}_{name}}}{{\\mathbf{{t}}\\cdot\\mathbf{{a}}}}$",
                  transform=ax.transAxes, fontsize=15, color=INK)
        finish(ax, f"{idx}) Наклон в плоскости $(\\mathbf{{a}},\\mathbf{{u}}_{name})$: $q_{name}$", azim)

    component(2, u1, tu1, U1C, "1", -72)
    component(3, u2, tu2, U2C, "2", -18)
    fig.subplots_adjust(left=0.0, right=1.0, top=0.9, bottom=0.03, wspace=0.0)
    save(fig, out, "r13_tilt_steps.png")


STATIC = [fig_a1_pipeline, fig_a2_chain, fig_a3_cross_section, fig_a4_tortuosity, fig_a5_assembly, fig_a6_flow_matching, fig_a7_spine_models,
          fig_b1_smoothing, fig_b2_t0, fig_b3_validity, fig_b4_vae, fig_b5_assembly, fig_b6_audit]


# ---------------------------------------------------------------------------
# C: plots from run results
# ---------------------------------------------------------------------------


def _metric_info() -> Dict[str, Dict[str, Any]]:
    try:
        from src.neuron_model.spine_generation.atlas import METRIC_INFO

        return METRIC_INFO
    except Exception as exc:  # heavy imports must not stop the plots
        print(f"[figures] note: atlas metric names unavailable ({exc}); using identifiers")
        return {}


def read_jsonl(path: Path) -> pd.DataFrame:
    return pd.read_json(path, lines=True) if path.exists() else pd.DataFrame()


SMOOTH_STEPS = 2000  # moving-average window of the loss curves, training steps


def fig_c1_training(out: Path, runs: Mapping[str, Path]) -> None:
    mogen, vae = {}, {}
    for name, run in runs.items():
        df = read_jsonl(Path(run) / "logs" / "metrics_train.jsonl")
        if df.empty:
            print(f"[figures] c1: no train log in {run}")
            continue
        (vae if "sdf_loss" in df.columns else mogen)[name] = df
    if not mogen and not vae:
        return
    fig, axes = plt.subplots(1, 2 if (mogen and vae) else 1, figsize=(13 if (mogen and vae) else 7.5, 4.6), squeeze=False)
    ai = 0
    if mogen:
        ax = axes[0][ai]; ai += 1
        for (name, df), col in zip(mogen.items(), SERIES):
            y = df["loss"]
            # smooth over a fixed number of TRAINING STEPS (not log lines), so that the runs are comparable
            spacing = float(np.median(np.diff(df["step"]))) if len(df) > 1 else 1.0
            window = max(5, int(round(SMOOTH_STEPS / max(spacing, 1.0))))
            ax.plot(df["step"], y, color=col, alpha=0.12, lw=0.7)
            ax.plot(df["step"], y.rolling(window, min_periods=max(1, window // 4), center=True).mean(), color=col, lw=2.2, label=name)
        ax.set_xlabel("шаг обучения"); ax.set_ylabel("loss Flow Matching")
        ax.set_title("MoGen: loss выходит на плато рано"); ax.legend()
        lo = min(df["loss"].quantile(0.02) for df in mogen.values()); ax.set_ylim(lo * 0.95, min(0.9, max(df["loss"].iloc[min(20, len(df) - 1)] for df in mogen.values()) * 1.05))
    if vae:
        ax = axes[0][ai]
        for name, df in vae.items():
            for column, col, lab in [("sdf_loss", PINK, "ошибка расстояния"), ("surface_loss", GREEN, "поверхность"), ("normal_loss", VIOLET, "нормаль"), ("eikonal_loss", GREY, "эйконал")]:
                if column in df.columns:
                    spacing = float(np.median(np.diff(df["step"]))) if len(df) > 1 else 1.0
                    window = max(5, int(round(SMOOTH_STEPS / max(spacing, 1.0))))
                    ax.plot(df["step"], df[column].rolling(window, min_periods=max(1, window // 4), center=True).mean(), color=col, lw=1.8, label=lab)
        ax.set_yscale("log"); ax.set_xlabel("шаг обучения"); ax.set_title("VAE: компоненты функции потерь"); ax.legend(fontsize=9)
    save(fig, out, "c1_training_curves.png")


def _metrics_file(run: Path, stream: str) -> Path:
    run = Path(run)
    for cand in (run / "logs" / f"metrics_{stream}.jsonl", run / f"metrics_{stream}.jsonl"):
        if cand.exists():
            return cand
    return run / "logs" / f"metrics_{stream}.jsonl"


def fig_a12_conditional_curves(out: Path, v2: Path, v3: Path) -> None:
    """Training loss of the conditional MoGen v2 -> v3 as it was LOGGED (no kept checkpoints, so no curves
    recomputed with one estimator): the train loss is the mean over the 100 training batches before each
    log line (condition dropout on), the validation loss exists only at the end of each run."""
    parts = []
    offset = 0
    for name, run in (("v2", v2), ("v3", v3)):
        df = read_jsonl(_metrics_file(run, "train"))
        val = read_jsonl(_metrics_file(run, "val"))
        if df.empty:
            print(f"[figures] a12: no train log in {run}")
            return
        parts.append((name, df.assign(gstep=df["step"] + offset), val.assign(gstep=val["step"] + offset) if not val.empty else val))
        offset += int(df["step"].max())
    fig, (ax, axd) = plt.subplots(2, 1, figsize=(9.5, 7.6), gridspec_kw={"height_ratios": [1.6, 1]})
    for (name, df, val), col in zip(parts, (PINK, GREEN)):
        window = max(5, int(round(1000 / float(np.median(np.diff(df["step"]))))))
        ax.plot(df["gstep"], df["loss"], color=col, alpha=0.25, lw=0.8)
        ax.plot(df["gstep"], df["loss"].rolling(window, min_periods=window // 2, center=True).mean(), color=col, lw=2.2,
                label=f"{name}: обучающая потеря (среднее по 1000 шагам)")
        if not val.empty:
            ax.scatter(val["gstep"], val["val_loss"], color=col, s=70, marker="D", edgecolor="white", zorder=5,
                       label=f"{name}: проверочная потеря {val['val_loss'].iloc[-1]:.4f}".replace(".", ","))
    lo = min(p[1]["loss"].quantile(0.01) for p in parts)
    hi = max(p[1]["loss"].quantile(0.99) for p in parts)
    ax.set_ylim(lo - 0.012, hi + 0.004)
    ax.axvline(parts[0][1]["gstep"].max(), color=GREY, lw=1, ls="--")
    ax.text(parts[0][1]["gstep"].max() + 300, hi + 0.003, "старт v3 от весов v2,\nlr слоя условий ×100", va="top", fontsize=9, color=INK,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.85))
    ax.set_xlabel("шаг обучения (v2: 0–30 000, v3: 30 000–45 000)")
    ax.set_ylabel("потеря сопоставления потоков")
    ax.set_title("а) Условная MoGen v2 → v3: потеря по журналу обучения")
    ax.legend(fontsize=8, loc="lower left", ncol=2, frameon=True)
    # v3 was started with the same data seed, so its batches repeat the first steps of v2: paired difference
    (_, d2, _), (_, d3, _) = parts
    m = d2[["step", "loss"]].merge(d3[["step", "loss"]], on="step", suffixes=("_v2", "_v3"))
    diff = m["loss_v3"] - m["loss_v2"]
    corr = float(np.corrcoef(m["loss_v2"], m["loss_v3"])[0, 1])
    se = float(diff.std(ddof=1) / np.sqrt(len(diff)))
    w = max(5, int(round(1000 / float(np.median(np.diff(m["step"]))))))
    axd.axhline(0, color=GREY, lw=1)
    axd.plot(m["step"], diff, color=VIOLET, alpha=0.3, lw=0.8)
    axd.plot(m["step"], diff.rolling(w, min_periods=w // 2, center=True).mean(), color=VIOLET, lw=2.2)
    axd.set_xlabel("шаг внутри запуска (одинаковые пакеты у v2 и v3)")
    axd.set_ylabel("v3 − v2")
    axd.set_title(f"б) Те же пакеты: корреляция журналов {corr:.3f}, средняя разность {diff.mean():+.5f} ± {se:.5f}".replace(".", ","), fontsize=13)
    fig.tight_layout()
    save(fig, out, "r14_conditional_v2_v3_loss.png")

    # slide version: only the logged training loss, legend "v2" / "v3", no end-of-run validation points
    # (they come from another estimator: EMA weights, fixed t and noise, no condition dropout)
    fig, ax = plt.subplots(figsize=(9.5, 4.4))
    for (name, df, _), col in zip(parts, (PINK, GREEN)):
        window = max(5, int(round(1000 / float(np.median(np.diff(df["step"]))))))
        ax.plot(df["gstep"], df["loss"], color=col, alpha=0.25, lw=0.8)
        ax.plot(df["gstep"], df["loss"].rolling(window, min_periods=window // 2, center=True).mean(), color=col, lw=2.2, label=name)
    ax.set_ylim(lo - 0.004, hi + 0.004)
    ax.axvline(parts[0][1]["gstep"].max(), color=GREY, lw=1, ls="--")
    ax.set_xlabel("шаг обучения")
    ax.set_ylabel("потеря сопоставления потоков")
    ax.legend(fontsize=10, loc="lower right", frameon=True)
    fig.tight_layout()
    save(fig, out, "r14b_conditional_v2_v3_train_loss.png")


V3_NAME = "Условная MoGen (бинарный тип клетки, подробный тип клетки)"
V4_NAME = "Условная MoGen (организм, бинарный тип клетки, подробный тип клетки)"
# legend names by a Latin key (Cyrillic arguments of a .bat file are garbled by the Windows console code page)
MODEL_NAMES = {"v3": V3_NAME, "v4": V4_NAME, "A": "MoGen (без условий)", "vae_flow": "VAE с нормализующим потоком"}


def _marginals(real: pd.DataFrame, gen: pd.DataFrame, floor: Optional[pd.DataFrame], name: str, path: Path, title: str) -> None:
    """Densities of the 11 morphometrics: real (grey), real after the generator's reconstruction chain (dashed), one model."""
    from src.neuron_model import quality_report as qr

    cap = lambda t: t[:1].upper() + t[1:]
    labels = {m: cap(t) for m, t in qr.S_LABELS.items()}
    log_cols = [m for m in qr.S_METRICS if m not in ("OpenAngle", "CVD", "ConvexHullRatio")]
    qr.distribution_grid(real, {name: gen}, floor, labels, path, "hist", log_cols, title,
                         scale={m: f for m, (f, _) in qr.S_DISPLAY.items()}, labels={"real": "Реальные", "floor": "Реальные после постобработки"})
    print(f"[figures] {path}")


def fig_a13_s_marginals(out: Path, evaluation: Path, model: str = "mogen_v3", name: str = V3_NAME, floor: str = "real_poisson",
                        file_name: str = "r15_s_marginals_mogen.png") -> None:
    """Slide version of ``s_marginals.png`` of the quality report from an ``evaluate_generators`` run: real spines of the
    evaluation neurons, the same real spines after the reconstruction chain of the model (``floor``: real_poisson for MoGen,
    real_mc for the VAE) and ONE model."""
    from src.neuron_model import quality_report as qr

    pp = Path(evaluation) / "postprocessed"
    real = pd.read_parquet(Path(evaluation) / "real_test_morphometrics.parquet", columns=["spine_key", *qr.S_METRICS])
    sets = qr._read_gen_sets(pp)
    if model not in sets:
        print(f"[figures] a13: no {model} in {pp} ({sorted(sets)})")
        return
    if floor not in sets:
        print(f"[figures] a13: no floor {floor} in {pp}, drawn without it")
    _marginals(real, sets[model], sets.get(floor), name, Path(out) / file_name, "Морфометрики шипиков: реальные против сгенерированных")


def fig_a14_species_marginals(out: Path, report: Path, name: str = V4_NAME, label: str = "v4-pw8", floor_label: str = "real-pw8val",
                              metadata_root: Path = REPO_ROOT / "datasets") -> None:
    """MoGen v4 by organism and binary cell type from a ``compare_species_generation.py`` report: for every group the real
    spines of the validation neurons of that organism and type (read again from the stage-8 tables of the report), the same
    real spines after the d8_pw8 chain (``floor_label``) and the CONDITIONAL v4 samples of that group (one figure per group)."""
    import importlib.util

    report = Path(report)
    info = json.loads((report / "run_info.json").read_text(encoding="utf-8"))
    gen = pd.read_parquet(report / "generated_metrics.parquet")

    def load(name_):
        spec = importlib.util.spec_from_file_location(name_, REPO_ROOT / "scripts/neuron_model" / f"{name_}.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    csg, mbc = load("compare_species_generation"), load("morphology_by_condition")
    splits = Path(info["splits"]) if info.get("splits") not in (None, "None") else None
    frames = {"mouse": mbc.load_spines(Path(info["minnie_table"]), Path(metadata_root).resolve(), splits, info.get("split")),
              "human": csg.load_h01_spines(Path(info["h01_table"]), Path(metadata_root).resolve(),
                                           Path(info["h01_detached"]) if info.get("h01_detached") not in (None, "None") else None, splits, info.get("split"))}
    species_ru = {"mouse": "мышь", "human": "человек"}
    type_ru = {"excitatory": "возбуждающие", "inhibitory": "тормозные"}
    for species, frame in frames.items():
        for ctype in ("excitatory", "inhibitory"):
            group = f"{species}/{ctype}"
            real = frame[frame["cell_type_binary"] == ctype]
            g = gen[(gen["value"] == group) & (gen["label"] == label)]
            fl = gen[(gen["value"] == group) & (gen["label"] == floor_label)]
            if not len(real) or not len(g):
                print(f"[figures] a14: {group}: real {len(real)}, generated {len(g)} - skipped")
                continue
            _marginals(real, g, fl if len(fl) else None, name, Path(out) / f"r16_s_marginals_v4_{species}_{ctype}.png",
                       f"Морфометрики шипиков: {species_ru[species]}, {type_ru[ctype]} клетки ({real['neuron_id'].nunique()} проверочных нейронов)")


def _eval_dir(evaluation: Path, kind: str) -> Path:
    return Path(evaluation) / kind


def fig_c2_morph_w1(out: Path, evaluation: Path, kind: str) -> None:
    files = sorted(_eval_dir(evaluation, kind).glob("univariate_*_seed*.csv"))
    if not files:
        print(f"[figures] c2: no univariate_*.csv in {_eval_dir(evaluation, kind)}")
        return
    info = _metric_info()
    tables: Dict[str, pd.DataFrame] = {}
    for f in files:
        name = f.stem[len("univariate_"):].rsplit("_seed", 1)[0]
        tables.setdefault(name, pd.read_csv(f))
    names = order_models(list(tables))
    features = [f for f in tables[names[0]]["feature"]]
    labels = [info.get(f, {}).get("name", f) for f in features]
    series = {model_label(n): [float(tables[n].set_index("feature").loc[f, "w1_normalized"]) if f in set(tables[n]["feature"]) else np.nan for f in features] for n in names}
    colors = [model_color(n, i) for i, n in enumerate(names)]
    # the untuned MoGen is 10-50 times worse than the rest and flattens the other bars: the axis is cut and the cut bars say so
    values = np.array([v for k, vals in series.items() for v in vals if k != model_label("base")], dtype=float)
    cap = float(np.nanmax(values)) * 1.35 if values.size else None
    fig, ax = plt.subplots(figsize=(11.5, 0.95 * len(features) + 2.2))
    clipped = {k: np.minimum(v, cap) if cap else np.asarray(v) for k, v in series.items()}
    grouped_bars(ax, labels, clipped, colors, horizontal=True, width=0.86, annotate=False)
    n = len(series)
    for i, (name, vals) in enumerate(series.items()):
        for j, v in enumerate(vals):
            if not np.isfinite(v):
                continue
            y = j - 0.86 / 2 + (0.86 / n) * (i + 0.5)
            cut = bool(cap) and v > cap
            ax.text(min(v, cap) if cap else v, y, (" ▶ " if cut else " ") + f"{v:.2f}", va="center", fontsize=9, color="#4a4d55")
    if cap:
        ax.set_xlim(0, cap * 1.12)
    ax.set_xlabel("W₁ / σ (расстояние Вассерштейна в долях разброса; меньше — лучше)" + ("\n▶ — столбец обрезан, число справа настоящее" if cap else ""))
    ax.set_title(f"Совпадение распределений морфометрик с реальными шипиками ({KIND_RU[kind]})", pad=34)
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=n, fontsize=10.5, handlelength=1.4)
    save(fig, out, "c2_morph_w1.png")


def fig_c3_morph_hist(out: Path, evaluation: Path, kind: str, features=("Length", "Volume", "Area", "OpenAngle", "CVD", "JunctionArea")) -> None:
    real_path = Path(evaluation) / "real_test_morphometrics.parquet"
    gen_files = sorted(_eval_dir(evaluation, kind).glob("morphometrics_*_seed*.parquet"))
    if not real_path.exists() or not gen_files:
        print(f"[figures] c3: needs real_test_morphometrics.parquet and {kind}/morphometrics_*.parquet (written by the current evaluate_generators.py)")
        return
    info = _metric_info()
    real = pd.read_parquet(real_path)
    gens_raw = {f.stem[len("morphometrics_"):].rsplit("_seed", 1)[0]: pd.read_parquet(f) for f in gen_files}
    gens = {model_label(n): gens_raw[n] for n in order_models(list(gens_raw))}
    gen_colors = [model_color(n, i) for i, n in enumerate(order_models(list(gens_raw)))]
    fig, axes = plt.subplots(2, 3, figsize=(13.4, 7.0))
    for ax, feat in zip(axes.ravel(), features):
        meta = info.get(feat, {"name": feat, "unit": "", "factor": 1.0, "log": False})
        r = pd.to_numeric(real[feat], errors="coerce").to_numpy(float) * meta["factor"]
        r = r[np.isfinite(r)]
        pool = [r] + [pd.to_numeric(g[feat], errors="coerce").to_numpy(float) * meta["factor"] for g in gens.values() if feat in g.columns]
        pool = [p[np.isfinite(p)] for p in pool]
        if meta.get("log"):
            pool = [p[p > 0] for p in pool]
            lo, hi = np.percentile(np.concatenate(pool), [0.5, 99.5])
            bins = np.logspace(np.log10(lo), np.log10(hi), 40); ax.set_xscale("log")
        else:
            lo, hi = np.percentile(np.concatenate(pool), [0.5, 99.5])
            bins = np.linspace(lo, hi, 40)
        ax.hist(pool[0], bins=bins, density=True, color=SOFT_GREY, ec="#c9cbd0", label=f"реальные (n = {len(pool[0])})")
        for (name, g), col, p in zip(gens.items(), gen_colors, pool[1:]):
            ax.hist(p, bins=bins, density=True, histtype="step", color=col, lw=2.0, label=f"{name} (n = {len(p)})")
        ax.set_title(meta["name"], fontsize=13); ax.set_xlabel(f"{meta['name']}, {meta['unit']}".rstrip(", ")); ax.set_yticks([])
    axes[0][0].legend(fontsize=8.5)
    fig.suptitle(f"Распределения морфометрик: тестовые реальные шипики и сгенерированные ({KIND_RU[kind]})", fontsize=14, x=0.01, ha="left")
    fig.tight_layout()
    save(fig, out, "c3_morph_hist.png")


CRITERION_NAMES = {
    "validity_raw.valid_mesh_rate": "валидные сетки, сырые", "validity_postprocessed.valid_mesh_rate": "валидные сетки, после постобработки",
    "mmd2_morphometrics": "MMD по морфометрии", "correlation_frobenius": "расхождение корреляций", "coverage_cd": "покрытие (COV-CD)",
    "one_nna_cd": "1-NNA", "mmd_cd": "MMD-CD", "diversity.chamfer_mean_ratio": "разнообразие (по формам)", "diversity.morphometric_mean_ratio": "разнообразие (по признакам)",
    "memorization.morphometric_dcr_median_ratio": "запоминание: DCR по признакам", "memorization.chamfer_dcr_median_ratio": "запоминание: DCR по формам",
    "memorization.morphometric_near_copy_rate": "доля почти копий (признаки)", "memorization.chamfer_near_copy_rate": "доля почти копий (формы)", "classifier_accuracy": "классификатор: точность",
}


def _leading_float(text: Any) -> float:
    try:
        return float(str(text).split("±")[0].strip())
    except ValueError:
        return float("nan")


def _target(text: str) -> tuple:
    text = str(text)
    if text.startswith("↑"):
        return "up", 0.0
    if text.startswith("↓"):
        return "down", 0.0
    digits = "".join(ch for ch in text.replace(",", ".").split("(")[0] if ch.isdigit() or ch == ".")
    return "to", float(digits) if digits else 0.0


def fig_c4_scorecard(out: Path, evaluation: Path, kind: str) -> None:
    path = _eval_dir(evaluation, kind) / "summary.csv"
    if not path.exists():
        print(f"[figures] c4: {path} not found")
        return
    table = pd.read_csv(path)
    models = order_models([c for c in table.columns if c not in ("criterion", "target")])
    cell_text, colors = [], []
    for _, row in table.iterrows():
        vals = np.array([_leading_float(row[m]) for m in models])
        mode, goal = _target(row["target"])
        score = vals if mode == "up" else (-vals if mode == "down" else -np.abs(vals - goal))
        finite = np.isfinite(score)
        best, worst = (np.nanmax(score), np.nanmin(score)) if finite.any() else (np.nan, np.nan)
        cell_text.append([CRITERION_NAMES.get(row["criterion"], row["criterion"]), row["target"], *[str(row[m]) for m in models]])
        colors.append(["white", "white", *[(SOFT_GREEN if np.isfinite(s) and s == best and best != worst else SOFT_PINK if np.isfinite(s) and s == worst and best != worst else "white") for s in score]])
    fig, ax = plt.subplots(figsize=(3.2 + 1.9 * len(models) + 3.5, 0.5 + 0.42 * len(table)))
    ax.axis("off"); ax.grid(False)
    tb = ax.table(cellText=cell_text, colLabels=["критерий", "ориентир", *[model_label(m) for m in models]], cellColours=colors, loc="center", cellLoc="center", colLoc="center")
    tb.auto_set_font_size(False); tb.set_fontsize(9.5); tb.scale(1, 1.5)
    tb.auto_set_column_width(list(range(len(models) + 2)))
    for (r, c), cell in tb.get_celld().items():
        cell.set_edgecolor("#e6e6ea")
        if r == 0:
            cell.set_text_props(fontweight="bold"); cell.set_facecolor("#f6f6f8")
        if c == 0 and r > 0:
            cell.set_text_props(ha="left"); cell._loc = "left"
    ax.set_title(f"Сводка критериев ({KIND_RU[kind]}): зелёный — лучшее значение в строке, розовый — худшее", fontsize=12.5, loc="left")
    save(fig, out, "c4_scorecard.png")


def fig_c5_t0(out: Path, run: Path) -> None:
    path = Path(run) / "report.json"
    if not path.exists():
        print(f"[figures] c5: {path} not found")
        return
    rep = json.loads(path.read_text(encoding="utf-8"))
    chosen = rep["chosen"]; var = rep["variants"][chosen]
    names = {"mean_radius_nm": "средний радиус", "radius_ratio_prox_dist": "сужение", "kappa_rms_per_nm": "среднеквадр. изгиб", "tortuosity": "извилистость", "radius_cv": "разброс радиуса", "kappa_autocorr_1um": "автокорреляция изгиба"}
    feats = list(var["functionals"].keys())
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.6), gridspec_kw={"width_ratios": [1.2, 1]})
    ax = axes[0]
    y = np.arange(len(feats))
    ax.barh(y - 0.2, [var["functionals"][f]["w1_rel"] for f in feats], height=0.38, color=PINK, label="сгенерированные")
    ax.barh(y + 0.2, [rep["noise_floor"][f]["w1_rel"] for f in feats], height=0.38, color=GREY, label="шумовой пол")
    ax.set_yticks(y, [names.get(f, f) for f in feats]); ax.invert_yaxis(); ax.grid(axis="y", visible=False)
    ax.set_xlabel("W₁ / σ"); ax.set_title(f"T0 ({chosen}): расхождение по функционалам"); ax.legend(loc="lower right")
    ax = axes[1]
    by = var.get("tortuosity_by_length", {})
    if by:
        labels = list(by.keys())
        grouped_bars(ax, labels, {"реальные, медиана": [by[k]["real_median"] for k in labels], "сгенерированные, медиана": [by[k]["gen_median"] for k in labels]}, [GREY, PINK], fmt="{:.2f}")
        ax.set_ylim(0.9, max(1.3, 1.1 * max(by[k]["gen_median"] for k in labels)))
        ax.set_title("Извилистость по классам длины"); ax.legend(fontsize=9, loc="upper left")
    save(fig, out, "c5_t0_report.png")


def fig_c6_assembled(out: Path, directory: Path) -> None:
    bpath, spath = Path(directory) / "evaluation_branches.csv", Path(directory) / "evaluation_spines.csv"
    if not bpath.exists():
        print(f"[figures] c6: {bpath} not found")
        return
    b = pd.read_csv(bpath)
    s = pd.read_csv(spath) if spath.exists() else pd.DataFrame()
    fig, axes = plt.subplots(2, 3, figsize=(14.0, 7.4))
    colors = {"real": GREY, "generated": PINK}
    names = {"real": "реальные ветви", "generated": "сгенерированные"}
    ax = axes[0][0]
    classes = [c for c in ["< 20 um", "20-50 um", ">= 50 um"] if c in set(b["length_class"].dropna())]
    series = {names[k]: [float(b[(b["kind"] == k) & (b["length_class"] == c)]["density_per_um"].median()) for c in classes] for k in ("real", "generated")}
    grouped_bars(ax, [c.replace("um", "мкм") for c in classes], series, [GREY, PINK], fmt="{:.2f}")
    ax.set_title("Плотность шипиков, на мкм (медиана)"); ax.legend(fontsize=9)
    ax = axes[0][1]
    for k in ("real", "generated"):
        v = b.loc[b["kind"] == k, "n_spines"].dropna()
        ax.hist(v, bins=np.linspace(0, max(1, b["n_spines"].max()), 22), density=True, histtype="stepfilled" if k == "real" else "step", color=colors[k], alpha=0.5 if k == "real" else 1.0, lw=2.0, label=f"{names[k]} (n = {len(v)})")
    ax.set_title("Число шипиков на ветви"); ax.legend(fontsize=9); ax.set_yticks([])
    ax = axes[0][2]
    for k in ("real", "generated"):
        v = b.loc[b["kind"] == k, "tortuosity"].dropna() if "tortuosity" in b.columns else pd.Series(dtype=float)
        if len(v):
            ax.hist(v, bins=np.linspace(1.0, min(2.0, v.max() + 0.01), 25), density=True, histtype="stepfilled" if k == "real" else "step", color=colors[k], alpha=0.5 if k == "real" else 1.0, lw=2.0, label=names[k])
    ax.set_title("Извилистость ствола"); ax.legend(fontsize=9); ax.set_yticks([])
    if not s.empty:
        for ax, col, title, xl in [(axes[1][0], "u", "Положение вдоль ветви", "u = s / L"), (axes[1][1], "nn_surface_nm", "Расстояние до ближайшего шипика", "нм"), (axes[1][2], "gap_nm", "Интервал между соседними шипиками", "нм")]:
            if col not in s.columns:
                continue
            vals = {k: s.loc[s["kind"] == k, col].dropna().to_numpy(float) for k in ("real", "generated")}
            hi = np.percentile(np.concatenate(list(vals.values())), 99) if col != "u" else 1.0
            bins = np.linspace(0, hi, 26)
            for k, v in vals.items():
                ax.hist(v[v <= hi], bins=bins, density=True, histtype="stepfilled" if k == "real" else "step", color=colors[k], alpha=0.5 if k == "real" else 1.0, lw=2.0, label=f"{names[k]} (n = {len(v)})")
            ax.set_title(title); ax.set_xlabel(xl); ax.legend(fontsize=8.5); ax.set_yticks([])
    else:
        for ax in axes[1]:
            ax.axis("off")
        axes[1][1].text(0.5, 0.5, "evaluation_spines.csv не найден:\nповторите evaluate_assembled_branches.py\nс текущим скриптом", ha="center", va="center", transform=axes[1][1].transAxes)
    fig.suptitle("Собранные ветви против реальных ветвей из тех же запросов", fontsize=14, x=0.01, ha="left")
    fig.tight_layout()
    save(fig, out, "c6_assembled_eval.png")


# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--static", action="store_true", help="schemes (A) and report numbers (B)")
    parser.add_argument("--training", nargs="*", default=[], help='"Name=<run dir>" ... (c1)')
    parser.add_argument("--evaluation", type=Path, help="generator-evaluation run directory (c2, c3, c4)")
    parser.add_argument("--kind", default="postprocessed", choices=("raw", "postprocessed"))
    parser.add_argument("--t0-run", type=Path, help="<id>_t0 run directory (c5)")
    parser.add_argument("--assembled-eval", type=Path, help="assembly directory with evaluation_branches.csv (c6)")
    parser.add_argument("--conditional-logs", nargs=2, type=Path, metavar=("V2", "V3"),
                        help="folders with metrics_train/val.jsonl of the conditional MoGen v2 and v3 (r14)")
    parser.add_argument("--s-marginals", type=Path, metavar="EVAL_RUN", help="evaluate_generators run: slide version of s_marginals (r15)")
    parser.add_argument("--s-marginals-model", default="mogen_v3")
    parser.add_argument("--s-marginals-name", default="v3", help="legend name of the model: a key of MODEL_NAMES (v3, v4, A, vae_flow) or the text")
    parser.add_argument("--s-marginals-floor", default="real_poisson", help="real spines through the model's chain: real_poisson (MoGen) / real_mc (VAE)")
    parser.add_argument("--s-marginals-file", default="r15_s_marginals_mogen.png")
    parser.add_argument("--species-marginals", type=Path, metavar="REPORT", help="compare_species_generation report of v4 (r16, needs the stage-8 tables)")
    args = parser.parse_args()

    if args.s_marginals:
        fig_a13_s_marginals(args.out, args.s_marginals, args.s_marginals_model, MODEL_NAMES.get(args.s_marginals_name, args.s_marginals_name),
                            args.s_marginals_floor, args.s_marginals_file)
    if args.species_marginals:
        fig_a14_species_marginals(args.out, args.species_marginals)
    if args.conditional_logs:
        fig_a12_conditional_curves(args.out, *args.conditional_logs)
    if args.static:
        for fn in STATIC:
            fn(args.out)
    if args.training:
        runs = {}
        for item in args.training:
            name, _, path = item.partition("=")
            runs[name] = Path(path)
        fig_c1_training(args.out, runs)
    if args.evaluation:
        fig_c2_morph_w1(args.out, args.evaluation, args.kind)
        fig_c3_morph_hist(args.out, args.evaluation, args.kind)
        fig_c4_scorecard(args.out, args.evaluation, args.kind)
    if args.t0_run:
        fig_c5_t0(args.out, args.t0_run)
    if args.assembled_eval:
        fig_c6_assembled(args.out, args.assembled_eval)
    if not (args.static or args.conditional_logs or args.s_marginals or args.species_marginals or args.training or args.evaluation or args.t0_run or args.assembled_eval):
        parser.error("nothing to do: give --static and/or --training / --evaluation / --t0-run / --assembled-eval")


if __name__ == "__main__":
    main()
