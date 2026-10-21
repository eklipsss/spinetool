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

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
    "font.family": "DejaVu Sans", "font.size": 11, "text.color": INK, "axes.labelcolor": INK,
    "axes.edgecolor": "#c9cbd0", "xtick.color": "#4a4d55", "ytick.color": "#4a4d55",
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": "#ececef", "grid.linewidth": 0.8,
    "axes.axisbelow": True, "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left",
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
        ax.set_title(title, fontsize=11)
        ax.text(0.02, 0.03, f"медиана извилистости на 80 мкм: {np.median(tort):.2f}", transform=ax.transAxes, fontsize=10, color=INK)
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("Одинаковый среднеквадратичный изгиб, разное направление (упрощённая двумерная иллюстрация, девять ветвей по 80 мкм из одной точки)", fontsize=10.5, x=0.01, ha="left")
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
    ax.set_title("Функция немонотонна: простой перебор «глубже, пока внутри» не работает", fontsize=11)
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
        ax.set_title({0.0: "t = 0: шум", 1.0: "t = 1: форма шипика"}.get(t, f"t = {t:g}"), fontsize=10.5)
    fig.suptitle("Flow Matching на плоскости: каждая точка движется по прямой от шума к форме; сеть учится угадывать эту скорость", fontsize=11, x=0.01, ha="left", y=0.98)
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
    fig.suptitle("Адаптивное сглаживание центральной линии: 120 ветвей, 10 нейронов", fontsize=11, x=0.01, ha="left")
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
        ax.set_xticks(range(3), labels, fontsize=8.6); ax.set_title(title, fontsize=10.5); ax.grid(axis="x", visible=False)
        ax.set_ylim(0, max(vals) * 1.18)
    fig.suptitle("VAE: три рецепта обучения (50 проверочных шипиков, шаг 30 000)", fontsize=11, x=0.01, ha="left")
    save(fig, out, "b4_vae_recipes.png")


def fig_b5_assembly(out: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.2))
    grouped_bars(axes[0], ["все ветви", "20–50 мкм", "≥ 50 мкм"], {"реальные ветви": [0.84, 1.01, 0.98], "сгенерированные": [0.53, 0.54, 0.70]}, [GREY, PINK])
    axes[0].set_title("Плотность шипиков, на мкм"); axes[0].set_ylim(0, 1.25); axes[0].legend(loc="upper right", fontsize=9)
    grouped_bars(axes[1], ["разброс радиуса\n(коэфф. вариации)"], {"реальные ветви": [0.12], "сгенерированные": [0.07]}, [GREY, PINK], width=0.6)
    axes[1].set_title("Разброс радиуса вдоль ветви"); axes[1].set_ylim(0, 0.16)
    grouped_bars(axes[2], ["расстояние до ближайшего\nшипика, нм (медиана)"], {"реальные ветви": [842], "сгенерированные": [802]}, [GREY, PINK], width=0.6, fmt="{:.0f}")
    axes[2].set_title("Расстояние до ближайшего соседа"); axes[2].set_ylim(0, 1000)
    fig.suptitle("Собранные ветви против реальных ветвей тех же запросов (50 ветвей; положения вдоль ветви: W₁/σ = 0,07, на уровне шумового пола)", fontsize=10.5, x=0.01, ha="left")
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
    fig.suptitle("Аудит обратной реконструкции: ошибку определяют выросты сегментации, эллипс почти не помогает", fontsize=11, x=0.01, ha="left")
    save(fig, out, "b6_audit_decomposition.png")


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
            ax.plot(df["step"], y, color=col, alpha=0.18, lw=0.8)
            ax.plot(df["step"], y.rolling(10, min_periods=1).mean(), color=col, lw=2.0, label=name)
        ax.set_xlabel("шаг обучения"); ax.set_ylabel("loss Flow Matching (начальный спад B обрезан по оси)")
        ax.set_title("MoGen: loss выходит на плато рано"); ax.legend()
        lo = min(df["loss"].quantile(0.02) for df in mogen.values()); ax.set_ylim(lo * 0.95, min(0.9, max(df["loss"].iloc[min(20, len(df) - 1)] for df in mogen.values()) * 1.05))
    if vae:
        ax = axes[0][ai]
        for name, df in vae.items():
            for column, col, lab in [("sdf_loss", PINK, "ошибка расстояния"), ("surface_loss", GREEN, "поверхность"), ("normal_loss", VIOLET, "нормаль"), ("eikonal_loss", GREY, "эйконал")]:
                if column in df.columns:
                    ax.plot(df["step"], df[column].rolling(10, min_periods=1).mean(), color=col, lw=1.8, label=lab)
        ax.set_yscale("log"); ax.set_xlabel("шаг обучения"); ax.set_title("VAE: компоненты функции потерь"); ax.legend(fontsize=9)
    save(fig, out, "c1_training_curves.png")


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
    features = [f for f in tables[next(iter(tables))]["feature"]]
    labels = [info.get(f, {}).get("name", f) for f in features]
    series = {n: [float(t.set_index("feature").loc[f, "w1_normalized"]) if f in set(t["feature"]) else np.nan for f in features] for n, t in tables.items()}
    fig, ax = plt.subplots(figsize=(10.5, 6.0))
    grouped_bars(ax, labels, series, SERIES, horizontal=True, fmt="{:.2f}")
    ax.set_xlabel("W₁ / σ (расстояние Вассерштейна в долях разброса; меньше — лучше)")
    ax.set_title(f"Совпадение распределений морфометрик с реальными шипиками ({KIND_RU[kind]})"); ax.legend(loc="lower right")
    save(fig, out, "c2_morph_w1.png")


def fig_c3_morph_hist(out: Path, evaluation: Path, kind: str, features=("Length", "Volume", "Area", "OpenAngle", "CVD", "JunctionArea")) -> None:
    real_path = Path(evaluation) / "real_test_morphometrics.parquet"
    gen_files = sorted(_eval_dir(evaluation, kind).glob("morphometrics_*_seed*.parquet"))
    if not real_path.exists() or not gen_files:
        print(f"[figures] c3: needs real_test_morphometrics.parquet and {kind}/morphometrics_*.parquet (written by the current evaluate_generators.py)")
        return
    info = _metric_info()
    real = pd.read_parquet(real_path)
    gens = {f.stem[len("morphometrics_"):].rsplit("_seed", 1)[0]: pd.read_parquet(f) for f in gen_files}
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
        for (name, g), col, p in zip(gens.items(), SERIES, pool[1:]):
            ax.hist(p, bins=bins, density=True, histtype="step", color=col, lw=2.0, label=f"{name} (n = {len(p)})")
        ax.set_title(meta["name"], fontsize=11); ax.set_xlabel(f"{meta['name']}, {meta['unit']}".rstrip(", ")); ax.set_yticks([])
    axes[0][0].legend(fontsize=8.5)
    fig.suptitle(f"Распределения морфометрик: тестовые реальные шипики и сгенерированные ({KIND_RU[kind]})", fontsize=12, x=0.01, ha="left")
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
    models = [c for c in table.columns if c not in ("criterion", "target")]
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
    tb = ax.table(cellText=cell_text, colLabels=["критерий", "ориентир", *models], cellColours=colors, loc="center", cellLoc="center", colLoc="center")
    tb.auto_set_font_size(False); tb.set_fontsize(9.5); tb.scale(1, 1.5)
    tb.auto_set_column_width(list(range(len(models) + 2)))
    for (r, c), cell in tb.get_celld().items():
        cell.set_edgecolor("#e6e6ea")
        if r == 0:
            cell.set_text_props(fontweight="bold"); cell.set_facecolor("#f6f6f8")
        if c == 0 and r > 0:
            cell.set_text_props(ha="left"); cell._loc = "left"
    ax.set_title(f"Сводка критериев ({KIND_RU[kind]}): зелёный — лучшее значение в строке, розовый — худшее", fontsize=10.5, loc="left")
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
    fig.suptitle("Собранные ветви против реальных ветвей из тех же запросов", fontsize=12, x=0.01, ha="left")
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
    args = parser.parse_args()

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
    if not (args.static or args.training or args.evaluation or args.t0_run or args.assembled_eval):
        parser.error("nothing to do: give --static and/or --training / --evaluation / --t0-run / --assembled-eval")


if __name__ == "__main__":
    main()
