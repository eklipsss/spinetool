"""``curves.png`` (matplotlib) and ``curves.html`` (project HTML kit) for :mod:`training_curves`."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Mapping

import numpy as np
import pandas as pd

from ...reports import html_kit as hk
from .training_curves import TITLES, metric_names

BAND = 2.0  # bands are +-2 standard errors


def _smooth(values: np.ndarray, window: int) -> np.ndarray:
    window = min(window, len(values))
    if window < 2:
        return values
    kernel = np.ones(window)
    num = np.convolve(np.nan_to_num(values), kernel, mode="same")
    den = np.convolve(np.isfinite(values).astype(float), kernel, mode="same")
    return num / np.maximum(den, 1)


def _panels(kind: str, curves: pd.DataFrame) -> List[str]:
    return metric_names(kind, curves.to_dict("records"))


def write_png(path: Path, kind: str, curves: pd.DataFrame, logs: Mapping[str, pd.DataFrame], summary: Mapping[str, Any]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    p = hk.PALETTE
    names = _panels(kind, curves)
    panels = names + ["gap"]
    ncols = 3 if len(panels) > 4 else 2
    nrows = int(np.ceil(len(panels) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.4 * ncols, 3.6 * nrows), squeeze=False)
    for ax, name in zip(axes.ravel(), panels):
        for which, frame in curves.groupby("weights"):
            frame = frame.sort_values("step")
            style = "-" if which == "ema" else "--"
            if name == "gap":
                main = "total_final" if kind == "vae" else "loss"
                y, se = frame[f"gap_{main}"].to_numpy(), frame[f"gap_{main}_se"].to_numpy()
                ax.plot(frame["step"], y, style, color=p["series_3"], label=f"val - train ({which})")
                ax.fill_between(frame["step"], y - BAND * se, y + BAND * se, color=p["series_3"], alpha=0.18, linewidth=0)
                ax.axhline(0, color=p["border"], linewidth=1)
                continue
            for split, color, label in (("train", p["series_1"], "train, фиксированная выборка"), ("val", p["series_2"], "validation")):
                if f"{split}_{name}" not in frame:
                    continue
                y, se = frame[f"{split}_{name}"].to_numpy(), frame[f"{split}_{name}_se"].to_numpy()
                ax.plot(frame["step"], y, style, color=color, marker="o", markersize=3, label=f"{label} ({which})")
                ax.fill_between(frame["step"], y - BAND * se, y + BAND * se, color=color, alpha=0.18, linewidth=0)
        if name != "gap":
            log = logs.get("train_log")
            if log is not None and name in log:
                ax.plot(log["step"], _smooth(log[name].to_numpy(), 5), color=p["series_1"], alpha=0.35, linewidth=1, label="train, бегущее среднее лога (train-режим)")
            inloop = logs.get("val_inloop")
            if inloop is not None and name in inloop:
                ax.plot(inloop["step"], inloop[name], "D", color=p["reference"], markersize=4, label="validation в цикле (in-loop)")
            main = "total_final" if kind == "vae" else "loss"
            best = summary.get("ema") or next((v for k, v in summary.items() if isinstance(v, dict)), None)
            if best and name == main:
                ax.axvline(best["best_step"], color=p["series_2"], linestyle=":", linewidth=1.2)
        ax.set_title(TITLES.get(name, "Разрыв val − train (±2 SE)") if name != "gap" else "Разрыв val − train (±2 SE)", fontsize=10, color=p["ink"])
        ax.set_xlabel("шаг", fontsize=9)
        ax.grid(color=p["grid"])
        ax.tick_params(labelsize=8)
        if name in ("kl", "far", "eikonal", "normal", "surface") and (curves.get(f"val_{name}", pd.Series([1.0])) > 0).all():
            ax.set_yscale("log")
    for ax in axes.ravel()[len(panels):]:
        ax.axis("off")
    axes.ravel()[0].legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=140, facecolor="white")
    plt.close(fig)


PAGE_JS = r"""
const D = PAYLOAD.data;
function band(x, y, se, color, name, dash) {
  const up = y.map((v, i) => v === null || se[i] === null ? null : v + PAYLOAD.band * se[i]);
  const dn = y.map((v, i) => v === null || se[i] === null ? null : v - PAYLOAD.band * se[i]);
  return [
    {x: x, y: up, mode: 'lines', line: {width: 0}, hoverinfo: 'skip', showlegend: false},
    {x: x, y: dn, mode: 'lines', line: {width: 0}, fill: 'tonexty', fillcolor: color + '2e', hoverinfo: 'skip', showlegend: false},
    {x: x, y: y, mode: 'lines+markers', name: name, line: {color: color, width: 2, dash: dash || 'solid'}, marker: {size: 5},
     customdata: se, hovertemplate: '%{x}: %{y:.5g} ± %{customdata:.2g} (SE)<extra>' + name + '</extra>'},
  ];
}
function draw(which) {
  const rows = D.rows.filter(r => r.weights === which);
  const steps = rows.map(r => r.step);
  const col = (k) => rows.map(r => r[k]);
  for (const m of D.panels) {
    const traces = [];
    if (m === 'gap') {
      traces.push(...band(steps, col('gap_' + D.main), col('gap_' + D.main + '_se'), P.series_3, 'val − train'));
    } else {
      if (D.train_log && D.train_log[m]) {
        const w = 5, y = D.train_log[m], sm = y.map((v, i) => { let s = 0, n = 0; for (let j = Math.max(0, i - 2); j <= Math.min(y.length - 1, i + 2); j++) if (y[j] !== null) { s += y[j]; n++; } return n ? s / n : null; });
        traces.push({x: D.train_log.step, y: sm, mode: 'lines', name: 'train: бегущее среднее лога (train-режим)', line: {color: P.series_1 + '66', width: 1.5}, hovertemplate: '%{x}: %{y:.5g}<extra>log</extra>'});
      }
      if (rows.length && rows[0]['train_' + m] !== undefined) traces.push(...band(steps, col('train_' + m), col('train_' + m + '_se'), P.series_1, 'train: фиксированная выборка'));
      if (rows.length && rows[0]['val_' + m] !== undefined) traces.push(...band(steps, col('val_' + m), col('val_' + m + '_se'), P.series_2, 'validation: фиксированная выборка'));
      if (D.val_inloop && D.val_inloop[m]) traces.push({x: D.val_inloop.step, y: D.val_inloop[m], mode: 'markers', name: 'validation в цикле (in-loop)', marker: {color: P.reference, symbol: 'diamond', size: 7}});
    }
    const shapes = [];
    if (m === D.main && D.best[which]) shapes.push({type: 'line', x0: D.best[which].best_step, x1: D.best[which].best_step, yref: 'paper', y0: 0, y1: 1, line: {color: P.series_2, dash: 'dot', width: 1.5}});
    if (m === 'gap') shapes.push({type: 'line', xref: 'paper', x0: 0, x1: 1, y0: 0, y1: 0, line: {color: P.border, width: 1}});
    Plotly.react('plot_' + m, traces, layout({xaxis: {title: axisTitle('шаг')}, yaxis: {title: axisTitle(D.titles[m]), type: D.log[m] ? 'log' : 'linear'}, shapes: shapes,
      legend: {orientation: 'h', y: -0.22}, hovermode: 'x unified'}), PLOT_CONFIG);
  }
}
const sel = $('weights');
for (const w of D.weights) sel.append(el('option', {value: w, text: w === 'ema' ? 'EMA-веса' : 'обычные веса'}));
sel.onchange = () => draw(sel.value);
draw(D.weights[0]);
"""


def write_html(path: Path, kind: str, curves: pd.DataFrame, logs: Mapping[str, pd.DataFrame], summary: Mapping[str, Any], cfg: Mapping[str, Any], run_name: str,
               settings: Mapping[str, Any]) -> None:
    names = _panels(kind, curves)
    main = "total_final" if kind == "vae" else "loss"
    panels = names + ["gap"]
    data = {
        "rows": curves.replace({np.nan: None}).to_dict("records"),
        "panels": panels, "main": main, "weights": sorted(curves["weights"].unique(), key=lambda w: w != "ema"),
        "titles": {**TITLES, "gap": "val − train"},
        "log": {n: bool(n in ("kl", "far", "eikonal", "normal", "surface") and (curves[f"val_{n}"] > 0).all()) for n in names},
        "best": {k: v for k, v in summary.items() if isinstance(v, dict)},
        "train_log": {k: [None if not np.isfinite(x) else float(x) for x in v] for k, v in logs["train_log"].items()} if "train_log" in logs else None,
        "val_inloop": {k: [None if not np.isfinite(x) else float(x) for x in v] for k, v in logs["val_inloop"].items()} if "val_inloop" in logs else None,
    }
    best = data["best"].get("ema") or next(iter(data["best"].values()))
    trend = best.get("gap_trend", {})
    tiles = [
        {"label": "Лучший чекпоинт по validation", "value": f"шаг {best['best_step']}", "note": f"{best['best_val']:.5g} ± {best['best_val_se']:.2g} (SE)", "accent": "green"},
        {"label": "Разрыв val − train на последнем шаге", "value": f"{best.get('gap_last', float('nan')):.4g}", "note": f"± {best.get('gap_last_se', float('nan')):.2g} (SE)", "accent": "pink"},
        {"label": "Тренд разрыва (вторая половина)", "value": "растёт" if trend.get("growing") else "не растёт",
         "note": f"наклон {trend.get('slope_per_1000', float('nan')):.3g} ± {trend.get('slope_se', float('nan')):.2g} на 1000 шагов"},
        {"label": "Выборка оценки", "value": f"{settings['n_spines']} + {settings['n_spines']} шипиков", "note": f"validation + фиксированное подмножество train; seed {settings['seed']}"},
    ]
    cards = "".join(hk.card(TITLES.get(n, "Разрыв val − train (±2 SE)") if n != "gap" else "Разрыв val − train", f'<div id="plot_{n}" class="plot s"></div>') for n in panels)
    if kind == "vae":
        method = ("Потери VAE считаются в режиме eval, код z = mu, для одних и тех же шипиков и одних и тех же точек запроса на каждом чекпоинте. "
                  "Суммарные потери показаны дважды: с весами КЛ и регуляризаторов на шаге чекпоинта (как обучалась модель, видно разогрев KL) и с весами финального шага "
                  "(фиксированная цель: кривая не искажена разогревом).")
    else:
        method = ("Потери Flow Matching считаются в режиме eval для одних и тех же шипиков; момент времени t и шум для каждого батча задаются фиксированным генератором "
                  f"({settings['n_draws']} независимых набора), одинаковым на всех чекпоинтах, для train и validation.")
    notes = hk.note([
        method,
        "Кривая train — оценка ТОЙ ЖЕ функцией на фиксированном случайном подмножестве обучающих шипиков того же размера, что и validation, поэтому train и validation сравнимы напрямую. "
        "Бегущее среднее из лога обучения (светлая линия) — другая величина: оно считается в режиме train, со случайной аугментацией и случайным кодом, на каждом шаге на других данных; "
        "оно показано для контроля и не должно совпадать с кривой фиксированной выборки.",
        f"Полосы — ±{BAND:g} стандартных ошибки по батчам (средневзвешенное по размеру батча, оценка отношения). Неопределённость отражает выборку шипиков, а не только шум оценки; "
        "чекпоинты оцениваются на одних и тех же батчах, поэтому разность соседних чекпоинтов точнее, чем следует из полос. Вертикальная пунктирная линия — лучший по validation чекпоинт.",
        "Ромбики — валидация внутри цикла обучения (если она шла): она использует другое подмножество шипиков, поэтому может отличаться от кривой фиксированной выборки на величину ошибки выборки.",
    ], title="Как читать", collapsible=True)
    body = (hk.header("Кривые обучения", f"{run_name}", [f"тип: {kind}", f"чекпоинтов: {curves['step'].nunique()}"]) + hk.tiles(tiles)
            + '<div class="row" style="margin-top:14px"><label class="small muted" for="weights">Веса:</label><select id="weights" style="width:auto"></select></div>'
            + notes + f'<div class="grid-2e" style="margin-top:14px">{cards}</div>')
    path.write_text(hk.page(f"Кривые обучения — {run_name}", body, {"data": data, "band": BAND}, PAGE_JS), encoding="utf-8")
