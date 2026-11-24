"""Quality of the generation, real vs generated, per module: tables and distribution figures (2026-11-24).

Protocol: ``docs/neuron-model/generation-quality-protocol.md``. Everything here READS the outputs of the existing
evaluations, nothing is regenerated:

* S, overall: an ``evaluate_generators.py`` run (``real_test_morphometrics.parquet`` + ``postprocessed/
  morphometrics_<model>_seed<k>.parquet`` + ``postprocessed/summary.csv``): the headline table (validity, MMD,
  correlations, COV / 1-NNA / MMD-CD, diversity, memorization, classifier, W1/sigma) and, per morphometric, the shift of the
  median d = ln(median gen / median real) with an interval from resampling the real NEURONS (and the generated samples),
  Benjamini-Hochberg q, W1/sigma; marginal densities and ECDFs of the 11 morphometrics, Spearman correlation matrices.
* S, per cell type: a ``conditional_generation_report.py`` folder (``generated_metrics.parquet`` with the type of every
  generated sample, ``same_type_*.csv``, ``concordance_*.csv``) + the real spines of the type (the real table with
  ``spine_key`` and a neuron -> type map).
* T and P: a ``generate_tp.py`` set and its ``evaluate_assembled_branches.py`` folder (``evaluation_branches.csv``,
  ``evaluation_spines.csv``): trunk functionals and branch-level counts (one row per branch, generated row i = request i,
  real row i = its source branch), spine-level statistics (nearest neighbour on the surface, gaps, u, theta); the
  difference of the medians with the interval from resampling NEURONS (a generated branch belongs to the neuron of its
  source), overall and per group (species, cell type, compartment).

Figures: real = grey filled, generated = pink line, floor (real through the same reconstruction chain) = green dashed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

S_METRICS = ("OpenAngle", "CVD", "AverageDistance", "LengthVolumeRatio", "LengthAreaRatio", "Length", "JunctionArea",
             "Area", "Volume", "ConvexHullVolume", "ConvexHullRatio")
S_NAMES = {"OpenAngle": "угол раскрытия", "CVD": "CVD", "AverageDistance": "среднее расстояние", "LengthVolumeRatio": "длина / объём",
           "LengthAreaRatio": "длина / площадь", "Length": "длина", "JunctionArea": "площадь основания", "Area": "площадь",
           "Volume": "объём", "ConvexHullVolume": "объём выпуклой оболочки", "ConvexHullRatio": "ConvexHullRatio"}
# display of the figures: (factor from nm units, unit) - the data stay in nm
S_DISPLAY = {"AverageDistance": (1e-3, "мкм"), "LengthVolumeRatio": (1e6, "1/мкм²"), "LengthAreaRatio": (1e3, "1/мкм"), "Length": (1e-3, "мкм"),
             "JunctionArea": (1e-6, "мкм²"), "Area": (1e-6, "мкм²"), "Volume": (1e-9, "мкм³"), "ConvexHullVolume": (1e-9, "мкм³"),
             "OpenAngle": (1.0, "рад")}
S_LABELS = {m: S_NAMES[m] + (f", {S_DISPLAY[m][1]}" if m in S_DISPLAY else "") for m in S_NAMES}
SUMMARY_NAMES = {
    "validity_postprocessed.valid_mesh_rate": "Валидных сеток (после постобработки)",
    "mmd2_morphometrics": "MMD² по морфометрике",
    "correlation_frobenius": "Расхождение корреляций (норма Фробениуса)",
    "coverage_cd": "Покрытие (COV-CD)",
    "one_nna_cd": "1-NNA (CD)",
    "mmd_cd": "MMD-CD",
    "diversity.chamfer_mean_ratio": "Разнообразие по формам (Chamfer, ген. / реал.)",
    "diversity.morphometric_mean_ratio": "Разнообразие по признакам (ген. / реал.)",
    "memorization.morphometric_dcr_median_ratio": "Запоминание: DCR по признакам (ген. / отложенные)",
    "memorization.chamfer_dcr_median_ratio": "Запоминание: DCR по формам (ген. / отложенные)",
    "memorization.morphometric_near_copy_rate": "Доля почти копий (признаки)",
    "memorization.chamfer_near_copy_rate": "Доля почти копий (формы)",
    "classifier_accuracy": "Классификатор «реальный / сгенерированный»: точность",
    "w1_normalized.Area": "W₁/σ площади", "w1_normalized.CVD": "W₁/σ CVD", "w1_normalized.Length": "W₁/σ длины",
    "w1_normalized.OpenAngle": "W₁/σ угла раскрытия", "w1_normalized.Volume": "W₁/σ объёма",
}
T_FUNCTIONALS = {"mean_radius_nm": "средний радиус, нм", "radius_ratio_prox_dist": "радиус прокс. / дист.", "radius_cv": "CV радиуса",
                 "kappa_rms_per_nm": "изгиб κ rms, 1/нм", "tortuosity": "извилистость", "kappa_autocorr_1um": "автокорреляция изгиба, 1 мкм"}
P_BRANCH = {"n_spines": "шипиков на ветвь", "density_per_um": "плотность, 1/мкм", "no_spines": "доля ветвей без шипиков"}
P_SPINE = {"nn_surface_nm": "ближайший сосед на поверхности, нм", "gap_nm": "интервал вдоль ветви, нм", "u": "положение вдоль ветви u", "theta": "угол вокруг оси θ"}

REAL_C, GEN_C, FLOOR_C = "#8a8a8a", "#c2185b", "#2e7d32"
GEN_EXTRA = ("#c2185b", "#6a1b9a", "#ef6c00", "#1565c0")


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _neuron(key: str) -> str:
    parts = str(key).split("/")
    return parts[1] if len(parts) > 1 else parts[0]


def _bh(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, float)
    q = np.full_like(p, np.nan)
    ok = np.isfinite(p)
    if ok.any():
        pv = p[ok]
        order = np.argsort(pv)
        qq = np.empty_like(pv)
        qq[order] = np.minimum.accumulate((pv[order] * len(pv) / np.arange(1, len(pv) + 1))[::-1])[::-1]
        q[ok] = np.minimum(qq, 1.0)
    return q


def w1_sd(a: np.ndarray, b: np.ndarray) -> float:
    """W1 between two samples in units of the sd of ``b`` (the real one)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2 or b.std() == 0:
        return float("nan")
    q = (np.arange(512) + 0.5) / 512
    return float(np.mean(np.abs(np.quantile(a, q) - np.quantile(b, q))) / b.std())


def _density(ax, v: np.ndarray, grid: np.ndarray, log: bool, **kw) -> None:
    from scipy.stats import gaussian_kde

    v = v[np.isfinite(v) & (v >= grid[0]) & (v <= grid[-1])]  # outliers would inflate the KDE bandwidth (Scott: sd-based)
    if len(v) < 5 or np.std(v) == 0:
        return
    x = np.log(v) if log else v
    g = np.log(grid) if log else grid
    dens = gaussian_kde(x)(g)
    ax.plot(grid, dens, **kw)


def _hist_panel(ax, real: np.ndarray, gens: Dict[str, np.ndarray], floor: Optional[np.ndarray], title: str, log: bool):
    """Smoothed densities (Gaussian KDE; on the log scale for size-like quantities): real filled grey, floor dashed green,
    generated lines. With a log x axis the density is per unit of log x, so the areas stay comparable."""
    vals = [v for v in [real, floor, *gens.values()] if v is not None and len(v)]
    allv = np.concatenate(vals)
    allv = allv[np.isfinite(allv)]
    if log:
        allv = allv[allv > 0]
    if not len(allv):
        ax.set_title(title)
        return
    lo, hi = np.quantile(allv, [0.005, 0.995])
    if not hi > lo:
        ax.set_title(f"{title}\n(константа {lo:.3g})", fontsize=9)
        ax.set_yticks([])
        return
    log = log and lo > 0
    grid = np.geomspace(lo, hi, 300) if log else np.linspace(lo, hi, 300)
    from scipy.stats import gaussian_kde

    r = real[np.isfinite(real) & (real >= lo) & (real <= hi)]
    if len(r) >= 5 and np.std(r) > 0:
        x = np.log(r) if log else r
        if len(x) > 20000:
            x = np.random.default_rng(0).choice(x, 20000, replace=False)
        ax.fill_between(grid, gaussian_kde(x)(np.log(grid) if log else grid), color=REAL_C, alpha=0.45, label="реальные", lw=0)
    if floor is not None and len(floor):
        _density(ax, floor, grid, log, color=FLOOR_C, lw=1.6, ls="--", label="пол (реальные через цепочку)")
    for (name, v), c in zip(gens.items(), GEN_EXTRA):
        _density(ax, v, grid, log, color=c, lw=1.8, label=name)
    if log:
        ax.set_xscale("log")
        from matplotlib.ticker import LogFormatterSciNotation, NullFormatter

        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.xaxis.set_major_formatter(LogFormatterSciNotation(labelOnlyBase=False, minor_thresholds=(2, 0.5)))
    ax.set_title(title, fontsize=9)
    ax.tick_params(labelsize=7)
    ax.set_yticks([])


def _ecdf_panel(ax, real, gens, floor, title, log):
    def line(v, **kw):
        v = np.sort(v[np.isfinite(v)])
        if len(v):
            ax.plot(v, np.arange(1, len(v) + 1) / len(v), **kw)
    line(real, color=REAL_C, lw=2.5, label="реальные")
    if floor is not None:
        line(floor, color=FLOOR_C, lw=1.5, ls="--", label="пол")
    for (name, v), c in zip(gens.items(), GEN_EXTRA):
        line(v, color=c, lw=1.5, label=name)
    if log:
        ax.set_xscale("log")
    ax.set_title(title, fontsize=9)
    ax.tick_params(labelsize=7)


def distribution_grid(real: pd.DataFrame, gens: Dict[str, pd.DataFrame], floor: Optional[pd.DataFrame], columns: Dict[str, str],
                      path: Path, kind: str = "hist", log_columns: Sequence[str] = (), suptitle: str = "", ncols: int = 4,
                      scale: Optional[Dict[str, float]] = None) -> Path:
    plt = _plt()
    cols = [c for c in columns if c in real.columns]
    nrows = int(np.ceil(len(cols) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.4 * ncols, 2.6 * nrows), squeeze=False)
    scale = scale or {}
    for ax, c in zip(axes.ravel(), cols):
        k = scale.get(c, 1.0)
        r = real[c].to_numpy(float) * k
        g = {n: f[c].to_numpy(float) * k for n, f in gens.items() if c in f.columns}
        fl = floor[c].to_numpy(float) * k if floor is not None and c in floor.columns else None
        (_hist_panel if kind == "hist" else _ecdf_panel)(ax, r, g, fl, columns[c], c in log_columns)
    for ax in axes.ravel()[len(cols):]:
        ax.axis("off")
    h, lab = axes.ravel()[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=min(len(lab), 4), fontsize=8, frameon=False)
    if suptitle:
        fig.suptitle(suptitle, fontsize=11)
    fig.tight_layout(rect=(0, 0.06, 1, 0.96 if suptitle else 1))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------------------------------------------- S


def _read_gen_sets(pp: Path) -> Dict[str, pd.DataFrame]:
    sets: Dict[str, List[pd.DataFrame]] = {}
    for f in sorted(pp.glob("morphometrics_*_seed*.parquet")):
        stem = f.stem[len("morphometrics_"):]
        model, seed = stem.rsplit("_seed", 1)
        sets.setdefault(model, []).append(pd.read_parquet(f).assign(generation_seed=int(seed)))
    return {m: pd.concat(v, ignore_index=True) for m, v in sets.items()}


def summary_table(summary_csv: Path) -> List[str]:
    t = pd.read_csv(summary_csv)
    models = [c for c in t.columns if c not in ("criterion", "target")]
    lines = ["| критерий | цель | " + " | ".join(models) + " |", "| --- | --- |" + " --- |" * len(models)]
    for _, r in t.iterrows():
        lines.append(f"| {SUMMARY_NAMES.get(r['criterion'], r['criterion'])} | {r['target']} | " + " | ".join(str(r[m]) for m in models) + " |")
    missing = [k for k in SUMMARY_NAMES if k not in set(t["criterion"])]
    if missing:
        lines += ["", "Нет в этой оценке (не считалось): " + "; ".join(SUMMARY_NAMES[k] for k in missing) + "."]
    return lines


def s_overall(eval_run: Path, out: Path, *, floor_model: str = "real_poisson", n_boot: int = 1000, seed: int = 0) -> List[str]:
    """Headline table, per-metric shifts and distribution figures of an evaluate_generators run."""
    from .spine_generation.evaluation.density_ratio import per_metric_shift

    eval_run, out = Path(eval_run), Path(out)
    real = pd.read_parquet(eval_run / "real_test_morphometrics.parquet", columns=["spine_key", *S_METRICS])
    sets = _read_gen_sets(eval_run / "postprocessed")
    floor = sets.get(floor_model)
    gens = {m: f for m, f in sets.items() if m != floor_model}
    lines = [f"## S: общая оценка (`{eval_run.name}`)", "",
             f"Реальные: {len(real)} шипиков {real['spine_key'].map(_neuron).nunique()} проверочных нейронов. Наборы: "
             + "; ".join(f"{m} — {len(f)} ({f['generation_seed'].nunique()} зерна)" for m, f in sets.items()) + ".", ""]
    summary_csv = eval_run / "postprocessed" / "summary.csv"
    if summary_csv.exists():
        lines += ["### Сводные критерии (среднее ± sd по зёрнам)", "", *summary_table(summary_csv), ""]
    table = per_metric_shift(real, sets, metrics=S_METRICS, n_boot=n_boot, seed=seed)
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "s_per_metric.csv", index=False)
    lines += ["### По каждой морфометрике", "",
              "d — отношение медиан сгенерированных и реальных − 1, %; интервал 95 %: реальные перевыбираются по нейронам, "
              "сгенерированные по образцам; * — значимо после поправки Бенджамини — Хохберга по 11 метрикам; W₁/σ — расстояние "
              "Вассерштейна логарифмов в единицах sd реальных.", "",
              "| метрика | " + " | ".join(f"{m}: d, % [95 %] | W₁/σ" for m in sets) + " |", "| --- |" + " --- | --- |" * len(sets)]
    for m in S_METRICS:
        cells = []
        for name in sets:
            r = table[(table["set"] == name) & (table["metric"] == m)].iloc[0]
            cells.append(f"{100 * np.expm1(r.d):+.1f}{'*' if r.significant else ''} [{100 * np.expm1(r.lo):+.0f}; {100 * np.expm1(r.hi):+.0f}] | {r.w1_sd:.2f}")
        lines.append(f"| {S_NAMES[m]} | " + " | ".join(cells) + " |")
    sig = {n: int(g["significant"].sum()) for n, g in table.groupby("set")}
    lines += ["", "Значимых отличий из 11: " + ", ".join(f"{n} {k}" for n, k in sig.items()) + ".", ""]
    log_cols = [m for m in S_METRICS if m not in ("OpenAngle", "CVD", "ConvexHullRatio")]
    sc = {m: f for m, (f, _) in S_DISPLAY.items()}
    distribution_grid(real, gens, floor, S_LABELS, out / "s_marginals.png", "hist", log_cols, "Морфометрики шипиков: реальные против сгенерированных", scale=sc)
    distribution_grid(real, gens, floor, S_LABELS, out / "s_ecdf.png", "ecdf", log_cols, "Эмпирические функции распределения", scale=sc)
    correlation_figure(real, {**gens, **({"пол": floor} if floor is not None else {})}, out / "s_correlations.png")
    lines += ["Рисунки: `s_marginals.png` (плотности), `s_ecdf.png` (функции распределения), `s_correlations.png` "
              "(матрицы корреляций Спирмена и их разность с реальными).", ""]
    return lines


def correlation_figure(real: pd.DataFrame, sets: Dict[str, pd.DataFrame], path: Path) -> Path:
    plt = _plt()
    cols = list(S_METRICS)
    cr = real[cols].corr(method="spearman").to_numpy()
    fig, axes = plt.subplots(1, 1 + len(sets), figsize=(3.6 * (1 + len(sets)), 3.6))
    axes = np.atleast_1d(axes)
    im = axes[0].imshow(cr, vmin=-1, vmax=1, cmap="PiYG")
    axes[0].set_title("реальные", fontsize=9)
    for ax, (name, f) in zip(axes[1:], sets.items()):
        d = f[cols].corr(method="spearman").to_numpy() - cr
        ax.imshow(d, vmin=-1, vmax=1, cmap="PiYG")
        ax.set_title(f"{name} − реальные\n(норма Фробениуса {np.sqrt(np.nansum(d ** 2)):.2f})", fontsize=9)
    labels = [S_NAMES[c] for c in cols]
    for ax in axes:
        ax.set_xticks(range(len(cols)), labels, rotation=90, fontsize=6)
        ax.set_yticks(range(len(cols)), labels if ax is axes[0] else [""] * len(cols), fontsize=6)
    fig.colorbar(im, ax=axes, shrink=0.7)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def s_by_type(report_dir: Path, real: pd.DataFrame, neuron_type: Dict[str, str], out: Path, *, floor_label: str = "real-pw8val") -> List[str]:
    """Per-type comparison: real spines of the type (``real`` with spine_key, mapped by ``neuron_type``) against the generated
    samples of that type (``generated_metrics.parquet``) and the floor; the verdicts of the conditional report."""
    report_dir, out = Path(report_dir), Path(out)
    gm = pd.read_parquet(report_dir / "generated_metrics.parquet")
    gm = gm[gm["valid"].fillna(False).astype(bool)] if "valid" in gm.columns else gm
    real = real.assign(type=real["spine_key"].map(lambda k: neuron_type.get(_neuron(k))))
    lines = [f"## S по типам клеток (`{report_dir.name}`)", ""]
    for value, g in gm.groupby("value"):
        r = real[real["type"] == value]
        labels = [l for l in pd.unique(g["label"]) if l != floor_label]
        floor = g[g["label"] == floor_label]
        if len(r) >= 20:
            distribution_grid(r, {l: g[g["label"] == l] for l in labels}, floor if len(floor) else None, S_LABELS,
                              out / f"s_type_{value}.png", "hist", [m for m in S_METRICS if m not in ("OpenAngle", "CVD", "ConvexHullRatio")],
                              f"{value}: реальные ({r['spine_key'].map(_neuron).nunique()} нейронов) против сгенерированных",
                              scale={m: f for m, (f, _) in S_DISPLAY.items()})
        lines += [f"### {value}", "", f"Реальных шипиков {len(r)} ({r['spine_key'].map(_neuron).nunique()} нейронов); рисунок `s_type_{value}.png`.", ""]
        for l in [*labels, floor_label]:
            f = report_dir / f"same_type_{value}@{l}_vs_{value}.csv"
            if not f.exists():
                continue
            t = pd.read_csv(f)
            counts = t["verdict"].value_counts().to_dict()
            lines.append(f"- {l}: " + ", ".join(f"{k} {v}" for k, v in counts.items())
                         + f"; медиана |d| {100 * np.expm1(t['log_ratio'].abs()).median():.1f} %, медиана наименьшей доказуемой границы "
                         f"эквивалентности {t['min_margin_sd'].median():.2f} σ")
        lines.append("")
    conc = sorted(report_dir.glob("concordance_*.csv"))
    if conc:
        lines += ["### Воспроизведение различий между типами", "",
                  "Для каждой метрики реальное различие типов (нейрон — единица, перестановки) сравнивается со сгенерированным: "
                  "«воспроизведено» — значимо, тот же знак и величина в пределах интервала; «согласуется» — тот же знак без значимости; "
                  "«неверный знак»; «не воспроизведено»; «лишний эффект» — у сгенерированных значимо при отсутствии реального.", "",
                  "| набор | реальных значимых различий | воспроизведено | согласуется | неверный знак | не воспроизведено | лишний эффект |",
                  "| --- | --- | --- | --- | --- | --- | --- |"]
        for f in conc:
            t = pd.read_csv(f)
            st = t["status"].astype(str)
            real_sig = int(t["real_sig"].astype(str).str.lower().eq("true").sum())
            def n(name):
                return int(st.eq(name).sum())
            lines.append(f"| {f.stem[len('concordance_'):]} | {real_sig} | {n('reproduced')} | {n('consistent')} | {n('wrong_sign')} | {n('not_reproduced')} | {n('spurious_effect')} |")
        lines.append("")
    return lines


# ------------------------------------------------------------------------------------------------------------ T, P


def _tp_frames(tp_dir: Path, evaluation: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    req = pd.read_parquet(tp_dir / "branches.parquet").sort_values("branch_index").reset_index(drop=True)
    ev = pd.read_csv(evaluation / "evaluation_branches.csv")
    gen = ev[ev["kind"] == "generated"].reset_index(drop=True)
    real = ev[ev["kind"] == "real"].reset_index(drop=True)
    if not (len(gen) == len(real) == len(req)):
        raise SystemExit(f"cannot pair {evaluation}: {len(req)} requests, {len(gen)} generated, {len(real)} real rows")
    groups = ["species", "cell_type_binary", "cell_type", "compartment"]
    meta = req[[c for c in groups if c in req.columns]].copy()
    meta["neuron"] = req["source_branch"].astype(str).map(_neuron)
    gen, real = pd.concat([gen, meta], axis=1), pd.concat([real, meta], axis=1)
    sp = pd.read_csv(evaluation / "evaluation_spines.csv") if (evaluation / "evaluation_spines.csv").exists() else pd.DataFrame()
    if len(sp):
        sp = sp.join(meta, on="branch", rsuffix="_m")
        if "neuron_m" in sp.columns:
            sp["neuron"] = sp["neuron_m"]
    gs = sp[sp["kind"] == "generated"] if len(sp) else sp
    rs = sp[sp["kind"] == "real"] if len(sp) else sp
    return gen, real, gs, rs


def _diff_rows(gen: pd.DataFrame, real: pd.DataFrame, columns: Dict[str, str], n_boot: int, seed: int) -> List[Dict]:
    from .significance import cluster_bootstrap_diff

    rows = []
    for c, label in columns.items():
        if c not in gen.columns or c not in real.columns:
            continue
        a, b = gen[c].to_numpy(float), real[c].to_numpy(float)
        if c == "no_spines":  # a share: compare the means
            ci = cluster_bootstrap_diff(a, gen["neuron"], b, real["neuron"], n_boot=n_boot, seed=seed, statistic=np.mean)
            ma, mb = np.nanmean(a), np.nanmean(b)
        else:
            ci = cluster_bootstrap_diff(a, gen["neuron"], b, real["neuron"], n_boot=n_boot, seed=seed)
            ma, mb = np.nanmedian(a), np.nanmedian(b)
        rows.append({"metric": c, "label": label, "n_gen": int(np.isfinite(a).sum()), "n_real": int(np.isfinite(b).sum()),
                     "gen": ma, "real": mb, "diff": ci["diff"], "lo": ci["lo"], "hi": ci["hi"], "p": ci["p"], "w1_sd": w1_sd(a, b)})
    return rows


def tp_report(tp_dir: Path, out: Path, *, label: str = "", evaluation: Optional[Path] = None, group_by: Sequence[str] = ("cell_type_binary", "compartment", "species"),
              min_branches: int = 30, n_boot: int = 1000, seed: int = 0) -> List[str]:
    """T functionals, P counts (branch level) and P spine statistics, overall and per group, with figures."""
    tp_dir, out = Path(tp_dir), Path(out)
    if evaluation is None:
        cands = [d for d in ("evaluation_paired", "evaluation") if (tp_dir / d / "evaluation_branches.csv").exists()]
        if not cands:
            raise SystemExit(f"no evaluation folder in {tp_dir}")
        evaluation = tp_dir / cands[0]
    gen, real, gs, rs = _tp_frames(tp_dir, Path(evaluation))
    tag = label or tp_dir.name
    lines = [f"## T и P: `{tag}` (оценка `{Path(evaluation).name}`)", "",
             f"{len(gen)} пар «сгенерированная ветвь — её реальный прототип», {real['neuron'].nunique()} нейронов. Разность медиан "
             "(для доли ветвей без шипиков — средних), интервал 95 % — бутстрэп нейронов (сгенерированная ветвь принадлежит нейрону "
             "прототипа), * — значимо после поправки Бенджамини — Хохберга внутри таблицы; W₁/σ — в единицах sd реальных.", ""]
    all_rows = []
    def table(rows, title):
        q = _bh(np.array([r["p"] for r in rows], float))
        out_lines = [f"#### {title}", "", "| величина | n ген. / реал. | сгенерированные | реальные | разность [95 %] | W₁/σ |", "| --- | --- | --- | --- | --- | --- |"]
        for r, qq in zip(rows, q):
            star = "*" if np.isfinite(qq) and qq < 0.05 else ""
            out_lines.append(f"| {r['label']} | {r['n_gen']} / {r['n_real']} | {r['gen']:.4g} | {r['real']:.4g} | {r['diff']:+.3g}{star} [{r['lo']:+.3g}; {r['hi']:+.3g}] | {r['w1_sd']:.2f} |")
            all_rows.append({**r, "q": qq, "table": title})
        return out_lines + [""]
    lines += table(_diff_rows(gen, real, {"length_nm": "длина, нм", **T_FUNCTIONALS}, n_boot, seed), "T: функционалы ствола, все ветви")
    lines += table(_diff_rows(gen, real, P_BRANCH, n_boot, seed), "P: число шипиков, все ветви")
    if len(gs):
        lines += table(_diff_rows(gs, rs, P_SPINE, n_boot, seed), "P: статистики по шипикам, все ветви")
    thin_g, thin_r = (gen["mean_radius_nm"] < 200).mean(), (real["mean_radius_nm"] < 200).mean()
    lines += [f"Доля тонких ветвей (средний радиус < 200 нм): сгенерированные {thin_g:.3f}, реальные {thin_r:.3f}.", ""]
    distribution_grid(real, {"сгенерированные": gen}, None, T_FUNCTIONALS, out / f"t_{tag}.png", "hist",
                      ["mean_radius_nm", "kappa_rms_per_nm", "radius_ratio_prox_dist"], f"Ствол: {tag}", ncols=3)
    distribution_grid(real, {"сгенерированные": gen}, None, {"n_spines": P_BRANCH["n_spines"], "density_per_um": P_BRANCH["density_per_um"]},
                      out / f"p_branch_{tag}.png", "ecdf", [], f"Шипики на ветви: {tag}", ncols=2)
    if len(gs):
        distribution_grid(rs, {"сгенерированные": gs}, None, P_SPINE, out / f"p_spines_{tag}.png", "hist", [], f"Расположение шипиков: {tag}", ncols=4)
    lines += [f"Рисунки: `t_{tag}.png`, `p_branch_{tag}.png`" + (f", `p_spines_{tag}.png`" if len(gs) else "") + ".", ""]
    for col in group_by:
        if col not in gen.columns or gen[col].nunique(dropna=True) < 2:
            continue
        lines += [f"### По группам: {col}", ""]
        for value, gg in gen.groupby(col):
            rr = real.loc[gg.index]
            if len(gg) < min_branches:
                lines += [f"- {value}: {len(gg)} ветвей < {min_branches}, не оценивается", ""]
                continue
            title = f"{col} = {value} ({len(gg)} ветвей, {rr['neuron'].nunique()} нейронов)"
            rows = _diff_rows(gg, rr, {**T_FUNCTIONALS, **P_BRANCH}, n_boot, seed)
            if len(gs):
                idx = set(gg.index)
                rows += _diff_rows(gs[gs["branch"].isin(idx)], rs[rs["branch"].isin(idx)], {"nn_surface_nm": P_SPINE["nn_surface_nm"], "gap_nm": P_SPINE["gap_nm"]}, n_boot, seed)
            lines += table(rows, title)
            safe = str(value).replace("/", "-")
            distribution_grid(rr, {"сгенерированные": gg}, None, {**T_FUNCTIONALS, "n_spines": P_BRANCH["n_spines"], "density_per_um": P_BRANCH["density_per_um"]},
                              out / f"tp_{tag}_{col}-{safe}.png", "hist", ["mean_radius_nm", "kappa_rms_per_nm"], f"{tag}: {title}", ncols=4)
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(all_rows).to_csv(out / f"tp_{tag}.csv", index=False)
    return lines
