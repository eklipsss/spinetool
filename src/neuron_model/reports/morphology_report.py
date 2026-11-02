"""HTML report of ``scripts/neuron_model/morphology_by_condition.py``: where the spine morphology differs significantly
between cell types and compartments (style: docs/neuron-model/html-materials-style-guide.md, kit: html_kit.py).

Reads the output folder of the analysis (the test tables + the neuron-level medians) and writes ``report.html`` next to
them. Page logic: ``assets/morphology_report.js``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from ..spine_generation.atlas import METRIC_INFO
from ..spine_morphometrics import SCALAR_METRICS
from .html_kit import card, clean_floats, esc, header, note, page, section, tiles

COMPARTMENT_RU = {"apical": "апикальный", "basal": "базальный", "uncertain": "неопределённый", "axon_like": "аксоноподобный"}
BINARY_RU = {"inhibitory": "тормозные", "excitatory": "возбуждающие"}


def _num(v: Any) -> Optional[float]:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if np.isfinite(f) else None


def _read(folder: Path, name: str) -> pd.DataFrame:
    path = folder / f"{name}.csv"
    return pd.read_csv(path) if path.exists() and path.stat().st_size > 1 else pd.DataFrame()


def _cell(row: Optional[pd.Series], rel_col: str, factor: float, absolute: bool = False) -> Dict[str, Any]:
    """``absolute``: the difference in the metric's own units instead of the relative one (shares, g(r), correlations)."""
    if row is None:
        return {"rel": None}
    sig = bool(row.get("significant")) if pd.notna(row.get("significant")) else False
    col = rel_col.replace("rel_diff", "abs_diff") if absolute else rel_col
    value = _num(row.get(col))
    return {
        "rel": None if value is None else (value * factor if absolute else value), "abs": absolute, "q": _num(row.get("q_bh")), "p": _num(row.get("p")), "sig": sig,
        "ma": None if _num(row.get("median_a")) is None else float(row["median_a"]) * factor,
        "mb": None if _num(row.get("median_b")) is None else float(row["median_b"]) * factor,
        "na": _num(row.get("n_neurons_a", row.get("n_neurons_paired"))), "nb": _num(row.get("n_neurons_b", row.get("n_neurons_paired"))),
        "delta": _num(row.get("cliffs_delta")), "share": _num(row.get("share_neurons_a_gt_b")),
    }


PROFILES: Dict[str, Dict[str, Any]] = {
    "spines": {
        "title": "Морфология шипиков по типу клетки и компартменту",
        "subtitle": "Где реальные шипики Minnie65 статистически значимо различаются по каждой из 11 морфометрик",
        "unit_word": "шипиков", "value_word": "медиана метрики по его шипикам",
        "intro": "Страница показывает, отличаются ли реальные шипики Minnie65 по каждой морфометрике между тормозными и возбуждающими клетками, между подробными типами клеток и между компартментами. Это нужно, чтобы решить, какие условия имеет смысл подавать генератору шипиков и насколько сильным должен быть их эффект в сгенерированных формах.",
        "neuron_value": "Для каждого нейрона берётся медиана метрики по его шипикам (не меньше 20)",
    },
    "branches": {
        "title": "Ствол и расположение шипиков по типу клетки и компартменту",
        "subtitle": "Где реальные ветви значимо различаются по метрикам модулей T (ствол) и P (число и расположение шипиков)",
        "unit_word": "ветвей", "value_word": "значение метрики нейрона (медиана по ветвям или статистика по всем его шипикам)",
        "intro": "Страница показывает, отличаются ли реальные ветви между тормозными и возбуждающими клетками, между подробными типами клеток и между компартментами по метрикам ствола (модуль T: радиус, сужение, сплюснутость сечения, извилистость, изгиб, длина) и расположения шипиков (модуль P: плотность, доля пустых ветвей, парная корреляционная функция положений вдоль ветви, регулярность интервалов, сходство соседних шипиков по объёму и длине, связь размера шипика с положением). Это нужно, чтобы решить, какие условия подавать моделям T и P и проверять ли их эффект в сгенерированных ветвях.",
        "neuron_value": "Для каждого нейрона (или нейрона и компартмента) метрики ствола и простые метрики шипиков — медианы по его ветвям (не меньше 10 пригодных ветвей), а пространственные метрики считаются по всем шипикам его ветвей вместе: на одной ветви их слишком мало для устойчивой оценки",
    },
}


def _profile(info: Dict[str, Any]):
    """(metric info, metric order, texts) of the analysis that wrote the folder (run_info.json ``profile``)."""
    name = info.get("profile", "spines")
    if name == "branches":
        from ..branch_statistics import ALL_METRICS, METRIC_INFO as BRANCH_INFO

        return BRANCH_INFO, list(ALL_METRICS), PROFILES["branches"]
    return METRIC_INFO, list(SCALAR_METRICS), PROFILES["spines"]


def build_payload(folder: Path) -> Dict[str, Any]:
    folder = Path(folder)
    info0 = json.loads((folder / "run_info.json").read_text(encoding="utf-8")) if (folder / "run_info.json").exists() else {}
    METRIC_INFO_, ORDER, _texts = _profile(info0)
    binary = _read(folder, "cell_type_binary")
    within = _read(folder, "cell_type_binary_within_compartment")
    comp = _read(folder, "compartment")
    pairs = _read(folder, "cell_type")
    omni = _read(folder, "cell_type_omnibus")
    med = _read(folder, "neuron_medians")
    cmed = _read(folder, "neuron_compartment_medians")
    info = json.loads((folder / "run_info.json").read_text(encoding="utf-8")) if (folder / "run_info.json").exists() else {}

    tested = set(binary["metric"]) if "metric" in binary.columns else set(ORDER)
    metrics = [m for m in ORDER if m in METRIC_INFO_ and m in tested and (med.empty or m in med.columns)]
    metric_meta = [{"id": m, "abs": METRIC_INFO_[m].get("diff") == "abs", **{k: METRIC_INFO_[m][k] for k in ("name", "unit", "factor", "about")}} for m in metrics]

    # ---- overview: binary (all, within compartments), compartment pairs
    columns: List[Dict[str, Any]] = []
    cells: Dict[str, List[Dict[str, Any]]] = {}

    def add_column(key, label, kind, rows: pd.DataFrame, rel_col):
        if rows.empty:
            return
        by_metric = {r["metric"]: r for _, r in rows.iterrows()}
        col = [_cell(by_metric.get(m), rel_col, METRIC_INFO_[m]["factor"], METRIC_INFO_[m].get("diff") == "abs") for m in metrics]
        if all(c["rel"] is None for c in col):
            return
        columns.append({"key": key, "label": label, "kind": kind})
        cells[key] = col

    if not binary.empty:
        add_column("binary_all", "Тормозные / возбуждающие<br>все " + _texts["unit_word"], "binary", binary, "rel_diff_a_vs_b")
    if not within.empty:
        for c in ("basal", "uncertain", "apical"):
            add_column(f"binary_{c}", f"Тормозные / возбуждающие<br>только {COMPARTMENT_RU[c]}", "binary", within[within["compartment"] == c], "rel_diff_a_vs_b")
    if not comp.empty:
        for (a, b), rows in comp.groupby(["a", "b"], sort=False):
            add_column(f"comp_{a}_{b}", f"{COMPARTMENT_RU.get(a, a).capitalize()} / {COMPARTMENT_RU.get(b, b)}<br>внутри нейрона", "compartment", rows, "median_paired_rel_diff")

    # ---- detailed types: matrices per metric (row type vs column type)
    types: List[str] = []
    type_info: List[Dict[str, Any]] = []
    matrices: Dict[str, Dict[str, Any]] = {}
    if not pairs.empty:
        counts = med.groupby("cell_type").size() if not med.empty else pd.Series(dtype=int)
        binary_of = med.groupby("cell_type")["cell_type_binary"].first() if not med.empty else pd.Series(dtype=object)
        names = sorted(set(pairs["a"]) | set(pairs["b"]), key=lambda t: (binary_of.get(t, "") != "excitatory", -int(counts.get(t, 0)), t))
        types = names
        type_info = [{"type": t, "binary": binary_of.get(t), "n": int(counts.get(t, 0))} for t in names]
        index = {t: i for i, t in enumerate(names)}
        for m in metrics:
            rel = [[None] * len(names) for _ in names]
            q = [[None] * len(names) for _ in names]
            sig = [[False] * len(names) for _ in names]
            for _, r in pairs[pairs["metric"] == m].iterrows():
                i, j = index[r["a"]], index[r["b"]]
                if METRIC_INFO_[m].get("diff") == "abs":
                    v = _num(r.get("abs_diff_a_vs_b"))
                    rel[i][j] = v
                    rel[j][i] = None if v is None else -v
                else:
                    v = _num(r.get("rel_diff_a_vs_b"))
                    rel[i][j] = v
                    rel[j][i] = None if v is None or v <= -1 else 1.0 / (1.0 + v) - 1.0
                q[i][j] = q[j][i] = _num(r.get("q_bh"))
                sig[i][j] = sig[j][i] = bool(r.get("significant")) if pd.notna(r.get("significant")) else False
            matrices[m] = {"rel": rel, "q": q, "sig": sig}
    omnibus = {r["metric"]: {"q": _num(r.get("q_bh")), "sig": bool(r.get("significant"))} for _, r in omni.iterrows()} if not omni.empty else {}

    # ---- neuron-level values for the distributions (display units)
    dist: Dict[str, Any] = {"by_type": {}, "by_binary": {}, "paired": {}}
    if not med.empty:
        for m in metrics:
            f = METRIC_INFO_[m]["factor"]
            dist["by_type"][m] = {t: clean_floats(med.loc[med["cell_type"] == t, m] * f) for t in types}
            dist["by_binary"][m] = {"all": {b: clean_floats(med.loc[med["cell_type_binary"] == b, m] * f) for b in BINARY_RU}}
    if not cmed.empty:
        present = [c for c in ("basal", "uncertain", "apical") if (cmed["compartment"] == c).any()]
        for m in metrics:
            f = METRIC_INFO_[m]["factor"]
            for c in present:
                sub = cmed[cmed["compartment"] == c]
                dist["by_binary"].setdefault(m, {})[c] = {b: clean_floats(sub.loc[sub["cell_type_binary"] == b, m] * f) for b in BINARY_RU}
        wide = cmed.pivot(index="neuron_id", columns="compartment", values=metrics)
        for a, b in [("apical", "basal"), ("apical", "uncertain"), ("basal", "uncertain")]:
            key = f"{a}_{b}"
            dist["paired"][key] = {}
            for m in metrics:
                if (m, a) in wide.columns and (m, b) in wide.columns:
                    pair = wide[[(m, a), (m, b)]].dropna()
                    dist["paired"][key][m] = clean_floats(100.0 * (pair[(m, a)] / pair[(m, b)] - 1.0))

    counts_binary = med.drop_duplicates("neuron_id")["cell_type_binary"].value_counts().to_dict() if not med.empty else {}
    # neurons of each binary type that have at least one limb with a known compartment (inhibitory: none in Minnie65)
    with_comp = cmed.drop_duplicates("neuron_id")["cell_type_binary"].value_counts().to_dict() if not cmed.empty else {}
    n_sig = {name: (int(t["significant"].sum()) if "significant" in t.columns else 0, int(t["p"].notna().sum()) if "p" in t.columns else 0)
             for name, t in (("binary", binary), ("within", within), ("compartment", comp), ("types", pairs), ("omnibus", omni))}
    return {"metrics": metric_meta, "columns": columns, "cells": cells, "types": type_info, "matrices": matrices, "omnibus": omnibus,
            "dist": dist, "metric_info": METRIC_INFO_, "compartment_ru": COMPARTMENT_RU, "binary_ru": BINARY_RU, "info": info, "counts_binary": counts_binary, "with_compartment": with_comp, "n_sig": n_sig}


def _table_html(df: pd.DataFrame, columns: List[tuple], factor_col: Optional[str] = None, metric_info: Optional[Dict[str, Any]] = None) -> str:
    """columns: (source column, header, kind) with kind in text / int / pct / sci / num / median."""
    if df.empty:
        return '<p class="muted small">Нет проверяемых сравнений.</p>'
    head = "".join(f"<th>{esc(h)}</th>" for _, h, _ in columns) + "<th>Итог</th>"
    rows = []
    for _, r in df.iterrows():
        tds = []
        for col, _, kind in columns:
            v = r.get(col)
            if kind == "text":
                txt = {"metric": (metric_info or METRIC_INFO).get(v, {}).get("name", v)}.get(col, COMPARTMENT_RU.get(v, BINARY_RU.get(v, v))) if isinstance(v, str) else ""
            elif _num(v) is None:
                txt = "—"
            elif kind == "int":
                txt = f"{int(float(v))}"
            elif kind == "pct":
                txt = f"{float(v) * 100:+.1f} %"
            elif kind == "sci":
                txt = f"{float(v):.2g}"
            elif kind == "median":
                info = (metric_info or METRIC_INFO).get(r.get("metric"), {})
                txt = f"{float(v) * info.get('factor', 1.0):.3g}" + (f" {info.get('unit')}" if info.get("unit") else "")
            else:
                txt = f"{float(v):.2f}"
            tds.append(f"<td>{esc(txt)}</td>")
        sig = bool(r.get("significant")) if pd.notna(r.get("significant")) else False
        badge = '<span class="pill good">значимо</span>' if sig else ('<span class="pill">нет</span>' if _num(r.get("p")) is not None else '<span class="pill">мало данных</span>')
        rows.append("<tr>" + "".join(tds) + f"<td>{badge}</td></tr>")
    return f'<div style="overflow-x:auto"><table class="metrics"><thead><tr>{head}</tr></thead><tbody>{"".join(rows)}</tbody></table></div>'


def build_report(folder: Path, out: Optional[Path] = None) -> Path:
    folder = Path(folder)
    payload = build_payload(folder)
    info = payload["info"]
    n_sig = payload["n_sig"]
    cb = payload["counts_binary"]

    texts = _profile(info)[2]
    body = [header(texts["title"], texts["subtitle"],
                   [f"нейронов {info.get('n_neurons', '—')}, {texts['unit_word']} {info.get('n_spines', info.get('n_branches', '—'))}", f"перестановок {info.get('n_permutations', '—')}",
                    f"создано {info.get('created', '—')}"])]
    body.append(tiles([
        {"label": "Нейронов: тормозных / возбуждающих", "value": f"{cb.get('inhibitory', 0)} / {cb.get('excitatory', 0)}", "note": f"не меньше {info.get('min_spines', 20)} {texts['unit_word']} на нейрон"},
        {"label": "Тормозные против возбуждающих", "value": f"{n_sig['binary'][0]} из {n_sig['binary'][1]}", "note": "метрик различаются значимо", "accent": "pink"},
        {"label": "Подробные типы, общий тест", "value": f"{n_sig['omnibus'][0]} из {n_sig['omnibus'][1]}", "note": "метрик различаются между типами"},
        {"label": "Пары подробных типов", "value": f"{n_sig['types'][0]} из {n_sig['types'][1]}", "note": "сравнений значимы"},
        {"label": "Компартменты внутри нейрона", "value": f"{n_sig['compartment'][0]} из {n_sig['compartment'][1]}", "note": "сравнений значимы", "accent": "green"},
    ]))
    body.append(note([
        texts["intro"],
        "**Единица сравнения — нейрон.** Шипики и ветви одного нейрона похожи друг на друга сильнее, чем у разных нейронов (общий «стиль» сегментации, размера клетки, условий), поэтому их нельзя считать независимыми: тогда значимым оказалось бы почти любое различие. " + texts["neuron_value"] + ", и группы нейронов сравниваются перестановочным тестом: метки типа клетки случайно переставляются между нейронами, и наблюдаемая разность медиан сравнивается с разностями при перестановках.",
        "**Компартменты сравниваются внутри нейрона.** Компартмент — свойство лимба, и у одного нейрона часто есть апикальные и базальные лимбы. Для каждого такого нейрона вычисляется разность медиан двух компартментов, а нулевое распределение получается случайной сменой знаков этих разностей. Так стиль нейрона сокращается, и остаётся только эффект компартмента.",
        "**Множественные сравнения.** Внутри каждой таблицы (метрики × пары) p-значения поправлены методом Бенджамини–Хохберга; значимым считается сравнение с q < 0,05. Значимость говорит о том, что различие не случайно, а не о том, что оно велико: при сотнях нейронов значимой может быть и разница в 1–2 %. Поэтому рядом всегда показан размер эффекта — относительная разность медиан.",
    ], title="О чём эта страница"))

    wc = payload["with_compartment"]
    if cb.get("inhibitory") and not wc.get("inhibitory"):
        body.append(note([
            f"**У тормозных нейронов нет ни одного лимба с известным компартментом** (в таблице компартментов есть лимбы {wc.get('excitatory', 0)} "
            f"из {cb.get('excitatory', 0)} возбуждающих нейронов и ни одного из {cb.get('inhibitory', 0)} тормозных). Поэтому сравнение типов клеток внутри "
            "компартмента невозможно, и нельзя проверить, не объясняется ли различие тормозных и возбуждающих клеток составом компартментов. "
            "Сравнение компартментов относится только к возбуждающим нейронам."], title="Ограничение данных", tone="pink"))
    body.append(section("Сводная карта", "Строки — морфометрики, столбцы — сравнения. Цвет — относительная разность медиан (первая группа к второй), серый — различие не значимо.",
                        card("Относительная разность медиан, %", '<div id="overview" class="plot" style="height:520px"></div>', hint="наведите на клетку: медианы, число нейронов, q")
                        + note(["Цветная клетка означает значимое различие (q < 0,05), её цвет — знак и величину эффекта: розовый — у первой группы значение больше, зелёный — меньше, насыщенность растёт с величиной разности (шкала обрезана на ±50 %). Серая клетка — различие не значимо, число в ней показывает наблюдаемую разность, которую нельзя отличить от случайной. Пустая клетка — сравнение невозможно (например, у тормозных клеток нет апикальных дендритов).",
                                "Столбцы «только базальный / неопределённый» повторяют сравнение типов клеток внутри одного компартмента. Это нужно, потому что у тормозных клеток нет апикальных дендритов: если бы различие типов было на самом деле различием компартментов, внутри одного компартмента оно бы исчезло. Столбцы компартментов показывают медиану относительных разностей внутри нейронов."],
                               title="Как читать сводную карту", collapsible=True), anchor="sec-overview"))

    body.append(section("Тормозные и возбуждающие клетки", "Распределение медиан нейронов по выбранной морфометрике: все шипики и внутри компартментов.",
                        card("Медианы нейронов", '<div style="max-width:320px;margin-bottom:8px"><select id="sel-binary"></select></div><div id="plot-binary" class="plot" style="height:420px"></div>')
                        + note(["Каждая точка ящика — один нейрон (" + texts["value_word"] + "). Ящик — межквартильный размах, линия — медиана нейронов. Розовый — тормозные клетки, зелёный — возбуждающие. Если ящики сдвинуты и при этом клетка сводной карты цветная, сдвиг больше, чем различие случайных групп нейронов."], title="Как читать", collapsible=True),
                        anchor="sec-binary"))

    body.append(section("Подробные типы клеток", "Типы с достаточным числом нейронов; матрица попарных сравнений и распределения медиан нейронов по выбранной морфометрике.",
                        '<div style="max-width:320px;margin-bottom:12px"><select id="sel-types"></select></div><div class="grid-2">'
                        + card("Попарные сравнения: строка / столбец, %", '<div id="plot-type-matrix" class="plot" style="height:480px"></div>', hint="серым — не значимо")
                        + card("Медианы нейронов по типам", '<div id="plot-type-box" class="plot" style="height:480px"></div>') + "</div>"
                        + '<p class="small muted" id="omnibus-line"></p>'
                        + note(["Клетка матрицы — относительная разность медиан типа строки к типу столбца (у пары она записана в обе стороны, поэтому матрица антисимметрична по знаку). Цвет — только у значимых пар (q < 0,05 после поправки по всем парам типов и всем метрикам вместе), серые клетки не значимы. Общий тест (строка под матрицей) проверяет, различаются ли типы вообще, по рангам медиан нейронов.",
                                "Типы упорядочены: сначала возбуждающие, затем тормозные, внутри — по числу нейронов. Типы, в которых меньше 10 нейронов, и типы «Unsure» в сравнение не входят. У малых типов (10–15 нейронов) интервал неопределённости широк, поэтому незначимость там не означает отсутствия различия."],
                               title="Как читать", collapsible=True),
                        anchor="sec-types"))

    body.append(section("Компартменты внутри нейрона", "Распределение относительных разностей медиан двух компартментов у одного и того же нейрона.",
                        card("Разность внутри нейрона, %", '<div style="max-width:320px;margin-bottom:8px"><select id="sel-comp"></select></div><div id="plot-comp" class="plot" style="height:400px"></div>')
                        + note(["Каждая точка — один нейрон, у которого есть оба компартмента: насколько медиана метрики в первом компартменте больше (положительно) или меньше (отрицательно), чем во втором. Если ящик стоит на нуле, у нейронов нет систематической разницы между компартментами, даже если отдельные нейроны различаются в обе стороны."],
                               title="Как читать", collapsible=True),
                        anchor="sec-comp"))

    folder_tables = []
    for name, title, cols in [
        ("cell_type_binary", "Тормозные / возбуждающие", [("metric", "Метрика", "text"), ("n_neurons_a", "Нейронов (торм.)", "int"), ("n_neurons_b", "Нейронов (возб.)", "int"),
                                                          ("median_a", "Медиана (торм.)", "median"), ("median_b", "Медиана (возб.)", "median"), ("rel_diff_a_vs_b", "Разность", "pct"),
                                                          ("cliffs_delta", "Дельта Клиффа", "num"), ("p", "p", "sci"), ("q_bh", "q", "sci")]),
        ("cell_type_binary_within_compartment", "Тормозные / возбуждающие внутри компартмента", [("compartment", "Компартмент", "text"), ("metric", "Метрика", "text"), ("n_neurons_a", "Нейронов (торм.)", "int"),
                                                                                              ("n_neurons_b", "Нейронов (возб.)", "int"), ("rel_diff_a_vs_b", "Разность", "pct"), ("p", "p", "sci"), ("q_bh", "q", "sci")]),
        ("compartment", "Компартменты внутри нейрона", [("a", "Первый", "text"), ("b", "Второй", "text"), ("metric", "Метрика", "text"), ("n_neurons_paired", "Нейронов", "int"),
                                                        ("median_paired_rel_diff", "Разность (медиана по нейронам)", "pct"), ("share_neurons_a_gt_b", "Доля нейронов «первый больше»", "num"), ("p", "p", "sci"), ("q_bh", "q", "sci")]),
        ("cell_type_omnibus", "Подробные типы: общий тест", [("metric", "Метрика", "text"), ("n_types", "Типов", "int"), ("n_neurons", "Нейронов", "int"), ("p", "p", "sci"), ("q_bh", "q", "sci")]),
    ]:
        df = _read(folder, name)
        folder_tables.append(f'<details class="note"><summary>{esc(title)}</summary>{_table_html(df, cols, metric_info=payload["metric_info"])}</details>')
    body.append(section("Таблицы", "Все сравнения с числами; попарные сравнения подробных типов — в файле cell_type.csv рядом со страницей.", "".join(folder_tables), anchor="sec-tables"))
    body.append(f'<footer class="page">Данные: таблица морфометрик этапа 8 предобработки S ({esc(info.get("morphometrics_table", "—"))}); типы клеток — '
                f'cell_types_with_nucleus.csv, компартменты — minnie65_limb_compartments_length_rule.parquet. Анализ: scripts/neuron_model/morphology_by_condition.py, '
                f'страница: src/neuron_model/reports/morphology_report.py. Значения показаны в мкм (пересчёт из нм).</footer>')

    js = (Path(__file__).parent / "assets" / "morphology_report.js").read_text(encoding="utf-8")
    out = Path(out) if out else folder / "report.html"
    out.write_text(page("Морфология по условиям", "".join(body), payload, js), encoding="utf-8")
    return out
