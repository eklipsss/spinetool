"""Global report: composition of the datasets, distributions of every metric, which metrics differ between dendrite types
(and which do NOT), with the confounding stated. Style: docs/neuron-model/html-materials-style-guide.md; kit: html_kit.py.

Input = the folder written by ``scripts/neuron_model/build_global_tables.py`` (``<table>.json`` records, ``run_info.json``).
Output = ``<date>_global-report.html`` next to it. Page logic: ``assets/global_report.js``. Design and statistics:
docs/neuron-model/global-report-design.md. Sections 0-3 of the design (generated dendrites = section 4 are not built yet).
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from .html_kit import card, esc, header, note, page, section, tiles

DATASET_RU = {"minnie65": "Minnie65 (мышь, ЭМ)", "h01": "H01 (человек, ЭМ)", "labid": "LabID (мышь, конфокал)"}
SHORT = {"minnie65": "Minnie65", "h01": "H01", "labid": "LabID"}


def scope_name(ds: str) -> str:
    if ds in DATASET_RU:
        return DATASET_RU[ds]
    return " и ".join(SHORT.get(x, x) for x in ds.split(" vs "))


def reason_ru(reason: Any) -> str:
    """Russian text of the engine's reason (factors.py)."""
    import re

    if not isinstance(reason, str):
        return ""
    m = re.match(r"factor not defined for (\w+)", reason)
    if m:
        return f"фактор не определён для {SHORT.get(m.group(1), m.group(1))}"
    m = re.match(r"too few units \(a=(\d+), b=(\d+), need (\d+)\)", reason)
    if m:
        return f"мало единиц ({m.group(1)} и {m.group(2)}, нужно не меньше {m.group(3)})"
    m = re.match(r"level has (\d+) units < (\d+)", reason)
    if m:
        return f"в типе {m.group(1)} единиц, нужно не меньше {m.group(2)}"
    m = re.match(r"too few units with both levels \((\d+), need (\d+)\)", reason)
    if m:
        return f"мало нейронов с обоими уровнями ({m.group(1)}, нужно не меньше {m.group(2)})"
    return {"metadata missing in this dataset (no values)": "метаданные в этом датасете отсутствуют",
            "level absent in this dataset / stratum": "один из уровней отсутствует",
            "both levels absent in this dataset / stratum": "оба уровня отсутствуют",
            "level absent in this dataset": "один из уровней отсутствует", "factor has no values here": "у фактора нет значений",
            "needs both EM datasets": "нужны оба датасета электронной микроскопии", "stratifying metadata missing": "нет метаданных для страт",
            "no stratum present in both datasets": "нет страты, общей для обоих датасетов"}.get(reason, reason)


WARN_RU = {
    "inhibitory cells have no apical dendrite: a pooled contrast can be a compartment contrast (see cell_type_binary | compartment)":
        "У тормозных клеток нет апикального дендрита: общее различие может быть различием компартментов (см. сравнение внутри компартмента).",
    "compartment of H01 comes from the direction rule (pia axis estimated from layers), not a ground truth":
        "Компартмент H01 получен по правилу направления к пиа (ось оценена по слоям), это оценка, а не разметка.",
    "unit = branch folder: animals / slices are unknown, folders of one animal are not independent (pseudo-replication possible)":
        "Единица — папка ветви: животные и срезы неизвестны, папки одного животного могут быть зависимыми (возможна псевдорепликация).",
    "treatment is confounded with the imaging / z-correction series until the laboratory confirms":
        "Лечение смешано с серией съёмки и поправкой по z, пока лаборатория не подтвердит обратное.",
    "species is perfectly confounded with the dataset: descriptive contrast only, NOT a test of species":
        "Вид полностью совпадает с датасетом: это описательное сравнение, а не проверка влияния вида.",
    "same cell class (e.g. pyramidal) in mouse EM vs human EM: still species = dataset, descriptive only":
        "Один и тот же класс клеток (например, пирамидные) у мыши и человека: вид по-прежнему совпадает с датасетом, сравнение описательное.",
    "same layer in mouse and human, Minnie65 layer is coarse (from the cell-type name): descriptive only":
        "Один и тот же слой у мыши и человека; слой Minnie65 грубый (из названия типа клетки): сравнение описательное.",
    "dataset contrast = species + modality + processing: descriptive only":
        "Различие датасетов = вид + метод съёмки + обработка: сравнение описательное.",
    "modality differs (EM vs confocal): size metrics depend on resolution":
        "Методы съёмки разные (электронная и конфокальная микроскопия): размеры зависят от разрешения.",
    "harmonised class: pyramidal / other_excitatory / interneuron, derived from the dataset's own cell-type names":
        "Унифицированный класс (пирамидные, другие возбуждающие, интернейроны) получен из названий типов в самих датасетах.",
    "Minnie65 layer is COARSE and derived from the cell-type NAME (excitatory cells only): it is a function of the type, not an independent measurement; H01 layers L2 and L3 are merged to L2/3":
        "Слой Minnie65 грубый и получен из НАЗВАНИЯ типа клетки (только возбуждающие): это функция типа, а не независимое измерение; слои L2 и L3 в H01 объединены в L2/3.",
    "H01 layer from h01_layer.csv, one value per neuron": "Слой H01 из h01_layer.csv, одно значение на нейрон.",
}
CF_RU = {"dataset": "датасет", "brain region and layer sampling": "область коры и выборка слоёв", "segmentation pipeline": "пайплайн сегментации",
         "age / donor": "возраст и донор", "brain region": "область коры", "coarse Minnie65 layer from the type name": "грубый слой Minnie65 из названия типа",
         "species": "вид", "imaging modality (EM vs confocal)": "метод съёмки (электронная и конфокальная микроскопия)",
         "resolution and z-stretch": "разрешение и растяжение по z", "segmentation": "сегментация",
         "z-rescaling pipeline (the 9009 groups are not rescaled by 0.7)": "поправка по z (группы 9009 не масштабированы на 0,7)"}


def warn_ru(text: Any) -> str:
    return " | ".join(WARN_RU.get(w, w) for w in str(text or "").split(" | ") if w)


def cf_ru(text: Any) -> str:
    if not text:
        return ""
    return json.dumps([CF_RU.get(x, x) for x in json.loads(text)], ensure_ascii=False)


FACTOR_RU = {
    "cell_type_binary": "Тормозные / возбуждающие",
    "cell_type_binary | compartment": "Тормозные / возбуждающие внутри компартмента",
    "cell_type": "Подробный тип клетки",
    "cell_class": "Класс клетки (унифицированный)",
    "layer": "Слой (унифицированный)",
    "layer_native": "Слой H01 (исходный)",
    "compartment": "Компартмент (внутри нейрона)",
    "health | treatment": "Здоровье (внутри групп лечения)",
    "treatment | health": "Лечение (внутри групп здоровья)",
    "species": "Вид: человек / мышь",
    "species | cell_class": "Вид внутри класса клетки",
    "species | layer": "Вид внутри слоя",
    "dataset": "Датасет против датасета",
}
LABEL_RU = {"significant": "значимо", "not_significant": "не значимо, мощности достаточно", "underpowered": "не значимо, мощности не хватает", "not_tested": "не проверялось"}
LEVEL_RU = {"spine": "Шипик (модуль S)", "branch": "Ветвь (модули T и P)"}


def _records(folder: Path, name: str) -> List[Dict[str, Any]]:
    path = folder / f"{name}.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() and path.stat().st_size > 2 else []


def _metric_info(level: str) -> Dict[str, Dict[str, Any]]:
    if level == "branch":
        from ..branch_statistics import METRIC_INFO
        return {k: {**v, "diff": v.get("diff")} for k, v in METRIC_INFO.items()}
    from ..spine_generation.atlas import METRIC_INFO
    return {k: {**v, "module": "S"} for k, v in METRIC_INFO.items()}


def _meta(metric: str, info: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    i = info.get(metric, {})
    return {"id": metric, "name": i.get("name", metric), "unit": i.get("unit", ""), "factor": i.get("factor", 1.0),
            "about": i.get("about", ""), "module": i.get("module", ""), "abs": i.get("diff") == "abs"}


def _num(v: Any) -> Optional[float]:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if np.isfinite(f) else None


def _scope(ds: str) -> str:
    return ds


def _level_payload(folder: Path, level: str) -> Optional[Dict[str, Any]]:
    results = _records(folder, f"results_{level}")
    comps = _records(folder, f"comparisons_{level}")
    desc = _records(folder, f"descriptive_{level}") + (_records(folder, "descriptive_spine_by_cell_type") if level == "spine" else [])
    if not (results or comps or desc):
        return None
    info = _metric_info(level)
    metrics = []
    for r in desc + results:
        if r["metric"] not in metrics:
            metrics.append(r["metric"])
    rows = []
    for r in results:
        na = r.get("n_neurons_a", r.get("n_neurons_paired"))
        nb = r.get("n_neurons_b", r.get("n_neurons_paired"))
        rows.append({"cid": r["comparison_id"], "ds": r["dataset"], "f": r["factor"], "st": r.get("stratum"), "a": r["a"], "b": r["b"], "m": r["metric"],
                     "eff": _num(r.get("effect")), "lo": _num(r.get("ci_low")), "hi": _num(r.get("ci_high")), "kind": r.get("effect_kind"),
                     "p": _num(r.get("p")), "q": _num(r.get("q_bh")), "lab": r.get("label"), "eq": bool(r.get("equivalent")),
                     "na": _num(na), "nb": _num(nb), "ma": _num(r.get("median_a")), "mb": _num(r.get("median_b")),
                     "mde": _num(r.get("mde")), "mg": _num(r.get("margin")), "causal": bool(r.get("causal")),
                     "w": warn_ru(r.get("warnings")), "cf": cf_ru(r.get("confounded_with")), "mode": r.get("mode")})
    comp_rows = [{"cid": c["comparison_id"], "ds": c["dataset"], "f": c["factor"], "a": c["a"], "b": c["b"], "st": c.get("stratum"), "valid": bool(c["valid"]),
                  "reason": reason_ru(c.get("reason")), "causal": bool(c["causal"]), "na": c.get("n_units_a"), "nb": c.get("n_units_b"), "np": c.get("n_units_paired")}
                 for c in comps]
    dist = [{k: (r.get(k) if k not in ("hist_edges", "hist_counts") else r.get(k)) for k in
             ("dataset", "cell_type_binary", "metric", "level", "n", "n_units", "mean", "sd", "q05", "q25", "q50", "q75", "q95", "icc", "hist_edges", "hist_counts")
             if k in r} for r in desc]
    return {"metrics": [_meta(m, info) for m in metrics], "results": rows, "comparisons": comp_rows, "dist": dist,
            "omnibus": [{k: r.get(k) for k in ("dataset", "factor", "metric", "n_types", "n_neurons", "p", "q_bh", "significant")} for r in _records(folder, f"omnibus_{level}")]}


def _pill(text: str, tone: str = "") -> str:
    return f'<span class="pill {tone}">{esc(text)}</span>'


def _matrix_html(comps: List[Dict[str, Any]]) -> str:
    """Factor x scope matrix: valid / descriptive only / cannot be tested (with the reason)."""
    scopes: List[str] = []
    factors: List[str] = []
    cell: Dict[Any, Dict[str, Any]] = {}
    for c in comps:
        if c["ds"] not in scopes:
            scopes.append(c["ds"])
        if c["f"] not in factors:
            factors.append(c["f"])
        key = (c["f"], c["ds"])
        state = "valid" if c["valid"] and c["causal"] else ("desc" if c["valid"] else "no")
        prev = cell.get(key)
        rank = {"valid": 0, "desc": 1, "no": 2}
        if prev is None or rank[state] < rank[prev["state"]]:
            cell[key] = {"state": state, "reason": c.get("reason"), "n": 1}
        elif state == prev["state"] and state != "no":
            prev["n"] += 1
    order = ["minnie65", "h01", "labid"]
    scopes.sort(key=lambda s: (" vs " in s, order.index(s) if s in order else 9, s))
    head = "".join(f"<th>{esc(scope_name(s))}</th>" for s in scopes)
    rows = []
    for f in factors:
        tds = []
        for s in scopes:
            c = cell.get((f, s))
            if c is None:
                tds.append('<td class="muted">—</td>')
            elif c["state"] == "valid":
                tds.append(f'<td>{_pill("можно проверять", "good")}</td>')
            elif c["state"] == "desc":
                tds.append(f'<td>{_pill("только описательно", "warn")}</td>')
            else:
                tds.append(f'<td title="{esc(reason_ru(c["reason"]))}">{_pill("нельзя", "bad")}<div class="small muted" style="margin-top:3px">{esc(reason_ru(c["reason"]))}</div></td>')
        rows.append(f"<tr><td>{esc(FACTOR_RU.get(f, f))}</td>{''.join(tds)}</tr>")
    return ('<style>table.vm td{text-align:left!important;vertical-align:top;min-width:150px}table.vm th{text-align:left!important}</style>'
            f'<div style="overflow-x:auto"><table class="metrics vm"><thead><tr><th>Фактор</th>{head}</tr></thead><tbody>{"".join(rows)}</tbody></table></div>')


def _summary_html(level_payloads: Dict[str, Dict[str, Any]]) -> str:
    """Counts of significant / not significant / underpowered tests per (level, module, scope, factor)."""
    rows = []
    for level, lp in level_payloads.items():
        mod = {m["id"]: m["module"] or ("S" if level == "spine" else "") for m in lp["metrics"]}
        agg: Dict[Any, Dict[str, int]] = {}
        for r in lp["results"]:
            key = (mod.get(r["m"], ""), r["ds"], r["f"], r["causal"])
            a = agg.setdefault(key, {"significant": 0, "not_significant": 0, "underpowered": 0, "not_tested": 0})
            a[r["lab"]] = a.get(r["lab"], 0) + 1
        for (module, ds, f, causal), a in sorted(agg.items(), key=lambda kv: (kv[0][0], kv[0][3] is False, kv[0][1], kv[0][2])):
            total = sum(a.values())
            rows.append(f"<tr><td>{esc(module)}</td><td>{esc(scope_name(ds))}</td><td>{esc(FACTOR_RU.get(f, f))}</td>"
                        f"<td>{'' if causal else _pill('описательно, смешано', 'warn')}</td><td>{total}</td><td>{a['significant']}</td><td>{a['not_significant']}</td>"
                        f"<td>{a['underpowered']}</td><td>{a['not_tested']}</td></tr>")
    if not rows:
        return '<p class="muted small">Нет проведённых тестов.</p>'
    return ('<div style="overflow-x:auto"><style>table.sm td:nth-child(-n+4),table.sm th:nth-child(-n+4){text-align:left}</style><table class="metrics sm"><thead><tr><th>Модуль</th><th>Датасет</th><th>Фактор</th><th>Статус</th><th>Тестов</th>'
            '<th>значимо</th><th>не значимо (мощности достаточно)</th><th>мощности не хватает</th><th>не проверялось</th></tr></thead>'
            f'<tbody>{"".join(rows)}</tbody></table></div>')


def _inventory_html(folder: Path) -> str:
    inv = _records(folder, "inventory_by_dataset")
    if not inv:
        return '<p class="muted small">Нет таблицы состава.</p>'
    rows = "".join(f"<tr><td>{esc(DATASET_RU.get(r['dataset'], r['dataset']))}</td><td>{esc({'neuron': 'нейрон', 'branch folder': 'папка ветви'}.get(r.get('unit'), r.get('unit')))}</td><td>{r.get('n_units')}</td><td>{int(r['n_limbs']) if r.get('n_limbs') is not None else '—'}</td>"
                   f"<td>{int(r['n_branches']) if r.get('n_branches') is not None else '—'}</td><td>{r.get('n_spines')}</td><td>{_num(r.get('spines_per_unit_median')) and round(r['spines_per_unit_median'])}</td></tr>" for r in inv)
    miss = _records(folder, "inventory_metadata_missing")
    mrows = "".join(f"<tr><td>{esc(DATASET_RU.get(r['dataset'], r['dataset']))}</td><td>{100 * r['units_without_cell_type']:.0f} %</td><td>{100 * r['spines_without_compartment']:.0f} %</td><td>{100 * r['units_without_health']:.0f} %</td></tr>" for r in miss)
    out = ('<div style="overflow-x:auto"><table class="metrics"><thead><tr><th>Датасет</th><th>Единица вывода</th><th>Единиц</th><th>Лимбов</th><th>Ветвей</th><th>Шипиков</th><th>Шипиков на единицу (медиана)</th></tr></thead>'
           f"<tbody>{rows}</tbody></table></div>")
    if mrows:
        out += ('<h3 style="margin-top:18px">Доля пропущенных метаданных</h3><div style="overflow-x:auto"><table class="metrics"><thead><tr><th>Датасет</th><th>Единиц без типа клетки</th>'
                f"<th>Шипиков без компартмента</th><th>Единиц без состояния здоровья</th></tr></thead><tbody>{mrows}</tbody></table></div>")
    el = _records(folder, "inventory_branch_eligibility")
    if el:
        erows = "".join(f"<tr><td>{esc(DATASET_RU.get(r['dataset'], r['dataset']))}</td><td>{r['n_branches']}</td><td>{int(r['n_ok'])}</td><td>{int(r['n_eligible_T'])}</td><td>{int(r['n_eligible_P'])}</td></tr>" for r in el)
        out += ('<h3 style="margin-top:18px">Ветви, пригодные для моделей T и P</h3><div style="overflow-x:auto"><table class="metrics"><thead><tr><th>Датасет</th><th>Ветвей</th><th>Обработано</th>'
                f"<th>Пригодны для T</th><th>Пригодны для P</th></tr></thead><tbody>{erows}</tbody></table></div>")
    return out


def _inventory_counts(folder: Path) -> Dict[str, List[Dict[str, Any]]]:
    out = {}
    for col in ("cell_type_binary", "cell_class", "cell_type", "layer", "compartment", "health", "treatment", "group"):
        rows = _records(folder, f"inventory_by_dataset_{col}")
        if rows:
            out[col] = [{"ds": r["dataset"], "v": r[col], "units": r["n_units"], "rows": r["n_rows"]} for r in rows]
    return out


def build_payload(folder: Path) -> Dict[str, Any]:
    folder = Path(folder)
    info = json.loads((folder / "run_info.json").read_text(encoding="utf-8")) if (folder / "run_info.json").exists() else {}
    levels = {lv: p for lv in ("spine", "branch") if (p := _level_payload(folder, lv))}
    return {"info": info, "levels": levels, "inventory": _inventory_counts(folder), "dataset_ru": {**DATASET_RU, **{f"{a} vs {b}": f"{SHORT[a]} и {SHORT[b]}" for a in SHORT for b in SHORT if a != b}}, "factor_ru": FACTOR_RU,
            "label_ru": LABEL_RU, "level_ru": LEVEL_RU}


def report_name(info: Dict[str, Any]) -> str:
    return f"{time.strftime('%Y-%m-%d')}_global-report.html"


def build_report(folder: Path, out: Optional[Path] = None) -> Path:
    folder = Path(folder)
    payload = build_payload(folder)
    info, levels = payload["info"], payload["levels"]
    inv = _records(folder, "inventory_by_dataset")
    n_tests = sum(len(lp["results"]) for lp in levels.values())
    n_sig = sum(1 for lp in levels.values() for r in lp["results"] if r["lab"] == "significant")
    n_under = sum(1 for lp in levels.values() for r in lp["results"] if r["lab"] == "underpowered")
    body = [header("Датасеты и дендриты разных типов: что различается, а что нет",
                   "Состав Minnie65, H01 и LabID, распределения морфометрик и статистические различия между типами дендритов для модулей S, T и P",
                   [f"датасетов {len(inv) or '—'}, перестановок {info.get('n_permutations', '—')}", f"создано {info.get('created', '—')}"])]
    body.append(tiles([
        {"label": "Датасетов", "value": str(len(inv)), "note": ", ".join(r["dataset"] for r in inv)},
        {"label": "Единиц вывода", "value": str(sum(r["n_units"] for r in inv)), "note": "нейроны (Minnie65, H01) и папки ветвей (LabID)"},
        {"label": "Шипиков", "value": f"{sum(r['n_spines'] for r in inv):,}".replace(",", " "), "note": "с валидным этапом 8"},
        {"label": "Проведено тестов", "value": str(n_tests), "note": "метрики × сравнения × уровни", "accent": "green"},
        {"label": "Значимо / мало мощности", "value": f"{n_sig} / {n_under}", "note": "q < 0,05 / различие не исключено, но мощности не хватает", "accent": "pink"},
    ]))
    body.append(note([
        "Страница отвечает на два вопроса: что известно о доступных датасетах и какие характеристики шипиков, ствола ветви и расположения шипиков **значимо различаются** у дендритов разных типов, а какие — нет. Результаты нужны, чтобы понимать, какие условия имеет смысл подавать генеративным моделям и какие правила сгенерированные дендриты должны воспроизводить.",
        "**Единица вывода — нейрон**, а в LabID, где нейронов нет, — папка ветви. Шипики и ветви одного нейрона похожи друг на друга сильнее, чем у разных нейронов, поэтому считать их независимыми нельзя: при тысячах шипиков значимым стало бы почти любое различие. Для каждой единицы берётся медиана метрики, группы единиц сравниваются перестановочным тестом, а внутри семейства сравнений p-значения поправлены методом Бенджамини–Хохберга (q < 0,05).",
        "**Смешение названо прямо.** Вид полностью совпадает с датасетом и методом съёмки (человек — H01, электронная микроскопия; мышь — Minnie65, электронная микроскопия, и LabID, конфокальная микроскопия); патология и лечение есть только в LabID; названия типов клеток в датасетах разные. Поэтому сравнения между датасетами и по лечению даны **только как описательные** и помечены отдельно: они не доказывают влияния вида или лечения.",
        "**«Не значимо» не значит «одинаково».** Для каждого теста считаются минимально обнаружимый эффект при имеющемся числе единиц и доверительный интервал эффекта. Если минимально обнаружимый эффект больше порога значимости, страница пишет «мощности не хватает», а не «различий нет».",
    ], title="О чём эта страница"))

    body.append(section("Что можно сравнивать", "Матрица «фактор × датасет»: где сравнение возможно, где оно только описательное, а где нельзя — с причиной.",
                        "".join('<div style="margin-bottom:14px">' + card(LEVEL_RU[lv] + ": допустимость сравнений", _matrix_html(lp["comparisons"])) + "</div>" for lv, lp in levels.items() if lp["comparisons"])
                        + note(["«Можно проверять» означает, что фактор есть в датасете, оба уровня представлены достаточным числом единиц (не меньше порога) и сравнение внутри датасета не смешано с другим фактором. «Только описательно» — сравнение возможно, но его результат смешан: вид с датасетом и методом съёмки, лечение с серией съёмки и поправкой по z. «Нельзя» — метаданных нет или единиц слишком мало; причина указана в клетке.",
                                "Единый словарь для разных датасетов: **класс клетки** (пирамидная, другая возбуждающая, интернейрон) и **слой** выводятся из исходных названий; слой Minnie65 грубый и получен из названия типа клетки (только возбуждающие), то есть он не независим от типа. Слои L2 и L3 в H01 объединены в L2/3 для сравнения с Minnie65, исходные слои H01 сравниваются отдельным фактором."],
                               title="Как читать матрицу", collapsible=True), anchor="sec-validity"))

    body.append(section("Состав датасетов", "Сколько в каждом датасете единиц, ветвей и шипиков, какие метаданные известны и какова доля пропусков.",
                        card("Сводная таблица", _inventory_html(folder))
                        + '<div style="height:14px"></div>' + card("Состав по выбранному признаку", '<div style="max-width:340px;margin-bottom:8px"><select id="sel-inv"></select></div><div id="plot-inv" class="plot m"></div>',
                               hint="число единиц (нейронов или папок ветвей)")
                        + note(["Столбцы показывают число единиц вывода с данным значением признака; «unknown» — метаданные отсутствуют. У Minnie65 компартмент известен только для возбуждающих нейронов (таблица компартментов не содержит тормозных клеток); у H01 компартмент оценён по направлению к поверхности коры и считается приближением; у LabID нет ни типа клетки, ни компартмента, но есть состояние здоровья и лечение."],
                               title="Как читать", collapsible=True), anchor="sec-inventory"))

    body.append(section("Распределения морфометрик", "Распределение выбранной метрики по датасетам (и типам клеток); внутри каждой группы — доля дисперсии между единицами.",
                        card("Распределение", '<div class="row" style="margin-bottom:8px"><div style="min-width:260px"><label class="field">Уровень</label><select id="sel-dist-level"></select></div>'
                             '<div style="min-width:320px"><label class="field">Метрика</label><select id="sel-dist-metric"></select></div>'
                             '<div style="min-width:260px"><label class="field">Группировка</label><select id="sel-dist-by"></select></div></div>'
                             '<div id="plot-dist" class="plot m"></div><div id="dist-table"></div>')
                        + note(["Гистограммы нормированы на площадь и построены на общих корзинах (диапазон 1–99 % всех значений, выбросы собраны в крайние корзины), поэтому датасеты сравнимы по форме. Таблица под графиком даёт число шипиков или групп, число единиц, медиану с квартилями, среднее, стандартное отклонение и **ICC** — долю дисперсии между единицами: чем она ближе к 1, тем меньше независимой информации в тысячах шипиков одного нейрона.",
                                "Абсолютные размеры шипиков LabID с электронной микроскопией сравнивать нельзя: разрешение и растяжение по z разные. У H01 часть шипиков сегментирована без шейки, поэтому длина и объём там занижены."],
                               title="Как читать", collapsible=True), anchor="sec-dist"))

    body.append(section("Что различается между типами дендритов", "Тепловая карта «метрики × сравнения» для выбранного датасета и фактора: цвет — размер эффекта у значимых различий.",
                        card("Значимость и эффект", '<div class="row" style="margin-bottom:8px"><div style="min-width:240px"><label class="field">Уровень</label><select id="sel-sig-level"></select></div>'
                             '<div style="min-width:300px"><label class="field">Датасет</label><select id="sel-sig-scope"></select></div>'
                             '<div style="min-width:360px"><label class="field">Фактор</label><select id="sel-sig-factor"></select></div></div>'
                             '<div id="sig-warning"></div><div class="legend" id="sig-legend"></div><div id="plot-sig" class="plot l"></div>')
                        + '<div style="height:14px"></div>' + card("Сводка по модулям и факторам", _summary_html(levels))
                        + note(["Клетка тепловой карты — одна метрика в одном сравнении (первая группа к второй). **Розовый или зелёный цвет** — значимое различие (q < 0,05), цвет и число — относительная разность медиан единиц (для долей и корреляций — абсолютная разность); шкала обрезана на ±50 %. **Серая клетка** — не значимо и мощности достаточно: различие, не меньшее порога значимости, было бы обнаружено. **Светлая клетка с вопросом** — не значимо, но мощности не хватает: при таком числе единиц нельзя исключить значимого различия, поэтому «нет различий» утверждать нельзя. Пустая клетка — метрика не проверялась.",
                                "При наведении видны медианы групп, число единиц, эффект с 95 % интервалом (бутстреп по единицам), минимально обнаружимый эффект и порог, p, q и признак эквивалентности (интервал целиком внутри порога). Для компартмента сравниваются два компартмента у одного и того же нейрона, поэтому стиль нейрона сокращается.",
                                "Сравнения между датасетами и по лечению помечены рамкой «описательно, смешано» с перечнем смешивающих факторов. Подробные типы клеток проверяются парами; общий тест по всем типам сразу — в таблице омнибус-тестов в файлах рядом со страницей."],
                               title="Как читать", collapsible=True), anchor="sec-sig"))

    body.append(section("Сгенерированные дендриты", "Соответствие сгенерированных данных этим правилам.",
                        note(["Этот раздел ещё не построен. План: те же сравнения (тип клетки, компартмент) считаются по сгенерированным шипикам и ветвям с условиями каждого типа; для каждой метрики сопоставляются эффект реальных данных и эффект сгенерированных, а внутри одного типа реальные и сгенерированные единицы проверяются на эквивалентность с порогом из шумового пола. Подробности генерации — в отчёте условной генерации."],
                             title="Статус раздела", tone="pink"), anchor="sec-generated"))

    body.append(section("Методы и ограничения", "Формулы тестов и список допущений, которые нужно помнить при чтении результатов.",
                        note(["**Статистика между единицами.** Для единицы u и метрики x берётся медиана m_u по её шипикам (или по её ветвям). Статистика сравнения групп A и B — T = |median(m_u, u ∈ A) − median(m_u, u ∈ B)|; нулевое распределение получается 10 000 случайными перестановками меток группы между единицами; p = (1 + число T* ≥ T) / (1 + 10 000).",
                              "**Внутри единицы (компартмент).** Для каждого нейрона с обоими компартментами d_u = m_u(A) − m_u(B); статистика T = |median(d_u)|, нуль — случайная смена знаков d_u; нужно не менее 5 таких нейронов.",
                              "**Общий тест по нескольким типам.** Ранги r_u значений m_u; статистика K = Σ_g (Σ_{u∈g} r_u)² / n_g (в духе Краскела–Уоллиса), нуль — перестановка меток типов между единицами.",
                              "**Поправка.** В каждом семействе (датасет, фактор, страта; метрики × пары) p-значения поправлены по Бенджамини–Хохбергу, q = min_k≥i (p_(k) · n / k); значимо при q < 0,05. Семейства разных датасетов и факторов не смешиваются.",
                              "**Эффект и интервал.** Эффект — ma / mb − 1 (относительный) или ma − mb (абсолютный, для долей, парной корреляционной функции, корреляций и метрик с медианой около 0); 95 % интервал — бутстреп по единицам внутри групп (в парном тесте по разностям нейронов).",
                              "**Мощность.** Минимально обнаружимый эффект MDE = (z_{1−α/2} + z_{мощность}) · √(se_A² + se_B²), se медианы = 1,2533 · s / √n, s = 1,4826 · MAD значений единиц; α = 0,05, мощность 0,8. Порог значимости эффекта: 10 % от опорной медианы для относительных эффектов, 0,5 робастного стандартного отклонения между единицами для абсолютных. Метка «мощности не хватает» — не значимо и MDE больше порога; «эквивалентно» — интервал эффекта целиком внутри ±порога."],
                             title="Методы"),
                        anchor="sec-methods"))
    body.append(note(["**Смешение.** Вид совпадает с датасетом, областью коры, пайплайном сегментации и донором; результат «человек против мыши» — описательное различие датасетов. LabID отличается от Minnie65 и H01 методом съёмки, разрешением и растяжением по z, так что размеры шипиков между ними несопоставимы. Лечение (9009) в LabID смешано с серией съёмки: группы 9009 не получили поправку по z, подтверждения лаборатории нет.",
                      "**Единицы.** У LabID нет сведений о животном и срезе, папки одного животного могут быть зависимыми (псевдорепликация); у H01 компартмент — оценка по направлению к пиа, а не разметка; слой Minnie65 получен из названия типа и грубый; у тормозных клеток Minnie65 нет таблицы компартментов.",
                      "**Что не утверждается.** Страница не делает причинных выводов о виде и лечении, не считает p > 0,05 доказательством сходства, не сопоставляет типы клеток разных датасетов по названию и не применяет общую поправку на множественность по всем семействам сразу."],
                     title="Ограничения", tone="pink"))
    body.append('<footer class="page">Данные: таблицы scripts/neuron_model/build_global_tables.py (морфометрики этапа 8 предобработки S, таблицы ветвей T и P, метаданные репозитория). '
                'Страница: src/neuron_model/reports/global_report.py, логика — assets/global_report.js; проект и схемы — docs/neuron-model/global-report-design.md. Значения в мкм (пересчёт из нм).</footer>')
    js = (Path(__file__).parent / "assets" / "global_report.js").read_text(encoding="utf-8")
    out = Path(out) if out else folder / report_name(info)
    out.write_text(page("Глобальный отчёт по датасетам", "".join(body), payload, js), encoding="utf-8")
    return out
