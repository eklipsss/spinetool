"""Characteristics of a generated branch for the web application: trunk (T), spine placement (P), every spine (S).

All values are computed and stored in nm (nm^2, nm^3, nm^-1); a metric description gives the display unit and the
``factor`` nm -> display (the page shows um, um^2, um^3, um^-1). The JSON for the page carries both: ``value`` (nm)
and ``display`` (= value * factor, in ``unit``).

Trunk metrics follow the branch preprocessing where a real counterpart exists (proximal / distal radius = mean over the
first / last 10 % of the length, tortuosity = length / chord, kappa rms over the nodes), so a generated branch can be
put next to the quantiles of real branches of the same condition (``reference_from_branches``). Placement statistics of
ONE branch (g(r), gap CV, neighbour correlations) rest on a handful of spines: they are shown with the number of
spines they come from and are noisy by nature.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np

from ..branch_statistics import PCF_BINS_NM, group_spatial_stats
from ..spine_generation.atlas import METRIC_INFO as SPINE_METRIC_INFO
from ..spine_morphometrics import SCALAR_METRICS
from ..trunk_generation.conditions import group_key
from ..trunk_generation.fit import functionals

QUANTILES = (5, 25, 50, 75, 95)

TRUNK_INFO: Dict[str, Dict[str, Any]] = {
    "length_nm": {"name": "Длина ветви", "unit": "мкм", "factor": 1e-3, "about": "длина центральной линии"},
    "chord_nm": {"name": "Расстояние между концами", "unit": "мкм", "factor": 1e-3, "about": "длина хорды от начала до конца ветви"},
    "tortuosity": {"name": "Извилистость", "unit": "", "factor": 1.0, "about": "длина линии / расстояние между концами (1 — прямая)"},
    "radius_median_nm": {"name": "Медианный радиус", "unit": "мкм", "factor": 1e-3, "about": "медиана радиуса ствола по узлам"},
    "radius_mean_nm": {"name": "Средний радиус", "unit": "мкм", "factor": 1e-3, "about": "среднее радиуса ствола по узлам"},
    "radius_min_nm": {"name": "Минимальный радиус", "unit": "мкм", "factor": 1e-3, "about": "наименьший радиус ствола"},
    "radius_max_nm": {"name": "Максимальный радиус", "unit": "мкм", "factor": 1e-3, "about": "наибольший радиус ствола"},
    "radius_proximal_nm": {"name": "Проксимальный радиус", "unit": "мкм", "factor": 1e-3, "about": "средний радиус на первых 10 % длины (ближе к соме)"},
    "radius_distal_nm": {"name": "Дистальный радиус", "unit": "мкм", "factor": 1e-3, "about": "средний радиус на последних 10 % длины"},
    "taper_ratio": {"name": "Сужение (прокс. / дист.)", "unit": "", "factor": 1.0, "about": "больше 1 — ветвь сужается к дистальному концу"},
    "radius_cv": {"name": "Вариация радиуса (CV)", "unit": "", "factor": 1.0, "about": "стандартное отклонение радиуса вдоль ветви / среднее"},
    "kappa_rms_per_nm": {"name": "Среднеквадратичный изгиб", "unit": "мкм⁻¹", "factor": 1e3, "about": "rms кривизны центральной линии"},
    "kappa_autocorr_1um": {"name": "Автокорреляция изгиба на 1 мкм", "unit": "", "factor": 1.0, "about": "насколько изгиб сохраняется на расстоянии 1 мкм"},
    "step_nm": {"name": "Шаг узлов", "unit": "мкм", "factor": 1e-3, "about": "расстояние между соседними узлами модели T"},
    "n_nodes": {"name": "Число узлов", "unit": "", "factor": 1.0, "about": "узлов центральной линии"},
    "trunk_attempts": {"name": "Попыток генерации ствола", "unit": "", "factor": 1.0, "about": "сколько раз ствол генерировался до прохождения геометрических проверок"},
}

PLACEMENT_INFO: Dict[str, Dict[str, Any]] = {
    "n_requested": {"name": "Шипиков по модели P", "unit": "", "factor": 1.0, "about": "число шипиков, выбранное моделью числа (P0)"},
    "n_placed": {"name": "Шипиков на сетке", "unit": "", "factor": 1.0, "about": "сколько шипиков удалось поставить при сборке"},
    "n_dropped": {"name": "Не поставлено", "unit": "", "factor": 1.0, "about": "положения, где ни один шипик библиотеки не встал (причины ниже)"},
    "density_requested_per_um": {"name": "Плотность по модели P", "unit": "мкм⁻¹", "factor": 1.0, "about": "шипиков модели P на микрометр длины"},
    "density_placed_per_um": {"name": "Плотность на сетке", "unit": "мкм⁻¹", "factor": 1.0, "about": "поставленных шипиков на микрометр длины"},
    "gap_median_nm": {"name": "Медианный интервал", "unit": "мкм", "factor": 1e-3, "about": "медиана расстояний вдоль ветви между соседними шипиками"},
    "gap_cv": {"name": "Вариация интервалов (CV)", "unit": "", "factor": 1.0, "about": "1 — как у случайных положений, меньше 1 — регулярнее"},
    "nn_along_nm": {"name": "До соседа вдоль ветви", "unit": "мкм", "factor": 1e-3, "about": "медиана расстояния до ближайшего шипика вдоль ветви"},
    "surface_min_distance_nm": {"name": "Наименьшее расстояние по поверхности", "unit": "мкм", "factor": 1e-3, "about": "между точками крепления двух шипиков; не меньше d_min модели P"},
    "d_min_nm": {"name": "d_min модели P", "unit": "мкм", "factor": 1e-3, "about": "наименьшее допустимое расстояние между основаниями"},
    "placement_attempts": {"name": "Попыток расстановки", "unit": "", "factor": 1.0, "about": "сколько раз положения перевыбирались из-за d_min"},
    "immersion_median_nm": {"name": "Медианное погружение основания", "unit": "мкм", "factor": 1e-3, "about": "на сколько основание шипика утоплено в ствол при сборке"},
    **{f"pcf_{int(a)}_{int(b)}": {"name": f"g(r), {a / 1000:g}–{b / 1000:g} мкм".replace(".", ","), "unit": "", "factor": 1.0,
                                  "about": "парная корреляция положений вдоль ветви; 1 — случайно, меньше 1 — отталкивание"} for a, b in PCF_BINS_NM},
    "nn_corr_log_volume": {"name": "Корреляция объёма соседей", "unit": "", "factor": 1.0, "about": "соседние шипики похожи по объёму (превышение над перестановкой, 0 — нет сходства)"},
    "nn_corr_length": {"name": "Корреляция длины соседей", "unit": "", "factor": 1.0, "about": "соседние шипики похожи по длине (превышение над перестановкой, 0 — нет сходства)"},
}

BRANCH_MESH_INFO: Dict[str, Dict[str, Any]] = {
    "area_nm2": {"name": "Площадь поверхности ветви", "unit": "мкм²", "factor": 1e-6, "about": "площадь собранной сетки"},
    "volume_nm3": {"name": "Объём ветви", "unit": "мкм³", "factor": 1e-9, "about": "объём собранной сетки"},
    "volume_trunk_nm3": {"name": "Объём ствола", "unit": "мкм³", "factor": 1e-9, "about": "объём трубки без шипиков"},
    "n_vertices": {"name": "Вершин", "unit": "", "factor": 1.0, "about": "в собранной сетке"},
    "n_faces": {"name": "Треугольников", "unit": "", "factor": 1.0, "about": "в собранной сетке"},
    "n_components": {"name": "Связных компонент", "unit": "", "factor": 1.0, "about": "1 — шипики и ствол образуют одну поверхность"},
    "genus": {"name": "Род поверхности", "unit": "", "factor": 1.0, "about": "число «ручек» (0 — без сквозных отверстий)"},
}

SPINE_COLUMNS: Dict[str, Dict[str, Any]] = {
    "index": {"name": "№", "unit": "", "factor": 1.0},
    "status": {"name": "Статус", "unit": "", "factor": 1.0},
    "s_nm": {"name": "Положение s", "unit": "мкм", "factor": 1e-3},
    "u": {"name": "u = s / L", "unit": "", "factor": 1.0},
    "theta": {"name": "Угол θ", "unit": "рад", "factor": 1.0},
    "radius_at_s_nm": {"name": "Радиус ствола в точке", "unit": "мкм", "factor": 1e-3},
    "immersion_nm": {"name": "Погружение основания", "unit": "мкм", "factor": 1e-3},
    "base_radius_nm": {"name": "Радиус основания", "unit": "мкм", "factor": 1e-3},
    "attempts": {"name": "Попыток", "unit": "", "factor": 1.0},
    "library_key": {"name": "Библиотека", "unit": "", "factor": 1.0},
    "spine_id": {"name": "Шипик библиотеки", "unit": "", "factor": 1.0},
    **{m: {"name": SPINE_METRIC_INFO[m]["name"], "unit": SPINE_METRIC_INFO[m]["unit"], "factor": SPINE_METRIC_INFO[m]["factor"],
           "about": SPINE_METRIC_INFO[m]["about"]} for m in SCALAR_METRICS if m in SPINE_METRIC_INFO},
}

FAILURE_REASONS_RU = {
    "base_not_immersed": "основание шире ствола (не погружается)",
    "penetrates_trunk": "тело шипика входит в ствол",
    "collides": "пересекается с уже поставленным шипиком",
}


def _finite(value: Any) -> Optional[float]:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def present(values: Mapping[str, Any], info: Mapping[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Rows for a metric table of the page: key, Russian name, display unit, value (nm) and display value."""
    rows = []
    for key, meta in info.items():
        if key not in values:
            continue
        v = _finite(values[key])
        rows.append({"key": key, "name": meta["name"], "unit": meta["unit"], "about": meta.get("about", ""),
                     "value": v, "display": None if v is None else v * meta["factor"]})
    return rows


# ---------------------------------------------------------------------------
# trunk
# ---------------------------------------------------------------------------


def trunk_metrics(trunk: Mapping[str, Any]) -> Dict[str, Any]:
    """Trunk record of ``generate_trunks`` -> metrics (nm)."""
    s, r, gamma = (np.asarray(trunk[k], dtype=float) for k in ("s", "radius", "gamma"))
    length = float(trunk.get("length", s[-1]))
    step = float(trunk.get("step", np.median(np.diff(s)) if len(s) > 1 else 0.0))
    f = functionals(trunk["kappa1"], trunk["kappa2"], np.log(r), step, gamma=gamma)
    prox, dist = r[s <= 0.1 * length], r[s >= 0.9 * length]
    out = {"length_nm": length, "chord_nm": float(np.linalg.norm(gamma[-1] - gamma[0])), "tortuosity": f["tortuosity"],
           "radius_median_nm": float(np.median(r)), "radius_mean_nm": float(r.mean()), "radius_min_nm": float(r.min()), "radius_max_nm": float(r.max()),
           "radius_proximal_nm": float(prox.mean()) if len(prox) else float(r[0]), "radius_distal_nm": float(dist.mean()) if len(dist) else float(r[-1]),
           "radius_cv": f["radius_cv"], "kappa_rms_per_nm": f["kappa_rms_per_nm"], "kappa_autocorr_1um": f["kappa_autocorr_1um"],
           "step_nm": step, "n_nodes": int(len(s)), "trunk_attempts": trunk.get("attempts")}
    out["taper_ratio"] = out["radius_proximal_nm"] / out["radius_distal_nm"] if out["radius_distal_nm"] > 0 else None
    return out


# ---------------------------------------------------------------------------
# placement and spines
# ---------------------------------------------------------------------------


def spine_rows(assembly_spines: Sequence[Mapping[str, Any]], placement: Mapping[str, Any], length_nm: float,
               library_metrics: Mapping[str, Mapping[str, Optional[float]]]) -> List[Dict[str, Any]]:
    """One row per placement (placed or dropped): position, pose, library spine and its morphometrics (nm).
    ``library_metrics``: pool key (``<library key>#<id>``) -> morphometrics of that library spine."""
    rows = []
    for k, a in enumerate(assembly_spines):
        key = a.get("spine_key")
        lib_key, spine_id = (key.rsplit("#", 1) if key else (None, None))
        row = {"index": int(a["spine_index"]), "status": a.get("status"), "s_nm": float(a["s"]), "u": float(a["s"]) / max(length_nm, 1e-9),
               "theta": float(a["theta"]), "radius_at_s_nm": _finite(a.get("radius_at_s")), "immersion_nm": _finite(a.get("delta_nm")),
               "base_radius_nm": _finite(a.get("base_radius_nm")), "attempts": a.get("attempts"), "library_key": lib_key,
               "spine_id": None if spine_id is None else int(spine_id), "failure_reasons": a.get("failure_reasons"),
               "anchor_nm": None if a.get("anchor_x") is None else [float(a["anchor_x"]), float(a["anchor_y"]), float(a["anchor_z"])]}
        if row["anchor_nm"] is None and placement.get("anchor") is not None and k < len(placement["anchor"]):
            row["anchor_nm"] = [float(x) for x in placement["anchor"][k]]  # dropped placement: the P anchor on the tube
        metrics = library_metrics.get(key, {}) if key else {}
        row.update({m: metrics.get(m) for m in SCALAR_METRICS})
        rows.append(row)
    return rows


def placement_metrics(trunk: Mapping[str, Any], placement: Mapping[str, Any], rows: Sequence[Mapping[str, Any]],
                      n_requested: int, d_min_nm: float) -> Dict[str, Any]:
    """Placement statistics of the PLACED spines (nm; densities per um) + failure counts."""
    from ..placement_generation.generate import surface_min_distance

    length = float(trunk.get("length", np.asarray(trunk["s"])[-1]))
    placed = [r for r in rows if r["status"] == "placed"]
    s = np.sort(np.array([r["s_nm"] for r in placed], dtype=float))
    out: Dict[str, Any] = {"n_requested": int(n_requested), "n_placed": len(placed), "n_dropped": int(len(rows) - len(placed)),
                           "density_requested_per_um": n_requested / (length / 1000.0), "density_placed_per_um": len(placed) / (length / 1000.0),
                           "d_min_nm": float(d_min_nm), "placement_attempts": placement.get("attempts")}
    gaps = np.diff(s)
    out["gap_median_nm"] = float(np.median(gaps)) if len(gaps) else None
    if len(placed):
        th = np.array([r["theta"] for r in placed], dtype=float)
        sp = np.array([r["s_nm"] for r in placed], dtype=float)
        out["surface_min_distance_nm"] = _finite(surface_min_distance(sp, th, np.interp(sp, trunk["s"], trunk["radius"])))
        out["immersion_median_nm"] = _finite(np.median([r["immersion_nm"] for r in placed if r["immersion_nm"] is not None] or [np.nan]))
    import pandas as pd

    frame = pd.DataFrame({"branch": 0, "s": [r["s_nm"] for r in placed], "u": [r["u"] for r in placed],
                          "log_volume": [math.log(r["Volume"]) if r.get("Volume") else np.nan for r in placed],
                          "length": [r["Length"] if r.get("Length") is not None else np.nan for r in placed]})
    frame = frame.drop(columns=[c for c in ("log_volume", "length") if not frame[c].notna().any()])  # library without morphometrics
    # one branch: few pairs, so low thresholds (the page states the number of spines next to these numbers)
    spatial = group_spatial_stats(frame, {0: length}, min_pairs=3, n_null=20, seed=0)
    out.update({k: _finite(v) for k, v in spatial.items() if k in PLACEMENT_INFO})
    reasons: Dict[str, int] = {}
    for r in rows:
        for reason in (r.get("failure_reasons") or "").split(";"):
            if reason:
                reasons[reason] = reasons.get(reason, 0) + 1
    out["failure_reasons"] = reasons
    return out


# ---------------------------------------------------------------------------
# reference: real branches of the same condition
# ---------------------------------------------------------------------------

# generated metric key -> how the branch preprocessing table gives it (module T: T-eligible branches, P: P-eligible)
REFERENCE_METRICS = {
    "length_nm": ("T", lambda t: t["length_nm"]),
    "radius_median_nm": ("T", lambda t: t["radius_median_nm"]),
    "radius_proximal_nm": ("T", lambda t: t["radius_proximal_nm"]),
    "radius_distal_nm": ("T", lambda t: t["radius_distal_nm"]),
    "taper_ratio": ("T", lambda t: t["radius_proximal_nm"] / t["radius_distal_nm"]),
    "tortuosity": ("T", lambda t: t["tortuosity"]),
    "kappa_rms_per_nm": ("T", lambda t: t["kappa_rms"]),
    "density_requested_per_um": ("P", lambda t: t["n_spines"] / (t["length_nm"] / 1000.0)),
    "density_placed_per_um": ("P", lambda t: t["n_spines"] / (t["length_nm"] / 1000.0)),
}


def reference_from_branches(table, *, exclude_compartments: Sequence[str] = ("axon_like",), min_branches: int = 20) -> Dict[str, Any]:
    """Quantiles (``QUANTILES``) of the real metrics per condition group ``species|cell_type_binary|compartment`` and over
    all branches (``all``); a group needs ``min_branches`` branches for a metric. ``table``: ``branches.parquet``."""
    import pandas as pd

    t = table[table["status"].astype(str) == "ok"] if "status" in table.columns else table
    if "compartment" in t.columns:
        t = t[~t["compartment"].astype(str).isin(set(exclude_compartments))]
    t = t.copy()
    cols = [t[c] if c in t.columns else pd.Series(None, index=t.index, dtype=object) for c in ("species", "cell_type_binary", "compartment")]
    t["_group"] = [group_key(*(None if pd.isna(v) else str(v) for v in vals)) for vals in zip(*cols)]
    eligible = {m: t[f"train_eligible_{m}"].fillna(False).astype(bool) if f"train_eligible_{m}" in t.columns else pd.Series(True, index=t.index) for m in ("T", "P")}

    def summary(rows) -> Dict[str, Any]:
        metrics, counts = {}, {}
        for key, (module, fn) in REFERENCE_METRICS.items():
            sub = rows[eligible[module].loc[rows.index]]
            try:
                v = np.asarray(fn(sub), dtype=float)
            except KeyError:  # a column missing in this table version
                continue
            v = v[np.isfinite(v)]
            counts[module] = int(len(sub))
            if len(v) >= min_branches:
                metrics[key] = [float(np.percentile(v, q)) for q in QUANTILES]
        return {"n_branches_T": counts.get("T", 0), "n_branches_P": counts.get("P", 0), "metrics": metrics}

    groups = {g: summary(rows) for g, rows in t.groupby("_group")}
    groups = {g: v for g, v in groups.items() if v["metrics"]}
    return {"quantiles": list(QUANTILES), "units": "nm", "groups": groups, "all": summary(t), "n_branches": int(len(t))}


def reference_rows(reference: Optional[Mapping[str, Any]], request: Mapping[str, Any], values: Mapping[str, Any]) -> Dict[str, Any]:
    """Generated values next to the real quantiles of the request's condition group (or of all branches)."""
    if not reference:
        return {"available": False}
    group = group_key(request.get("species"), request.get("cell_type_binary"), request.get("compartment"))
    data = reference["groups"].get(group)
    used = group if data else "all"
    data = data or reference["all"]
    info = {**TRUNK_INFO, **PLACEMENT_INFO}
    rows = []
    for key, q in data["metrics"].items():
        meta = info[key]
        v = _finite(values.get(key))
        rows.append({"key": key, "name": meta["name"], "unit": meta["unit"], "generated": None if v is None else v * meta["factor"],
                     "quantiles": [x * meta["factor"] for x in q]})
    return {"available": True, "group": used, "requested_group": group, "quantiles": list(reference["quantiles"]),
            "n_branches_T": data["n_branches_T"], "n_branches_P": data["n_branches_P"], "rows": rows}
