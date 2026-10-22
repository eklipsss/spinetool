"""One self-contained HTML report of assembled (generated) branches: an interactive 3-D view of each branch (trunk and spines in
different colours, centre line, spine bases) and statistics of the whole set. Opens offline, can be sent as a single file.

    python scripts/neuron_model/build_branch_report.py --assembly <tp-dir>/assembly_<date> \
        [--evaluation <evaluate_assembled_branches output>] [--max-branches 10] [--max-faces 50000]

Reads only files written by ``generate_tp.py`` (trunks, placements), ``assemble_branches.py`` (meshes and tables) and, optionally,
``evaluate_assembled_branches.py`` (``evaluation_branches.csv``: generated vs real branches of the same queries).
Output: ``<assembly>/branch_report.html``.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd

from src.neuron_model.branch_assembly import load_tp_set
from src.neuron_model.reports import html_kit as hk

ASSET = REPO_ROOT / "src/neuron_model/reports/assets/branch_report.js"
REASONS = {
    "base_not_immersed": "основание шире ствола",
    "penetrates_trunk": "шипик пронзает ствол",
    "collides": "пересечение с другим шипиком",
    "trunk_mesh": "не построился ствол",
    "empty": "пустая сетка",
    "not_watertight": "сетка не замкнута",
    "inconsistent_winding": "несогласованные нормали",
    "non_positive_volume": "неположительный объём",
    "no_attachment_region": "не найдено основание шипика",
}
EVAL_METRICS = [
    ("n_spines", "Число шипиков на ветвь", "Для реальных ветвей тоже учитываются только размеченные шипики."),
    ("density_per_um", "Плотность, шипиков на мкм", "Число шипиков делится на длину ветви."),
    ("tortuosity", "Извилистость ствола", "Длина пути, делённая на расстояние между концами. Для прямой линии 1."),
    ("radius_cv", "Разброс радиуса вдоль ветви (коэффициент вариации)", "Стандартное отклонение радиуса, делённое на среднее."),
    ("mean_radius_nm", "Средний радиус, нм", ""),
]


def _f(x):
    return None if x is None or (isinstance(x, float) and not np.isfinite(x)) else float(x)


def _classify(vertices: np.ndarray, faces: np.ndarray, gamma: np.ndarray, s: np.ndarray, radius: np.ndarray):
    """Faces of the trunk tube vs faces of spines: a face whose mean distance to the centre line exceeds 1.15 of the local radius is a spine."""
    from scipy.spatial import cKDTree

    dense_s = np.arange(0.0, float(s[-1]) + 50.0, 50.0)
    dense = np.stack([np.interp(dense_s, s, gamma[:, k]) for k in range(3)], axis=1)
    dense_r = np.interp(dense_s, s, radius)
    d, k = cKDTree(dense).query(vertices)
    ratio = d / np.maximum(dense_r[k], 1.0)
    face_ratio = ratio[faces].mean(axis=1)
    return face_ratio > 1.15


def _pack_subset(vertices_um: np.ndarray, faces: np.ndarray):
    if not len(faces):
        return None
    used, inverse = np.unique(faces.ravel(), return_inverse=True)
    return hk.pack_mesh(vertices_um[used], inverse.reshape(-1, 3), max_faces=10**9)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--assembly", type=Path, required=True)
    parser.add_argument("--tp-dir", type=Path, help="default: tp_dir from assembly_info.json, else the parent of --assembly")
    parser.add_argument("--evaluation", type=Path, help="output of evaluate_assembled_branches.py (evaluation_branches.csv); default: look in --assembly")
    parser.add_argument("--max-branches", type=int, default=10, help="branches embedded as 3-D meshes (the statistics cover all branches)")
    parser.add_argument("--max-faces", type=int, default=50_000, help="display limit of faces per branch")
    parser.add_argument("--title", default="Сгенерированные ветви дендрита")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    import trimesh

    asm = args.assembly
    info = json.loads((asm / "assembly_info.json").read_text(encoding="utf-8")) if (asm / "assembly_info.json").exists() else {}
    tp_dir = args.tp_dir or (Path(info["tp_dir"]) if info.get("tp_dir") and Path(info["tp_dir"]).exists() else asm.parent)
    bt = pd.read_parquet(asm / "assembly_branches.parquet")
    st = pd.read_parquet(asm / "assembly_spines.parquet") if (asm / "assembly_spines.parquet").exists() else pd.DataFrame()
    rej = pd.read_parquet(asm / "spine_pool_rejected.parquet") if (asm / "spine_pool_rejected.parquet").exists() else pd.DataFrame(columns=["key", "reason"])
    _, trunks, _ = load_tp_set(tp_dir)
    assembled = bt[bt["status"].isin(["ok", "invalid"]) & bt.get("mesh_path", pd.Series(dtype=object)).notna()].sort_values("branch_index")

    def trunk_stats(i: int) -> dict:
        t = trunks[i]
        gamma, radius, s = np.asarray(t["gamma"], float), np.asarray(t["radius"], float), np.asarray(t["s"], float)
        path = float(np.linalg.norm(np.diff(gamma, axis=0), axis=1).sum())
        chord = float(np.linalg.norm(gamma[-1] - gamma[0]))
        return {"radius_median_nm": float(np.median(radius)), "tortuosity": path / chord if chord > 0 else None}

    def spines_of(i: int, prof_s: np.ndarray, prof_r: np.ndarray, length_nm: float) -> list:
        sub = st[st["branch_index"] == i].sort_values("spine_index") if len(st) else st
        out = []
        for r in sub.itertuples():
            ok = r.status == "placed"
            out.append({"k": int(r.spine_index), "status": r.status, "s_um": r.s / 1000.0, "u": r.s / length_nm, "theta_deg": float(np.degrees(r.theta) % 360.0),
                        "r_nm": float(np.interp(r.s, prof_s, prof_r)), "delta_nm": _f(getattr(r, "delta_nm", None)) if ok else None,
                        "reasons": ", ".join(REASONS.get(x, x) for x in str(r.failure_reasons).split(";")) if isinstance(r.failure_reasons, str) else None,
                        "ax": _f(getattr(r, "anchor_x", None)) / 1000.0 if ok and _f(getattr(r, "anchor_x", None)) is not None else None,
                        "ay": _f(getattr(r, "anchor_y", None)) / 1000.0 if ok and _f(getattr(r, "anchor_y", None)) is not None else None,
                        "az": _f(getattr(r, "anchor_z", None)) / 1000.0 if ok and _f(getattr(r, "anchor_z", None)) is not None else None})
        return out

    # ---- table over all branches
    table = []
    for r in assembled.itertuples():
        i = int(r.branch_index)
        sub = st[st["branch_index"] == i] if len(st) else st
        placed = int((sub["status"] == "placed").sum()) if len(sub) else 0
        length_um = float(r.length_nm) / 1000.0
        table.append({"index": i, "length_um": length_um, "placements": int(len(sub)), "placed": placed, "dropped": int(len(sub)) - placed, "density": placed / length_um if length_um > 0 else None,
                      "watertight": bool(getattr(r, "watertight", False)), "n_components": int(getattr(r, "n_components", 0)), "seconds": _f(getattr(r, "seconds", None)),
                      "status": r.status, "reason": getattr(r, "reason", None) if isinstance(getattr(r, "reason", None), str) else None})

    # ---- embedded branches (meshes)
    embedded = []
    for r in assembled.head(args.max_branches).itertuples():
        i = int(r.branch_index)
        mesh = trimesh.load(asm / r.mesh_path, process=False)
        v, f = hk.decimate(np.asarray(mesh.vertices, float), np.asarray(mesh.faces, np.int64), args.max_faces)
        t = trunks[i]
        gamma, radius, s = np.asarray(t["gamma"], float), np.asarray(t["radius"], float), np.asarray(t["s"], float)
        is_spine = _classify(v, f, gamma, s, radius)
        vu = v / 1000.0
        stats = trunk_stats(i)
        row = next(x for x in table if x["index"] == i)
        embedded.append({
            "index": i, "length_um": row["length_um"], "placements": row["placements"], "placed": row["placed"], "dropped": row["dropped"], "density": row["density"],
            "status": r.status, "reason": row["reason"], "watertight": row["watertight"], "n_components": row["n_components"], "genus": int(getattr(r, "genus", -1)) if hasattr(r, "genus") else None,
            "n_faces": int(getattr(r, "n_faces", len(mesh.faces))), "shown_faces": int(len(f)), "area_um2": _f(getattr(r, "area_nm2", np.nan)) / 1e6 if _f(getattr(r, "area_nm2", None)) is not None else None,
            "volume_um3": _f(getattr(r, "volume_nm3", np.nan)) / 1e9 if _f(getattr(r, "volume_nm3", None)) is not None else None, "seconds": row["seconds"],
            "radius_median_nm": stats["radius_median_nm"], "tortuosity": stats["tortuosity"],
            "mesh_trunk": _pack_subset(vu, f[~is_spine]), "mesh_spines": _pack_subset(vu, f[is_spine]),
            "center": {"x": (gamma[:, 0] / 1000.0).tolist(), "y": (gamma[:, 1] / 1000.0).tolist(), "z": (gamma[:, 2] / 1000.0).tolist()},
            "profile": {"s_um": (s / 1000.0).tolist(), "r_nm": radius.tolist()},
            "spines": spines_of(i, s, radius, float(s[-1])),
        })
        print(f"[report] branch {i}: {len(mesh.faces)} faces -> {len(f)} shown, spines {row['placed']}/{row['placements']}", flush=True)

    # ---- spine / pool statistics over all branches
    placed_df = st[st["status"] == "placed"] if len(st) else st
    lengths = {int(r.branch_index): float(r.length_nm) for r in assembled.itertuples()}
    u_all = [float(r.s) / lengths[int(r.branch_index)] for r in placed_df.itertuples() if int(r.branch_index) in lengths]
    placed_spines = {"delta_nm": [_f(x) for x in placed_df.get("delta_nm", pd.Series(dtype=float))], "u": u_all,
                     "theta_deg": [float(np.degrees(x) % 360.0) for x in placed_df["theta"]] if len(placed_df) else []}
    failure = {}
    if len(st) and "failure_reasons" in st.columns:
        failure = st["failure_reasons"].dropna().str.split(";").explode().value_counts().to_dict()
    pool_rej = rej["reason"].value_counts().to_dict() if len(rej) else {}
    n_pool = int(info.get("pool_size", 0))
    used = int(placed_df["spine_key"].nunique()) if len(placed_df) and "spine_key" in placed_df.columns else 0
    methods = placed_df["attachment_method"].value_counts().to_dict() if len(placed_df) and "attachment_method" in placed_df.columns else {}

    # ---- optional: generated vs real branches of the same queries
    evaluation = None
    ev_path = (args.evaluation or asm) / "evaluation_branches.csv"
    if ev_path.exists():
        ev = pd.read_csv(ev_path)
        have = [(k, n, a) for k, n, a in EVAL_METRICS if k in ev.columns]
        if have and "kind" in ev.columns:
            evaluation = {"metrics": [{"key": k, "name": n, "about": a} for k, n, a in have],
                          "values": {kind: {k: [_f(x) for x in ev.loc[ev["kind"] == kind, k]] for k, _, _ in have} for kind in ("real", "generated")}}

    ok_n = int((assembled["status"] == "ok").sum())
    tiles = [
        {"label": "Ветвей собрано", "value": f"{len(assembled)}", "note": f"без замечаний {ok_n}", "accent": "pink"},
        {"label": "Замкнутых и связных", "value": f"{100 * np.mean([r['watertight'] and r['n_components'] == 1 for r in table]):.0f} %" if table else "—", "note": "одна замкнутая поверхность на ветвь", "accent": "green"},
        {"label": "Шипиков поставлено", "value": f"{int((st['status'] == 'placed').sum()) if len(st) else 0:,}".replace(",", " "),
         "note": f"не удалось поставить {int((st['status'] != 'placed').sum()) if len(st) else 0}"},
        {"label": "Время сборки ветви", "value": f"{np.nanmedian([r['seconds'] for r in table if r['seconds'] is not None]):.1f} с" if table else "—", "note": "медиана"},
        {"label": "Пул сеток шипиков", "value": f"{n_pool}", "note": f"использовано {used}, отбраковано при загрузке {len(rej)}"},
    ]
    meta_lines = [f"Набор T и P: {tp_dir.name}", f"Сетки шипиков: {Path(info.get('spines', '—')).parent.parent.name if info.get('spines') else '—'}", f"Построено {_dt.datetime.now():%d.%m.%Y %H:%M}"]
    payload = {"branches": embedded, "table": table, "placed_spines": placed_spines, "failure_reasons": failure, "reason_names": REASONS,
               "pool": {"size": n_pool, "rejected_reasons": pool_rej, "attachment_methods": methods}, "evaluation": evaluation}

    head = hk.header(args.title, "Ветви целиком сгенерированы: ствол, число и положения шипиков, а затем сами шипики.", meta_lines)
    intro = hk.note([
        "Здесь собраны ветви дендрита, у которых всё порождено моделями. Форму ствола задаёт модель T, число шипиков и их положения модель P, форму каждого шипика модель MoGen после дообучения. "
        "Затем шипики вставляются в ствол и объединяются с ним в одну замкнутую поверхность. Условия генерации (тип клетки, длина ветви и другие) взяты у реальных ветвей, поэтому каждой "
        "собранной ветви можно сопоставить реальную ветвь с теми же условиями.",
        "**Что показано.** Ствол серый, шипики розовые. Флажками можно добавить центральную линию ствола (чёрная) и зелёные точки в местах, где основания шипиков вставлены в ствол. "
        "Вращать ветвь можно мышью, колесо меняет масштаб. Ниже список ветвей и статистика по всему набору. "
        "Показаны первые ветви набора, но таблица и графики охватывают все собранные ветви.",
    ], title="О чём этот отчёт")
    viewer = ('<div class="grid-2">'
              + hk.card("Ветвь", '<div class="row" style="margin-bottom:6px;gap:16px;align-items:center"><select id="branchSelect" aria-label="Ветвь" style="max-width:360px"></select>'
                        '<label class="check"><input type="checkbox" id="showCenter"> центральная линия</label>'
                        '<label class="check"><input type="checkbox" id="showAnchors"> основания шипиков</label></div><div id="view" class="plot l"></div><div id="viewNote" class="small muted"></div>',
                        hint="вращение мышью")
              + hk.card("Параметры выбранной ветви", '<table class="metrics" id="branchStats"></table>')
              + "</div>"
              + hk.card("Радиус ствола и положения шипиков вдоль ветви", '<div id="profile" class="plot m"></div>'))
    glob = ('<div class="grid-2e">'
            + hk.card("Длина ветви и число шипиков", '<div id="lenSpines" class="plot s"></div>')
            + hk.card("Плотность шипиков", '<div id="densityHist" class="plot s"></div>')
            + hk.card("Положения шипиков вдоль ветви", '<div id="uHist" class="plot s"></div>', hint="пунктир — равномерное распределение")
            + hk.card("Углы шипиков вокруг оси", '<div id="thetaHist" class="plot s"></div>', hint="пунктир — равномерное распределение")
            + hk.card("Глубина погружения шипика в ствол", '<div id="deltaHist" class="plot s"></div>')
            + hk.card("Почему шипик пришлось заменить или не удалось поставить", '<div id="attemptReasons" class="plot s"></div>')
            + hk.card("Какие сгенерированные шипики не попали в пул", '<div id="poolReasons" class="plot s"></div>')
            + "</div>")
    glob_note = hk.note([
        "Плотность и положения показывают, насколько ветви похожи на реальные. Положения вдоль ветви и углы должны быть близки к равномерным, как в реальных данных. "
        "Глубина погружения показывает, насколько глубоко основание шипика уходит в ствол, чтобы крышка основания оказалась внутри.",
        "Если шипик не удалось поставить, берётся другой из пула (до нескольких попыток). Причина записывается на каждой попытке, поэтому число случаев больше числа неудавшихся шипиков. "
        "В пул попадают только замкнутые сгенерированные сетки с положительным объёмом и найденным основанием.",
    ], title="Как читать статистику", collapsible=True)
    table_html = hk.card("Все ветви", '<div style="overflow:auto;max-height:420px"><table class="metrics"><thead><tr><th>Ветвь</th><th>Длина, мкм</th><th>Запрошено</th><th>Поставлено</th>'
                         '<th>Не поставлено</th><th>Плотность, на мкм</th><th>Замкнута</th><th>Компонент</th><th>Время, с</th><th>Статус</th></tr></thead><tbody id="tableBody"></tbody></table></div>')
    evalsec = ""
    if evaluation:
        evalsec = hk.section("Сравнение с реальными ветвями", "Собранные ветви и реальные ветви с теми же условиями и длиной.",
                             hk.card("Распределение показателя", '<select id="evalMetric" aria-label="Показатель" style="max-width:420px"></select><div id="evalPlot" class="plot s"></div><div id="evalAbout" class="small muted"></div>'))
    body = (head + hk.tiles(tiles) + intro + hk.section("Ветви в 3D", "Выберите ветвь, чтобы рассмотреть её со всех сторон.", viewer, anchor="sec-view")
            + hk.section("Статистика набора", "Число шипиков, плотность, положения и причины неудач по всем собранным ветвям.", glob + glob_note + table_html, anchor="sec-stats")
            + evalsec + f'<footer class="page">{hk.esc("Сетки в отчёте упрощены для показа, все числа посчитаны по полным сеткам. Длины в мкм, радиусы в нм.")}</footer>')
    out = args.out or asm / "branch_report.html"
    out.write_text(hk.page(args.title, body, payload, ASSET.read_text(encoding="utf-8")), encoding="utf-8")
    print(f"[report] {len(embedded)} branches embedded, {out.stat().st_size / 1e6:.1f} MB -> {out}", flush=True)


if __name__ == "__main__":
    main()
