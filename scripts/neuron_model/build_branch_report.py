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

from src.neuron_model.branch_assembly import densify, load_tp_set, trunk_mesh
from src.neuron_model.branch_assembly.evaluate import trunk_functionals
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


COND_RU = {
    "mouse": "мышь", "human": "человек", "excitatory": "возбуждающая", "inhibitory": "тормозная",
    "apical": "апикальный", "basal": "базальный", "uncertain": "неопределённый",
}


def _cond_label(value) -> str:
    return "не указано" if value is None or (isinstance(value, float) and not np.isfinite(value)) else COND_RU.get(str(value), str(value))


def conditions_of(row) -> list:
    """The conditions of T and P (conditions.py: species, cell type, compartment, length; P also: terminal flag, distance to the soma)."""
    soma = row.get("soma_distance_start_nm")
    has_soma = soma is not None and np.isfinite(soma)
    return [["Вид", _cond_label(row.get("species"))], ["Тип клетки", _cond_label(row.get("cell_type_binary"))], ["Компартмент", _cond_label(row.get("compartment"))],
            ["Длина ветви, мкм", f"{float(row['length_nm']) / 1000.0:.1f}"], ["Концевая ветвь", "да" if bool(row.get("is_terminal")) else "нет"],
            ["Расстояние от сомы до начала ветви, мкм", f"{float(soma) / 1000.0:.1f}" if has_soma else "не указано"]]


def shared_stats(record: dict, s_nm, n_spines) -> dict:
    """Numbers shown for the generated and for the real branch in the same table."""
    f = trunk_functionals(record)
    s_nm = np.sort(np.asarray(s_nm, float))
    length_um = float(record["length"]) / 1000.0
    gaps = np.diff(s_nm) if len(s_nm) > 1 else np.zeros(0)
    return {"length_um": length_um, "radius_mean_nm": _f(f.get("mean_radius_nm")), "radius_cv": _f(f.get("radius_cv")), "tortuosity": _f(f.get("tortuosity")),
            "kappa_rms": _f(f.get("kappa_rms_per_nm")), "n_spines": None if n_spines is None else int(n_spines),
            "density": None if n_spines is None else n_spines / length_um, "gap_median_nm": _f(np.median(gaps)) if len(gaps) else None}


def real_entry(record: dict, row: dict, key: str, sp_rows, max_faces: int, tube_theta: int = 32):
    """The real source branch of a request: a round tube along its measured centre line and radius (the real surface is not round,
    see the audit of the inverse reconstruction), the places where its spines are attached, the table numbers."""
    dense = densify(record, 100.0)
    v, f = trunk_mesh(dense, tube_theta)
    v, f = hk.decimate(v, f, max_faces)
    s_sp = np.asarray(sp_rows["s"], float) if sp_rows is not None else np.zeros(0)
    th_sp = np.asarray(sp_rows["theta"], float) if sp_rows is not None else np.zeros(0)
    eligible = bool(row.get("train_eligible_P")) and row.get("n_spines_in_range") is not None and bool(np.isfinite(row.get("n_spines_in_range")))
    n = int(row["n_spines_in_range"]) if eligible else None
    s, radius = np.asarray(record["s"], float), np.asarray(record["radius"], float)
    anchors = dense.surface_frame(s_sp, th_sp)["anchor"] / 1000.0 if len(s_sp) else np.zeros((0, 3))
    parts = key.split("/")
    return {"key": key, "neuron": parts[1] if len(parts) > 1 else None, "conditions": conditions_of({**row, "length_nm": float(record["length"])}),
            "stats": shared_stats(record, s_sp, n), "eligible_for_spines": eligible,
            "mesh": hk.pack_mesh(v / 1000.0, f, max_faces=10**9),
            "profile": {"s_um": (s / 1000.0).tolist(), "r_nm": radius.tolist()},
            "spines": [{"s_um": float(a / 1000.0), "theta_deg": float(np.degrees(t) % 360.0), "r_nm": float(np.interp(a, s, radius)), "ax": float(p[0]), "ay": float(p[1]), "az": float(p[2])}
                       for a, t, p in zip(s_sp, th_sp, anchors)]}


def trunk_record_of(pack, key: str) -> dict:
    from src.neuron_model.branch_data import trunk_record

    return trunk_record(pack.get(key))


def load_real_source(tp_dir: Path):
    """(rows by key, spine tables by key, geometry pack) of the P0 run that made the set: the same local data as evaluate_assembled_branches.py."""
    from src.neuron_model.branch_data import key_series, prepare_training_data
    from src.neuron_model.spine_generation.experiment.config import resolve_path
    from src.neuron_model.spine_generation.experiment.run import open_run

    gen_info = json.loads((tp_dir / "generation_info.json").read_text(encoding="utf-8"))
    d = open_run(Path(gen_info["p0_run"])).load_config()["data"]
    table, spines, pack = prepare_training_data(resolve_path(d["branches_root"]), resolve_path(d["pack_dir"]),
                                                splits=resolve_path(d["splits"]) if d.get("splits") else None, io_threads=int(d["io_threads"]))
    rows = dict(zip(key_series(table), table.to_dict("records")))
    sp = spines[(spines["status"].astype(str) == "valid") & spines["has_position"].fillna(False).astype(bool)] if len(spines) else spines
    sp_by_key = {k: g for k, g in sp.groupby(key_series(sp))} if len(sp) else {}
    return rows, sp_by_key, pack


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
    parser.add_argument("--no-real", action="store_true", help="do not load the real source branches (the comparison with the real branch of the same conditions is skipped)")
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
    branches_df, trunks, _ = load_tp_set(tp_dir)
    req = branches_df.set_index("branch_index").to_dict("index")
    real_src = None
    if not args.no_real:
        try:
            real_src = load_real_source(tp_dir)
            print(f"[report] real source branches: {len(real_src[2])} in the geometry pack", flush=True)
        except Exception as exc:  # the report is still useful without the real branches
            print(f"[report] real source branches unavailable ({exc!r}): the comparison with the real branch is skipped", flush=True)
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
        sp_list = spines_of(i, s, radius, float(s[-1]))
        placed_s = [x["s_um"] * 1000.0 for x in sp_list if x["status"] == "placed"]
        request = req.get(i, {})
        real = None
        if real_src is not None and request.get("source_branch") in real_src[2]:
            key = request["source_branch"]
            try:
                real = real_entry(trunk_record_of(real_src[2], key), real_src[0].get(key, {}), key, real_src[1].get(key), args.max_faces)
            except Exception as exc:
                print(f"[report] real branch {key}: {exc!r}", flush=True)
        embedded.append({
            "index": i, "length_um": row["length_um"], "placements": row["placements"], "placed": row["placed"], "dropped": row["dropped"], "density": row["density"],
            "status": r.status, "reason": row["reason"], "watertight": row["watertight"], "n_components": row["n_components"], "genus": int(getattr(r, "genus", -1)) if hasattr(r, "genus") else None,
            "n_faces": int(getattr(r, "n_faces", len(mesh.faces))), "shown_faces": int(len(f)), "area_um2": _f(getattr(r, "area_nm2", np.nan)) / 1e6 if _f(getattr(r, "area_nm2", None)) is not None else None,
            "volume_um3": _f(getattr(r, "volume_nm3", np.nan)) / 1e9 if _f(getattr(r, "volume_nm3", None)) is not None else None, "seconds": row["seconds"],
            "radius_median_nm": stats["radius_median_nm"], "tortuosity": stats["tortuosity"],
            "mesh_trunk": _pack_subset(vu, f[~is_spine]), "mesh_spines": _pack_subset(vu, f[is_spine]),
            "center": {"x": (gamma[:, 0] / 1000.0).tolist(), "y": (gamma[:, 1] / 1000.0).tolist(), "z": (gamma[:, 2] / 1000.0).tolist()},
            "profile": {"s_um": (s / 1000.0).tolist(), "r_nm": radius.tolist()},
            "spines": sp_list, "conditions": conditions_of(request) if request else None,
            "stats": shared_stats(t, placed_s, row["placed"]), "real": real,
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
        "Вращать ветвь можно мышью, колесо меняет масштаб. Кнопки над рисунком позволяют сравнить собранную ветвь с реальной ветвью, у которой те же условия и длина. "
        "Ниже список ветвей и статистика по всему набору. Показаны первые ветви набора, но таблица и графики охватывают все собранные ветви.",
        "**Важно.** Форма ствола, число шипиков и их положения зависят от условий ветви (вид, тип клетки, компартмент, длина, концевая ветвь, расстояние до сомы). "
        "Форма самих шипиков пока безусловна: шипик выбирается из пула случайно. Условная генерация шипиков — следующий шаг.",
    ], title="О чём этот отчёт")
    viewer = ('<div class="grid-2">'
              + hk.card("Ветвь", '<div class="row" style="margin-bottom:6px;gap:16px;align-items:center"><select id="branchSelect" aria-label="Ветвь" style="max-width:360px"></select>'
                        '<label class="check"><input type="checkbox" id="showCenter"> центральная линия</label>'
                        '<label class="check"><input type="checkbox" id="showAnchors"> основания шипиков</label></div>'
                        '<div id="viewModeBox" class="row" style="margin-bottom:6px;gap:16px"><span class="muted">Показать:</span>'
                        '<label class="check"><input type="radio" name="viewMode" value="gen" checked> собранную</label>'
                        '<label class="check"><input type="radio" name="viewMode" value="real"> реальную с теми же условиями</label>'
                        '<label class="check"><input type="radio" name="viewMode" value="both"> рядом</label></div>'
                        '<div id="view" class="plot l"></div><div id="viewNote" class="small muted"></div>', hint="вращение мышью")
              + "<div>"
              + hk.card("Условия генерации этой ветви", '<table class="metrics" id="condTable"></table><div id="realSource" class="small muted" style="margin-top:6px"></div>'
                        '<div class="small" style="margin-top:8px"><b>Форма шипиков пока генерируется безусловно.</b> Ствол и число с положениями шипиков зависят от условий выше, '
                        'а сам шипик берётся из пула сгенерированных MoGen сеток случайно, независимо от условий и места на ветви.</div>')
              + '<div style="height:var(--gap)"></div>'
              + hk.card("Сравнение с реальной ветвью", '<table class="metrics" id="branchStats"></table><div id="statsNote" class="small muted"></div>')
              + "</div></div>"
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
