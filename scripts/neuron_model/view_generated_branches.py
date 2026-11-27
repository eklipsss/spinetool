"""Generated branches next to the real branches they were requested from: one interactive HTML page.

    python scripts/neuron_model/view_generated_branches.py --tp-dir <generate_tp set> [--assembly <tp>/assembly_<x>] ^
        [--raw-data-root O:\\Datasets --dataset minnie65] [--n 30] [--out <page.html>]

For every chosen request (a dropdown on the page): the real source branch (its mesh from the raw segment meshes when
``--raw-data-root`` is given, spines coloured; otherwise the tube of its centreline and radius) and the generated branch
(the assembled mesh when ``--assembly`` is given, spines coloured by the distance from the trunk; otherwise the tube with
the spine attachment points). Below: radius along the branch (real and generated on one plot), spine positions along
the branch and around it (s, theta; both on one plot), nearest-neighbour distances on the trunk surface, and a table
of the branch metrics (the same functionals and spine statistics as ``evaluate_assembled_branches.py``).
Real reference = the source branch of every request; its tables and geometry come from the P0 run of the set
(``data.branches_root`` / ``data.pack_dir``), the meshes from the raw dataset (NAS). Only reads files. Lengths in the
page are in micrometres (converted from nm).
"""

from __future__ import annotations

import argparse
import base64
import datetime as _dt
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if sys.platform == "win32":
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch  # noqa: F401,E402  (torch FIRST on Windows, see fit_t0.py)
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.neuron_model.branch_assembly import densify, load_tp_set, trunk_mesh  # noqa: E402
from src.neuron_model.branch_assembly.evaluate import spine_geometry_stats, trunk_functionals  # noqa: E402
from src.neuron_model.branch_data import key_series, load_branches, prepare_training_data, trunk_record  # noqa: E402
from src.neuron_model.spine_generation.experiment.config import resolve_path  # noqa: E402
from src.neuron_model.spine_generation.experiment.run import open_run  # noqa: E402

PINK, GREEN, INK, GREY = "#d6418f", "#3f8a2a", "#1d1d1f", "#8e9196"
FUNCTIONALS = [("mean_radius_nm", "средний радиус, нм", 1.0, 1), ("radius_ratio_prox_dist", "отношение радиусов прокс./дист.", 1.0, 3),
               ("radius_cv", "CV радиуса", 1.0, 3), ("kappa_rms_per_nm", "изгиб κ rms, мкм⁻¹", 1e3, 3),
               ("tortuosity", "извилистость", 1.0, 3), ("kappa_autocorr_1um", "автокорреляция изгиба, 1 мкм", 1.0, 3)]


# ---------------------------------------------------------------------------
# meshes
# ---------------------------------------------------------------------------


def cluster_decimate(vertices: np.ndarray, faces: np.ndarray, max_faces: int, flags: np.ndarray | None = None):
    """Vertex-clustering decimation (display only): vertices are merged on a grid whose cell grows until at most
    ``max_faces`` faces remain. ``flags`` (bool per vertex) -> True for a merged vertex if any of its vertices was."""
    v, f = np.asarray(vertices, float), np.asarray(faces, np.int64)
    fl = None if flags is None else np.asarray(flags, bool)
    if len(f) <= max_faces:
        return v, f, fl
    edge = np.linalg.norm(v[f[:, 0]] - v[f[:, 1]], axis=1)
    h = float(np.median(edge)) if len(edge) else 1.0
    for _ in range(40):
        h *= 1.25
        key = np.floor((v - v.min(axis=0)) / h).astype(np.int64)
        _, inv = np.unique(key, axis=0, return_inverse=True)
        inv = inv.reshape(-1)
        n = int(inv.max()) + 1
        cnt = np.bincount(inv, minlength=n).astype(float)
        nv = np.column_stack([np.bincount(inv, weights=v[:, d], minlength=n) for d in range(3)]) / cnt[:, None]
        nf = inv[f]
        nf = nf[(nf[:, 0] != nf[:, 1]) & (nf[:, 1] != nf[:, 2]) & (nf[:, 0] != nf[:, 2])]
        if len(nf):
            first = np.unique(np.sort(nf, axis=1), axis=0, return_index=True)[1]
            nf = nf[np.sort(first)]  # duplicated faces after merging, orientation of the first kept
        nfl = None if fl is None else np.bincount(inv, weights=fl.astype(float), minlength=n) > 0
        if len(nf) <= max_faces:
            return nv, nf, nfl
    return nv, nf, nfl


def _b64(a: np.ndarray, dtype) -> str:
    return base64.b64encode(np.ascontiguousarray(a, dtype=dtype).tobytes()).decode("ascii")


def pack(vertices: np.ndarray, faces: np.ndarray, flags: np.ndarray | None, origin: np.ndarray) -> dict:
    """Vertices in nm relative to ``origin`` as int32, faces as uint32, flags as uint8 (base64)."""
    v = np.round(np.asarray(vertices, float) - origin).astype(np.int32)
    out = {"v": _b64(v, "<i4"), "f": _b64(faces, "<u4"), "nv": int(len(v)), "nf": int(len(faces))}
    if flags is not None:
        out["flag"] = _b64(np.asarray(flags, np.uint8), "u1")
    return out


def real_branch_mesh(raw_root: Path, dataset: str, row: dict, io_threads: int = 8):
    """(vertices, faces, spine flags) of the merged branch from the raw segment meshes (as the reconstruction audit)."""
    from src.neuron_model.branch_preprocessing.config import BranchPreprocessingConfig
    from src.neuron_model.branch_preprocessing.process import branch_segments, load_branch_surface

    cfg = BranchPreprocessingConfig(dataset=dataset, raw_data_root=Path(raw_root), io_threads=io_threads)
    limb = int(row["limb_id"])
    members = [int(b) for b in str(row["member_branch_ids"]).split("|")]
    neuron_dir = cfg.dataset_root / str(row["neuron_id"])
    member_dirs = {b: neuron_dir / f"limb_{limb:03d}" / f"branch_{b:03d}" for b in members}
    return load_branch_surface(cfg, member_dirs, branch_segments(member_dirs, {}, limb))


# ---------------------------------------------------------------------------
# per-branch payload
# ---------------------------------------------------------------------------


def _num(x, nd=3):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return None if not np.isfinite(x) else round(x, nd)


def branch_metrics(trunk: dict, s: np.ndarray, theta: np.ndarray) -> dict:
    s = np.asarray(s, float)
    r = np.interp(s, trunk["s"], trunk["radius"]) if len(s) else np.zeros(0)
    st = spine_geometry_stats(s, np.asarray(theta, float), r, float(trunk["length"]))
    out = {k: _num(v, 6) for k, v in trunk_functionals(trunk).items()}
    length = float(trunk["length"])
    out.update({"length_um": _num(length / 1000.0), "n_spines": int(len(s)), "density_per_um": _num(len(s) / max(length / 1000.0, 1e-9)),
                "nn_surface_median_nm": _num(np.nanmedian(st["nn_surface_nm"]) if len(s) >= 2 else np.nan, 1),
                "gap_median_nm": _num(np.median(st["gap_nm"]) if len(st["gap_nm"]) else np.nan, 1)})
    return out, st


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tp-dir", type=Path, required=True, help="output directory of generate_tp.py")
    parser.add_argument("--assembly", type=Path, help="assemble_branches.py output on this set (generated meshes with spines)")
    parser.add_argument("--raw-data-root", type=Path, help="raw dataset root (NAS: O:\\Datasets) for the real branch meshes; default: tubes")
    parser.add_argument("--dataset", default="minnie65")
    parser.add_argument("--n", type=int, default=30, help="branches on the page")
    parser.add_argument("--only-with-spines", action="store_true", help="only requests whose real source has spines with a position")
    parser.add_argument("--max-faces", type=int, default=20000, help="display meshes are decimated to at most this many faces")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    info = json.loads((args.tp_dir / "generation_info.json").read_text(encoding="utf-8"))
    d = open_run(Path(info["p0_run"])).load_config()["data"]
    root = resolve_path(d["branches_root"])
    table, spines, gpack = prepare_training_data(root, resolve_path(d["pack_dir"]), splits=resolve_path(d["splits"]) if d.get("splits") else None,
                                                 io_threads=int(d["io_threads"]))
    rows_by_key = dict(zip(key_series(table), table.to_dict("records")))
    members = {}
    if args.raw_data_root:
        mt = load_branches(root, columns=["dataset", "neuron_id", "limb_id", "merged_branch_id", "member_branch_ids"])
        members = dict(zip(key_series(mt), mt.to_dict("records")))
    sp = spines[(spines["status"].astype(str) == "valid") & spines["has_position"].fillna(False).astype(bool)] if len(spines) else spines
    sp_by_key = {k: g for k, g in sp.groupby(key_series(sp))} if len(sp) else {}

    branches, trunks, placements = load_tp_set(args.tp_dir)
    candidates = list(range(len(trunks)))
    asm, placed = None, None
    if args.assembly:
        asm = pd.read_parquet(args.assembly / "assembly_branches.parquet")
        candidates = sorted(int(i) for i in asm.loc[asm["mesh_path"].notna(), "branch_index"]) if "mesh_path" in asm.columns else []
        st = pd.read_parquet(args.assembly / "assembly_spines.parquet")
        placed = {int(k): set(g.loc[g["status"] == "placed", "spine_index"].astype(int)) for k, g in st.groupby("branch_index")}
    src_of = dict(zip(branches["branch_index"].astype(int), branches["source_branch"].astype(str)))
    candidates = [i for i in candidates if src_of.get(i) in gpack and src_of.get(i) in rows_by_key]
    if args.only_with_spines:
        candidates = [i for i in candidates if src_of[i] in sp_by_key]
    rng = np.random.default_rng(args.seed)
    chosen = sorted(rng.choice(candidates, size=min(args.n, len(candidates)), replace=False).tolist()) if candidates else []
    if not chosen:
        raise SystemExit("no generated branch with a real source in the tables of the P0 run")
    brow = branches.set_index("branch_index")

    items = []
    for n_done, i in enumerate(chosen, start=1):
        key = src_of[i]
        row = rows_by_key[key]
        real = trunk_record(gpack.get(key))
        gen = trunks[i]
        # real spines: valid with a position (only when the branch is eligible for P, otherwise its count is unknown)
        g = sp_by_key.get(key)
        eligible_p = bool(row.get("train_eligible_P"))
        rs = g["s"].to_numpy(float) if g is not None else np.zeros(0)
        rt = g["theta"].to_numpy(float) if g is not None else np.zeros(0)
        p = placements.get(i)
        if p is not None and placed is not None:
            p = p[p["spine_index"].astype(int).isin(placed.get(i, set()))]
        gs = p["s"].to_numpy(float) if p is not None else np.zeros(0)
        gt = p["theta"].to_numpy(float) if p is not None else np.zeros(0)
        m_real, st_real = branch_metrics(real, rs, rt)
        m_gen, st_gen = branch_metrics(gen, gs, gt)
        if not eligible_p:
            m_real["n_spines_note"] = "ветвь не пригодна для P: число шипиков не определено"

        # meshes
        o_real = np.asarray(real["gamma"], float)[0]
        mesh_real, real_kind = None, "tube"
        if args.raw_data_root and key in members:
            try:
                v, f, flag = real_branch_mesh(args.raw_data_root, args.dataset, members[key])
                v, f, flag = cluster_decimate(v, f, args.max_faces, flag)
                mesh_real, real_kind = pack(v, f, flag, o_real), "raw"
            except Exception as exc:  # noqa: BLE001  (a missing file on the NAS: fall back to the tube)
                print(f"[view] {key}: raw mesh not read ({exc!r}); tube shown", flush=True)
        if mesh_real is None:
            v, f = trunk_mesh(densify(real))
            mesh_real = pack(v, f, None, o_real)
        o_gen = np.asarray(gen["gamma"], float)[0]
        gen_kind = "tube"
        if asm is not None:
            arow = asm.loc[asm["branch_index"] == i].iloc[0]
            import trimesh

            m = trimesh.load(args.assembly / arow["mesh_path"], process=False)
            v, f = np.asarray(m.vertices, float), np.asarray(m.faces)
            dense = densify(gen)
            flag = dense.signed_distance(v) > 40.0  # spine = farther than 40 nm outside the generated trunk surface
            v, f, flag = cluster_decimate(v, f, args.max_faces, flag)
            mesh_gen, gen_kind = pack(v, f, flag, o_gen), "assembled"
        else:
            v, f = trunk_mesh(densify(gen))
            mesh_gen = pack(v, f, None, o_gen)

        def anchors(trunk, s, theta, origin):
            if not len(s):
                return []
            fr = densify(trunk).surface_frame(np.asarray(s, float), np.asarray(theta, float))
            return np.round((fr["anchor"] - origin) / 1000.0, 4).tolist()

        b = brow.loc[i]
        items.append({
            "index": int(i), "source": key, "neuron": key.split("/")[1],
            "conditions": {k: (None if pd.isna(b.get(k)) else str(b.get(k))) for k in ("species", "cell_type_binary", "compartment", "is_terminal")}
                          | {"soma_distance_um": _num(b.get("soma_distance_start_nm", np.nan) / 1000.0, 1),
                             "cell_type": None if pd.isna(row.get("cell_type")) else str(row.get("cell_type"))},
            "real": {"s_um": np.round(np.asarray(real["s"]) / 1000.0, 3).tolist(), "r_nm": np.round(real["radius"], 1).tolist(),
                     "spine_s_um": np.round(rs / 1000.0, 3).tolist(), "spine_theta": np.round(rt, 3).tolist(),
                     "nn_nm": [_num(x, 1) for x in st_real["nn_surface_nm"]], "line_um": np.round((np.asarray(real["gamma"]) - o_real) / 1000.0, 4).tolist(),
                     "anchors_um": anchors(real, rs, rt, o_real), "mesh": mesh_real, "mesh_kind": real_kind, "metrics": m_real},
            "gen": {"s_um": np.round(np.asarray(gen["s"]) / 1000.0, 3).tolist(), "r_nm": np.round(gen["radius"], 1).tolist(),
                    "spine_s_um": np.round(gs / 1000.0, 3).tolist(), "spine_theta": np.round(gt, 3).tolist(),
                    "nn_nm": [_num(x, 1) for x in st_gen["nn_surface_nm"]], "line_um": np.round((np.asarray(gen["gamma"]) - o_gen) / 1000.0, 4).tolist(),
                    "anchors_um": anchors(gen, gs, gt, o_gen), "mesh": mesh_gen, "mesh_kind": gen_kind, "metrics": m_gen},
        })
        print(f"[view] {n_done}/{len(chosen)}: #{i} {key} ({real_kind} / {gen_kind})", flush=True)

    meta = {"tp_dir": str(args.tp_dir), "assembly": None if args.assembly is None else str(args.assembly),
            "t0_run": info.get("t0_run"), "p0_run": info.get("p0_run"), "mode": info.get("mode"), "counts": info.get("counts"),
            "t0_neuron": info.get("t0_neuron"), "built": f"{_dt.datetime.now():%d.%m.%Y %H:%M}", "n_candidates": len(candidates)}
    payload = {"meta": meta, "items": items, "functionals": FUNCTIONALS}
    out = args.out or ((args.assembly or args.tp_dir) / "view_generated_branches.html")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(PAGE.replace("__PAYLOAD__", json.dumps(payload, ensure_ascii=False, separators=(",", ":"))), encoding="utf-8")
    print(f"[view] {len(items)} branches, {out.stat().st_size / 1e6:.1f} MB -> {out}", flush=True)


PAGE = r"""<!doctype html>
<html lang="ru"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Реальные и сгенерированные ветви</title>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<style>
:root{--ink:#1d1d1f;--muted:#5f6368;--line:#e3e3e6;--pink:#d6418f;--green:#3f8a2a;--bg:#ffffff;--card:#fafafa}
body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.45 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif}
main{max-width:1400px;margin:0 auto;padding:20px 16px 40px}
h1{font-size:22px;margin:4px 0 2px}.sub{color:var(--muted);margin-bottom:14px}
.bar{display:flex;flex-wrap:wrap;gap:12px;align-items:center;margin:10px 0 14px}
select{font:inherit;padding:6px 8px;border:1px solid var(--line);border-radius:8px;max-width:100%}
button{font:inherit;padding:6px 12px;border:1px solid var(--line);border-radius:8px;background:#fff;cursor:pointer}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:14px}
.card{border:1px solid var(--line);border-radius:12px;padding:12px;background:#fff;min-width:0}
.card h3{margin:0 0 6px;font-size:15px}.plot{height:430px}.plot.s{height:340px}
table{border-collapse:collapse;width:100%}td,th{padding:5px 8px;border-bottom:1px solid var(--line);text-align:right}
td:first-child,th:first-child{text-align:left}.real{color:var(--green)}.gen{color:var(--pink)}
.small{font-size:12px;color:var(--muted)}.cond{display:flex;flex-wrap:wrap;gap:6px 14px;margin:2px 0 10px}
.cond span b{font-weight:600}.note{color:var(--muted);font-size:13px;margin-top:6px}
@media (max-width:900px){.grid{grid-template-columns:1fr}}
</style></head><body><main>
<h1>Реальные и сгенерированные ветви</h1>
<div class="sub" id="sub"></div>
<div class="bar"><label for="pick">Ветвь-источник:</label><select id="pick"></select>
<button id="prev">←</button><button id="next">→</button>
<label><input type="checkbox" id="showLine" checked> центральная линия</label>
<label><input type="checkbox" id="showAnchors"> точки крепления шипиков</label></div>
<div class="cond" id="cond"></div>
<div class="grid">
 <div class="card"><h3 class="real">Реальная ветвь</h3><div id="m_real" class="plot"></div><div class="small" id="k_real"></div></div>
 <div class="card"><h3 class="gen">Сгенерированная ветвь</h3><div id="m_gen" class="plot"></div><div class="small" id="k_gen"></div></div>
 <div class="card"><h3>Радиус вдоль ветви</h3><div id="radius" class="plot s"></div></div>
 <div class="card"><h3>Положения шипиков: вдоль ветви и вокруг неё</h3><div id="pos" class="plot s"></div></div>
 <div class="card"><h3>Шипики вдоль ветви (по 2 мкм)</h3><div id="hist" class="plot s"></div></div>
 <div class="card"><h3>Ближайший сосед на поверхности ствола</h3><div id="nn" class="plot s"></div></div>
</div>
<div class="card" style="margin-top:14px"><h3>Метрики ветви</h3><table id="tab"></table>
<div class="note">Сгенерированная ветвь получена по условиям ветви-источника (вид, тип клетки, компартмент, расстояние до сомы, концевая или нет) и с её длиной.
Функционалы ствола считаются так же, как в оценке модуля T; статистики шипиков — как в оценке собранных ветвей.
У реальной ветви учитываются шипики, прошедшие проверку, с определённым положением; у сгенерированной — поставленные при сборке.
Цвет сеток: ствол светлее, шипики насыщеннее (у реальной ветви — сегменты шипиков исходных данных, у сгенерированной — всё, что дальше 40 нм от поверхности сгенерированного ствола).</div></div>
<div class="small" id="foot" style="margin-top:10px"></div>
</main>
<script>
const D = __PAYLOAD__;
const $ = id => document.getElementById(id);
const C = {real: '#3f8a2a', realLight: '#a9c99f', gen: '#d6418f', genLight: '#ecaacb', ink: '#1d1d1f', grid: '#ececef'};
const CFG = {displaylogo: false, responsive: true};
const fmt = (v, d) => (v === null || v === undefined || Number.isNaN(v)) ? '—' : Number(v).toLocaleString('ru-RU', {maximumFractionDigits: d === undefined ? 3 : d});
function dec(b64, T) { const s = atob(b64), u = new Uint8Array(s.length); for (let i = 0; i < s.length; i++) u[i] = s.charCodeAt(i); return new T(u.buffer); }
const cache = new Map();
function mesh(m) {
  if (cache.has(m)) return cache.get(m);
  const v = dec(m.v, Int32Array), f = dec(m.f, Uint32Array), fl = m.flag ? dec(m.flag, Uint8Array) : null;
  const x = [], y = [], z = [], I = [], J = [], K = [];
  for (let i = 0; i < m.nv; i++) { x.push(v[3*i] / 1000); y.push(v[3*i+1] / 1000); z.push(v[3*i+2] / 1000); }
  for (let i = 0; i < m.nf; i++) { I.push(f[3*i]); J.push(f[3*i+1]); K.push(f[3*i+2]); }
  const out = {x, y, z, i: I, j: J, k: K, flag: fl ? Array.from(fl) : null}; cache.set(m, out); return out;
}
const base = (extra) => Object.assign({paper_bgcolor: '#fff', plot_bgcolor: '#fff', font: {family: '-apple-system,Segoe UI,Roboto,sans-serif', size: 12, color: C.ink},
  margin: {l: 52, r: 12, t: 10, b: 44}, legend: {orientation: 'h', y: 1.12, x: 0}, hovermode: 'closest'}, extra || {});
const ax = t => ({title: {text: t}, gridcolor: C.grid, zeroline: false});
function draw3d(div, side, item, light, dark) {
  const b = item[side], m = mesh(b.mesh);
  const tr = [{type: 'mesh3d', x: m.x, y: m.y, z: m.z, i: m.i, j: m.j, k: m.k, hoverinfo: 'skip', name: 'сетка', flatshading: false,
    lighting: {ambient: 0.5, diffuse: 0.8, specular: 0.15, roughness: 0.6}, lightposition: {x: 1000, y: 2000, z: 1500}}];
  if (m.flag) { tr[0].intensity = m.flag; tr[0].colorscale = [[0, light], [1, dark]]; tr[0].showscale = false; tr[0].cmin = 0; tr[0].cmax = 1; }
  else tr[0].color = light;
  if ($('showLine').checked) tr.push({type: 'scatter3d', mode: 'lines', x: b.line_um.map(p => p[0]), y: b.line_um.map(p => p[1]), z: b.line_um.map(p => p[2]),
    line: {color: C.ink, width: 4}, name: 'центральная линия', hoverinfo: 'skip'});
  if ($('showAnchors').checked && b.anchors_um.length) tr.push({type: 'scatter3d', mode: 'markers', x: b.anchors_um.map(p => p[0]), y: b.anchors_um.map(p => p[1]),
    z: b.anchors_um.map(p => p[2]), marker: {size: 3, color: dark}, name: 'точки крепления', hoverinfo: 'skip'});
  Plotly.react(div, tr, base({margin: {l: 0, r: 0, t: 0, b: 0}, showlegend: false,
    scene: {aspectmode: 'data', xaxis: {title: 'x, мкм'}, yaxis: {title: 'y, мкм'}, zaxis: {title: 'z, мкм'}}}), CFG);
}
function show(k) {
  const it = D.items[k]; $('pick').value = k;
  const c = it.conditions, names = {excitatory: 'возбуждающая', inhibitory: 'тормозная', apical: 'апикальный', basal: 'базальный', uncertain: 'не определён',
    inhibitory_dendrite: 'дендрит тормозной клетки', mouse: 'мышь', human: 'человек', True: 'да', False: 'нет', true: 'да', false: 'нет'};
  const nm = v => v === null || v === undefined ? '—' : (names[v] || v);
  $('cond').innerHTML = [['вид', nm(c.species)], ['тип клетки', nm(c.cell_type_binary) + (c.cell_type ? ' (' + c.cell_type + ')' : '')], ['компартмент', nm(c.compartment)],
    ['концевая', nm(c.is_terminal)], ['расстояние до сомы', fmt(c.soma_distance_um, 0) + ' мкм'], ['длина', fmt(it.real.metrics.length_um, 1) + ' мкм'],
    ['нейрон', it.neuron]].map(([a, b]) => `<span>${a}: <b>${b}</b></span>`).join('');
  draw3d('m_real', 'real', it, C.realLight, C.real); draw3d('m_gen', 'gen', it, C.genLight, C.gen);
  $('k_real').textContent = (it.real.mesh_kind === 'raw' ? 'исходная сетка ветви (сегменты данных)' : 'трубка по центральной линии и радиусу (исходная сетка не загружалась)') + ' · ' + it.source;
  $('k_gen').textContent = it.gen.mesh_kind === 'assembled' ? 'собранная ветвь: ствол T0 + шипики (положения P, форма S)' : 'трубка ствола T0, шипики не собраны';
  Plotly.react('radius', [
    {x: it.real.s_um, y: it.real.r_nm, mode: 'lines', name: 'реальная', line: {color: C.real, width: 2.2}},
    {x: it.gen.s_um, y: it.gen.r_nm, mode: 'lines', name: 'сгенерированная', line: {color: C.gen, width: 2.2}}],
    base({xaxis: ax('расстояние вдоль ветви s, мкм'), yaxis: ax('радиус, нм')}), CFG);
  Plotly.react('pos', [
    {x: it.real.spine_s_um, y: it.real.spine_theta, mode: 'markers', name: `реальная (${it.real.spine_s_um.length})`, marker: {color: C.real, size: 8, symbol: 'circle'}},
    {x: it.gen.spine_s_um, y: it.gen.spine_theta, mode: 'markers', name: `сгенерированная (${it.gen.spine_s_um.length})`, marker: {color: C.gen, size: 8, symbol: 'diamond'}}],
    base({xaxis: Object.assign(ax('s, мкм'), {range: [0, Math.max(it.real.metrics.length_um, it.gen.metrics.length_um)]}),
          yaxis: Object.assign(ax('угол вокруг ствола θ, рад'), {range: [-3.3, 3.3]})}), CFG);
  const L = Math.max(it.real.metrics.length_um, it.gen.metrics.length_um);
  const hb = {start: 0, end: Math.ceil(L / 2) * 2, size: 2};
  Plotly.react('hist', [
    {type: 'histogram', x: it.real.spine_s_um, xbins: hb, name: 'реальная', marker: {color: C.real}, opacity: 0.6},
    {type: 'histogram', x: it.gen.spine_s_um, xbins: hb, name: 'сгенерированная', marker: {color: C.gen}, opacity: 0.6}],
    base({barmode: 'overlay', xaxis: ax('s, мкм'), yaxis: ax('шипиков')}), CFG);
  const ecdf = a => { const s = a.filter(v => v !== null).sort((p, q) => p - q); return {x: s, y: s.map((_, i) => (i + 1) / s.length)}; };
  const er = ecdf(it.real.nn_nm), eg = ecdf(it.gen.nn_nm);
  Plotly.react('nn', [
    {x: er.x, y: er.y, mode: 'lines', line: {shape: 'hv', color: C.real, width: 2.2}, name: 'реальная'},
    {x: eg.x, y: eg.y, mode: 'lines', line: {shape: 'hv', color: C.gen, width: 2.2}, name: 'сгенерированная'}],
    base({xaxis: ax('расстояние до ближайшего шипика по поверхности, нм'), yaxis: Object.assign(ax('доля шипиков'), {range: [0, 1.02]})}), CFG);
  const mr = it.real.metrics, mg = it.gen.metrics;
  const rows = [['длина, мкм', 'length_um', 1, 2], ['шипиков', 'n_spines', 1, 0], ['плотность, мкм⁻¹', 'density_per_um', 1, 3],
    ...D.functionals.map(([k, name, f, d]) => [name, k, f, d]),
    ['ближайший сосед на поверхности (медиана), нм', 'nn_surface_median_nm', 1, 0], ['интервал вдоль ветви (медиана), нм', 'gap_median_nm', 1, 0]];
  const v = (m, k, f) => m[k] === null || m[k] === undefined ? null : m[k] * f;
  $('tab').innerHTML = '<tr><th>Метрика</th><th class="real">реальная</th><th class="gen">сгенерированная</th><th>разность</th></tr>' + rows.map(([name, k, f, d]) => {
    const a = v(mr, k, f), b = v(mg, k, f);
    return `<tr><td>${name}</td><td>${fmt(a, d)}</td><td>${fmt(b, d)}</td><td>${a === null || b === null ? '—' : (b - a >= 0 ? '+' : '') + fmt(b - a, d)}</td></tr>`;
  }).join('') + (mr.n_spines_note ? `<tr><td colspan="4" class="small">${mr.n_spines_note}</td></tr>` : '');
}
const sel = $('pick');
D.items.forEach((it, k) => sel.add(new Option(`#${it.index} · ${it.neuron} · ${it.conditions.compartment || '—'} · ${fmt(it.real.metrics.length_um, 1)} мкм · шипиков ${it.real.metrics.n_spines} / ${it.gen.metrics.n_spines}`, k)));
sel.onchange = () => show(+sel.value);
$('prev').onclick = () => show((+sel.value - 1 + D.items.length) % D.items.length);
$('next').onclick = () => show((+sel.value + 1) % D.items.length);
$('showLine').onchange = $('showAnchors').onchange = () => show(+sel.value);
const M = D.meta;
$('sub').textContent = `Набор ${M.tp_dir}` + (M.assembly ? `, сборка ${M.assembly}` : '') + ` · ${D.items.length} ветвей из ${M.n_candidates} · построено ${M.built}`;
$('foot').textContent = `T0: ${M.t0_run || '—'} · P: ${M.p0_run || '—'} · режим ${M.mode || '—'}, число шипиков ${M.counts || '—'}, эффект нейрона T0 ${M.t0_neuron || '—'}. Длины в мкм (пересчёт из нм), радиус в нм.`;
show(0);
</script></body></html>
"""


if __name__ == "__main__":
    main()
