"""Interactive atlas of spine generators: one self-contained HTML (opens offline, no server).

    python scripts/neuron_model/build_spine_atlas.py --vae-run runs/training/neuron-model/<vae_run> \
        --manifest O:/Datasets/Minnie65/preprocessed/manifest.parquet \
        --model VAE=runs/training/neuron-model/<vae_run>/meshes/seed_1 \
        --model MoGen=runs/training/neuron-model/<mogen_run>/meshes/seed_1 \
        [--n-real 1500 --n-gen 600 --n-train 1000 --traversal-spines 4]

What is inside (data: src/neuron_model/spine_generation/atlas.py; page: src/neuron_model/reports/assets/spine_atlas.js;
style: docs/neuron-model/html-materials-style-guide.md):

1. summary tiles and a comparison table of every generated set against the real reference;
2. a map of real and generated spines in the latent space of the VAE encoder (UMAP fitted on real spines) and in the
   space of standardised morphometrics (PCA), coloured by source, cell type, compartment, any morphometric or surface
   quality; a click opens the 3-D mesh and the morphometric profile (percentiles among real spines), a second spine can
   be pinned for side-by-side comparison;
3. distributions of every morphometric per source, the W1 / KS table, correlation matrices (real and the difference);
4. surface quality: validity rates, roughness, needle-like triangles, area, components;
5. latent traversals: meshes along the first principal directions of the real latent codes for a few real spines,
   with the chosen morphometric along the path against the real 10-90 % band.

The real reference is the ``--split`` (default test) of the VAE run's split; ``--n-train`` train spines are encoded only
for the memorisation ratio. Real meshes and morphometrics are read from ``--manifest`` (the NAS manifest: the training
cache holds point clouds only). Windows + GPU: ~10-20 min for the defaults (CGAL morphometrics of generated meshes and
the traversal meshes dominate). Output: ``runs/analysis/neuron-model/atlas/<date>/spine_atlas.html`` (10-20 MB).
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

import os

if sys.platform == "win32":
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch  # noqa: F401,E402  (first on Windows, see fit_t0.py)

import numpy as np
import pandas as pd

from src.neuron_model.reports import html_kit as hk
from src.neuron_model.spine_generation import atlas as A
from src.neuron_model.spine_generation.data.index import load_spine_index
from src.neuron_model.spine_generation.data.splits import load_splits
from src.neuron_model.spine_generation.experiment.config import resolve_path

ASSET = REPO_ROOT / "src/neuron_model/reports/assets/spine_atlas.js"


def _model_arg(value: str):
    name, sep, path = value.partition("=")
    if not sep or not name or not path:
        raise argparse.ArgumentTypeError("--model expects NAME=<seed directory with sample_* folders>")
    return name, Path(path)


def body_html(meta: dict, sources: list, summary_rows: list) -> str:
    gen = [s for s in sources if s["kind"] == "generated"]
    real = next(s for s in sources if s["kind"] == "real")
    tiles = [{"label": "Реальных шипиков в эталоне", "value": f"{real['n']:,}".replace(",", " "), "note": f"сплит {meta['split']}, {meta['n_neurons']} нейронов"}]
    for s in gen:
        r = next((x for x in summary_rows if x["id"] == s["id"]), {})
        v = r.get("valid")
        tiles.append({"label": f"{s['name']}: валидных сеток", "value": "—" if v is None else f"{100 * v:.0f} %",
                      "note": f"среднее W₁/σ по признакам {r['w1']:.2f}" if r.get("w1") is not None else "", "accent": "pink" if s is gen[0] else "green"})
    head = hk.header(meta["title"], meta["subtitle"], meta["meta_lines"])
    summary = '<div class="card" style="margin-top:14px"><div class="card-body"><table class="metrics" id="summaryTable"></table></div></div>'
    intro = hk.note([
        "Атлас собирает в одном месте реальные шипики и шипики, сгенерированные моделями, и позволяет сравнить их тремя способами: "
        "по положению в общем пространстве форм, по распределениям морфометрических признаков и по качеству поверхности. "
        "Все модели сравниваются с одной и той же эталонной выборкой реальных шипиков, которая не участвовала в обучении.",
        "Общее пространство форм задаёт энкодер обученного VAE: каждый шипик, реальный или сгенерированный любой моделью, "
        "переводится в облако точек поверхности и кодируется одним и тем же энкодером. У MoGen собственного энкодера нет, поэтому "
        "другого пространства, общего для всех генераторов, не существует. Двумерная карта строится только по реальным шипикам, "
        "а сгенерированные проецируются на неё, поэтому карта реальных данных не меняется при добавлении новой модели.",
        "**Как читать сводку.** W₁/σ — расстояние Вассерштейна между распределениями признака у модели и у реальных шипиков в "
        "единицах стандартного отклонения реальных; значения меньше 0,2 означают близкие распределения, больше 0,5 — заметное расхождение. "
        "Покрытие показывает, какую долю реальных форм модель воспроизводит хотя бы одним образцом. Точность 1-NNA близка к 0,5, "
        "когда сгенерированные и реальные шипики неразличимы по ближайшему соседу, и к 1, когда они разделимы. Отношение близости "
        "к обучающим шипикам заметно меньше единицы указывает на копирование обучающей выборки.",
    ], title="О чём этот атлас")
    workspace = f"""
<div class="workspace">
 <aside class="sidebar">
  <div class="panel"><h3>Пространство</h3><select id="mapSelect" aria-label="Пространство карты"></select>
   <label class="field" for="colorSelect">Цвет точек</label><select id="colorSelect"></select></div>
  <div class="panel"><h3>Источники</h3><div id="sources"></div>
   <label class="check"><input type="checkbox" id="onlyValid"> только валидные сгенерированные сетки</label></div>
  <div class="panel"><h3>Реальные шипики</h3>
   <label class="field" for="cellTypeSelect">Тип клетки</label><select id="cellTypeSelect"></select>
   <label class="field" for="compartmentSelect">Компартмент</label><select id="compartmentSelect"></select></div>
 </aside>
 <div>
  <div class="grid-2">
   {hk.card("Карта форм", '<div id="map" class="plot l"></div><div id="mapLegend" class="legend"></div><div id="mapCounts" class="small muted"></div>', hint="колесо — масштаб, двойной щелчок — сброс")}
   {hk.card("Выбранный шипик", '<div class="row" style="margin-bottom:6px"><button id="pin">Закрепить для сравнения</button><button id="unpin">Открепить</button></div><div id="viewer" class="plot m"></div><div id="viewerNote" class="small muted"></div><div id="keyLine" class="mono muted" style="margin:6px 0"></div><table class="metrics" id="metricTable"></table>', hint="вращение мышью")}
  </div>
 </div>
</div>"""
    map_note = hk.note([
        "Каждая точка — один шипик. Близкие точки соответствуют шипикам, которые энкодер считает похожими по форме, однако расстояния на "
        "двумерной проекции UMAP передают только соседство, а не величину различий, поэтому сравнивать длины отрезков на карте нельзя. "
        "Вторая карта построена по стандартизованным морфометрическим признакам методом главных компонент; на ней оси и расстояния "
        "сохраняют смысл, но различия формы, не отражённые признаками, на ней не видны.",
        "**Хорошая модель** покрывает те же области карты, что и реальные шипики, с похожей плотностью. Сгенерированные точки, "
        "лежащие вне облака реальных, — это формы, которых в данных нет; пустые области внутри облака реальных точек указывают на "
        "формы, которые модель не воспроизводит. Цвет по морфометрике показывает, как признак меняется по карте: если граница цвета "
        "проходит одинаково у реальных и сгенерированных точек, модель связывает форму и размер так же, как в данных.",
        "Таблица под сеткой показывает значение каждого признака и его перцентиль среди реальных шипиков эталона. Перцентиль ниже 5 "
        "или выше 95 выделен красным: такое значение встречается у реальных шипиков редко. Трёхмерные сетки встроены не для всех точек; "
        "для точки без сетки показывается ближайший на карте шипик того же источника, и это отмечено звёздочкой.",
    ], title="Как читать карту", collapsible=True)
    morph = f"""
<div class="grid-2">
 {hk.card("Распределение признака", '<select id="metricSelect" aria-label="Признак" style="max-width:340px"></select><div id="dist" class="plot m"></div><div id="distAbout" class="small muted"></div>')}
 {hk.card("Корреляции признаков", '<select id="corrModel" aria-label="Модель" style="max-width:340px"></select><div id="corr" class="plot m"></div>')}
</div>
{hk.card("Сравнение распределений по всем признакам", '<div style="overflow:auto"><table class="metrics" id="compareTable"></table></div>')}"""
    morph_note = hk.note([
        "Морфометрические признаки сгенерированных шипиков вычисляются тем же кодом, что и у реальных на этапе предобработки. "
        "Основание сгенерированного шипика неизвестно, поэтому оно находится геометрически, как в процедуре оценки моделей; "
        "признаки, зависящие от основания (длина, площадь основания, угол раскрытия и другие), несут небольшую систематическую "
        "погрешность этого поиска, которую не следует приписывать генератору.",
        "Справа сравнивается структура связей между признаками: слева корреляции у реальных шипиков, справа их разность у выбранной "
        "модели. Модель может верно воспроизводить каждый признак по отдельности и при этом нарушать связи между ними, например "
        "давать длинные шипики с объёмом коротких; такие ошибки видны только на этой карте разностей.",
    ], title="Как читать морфометрику", collapsible=True)
    quality = f"""
<div class="grid-2e">
 {hk.card("Доли валидных сеток", '<div id="qualityRates" class="plot s"></div>')}
 {hk.card("Распределение показателя качества", '<select id="qualitySelect" aria-label="Показатель" style="max-width:340px"></select><div id="qualityBox" class="plot s"></div><div id="qualityAbout" class="small muted"></div>')}
</div>"""
    quality_note = hk.note([
        "Валидной считается замкнутая многообразная сетка из одной компоненты с положительным объёмом, без вырожденных граней и "
        "самопересечений. Для реальных шипиков доли посчитаны по встроенным в атлас запаянным сеткам; для сгенерированных — по "
        "отчёту генератора о каждой сырой сетке.",
        "Шероховатость — средний угол между нормалями соседних треугольников: у гладкой поверхности с мелкими треугольниками он "
        "мал, а рябь и ступеньки поверхности его увеличивают. Доля игольчатых треугольников показывает, насколько сетка пригодна "
        "для дальнейших вычислений: такие треугольники ухудшают оценку кривизны и площади.",
    ], title="Как читать качество поверхности", collapsible=True)
    trav = f"""
<div class="grid-2">
 {hk.card("Сетка на выбранном шаге", '<div class="row"><select id="travBase" aria-label="Базовый шипик" style="max-width:420px"></select><select id="travDir" aria-label="Направление" style="max-width:300px"></select></div><div class="row" style="margin:10px 0"><input type="range" id="travStep" min="0" max="4" step="1" style="max-width:320px"><span id="travStepText" class="mono"></span><button id="travPlay">Пройти</button></div><div id="travView" class="plot m"></div>')}
 {hk.card("Признак вдоль направления", '<select id="travMetric" aria-label="Признак" style="max-width:340px"></select><div id="travCurve" class="plot m"></div><div id="travAbout" class="small muted"></div>')}
</div>"""
    trav_note = hk.note([
        "Латентный код реального шипика сдвигается вдоль одного из главных направлений разброса кодов реальных шипиков, и по каждому "
        "сдвинутому коду декодер строит поверхность. Сдвиг измеряется в стандартных отклонениях кодов вдоль этого направления, "
        "поэтому сдвиг на ±2σ ещё остаётся в пределах, типичных для данных.",
        "Справа показано, как выбранный признак меняется вдоль пути, на фоне диапазона от 10-го до 90-го перцентиля реальных шипиков. "
        "Плавная монотонная кривая означает, что направление кодирует понятное свойство формы; резкие скачки или выход за пределы "
        "реального диапазона при малых сдвигах указывают на участки латентного пространства, где модель ведёт себя ненадёжно.",
    ], title="Как читать латентные направления", collapsible=True)
    return (head + hk.tiles(tiles) + summary + intro
            + hk.section("Карта форм", "Реальные и сгенерированные шипики в общем пространстве; щелчок по точке открывает её сетку.", workspace + map_note, anchor="sec-map")
            + hk.section("Морфометрические признаки", "Насколько распределения признаков и связи между ними у моделей совпадают с реальными.", morph + morph_note, anchor="sec-morph")
            + hk.section("Качество поверхности", "Пригодность сеток для анализа: замкнутость, связность, гладкость.", quality + quality_note, anchor="sec-quality")
            + hk.section("Латентные направления", "Как меняется форма шипика при движении по главным направлениям латентного пространства VAE.", trav + trav_note, anchor="sec-traversal")
            + f'<footer class="page">{hk.esc(meta["footer"])}</footer>')


SUMMARY_JS = r"""
(function () {
  const rows = PAYLOAD.summary, t = $('summaryTable');
  if (!rows.length) { t.innerHTML = '<tr><td>Сгенерированных наборов нет.</td></tr>'; return; }
  const pill = (v, good, bad, invert) => v === null || v === undefined ? '—' :
    `<span class="pill ${(invert ? v > good : v < good) ? 'good' : (invert ? v > bad : v < bad) ? 'warn' : 'bad'}">${fmt(v, 2)}</span>`;
  let h = '<tr><th>Модель</th><th>Валидных сеток</th><th>Среднее W₁/σ</th><th>Худший признак</th><th>Макс. |ΔR|</th><th>Покрытие</th><th>1-NNA</th><th>Близость к обучающим</th></tr>';
  for (const r of rows) {
    h += `<tr><td><span class="swatch" style="background:${r.color};margin-right:8px"></span>${r.name}</td><td>${r.valid === null ? '—' : (100 * r.valid).toFixed(0) + ' %'}</td>`
      + `<td>${pill(r.w1, 0.2, 0.5)}</td><td>${r.worst || '—'}</td><td>${pill(r.max_gap, 0.15, 0.3)}</td><td>${pill(r.coverage, 0.5, 0.3, true)}</td>`
      + `<td>${pill(r.one_nna === null ? null : Math.abs(r.one_nna - 0.5) + 0.5, 0.6, 0.8)}</td><td>${pill(r.memorization_ratio, 2, 0.8, true)}</td></tr>`;
  }
  t.innerHTML = h;
})();
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--vae-run", type=Path, required=True)
    parser.add_argument("--checkpoint", default="latest")
    parser.add_argument("--manifest", type=Path, nargs="+", required=True, help="manifest(s) with real meshes and morphometrics (NAS)")
    parser.add_argument("--splits", type=Path, help="default: data.splits of the VAE run")
    parser.add_argument("--split", default="test")
    parser.add_argument("--n-real", type=int, default=1500)
    parser.add_argument("--real-meshes", type=int, default=150)
    parser.add_argument("--n-train", type=int, default=1000, help="train spines encoded for the memorisation ratio (0 = off)")
    parser.add_argument("--model", type=_model_arg, action="append", default=[], help="NAME=<seed dir with sample_*> (repeatable, up to 3)")
    parser.add_argument("--n-gen", type=int, default=600)
    parser.add_argument("--gen-meshes", type=int, default=120)
    parser.add_argument("--mesh-kind", default="postprocessed", choices=["raw", "postprocessed"])
    parser.add_argument("--traversal-spines", type=int, default=4)
    parser.add_argument("--traversal-dirs", type=int, default=4)
    parser.add_argument("--traversal-steps", default="-2,-1,0,1,2")
    parser.add_argument("--traversal-resolution", type=int, default=96)
    parser.add_argument("--max-faces", type=int, default=2500, help="display meshes are decimated to at most this many faces")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--title", default="Атлас генераторов шипиков")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if len(args.model) > 3:
        parser.error("at most 3 generated sets (the map palette validates three series for scatter plots)")

    vae = A.load_vae(args.vae_run, args.checkpoint)
    d = vae.cfg["data"]
    index = load_spine_index(list(args.manifest), path_mode=d.get("path_mode", "relative"), check_files=())
    splits = load_splits(args.splits or resolve_path(d["splits"]))
    real_index = index.with_split(splits, args.split, drop_unassigned=True)
    if not len(real_index):
        raise SystemExit(f"no spines of split {args.split!r} in the manifest")
    print(f"[atlas] reference: {len(real_index)} spines of split {args.split}; VAE {vae.checkpoint_id}", flush=True)
    real = A.real_records(vae, real_index, args.n_real, seed=args.seed, n_meshes=args.real_meshes, source="real")
    train = []
    if args.n_train > 0:
        train_index = index.with_split(splits, "train", drop_unassigned=True)
        train = A.real_records(vae, train_index, args.n_train, seed=args.seed + 1, n_meshes=0, source="train") if len(train_index) else []
    gens = {name: A.generated_records(vae, path, name, n=args.n_gen, n_meshes=args.gen_meshes, mesh_kind=args.mesh_kind, seed=args.seed) for name, path in args.model}

    # ---- sources and points (real first, then the generated sets)
    colors = list(hk.SERIES)
    sources = [{"id": "real", "name": f"Реальные ({args.split})", "kind": "real", "color": hk.PALETTE["reference"], "n": len(real)}]
    for k, name in enumerate(gens):
        sources.append({"id": name, "name": name, "kind": "generated", "color": colors[k], "n": len(gens[name])})
    records = list(real) + [r for name in gens for r in gens[name]]
    src_index = {s["id"]: i for i, s in enumerate(sources)}

    lat = lambda rs: np.array([r.latent for r in rs if r.latent is not None]) if any(r.latent is not None for r in rs) else np.zeros((0, 1))
    real_lat = lat(real)
    gen_lat = {name: lat(rs) for name, rs in gens.items()}
    xy_real, xy_gens, method = A.fit_map(real_lat, list(gen_lat.values()), method="umap", seed=args.seed)
    xr, ref_stats = A.morph_matrix(real)
    xg = {name: A.morph_matrix(rs, ref_stats)[0] for name, rs in gens.items()}
    pr, pg, _ = A.fit_map(xr, list(xg.values()), method="pca", seed=args.seed)

    def place(rs, coords):
        out, k = [], 0
        for r in rs:
            if r.latent is None:
                out.append(None)
            else:
                out.append([round(float(coords[k][0]), 4), round(float(coords[k][1]), 4)])
                k += 1
        return out

    latent_xy = place(real, xy_real)
    for (name, rs), c in zip(gens.items(), xy_gens):
        latent_xy += place(rs, c)
    morph_xy = [[round(float(a), 4), round(float(b), 4)] for a, b in pr]
    for c in pg:
        morph_xy += [[round(float(a), 4), round(float(b), 4)] for a, b in c]

    meshes, mesh_ref = [], []
    for r in records:
        if r.mesh is not None and len(r.mesh.faces):
            mesh_ref.append(len(meshes))
            meshes.append(hk.pack_mesh(np.asarray(r.mesh.vertices), np.asarray(r.mesh.faces), max_faces=args.max_faces))
        else:
            mesh_ref.append(-1)

    metric_cols = list(A.METRIC_INFO)
    qual_cols = list(A.QUALITY_INFO)
    points = {
        "source": [src_index[r.source] for r in records],
        "key": [r.key for r in records],
        "cell_type": [r.meta.get("cell_type") for r in records],
        "ctb": [r.meta.get("cell_type_binary") for r in records],
        "compartment": [r.meta.get("compartment") for r in records],
        "valid": [r.meta.get("valid") for r in records],
        "mesh": mesh_ref,
        "maps": {"latent": {"name": f"Латентное пространство VAE ({method})", "axis": method, "xy": latent_xy},
                 "morph": {"name": "Морфометрические признаки (PCA)", "axis": "ГК", "xy": morph_xy}},
        "metrics": {m: hk.clean_floats(r.metrics.get(m) for r in records) for m in metric_cols},
        "quality": {q: hk.clean_floats(r.quality.get(q) for r in records) for q in qual_cols},
    }

    # ---- comparison numbers
    comparison, summary, quality_rates = {}, [], {}
    real_vals = {m: np.array([np.nan if r.metrics.get(m) is None else r.metrics[m] for r in real], float) for m in metric_cols}
    train_lat = lat(train) if train else None
    for k, (name, rs) in enumerate(gens.items()):
        mets = {m: A.compare_metric(np.array([np.nan if r.metrics.get(m) is None else r.metrics[m] for r in rs], float), real_vals[m]) for m in metric_cols}
        corr = A.correlation_gap(np.column_stack([np.array([np.nan if r.metrics.get(m) is None else r.metrics[m] for r in rs], float) for m in metric_cols]),
                                 np.column_stack([real_vals[m] for m in metric_cols]))
        latent = A.latent_scores(gen_lat[name], real_lat, train_lat if train_lat is not None and len(train_lat) else None)
        comparison[name] = {"metrics": mets, "corr": corr, "latent": latent}
        w = [(v["w1_rel"], m) for m, v in mets.items() if v["w1_rel"] is not None]
        valid = np.mean([bool(r.meta.get("valid")) for r in rs]) if rs else None
        summary.append({"id": name, "name": name, "color": colors[k], "valid": None if valid is None else float(valid),
                        "w1": float(np.mean([x for x, _ in w])) if w else None, "worst": A.METRIC_INFO[max(w)[1]]["name"] if w else None,
                        "max_gap": corr["max_gap"], **latent})
        present = [r for r in rs if r.quality.get("mesh_present")]
        quality_rates[name] = {"valid": None if valid is None else float(valid),
                               "watertight": float(np.mean([r.quality["watertight"] for r in present])) if present else None,
                               "single": float(np.mean([r.quality.get("n_components") == 1 for r in present])) if present else None,
                               "attachment": float(np.mean([bool(r.meta.get("attachment_found")) for r in present])) if present else None}
    rq = [r for r in real if r.quality.get("mesh_present")]
    quality_rates["real"] = {"valid": float(np.mean([r.quality["watertight"] and r.quality.get("n_components") == 1 for r in rq])) if rq else None,
                             "watertight": float(np.mean([r.quality["watertight"] for r in rq])) if rq else None,
                             "single": float(np.mean([r.quality.get("n_components") == 1 for r in rq])) if rq else None, "attachment": None}

    # ---- latent traversal (meshes appended to the same mesh list)
    steps = [float(x) for x in args.traversal_steps.split(",")]
    traversal_payload = None
    if args.traversal_spines > 0 and len(real_lat) > 2:
        rng = np.random.default_rng(args.seed + 2)
        cand = [r for r in real if r.latent is not None]
        base = [(cand[i].key, cand[i].latent) for i in rng.permutation(len(cand))[: args.traversal_spines]]
        tr = A.traversal(vae, base, real_lat, n_directions=args.traversal_dirs, steps=steps, resolution=args.traversal_resolution)
        frames = []
        for b in range(len(base)):
            per_dir = []
            for dd in range(len(tr["directions"])):
                row = []
                for t in steps:
                    f = tr["frames"][(b, dd, t)]
                    idx = -1
                    if f["mesh"] is not None and len(f["mesh"].faces):
                        idx = len(meshes)
                        meshes.append(hk.pack_mesh(np.asarray(f["mesh"].vertices), np.asarray(f["mesh"].faces), max_faces=args.max_faces))
                    row.append({"mesh": idx, "valid": f["valid"], "metrics": {m: hk.clean_floats([f["metrics"].get(m)])[0] for m in metric_cols}})
                per_dir.append(row)
            frames.append(per_dir)
        traversal_payload = {"directions": tr["directions"], "steps": steps, "base": tr["base"], "frames": frames}

    neurons = {r.meta.get("neuron_id") for r in real if r.meta.get("neuron_id")}
    meta = {
        "title": args.title,
        "subtitle": "Реальные и сгенерированные шипики в общем пространстве форм, их морфометрические признаки, качество поверхности и латентные направления.",
        "meta_lines": [f"VAE: {vae.checkpoint_id}", f"Эталон: {len(real)} шипиков, сплит {args.split}", f"Построено {_dt.datetime.now():%d.%m.%Y %H:%M}"],
        "split": args.split, "n_neurons": len(neurons),
        "footer": (f"Эталон — {len(real)} случайных шипиков сплита {args.split} из {len(neurons)} нейронов; для оценки копирования закодировано "
                   f"{len(train)} обучающих шипиков. Сгенерированные наборы: " + "; ".join(f"{n} — {len(rs)} образцов ({p})" for (n, p), rs in zip(args.model, gens.values()))
                   + f". Сетки в атласе упрощены для показа до {args.max_faces} треугольников; все числа посчитаны по полным сеткам. Длины в мкм, объёмы в мкм³ (пересчёт из нм)."),
    }
    payload = {
        "meta": meta, "sources": sources, "points": points, "meshes": meshes, "summary": summary, "comparison": comparison, "quality_rates": quality_rates,
        "metric_info": A.METRIC_INFO, "quality_info": A.QUALITY_INFO, "traversal": traversal_payload,
        "categories": {"cell_type": sorted({v for v in points["cell_type"] if v}), "compartment": sorted({v for v in points["compartment"] if v})},
    }
    out = args.out or REPO_ROOT / "runs/analysis/neuron-model/atlas" / f"{_dt.datetime.now():%Y-%m-%d_%H%M%S}" / "spine_atlas.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    page_js = ASSET.read_text(encoding="utf-8") + "\n" + SUMMARY_JS
    out.write_text(hk.page(args.title, body_html(meta, sources, summary), payload, page_js), encoding="utf-8")
    (out.parent / "atlas_summary.json").write_text(json.dumps({"summary": summary, "quality_rates": quality_rates}, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"[atlas] {len(records)} spines, {len(meshes)} meshes, {out.stat().st_size / 1e6:.1f} MB -> {out}", flush=True)


if __name__ == "__main__":
    main()
