"""Interactive page with whole neurons and their compartments (for looking and for report figures).

What is on the page (data: src/neuron_model/compartment_viewer.py; page: src/neuron_model/reports/assets/neuron_compartments.js):
- whole neurons: every branch skeleton coloured by the compartment of its limb - our labels (direction rule + thin-long
  axon of interneurons) or the colleague's length-rule table - or by the median radius of its branch; soma mesh, the
  direction to the pia, the initial direction of every limb (with --features), presynaptic / postsynaptic points of the
  synapse check (validate_axon_rule.py); a table of the limbs under the plot;
- close-ups of single branches: branch mesh + the meshes of its segmented "spines" + the synapse points on it - axon
  branches carrying presynapses AND segmented spines (are those boutons?), and a spiny dendrite for comparison.

Neurons chosen automatically (unless --neurons): excitatory neurons where the colleague's "apical" limb points away from
the pia (disputed), one neuron where both rules agree, the inhibitory neurons best covered by presynapses, inhibitory
neurons whose thin-long limb has no presynapse but some postsynapses. Reads the NAS: skeletons, soma meshes (~10-60 s per
neuron) and a few branch / spine meshes.

    python scripts\\neuron_model\\view_neuron_compartments.py ^
        --limbs runs\\analysis\\neuron-model\\2026-11-10_apical-by-direction-minnie65-inhibitory\\limbs.csv ^
        --compartment-table runs\\analysis\\neuron-model\\2026-11-10_apical-by-direction-minnie65-inhibitory\\limb_compartments_direction.parquet ^
        --features runs\\analysis\\neuron-model\\2026-11-02_195824_apical-by-direction-minnie65\\limb_features.parquet ^
        --points runs\\analysis\\neuron-model\\2026-11-10_030133_axon-rule-check\\points.parquet ^
        --branches C:\\Users\\Student\\LipsEkaterina\\data\\minnie65\\preprocessed_branches\\branches.parquet ^
        --dataset-root O:\\Datasets\\Minnie65

Writes ``runs/analysis/neuron-model/<date>_neuron-compartments.html`` (or --out).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

NO_APICAL = {"BC", "BPC", "MC", "NGC", "Unsure I"}


def our_labels(limbs: pd.DataFrame, table_path: Path | None, cell_type: dict) -> pd.Series:
    """Our compartment per limb: from the compartment table when given, else rebuilt from limbs.csv (direction labels,
    axons = table OR branch rule, interneurons by the thin-long rule)."""
    if table_path is not None:
        t = pd.read_parquet(table_path)
        m = dict(zip(t["limb"].astype(str), t["compartment"].astype(str)))
        return limbs["limb"].astype(str).map(m).fillna("not_applicable")
    from src.neuron_model.compartments import label_no_apical_neurons

    l = limbs.copy()
    ax = l["axon_like_table"].astype("boolean").fillna(False).astype(bool) | l["axon_like_branches"].astype("boolean").fillna(False).astype(bool)
    l["compartment_dir"] = l["compartment_dir"].where(~ax, "axon_like")
    l["axon_like"] = ax
    applicable = {n: cell_type.get(n) not in NO_APICAL for n in l["neuron_id"].unique()}
    return label_no_apical_neurons(l, applicable, cell_type=cell_type)["compartment_dir"]


def main() -> None:
    from src.neuron_model.compartment_viewer import branch_radius_map, neuron_payload, spine_example_payload
    from src.neuron_model.reports.html_kit import card, header, note, page, section, tiles

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--limbs", type=Path, required=True, help="limbs.csv of apical_by_direction.py")
    ap.add_argument("--compartment-table", type=Path, help="limb_compartments_direction.parquet (our labels); default: rebuilt from --limbs")
    ap.add_argument("--features", type=Path, help="limb_features.parquet (root and initial direction of every limb -> arrows)")
    ap.add_argument("--points", type=Path, required=True, help="points.parquet of validate_axon_rule.py")
    ap.add_argument("--branches", type=Path, required=True, help="branches.parquet of the branch preprocessing (radius of every branch)")
    ap.add_argument("--dataset-root", type=Path, required=True)
    ap.add_argument("--metadata-root", type=Path, default=REPO_ROOT / "datasets")
    ap.add_argument("--neurons", nargs="*", help="these neurons instead of the automatic choice")
    ap.add_argument("--n-disputed", type=int, default=3)
    ap.add_argument("--n-inhibitory", type=int, default=2)
    ap.add_argument("--n-examples", type=int, default=4)
    ap.add_argument("--pia-axis", default="0,-1,0")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    t0 = time.time()

    limbs = pd.read_csv(args.limbs)
    limbs["neuron_id"] = limbs["neuron_id"].astype(str)
    if "limb" not in limbs.columns:
        limbs["limb"] = limbs["neuron_id"] + "_limb_" + limbs["limb_id"].map("{:03d}".format)
    types = pd.read_csv(args.metadata_root / "minnie65" / "cell_types_with_nucleus.csv")
    cell_type = dict(zip(types["segment_id"].astype(str), types["cell_type"]))
    limbs["ours"] = our_labels(limbs, args.compartment_table, cell_type).to_numpy()
    colleague = pd.read_parquet(args.metadata_root / "minnie65" / "minnie65_limb_compartments_length_rule.parquet", columns=["limb", "compartment"])
    limbs["table"] = limbs["limb"].map(dict(zip(colleague["limb"].astype(str), colleague["compartment"].astype(str))))
    points = pd.read_parquet(args.points)
    points["neuron_id"] = points["neuron_id"].astype(str)
    branches = pd.read_parquet(args.branches, columns=["neuron_id", "limb_id", "member_branch_ids", "radius_median_nm"])
    features = None
    if args.features:
        features = pd.read_parquet(args.features, columns=["neuron_id", "limb_id", "root", "dir_init"])
        features["neuron_id"] = features["neuron_id"].astype(str)
    pia = [float(v) for v in args.pia_axis.split(",")]
    near = points[points["dist_nm"] <= 3000]
    npre = near[near["kind"] == "pre"].groupby("neuron_id").size()
    npost = near[near["kind"].isin(["post_shaft", "post_spine"])].groupby("neuron_id").size()
    is_inh = limbs["neuron_id"].map(cell_type).isin(NO_APICAL)

    # ---- which neurons
    chosen: list[tuple[str, str, str]] = []  # (neuron, group, why)
    if args.neurons:
        chosen = [(n, "выбран вручную", "") for n in args.neurons]
    else:
        disputed = limbs[(limbs["table"] == "apical") & (limbs["cos_pia"] < 0) & ~is_inh]
        disputed = disputed.assign(post=disputed["neuron_id"].map(npost).fillna(0)).sort_values(["post", "cable_nm"], ascending=False)
        for _, r in disputed.head(args.n_disputed).iterrows():
            chosen.append((r["neuron_id"], "спорный", f"Таблица коллеги называет апикальным лимб {int(r['limb_id'])}: он направлен вглубь (cos к пиа {r['cos_pia']:.2f}), "
                           f"медианный радиус ветвей {r['limb_radius_nm']:.0f} нм, кабель {r['cable_nm'] / 1e6:.1f} мм, плотность шипиков {r['limb_density_per_um']:.2f} на мкм."))
        agree = limbs[(limbs["table"] == "apical") & (limbs["ours"] == "apical") & (limbs["cos_pia"] > 0.9)]
        agree = agree.assign(post=agree["neuron_id"].map(npost).fillna(0)).sort_values("post", ascending=False)
        for _, r in agree.head(1).iterrows():
            chosen.append((r["neuron_id"], "согласованный", f"Оба правила выбирают апикальным лимб {int(r['limb_id'])} (cos к пиа {r['cos_pia']:.2f})."))
        inh_ids = sorted(set(limbs.loc[is_inh, "neuron_id"]), key=lambda n: -npre.get(n, 0))
        for n in inh_ids[: args.n_inhibitory]:
            chosen.append((n, "тормозный, пресинапсы", f"Тормозный нейрон с полной разметкой пресинапсов: {int(npre.get(n, 0))} пресинапсов у скелетов."))
        thin = limbs[is_inh & (limbs["ours"] == "axon_like")]
        thin = thin[thin["neuron_id"].map(npre).fillna(0).eq(0) & thin["neuron_id"].map(npost).fillna(0).ge(3)]
        for n in list(dict.fromkeys(thin["neuron_id"]))[:2]:
            chosen.append((n, "тормозный, без пресинапсов", "Тормозный нейрон без размеченных пресинапсов: аксон выделен только по толщине и длине лимба, "
                           f"на нейроне {int(npost.get(n, 0))} постсинапсов."))
    seen = set()
    chosen = [c for c in chosen if not (c[0] in seen or seen.add(c[0]))]

    neurons = []
    for n, group, why in chosen:
        d = args.dataset_root / n
        if not (d / "soma" / "soma_mesh.off").exists():
            print(f"[view] skip {n}: no soma mesh under {d}", flush=True)
            continue
        li = limbs[limbs["neuron_id"] == n].sort_values("limb_id")
        p = neuron_payload(d, neuron_id=n, cell_type=cell_type.get(n), ours=dict(zip(li["limb_id"].astype(int), li["ours"])),
                           table={int(k): v for k, v in zip(li["limb_id"], li["table"]) if isinstance(v, str)}, limb_info=li,
                           radius=branch_radius_map(branches, n), points=points, pia=pia, features=features, why=why)
        p["group"] = group
        neurons.append(p)
        print(f"[view] {group}: {n} ({time.time() - t0:.0f} s)", flush=True)

    # ---- close-ups: axon branches with presynapses and segmented spines, and a spiny dendrite
    label_of = dict(zip(zip(limbs["neuron_id"], limbs["limb_id"].astype(int)), limbs["ours"]))
    bp =near[near["branch_id"] >= 0].groupby(["neuron_id", "limb_id", "branch_id"]).agg(
        pre=("kind", lambda s: int((s == "pre").sum())), post=("kind", lambda s: int(s.isin(["post_shaft", "post_spine"]).sum())),
        spines=("branch_spines", "max")).reset_index()
    bp["label"] = [label_of.get((n, int(l))) for n, l in zip(bp["neuron_id"], bp["limb_id"])]
    if args.neurons:
        bp = bp[bp["neuron_id"].isin(set(args.neurons))]
    ax = bp[(bp["label"] == "axon_like") & (bp["spines"] > 0) & (bp["pre"] >= 3)]
    ax = ax.assign(inh=ax["neuron_id"].map(cell_type).isin(NO_APICAL)).sort_values(["inh", "pre"], ascending=False)
    picks = []
    for inh in (True, False):
        sub = ax[ax["inh"] == inh].drop_duplicates("neuron_id")
        picks += [(r, "аксонная ветвь с «шипиками»") for _, r in sub.head((args.n_examples + 1) // 2 if inh else args.n_examples // 2).iterrows()]
    den = bp[bp["label"].isin(["basal", "apical"]) & (bp["spines"] > 0) & (bp["post"] >= 3)].sort_values("post", ascending=False)
    picks += [(r, "дендрит с шипиками (для сравнения)") for _, r in den.head(1).iterrows()]
    examples = []
    for r, title in picks:
        n, limb, br = r["neuron_id"], int(r["limb_id"]), int(r["branch_id"])
        rm = branch_radius_map(branches[branches["neuron_id"].astype(str) == n], n)
        try:
            examples.append(spine_example_payload(args.dataset_root / n, neuron_id=n, limb=limb, branch=br, points=near, title=title,
                                                  info={"cell_type": cell_type.get(n), "radius_nm": rm.get((limb, br)), "label": label_of.get((n, limb))}))
            print(f"[view] close-up {title}: {n} limb {limb} branch {br} ({int(r['pre'])} pre, {int(r['post'])} post, {int(r['spines'])} spines)", flush=True)
        except FileNotFoundError as e:
            print(f"[view] skip close-up {n}/{limb}/{br}: {e}", flush=True)

    n_inh = sum(1 for n in neurons if n["cell_type"] in NO_APICAL)
    body = [
        header("Компартменты нейронов", "Апикальный и базальные дендриты, аксон и синапсы на целых нейронах Minnie65",
               [f"нейронов: {len(neurons)}", f"крупных планов ветвей: {len(examples)}", time.strftime("%Y-%m-%d")]),
        tiles([{"label": "Нейронов на странице", "value": str(len(neurons))}, {"label": "Из них тормозных", "value": str(n_inh), "accent": "green"},
               {"label": "Крупных планов ветвей", "value": str(len(examples)), "accent": "pink"}]),
        note(["Страница показывает целые нейроны с разметкой лимбов, по которой отбираются обучающие ветви модулей T и P. "
              "Каждая ветвь нарисована линией по своему скелету и окрашена по компартменту своего лимба. Разметку можно переключить "
              "на таблицу коллеги (правило по длине) или на радиус ветви. Точки — синапсы из ручной разметки MICrONS: чёрные ромбы — "
              "пресинапсы (по биологии лежат на аксоне), серые кольца — постсинапсы (лежат на дендритах).",
              "**Метод и проверка описаны в docs/neuron-model/compartment-by-direction.md** (разделы 4 и 9). Пунктирная линия от сомы — "
              "направление на пиа, толстые короткие отрезки у сомы — начальные направления лимбов (10–30 мкм пути), по которым выбирается апикальный дендрит."],
             title="Что на странице"),
        section("Нейроны", "Выберите нейрон и способ окраски; вращение — мышью, колесо — масштаб, двойной щелчок — исходный вид.",
                card("Нейрон", '<div class="row" style="margin-bottom:10px"><select id="neuron-select" style="max-width:520px"></select>'
                     '<div class="seg" id="mode-seg"><button data-mode="ours" aria-pressed="true">Наша разметка</button>'
                     '<button data-mode="table" aria-pressed="false">Таблица коллеги</button><button data-mode="radius" aria-pressed="false">Радиус ветви</button></div>'
                     '<label class="check"><input type="checkbox" id="show-pre" checked>пресинапсы</label>'
                     '<label class="check"><input type="checkbox" id="show-post" checked>постсинапсы</label>'
                     '<label class="check"><input type="checkbox" id="show-arrows" checked>начальные направления</label></div>'
                     '<div class="legend" id="label-legend"></div><p class="muted" id="neuron-title"></p><p id="neuron-why"></p>'
                     '<div id="plot-neuron" class="plot l"></div><table class="metrics" id="limb-table"></table>', hint="мкм от центра сомы")
                + note(["Апикальный дендрит — розовый, базальные дендриты и дендриты тормозных нейронов — зелёные (в одном нейроне они не встречаются), "
                        "аксон — фиолетовый, «не определён» — серый, лимбы без метки — светло-серые. В режиме «Радиус ветви» цвет — медианный "
                        "радиус объединённой ветви (светлый — тонкая, тёмный — толстая, шкала 80–450 нм): аксон тоньше 180 нм, дендриты около 300 нм.",
                        "Спорные нейроны — те, где таблица коллеги назвала апикальным лимб, направленный вглубь: по нашему разбору это аксоны. "
                        "У тормозных нейронов аксон — единственный тонкий и длинный лимб; у нейронов с размеченными пресинапсами почти все они должны лежать на нём."],
                       title="Как читать", collapsible=True), anchor="sec-neurons"),
        section("Ветви крупным планом", "Аксонные ветви, на которых сегментация нашла «шипики», и дендрит с шипиками для сравнения.",
                card("Ветвь", '<div class="row" style="margin-bottom:10px"><select id="example-select" style="max-width:640px"></select></div>'
                     '<p id="example-why"></p><div id="plot-example" class="plot l"></div>', hint="мкм от центра ветви")
                + note(["Серая полупрозрачная поверхность — сетка ветви, розовые — сетки «шипиков», выделенных сегментацией, чёрные ромбы и серые кольца — "
                        "пре- и постсинапсы на этой ветви. Внутри аксонных лимбов пресинапсы чаще лежат на ветвях, где сегментация нашла шипики, "
                        "чем на ветвях без них; здесь можно посмотреть, что это за выросты: бутоны аксона или настоящие шипики.",
                        "Сетки упрощены для показа (ветвь до 25 000 треугольников, шипик до 400), все числа посчитаны по полным данным."],
                       title="Как читать", collapsible=True), anchor="sec-examples"),
        '<footer class="muted small">Данные: скелеты ветвей и сетки Minnie65 (NAS), разметка лимбов — '
        f'{args.compartment_table or args.limbs}, синапсы — {args.points}, радиус ветвей — {args.branches}. Координаты в мкм.</footer>',
    ]
    js = (REPO_ROOT / "src/neuron_model/reports/assets/neuron_compartments.js").read_text(encoding="utf-8")
    out = args.out or REPO_ROOT / "runs/analysis/neuron-model" / f"{time.strftime('%Y-%m-%d')}_neuron-compartments.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(page("Компартменты нейронов", "".join(body), {"neurons": neurons, "examples": examples}, js), encoding="utf-8")
    print(f"-> {out} ({out.stat().st_size / 1e6:.1f} MB, {time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
