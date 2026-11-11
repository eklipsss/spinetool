"""Section 4 of the global report (global_report/generated.py) on synthetic data: fake generate_tp folders, a fake generated
spine table; a planted difference is reproduced, an absent one is not; equivalence and difference are told apart; the page renders."""

from __future__ import annotations

import json
import re

import numpy as np
import pandas as pd

from src.neuron_model.branch_assembly import load_tp_set
from src.neuron_model.global_report import generated as gen
from src.neuron_model.global_report.significance import save_tables
from tests.neuron_model.test_global_report import METRICS, _build

KIND = "minnie65"
FAST = dict(min_neurons=5, n_permutations=600, n_boot=60, n_boot_difference=300, n_halves=200, seed=0, log=lambda *_: None)


def _trunk(length, radius, rng, step=250.0, wiggle=150.0):
    s = np.arange(0.0, length + 1e-6, step)
    phase = rng.uniform(0, 2 * np.pi)
    gamma = np.stack([s, wiggle * np.sin(2 * np.pi * s / 6000.0 + phase), 0.3 * wiggle * np.cos(2 * np.pi * s / 9000.0 + phase)], axis=1)
    t = np.gradient(gamma, axis=0)
    t /= np.linalg.norm(t, axis=1, keepdims=True)
    n = np.cross(t, [0.0, 0.0, 1.0])
    n /= np.linalg.norm(n, axis=1, keepdims=True)
    b = np.cross(t, n)
    r = radius * np.exp(rng.normal(0, 0.04, len(s)))
    return {"s": s, "gamma": gamma, "radius": r, "tangent": t, "normal": n, "binormal": b, "length": float(s[-1])}


def _write_tp(folder, trunks, rows, placements):
    """A generate_tp.py folder: trunks.npz (packed + offsets), branches.parquet, placements.parquet."""
    folder.mkdir(parents=True, exist_ok=True)
    n = np.array([len(t["s"]) for t in trunks])
    packed = {"offsets": np.r_[0, np.cumsum(n)]}
    for f in ("s", "gamma", "radius", "tangent", "normal", "binormal"):
        packed[f] = np.concatenate([t[f] for t in trunks])
    np.savez_compressed(folder / "trunks.npz", **packed)
    pd.DataFrame(rows).to_parquet(folder / "branches.parquet", index=False)
    pd.DataFrame(placements).to_parquet(folder / "placements.parquet", index=False)


def _tp_world(tmp_path, seed=0):
    """24 source neurons (12 inhibitory BC, 12 excitatory 23P with apical limb 0 / basal limb 1), 8 source branches each.
    Real: inhibitory radius 400 nm vs 250 nm, density 0.1 vs 0.4 per um. Generated: radius follows the source (reproduced),
    density 0.25 per um for every branch (the real difference is NOT reproduced)."""
    rng = np.random.default_rng(seed)
    neurons = [str(864000 + i) for i in range(24)]
    types = {n: ("BC" if i < 12 else "23P") for i, n in enumerate(neurons)}
    meta = {"cell_types": {KIND: types}, "compartments": {KIND: {}}, "layers": {}}
    for n in neurons:
        if types[n] == "23P":
            meta["compartments"][KIND].update({(n, 0): "apical", (n, 1): "basal"})
    source_rows, source_trunks, source_spines = {}, {}, {}
    rows, trunks, placements = [], [], []
    for n in neurons:
        style = rng.normal(0, 0.08)
        inh = types[n] == "BC"
        for k in range(8):
            limb, length = k % 2, float(rng.uniform(15000, 30000))
            key = f"{KIND}/{n}/{limb:03d}/{k}"
            radius = (400.0 if inh else 250.0) * np.exp(style)
            rt = _trunk(length, radius, rng)
            dens = 0.1 if inh else 0.4
            n_real = rng.poisson(dens * length / 1000.0)
            s_real = np.sort(rng.uniform(0, rt["length"], n_real))
            source_trunks[key] = rt
            source_rows[key] = {"train_eligible_T": True, "train_eligible_P": True, "n_spines_in_range": n_real, "n_with_position": n_real}
            source_spines[key] = pd.DataFrame({"s": s_real, "u": s_real / rt["length"]})
            # generated copy of this request
            i = len(trunks)
            gt = _trunk(length, radius * np.exp(rng.normal(0, 0.02)), rng)
            trunks.append(gt)
            n_gen = rng.poisson(0.25 * length / 1000.0)
            s_gen = np.sort(rng.uniform(0, gt["length"], n_gen))
            rows.append({"branch_index": i, "species": "mouse", "cell_type_binary": "inhibitory" if inh else "excitatory", "compartment": None,
                         "is_terminal": True, "soma_distance_start_nm": 1e5, "source_branch": key, "length_nm": gt["length"], "n_spines": int(n_gen)})
            placements += [{"branch_index": i, "spine_index": j, "s": float(x), "u": float(x / gt["length"]), "theta": float(rng.uniform(-np.pi, np.pi))}
                           for j, x in enumerate(s_gen)]
    folder = tmp_path / "2026-11-11_000000_generated_seed1"
    _write_tp(folder, trunks, rows, placements)
    return folder, meta, source_rows, source_trunks, source_spines


def _tp_frame(tmp_path):
    folder, meta, rows, trunks, spines = _tp_world(tmp_path)
    branches, tr, pl = load_tp_set(folder)
    raw = gen.tp_branch_tables(branches, tr, pl, rows, trunks, spines, thin=True, seed=0)
    assert raw["kind"] == KIND and raw["n_skipped"] == 0 and abs(raw["mean_keep_fraction"] - 1.0) < 1e-9
    return gen.tidy_generated_tp(raw, meta, label="T0+P0", min_branches=3)


def _tp_tables(frame):
    from src.neuron_model.branch_statistics import METRIC_INFO

    metrics = [m for m in gen.TP_METRICS if m in frame.columns]
    modules = {lv: [m for m in metrics if METRIC_INFO[m]["module"] == mod] for lv, mod in (("trunk", "T"), ("placement", "P"))}
    abs_metrics = [m for m in metrics if METRIC_INFO[m].get("diff") == "abs"]
    return gen.generated_tables(frame, metrics, level="branch", set_label="T0+P0", modules=modules, abs_metrics=abs_metrics, min_rows=1, **FAST)


def test_tp_adapter_pairs_sides_by_source_neuron(tmp_path):
    frame = _tp_frame(tmp_path)
    assert set(frame["dataset"]) == {"real", "generated"}
    g, r = frame[frame["dataset"] == "generated"], frame[frame["dataset"] == "real"]
    assert set(g["unit_id"]) == set(r["unit_id"]) and all(u.startswith(f"{KIND}:") for u in g["unit_id"])
    # metadata come from the source neuron: compartments only for the excitatory neurons, cell type on both sides
    assert set(g["cell_type_binary"]) == {"inhibitory", "excitatory"} and set(g["compartment"].dropna()) == {"apical", "basal"}
    # the same per-branch definitions on both sides: the generated radius follows the source radius
    m = g.set_index(["unit_id", g["compartment"].fillna("none")])["radius_median_nm"]
    mr = r.set_index(["unit_id", r["compartment"].fillna("none")])["radius_median_nm"]
    assert np.allclose(m.sort_index().to_numpy(), mr.sort_index().to_numpy(), rtol=0.1)


def test_thinning_keeps_the_positioned_share():
    rng = np.random.default_rng(1)
    t = _trunk(20000.0, 300.0, rng)
    branches = pd.DataFrame({"branch_index": [0], "source_branch": [f"{KIND}/1/000/0"]})
    pl = {0: pd.DataFrame({"spine_index": np.arange(400), "s": np.linspace(0, 19000, 400), "theta": 0.0})}
    raw = gen.tp_branch_tables(branches, [t], pl, {f"{KIND}/1/000/0": {"n_spines_in_range": 100, "n_with_position": 50, "train_eligible_P": True}},
                               {f"{KIND}/1/000/0": t}, {}, thin=True, seed=0)
    assert raw["generated_branches"]["n_spines"].iloc[0] == 400          # counts: all spines
    assert 150 < len(raw["generated_spines"]) < 250                       # positions: ~ half kept


def test_tp_reproduction_planted_vs_absent(tmp_path):
    out = _tp_tables(_tp_frame(tmp_path))
    pairs = out["reproduction"]
    ctb = pairs[(pairs["factor"] == "cell_type_binary")].set_index("metric")
    assert ctb.loc["radius_median_nm", "verdict"] == "consistent"            # planted in both
    assert ctb.loc["radius_median_nm", "same_sign"]
    assert ctb.loc["density_per_um", "verdict"] == "real_only"              # real difference not reproduced
    assert ctb.loc["density_per_um", "difference_ci_high"] < 0 or ctb.loc["density_per_um", "difference_ci_low"] > 0
    summ = out["reproduction_summary"]
    p = summ[(summ["factor"] == "cell_type_binary") & (summ["level"] == "placement")].iloc[0]
    assert p["n_present_real"] >= 1 and p["share_reproduced"] < 1.0
    t = summ[(summ["factor"] == "cell_type_binary") & (summ["level"] == "trunk")].iloc[0]
    assert t["n_reproduced_same_sign"] >= 1


def test_tp_equivalence(tmp_path):
    out = _tp_tables(_tp_frame(tmp_path))
    eq = out["equivalence"]
    allu = eq[eq["stratum"] == "all"].set_index("metric")
    assert (allu["pairing"] == "source neuron").all()
    assert allu.loc["radius_median_nm", "verdict"] == "equivalent"          # generated radius = source radius
    assert allu.loc["length_nm", "verdict"] == "equivalent"                # length copied from the request
    inh = eq[(eq["stratum_column"] == "cell_type_binary") & (eq["stratum"] == "inhibitory")].set_index("metric")
    exc = eq[(eq["stratum_column"] == "cell_type_binary") & (eq["stratum"] == "excitatory")].set_index("metric")
    assert inh.loc["density_per_um", "verdict"] == "differs" and inh.loc["density_per_um", "effect"] > 0
    assert exc.loc["density_per_um", "verdict"] == "differs" and exc.loc["density_per_um", "effect"] < 0
    assert {"trunk", "placement"} <= set(eq["module"].dropna())
    s = out["equivalence_summary"]
    assert int(s.loc[s["stratum"] == "all", "n_equivalent"].sum()) >= 2


# ---------------------------------------------------------------------------
# S: the generated spine table of the conditional report
# ---------------------------------------------------------------------------


def _spine_table(rng, n=400):
    """Generated table as written by conditional_generation_report.py: M_cell inhibitory x1.5 (reproduced), M_null inhibitory x1.5
    (spurious: no real difference), M_species x1.6 for everything (differs from the real within a type)."""
    rows = []
    for value in ("excitatory", "inhibitory"):
        for i in range(n):
            row = {"sample_id": f"sample_{i:04d}", "value": value, "label": "g1.5", "block": "cell_type_binary",
                   "condition_cell_type_binary": value, "valid": True}
            for m in METRICS:
                row[m] = float(np.exp(rng.normal(0, 0.34)))
            if value == "inhibitory":
                row["M_cell"] *= 1.5
                row["M_null"] *= 1.5
            row["M_species"] *= 1.6
            rows.append(row)
    rows += [{"sample_id": "u0", "value": "unconditional", "label": "g1.5", "block": "cell_type_binary", **{m: 1.0 for m in METRICS}}]
    return pd.DataFrame(rows)


def _spine_frame(table):
    real = gen.real_reference_spines(_build(), KIND)
    g = gen.tidy_generated_spines(table, METRICS, kind=KIND, label="g1.5", unit_size=20, seed=0)
    return pd.concat([real.assign(set="g1.5"), g], ignore_index=True), g


def test_spine_adapter_pseudo_units_and_conditions():
    table = _spine_table(np.random.default_rng(2))
    frame, g = _spine_frame(table)
    assert len(g) == 800 and (g["dataset"] == "generated").all()               # unconditional row dropped
    assert g.groupby("cell_type_binary")["unit_id"].nunique().to_dict() == {"excitatory": 20, "inhibitory": 20}
    assert g.groupby("unit_id").size().min() == 20
    assert g["unit_mode"].iloc[0].startswith("pseudo-unit")
    # with a neuron_id column the source neuron is the unit (same id space as the real frame)
    t2 = table[table["value"] != "unconditional"].assign(neuron_id=lambda d: ["m" + str(i % 10) for i in range(len(d))])
    g2 = gen.tidy_generated_spines(t2, METRICS, kind=KIND)
    assert set(g2["unit_id"]) <= {f"{KIND}:m{i}" for i in range(10)} and g2["unit_mode"].iloc[0] == "source neuron"


def test_spine_reproduction_and_equivalence():
    frame, _ = _spine_frame(_spine_table(np.random.default_rng(3)))
    out = gen.generated_tables(frame, METRICS, level="spine", set_label="g1.5", min_rows={"default": 5, "generated": 5}, include_all=False, **FAST)
    pairs = out["reproduction"]
    ctb = pairs[pairs["factor"] == "cell_type_binary"].set_index("metric")
    assert ctb.loc["M_cell", "verdict"] in ("consistent", "same_sign_different_size") and ctb.loc["M_cell", "same_sign"]
    assert ctb.loc["M_null", "verdict"] == "generated_only"                 # spurious effect
    assert ctb.loc["M_comp", "verdict"] == "neither"
    eq = out["equivalence"]
    exc = eq[eq["stratum"] == "excitatory"].set_index("metric")
    assert exc.loc["M_species", "verdict"] == "differs"
    assert exc.loc["M_null", "verdict"] == "equivalent"                      # same distribution as the real excitatory spines
    assert exc.loc["M_comp", "verdict"] == "differs"                         # real M_comp carries the apical x1.3 of half the spines
    assert eq[eq["stratum"] == "inhibitory"].set_index("metric").loc["M_null", "verdict"] == "differs"
    assert (eq["pairing"] == "independent").all() and "all" not in set(eq["stratum"])


def test_page_renders_section_4(tmp_path):
    from src.neuron_model.reports.global_report import build_report
    from tests.neuron_model.test_global_report_page import write_tables

    write_tables(tmp_path, n_perm=300)
    frame, _ = _spine_frame(_spine_table(np.random.default_rng(4)))
    sp = gen.generated_tables(frame, METRICS, level="spine", set_label="g1.5", min_rows=5, include_all=False, **FAST)
    tp_frame = _tp_frame(tmp_path / "tp")
    br = _tp_tables(tp_frame)
    tables = {**gen.merge_tables([sp], "spine"), **gen.merge_tables([br], "branch")}
    tables["generated_sets"] = pd.DataFrame(gen.set_inventory(frame, {"set": "g1.5", "source": "x", "source_dataset": KIND, "unit": "pseudo-unit of 20 spines"}, "spine")
                                            + gen.set_inventory(tp_frame, {"set": "T0+P0", "source": "y", "source_dataset": KIND, "unit": "source neuron",
                                                                           "mean_keep_fraction": 0.93}, "branch"))
    save_tables(tables, tmp_path)
    html = build_report(tmp_path).read_text(encoding="utf-8")
    payload = re.search(r'<script type="application/json" id="payload">(.*?)</script>', html, re.S).group(1)

    def no_const(c):
        raise AssertionError(f"non-JSON constant {c} in payload")

    data = json.loads(payload, parse_constant=no_const)
    g = data["generated"]
    assert set(g["levels"]) == {"spine", "trunk", "placement"}
    assert any(r["ver"] == "generated_only" for r in g["levels"]["spine"]["repro"])
    assert {r["m"] for r in g["levels"]["trunk"]["eq"]} <= set(gen.TP_METRICS) and g["levels"]["placement"]["summary"]
    assert "sel-gen-set" in html and "plot-gen-eq" in html and "Таблиц сгенерированных наборов в этой папке нет" not in html


def test_page_keeps_stub_without_generated_tables(tmp_path):
    from src.neuron_model.reports.global_report import build_report
    from tests.neuron_model.test_global_report_page import write_tables

    write_tables(tmp_path, n_perm=200)
    html = build_report(tmp_path).read_text(encoding="utf-8")
    assert "sec-generated" in html and "Таблиц сгенерированных наборов в этой папке нет" in html and re.search(r'"generated":\s*null', html)
