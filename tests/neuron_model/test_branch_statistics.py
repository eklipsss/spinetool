"""Branch / placement statistics per group (src/neuron_model/branch_statistics.py)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.neuron_model.branch_statistics import expected_pairs, group_spatial_stats, group_table


def _spines(rng, n_branches=300, length=30000.0, mode="poisson", density=0.8e-3, mark_mode="independent"):
    rows, lengths = [], {}
    for b in range(n_branches):
        n = rng.poisson(density * length)
        if mode == "poisson":
            s = rng.uniform(0, length, n)
        else:  # regular: hard-core 700 nm via a jittered lattice
            step = length / max(n, 1)
            s = (np.arange(n) + 0.5) * step + rng.uniform(-0.15, 0.15, n) * step
        s = np.sort(s)
        if mark_mode == "independent":
            vol = rng.normal(0, 1, n)
        else:  # smooth along the branch: neighbours similar
            vol = np.sin(s / 2000.0 + rng.uniform(0, 6)) * 2 + rng.normal(0, 0.3, n)
        lengths[b] = length
        for i in range(n):
            rows.append({"branch": b, "s": s[i], "u": s[i] / length, "log_volume": vol[i] + 0.5 * b % 3, "length": vol[i]})
    return pd.DataFrame(rows), lengths


def test_expected_pairs_matches_monte_carlo():
    rng = np.random.default_rng(0)
    L, n = 10000.0, 12
    sims = []
    for _ in range(3000):
        s = rng.uniform(0, L, n)
        d = np.abs(s[:, None] - s[None, :])[~np.eye(n, dtype=bool)]
        sims.append(((d >= 500) & (d < 1000)).sum())
    assert expected_pairs(np.array([L]), np.array([n]), 500.0, 1000.0) == pytest.approx(np.mean(sims), rel=0.03)


def test_pcf_is_one_for_random_positions_and_below_one_for_regular():
    rng = np.random.default_rng(1)
    poisson = group_spatial_stats(*_spines(rng))
    regular = group_spatial_stats(*_spines(rng, mode="regular"))
    for k in ("pcf_0_500", "pcf_500_1000", "pcf_1000_2000"):
        assert poisson[k] == pytest.approx(1.0, abs=0.06)
    assert regular["pcf_0_500"] < 0.3 and regular["gap_cv"] < 0.4 < poisson["gap_cv"]
    assert poisson["gap_cv"] == pytest.approx(1.0, abs=0.1)


def test_mark_correlation_sees_local_similarity_not_the_branch_level():
    rng = np.random.default_rng(2)
    independent = group_spatial_stats(*_spines(rng))
    smooth = group_spatial_stats(*_spines(rng, mark_mode="smooth"))
    assert abs(independent["nn_corr_log_volume"]) < 0.08  # the branch offsets (0.5 * b % 3) are removed by centring
    assert smooth["nn_corr_log_volume"] > 0.5


def test_mark_correlation_is_not_biased_on_branches_with_few_spines():
    # 3-4 spines per branch, independent marks: centring by the branch mean alone gives about -1/(n-1)
    rng = np.random.default_rng(4)
    few = group_spatial_stats(*_spines(rng, n_branches=1500, length=4000.0))
    assert abs(few["nn_corr_log_volume"]) < 0.05 and abs(few["nn_corr_length"]) < 0.05
    raw = group_spatial_stats(*_spines(np.random.default_rng(4), n_branches=1500, length=4000.0), n_null=0)
    assert raw["nn_corr_log_volume"] < -0.15  # the bias the permutation removes


def test_group_table_rows_and_eligibility():
    rng = np.random.default_rng(3)
    sp, lengths = _spines(rng, n_branches=40)
    branches = pd.DataFrame({
        "neuron_id": ["n1"] * 20 + ["n2"] * 20, "limb_id": [0] * 40, "merged_branch_id": list(range(40)), "status": ["ok"] * 40,
        "length_nm": [30000.0] * 40, "radius_median_nm": rng.uniform(200, 400, 40), "radius_proximal_nm": 300.0, "radius_distal_nm": 250.0,
        "ellipse_ratio_median": 0.7, "tortuosity": 1.05, "kappa_rms": 2e-4, "n_spines": sp.groupby("branch").size().reindex(range(40), fill_value=0).to_numpy(),
        "train_eligible_T": True, "train_eligible_P": [True] * 25 + [False] * 15, "cell_type": ["23P"] * 20 + ["BC"] * 20,
        "cell_type_binary": ["excitatory"] * 20 + ["inhibitory"] * 20, "compartment": ["basal"] * 20 + [None] * 20,
    })
    sp = sp.assign(neuron_id=np.where(sp["branch"] < 20, "n1", "n2"), limb_id=0, merged_branch_id=sp["branch"])
    table = group_table(branches, sp, min_branches=10, min_pairs=10)
    assert set(table["neuron_id"]) == {"n1", "n2"} and set(table["compartment"]) == {"basal", "none"}
    n1 = table[table.neuron_id == "n1"].iloc[0]
    assert n1["radius_ratio"] == pytest.approx(1.2) and np.isfinite(n1["pcf_0_500"])
    n2 = table[table.neuron_id == "n2"].iloc[0]
    assert np.isnan(n2["density_per_um"]) and np.isfinite(n2["radius_median_nm"])  # 5 P-eligible branches < min_branches; T: all 20


def test_branch_profile_report_builds(tmp_path):
    import json

    from src.neuron_model.reports.morphology_report import build_payload, build_report
    from src.neuron_model.spine_generation.evaluation.morphology_by_condition import morphology_by_condition, neuron_medians

    rng = np.random.default_rng(4)
    rows = []
    for n in range(60):
        inh = n % 3 == 0
        for comp in (["none"] if inh else ["apical", "basal"]):
            rows.append({"neuron_id": f"n{n}", "compartment": comp, "cell_type": "BC" if inh else ("23P" if n % 2 else "4P"),
                         "cell_type_binary": "inhibitory" if inh else "excitatory", "radius_median_nm": rng.normal(200 if inh else 300, 20),
                         "share_empty": rng.uniform(0.4, 0.6) if inh else rng.uniform(0, 0.2), "pcf_0_500": rng.normal(1.0 if inh else 0.4, 0.1)})
    groups = pd.DataFrame(rows)
    metrics = ["radius_median_nm", "share_empty", "pcf_0_500"]
    tables = morphology_by_condition(groups, metrics, min_neurons=10, min_spines=1, n_permutations=500, log=lambda *_: None)
    for name, table in tables.items():
        table.to_csv(tmp_path / f"{name}.csv", index=False)
    info = groups.groupby("neuron_id")[["cell_type", "cell_type_binary"]].first()
    neuron_medians(groups, metrics, ["neuron_id"], 1).merge(info, on="neuron_id").to_csv(tmp_path / "neuron_medians.csv", index=False)
    cm = groups[groups.compartment != "none"]
    neuron_medians(cm, metrics, ["neuron_id", "compartment"], 1).merge(info, on="neuron_id").to_csv(tmp_path / "neuron_compartment_medians.csv", index=False)
    (tmp_path / "run_info.json").write_text(json.dumps({"profile": "branches", "n_neurons": 60, "n_branches": 999}))
    payload = build_payload(tmp_path)
    by_id = {m["id"]: m for m in payload["metrics"]}
    assert by_id["share_empty"]["abs"] and not by_id["radius_median_nm"]["abs"]
    i = [m["id"] for m in payload["metrics"]].index("pcf_0_500")
    cell = payload["cells"]["binary_all"][i]
    assert cell["sig"] and cell["rel"] == pytest.approx(0.6, abs=0.1)  # absolute difference of g, not a ratio
    html = build_report(tmp_path).read_text(encoding="utf-8")
    assert "Ствол и расположение шипиков" in html


def test_join_spine_morphometrics_by_original_branch_and_string_limbs():
    import numpy as np
    import pandas as pd

    from src.neuron_model.branch_statistics import join_spine_morphometrics

    spines = pd.DataFrame({"neuron_id": ["n1", "n1", "n1"], "limb_id": [3, 3, 3], "merged_branch_id": [0, 0, 0],
                           "original_branch_id": [1, 2, 2], "spine_id": ["5", "5", "6"], "s": [1.0, 2.0, 3.0]})
    morph = pd.DataFrame({"neuron_id": ["n1", "n1", "n1"], "limb_id": ["limb_003", "limb_003", "limb_003"], "branch_id": ["branch_001", "2", "2"],
                          "spine_id": ["5", "5", "6"], "status": ["success", "success", "failed"], "Volume": [8.0, 27.0, 1.0], "Length": [100.0, 200.0, 300.0]})
    out = join_spine_morphometrics(spines, morph)
    assert len(out) == 3 and np.allclose(out["log_volume"].iloc[:2], np.log([8.0, 27.0]))
    assert out["length"].iloc[:2].tolist() == [100.0, 200.0]
    assert np.isnan(out["log_volume"].iloc[2])          # stage 8 failed -> no mark, the spine still counts for placement
