"""Statistics of the conditional-generation report (evaluation/conditional_tests.py) on synthetic data with a neuron structure:
real spines = neuron random effect (style) + a shared size factor + a type shift in Volume and Length only; generated sets =
a faithful generator (fresh style per spine = the marginal mixture), a condition-blind one and an over-amplified one."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.neuron_model.spine_generation.evaluation import conditional_tests as ct

METRICS = ["Volume", "Length", "Area", "ConvexHullVolume", "CVD"]
MU = np.log([5e7, 1000.0, 2e6, 9e7, 0.5])
LOAD = np.array([0.5, 0.25, 0.4, 0.5, 0.0])
SHIFT = {"inhibitory": np.log([0.7, 0.8, 1.0, 1.0, 1.0]), "excitatory": np.zeros(5)}
NOISE = np.array([0.4, 0.25, 0.35, 0.4, 0.3])


def _spines(rng, n, kind, style, amplitude=1.0):
    """``style``: [n, 5] neuron effects (the same row repeated for the spines of one neuron)."""
    z = rng.normal(size=(n, 1))
    log = MU + style + LOAD * z + rng.normal(size=(n, 5)) * NOISE + amplitude * SHIFT[kind]
    return np.exp(log)


def make_real(seed=0, n_neurons=120, spines=80, style_sd=0.2, inhibitory_share=0.3):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_neurons):
        kind = "inhibitory" if i < int(n_neurons * inhibitory_share) else "excitatory"
        x = _spines(rng, spines, kind, rng.normal(0, style_sd, 5)[None, :])
        f = pd.DataFrame(x, columns=METRICS)
        f["neuron_id"], f["cell_type_binary"] = f"n{i}", kind
        rows.append(f)
    return pd.concat(rows, ignore_index=True)


def make_gen(seed, n, kind, style_sd=0.2, amplitude=1.0, shift_kind=None):
    rng = np.random.default_rng(seed)
    x = _spines(rng, n, shift_kind or kind, rng.normal(0, style_sd, (n, 5)), amplitude)
    f = pd.DataFrame(x, columns=METRICS)
    f["mesh_present"], f["attachment_found"], f["valid"] = True, True, True
    return f


@pytest.fixture(scope="module")
def real():
    return make_real()


def _real_effect(real):
    return ct.type_effect_real(real, "cell_type_binary", "inhibitory", "excitatory", METRICS, n_permutations=2000, n_boot=300, seed=0)


def test_real_effect_found_only_where_built_in(real):
    tbl, boots = _real_effect(real)
    t = tbl.set_index("metric")
    assert t.loc["Volume", "significant"] and t.loc["Length", "significant"]
    assert not t.loc[["Area", "ConvexHullVolume", "CVD"], "significant"].any()
    assert t.loc["Volume", "log_ratio"] == pytest.approx(np.log(0.7), abs=0.12)
    assert t.loc["Volume", "lr_lo"] < t.loc["Volume", "log_ratio"] < t.loc["Volume", "lr_hi"]


def test_concordance_faithful_blind_and_over_amplified(real):
    rtbl, rb = _real_effect(real)
    out = {}
    for name, kw in {"faithful": {}, "blind": {"shift_kind": "excitatory"}, "over": {"amplitude": 2.5}}.items():
        seeds = {"faithful": (1, 2), "blind": (21, 22), "over": (31, 32)}[name]
        ga = make_gen(seeds[0], 600, "inhibitory", **kw)
        gb = make_gen(seeds[1], 600, "excitatory", **({"amplitude": 2.5} if name == "over" else {}))
        gtbl, gb_boots = ct.type_effect_generated(ga, gb, METRICS, n_permutations=1000, n_boot=300, seed=3)
        out[name] = ct.concordance(rtbl, rb, gtbl, gb_boots).set_index("metric")
    f, b, o = out["faithful"], out["blind"], out["over"]
    assert f.loc["Volume", "status"] == "reproduced" and f.loc["Length", "status"] == "reproduced"
    assert f["concordant"].sum() == 5
    assert b.loc["Volume", "status"] in ("not_reproduced", "underpowered", "amplitude_under") and not b.loc["Volume", "concordant"]
    assert b.loc["Length", "status"] != "reproduced"
    assert o.loc["Volume", "status"] == "amplitude_over"


def test_underpowered_is_flagged_for_a_small_generated_sample(real):
    rtbl, rb = _real_effect(real)
    ga, gb = make_gen(1, 25, "inhibitory"), make_gen(2, 25, "excitatory")
    gtbl, gb_boots = ct.type_effect_generated(ga, gb, METRICS, n_permutations=500, n_boot=200, seed=3)
    conc = ct.concordance(rtbl, rb, gtbl, gb_boots).set_index("metric")
    assert (conc.loc[["Volume", "Length"], "status"].isin(["reproduced", "underpowered"])).all()
    assert gtbl["mde"].min() > 0.1          # 25 samples cannot see a 10 % effect


def test_metric_status_rules():
    s = ct.metric_status
    assert s(True, True, -0.3, -0.4, -0.2, 0.3, 0.2, 0.4, 0.05) == "wrong_sign"
    assert s(True, False, -0.1, -0.2, 0.0, -0.05, -0.2, 0.1, 0.3) == "underpowered"
    assert s(True, False, -0.5, -0.6, -0.4, -0.02, -0.1, 0.06, 0.1) == "not_reproduced"
    assert s(False, True, 0.0, -0.05, 0.05, 0.4, 0.3, 0.5, 0.1) == "spurious_effect"
    assert s(False, False, 0.0, -0.05, 0.05, 0.01, -0.1, 0.1, 0.1) == "consistent"


def _tester(real, n_reference=200, kinds=("inhibitory", "excitatory")):
    pools = {k: ct.RealPool(real[real.cell_type_binary == k], METRICS) for k in kinds}
    pools["mixture"] = ct.RealPool(real, METRICS)
    return ct.SameTypeTester(pools, METRICS, n_reference=n_reference, n_observed_draws=10, seed=0)


def test_same_type_accepts_faithful_and_rejects_shifted(real):
    tester = _tester(real)
    ok = tester.test(make_gen(5, 300, "excitatory"), "excitatory")
    assert (ok["table"]["q_boot"] > 0.05).all() and ok["mv"]["p_ed"] > 0.05
    # generated with the OTHER condition: the Volume / Length shift must be found (control of the test's sensitivity)
    bad = tester.test(make_gen(6, 300, "inhibitory"), "excitatory")
    t = bad["table"].set_index("metric")
    assert t.loc["Volume", "q_boot"] < 0.05 and t.loc["Volume", "d_hi"] < 0 and bad["mv"]["p_ed"] < 0.05
    assert t.loc["Area", "verdict"] != "differs"


def test_equivalence_needs_proof_and_min_margin_is_reported():
    big = make_real(seed=11, n_neurons=300, spines=30, inhibitory_share=0.0)
    pools = {"excitatory": ct.RealPool(big, METRICS)}
    tester = ct.SameTypeTester(pools, METRICS, n_reference=50, n_observed_draws=10, seed=0, margin_sd=0.35, n_boot=300)
    t = tester.test(make_gen(5, 500, "excitatory"), "excitatory")["table"].set_index("metric")
    assert (t["verdict"] == "equivalent").sum() >= 3
    # the margin-free statement: equivalence is proven exactly for the margins above min_margin_sd
    assert ((t["min_margin_sd"] < 0.35) == (t["q_tost"] < 0.05)).sum() >= 4 and (t["min_margin_sd"] > 0).all()
    shifted = make_gen(5, 500, "excitatory")
    shifted["Area"] *= np.exp(0.6 * float(np.log(big["Area"]).std()))
    t2 = tester.test(shifted, "excitatory")["table"].set_index("metric")
    assert t2.loc["Area", "verdict"] == "differs" and t2.loc["Area", "min_margin_sd"] > 0.6
    strict = ct.SameTypeTester(pools, METRICS, n_reference=50, n_observed_draws=10, seed=0, n_boot=300).test(make_gen(5, 500, "excitatory"), "excitatory")["table"]
    assert (strict["verdict"] == "equivalent").sum() <= (t["verdict"] == "equivalent").sum()


def test_type_one_error_and_power_of_the_bootstrap_test_and_neuron_awareness():
    """Generated = fresh draws of the true population (new neuron style per spine). The cluster-bootstrap test of the signed
    median difference must hold ~5 % (>= 200 repeats of the real dataset), have power against a 0.3 sd shift, and the naive
    spine-level KS against the pooled real spines must reject far more often (it ignores the neuron sampling error)."""
    from scipy.stats import ks_2samp

    rej = tot = naive = 0
    power = []
    for s in range(200):
        real = make_real(seed=1000 + s, n_neurons=50, spines=30, style_sd=0.3, inhibitory_share=0.0)
        tester = ct.SameTypeTester({"excitatory": ct.RealPool(real, METRICS)}, METRICS, n_reference=10, n_observed_draws=5, seed=s, n_boot=150, n_real_draw=3000)
        gen = make_gen(5000 + s, 300, "excitatory", style_sd=0.3)
        tb = tester.test(gen, "excitatory", descriptive=False)["table"]
        rej += int((tb["p_boot"] < 0.05).sum())
        tot += len(tb)
        naive += sum(ks_2samp(gen[m], real[m]).pvalue < 0.05 for m in METRICS)
        if s < 60:
            sh = gen.copy()
            sh["Area"] *= np.exp(0.3 * float(np.log(real["Area"]).std()))
            power.append(tester.test(sh, "excitatory", descriptive=False)["table"].set_index("metric").loc["Area", "p_boot"] < 0.05)
    assert 0.02 <= rej / tot <= 0.08
    assert np.mean(power) > 0.7
    assert naive / tot > rej / tot + 0.1


def test_correlation_structure_is_compared(real):
    tester = _tester(real)
    gen = make_gen(5, 300, "excitatory")
    # destroy the dependence between the metrics while keeping every marginal: shuffle each column independently
    rng = np.random.default_rng(0)
    broken = gen.copy()
    for m in METRICS:
        broken[m] = rng.permutation(broken[m].to_numpy())
    good, bad = tester.test(gen, "excitatory")["mv"], tester.test(broken, "excitatory")["mv"]
    assert good["p_corr"] > 0.05 and bad["p_corr"] < 0.05


def test_condition_classifier_follows_the_condition(real):
    clf = ct.fit_condition_classifier(real, "cell_type_binary", "inhibitory", "excitatory", METRICS, n_boot=100, seed=0)
    assert 0.55 < clf.auc < 0.9 and clf.auc_ci[0] <= clf.auc <= clf.auc_ci[1]
    follow_a = ct.apply_condition_classifier(clf, make_gen(1, 500, "inhibitory"))
    follow_b = ct.apply_condition_classifier(clf, make_gen(2, 500, "excitatory"))
    blind_a = ct.apply_condition_classifier(clf, make_gen(3, 500, "inhibitory", shift_kind="excitatory"))
    assert (follow_a > 0.5).mean() == pytest.approx(clf.tpr, abs=0.1) and (follow_b > 0.5).mean() == pytest.approx(clf.fpr, abs=0.1)
    assert (follow_a > 0.5).mean() > (blind_a > 0.5).mean() + 0.1
    auc, ci = ct.mann_whitney_auc(follow_a, follow_b, n_boot=200)
    assert auc == pytest.approx(clf.auc, abs=0.08)


def test_rogan_gladen_recovers_the_mixture_share():
    real = make_real(seed=11, n_neurons=400, spines=30)
    clf = ct.fit_condition_classifier(real, "cell_type_binary", "inhibitory", "excitatory", METRICS, n_boot=200, seed=0)
    n = 4000
    mix = pd.concat([make_gen(1, int(n * 0.2), "inhibitory"), make_gen(2, n - int(n * 0.2), "excitatory")])
    s = ct.apply_condition_classifier(clf, mix)
    pi, ci = ct.rogan_gladen(float((s > 0.5).mean()), len(s), clf.boot_tpr, clf.boot_fpr, tpr=clf.tpr, fpr=clf.fpr)
    assert ci[0] <= 0.2 <= ci[1] and pi == pytest.approx(0.2, abs=0.1)


def test_real_vs_generated_classifier(real):
    pool = ct.RealPool(real[real.cell_type_binary == "excitatory"], METRICS)
    good = ct.real_vs_generated_auc(pool, ct.complete_rows(make_gen(5, 300, "excitatory"), METRICS), n_reference=40, seed=1)
    bad = ct.real_vs_generated_auc(pool, ct.complete_rows(make_gen(5, 300, "excitatory", amplitude=-3.0, shift_kind="inhibitory"), METRICS), n_reference=40, seed=1)
    assert good["p"] > 0.05 and bad["p"] < 0.05 and bad["auc"] > good["auc"] + 0.1


def test_dose_response_slope_tracks_the_amplitude(real):
    rtbl, rb = _real_effect(real)
    effects = {}
    for w, amp in ((1.0, 0.5), (2.0, 1.0), (3.0, 1.6)):
        effects[w] = ct.type_effect_generated(make_gen(1, 800, "inhibitory", amplitude=amp), make_gen(2, 800, "excitatory"), METRICS, n_permutations=200, n_boot=200, seed=4)
    d = ct.dose_response(effects, rtbl, rb, gen_ed={1.0: 0.1, 2.0: 0.3, 3.0: 0.6}, real_ed=(0.28, 0.2, 0.35)).set_index("w")
    assert d.loc[1.0, "beta"] == pytest.approx(0.5, abs=0.2) and d.loc[2.0, "beta"] == pytest.approx(1.0, abs=0.2) and d.loc[3.0, "beta"] > 1.3
    assert d["closest"].sum() == 1 and bool(d.loc[2.0, "closest"])


def test_validity_table_reports_rates():
    g = make_gen(1, 100, "excitatory")
    g.loc[:19, "valid"] = False
    g.loc[:9, "Volume"] = np.nan
    v = ct.validity_table({("excitatory", "g1"): g}, METRICS).iloc[0]
    assert v["valid_rate"] == pytest.approx(0.8) and v["complete_rate"] == pytest.approx(0.9) and v["valid_lo"] < 0.8 < v["valid_hi"]


def test_orchestration_end_to_end(real):
    sets = {("inhibitory", "g1"): make_gen(1, 200, "inhibitory"), ("excitatory", "g1"): make_gen(2, 200, "excitatory"),
            ("unconditional", "default"): pd.concat([make_gen(3, 40, "inhibitory"), make_gen(4, 160, "excitatory")], ignore_index=True)}
    res = ct.run_conditional_tests(real, sets, "cell_type_binary", metrics=METRICS, n_permutations=300, n_boot=100, n_reference=60, n_reference_classifier=10, n_real_draw=600, log=lambda *_: None)
    assert res["effects"]["g1"]["n_concordant"] >= 4 and set(res["same_type"]) == {"inhibitory@g1", "excitatory@g1", "unconditional@default"}
    assert res["same_type"]["unconditional@default"]["own"] == "mixture"
    assert "logistic/all" in res["classifier"] and "unconditional@default" in res["discriminability"]
    import json

    json.dumps(ct.to_jsonable({k: v for k, v in res.items()}))


def _small_run(real):
    sets = {("inhibitory", "g1.25"): make_gen(1, 150, "inhibitory"), ("excitatory", "g1.25"): make_gen(2, 150, "excitatory"),
            ("unconditional", "default"): pd.concat([make_gen(3, 30, "inhibitory"), make_gen(4, 120, "excitatory")], ignore_index=True)}
    return sets, ct.run_conditional_tests(real, sets, "cell_type_binary", metrics=METRICS, n_permutations=200, n_boot=60, n_reference=40, n_reference_classifier=8, n_real_draw=600, log=lambda *_: None)


def test_report_builds_and_embeds_valid_payload(real, tmp_path):
    import json

    from src.neuron_model.reports.conditional_report import build_report

    sets, res = _small_run(real)
    (tmp_path / "results.json").write_text(json.dumps(ct.to_jsonable(res), ensure_ascii=False), encoding="utf-8")
    (tmp_path / "run_info.json").write_text(json.dumps({"created": "2026-11-04 10:00", "n_real_spines": len(real), "n_real_neurons": 120, "sets": {"inhibitory@g1.25": {"n": 150, "checkpoint": "ck", "guidance": 1.25, "seed": 1}}}), encoding="utf-8")
    html = build_report(tmp_path).read_text(encoding="utf-8")
    start = html.index('id="payload">') + len('id="payload">')
    payload = json.loads(html[start:html.index("</script>", start)])
    assert payload["results"]["effects"]["g1.25"]["n_metrics"] == 5 and "sec-a" in html and "sec-g" in html and "Ограничения" in html
    assert "NaN" not in html[start:html.index("</script>", start)]


def test_cli_loads_a_set_from_a_combined_parquet(real, tmp_path):
    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location("cgr", Path(__file__).resolve().parents[2] / "scripts/neuron_model/conditional_generation_report.py")
    cgr = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cgr)
    sets, _ = _small_run(real)
    pd.concat([f.assign(value=v, label=lab, folder=str(tmp_path / v)) for (v, lab), f in sets.items()], ignore_index=True).to_parquet(tmp_path / "generated_metrics.parquet")
    frame, info = cgr.load_set("inhibitory", "g1.25", tmp_path, "postprocessed")
    assert len(frame) == 150 and info["source"] == str(tmp_path)


TYPE_SHIFT = {"4P": np.zeros(5), "6P": np.array([0, 0.25, 0, 0, 0]), "BC": np.array([-0.4, -0.2, 0, 0, 0]), "MC": np.array([0, 0, 0.3, 0, 0.2])}


def _typed_spines(rng, n, t, style, amp=1.0):
    z = rng.normal(size=(n, 1))
    return np.exp(MU + style + LOAD * z + rng.normal(size=(n, 5)) * NOISE + amp * TYPE_SHIFT[t])


def make_typed_real(seed=0, neurons_per_type=40, spines=40, style_sd=0.2):
    rng = np.random.default_rng(seed)
    rows = []
    for t in TYPE_SHIFT:
        for i in range(neurons_per_type):
            f = pd.DataFrame(_typed_spines(rng, spines, t, rng.normal(0, style_sd, 5)[None, :]), columns=METRICS)
            f["neuron_id"], f["cell_type"] = f"{t}_{i}", t
            rows.append(f)
    return pd.concat(rows, ignore_index=True)


def make_typed_gen(seed, n, t, shift_type=None, style_sd=0.2):
    rng = np.random.default_rng(seed)
    f = pd.DataFrame(_typed_spines(rng, n, shift_type or t, rng.normal(0, style_sd, (n, 5))), columns=METRICS)
    f["mesh_present"], f["attachment_found"], f["valid"] = True, True, True
    return f


def test_detailed_cell_types_end_to_end():
    real = make_typed_real()
    sets = {(t, "g1"): make_typed_gen(10 + i, 300, t) for i, t in enumerate(TYPE_SHIFT)}
    sets[("BC", "blind")] = make_typed_gen(40, 300, "BC", shift_type="4P")     # asked for BC, generated like 4P
    sets[("4P", "blind")] = make_typed_gen(41, 300, "4P")
    res = ct.run_conditional_tests(real, sets, "cell_type", metrics=METRICS, n_permutations=500, n_boot=250, n_reference=40, n_reference_classifier=5, n_real_draw=600, log=lambda *_: None)
    # A: all 6 pairs of types, BH across pairs
    eff = {k: v for k, v in res["effects"].items() if k.startswith("g1")}
    assert len(eff) == 6 and res["pair"] is None and len(res["real_effects"]) == 6
    pair = next(v for v in eff.values() if set(v["pair"]) == {"4P", "6P"})["concordance"].set_index("metric")
    assert pair.loc["Length", "real_sig"] and pair.loc["Length", "status"] == "reproduced" and pair.loc["Volume", "status"] == "consistent"
    assert all(v["n_concordant"] >= 4 for v in eff.values())
    # blind pair (BC vs 4P generated alike): the Volume / Length difference must not be reproduced
    blind = next(v for k, v in res["effects"].items() if k.startswith("blind") and set(v["pair"]) == {"4P", "BC"})["concordance"].set_index("metric")
    assert blind.loc["Volume", "status"] != "reproduced"
    # B: every type has its own pool
    assert {k.split("@")[0] for k in res["same_type"]} == set(TYPE_SHIFT)
    t = res["same_type"]["MC@g1"]["by_pool"]["MC"]["table"].set_index("metric")
    assert (t["q_boot"] > 0.05).all()
    wrong = res["same_type"]["MC@g1"]["by_pool"]["4P"]["table"].set_index("metric")
    assert wrong.loc["Area", "q_boot"] < 0.05
    # C: multiclass classifier recovers the requested type as often as real spines of that type are recovered
    tc = res["type_classifier"]
    assert tc["balanced_accuracy"] > tc["chance"] + 0.1 and len(tc["classes"]) == 4
    assert np.allclose(np.array(tc["confusion"]).sum(1), 1.0)
    ok = [tc["sets"][f"{t}@g1"]["within_expected"] for t in TYPE_SHIFT]
    assert sum(ok) >= 3 and not tc["sets"]["BC@blind"]["within_expected"]


def test_detailed_types_with_few_neurons_are_flagged():
    real = make_typed_real()
    rare = make_typed_real(seed=5, neurons_per_type=7)
    rare = rare[rare.cell_type == "MC"].assign(cell_type="RARE", neuron_id=lambda d: "r" + d.neuron_id)
    real = pd.concat([real, rare], ignore_index=True)
    sets = {("RARE", "g1"): make_typed_gen(1, 200, "MC"), ("4P", "g1"): make_typed_gen(2, 200, "4P")}
    res = ct.run_conditional_tests(real, sets, "cell_type", metrics=METRICS, n_permutations=200, n_boot=50, n_reference=20, n_reference_classifier=3, n_real_draw=600, log=lambda *_: None)
    assert res["pools"]["RARE"]["underpowered"] and any("RARE" in w for w in res["warnings"])
    assert "RARE@g1" in res["type_classifier"]["sets"] and res["type_classifier"]["sets"]["RARE@g1"]["underpowered"]


def test_gbm_classifier_option_runs(real):
    from threadpoolctl import threadpool_limits

    sets = {("inhibitory", "g1"): make_gen(1, 150, "inhibitory"), ("excitatory", "g1"): make_gen(2, 150, "excitatory")}
    with threadpool_limits(limits=1):       # OpenMP oversubscription makes HistGradientBoosting 100x slower on small data
        res = ct.run_conditional_tests(real, sets, "cell_type_binary", metrics=METRICS, n_permutations=100, n_boot=30, n_reference=20, n_reference_classifier=3,
                                       classifiers=("logistic", "gbm"), n_real_draw=600, log=lambda *_: None)
    assert "gbm/all" in res["classifier"] and 0.5 < res["classifier"]["gbm/all"]["auc"] <= 1.0
    assert "gbm" in res["discriminability"]["inhibitory@g1"]
