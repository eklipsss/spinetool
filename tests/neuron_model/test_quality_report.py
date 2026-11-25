"""``quality_report``: tables and figures of the generation quality from evaluation outputs (synthetic outputs)."""

import numpy as np
import pandas as pd

from src.neuron_model import quality_report as Q


def _tp_set(tmp_path, shift_radius=0.0, n=120, eval_meta=False):
    rng = np.random.default_rng(0)
    d = tmp_path / "set"
    (d / "evaluation_paired").mkdir(parents=True)
    neurons = [f"n{i % 12}" for i in range(n)]
    req = pd.DataFrame({"branch_index": range(n), "species": "mouse", "cell_type_binary": ["excitatory" if i % 3 else "inhibitory" for i in range(n)],
                        "compartment": "basal", "source_branch": [f"minnie65/{nn}/limb_000/{i}" for i, nn in enumerate(neurons)]})
    req.to_parquet(d / "branches.parquet")
    def rows(kind, k):
        r = rng.lognormal(np.log(280.0) + k, 0.3, n)
        ns = rng.poisson(10, n).astype(float)
        return pd.DataFrame({"branch_index": range(n), "length_nm": 3e4, "n_spines": ns, "mean_radius_nm": r, "radius_ratio_prox_dist": 1.1,
                             "kappa_rms_per_nm": rng.lognormal(-8.4, 0.2, n), "tortuosity": 1 + rng.exponential(0.05, n), "radius_cv": 0.1,
                             "kappa_autocorr_1um": rng.normal(0.4, 0.1, n), "density_per_um": ns / 30, "no_spines": (ns == 0).astype(float), "kind": kind})
    ev = pd.concat([rows("generated", shift_radius), rows("real", 0.0)])
    if eval_meta:  # real evaluation tables also carry metadata columns (duplicate names after the join with the requests)
        ev = ev.assign(neuron=np.tile(neurons, 2), species="mouse", compartment="basal")
    ev.to_csv(d / "evaluation_paired" / "evaluation_branches.csv", index=False)
    sp = []
    for kind in ("generated", "real"):
        for b in range(n):
            sp.append(pd.DataFrame({"branch": b, "u": rng.random(5), "theta": rng.uniform(-np.pi, np.pi, 5), "nn_surface_nm": rng.gamma(4, 200, 5), "kind": kind}))
            sp.append(pd.DataFrame({"branch": b, "gap_nm": rng.exponential(1500, 4), "kind": kind}))
    pd.concat(sp).to_csv(d / "evaluation_paired" / "evaluation_spines.csv", index=False)
    return d


def test_tp_report_finds_a_planted_radius_shift_and_writes_figures(tmp_path):
    d = _tp_set(tmp_path, shift_radius=0.3)
    lines = Q.tp_report(d, tmp_path / "out", label="syn", n_boot=200)
    t = pd.read_csv(tmp_path / "out" / "tp_syn.csv")
    overall = t[t["table"].str.startswith("T: ")]
    radius = overall[overall["metric"] == "mean_radius_nm"].iloc[0]
    assert radius["lo"] > 0 and radius["q"] < 0.05  # planted +35 % radius
    assert (overall[overall["metric"] == "tortuosity"]["q"] > 0.05).all()  # nothing planted
    assert (tmp_path / "out" / "t_syn.png").exists() and (tmp_path / "out" / "p_spines_syn.png").exists()
    assert any("cell_type_binary = inhibitory" in l for l in lines)


def test_tp_report_with_metadata_columns_in_the_evaluation_table(tmp_path):
    d = _tp_set(tmp_path, shift_radius=0.3, eval_meta=True)
    Q.tp_report(d, tmp_path / "out", label="syn", n_boot=100)
    t = pd.read_csv(tmp_path / "out" / "tp_syn.csv")
    assert (t["metric"] == "mean_radius_nm").any()


def test_s_overall_reads_an_evaluation_run(tmp_path):
    rng = np.random.default_rng(1)
    run = tmp_path / "eval"
    (run / "postprocessed").mkdir(parents=True)
    keys = [f"minnie65/n{i % 20}/limb_000/branch_000/spine_{i}" for i in range(2000)]
    def metrics(n, k):
        return {m: rng.lognormal(k if m == "Volume" else 0.0, 0.4, n) for m in Q.S_METRICS}
    pd.DataFrame({"spine_key": keys, **metrics(2000, 0.0)}).to_parquet(run / "real_test_morphometrics.parquet")
    for s in (1, 2):
        pd.DataFrame({"sample_id": range(300), **metrics(300, 0.4)}).to_parquet(run / "postprocessed" / f"morphometrics_mogen_A_seed{s}.parquet")
        pd.DataFrame({"sample_id": range(300), **metrics(300, 0.0)}).to_parquet(run / "postprocessed" / f"morphometrics_real_poisson_seed{s}.parquet")
    lines = Q.s_overall(run, tmp_path / "out", n_boot=200)
    t = pd.read_csv(tmp_path / "out" / "s_per_metric.csv")
    vol = t[(t["set"] == "mogen_A") & (t["metric"] == "Volume")].iloc[0]
    assert vol["significant"] and vol["d"] > 0.25
    assert not t[(t["set"] == "real_poisson")]["significant"].any()
    assert (tmp_path / "out" / "s_marginals.png").exists() and any("Значимых отличий" in l for l in lines)
