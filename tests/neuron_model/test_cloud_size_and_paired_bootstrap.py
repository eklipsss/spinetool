from __future__ import annotations

import numpy as np
import pandas as pd

from src.neuron_model.spine_generation.evaluation import cloud_size as cs
from src.neuron_model.spine_generation.evaluation import paired_bootstrap as pb
from src.neuron_model.spine_generation.experiment.artifacts import write_pointcloud_ply


def cylinder(n, radius, length, noise, rng):
    t = rng.uniform(0, 2 * np.pi, n)
    y = rng.uniform(0, length, n)
    p = np.c_[radius * np.cos(t), y, radius * np.sin(t)]
    return p + rng.normal(0, noise, p.shape)


def test_cloud_measures_recover_radius_length_and_noise():
    rng = np.random.default_rng(0)
    m = cs.cloud_measures(cylinder(8192, 200.0, 1500.0, 0.0, rng))
    assert abs(m["length_nm"] - 1500) < 30 and abs(m["radial_rms_nm"] - 200) < 3 and m["surface_noise_nm"] < 2
    noisy = cs.cloud_measures(cylinder(8192, 200.0, 1500.0, 20.0, rng))
    assert noisy["surface_noise_nm"] > 5 * m["surface_noise_nm"] and abs(noisy["radial_rms_nm"] - 200) < 5


def test_generated_dir_and_summary_ratio(tmp_path):
    rng = np.random.default_rng(1)
    for i in range(12):
        d = tmp_path / "gen" / f"sample_{i:05d}"
        d.mkdir(parents=True)
        write_pointcloud_ply(d / "generated_pointcloud.ply", cylinder(2048, 300.0, 1500.0, 0.0, rng).astype(np.float32))
    gen = cs.measure_generated_dir(tmp_path / "gen")
    real = pd.DataFrame([{"spine_key": f"minnie65/{i % 4}/000/000/{i:03d}", **cs.cloud_measures(cylinder(2048, 200.0, 1500.0, 0.0, rng)),
                          "mesh_volume_nm3": np.nan} for i in range(12)])
    s = cs.summarise({"real": real, "gen": gen}, "real", n_boot=200).set_index(["set", "measure"])
    assert abs(s.loc[("gen", "radial_rms_nm"), "ratio_to_ref"] - 1.5) < 0.05
    assert abs(s.loc[("gen", "length_nm"), "ratio_to_ref"] - 1.0) < 0.05


def test_paired_bootstrap_ranks_a_shifted_generator_worse():
    rng = np.random.default_rng(2)
    n_neur, per = 30, 60
    keys = [f"minnie65/{n}/000/000/{i:03d}" for n in range(n_neur) for i in range(per)]
    effect = np.repeat(rng.normal(0, 0.2, n_neur), per)
    base = np.exp(rng.normal(0, 1, (len(keys), 3)) + effect[:, None])
    metrics = ("Length", "Area", "Volume")
    real = pd.DataFrame(base, columns=metrics).assign(spine_key=keys)

    def gen(shift):
        x = np.exp(rng.normal(0, 1, (400, 3)) + rng.normal(0, 0.2, (400, 1)) + shift)
        return pd.DataFrame(x, columns=metrics).assign(generation_seed=np.repeat([1, 2], 200))

    points, reps = pb.paired_bootstrap(real, {"good": gen(0.0), "bad": gen(0.6)}, metrics=metrics, n_boot=40, n_real=800, auc=True, log=lambda *_: None)
    d = pb.pair_differences(points, reps, [("good", "bad")]).set_index("statistic")
    assert d.loc["w1_mean", "hi"] < 0 and d.loc["energy", "hi"] < 0 and d.loc["auc", "hi"] < 0
    assert d.loc["w1_mean", "share_a_better"] == 1.0
