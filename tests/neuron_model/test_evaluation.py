"""Stage-7 evaluation metrics on synthetic data with known answers.

Scenarios: same distribution (metrics at their "indistinguishable" values), shift, mode
collapse (all generated samples ~ one shape), memorization (generated = train copies).
"""

import numpy as np
import pandas as pd
import pytest
import trimesh

from src.neuron_model.spine_generation.evaluation.bootstrap import aggregate_over_seeds, bootstrap_ci
from src.neuron_model.spine_generation.evaluation.classifier import real_vs_generated_classifier
from src.neuron_model.spine_generation.evaluation.distribution_metrics import (
    JOINT_FEATURES,
    Standardizer,
    correlation_difference,
    histogram_distances,
    mmd_permutation_pvalue,
    rbf_mmd2,
    univariate_distances,
    univariate_table,
)
from src.neuron_model.spine_generation.evaluation.diversity import diversity_report
from src.neuron_model.spine_generation.evaluation.memorization import (
    chamfer_dcr,
    closer_to_train_matrices,
    closer_to_train_points,
    morphometric_dcr,
)
from src.neuron_model.spine_generation.evaluation.morphometrics import generated_morphometrics, real_morphometrics
from src.neuron_model.spine_generation.evaluation.point_cloud_metrics import (
    chamfer_matrix,
    coverage_cd,
    mmd_cd,
    nearest_chamfer_with_candidates,
    one_nna,
)
from src.neuron_model.spine_generation.evaluation.report import (
    EvaluationConfig,
    build_real_reference,
    evaluate_model,
    write_report,
)
from src.neuron_model.spine_morphometrics import JUNCTION_METRICS, AttachmentRegion

# ---------------------------------------------------------------------------
# synthetic data
# ---------------------------------------------------------------------------


def morph_table(n, rng, shift=0.0, collapse=False):
    """Correlated 'morphometrics': a latent size drives Length/Area/Volume (like real spines)."""
    size = rng.normal(size=n) if not collapse else np.full(n, 0.3) + 0.01 * rng.normal(size=n)
    noise = lambda: 0.3 * rng.normal(size=n) if not collapse else 0.003 * rng.normal(size=n)
    cols = {
        "Length": 1000 + 300 * (size + shift) + 50 * noise(),
        "Area": 5e5 + 2e5 * (size + shift) + 3e4 * noise(),
        "Volume": 3e7 + 1e7 * (size + shift) + 2e6 * noise(),
        "OpenAngle": 1.0 + 0.2 * noise() + 0.2 * shift,
        "CVD": 0.3 + 0.05 * noise(),
        "AverageDistance": 600 + 150 * (size + shift) + 30 * noise(),
        "LengthVolumeRatio": 3e-5 + 5e-6 * noise(),
        "LengthAreaRatio": 2e-3 + 2e-4 * noise(),
        "ConvexHullVolume": 3.5e7 + 1.1e7 * (size + shift) + 2e6 * noise(),
        "ConvexHullRatio": 0.85 + 0.03 * noise(),
        "JunctionArea": 3e4 + 5e3 * noise(),
    }
    frame = pd.DataFrame(cols)
    peaks = np.clip(0.4 + 0.1 * rng.normal(size=n) + 0.1 * shift, 0.05, 0.95)
    bins = (np.arange(100) + 0.5) / 100
    frame["OldChordDistribution"] = [list(np.exp(-((bins - p) ** 2) / 0.02) / np.exp(-((bins - p) ** 2) / 0.02).sum() * 100) for p in peaks]
    return frame


def ellipsoid_clouds(n, rng, n_points=96, collapse=False, axes_shift=0.0):
    """'Spines' = ellipsoids with random axes (nm-like scale); collapse -> all ~one shape."""
    clouds = []
    for _ in range(n):
        axes = np.array([400.0, 500.0, 1200.0]) * (np.exp(0.25 * rng.normal(size=3)) if not collapse else np.exp(0.003 * rng.normal(size=3)))
        axes = axes * (1 + axes_shift)
        d = rng.normal(size=(n_points, 3))
        clouds.append(d / np.linalg.norm(d, axis=1, keepdims=True) * axes)
    return np.stack(clouds).astype(np.float32)


# ---------------------------------------------------------------------------
# distribution metrics
# ---------------------------------------------------------------------------


def test_univariate_same_vs_shifted_and_missing_values():
    rng = np.random.default_rng(0)
    real = rng.normal(size=2000)
    same = univariate_distances(real, rng.normal(size=2000))
    shifted = univariate_distances(real, rng.normal(size=2000) + 1.0)
    assert same["w1_normalized"] < 0.1 and same["ks_statistic"] < 0.06
    assert shifted["w1_normalized"] == pytest.approx(1.0, abs=0.1)  # 1-std shift
    assert shifted["ks_pvalue"] < 1e-10 and shifted["energy"] > 10 * same["energy"]
    with_nan = univariate_distances(real, np.r_[rng.normal(size=10), np.nan, np.inf])
    assert with_nan["n_generated"] == 10 and with_nan["n_generated_missing"] == 2


def test_univariate_table_uses_given_scales():
    rng = np.random.default_rng(1)
    real, gen = morph_table(300, rng), morph_table(300, rng, shift=1.0)
    table = univariate_table(real, gen, ["Length", "Volume"], scales={"Length": 300.0, "Volume": 1e7})
    row = table.set_index("feature").loc["Length"]
    assert row["w1_normalized"] == pytest.approx(row["w1"] / 300.0)
    assert set(table["feature"]) == {"Length", "Volume"}


def test_standardizer_train_only_and_roundtrip():
    rng = np.random.default_rng(2)
    train = morph_table(500, rng)
    s = Standardizer.fit(train, JOINT_FEATURES)
    x = s.transform(train)
    assert np.allclose(x.mean(axis=0), 0, atol=1e-9) and np.allclose(x.std(axis=0), 1, atol=1e-9)
    assert Standardizer.from_dict(s.to_dict()) == s
    with pytest.raises(ValueError):
        Standardizer.fit(train.assign(CVD=1.0), JOINT_FEATURES)
    with_nan = train.copy()
    with_nan.loc[0, "Length"] = np.nan
    assert s.transform(with_nan).shape[0] == 499  # incomplete rows dropped


def test_mmd_and_permutation_test():
    rng = np.random.default_rng(3)
    x = rng.normal(size=(300, 4))
    same = rng.normal(size=(300, 4))
    shifted = rng.normal(size=(300, 4)) + 0.5
    mmd_same, sigma = rbf_mmd2(x, same)
    mmd_shift, _ = rbf_mmd2(x, shifted, sigma=sigma)
    assert abs(mmd_same) < 0.01 and mmd_shift > 0.02 and mmd_shift > 20 * abs(mmd_same)
    assert mmd_permutation_pvalue(x, same, sigma=sigma, n_permutations=50) > 0.05
    assert mmd_permutation_pvalue(x, shifted, sigma=sigma, n_permutations=50) < 0.05


def test_correlation_difference_detects_broken_relations():
    rng = np.random.default_rng(4)
    a = rng.normal(size=1000)
    real = np.c_[a, a + 0.1 * rng.normal(size=1000)]  # strongly correlated
    same_marginals_independent = np.c_[rng.normal(size=1000), rng.normal(size=1000)]
    assert correlation_difference(real, real.copy())["frobenius"] < 1e-12
    broken = correlation_difference(real, same_marginals_independent)
    assert broken["frobenius"] == pytest.approx(np.sqrt(2) * 0.995, abs=0.1)  # two off-diagonal entries ~0.995


def test_histogram_metric_comparison():
    rng = np.random.default_rng(5)
    real, gen_same, gen_shift = morph_table(200, rng), morph_table(200, rng), morph_table(200, rng, shift=2.0)
    same = histogram_distances(real["OldChordDistribution"], gen_same["OldChordDistribution"])
    shift = histogram_distances(real["OldChordDistribution"], gen_shift["OldChordDistribution"])
    assert same["w1_mean_density"] < 0.02
    assert shift["w1_mean_density"] == pytest.approx(0.2, abs=0.05)  # peaks moved by 0.1*shift = 0.2
    assert shift["mmd2"] > 10 * abs(same["mmd2"])


# ---------------------------------------------------------------------------
# surface (point cloud) metrics
# ---------------------------------------------------------------------------


def test_chamfer_matrix_matches_brute_force_and_symmetry():
    rng = np.random.default_rng(6)
    a, b = rng.normal(size=(3, 20, 3)), rng.normal(size=(4, 25, 3))
    d = chamfer_matrix(a, b, pair_chunk=2)
    for i in range(3):
        for j in range(4):
            dd = ((a[i][:, None] - b[j][None]) ** 2).sum(-1)
            assert d[i, j] == pytest.approx(dd.min(1).mean() + dd.min(0).mean(), rel=1e-5)
    sym = chamfer_matrix(a)
    assert np.allclose(sym, sym.T) and np.allclose(np.diag(sym), 0)
    assert np.allclose(sym, chamfer_matrix(a, a), atol=1e-5)
    assert np.allclose(chamfer_matrix(a, b, scale=0.5), 0.25 * d, rtol=1e-5)  # squared distances


def test_surface_metrics_same_distribution_vs_mode_collapse():
    rng = np.random.default_rng(7)
    n = 60
    real = ellipsoid_clouds(n, rng)
    good = ellipsoid_clouds(n, rng)
    collapsed = ellipsoid_clouds(n, rng, collapse=True)
    d_rr = chamfer_matrix(real)
    good_m = (d_rr, chamfer_matrix(good), chamfer_matrix(real, good))
    bad_m = (d_rr, chamfer_matrix(collapsed), chamfer_matrix(real, collapsed))
    nna_good, nna_bad = one_nna(*good_m), one_nna(*bad_m)
    assert 0.3 < nna_good["accuracy"] < 0.7          # indistinguishable -> ~0.5
    # collapsed samples' NN is (almost) always another collapsed sample; a few real spines near the
    # collapsed shape do get a generated NN, so overall accuracy is high but not exactly 1
    assert nna_bad["accuracy_generated"] > 0.9 and nna_bad["accuracy"] > 0.8 > nna_good["accuracy"]
    assert coverage_cd(good_m[2]) > 3 * coverage_cd(bad_m[2])
    assert mmd_cd(good_m[2]) < mmd_cd(bad_m[2])
    with pytest.raises(ValueError):
        one_nna(d_rr, chamfer_matrix(good[:10]), chamfer_matrix(real, good[:10]))


def test_one_nna_detects_copies():
    rng = np.random.default_rng(8)
    real = ellipsoid_clouds(30, rng)
    copies = real + rng.normal(scale=1e-3, size=real.shape).astype(np.float32)
    result = one_nna(chamfer_matrix(real), chamfer_matrix(copies), chamfer_matrix(real, copies))
    assert result["accuracy"] < 0.05  # every sample's NN is its copy in the other class -> ~0 (not 0.5!)


def test_candidate_reranking_finds_exact_nearest():
    rng = np.random.default_rng(9)
    refs = ellipsoid_clouds(40, rng)
    queries = refs[[3, 17]] + 1.0
    exact = chamfer_matrix(queries, refs).argmin(axis=1)
    candidates = np.array([[0, 3, 5, 9], [17, 2, 30, 31]])
    found = nearest_chamfer_with_candidates(queries, refs, candidates)
    assert list(found["index"]) == list(exact) == [3, 17]


# ---------------------------------------------------------------------------
# diversity / memorization / classifier
# ---------------------------------------------------------------------------


def test_diversity_flags_mode_collapse():
    rng = np.random.default_rng(10)
    real_c, good_c, bad_c = ellipsoid_clouds(40, rng), ellipsoid_clouds(40, rng), ellipsoid_clouds(40, rng, collapse=True)
    train = morph_table(500, rng)
    s = Standardizer.fit(train)
    real_x, good_x, bad_x = s.transform(morph_table(200, rng)), s.transform(morph_table(200, rng)), s.transform(morph_table(200, rng, collapse=True))
    good = diversity_report(real_real_chamfer=chamfer_matrix(real_c), gen_gen_chamfer=chamfer_matrix(good_c), real_x=real_x, gen_x=good_x, features=s.features)
    bad = diversity_report(real_real_chamfer=chamfer_matrix(real_c), gen_gen_chamfer=chamfer_matrix(bad_c), real_x=real_x, gen_x=bad_x, features=s.features)
    assert 0.8 < good["chamfer_mean_ratio"] < 1.25 and 0.8 < good["morphometric_mean_ratio"] < 1.25
    # Chamfer between two random samplings of the SAME surface is not 0 (sampling-noise floor),
    # so even a perfect collapse keeps chamfer_mean_ratio well above 0 - but far below 1
    assert bad["chamfer_mean_ratio"] < 0.5 and bad["morphometric_mean_ratio"] < 0.1
    assert bad["variance_ratio_geometric_mean"] < 0.01 < 0.5 < good["variance_ratio_geometric_mean"]


def test_memorization_dcr_independent_vs_copies_with_unequal_reference_sizes():
    rng = np.random.default_rng(11)
    train = rng.normal(size=(4000, 10))   # 8x larger than test, like a real split
    test = rng.normal(size=(500, 10))
    independent = rng.normal(size=(300, 10))
    copies = train[rng.choice(4000, 300, replace=False)] + 1e-3 * rng.normal(size=(300, 10))

    fair = morphometric_dcr(independent, test, train)
    assert 0.85 < fair["dcr_median_ratio"] < 1.15
    assert fair["near_copy_rate"] < 0.12 and fair["mannwhitney_pvalue"] > 0.01
    mem = morphometric_dcr(copies, test, train)
    assert mem["dcr_median_ratio"] < 0.05 and mem["near_copy_rate"] > 0.95 and mem["mannwhitney_pvalue"] < 1e-10


def test_spec_literal_check_is_diluted_by_train_subsampling():
    """Documented limitation: subsampling train to the test size (needed against set-size bias)
    keeps the copied spine only with probability |test|/|train| -> a pure copier scores ~0.56."""
    rng = np.random.default_rng(12)
    train, test = rng.normal(size=(4000, 10)), rng.normal(size=(500, 10))
    independent = rng.normal(size=(300, 10))
    copies = train[rng.choice(4000, 300, replace=False)] + 1e-3 * rng.normal(size=(300, 10))
    fair = closer_to_train_points(independent, train, test, n_repeats=10)
    assert 0.4 < fair["fraction_closer_to_train"] < 0.6
    copier = closer_to_train_points(copies, train, test, n_repeats=10)
    assert copier["fraction_closer_to_train_if_copying"] == pytest.approx(0.5625)
    assert copier["fraction_closer_to_train"] == pytest.approx(0.5625, abs=0.05)
    from scipy.spatial import cKDTree  # and without subsampling, the bigger train set wins by chance:
    assert (cKDTree(train).query(independent)[0] < cKDTree(test).query(independent)[0]).mean() > 0.75


def test_chamfer_dcr_with_candidate_reranking():
    rng = np.random.default_rng(16)
    train, holdout = ellipsoid_clouds(80, rng), ellipsoid_clouds(20, rng)
    independent = ellipsoid_clouds(20, rng)
    copies = train[rng.choice(80, 20, replace=False)] + rng.normal(scale=0.5, size=(20, 96, 3)).astype(np.float32)
    fair = chamfer_dcr(independent, holdout, train, k_candidates=15, scale=1e-3)
    mem = chamfer_dcr(copies, holdout, train, k_candidates=15, scale=1e-3)
    assert 0.5 < fair["dcr_median_ratio"] < 2.0
    assert mem["dcr_median_ratio"] < 0.5 * fair["dcr_median_ratio"] and mem["near_copy_rate"] > 0.5


def test_memorization_from_chamfer_matrices():
    rng = np.random.default_rng(12)
    train, test = ellipsoid_clouds(30, rng), ellipsoid_clouds(30, rng)
    copies = train[:20] + rng.normal(scale=1e-3, size=train[:20].shape).astype(np.float32)
    result = closer_to_train_matrices(chamfer_matrix(copies, train), chamfer_matrix(copies, test))
    assert result["fraction_closer_to_train"] == 1.0


def test_classifier_chance_vs_shift():
    rng = np.random.default_rng(13)
    s = Standardizer.fit(morph_table(500, rng))
    real = s.transform(morph_table(400, rng))
    same = real_vs_generated_classifier(real, s.transform(morph_table(400, rng)), seed=0)
    shifted = real_vs_generated_classifier(real, s.transform(morph_table(400, rng, shift=1.5)), seed=0)
    assert 0.4 < same["accuracy"] < 0.6
    assert shifted["accuracy"] > 0.85 and shifted["roc_auc"] > 0.9
    assert real_vs_generated_classifier(real, s.transform(morph_table(400, rng, shift=1.5)), model="gbm")["accuracy"] > 0.85


def test_bootstrap_ci_and_seed_aggregation():
    rng = np.random.default_rng(14)
    real, gen = rng.normal(size=500), rng.normal(size=500) + 0.5
    ci = bootstrap_ci(lambda r, g: g.mean() - r.mean(), real, gen, n_boot=500)
    assert ci["ci_low"] < ci["value"] < ci["ci_high"]
    expected_width = 2 * 1.96 * np.sqrt(2 / 500)  # normal approximation, unit variances
    assert ci["ci_high"] - ci["ci_low"] == pytest.approx(expected_width, rel=0.25)
    agg = aggregate_over_seeds({1: {"a": 1.0, "flag": True}, 2: {"a": 3.0}, 3: {"a": 2.0, "b": np.nan}})
    assert agg["a"] == {"mean": 2.0, "std": 1.0, "min": 1.0, "max": 3.0, "n_seeds": 3}
    assert "flag" not in agg and "b" not in agg


# ---------------------------------------------------------------------------
# morphometrics of meshes (shared with preprocessing) + end-to-end report
# ---------------------------------------------------------------------------


def test_generated_morphometrics_attachment_free_and_with_finder():
    sphere = trimesh.creation.icosphere(subdivisions=2, radius=500.0)
    no_finder = generated_morphometrics([sphere, None], compute_chords=False)
    row = no_finder.iloc[0]
    assert row["mesh_present"] and not row["attachment_found"]
    assert all(np.isnan(row[m]) for m in JUNCTION_METRICS)
    assert row["Volume"] == pytest.approx(sphere.volume, rel=1e-6)
    assert not no_finder.iloc[1]["mesh_present"] and np.isnan(no_finder.iloc[1]["Volume"])

    bottom = sphere.vertices[sphere.vertices[:, 2].argmin()]
    finder = lambda mesh: AttachmentRegion(center=bottom, cap_area=1000.0, loop_area=900.0)
    with_finder = generated_morphometrics([sphere], attachment_finder=finder, compute_chords=False).iloc[0]
    assert with_finder["attachment_found"]
    assert with_finder["JunctionArea"] == 900.0 and with_finder["Area"] == pytest.approx(sphere.area - 1000.0)
    assert with_finder["Length"] > 900  # bottom point to the far side of a r=500 sphere


def test_generated_morphometrics_chord_distribution_shape():
    sphere = trimesh.creation.icosphere(subdivisions=1, radius=500.0)
    row = generated_morphometrics([sphere], compute_chords=True).iloc[0]
    assert len(row["OldChordDistribution"]) == 100


def test_real_morphometrics_reads_manifest_paths(tmp_path):
    import json

    from src.neuron_model.spine_generation.data.index import SpineIndex

    path = tmp_path / "morphometrics.json"
    path.write_text(json.dumps({"Length": 1.0, "Volume": 2.0, "OldChordDistribution": [0.5] * 100}))
    index = SpineIndex(pd.DataFrame({"spine_key": ["a", "b"], "morphometrics_path": [str(path), None]}), ())
    frame = real_morphometrics(index)
    assert frame.loc[0, "Length"] == 1.0 and pd.isna(frame.loc[1, "Length"]) and len(frame.loc[0, "OldChordDistribution"]) == 100


def test_end_to_end_report_ranks_good_model_above_collapsed(tmp_path):
    rng = np.random.default_rng(15)
    real = build_real_reference(
        morph_table(800, rng), morph_table(200, rng),
        test_clouds=ellipsoid_clouds(40, rng), train_clouds=ellipsoid_clouds(80, rng),
    )
    cfg = EvaluationConfig(surface_reference_size=40, memorization_repeats=3, chamfer_scale=1e-3)
    valid = [{"mesh_present": True, "is_valid": True, "is_watertight": True, "n_connected_components": 1, "genus": 0}] * 5
    good = evaluate_model(real, {s: {"gen_morph": morph_table(200, np.random.default_rng(s)), "gen_clouds": ellipsoid_clouds(40, np.random.default_rng(100 + s)), "validity_raw": valid} for s in (1, 2)}, cfg)
    bad = evaluate_model(real, {s: {"gen_morph": morph_table(200, np.random.default_rng(s), collapse=True), "gen_clouds": ellipsoid_clouds(40, np.random.default_rng(100 + s), collapse=True)} for s in (1, 2)}, cfg)
    g, b = good["headline"], bad["headline"]
    assert g["validity_raw.valid_mesh_rate"]["mean"] == 1.0 and "validity_raw.valid_mesh_rate" not in b
    assert g["mmd2_morphometrics"]["mean"] < b["mmd2_morphometrics"]["mean"]
    assert g["coverage_cd"]["mean"] > b["coverage_cd"]["mean"]
    assert abs(g["one_nna_cd"]["mean"] - 0.5) < abs(b["one_nna_cd"]["mean"] - 0.5)
    assert abs(g["diversity.chamfer_mean_ratio"]["mean"] - 1) < abs(b["diversity.chamfer_mean_ratio"]["mean"] - 1)
    assert g["classifier_accuracy"]["mean"] < b["classifier_accuracy"]["mean"]
    assert g["mmd2_morphometrics"]["n_seeds"] == 2 and "chamfer_dcr_median_ratio" in good["per_seed"][1]["memorization"]
    summary = write_report({"good": good, "collapsed": bad}, tmp_path)
    text = summary.read_text()
    assert "coverage_cd" in text and "| good | collapsed |" in text.replace("| criterion | target ", "")
    assert (tmp_path / "report.json").exists() and (tmp_path / "univariate_good_seed1.csv").exists()


def test_generated_set_loader_keeps_missing_meshes(tmp_path):
    from src.neuron_model.spine_generation.evaluation.inputs import discover_seed_dirs, load_generated_set

    seed_dir = tmp_path / "seed_3"
    for i in range(2):
        (seed_dir / f"sample_{i:04d}").mkdir(parents=True)
    trimesh.creation.icosphere(subdivisions=1).export(seed_dir / "sample_0000" / "generated_mesh_raw.off")
    (tmp_path / "seed_x").mkdir()  # ignored: not a seed directory
    assert list(discover_seed_dirs(tmp_path)) == [3]
    meshes, reports, ids = load_generated_set(seed_dir, "raw")
    assert ids == ["sample_0000", "sample_0001"]
    assert meshes[0] is not None and meshes[1] is None
    assert reports[1]["mesh_present"] is False and reports[1]["is_valid"] is False
    with pytest.raises(ValueError):
        load_generated_set(seed_dir, "smoothed")


def test_cloud_cache_roundtrip_and_key_check(tmp_path):
    from src.neuron_model.spine_generation.data.index import SpineIndex
    from src.neuron_model.spine_generation.evaluation.inputs import build_cloud_cache, load_real_clouds, open_cloud_cache

    clouds = ellipsoid_clouds(3, np.random.default_rng(0), n_points=16).astype(np.float32)
    paths = []
    for i, cloud in enumerate(clouds):
        path = tmp_path / f"s{i}" / "pointcloud_16_1.npz"
        path.parent.mkdir()
        np.savez(path, points=cloud)
        paths.append(str(path))
    index = SpineIndex(pd.DataFrame({"spine_key": ["a", "b", "c"], "pointcloud_16_path": paths}), ())
    np.testing.assert_allclose(load_real_clouds(index, 16), clouds, rtol=1e-6)
    cache = build_cloud_cache(index, 16, tmp_path / "cache" / "train_16.npy")
    np.testing.assert_allclose(open_cloud_cache(cache, index), clouds, rtol=1e-6)
    with pytest.raises(ValueError, match="different spine list"):
        open_cloud_cache(cache, index.subset(["a", "b"]))


def _standing_cylinder(radius=200.0, height=1500.0, cone_depth=0.0):
    """Spine-like solid in the local frame: axis +y, flat base disc at y = 0
    (or, with ``cone_depth``, a base that comes to a point at y = -cone_depth)."""
    mesh = trimesh.creation.cylinder(radius=radius, height=height, sections=64)
    mesh.apply_translation([0, 0, height / 2])
    if cone_depth:
        vertices = mesh.vertices.copy()
        centre = np.flatnonzero(np.linalg.norm(vertices, axis=1) < 1e-9)
        vertices[centre, 2] = -cone_depth
        mesh = trimesh.Trimesh(vertices, mesh.faces, process=False)
    mesh = mesh.subdivide().subdivide()
    mesh.apply_transform(trimesh.transformations.rotation_matrix(-np.pi / 2, [1, 0, 0]))  # z -> y
    return mesh


def test_down_facing_cap_finds_flat_base():
    from src.neuron_model.spine_generation.evaluation.attachment import DownFacingCapFinder

    radius = 200.0
    mesh = _standing_cylinder(radius)
    region = DownFacingCapFinder()(mesh)
    disc = np.pi * radius ** 2 * (64 / (2 * np.pi)) * np.sin(2 * np.pi / 64)  # inscribed 64-gon
    assert region.method == "down_facing_cap:axis_ray"
    assert region.loop_area == pytest.approx(disc, rel=1e-6)
    assert region.cap_area == pytest.approx(disc, rel=1e-6)
    np.testing.assert_allclose(region.center, 0.0, atol=1e-6 * radius)


def test_section_fallback_when_base_does_not_face_down():
    from src.neuron_model.spine_generation.evaluation.attachment import DownFacingCapFinder

    radius = 200.0
    mesh = _standing_cylinder(radius, cone_depth=400.0)  # base faces tilted 63 deg from -y
    finder = DownFacingCapFinder(max_tilt_deg=60.0)  # -> no down-facing seed -> section fallback
    region = finder(mesh)
    disc = np.pi * radius ** 2 * (64 / (2 * np.pi)) * np.sin(2 * np.pi / 64)
    assert region.method == "section"
    assert region.loop_area == pytest.approx(disc, rel=1e-6)
    assert abs(region.center[0]) < 1e-6 and abs(region.center[2]) < 1e-6 and region.center[1] > 0
    assert region.cap_area > disc  # pointed base + the wall below the cutting plane
    assert DownFacingCapFinder(max_tilt_deg=60.0, section_fallback=False)(mesh) is None
    assert DownFacingCapFinder(max_tilt_deg=70.0)(mesh).method == "down_facing_cap:axis_ray"


def test_generated_morphometrics_with_finder_and_config():
    from src.neuron_model.spine_generation.evaluation.attachment import DownFacingCapFinder, finder_from_config

    assert finder_from_config(None) is None
    assert finder_from_config({"name": "none"}) is None
    finder = finder_from_config({"name": "down_facing_cap", "max_tilt_deg": 45.0})
    assert finder == DownFacingCapFinder(max_tilt_deg=45.0)
    with pytest.raises(ValueError):
        finder_from_config({"name": "magic"})
    mesh = _standing_cylinder()
    table = generated_morphometrics([mesh, None], attachment_finder=finder, compute_chords=False)
    assert table["attachment_found"].tolist() == [True, False]
    assert table.loc[0, "attachment_method"] == "down_facing_cap:axis_ray"
    assert table.loc[0, list(JUNCTION_METRICS)].notna().all()
    assert table.loc[0, "Length"] > 1000  # base centre -> tip of a 1500 nm cylinder
