"""Web application of the branch generator on a tiny synthetic bundle (synthetic T0 / P0 fits, mushroom spine meshes)."""

from __future__ import annotations

import io
import json
import math
import zipfile

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("manifold3d")
import trimesh

from src.neuron_model.placement_generation.covariates import BranchCovariates, CountDesign
from src.neuron_model.placement_generation.models.count import CountModel, build_count_data
from src.neuron_model.placement_generation.models.position import PositionBranch, PositionModel
from src.neuron_model.trunk_generation.conditions import DesignEncoder, group_key
from src.neuron_model.trunk_generation.constraints import TrunkLimits
from src.neuron_model.trunk_generation.models.t0_statistical import T0Branch, T0Model
from src.neuron_model.webapp import BranchGenerator, ModelBundle, RequestError, build_bundle
from src.neuron_model.webapp import export, metrics
from src.neuron_model.webapp.bundle import SpineLibrary, library_key_chain

GROUP = ("mouse", "excitatory", "apical")


def _t0_branches(rng, n=60):
    """Straight-ish trunks with a parabolic radius profile (as in test_t0._simulate, one group)."""
    enc = DesignEncoder.fit([6000.0, 30000.0])
    out = []
    for _ in range(n):
        length = float(np.exp(rng.uniform(math.log(6000.0), math.log(30000.0))))
        m = max(int(math.ceil(length / 500.0)) + 1, 6)
        step = length / (m - 1)
        u = np.arange(m) / (m - 1)
        r_l = rng.uniform(220.0, 320.0)
        log_r = np.log(r_l * (1 - (1 - u) ** 2) + 1.3 * r_l * (1 - u) ** 2) + rng.normal(0, 0.03, m)
        out.append(T0Branch(rng.normal(0, 3e-5, m - 1), rng.normal(0, 3e-5, m - 1), log_r, step, enc.row(*GROUP, length), group_key(*GROUP)))
    return out, enc


def _p0(rng):
    cov, pos = [], []
    for i in range(120):
        length = float(rng.uniform(5000, 40000))
        radius = float(rng.uniform(200, 350))
        c = BranchCovariates(f"n{i // 6}", -1, length, 2 * np.pi * radius * length, radius, float(rng.uniform(5000, 150000)), 1.05, bool(i % 3 == 0), *GROUP)
        c.n_spines = int(rng.poisson(0.8 * length / 1000.0))
        cov.append(c)
        s = np.linspace(0, length, int(length / 500) + 1)
        b = PositionBranch(s, np.full(len(s), radius), np.full(len(s), 1e-4), c.soma_distance_nm)
        b.positions = np.sort(rng.uniform(0, length, c.n_spines))
        pos.append(b)
    design = CountDesign().fit(cov)
    count = CountModel("poisson", random_effect=False).fit(build_count_data(design, cov, "length"), max_iter=60)
    position = PositionModel(use=()).fit(pos)
    return {"design": design.to_dict(), "count_model": count.to_dict(), "position_model": position.to_dict(), "d_min_nm": 200.0}


def _spine(neck_r=60.0, neck_len=600.0, head_r=200.0):
    """Mushroom spine in the local frame (as in test_branch_assembly): flat base at y = -20, head along +y."""
    import manifold3d as m3d

    def mf(mesh):
        return m3d.Manifold(m3d.Mesh64(vert_properties=np.asarray(mesh.vertices, float), tri_verts=np.asarray(mesh.faces, np.uint64)))

    neck = trimesh.creation.cylinder(radius=neck_r, height=neck_len, sections=16)
    neck.apply_transform(trimesh.transformations.rotation_matrix(-np.pi / 2, [1, 0, 0]))
    neck.apply_translation([0, neck_len / 2 - 20.0, 0])
    head = trimesh.creation.icosphere(2, head_r)
    head.apply_translation([0, neck_len - 20.0 + 0.7 * head_r, 0])
    out = (mf(neck) + mf(head)).to_mesh64()
    return trimesh.Trimesh(np.asarray(out.vert_properties)[:, :3], np.asarray(out.tri_verts), process=True)


def _write_spines(folder, params):
    folder.mkdir(parents=True, exist_ok=True)
    for i, p in enumerate(params):
        _spine(*p).export(folder / f"spine_{i}.off")
    (folder / "broken.off").write_text("OFF\n3 1 0\n0 0 0\n1 0 0\n0 1 0\n3 0 1 2\n")  # open: rejected
    return folder


def _branches_table(rng, n=60):
    """Minimal branches.parquet for the reference quantiles."""
    length = rng.uniform(5000, 60000, n)
    prox = rng.uniform(250, 400, n)
    return pd.DataFrame({"status": "ok", "species": "mouse", "cell_type_binary": "excitatory", "compartment": "apical",
                         "train_eligible_T": True, "train_eligible_P": True, "length_nm": length, "radius_median_nm": prox * 0.9,
                         "radius_proximal_nm": prox, "radius_distal_nm": prox * 0.8, "tortuosity": rng.uniform(1.0, 1.2, n),
                         "kappa_rms": rng.uniform(1e-4, 3e-4, n), "n_spines": rng.poisson(length / 1000.0)})


@pytest.fixture(scope="module")
def bundle_dir(tmp_path_factory):
    rng = np.random.default_rng(0)
    tmp = tmp_path_factory.mktemp("webapp")
    t0_run, p0_run = tmp / "t0_run", tmp / "p0_run"
    t0_run.mkdir()
    p0_run.mkdir()
    branches, enc = _t0_branches(rng)
    T0Model("parabola", direction="kappa", radius_residual="pooled").fit(branches, encoder=enc, max_iter=30).save(t0_run / "t0_parabola.json")
    (t0_run / "limits.json").write_text(json.dumps(TrunkLimits().to_dict()))
    (t0_run / "report.json").write_text(json.dumps({"chosen": "parabola"}))
    (p0_run / "p0_bundle.json").write_text(json.dumps(_p0(rng)))
    spines = {"any": _write_spines(tmp / "spines_any", [(60, 600, 200), (70, 500, 180)]),
              "mouse/excitatory/BPC": _write_spines(tmp / "spines_bpc", [(50, 700, 220)])}
    return build_bundle(tmp / "bundles", t0_run=t0_run, p0_run=p0_run, spines=spines, bundle_id="test", compute_morphometrics=False,
                        branches_reference=_branches_table(rng), limits={"max_length_nm": 20000.0}, log=lambda *_: None)


@pytest.fixture(scope="module")
def generator(bundle_dir):
    return BranchGenerator(ModelBundle.load(bundle_dir))


REQUEST = {"species": "mouse", "cell_type_binary": "excitatory", "compartment": "apical", "length_nm": 8000.0,
           "soma_distance_start_nm": 30000.0, "is_terminal": False, "seed": 3}


def test_bundle_round_trip(bundle_dir):
    b = ModelBundle.load(bundle_dir)
    man = b.manifest
    assert man["bundle_id"] == "test" and man["units"] == "nm"
    assert man["categories"]["species"] == ["mouse"] and man["categories"]["spine_cell_type"] == ["BPC"]
    assert man["spine_cell_type_binary"] == {"BPC": "excitatory"}
    assert man["spine_library"]["any"]["n_spines"] == 2 and man["spine_library"]["any"]["n_rejected"] == 1
    assert man["limits"]["max_length_nm"] == 20000.0 and man["trained_groups"] == ["mouse|excitatory|apical"]
    lib = b.library("any")
    assert len(lib) == 2 and np.isfinite(lib.base_radius_nm).all()
    # attachment centre of the synthetic spine: the flat base at y = -20 on the axis
    assert np.allclose(lib.center[:, 1], -20.0, atol=5.0)
    path = bundle_dir / "copy.npz"
    lib.save(path)
    again = SpineLibrary.load("any", path)
    assert all(np.array_equal(a, c) for a, c in zip(lib.vertices, again.vertices)) and again.source == lib.source
    assert set(b.reference["trunk_placement"]["groups"]) == {"mouse|excitatory|apical"}


def test_library_key_chain():
    assert library_key_chain("mouse", "excitatory", "BPC") == ["mouse/excitatory/BPC", "mouse/excitatory", "mouse", "any"]
    assert library_key_chain("mouse", "excitatory", None) == ["mouse/excitatory", "mouse", "any"]


def test_generation_is_deterministic_for_a_seed(generator):
    a, b = generator.generate(REQUEST), generator.generate(dict(REQUEST))
    assert a.result_id == b.result_id
    assert np.array_equal(a.branch_mesh[0], b.branch_mesh[0]) and np.array_equal(a.branch_mesh[1], b.branch_mesh[1])
    assert [r["s_nm"] for r in a.metrics["spines"]] == [r["s_nm"] for r in b.metrics["spines"]]
    c = generator.generate({**REQUEST, "seed": 4})
    assert c.result_id != a.result_id and not np.array_equal(a.trunk["gamma"], c.trunk["gamma"])


def test_generated_branch_is_consistent(generator):
    res = generator.generate(REQUEST)
    assert res.trunk["length"] == pytest.approx(8000.0)
    assert res.assembly.branch["status"] in ("ok", "invalid")
    placed = [r for r in res.metrics["spines"] if r["status"] == "placed"]
    assert len(placed) == len(res.spines_global) == len(res.spines_local) == res.metrics["placement"]["n_placed"]
    for r in placed:
        v, _ = res.spines_global[r["index"]]
        # the placed spine reaches its anchor: some vertex within the base band of the anchor point
        assert np.linalg.norm(v - np.asarray(r["anchor_nm"]), axis=1).min() < 60.0


def test_request_validation_collects_every_error(generator):
    with pytest.raises(RequestError) as err:
        generator.generate({**REQUEST, "species": "rat", "length_nm": 1e9, "seed": -1, "spine_cell_type": "XYZ", "colour": "red"})
    fields = {e["field"] for e in err.value.errors}
    assert {"species", "length_nm", "seed", "spine_cell_type", "colour"} <= fields
    with pytest.raises(RequestError):
        generator.generate({**REQUEST, "is_terminal": "maybe"})
    with pytest.raises(RequestError):
        generator.generate({**REQUEST, "seed": 1.5})
    q = generator.normalize({"seed": "7"})  # defaults fill the rest
    assert q["seed"] == 7 and q["species"] == "mouse"


def test_library_fallback_is_reported(generator):
    res = generator.generate(REQUEST)
    assert res.library["used_key"] == "any" and res.library["fallback"] and res.library["requested_keys"][0] == "mouse/excitatory"
    assert any("библиотеки шипиков" in w for w in res.warnings)
    exact = generator.generate({**REQUEST, "spine_cell_type": "BPC"})
    assert exact.library["used_key"] == "mouse/excitatory/BPC" and not exact.library["fallback"]
    assert {r["library_key"] for r in exact.metrics["spines"] if r["status"] == "placed"} <= {"mouse/excitatory/BPC"}


def test_metrics_keys_and_units(generator):
    res = generator.generate(REQUEST)
    s = res.summary()
    trunk = {r["key"]: r for r in s["trunk"]}
    for key in ("length_nm", "tortuosity", "radius_median_nm", "radius_proximal_nm", "radius_distal_nm", "taper_ratio", "kappa_rms_per_nm"):
        assert key in trunk
    assert trunk["length_nm"]["unit"] == "мкм" and trunk["length_nm"]["display"] == pytest.approx(8.0)
    assert trunk["kappa_rms_per_nm"]["display"] == pytest.approx(trunk["kappa_rms_per_nm"]["value"] * 1e3)
    place = {r["key"]: r for r in s["placement"]}
    for key in ("n_requested", "n_placed", "density_placed_per_um", "gap_cv", "pcf_0_500", "d_min_nm"):
        assert key in place
    assert place["density_placed_per_um"]["value"] == pytest.approx(place["n_placed"]["value"] / 8.0)
    cols = {c["key"] for c in s["spine_columns"]}
    assert {"s_nm", "u", "theta", "immersion_nm", "Volume", "Length"} <= cols
    assert s["reference"]["available"] and s["reference"]["group"] == "mouse|excitatory|apical"
    assert {r["key"] for r in s["reference"]["rows"]} >= {"length_nm", "tortuosity"}
    json.loads(export.dumps(s))  # JSON-safe (no NaN)


def test_zip_contents(generator):
    res = generator.generate(REQUEST)
    z = zipfile.ZipFile(io.BytesIO(export.build_zip(res)))
    names = set(z.namelist())
    assert {"branch.off", "trunk.off", "metrics.json", "spines.csv", "trunk_centerline.csv", "request.json", "README.txt"} <= names
    placed = sorted(res.spines_global)
    assert {f"spines_local/spine_{k:03d}.off" for k in placed} | {f"spines_global/spine_{k:03d}.off" for k in placed} <= names
    req = json.loads(z.read("request.json"))
    assert req["bundle_id"] == "test" and req["seed"] == 3
    again = trimesh.load(io.BytesIO(z.read("branch.off")), file_type="off", process=False)
    assert len(again.faces) == len(res.branch_mesh[1])
    spines = pd.read_csv(io.BytesIO(z.read("spines.csv")))
    assert len(spines) == len(res.metrics["spines"]) and "anchor_x_nm" in spines.columns
    line = pd.read_csv(io.BytesIO(z.read("trunk_centerline.csv")))
    assert list(line.columns) == ["s_nm", "x_nm", "y_nm", "z_nm", "r_nm"] and line["s_nm"].iloc[-1] == pytest.approx(8000.0)
    # the reproduction recipe: request.json without the bundle id regenerates the same branch
    req.pop("bundle_id")
    assert generator.generate(req).result_id == res.result_id


def test_reference_from_branches_quantiles():
    rng = np.random.default_rng(1)
    t = _branches_table(rng, 80)
    ref = metrics.reference_from_branches(t)
    q = ref["groups"]["mouse|excitatory|apical"]["metrics"]["length_nm"]
    assert q == sorted(q) and q[2] == pytest.approx(np.median(t["length_nm"]))
    assert ref["all"]["n_branches_T"] == 80


def test_library_morphometrics(tmp_path):
    pytest.importorskip("CGAL")
    from src.neuron_model.webapp.bundle import build_spine_library

    folder = _write_spines(tmp_path / "s", [(60, 600, 200)])
    lib, rejected = build_spine_library("any", sorted(folder.glob("*.off")), folder, compute_morphometrics=True)
    assert len(lib) == 1 and len(rejected) == 1
    m = lib.spine_metrics(0)
    assert m["Volume"] and m["Volume"] > 0 and m["Length"] and 700 < m["Length"] < 1200


# ---------------------------------------------------------------------------
# server (skipped without fastapi)
# ---------------------------------------------------------------------------


def test_server_endpoints(bundle_dir):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from src.neuron_model.webapp.server import create_app

    client = TestClient(create_app(bundle_dir))
    assert client.get("/api/health").json()["status"] == "ok"
    opts = client.get("/api/options").json()
    assert opts["bundle"]["bundle_id"] == "test" and "species" in opts["categories"]
    bad = client.post("/api/generate", json={**REQUEST, "species": "rat"})
    assert bad.status_code == 422 and bad.json()["errors"][0]["field"] == "species"
    r = client.post("/api/generate", json=REQUEST).json()
    rid = r["id"]
    assert client.get(f"/api/results/{rid}/branch.off").text.startswith("OFF")
    assert client.get(f"/api/results/{rid}/trunk.off").status_code == 200
    placed = [s["index"] for s in r["spines"] if s["status"] == "placed"]
    if placed:
        assert client.get(f"/api/results/{rid}/spines/{placed[0]}.off?frame=global").text.startswith("OFF")
    assert client.get(f"/api/results/{rid}/spines/9999.off").status_code == 404
    z = zipfile.ZipFile(io.BytesIO(client.get(f"/api/results/{rid}/download.zip").content))
    assert "branch.off" in z.namelist()
    assert client.get(f"/api/results/{rid}/metrics.json").json()["request"]["seed"] == 3
    assert client.get("/api/results/0000000000000000/branch.off").status_code == 404
    assert client.get("/").status_code == 200


def test_result_store_evicts_and_regenerates_from_disk(generator, tmp_path):
    from src.neuron_model.webapp.server import ResultStore

    store = ResultStore(generator, size=1, cache_dir=tmp_path)
    a, b = generator.generate(REQUEST), generator.generate({**REQUEST, "seed": 9})
    store.put(a)
    store.put(b)  # evicts a from memory
    assert store.peek(a.result_id) is None and store.peek(b.result_id) is not None
    again, _ = store.get(a.result_id)  # regenerated from the request on disk
    assert np.array_equal(again.branch_mesh[0], a.branch_mesh[0])
    assert store.get("ffffffffffffffff") is None and store.get("../etc") is None
