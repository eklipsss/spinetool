"""Nearest spines of every source in the VAE latent space (build_spine_atlas.latent_neighbors)."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "neuron_model" / "build_spine_atlas.py"


def _load():
    spec = importlib.util.spec_from_file_location("build_spine_atlas", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_neighbors_per_source_start_with_the_point_itself_and_skip_points_without_mesh():
    mod = _load()
    rng = np.random.default_rng(0)
    z = rng.normal(size=(12, 8))
    source = [0] * 6 + [1] * 6
    mesh = [k if k != 2 else -1 for k in range(12)]  # point 2 has no mesh in the page
    records = [SimpleNamespace(latent=z[i]) for i in range(12)]
    records[11] = SimpleNamespace(latent=None)
    nb = mod.latent_neighbors(records, mesh, source, 2, k=3)
    assert nb[11] is None
    for i in (0, 4, 7):
        own = nb[i][source[i]]
        assert own[0] == [i, 0.0]
        d = [x[1] for x in own]
        assert d == sorted(d)
    for i in range(11):
        for row in nb[i]:
            assert all(j != 2 and j != 11 for j, _ in row)
            assert len(row) == 3
    # the distances are the full latent distances
    j, d = nb[0][1][0]
    assert abs(d - np.linalg.norm(z[0] - z[j])) < 1e-3
    assert d == min(round(float(np.linalg.norm(z[0] - z[t])), 3) for t in range(6, 11))
