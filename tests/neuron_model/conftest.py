import functools
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.neuron_model.spine_generation.data.index import spine_relative_dir  # noqa: E402

_POISSON_PROBE = """
import numpy as np, pymeshlab
p = np.random.default_rng(0).normal(size=(500, 3)); p /= np.linalg.norm(p, axis=1, keepdims=True)
ms = pymeshlab.MeshSet(); ms.add_mesh(pymeshlab.Mesh(vertex_matrix=p, v_normals_matrix=p.copy()))
ms.generate_surface_reconstruction_screened_poisson(depth=5, threads=1)
assert ms.current_mesh().face_number() > 0
"""


@functools.lru_cache(maxsize=1)
def poisson_backend_works() -> bool:
    """pymeshlab Screened Poisson usable in this environment? Probed in a subprocess, because on
    the macOS conda env the pip pymeshlab's bundled libomp clashes with conda's (OpenBLAS) and
    aborts the process ("OMP: Error #15") - see CLAUDE.md "Модели генерации шипиков"."""
    try:
        result = subprocess.run([sys.executable, "-c", _POISSON_PROBE], capture_output=True, text=True, timeout=120)
    except (OSError, subprocess.TimeoutExpired):
        return False
    return result.returncode == 0 and "OMP: Error" not in result.stderr


# --- fake preprocessed dataset (manifest with Windows-style paths + npz files) ---

N_SURFACE, N_NEAR, N_UNIFORM = 20, 30, 10


def _write_spine(root, neuron, limb, branch, spine, rng, scale):
    spine_dir = root / spine_relative_dir(neuron, limb, branch, spine)
    spine_dir.mkdir(parents=True)
    for n in (2048, 4096, 8192):
        for variant in (1, 2, 3, 4):
            points = rng.normal(size=(n, 3)) * scale + variant  # variant-dependent offset
            normals = points / np.linalg.norm(points, axis=1, keepdims=True)
            np.savez_compressed(spine_dir / f"pointcloud_{n}_{variant}.npz", points=points.astype(np.float32), normals=normals.astype(np.float32))
    n_total = N_SURFACE + N_NEAR + N_UNIFORM
    sample_type = np.array([0] * N_SURFACE + [1] * N_NEAR + [2] * N_UNIFORM, dtype=np.uint8)
    normals = rng.normal(size=(n_total, 3)).astype(np.float32)
    normals[sample_type == 2] = np.nan
    np.savez_compressed(
        spine_dir / "sdf_samples.npz",
        query_points=rng.normal(size=(n_total, 3)).astype(np.float32),
        sdf=np.arange(n_total, dtype=np.float32),
        sample_type=sample_type,
        surface_normals=normals,
    )
    return spine_dir


@pytest.fixture()
def fake_dataset(tmp_path):
    """12 neurons x 2 spines; manifest paths look like they came from the Windows workstation."""
    rng = np.random.default_rng(0)
    root = tmp_path / "preprocessed" / "minnie65"
    rows = []
    for n_idx in range(12):
        neuron = f"86469{n_idx:03d}"
        for spine in ("000", "001"):
            _write_spine(root, neuron, "000", "001", spine, rng, scale=100.0 + n_idx)
            win_dir = f"O:\\Datasets\\Minnie65\\preprocessed\\minnie65\\{neuron}\\limb_000\\branch_001\\spines\\spine_{spine}"
            rows.append(
                {
                    "dataset": "minnie65", "neuron_id": neuron, "limb_id": "000", "branch_id": "001", "spine_id": spine,
                    "train_eligible": not (n_idx == 0 and spine == "001"),
                    "species": "mouse", "health": "healthy",
                    "cell_type_binary": "excitatory" if n_idx % 3 else "inhibitory",
                    "compartment": "apical" if n_idx % 2 else "basal",
                    **{f"pointcloud_{n}_path": f"{win_dir}\\pointcloud_{n}_1.npz" for n in (2048, 4096, 8192)},
                    "sdf_samples_path": f"{win_dir}\\sdf_samples.npz",
                    "local_sealed_mesh_path": f"{win_dir}\\local_sealed_spine_{spine}.off",
                    "preprocessing_version": "v1", "config_hash": "abc123",
                }
            )
    pd.DataFrame(rows).to_parquet(root / "manifest.parquet", index=False)
    return root
