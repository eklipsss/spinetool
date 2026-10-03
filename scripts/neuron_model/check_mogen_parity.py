"""PyTorch port vs official JAX MoGen: numerical parity check (plan stage 4.3).

Inputs are the files written by export_mogen_weights.py (JAX env). Run in the
`neuron-model` env (CPU is fine):

    python scripts/neuron_model/check_mogen_parity.py --export data/external/mogen/mouse_mixed_750000

Passes if the network velocity and the short midpoint trajectory agree with the
JAX reference within --atol / --rtol (fp32). Exit code 1 otherwise.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch

from src.neuron_model.spine_generation.models.spine_mogen.flow import sample_midpoint, t_schedule
from src.neuron_model.spine_generation.models.spine_mogen.weights import load_flax_npz, load_pointinfinity_from_flax


def report(name: str, ours: np.ndarray, ref: np.ndarray, atol: float, rtol: float) -> bool:
    diff = np.abs(ours - ref)
    ok = bool(np.allclose(ours, ref, atol=atol, rtol=rtol))
    print(f"{name:16s} max_abs={diff.max():.3e} mean_abs={diff.mean():.3e} ref_scale={np.abs(ref).mean():.3e} -> {'OK' if ok else 'MISMATCH'}")
    return ok


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--export", type=Path, required=True, help="output prefix given to export_mogen_weights.py")
    parser.add_argument("--atol", type=float, default=1e-4)
    parser.add_argument("--rtol", type=float, default=1e-3)
    args = parser.parse_args()

    model = load_pointinfinity_from_flax(load_flax_npz(Path(str(args.export) + "_weights.npz"), "ema_params")).eval()
    print("config:", json.dumps(model.cfg.to_dict()))
    ref = np.load(str(args.export) + "_reference.npz")
    coord = torch.from_numpy(ref["coord"])
    t = torch.from_numpy(ref["t"])
    cond = torch.from_numpy(ref["cond"]) if ref["cond"].shape[1] else None
    schedule = str(ref["sample_schedule"])
    n_steps = len(ref["timesteps"]) - 1

    with torch.no_grad():
        velocity = model(coord, t, cond).numpy()
        timesteps = t_schedule(torch.linspace(0.0, 1.0, n_steps + 1), schedule).numpy()
        end = sample_midpoint(model, coord, n_steps=n_steps, schedule=schedule, cond=cond).numpy()

    ok = report("timesteps", timesteps, ref["timesteps"], 1e-6, 0.0)
    ok &= report("velocity", velocity, ref["velocity"], args.atol, args.rtol)
    ok &= report("trajectory_end", end, ref["trajectory_end"], args.atol * 10, args.rtol)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
