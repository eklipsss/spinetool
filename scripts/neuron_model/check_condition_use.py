"""Does a conditional MoGen run actually USE its condition? (diagnostic of the pilot training, CPU or GPU, minutes)

For spines of each value of the first enabled condition block (e.g. cell_type_binary: inhibitory / excitatory, equal
numbers) the flow-matching loss is computed with the SAME noise and times under: the true condition, the null (zero)
condition, and the swapped value. A model that uses the condition has loss(true) < loss(swapped), and the gap is larger
than the noise of the estimate (printed: mean +- s.e. of the per-spine paired difference). Also printed: the size of
the columns of the condition input (Dense_0) per block, against the same columns of the run it was initialised from.

    python scripts/neuron_model/check_condition_use.py --run runs/training/neuron-model/2026-10-28_025253_mogen-conditional \
        --split train --n-per-value 400
"""

from __future__ import annotations

import torch  # noqa: F401  # first: Windows (WinError 127, shm.dll)

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.neuron_model.spine_generation.conditioning import ConditionEncoder, spine_conditions  # noqa: E402
from src.neuron_model.spine_generation.data.index import load_spine_index  # noqa: E402
from src.neuron_model.spine_generation.data.splits import load_splits  # noqa: E402
from src.neuron_model.spine_generation.experiment.config import resolve_path  # noqa: E402
from src.neuron_model.spine_generation.experiment.run import open_run  # noqa: E402
from src.neuron_model.spine_generation.experiment.training import get_device  # noqa: E402
from src.neuron_model.spine_generation.models.spine_mogen.flow import flow_matching_loss  # noqa: E402
from src.neuron_model.spine_generation.models.spine_mogen.train import build_model, point_cloud_datasets  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--split", default="train", choices=["train", "val"])
    parser.add_argument("--n-per-value", type=int, default=400)
    parser.add_argument("--block", default=None, help="condition block to test (default: the first enabled one)")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    run = open_run(args.run)
    cfg = run.load_config()
    d, c = cfg["data"], cfg["condition"]
    enc = ConditionEncoder.load(run.root / "condition_encoder.json")
    block = args.block or (enc.enabled or enc.names)[0]
    device = get_device(cfg.get("device", "auto"))

    index = load_spine_index([resolve_path(p) for p in d["manifests"]], path_mode=d.get("path_mode", "relative"), check_files=())
    index = index.with_split(load_splits(resolve_path(d["splits"])), drop_unassigned=bool(d.get("drop_unassigned", False)))
    conditions = spine_conditions(index.frame, resolve_path(c.get("metadata_root", "datasets")))
    keep = ~conditions["compartment"].isin(set(c.get("exclude_compartments", ["axon_like"])))
    vectors = enc.encode_frame(conditions[keep])
    cond_by_key = dict(zip(index.frame.loc[keep, "spine_key"], vectors))
    train_ds, val_ds = point_cloud_datasets(index, {**d, "jitter_std": 0.0}, cond_by_key)
    ds = train_ds if args.split == "train" else val_ds
    sl = enc.slices()[block]
    values = dict(DEFAULT_VALUES(enc, block))
    onehot = ds.cond[:, sl]
    ok = onehot.sum(1) > 0
    print(f"[use] run {run.root.name}, block {block}, split {args.split}: {int(ok.sum())} spines with a value; per value {onehot.sum(0).astype(int).tolist()} {list(values.values())}")

    rng = np.random.default_rng(args.seed)
    pick = {}
    n = min(args.n_per_value, int(onehot.sum(0)[onehot.sum(0) > 0].min()))
    for v in range(onehot.shape[1]):
        idx = np.flatnonzero(onehot[:, v] > 0)
        if len(idx):
            pick[v] = rng.choice(idx, n, replace=False)

    model = build_model({"init": "run", "init_run": str(run.root), "init_checkpoint": "latest"}).to(device).eval()
    weight = model.Dense_0.weight.detach().cpu().numpy()  # [latent, cond_dim]
    print(f"[use] Dense_0 input columns of block '{block}': norm per column {np.linalg.norm(weight[:, sl], axis=0).round(4).tolist()}; "
          f"other enabled columns {np.linalg.norm(weight[:, : sl.start], axis=0).round(4).tolist() if sl.start else []}; bias norm {float(np.linalg.norm(model.Dense_0.bias.detach().cpu().numpy())):.4f}")

    def losses(indices, cond_fn):
        out = []
        with torch.no_grad():
            for b in range(0, len(indices), args.batch_size):
                items = [ds[int(i)] for i in indices[b: b + args.batch_size]]
                x1 = torch.stack([it["points"] for it in items]).to(device)
                cond = torch.stack([it["cond"] for it in items])
                gen = torch.Generator().manual_seed(args.seed + b)
                # per-spine loss: one spine at a time shares the draws of the same batch position
                for k in range(len(items)):
                    g = torch.Generator().manual_seed(args.seed * 100003 + b + k)
                    out.append(float(flow_matching_loss(model, x1[k: k + 1], schedule=cfg["flow"].get("time_schedule", "mogen_cosine"),
                                                        cond=cond_fn(cond[k: k + 1]).to(device), generator=g)))
        return np.array(out)

    def swap(cond, to):
        out = cond.clone()
        out[:, sl] = 0
        out[:, sl.start + to] = 1.0
        return out

    for v, idx in pick.items():
        true = losses(idx, lambda x: x)
        null = losses(idx, lambda x: torch.zeros_like(x))
        print(f"\n[use] spines with value '{list(values.values())[v]}' (n {len(idx)}): loss true {true.mean():.5f}, null {null.mean():.5f} (null - true {np.mean(null - true):+.5f} +- {np.std(null - true) / np.sqrt(len(idx)):.5f})")
        for w in pick:
            if w == v:
                continue
            other = losses(idx, lambda x, w=w: swap(x, w))
            diff = other - true
            print(f"       swapped to '{list(values.values())[w]}': loss {other.mean():.5f}; swapped - true {diff.mean():+.5f} +- {diff.std() / np.sqrt(len(idx)):.5f}  (> 0 = the condition is used)")


def DEFAULT_VALUES(enc, block):
    from src.neuron_model.spine_generation.conditioning import DEFAULT_BLOCKS

    return [(i, v) for name, values in DEFAULT_BLOCKS if name == block for i, v in enumerate(values)]


if __name__ == "__main__":
    main()
