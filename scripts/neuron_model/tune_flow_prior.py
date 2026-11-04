"""Clean protocol for the latent-prior flow: choose the flow settings on VALIDATION, report once on the held-out TEST split.

1. Codes of the frozen VAE: train (cached ``codes_train.npz`` of ``--codes-dir``), validation (cached) and TEST (encoded here once,
   first resampling, cached as ``codes_test.npz``; the test split is that of the run's split file, neurons the VAE never saw).
2. Grid over flow settings (``--noise``, ``--hidden``, ``--posterior-samples``). Each setting is fitted on the train codes with
   early stopping on the validation NLL (``fit_flow``). The setting is CHOSEN by the validation NLL only.
3. The chosen setting AND the reference setting (noise 0.05, hidden 256, no posterior draws = the flow of the report) are then
   evaluated on the test codes: NLL per dimension (mean and SE over codes), hole ratio and coverage ratio of 4000 samples, for
   N(0, I), the fitted Gaussian, the flows and the floor (real train codes). Test numbers are never used to choose anything.

    python scripts/neuron_model/tune_flow_prior.py --run <vae run> --codes-dir latent_prior --device cpu

Writes ``<run>/<out-name>/{grid.csv, test_report.json, flow_<name>.pt, flow_curves_<name>.png}``.
"""

from __future__ import annotations

import argparse
import itertools
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

if sys.platform == "win32":
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.neuron_model.spine_generation.experiment.artifacts import write_json  # noqa: E402
from src.neuron_model.spine_generation.models.spine_vae.flow_curves import write_flow_curves  # noqa: E402
from src.neuron_model.spine_generation.models.spine_vae.latent_prior import (  # noqa: E402
    GaussianPrior, StandardNormalPrior, encode_spines, fit_flow, hole_ratio, load_split_index, load_vae_run, save_prior,
)

REFERENCE = {"noise": 0.05, "hidden": 256, "posterior": 0}


def cached_codes(model, cfg, args, split, n, device, checkpoint_id, variants, out_dir, codes_dir):
    tag = f"{checkpoint_id}|{split}|{n}|{args.seed}|{variants}"
    for folder in (codes_dir, out_dir):
        path = folder / f"codes_{split}.npz"
        if path.exists():
            data = np.load(path, allow_pickle=False)
            if str(data["tag"]) == tag:
                print(f"[tune] reuse {path} ({data['mu'].shape[0]} spines)", flush=True)
                return data["mu"], data["logvar"]
    index = load_split_index(cfg, split, n, args.seed, manifests=args.manifest, splits=args.splits, path_mode=args.path_mode)
    print(f"[tune] encoding {len(index)} {split} spines", flush=True)
    codes = encode_spines(model, index, cfg["data"], variants=variants, device=device, batch_size=args.batch_size, progress=True)
    np.savez(out_dir / f"codes_{split}.npz", mu=codes["mu"], logvar=codes["logvar"], keys=np.array(codes["keys"]), tag=np.array(tag))
    return codes["mu"], codes["logvar"]


def nll_stats(prior, codes):
    per_code = -prior.log_prob(codes) / codes.shape[1]
    return float(per_code.mean()), float(per_code.std(ddof=1) / np.sqrt(len(per_code)))


def name_of(c):
    return f"noise{c['noise']}_h{c['hidden']}_post{c['posterior']}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--checkpoint", default="latest")
    ap.add_argument("--codes-dir", default="latent_prior")
    ap.add_argument("--out-name", default="latent_prior_tuning")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--n-train", type=int, default=15000)
    ap.add_argument("--n-val", type=int, default=1500)
    ap.add_argument("--n-test", type=int, default=1500)
    ap.add_argument("--variants", type=int, nargs="+")
    ap.add_argument("--noise", type=float, nargs="+", default=[0.05, 0.1, 0.2])
    ap.add_argument("--hidden", type=int, nargs="+", default=[128, 256])
    ap.add_argument("--posterior-samples", type=int, nargs="+", default=[0, 2])
    ap.add_argument("--flow-layers", type=int, default=8)
    ap.add_argument("--flow-epochs", type=int, default=300)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--manifest", type=Path, nargs="+")
    ap.add_argument("--splits", type=Path)
    ap.add_argument("--path-mode", choices=("relative", "absolute"))
    args = ap.parse_args()

    model, run, cfg, checkpoint_id, device = load_vae_run(args.run, args.checkpoint, device=args.device)
    variants = args.variants or [int(v) for v in cfg["data"].get("variants", (1, 2, 3, 4))]
    out_dir = run.root / args.out_name
    out_dir.mkdir(parents=True, exist_ok=True)
    codes_dir = run.root / args.codes_dir
    mu_tr, lv_tr = cached_codes(model, cfg, args, "train", args.n_train, device, checkpoint_id, variants, out_dir, codes_dir)
    mu_va, _ = cached_codes(model, cfg, args, "val", args.n_val, device, checkpoint_id, variants, out_dir, codes_dir)
    mu_te, _ = cached_codes(model, cfg, args, "test", args.n_test, device, checkpoint_id, variants, out_dir, codes_dir)
    val, test = mu_va[:, 0], mu_te[:, 0]
    rng = np.random.default_rng(args.seed)
    dim = mu_tr.shape[-1]

    def train_codes(n_post):
        base = mu_tr.reshape(-1, dim)
        if n_post <= 0:
            return base
        sig = np.exp(0.5 * lv_tr)
        return np.concatenate([base] + [(mu_tr + sig * rng.standard_normal(mu_tr.shape)).reshape(-1, dim) for _ in range(n_post)])

    grid = [{"noise": a, "hidden": b, "posterior": c} for a, b, c in itertools.product(args.noise, args.hidden, args.posterior_samples)]
    if REFERENCE not in grid:
        grid.insert(0, dict(REFERENCE))
    rows, flows = [], {}
    for c in grid:
        name = name_of(c)
        print(f"[tune] fit {name}", flush=True)
        flow, info = fit_flow(train_codes(c["posterior"]), val, n_layers=args.flow_layers, hidden=c["hidden"], epochs=args.flow_epochs,
                              noise_std=c["noise"], seed=args.seed, log=None)
        hist = pd.DataFrame(info["history"])
        best = hist.iloc[info["best_epoch"]]
        rows.append({**c, "name": name, "best_epoch": info["best_epoch"] + 1, "val_nll": float(best["val_nll_per_dim"]), "val_se": float(best["val_se"]),
                     "train_eval_nll": float(best["train_eval_nll_per_dim"]), "gap": float(best["val_nll_per_dim"] - best["train_eval_nll_per_dim"])})
        print(f"[tune]   {name}: best epoch {rows[-1]['best_epoch']}, val NLL/dim {rows[-1]['val_nll']:.4f} (gap {rows[-1]['gap']:.3f})", flush=True)
        flows[name] = flow
        write_flow_curves(out_dir / f"curves_{name}", info)
    table = pd.DataFrame(rows)
    table.to_csv(out_dir / "grid.csv", index=False)
    chosen = table.sort_values("val_nll").iloc[0]["name"]
    reference = name_of(REFERENCE)
    print(f"[tune] CHOSEN by validation NLL: {chosen} (reference setting: {reference})", flush=True)

    m = 4000
    report = {"checkpoint_id": checkpoint_id, "n_test_codes": int(len(test)), "chosen": chosen, "reference": reference,
              "selection": "validation NLL per dimension at the early-stopped epoch", "priors": {}}
    gauss = GaussianPrior.fit(train_codes(0))
    candidates = {"standard_normal": StandardNormalPrior(dim), "gaussian": gauss}
    for name in dict.fromkeys([reference, chosen]):
        candidates[f"flow[{name}]"] = flows[name]
        save_prior(flows[name], out_dir / f"flow_{name}")
    for key, prior in candidates.items():
        nll, se = nll_stats(prior, test)
        h = hole_ratio(prior.sample(m, seed=args.seed), test)
        report["priors"][key] = {"test_nll_per_dim": nll, "test_nll_se": se, "hole_ratio": h}
        print(f"  TEST {key:44s} NLL/dim {nll:8.4f} +- {se:.4f}   holes {h['median_ratio']:.3f}   coverage {h['coverage_ratio']:.3f}", flush=True)
    floor = hole_ratio(mu_tr[rng.permutation(len(mu_tr))[:m], 0], test)
    report["priors"]["real_train_codes"] = {"hole_ratio": floor}
    print(f"  TEST {'real train codes (floor)':44s} {'':26s} holes {floor['median_ratio']:.3f}   coverage {floor['coverage_ratio']:.3f}", flush=True)
    write_json(out_dir / "test_report.json", report)
    print(f"-> {out_dir}")


if __name__ == "__main__":
    main()
