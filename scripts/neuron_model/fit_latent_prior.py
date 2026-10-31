"""Learned prior for a trained VAE: fit N(0, I) alternatives to the encoder codes of real spines and compare them.

The codes (aggregate posterior) of train spines are fitted by
    gaussian   one full-covariance Gaussian (what ``generate_vae.py --prior data`` does on the fly),
    gmm<k>     Gaussian mixtures (--gmm 4 8 ...),
    flow       RealNVP normalizing flow: whitening by the fitted Gaussian + affine coupling layers, z <-> eps ~ N(0, I)
               bijectively; it starts as the Gaussian and learns only the non-Gaussian remainder.
Compared on VALIDATION codes (other neurons): mean negative log-likelihood per latent dimension (lower = better density
model of real codes) and the hole ratio of 4000 prior samples (NN distance to a validation code relative to the NN
distance between validation codes; ~1 = samples land where real spines are, >> 1 = in empty regions) and the coverage
ratio (NN distance from a validation code to the nearest sample; large = real spines without a sample near them, i.e.
a too narrow prior). Both are compared with the same number of real train codes (the floor).

Training codes: mu of every resampling in the run's ``data.variants`` (each resampling = one more sample of the
aggregate posterior) plus --posterior-samples draws z ~ q(z|x) per spine (0 = only mu).

    python scripts/neuron_model/fit_latent_prior.py --run runs/training/neuron-model/2026-10-19_003408_vae-dev-reg0-cos \
        --n-train 15000 --n-val 1500 --gmm 4 8 16

Then meshes from the flow (``meshes_flow_prior/``), evaluated like any generated set:

    python scripts/neuron_model/generate_vae.py --run <run> --prior learned --latent-prior <run>/latent_prior/flow_prior.pt \
        --n-samples 200 --seed 1 --temperature 1.0

Writes ``<run>/latent_prior/{codes_train.npz, codes_val.npz, gaussian_prior.npz, flow_prior.pt, report.json}``; encoded
codes are reused on a re-run with the same checkpoint / counts / seed (``--re-encode`` forces).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

if sys.platform == "win32":
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np

from src.neuron_model.spine_generation.experiment.artifacts import write_json  # noqa: E402
from src.neuron_model.spine_generation.models.spine_vae.latent_prior import (  # noqa: E402
    GaussianPrior, StandardNormalPrior, encode_spines, fit_flow, hole_ratio, load_split_index, load_vae_run, nll_per_dim, save_prior,
)


def get_codes(model, cfg, args, split, n, device, out_dir, checkpoint_id, variants):
    path = out_dir / f"codes_{split}.npz"
    tag = f"{checkpoint_id}|{split}|{n}|{args.seed}|{variants}"
    if path.exists() and not args.re_encode:
        data = np.load(path, allow_pickle=False)
        if str(data["tag"]) == tag:
            print(f"[prior] reuse {path.name} ({data['mu'].shape[0]} spines)", flush=True)
            return data["mu"], data["logvar"]
    index = load_split_index(cfg, split, n, args.seed, manifests=args.manifest, splits=args.splits, path_mode=args.path_mode)
    codes = encode_spines(model, index, cfg["data"], variants=variants, device=device, batch_size=args.batch_size, progress=True)
    np.savez(path, mu=codes["mu"], logvar=codes["logvar"], keys=np.array(codes["keys"]), tag=np.array(tag))
    return codes["mu"], codes["logvar"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--checkpoint", default="latest")
    parser.add_argument("--n-train", type=int, default=15000)
    parser.add_argument("--n-val", type=int, default=1500)
    parser.add_argument("--variants", type=int, nargs="+")
    parser.add_argument("--posterior-samples", type=int, default=0)
    parser.add_argument("--gmm", type=int, nargs="*", default=[4, 8, 16])
    parser.add_argument("--flow-layers", type=int, default=8)
    parser.add_argument("--flow-hidden", type=int, default=256)
    parser.add_argument("--flow-epochs", type=int, default=300)
    parser.add_argument("--flow-noise", type=float, default=0.05, help="training noise in whitened units (regularisation)")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--re-encode", action="store_true")
    parser.add_argument("--manifest", type=Path, nargs="+")
    parser.add_argument("--splits", type=Path)
    parser.add_argument("--path-mode", choices=("relative", "absolute"))
    args = parser.parse_args()

    model, run, cfg, checkpoint_id, device = load_vae_run(args.run, args.checkpoint)
    variants = args.variants or [int(v) for v in cfg["data"].get("variants", (1, 2, 3, 4))]
    out_dir = run.root / "latent_prior"
    out_dir.mkdir(parents=True, exist_ok=True)
    mu_tr, lv_tr = get_codes(model, cfg, args, "train", args.n_train, device, out_dir, checkpoint_id, variants)
    mu_va, _ = get_codes(model, cfg, args, "val", args.n_val, device, out_dir, checkpoint_id, variants)

    rng = np.random.default_rng(args.seed)
    train = mu_tr.reshape(-1, mu_tr.shape[-1])
    if args.posterior_samples > 0:
        sig = np.exp(0.5 * lv_tr)
        draws = [(mu_tr + sig * rng.standard_normal(mu_tr.shape)).reshape(-1, mu_tr.shape[-1]) for _ in range(args.posterior_samples)]
        train = np.concatenate([train, *draws])
    val = mu_va[:, 0]  # one code per validation spine (first resampling)
    dim = train.shape[1]
    print(f"[prior] {checkpoint_id}: {len(train)} training codes ({mu_tr.shape[0]} spines x {len(variants)} resamplings"
          f"{f' + {args.posterior_samples} posterior draws' if args.posterior_samples else ''}), {len(val)} validation codes, D {dim}", flush=True)

    m = 4000
    report = {"checkpoint_id": checkpoint_id, "n_train_codes": int(len(train)), "n_val_codes": int(len(val)), "priors": {}}

    def score(name, prior, samples):
        entry = {"val_nll_per_dim": nll_per_dim(prior, val) if prior is not None else None, "hole_ratio": hole_ratio(samples, val)}
        report["priors"][name] = entry
        nll = "-" if entry["val_nll_per_dim"] is None else f"{entry['val_nll_per_dim']:.4f}"
        h = entry["hole_ratio"]
        print(f"  {name:16s} val NLL/dim {nll:>8s}   holes (sample->code) {h['median_ratio']:.3f}, p90 {h['p90_ratio']:.3f}; coverage (code->sample) {h['coverage_ratio']:.3f}", flush=True)

    print("[prior] validation codes:")
    score("standard_normal", StandardNormalPrior(dim), StandardNormalPrior(dim).sample(m, seed=args.seed))
    gauss = GaussianPrior.fit(train)
    score("gaussian", gauss, gauss.sample(m, seed=args.seed))
    save_prior(gauss, out_dir)
    for k in args.gmm or []:
        from sklearn.mixture import GaussianMixture

        gmm = GaussianMixture(n_components=k, covariance_type="full", reg_covar=1e-4, random_state=args.seed).fit(train)

        class _G:
            def log_prob(self, z, g=gmm):
                return g.score_samples(z)

        score(f"gmm{k}", _G(), gmm.sample(m)[0])
    flow, info = fit_flow(train, val, n_layers=args.flow_layers, hidden=args.flow_hidden, epochs=args.flow_epochs,
                          noise_std=args.flow_noise, seed=args.seed)
    score("flow", flow, flow.sample(m, seed=args.seed))
    report["flow"] = {k: v for k, v in info.items() if k != "history"}
    report["flow"]["config"] = {"layers": args.flow_layers, "hidden": args.flow_hidden, "noise": args.flow_noise}
    for t in (0.7, 0.85):
        report["priors"][f"flow_T{t}"] = {"hole_ratio": hole_ratio(flow.sample(m, temperature=t, seed=args.seed), val)}
        h = report["priors"][f"flow_T{t}"]["hole_ratio"]
        print(f"  flow T={t:<10} {'':22s} holes {h['median_ratio']:.3f}; coverage {h['coverage_ratio']:.3f}")
    report["priors"]["real_train_codes"] = {"hole_ratio": hole_ratio(mu_tr[rng.permutation(len(mu_tr))[:m], 0], val)}
    h = report["priors"]["real_train_codes"]["hole_ratio"]
    print(f"  {'real train codes':16s} {'':22s} holes {h['median_ratio']:.3f}; coverage {h['coverage_ratio']:.3f}  (the floor: the same number of real codes)")
    save_prior(flow, out_dir)
    write_json(out_dir / "report.json", report)
    print(f"-> {out_dir}")


if __name__ == "__main__":
    main()
