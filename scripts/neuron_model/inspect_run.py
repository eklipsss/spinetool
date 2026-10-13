"""Summarise a training run: text report + curves PNG + one HTML page with the saved meshes.

    python scripts/neuron_model/inspect_run.py runs/training/neuron-model/<experiment_id> \
        [--manifest O:/Datasets/Minnie65/preprocessed/manifest.parquet]

Writes into ``<run>/report/``:

- ``summary.txt``   what to paste into the chat: speed, loss trend, latent statistics, NaN check,
                    mesh validity per sampling step (also printed);
- ``curves.png``    train/val loss (all logged loss terms) and latent statistics over steps;
- ``samples_step_<N>.html``  generated meshes of the latest (or ``--step``) sampling step. With
                    ``--manifest`` every reconstruction is shown next to the real spine it encodes.

Safe to run while training is still going (it only reads the logs / meshes).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd

from src.neuron_model.spine_generation.experiment.metrics_logger import read_metrics

LOSS_SUFFIXES = ("loss", "sdf", "kl", "surface", "eikonal", "normal")


def _fmt(value) -> str:
    return "nan" if value is None or (isinstance(value, float) and np.isnan(value)) else f"{value:.4g}"


def build_summary(run: Path) -> str:
    train = read_metrics(run / "logs" / "metrics_train.jsonl")
    val = read_metrics(run / "logs" / "metrics_val.jsonl")
    samples = read_metrics(run / "logs" / "metrics_samples.jsonl")
    lines = [f"run: {run.name}"]
    info = run / "run_info.json"
    if info.exists():
        meta = json.loads(info.read_text(encoding="utf-8"))
        lines.append(f"split: {meta.get('split_version')}  n_train={meta.get('n_train')} n_val={meta.get('n_val')}  git={str(meta.get('git', ''))[:80]}")
    if train.empty:
        return "\n".join(lines + ["no train log yet"])

    last = train.iloc[-1]
    lines.append(f"\n== training: {len(train)} log points, last step {int(last['step'])} ==")
    if "steps_per_second" in train:
        sps = train["steps_per_second"].tail(10)
        lines.append(f"speed: {_fmt(sps.mean())} steps/s (last 10 log points; min {_fmt(sps.min())}, max {_fmt(sps.max())})")
    nan_cols = [c for c in train.columns if c not in ("step", "time") and train[c].isna().any()]
    lines.append("NaN in train log: " + (", ".join(nan_cols) if nan_cols else "none"))
    loss_cols = [c for c in train.columns if c in ("total_loss", "loss") or c.split("_")[0] in LOSS_SUFFIXES and c != "steps_per_second"]
    first_rows = train.head(1).iloc[0]
    for column in dict.fromkeys(loss_cols):
        lines.append(f"  {column}: first {_fmt(first_rows[column])} -> last {_fmt(last[column])}")
    for column in ("grad_norm", "lr"):
        if column in train:
            lines.append(f"  {column}: last {_fmt(last[column])}; max {_fmt(train[column].max())}")

    if not val.empty:
        lines.append(f"\n== validation ({len(val)} points) ==")
        cols = [c for c in val.columns if c not in ("time",)]
        lines.append(val[cols].tail(8).to_string(index=False, float_format=lambda v: f"{v:.4g}"))
        if "active_latent_dimensions" in "".join(val.columns) and "val_active_latent_dimensions" in val:
            active = float(val["val_active_latent_dimensions"].iloc[-1])
            lines.append(f"active latent dims (last): {active:g}" + ("   <-- posterior collapse?" if active < 3 else ""))
    if not samples.empty:
        lines.append(f"\n== mesh sampling ({len(samples)} points) ==")
        cols = [c for c in samples.columns if c not in ("time",)]
        lines.append(samples[cols].to_string(index=False, float_format=lambda v: f"{v:.4g}"))
    return "\n".join(lines)


def plot_curves(run: Path, out: Path) -> bool:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    train = read_metrics(run / "logs" / "metrics_train.jsonl")
    val = read_metrics(run / "logs" / "metrics_val.jsonl")
    if train.empty:
        return False
    skip = {"step", "time", "steps_per_second", "lr"}
    terms = [c for c in train.columns if c not in skip and c != "grad_norm"]
    latent_cols = [c for c in val.columns if "latent" in c or c in ("val_mean_abs_mu", "val_mean_std_z", "val_kl_raw")] if not val.empty else []
    n = 2 + (1 if latent_cols else 0)
    fig, axes = plt.subplots(1, n, figsize=(5.5 * n, 4))
    ax = axes[0]
    for c in terms:
        if c.startswith(("total", "loss")):
            ax.plot(train["step"], train[c], label=f"train {c}")
    if not val.empty and "val_loss" in val:
        ax.plot(val["step"], val["val_loss"], "o-", label="val_loss")
    ax.set_xlabel("step")
    ax.set_title("total loss")
    ax.set_yscale("log")
    ax.legend()
    ax = axes[1]
    for c in terms:
        if not c.startswith(("total", "loss")):
            ax.plot(train["step"], train[c], label=c)
    ax.set_xlabel("step")
    ax.set_title("loss terms (train)")
    ax.set_yscale("log")
    ax.legend()
    if latent_cols:
        ax = axes[2]
        for c in latent_cols:
            ax.plot(val["step"], val[c], "o-", label=c)
        ax.set_xlabel("step")
        ax.set_title("latent statistics (val)")
        ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return True


def _mesh_trace(path: Path, color: str, name: str, x_offset: float, z_offset: float = 0.0):
    import plotly.graph_objects as go
    import trimesh

    mesh = trimesh.load(path, process=False)
    v = np.asarray(mesh.vertices, dtype=float).copy()
    v[:, 0] += x_offset
    v[:, 2] += z_offset
    f = np.asarray(mesh.faces)
    return go.Mesh3d(x=v[:, 0], y=v[:, 1], z=v[:, 2], i=f[:, 0], j=f[:, 1], k=f[:, 2], color=color, opacity=1.0, name=name, flatshading=True, showlegend=True)


def render_samples(run: Path, step: int | None, manifest: Path | None, out_dir: Path) -> Path | None:
    import plotly.graph_objects as go

    meshes = run / "meshes"
    tags = sorted(p.name for p in meshes.glob("step_*") if p.is_dir()) if meshes.exists() else []
    if not tags:
        return None
    tag = f"step_{step:07d}" if step is not None else tags[-1]
    real_paths = {}
    if manifest is not None:
        from src.neuron_model.spine_generation.data.index import load_spine_index

        index = load_spine_index([manifest], check_files=())
        real_paths = dict(zip(index.frame["spine_key"], index.frame["local_sealed_mesh_path"]))

    fig = go.Figure()
    gap = 4500.0  # nm between shapes along x (shapes are ~1-3 um)
    placed = 0
    for mode, row in (("reconstruction", 0), ("prior", 1)):
        for sample_dir in sorted((meshes / tag / mode).glob("sample_*")):
            mesh_path = sample_dir / "generated_mesh_postprocessed.off"
            if not mesh_path.exists():
                mesh_path = sample_dir / "generated_mesh_raw.off"
            if not mesh_path.exists():
                continue
            meta = json.loads((sample_dir / "generation_metadata.json").read_text(encoding="utf-8"))
            key = meta.get("generation_config", {}).get("spine_key")
            index = int(meta["sample_index"])
            x0 = index * gap
            z0 = row * 2 * gap
            fig.add_trace(_mesh_trace(mesh_path, "#FD89C3", f"{mode} {index} (generated)", x0, z0))
            placed += 1
            if mode == "reconstruction" and key in real_paths and Path(real_paths[key]).exists():
                fig.add_trace(_mesh_trace(Path(real_paths[key]), "#B5B8BC", f"{mode} {index} (real)", x0 + gap / 2, z0))
    if not placed:
        return None
    fig.update_layout(
        title=f"{run.name} - {tag}: reconstructions (pink; grey = the real spine, if --manifest) and prior samples",
        scene=dict(aspectmode="data"),
        height=700,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"samples_{tag}.html"
    fig.write_html(out)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run", type=Path)
    parser.add_argument("--manifest", type=Path, help="manifest.parquet: also draw the real spine next to each reconstruction")
    parser.add_argument("--step", type=int, help="sampling step to render (default: latest)")
    args = parser.parse_args()
    run = args.run
    report = run / "report"
    report.mkdir(parents=True, exist_ok=True)

    summary = build_summary(run)
    (report / "summary.txt").write_text(summary, encoding="utf-8")
    print(summary)
    if plot_curves(run, report / "curves.png"):
        print(f"\ncurves  -> {report / 'curves.png'}")
    html = render_samples(run, args.step, args.manifest, report)
    print(f"samples -> {html}" if html else "samples: none yet (first ones appear at train.sample_every)")


if __name__ == "__main__":
    main()
