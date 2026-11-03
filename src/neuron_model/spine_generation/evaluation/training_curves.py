"""Post-hoc training curves: every periodic checkpoint of a finished run evaluated with ONE estimator.

Why post hoc (and not inside the training loop): the running mean of the training log is a different quantity from the
validation loss (train mode vs eval mode, random augmentation / latent sampling / condition dropout vs fixed inputs, a
different subset of spines), so the two curves are not comparable, and validation inside the VAE loop slows the Windows
training ~10x for good. Here both curves are computed by the SAME function in a separate process:

- a FIXED list of validation spines and a FIXED random subset of training spines of the same size (seeded permutation of
  the sorted spine keys), inputs materialised ONCE (point cloud variant 1, no jitter, the first query points of each
  type, far-field points drawn once from a seeded numpy RNG) and fed unchanged to every checkpoint;
- eval mode, ``z = mu`` for the VAE; for MoGen the flow-matching ``(t, noise)`` of every batch come from a seeded
  generator (``n_draws`` independent draws), the same for every checkpoint and for train and validation;
- the VAE total loss twice: with the loss weights AT THE CHECKPOINT STEP (KL warm-up, as trained) and with the FINAL
  weights (a fixed objective, so the curve is not distorted by the warm-up);
- mean and standard error over batches (ratio estimator, batches weighted by size) per checkpoint.

Repeated evaluation of the same checkpoint is bit-identical (same device, same threads).
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from ..data.index import SpineIndex

VAE_COMPONENTS = ("sdf", "surface", "kl", "far", "eikonal", "normal")
TERM_OF = {"sdf": "sdf_loss", "surface": "surface_loss", "kl": "kl_loss", "far": "far_loss", "eikonal": "eikonal_loss", "normal": "normal_loss"}
TITLES = {
    "total_final": "Суммарные потери, веса финального шага",
    "total": "Суммарные потери, веса на шаге чекпоинта",
    "sdf": "SDF (L1)", "surface": "Поверхность |f| на поверхности", "kl": "KL", "far": "Дальняя область (far)",
    "eikonal": "Эйконал", "normal": "Нормаль (1 − cos)",
    "loss": "Flow Matching loss", "loss_null_cond": "Flow Matching loss, нулевое условие",
}


# ----------------------------------------------------------------------------------------------------------------------
# run introspection
# ----------------------------------------------------------------------------------------------------------------------


def detect_kind(cfg: Mapping[str, Any]) -> str:
    """``vae`` (config has ``loss`` + ``sampling``) or ``mogen`` (``flow``)."""
    if "flow" in cfg:
        return "mogen"
    if "loss" in cfg:
        return "vae"
    raise ValueError("cannot tell the run kind (no 'flow' or 'loss' section in its config)")


def list_checkpoints(checkpoint_dir: Path, *, include_latest: bool = True) -> List[Tuple[int, Path]]:
    """``[(step, path)]`` of ``step_<N>.pt`` sorted by step; ``latest.pt`` is added when its step is not among them
    (the last step of a run is rarely a multiple of ``keep_every``)."""
    out: Dict[int, Path] = {}
    for path in Path(checkpoint_dir).glob("step_*.pt"):
        try:
            out[int(path.stem.split("_", 1)[1])] = path
        except ValueError:
            continue
    latest = Path(checkpoint_dir) / "latest.pt"
    if include_latest and latest.exists():
        index_path = Path(checkpoint_dir) / "index.json"
        step = None
        if index_path.exists():
            step = json.loads(index_path.read_text(encoding="utf-8")).get("latest", {}).get("step")
        if step is not None and int(step) not in out:
            out[int(step)] = latest
    return sorted(out.items())


def load_weights(model, payload: Mapping[str, Any], which: str):
    """Load ``ema`` or ``raw`` weights of a checkpoint payload into ``model``; returns False when ``ema`` is absent."""
    state = payload["state"]
    if which == "ema":
        if "ema" not in state:
            return False
        model.load_state_dict(state["ema"]["model"])
    else:
        model.load_state_dict(state["model"])
    return True


def build_model(kind: str, run_root: Path):
    from ..models.spine_mogen.pointinfinity import PointInfinity, PointInfinityConfig
    from ..models.spine_vae.model import SpineVAE, SpineVAEConfig

    cfg_json = json.loads((Path(run_root) / "model_config.json").read_text(encoding="utf-8"))
    if kind == "vae":
        return SpineVAE(SpineVAEConfig.from_mapping(cfg_json))
    return PointInfinity(PointInfinityConfig(**cfg_json))


# ----------------------------------------------------------------------------------------------------------------------
# fixed evaluation inputs
# ----------------------------------------------------------------------------------------------------------------------


def _fixed_keys(index: SpineIndex, split: str, n: int, seed: int) -> List[str]:
    keys = sorted(index.frame.loc[index.frame["split"] == split, "spine_key"])
    order = np.random.default_rng(seed).permutation(len(keys))[:n]
    return [keys[i] for i in sorted(order)]


@dataclass
class EvalInputs:
    """Materialised fixed batches (CPU tensors) of the validation spines and of the training-subset spines."""
    val_batches: List[Dict[str, Any]]
    train_batches: List[Dict[str, Any]]
    val_keys: List[str]
    train_keys: List[str]
    batch_size: int
    seed: int
    fingerprint: str = ""


def _materialise(dataset, batch_size: int, seed: int, num_workers: int) -> List[Dict[str, Any]]:
    from ..data.datasets import make_loader

    np.random.seed(seed % (2**32))  # far-field points are drawn from numpy's global RNG
    loader = make_loader(dataset, batch_size=batch_size, shuffle=False, seed=seed, num_workers=num_workers, pin_memory=False)
    return list(loader)


def build_inputs(kind: str, cfg: Mapping[str, Any], run_root: Path, *, n_spines: int, batch_size: int, seed: int, num_workers: int = 0,
                 manifests: Optional[Sequence[Path]] = None, splits: Optional[Path] = None, path_mode: Optional[str] = None) -> EvalInputs:
    """Rebuild the run's dataset and split (as ``train_vae.py`` / ``train_mogen.py`` do) and fix the evaluation spines.

    The training subset goes through the SAME dataset class as validation: its spines are relabelled ``val`` and the
    ``val`` dataset factory of the run (``sdf_datasets`` / ``point_cloud_datasets``: variant 1, no jitter) builds it."""
    from ..data.bbox import FrozenBBox
    from ..data.index import load_spine_index
    from ..data.splits import load_splits
    from ..experiment.config import resolve_path

    data_cfg = cfg["data"]
    manifests = list(manifests) if manifests else [resolve_path(p) for p in data_cfg["manifests"]]
    mode = path_mode or data_cfg.get("path_mode", "relative")
    index = load_spine_index(manifests, path_mode=mode, check_files=())
    index = index.with_split(load_splits(splits or resolve_path(data_cfg["splits"])), drop_unassigned=bool(data_cfg.get("drop_unassigned", False)))

    cond_by_key = None
    if kind == "mogen" and (cfg.get("condition") or {}).get("enabled"):
        cond_by_key = _mogen_conditions(index, cfg, Path(run_root))
        index = index.subset([k for k in index.frame["spine_key"] if k in cond_by_key])

    n_val_all = int((index.frame["split"] == "val").sum())
    n_train_all = int((index.frame["split"] == "train").sum())
    n = min(int(n_spines), n_val_all, n_train_all)
    if n == 0:
        raise ValueError(f"no spines to evaluate (val {n_val_all}, train {n_train_all})")
    val_keys = _fixed_keys(index, "val", n, seed)
    train_keys = _fixed_keys(index, "train", n, seed + 1)

    def dataset_of(keys: Sequence[str]):
        """The run's own 'val' dataset (deterministic variant / queries / no jitter) over exactly ``keys``."""
        sub = index.subset(keys)
        frame = sub.frame.copy()
        frame["split"] = "val"
        relabelled = SpineIndex(frame, sub.dataset_versions)
        if kind == "vae":
            from ..models.spine_vae.train import sdf_datasets

            bbox = FrozenBBox.load(resolve_path(data_cfg["bbox"]))
            return sdf_datasets(relabelled, data_cfg, far_bbox=(bbox.low, bbox.high))[1]
        from ..models.spine_mogen.train import point_cloud_datasets

        return point_cloud_datasets(relabelled, data_cfg, cond_by_key)[1]

    val_batches = _materialise(dataset_of(val_keys), batch_size, seed, num_workers)
    train_batches = _materialise(dataset_of(train_keys), batch_size, seed, num_workers)
    fingerprint = hashlib.sha1(json.dumps([val_keys, train_keys, batch_size, seed]).encode()).hexdigest()[:12]
    return EvalInputs(val_batches, train_batches, val_keys, train_keys, batch_size, seed, fingerprint)


def _mogen_conditions(index: SpineIndex, cfg: Mapping[str, Any], run_root: Path):
    """Condition vectors of a conditional MoGen run: the run's saved encoder + ``train_mogen.prepare_conditions``."""
    import importlib.util

    script = Path(__file__).resolve().parents[4] / "scripts" / "neuron_model" / "train_mogen.py"
    spec = importlib.util.spec_from_file_location("_train_mogen_script", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # defines functions only; main() is behind __main__
    from ..models.spine_mogen.pointinfinity import PointInfinityConfig

    cond_dim = PointInfinityConfig(**json.loads((run_root / "model_config.json").read_text(encoding="utf-8"))).cond_dim
    return module.prepare_conditions(index, cfg["condition"], cond_dim, resume_dir=run_root)[1]


# ----------------------------------------------------------------------------------------------------------------------
# estimators
# ----------------------------------------------------------------------------------------------------------------------


@dataclass
class BatchLosses:
    """Per-batch values: ``values[metric]`` has shape ``[n_draws, n_batches]``; ``sizes`` = spines per batch."""
    values: Dict[str, np.ndarray] = field(default_factory=dict)
    sizes: np.ndarray = field(default_factory=lambda: np.zeros(0))
    extra: Dict[str, float] = field(default_factory=dict)


def summarize(values: np.ndarray, sizes: np.ndarray) -> Dict[str, float]:
    """Size-weighted mean over batches (draws averaged first) and the ratio-estimator standard error over batches.
    ``draw_sd`` = spread of the overall mean between independent (t, noise) draws (NaN for one draw)."""
    values = np.atleast_2d(np.asarray(values, dtype=np.float64))
    sizes = np.asarray(sizes, dtype=np.float64)
    per_batch = values.mean(axis=0)
    total = sizes.sum()
    mean = float((per_batch * sizes).sum() / total)
    nb = len(per_batch)
    se = float(math.sqrt(nb / (nb - 1)) * math.sqrt((sizes**2 * (per_batch - mean) ** 2).sum()) / total) if nb > 1 else float("nan")
    draw_means = (values * sizes).sum(axis=1) / total
    draw_sd = float(draw_means.std(ddof=1)) if len(draw_means) > 1 else float("nan")
    return {"mean": mean, "se": se, "draw_sd": draw_sd, "n_batches": nb}


def evaluate_vae(model, batches: Sequence[Mapping[str, Any]], weights, *, step: int, final_step: int, device, seed: int = 0) -> BatchLosses:
    """All VAE loss terms per batch, ``z = mu``. ``total`` uses the KL weight of ``step``, ``total_final`` that of ``final_step``.

    The encoder is not deterministic by itself: farthest point sampling starts from a RANDOM point (global torch RNG) and
    "random" samplers draw permutations. The RNG is therefore forked and seeded here, so every checkpoint, the train subset
    and the validation spines see the same draws (and a repeated evaluation is bit-identical)."""
    import torch

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        return _evaluate_vae(model, batches, weights, step=step, final_step=final_step, device=device)


def _evaluate_vae(model, batches, weights, *, step: int, final_step: int, device) -> BatchLosses:
    import torch

    from ..models.spine_vae.latent import kl_weight
    from ..models.spine_vae.losses import combine
    from ..models.spine_vae.train import to_device

    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)  # the eikonal / normal terms only need d f / d query
    kl_w_final = kl_weight(final_step, weights.beta, weights.kl_warmup_steps, weights.kl_start_steps)
    acc: Dict[str, List[float]] = {}
    sizes: List[int] = []
    for batch in batches:
        with torch.enable_grad():
            terms = model.compute_losses(to_device(batch, device), weights, step, sample=False)
        for key, term in TERM_OF.items():
            if term in terms:
                acc.setdefault(key, []).append(float(terms[term].detach()))
        acc.setdefault("total", []).append(float(terms["total_loss"].detach()))
        acc.setdefault("total_final", []).append(float(combine(terms, weights, kl_w_final).detach()))
        sizes.append(int(batch["points"].shape[0]))
    out = BatchLosses({k: np.asarray(v)[None, :] for k, v in acc.items()}, np.asarray(sizes))
    out.extra["kl_weight"] = float(kl_weight(step, weights.beta, weights.kl_warmup_steps, weights.kl_start_steps))
    out.extra["kl_weight_final"] = float(kl_w_final)
    return out


def evaluate_mogen(model, batches: Sequence[Mapping[str, Any]], flow_cfg: Mapping[str, Any], *, device, seed: int, n_draws: int) -> BatchLosses:
    """Flow-matching loss per batch for ``n_draws`` independent fixed ``(t, noise)`` draws (generator seed
    ``seed + 7919 * draw + batch``), with the condition and, for conditional runs, with the null condition."""
    import torch

    from ..models.spine_mogen.flow import flow_matching_loss

    model.eval()
    schedule = flow_cfg.get("time_schedule", "mogen_cosine")
    loss = np.zeros((n_draws, len(batches)))
    null = np.zeros_like(loss)
    has_cond = False
    sizes: List[int] = []
    with torch.no_grad():
        for i, batch in enumerate(batches):
            x1 = batch["points"].to(device)
            cond = batch.get("cond")
            has_cond = has_cond or cond is not None
            sizes.append(int(x1.shape[0]))
            for d in range(n_draws):
                gen = torch.Generator().manual_seed(seed + 7919 * d + i)
                loss[d, i] = float(flow_matching_loss(model, x1, schedule=schedule, cond=None if cond is None else cond.to(device), generator=gen))
                if cond is not None:
                    gen = torch.Generator().manual_seed(seed + 7919 * d + i)  # same noise and times
                    null[d, i] = float(flow_matching_loss(model, x1, schedule=schedule, cond=torch.zeros_like(cond).to(device), generator=gen))
    values = {"loss": loss}
    if has_cond:
        values["loss_null_cond"] = null
    return BatchLosses(values, np.asarray(sizes))


# ----------------------------------------------------------------------------------------------------------------------
# curves
# ----------------------------------------------------------------------------------------------------------------------


def metric_names(kind: str, rows: Sequence[Mapping[str, Any]]) -> List[str]:
    if kind == "vae":
        names = ["total_final", "total"] + list(VAE_COMPONENTS)
    else:
        names = ["loss", "loss_null_cond"]
    return [m for m in names if any(f"val_{m}" in r for r in rows)]


def evaluate_checkpoint(kind: str, model, inputs: EvalInputs, cfg: Mapping[str, Any], *, step: int, final_step: int, device, n_draws: int, seed: int) -> Dict[str, Any]:
    """One curves row: val and train-subset estimates of the same function for the loaded model."""
    row: Dict[str, Any] = {"step": int(step), "n_val": len(inputs.val_keys), "n_train": len(inputs.train_keys)}
    for split, batches in (("val", inputs.val_batches), ("train", inputs.train_batches)):
        if kind == "vae":
            from ..models.spine_vae.train import loss_weights

            result = evaluate_vae(model, batches, loss_weights(cfg["loss"]), step=step, final_step=final_step, device=device, seed=seed)
            row.update({"kl_weight": result.extra["kl_weight"], "kl_weight_final": result.extra["kl_weight_final"]})
        else:
            result = evaluate_mogen(model, batches, cfg["flow"], device=device, seed=seed, n_draws=n_draws)
        for name, values in result.values.items():
            s = summarize(values, result.sizes)
            row[f"{split}_{name}"] = s["mean"]
            row[f"{split}_{name}_se"] = s["se"]
            row[f"{split}_{name}_draw_sd"] = s["draw_sd"]
    for name in metric_names(kind, [row]):
        if f"train_{name}" in row:
            row[f"gap_{name}"] = row[f"val_{name}"] - row[f"train_{name}"]
            row[f"gap_{name}_se"] = math.sqrt(row[f"val_{name}_se"] ** 2 + row[f"train_{name}_se"] ** 2)
    return row


def total_from_components(frame: pd.DataFrame, weights, kl_w: Optional[float]) -> pd.Series:
    """Weighted total of per-window component means (linear, so exact); ``kl_w`` None -> the logged ``kl_weight`` column."""
    kl = frame["kl_weight"] if kl_w is None else kl_w
    total = weights.sdf * frame["sdf_loss"] + kl * frame["kl_loss"]
    for name, w in (("surface", weights.surface), ("eikonal", weights.eikonal), ("normal", weights.normal)):
        if w and f"{name}_loss" in frame:
            total = total + w * frame[f"{name}_loss"]
    if weights.far and "far_loss" in frame:
        total = total + weights.far * frame["far_loss"]
    return total


def read_logs(run_root: Path, kind: str, cfg: Mapping[str, Any], final_step: int) -> Dict[str, pd.DataFrame]:
    """Training log (window means of the loop, train mode) and in-loop validation points, renamed to curve metric names."""
    from ..experiment.metrics_logger import read_metrics

    logs = Path(run_root) / "logs"
    train = read_metrics(logs / "metrics_train.jsonl")
    val = read_metrics(logs / "metrics_val.jsonl")
    out: Dict[str, pd.DataFrame] = {}
    if len(train):
        t = pd.DataFrame({"step": train["step"]})
        if kind == "vae":
            from ..models.spine_vae.latent import kl_weight
            from ..models.spine_vae.train import loss_weights

            w = loss_weights(cfg["loss"])
            for key, term in TERM_OF.items():
                if term in train:
                    t[key] = train[term]
            t["total"] = train["total_loss"]
            t["total_final"] = total_from_components(train, w, kl_weight(final_step, w.beta, w.kl_warmup_steps, w.kl_start_steps))
        else:
            t["loss"] = train["loss"]
        out["train_log"] = t
    if len(val):
        v = pd.DataFrame({"step": val["step"]})
        if kind == "vae":
            for key, term in TERM_OF.items():
                if f"val_{term}" in val:
                    v[key] = val[f"val_{term}"]
            if "val_loss" in val:
                v["total"] = val["val_loss"]
        else:
            if "val_loss" in val:
                v["loss"] = val["val_loss"]
            if "val_loss_null_cond" in val:
                v["loss_null_cond"] = val["val_loss_null_cond"]
        out["val_inloop"] = v
    return out


def gap_trend(steps: np.ndarray, gap: np.ndarray, gap_se: np.ndarray) -> Dict[str, float]:
    """Weighted least-squares slope of the val - train gap over the second half of the checkpoints (per 1000 steps) and
    its standard error; ``growing`` only when the slope exceeds two of its standard errors."""
    ok = np.isfinite(gap) & np.isfinite(gap_se) & (gap_se > 0)
    steps, gap, gap_se = steps[ok], gap[ok], gap_se[ok]
    half = len(steps) // 2
    steps, gap, gap_se = steps[half:], gap[half:], gap_se[half:]
    if len(steps) < 3:
        return {"slope_per_1000": float("nan"), "slope_se": float("nan"), "growing": False, "n_points": int(len(steps))}
    w = 1.0 / gap_se**2
    x = steps / 1000.0
    xm = (w * x).sum() / w.sum()
    sxx = (w * (x - xm) ** 2).sum()
    slope = float((w * (x - xm) * (gap - (w * gap).sum() / w.sum())).sum() / sxx)
    se = float(math.sqrt(1.0 / sxx))
    return {"slope_per_1000": slope, "slope_se": se, "growing": bool(slope > 2 * se), "n_points": int(len(steps))}


def summary_of(kind: str, curves: pd.DataFrame) -> Dict[str, Any]:
    """Best validation checkpoint and the overfitting indicator per weight kind."""
    main = "total_final" if kind == "vae" else "loss"
    out: Dict[str, Any] = {"main_metric": main}
    for which, frame in curves.groupby("weights"):
        frame = frame.sort_values("step")
        best = frame.loc[frame[f"val_{main}"].idxmin()]
        info = {"best_step": int(best["step"]), "best_val": float(best[f"val_{main}"]), "best_val_se": float(best[f"val_{main}_se"]),
                "last_step": int(frame["step"].iloc[-1])}
        if f"gap_{main}" in frame:
            info["gap_last"] = float(frame[f"gap_{main}"].iloc[-1])
            info["gap_last_se"] = float(frame[f"gap_{main}_se"].iloc[-1])
            info["gap_trend"] = gap_trend(frame["step"].to_numpy(), frame[f"gap_{main}"].to_numpy(), frame[f"gap_{main}_se"].to_numpy())
        out[which] = info
    return out


# ----------------------------------------------------------------------------------------------------------------------
# orchestration
# ----------------------------------------------------------------------------------------------------------------------


def run_curves(run_root: Path, *, weights_kinds: Sequence[str] = ("ema",), n_spines: int = 1000, batch_size: Optional[int] = None, n_draws: int = 3,
               seed: int = 12345, device: Any = None, num_workers: int = 0, final_step: Optional[int] = None, steps: Optional[Sequence[int]] = None,
               reuse: bool = True, manifests: Optional[Sequence[Path]] = None, splits: Optional[Path] = None, path_mode: Optional[str] = None,
               out_dir: Optional[Path] = None, log=print) -> Dict[str, Any]:
    """Evaluate all periodic checkpoints of ``run_root``; writes ``training_curves/curves.{csv,json}`` and returns the result."""
    import time

    import torch

    from ..experiment.run import open_run
    from ..experiment.training import get_device

    run = open_run(Path(run_root))
    cfg = run.load_config()
    kind = detect_kind(cfg)
    device = torch.device(device) if device is not None else get_device(cfg.get("device", "auto"))
    out_dir = Path(out_dir) if out_dir else run.root / "training_curves"
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoints = list_checkpoints(run.checkpoints)
    if steps:
        checkpoints = [(s, p) for s, p in checkpoints if s in set(int(x) for x in steps)]
    if not checkpoints:
        raise SystemExit(f"no step_<N>.pt checkpoints in {run.checkpoints} (train with train.keep_every=<N>)")
    final_step = int(final_step or cfg["train"].get("max_steps") or checkpoints[-1][0])
    batch_size = int(batch_size or cfg["data"].get("batch_size", 8))
    log(f"[curves] {run.experiment_id}: kind {kind}, {len(checkpoints)} checkpoints (steps {checkpoints[0][0]}..{checkpoints[-1][0]}), final step {final_step}, device {device}")

    t0 = time.perf_counter()
    inputs = build_inputs(kind, cfg, run.root, n_spines=n_spines, batch_size=batch_size, seed=seed, num_workers=num_workers, manifests=manifests, splits=splits, path_mode=path_mode)
    log(f"[curves] fixed inputs: {len(inputs.val_keys)} val + {len(inputs.train_keys)} train spines, {len(inputs.val_batches)}+{len(inputs.train_batches)} batches of {batch_size} ({time.perf_counter() - t0:.1f} s)")

    settings = {"kind": kind, "n_spines": len(inputs.val_keys), "batch_size": batch_size, "n_draws": n_draws, "seed": seed, "final_step": final_step, "fingerprint": inputs.fingerprint}
    json_path = out_dir / "curves.json"
    done: Dict[Tuple[int, str], Dict[str, Any]] = {}
    if reuse and json_path.exists():
        previous = json.loads(json_path.read_text(encoding="utf-8"))
        if previous.get("settings") == settings:
            done = {(int(r["step"]), r["weights"]): r for r in previous["rows"]}
            log(f"[curves] reusing {len(done)} rows of the previous evaluation (same settings)")

    model = build_model(kind, run.root)
    if not cfg["train"].get("ema_decay") and "ema" in weights_kinds:
        log("[curves] the run was trained without EMA (train.ema_decay is not set): 'ema' -> 'raw' weights")
        weights_kinds = tuple(dict.fromkeys("raw" if w == "ema" else w for w in weights_kinds))
    rows: List[Dict[str, Any]] = []
    for step, path in checkpoints:
        payload = None
        for which in weights_kinds:
            if (step, which) in done:
                rows.append(done[(step, which)])
                continue
            if payload is None:
                payload = torch.load(path, map_location="cpu", weights_only=False)
            if not load_weights(model, payload, which):
                log(f"[curves] step {step}: no EMA weights in the checkpoint, skipped ({which})")
                continue
            model.to(device)
            t1 = time.perf_counter()
            row = evaluate_checkpoint(kind, model, inputs, cfg, step=step, final_step=final_step, device=device, n_draws=n_draws, seed=seed)
            row["weights"] = which
            row["checkpoint"] = path.name
            rows.append(row)
            main = "total_final" if kind == "vae" else "loss"
            log(f"[curves] step {step:>7} {which:<3} val {row[f'val_{main}']:.5g} +- {row[f'val_{main}_se']:.2g}  train {row[f'train_{main}']:.5g} +- {row[f'train_{main}_se']:.2g}  ({time.perf_counter() - t1:.1f} s)")
        del payload
        _write_partial(out_dir, settings, rows)

    curves = pd.DataFrame(rows)
    for col in curves.columns:  # rows reused from curves.json carry None for NaN
        if col not in ("weights", "checkpoint"):
            curves[col] = pd.to_numeric(curves[col], errors="coerce")
    logs = read_logs(run.root, kind, cfg, final_step)
    result = {"settings": settings, "rows": rows, "summary": summary_of(kind, curves), "run": run.experiment_id}
    curves.to_csv(out_dir / "curves.csv", index=False)
    for name, frame in logs.items():
        frame.to_csv(out_dir / f"{name}.csv", index=False)
    json_path.write_text(json.dumps(_clean(result), indent=2), encoding="utf-8")
    return {"result": result, "curves": curves, "logs": logs, "kind": kind, "cfg": cfg, "out_dir": out_dir}


def _write_partial(out_dir: Path, settings: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]) -> None:
    (Path(out_dir) / "curves.json").write_text(json.dumps(_clean({"settings": dict(settings), "rows": list(rows), "summary": {}}), indent=2), encoding="utf-8")


def _clean(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        v = float(obj)
        return v if math.isfinite(v) else None
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj
