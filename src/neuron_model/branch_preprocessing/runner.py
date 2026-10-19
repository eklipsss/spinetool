"""Runner: one task = one neuron (all its merged branches), resumable, parallel over neurons.

A neuron is done when ``<out_root>/<neuron>/_done.json`` exists with the current config hash
(written after the neuron's tables). ``consolidate`` concatenates the per-neuron tables into
``<out_root>/branches.parquet`` and ``<out_root>/branch_spines.parquet``.
"""

from __future__ import annotations

import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from .config import BranchPreprocessingConfig
from .inputs import available_neurons, load_members, load_metadata, load_soma_midpoints, load_spine_table, neuron_metadata, neuron_split_lookup
from .process import process_neuron


def _is_done(cfg: BranchPreprocessingConfig, neuron_id: str, config_hash: str) -> bool:
    marker = cfg.out_root / neuron_id / "_done.json"
    if not marker.exists():
        return False
    try:
        info = json.loads(marker.read_text(encoding="utf-8"))
        return info.get("config_hash") == config_hash and not info.get("partial", False)
    except (OSError, ValueError):
        return False


def build_payloads(cfg: BranchPreprocessingConfig, neurons: List[str]) -> List[Dict[str, Any]]:
    members = load_members(cfg)
    members = members[members["neuron_id"].isin(set(neurons))]
    spines = load_spine_table(cfg, neurons)
    metadata = load_metadata(cfg)
    split = neuron_split_lookup(cfg.splits)
    midpoints = load_soma_midpoints(cfg)
    config_hash = cfg.config_hash()
    spine_groups = {n: g for n, g in spines.groupby("neuron_id")} if len(spines) else {}
    missing = sorted(set(map(str, neurons)) - set(map(str, spine_groups)))
    if missing:
        # without S verdicts every segment is 'unknown': counts and positions are undefined and the branch is not
        # eligible for P (T is not affected). Usually the S preprocessing has not reached these neurons yet.
        print(f"[branches] WARNING: {len(missing)} of {len(neurons)} neurons have no rows in the S manifest ({cfg.s_root}); "
              f"their spines will be 'unknown' (S preprocessing not finished?). First: {missing[:5]}", flush=True)
    status_known = spines["status"].isin(["valid", "invalid"]).mean() if len(spines) and "status" in spines.columns else 0.0
    print(f"[branches] S tables: {len(spines)} spine rows for {len(spine_groups)} neurons, false-spine verdict known for {status_known:.1%}", flush=True)
    keep_cols = ["limb_id", "branch_id", "spine_id", "status", "seal_status", "attachment_ambiguous", "attachment_method", "attachment_region_path", "origin_global", "attachment_loop_area", "attachment_cap_area"]
    payloads = []
    for neuron_id, group in members.groupby("neuron_id"):
        sp = spine_groups.get(neuron_id)
        spine_records = [] if sp is None else sp[[c for c in keep_cols if c in sp.columns]].to_dict("records")
        payloads.append(
            {
                "neuron_id": neuron_id,
                "config": cfg.to_dict(),
                "config_hash": config_hash,
                "members": group.to_dict("records"),
                "spines": spine_records,
                "metadata": neuron_metadata(cfg, metadata, neuron_id, sorted(group["limb_id"].unique())),
                "split": split.get(neuron_id),
                "soma_midpoints": {p: midpoints[p] for p in group["prefix"] if p in midpoints},
            }
        )
    return payloads


def run(cfg: BranchPreprocessingConfig, *, consolidate_after: bool = True) -> Dict[str, Any]:
    t0 = time.perf_counter()
    config_hash = cfg.config_hash()
    members = load_members(cfg)
    neurons = available_neurons(cfg, members)
    todo = neurons if cfg.force else [n for n in neurons if not _is_done(cfg, n, config_hash)]
    print(f"[branches] {cfg.dataset}: {len(neurons)} neurons, {len(neurons) - len(todo)} done, {len(todo)} to process -> {cfg.out_root} (config {config_hash})", flush=True)
    summaries: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []
    if todo:
        payloads = build_payloads(cfg, todo)
        print(f"[branches] inputs loaded in {time.perf_counter() - t0:.1f}s", flush=True)
        from tqdm.auto import tqdm

        bar = tqdm(total=len(payloads), desc="neurons", unit="neuron")
        if cfg.workers <= 1:
            for payload in payloads:
                summaries.append(_run_one(payload, failures))
                bar.update()
        else:
            with ProcessPoolExecutor(max_workers=int(cfg.workers)) as pool:
                futures = {pool.submit(process_neuron, p): p["neuron_id"] for p in payloads}
                for future in as_completed(futures):
                    try:
                        summaries.append(future.result())
                    except Exception as exc:  # a whole neuron failed (e.g. unreadable files)
                        failures.append({"neuron_id": futures[future], "error": repr(exc)})
                    bar.update()
        bar.close()
        summaries = [s for s in summaries if s]
    if failures:
        cfg.out_root.mkdir(parents=True, exist_ok=True)
        (cfg.out_root / "_failed_neurons.json").write_text(json.dumps(failures, indent=2), encoding="utf-8")
        print(f"[branches] {len(failures)} neurons failed, see {cfg.out_root / '_failed_neurons.json'}", flush=True)
    result = {"neurons": len(neurons), "processed": len(summaries), "failed_neurons": len(failures), "seconds": round(time.perf_counter() - t0, 1)}
    if consolidate_after:
        result.update(consolidate(cfg, neurons))
    print(f"[branches] done: {result}", flush=True)
    return result


def _run_one(payload: Dict[str, Any], failures: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    try:
        return process_neuron(payload)
    except Exception as exc:
        failures.append({"neuron_id": payload["neuron_id"], "error": repr(exc)})
        return None


def consolidate(cfg: BranchPreprocessingConfig, neurons: Optional[List[str]] = None) -> Dict[str, Any]:
    """Concatenate per-neuron tables of finished neurons into the dataset-level tables."""
    root = cfg.out_root
    if neurons is None:
        neurons = sorted(p.name for p in root.iterdir() if p.is_dir()) if root.is_dir() else []
    branch_frames, spine_frames = [], []
    for neuron_id in neurons:
        d = root / neuron_id
        if not (d / "_done.json").exists():
            continue
        branch_frames.append(pd.read_parquet(d / "branches.parquet"))
        spines_path = d / "branch_spines.parquet"
        if spines_path.exists():
            frame = pd.read_parquet(spines_path)
            if len(frame):
                spine_frames.append(frame)
    out: Dict[str, Any] = {"branches_rows": 0, "spines_rows": 0}
    if branch_frames:
        branches = pd.concat(branch_frames, ignore_index=True)
        _write(branches, root / "branches.parquet")
        out["branches_rows"] = int(len(branches))
        out["train_eligible_T"] = int(branches["train_eligible_T"].sum())
        out["train_eligible_P"] = int(branches["train_eligible_P"].sum())
        out["status_counts"] = branches["status"].value_counts().to_dict()
    if spine_frames:
        spines = pd.concat(spine_frames, ignore_index=True)
        _write(spines, root / "branch_spines.parquet")
        out["spines_rows"] = int(len(spines))
    return out


def _write(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    frame.to_parquet(tmp, index=False)
    os.replace(tmp, path)


def reapply_eligibility(cfg: BranchPreprocessingConfig, *, write: bool = True, neurons: Optional[List[str]] = None) -> Dict[str, Any]:
    """Recompute the QC flags and ``train_eligible_T/P`` of finished neurons from the STORED quantities with the
    thresholds of ``cfg`` (no mesh is read, minutes for the whole dataset). ``write=False``: only count.
    Returns counts and, per check, how many ok branches fail it."""
    from collections import Counter

    from .process import _finalize_eligibility

    root = cfg.out_root
    names = neurons if neurons is not None else sorted(p.name for p in root.iterdir() if p.is_dir()) if root.is_dir() else []
    n_t = n_p = n_ok = n_total = 0
    reasons: Counter = Counter()
    for neuron_id in names:
        d = root / neuron_id
        if not (d / "_done.json").exists():
            continue
        frame = pd.read_parquet(d / "branches.parquet")
        rows = []
        for record in frame.to_dict("records"):
            # a missing value is an ABSENT key, as in the rows the processing builds (``row.get(key, default)``)
            record = {k: v for k, v in record.items() if isinstance(v, (list, tuple, np.ndarray)) or not pd.isna(v)}
            _finalize_eligibility(cfg, record)
            rows.append(record)
        new = pd.DataFrame(rows).reindex(columns=list(frame.columns))
        n_total += len(new)
        n_ok += int((new["status"] == "ok").sum())
        n_t += int(new["train_eligible_T"].sum())
        n_p += int(new["train_eligible_P"].sum())
        for r in new.loc[new["status"] == "ok", "qc_reason"].fillna(""):
            reasons.update([x for x in str(r).split(";") if x])
        if write:
            _write(new, d / "branches.parquet")
    out = {"branches": n_total, "ok": n_ok, "train_eligible_T": n_t, "train_eligible_P": n_p, "failed_checks_among_ok": dict(reasons.most_common())}
    if write:
        out.update(consolidate(cfg, names))
    return out
