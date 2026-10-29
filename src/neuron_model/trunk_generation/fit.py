"""Fitting and validation of T0 (t-module-technical-spec.md, sections 4, 7, 10).

``fit_t0`` fits T0 with each mean-profile variant on the train branches, calibrates the geometric limits on the
train branches (real branches must pass), generates one branch per validation request (same conditions and
length as the validation branch) and compares branch functionals of generated and validation branches
(W1, KS); the noise floor is the same comparison between two halves of the validation set (split by neuron).
Units: nm.
"""

from __future__ import annotations

import functools

from typing import Any, Dict, List, Sequence

import numpy as np

from ..significance import cluster_bootstrap_diff, ks_pvalue
from .conditions import DesignEncoder
from .constraints import TrunkLimits, check_trunk
from .generate import generate_trunks
from .models.t0_statistical import T0Branch, T0Model
from .representation import geometry_from_features

FUNCTIONALS = ("mean_radius_nm", "radius_ratio_prox_dist", "kappa_rms_per_nm", "tortuosity", "radius_cv", "kappa_autocorr_1um")


def functionals(kappa1: np.ndarray, kappa2: np.ndarray, log_radius: np.ndarray, step: float, gamma: np.ndarray = None) -> Dict[str, float]:
    r = np.exp(np.asarray(log_radius, dtype=float))
    m = len(r)
    # kappa carries M-1 steps; trunk records may append a NaN for the last node
    kappa1, kappa2 = np.asarray(kappa1, dtype=float)[: m - 1], np.asarray(kappa2, dtype=float)[: m - 1]
    k = np.hypot(kappa1, kappa2)
    q = max(1, m // 5)
    if gamma is None:
        gamma = geometry_from_features(kappa1, kappa2, log_radius, step)["gamma"]
    chord = float(np.linalg.norm(gamma[-1] - gamma[0]))
    lag = max(1, int(round(1000.0 / step)))
    kc = np.asarray(kappa1, dtype=float) - np.mean(kappa1)
    ac = float(np.corrcoef(kc[:-lag], kc[lag:])[0, 1]) if len(kc) > lag + 2 and kc.std() > 0 else float("nan")
    return {"mean_radius_nm": float(r.mean()), "radius_ratio_prox_dist": float(r[:q].mean() / r[-q:].mean()),
            "kappa_rms_per_nm": float(np.sqrt(np.mean(k ** 2))), "tortuosity": float((m - 1) * step / max(chord, 1e-9)),
            "radius_cv": float(r.std() / r.mean()), "kappa_autocorr_1um": ac}


def compare(a: Sequence[float], b: Sequence[float]) -> Dict[str, float]:
    from scipy import stats

    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        return {"w1": float("nan"), "w1_rel": float("nan"), "ks": float("nan"), "median_a": float("nan"), "median_b": float("nan")}
    w1 = float(stats.wasserstein_distance(a, b))
    return {"w1": w1, "w1_rel": w1 / max(float(np.std(b)), 1e-12), "ks": float(stats.ks_2samp(a, b).statistic),
            "median_a": float(np.median(a)), "median_b": float(np.median(b))}


LENGTH_CLASSES = (("< 20 um", 0.0, 20000.0), ("20-50 um", 20000.0, 50000.0), (">= 50 um", 50000.0, 1e12))
RADIUS_CLASSES = (("r < 300 nm", 0.0, 300.0), ("300-500 nm", 300.0, 500.0), ("r >= 500 nm", 500.0, 1e12))


def _raw_radius_cv_by_radius(model: T0Model, requests: Sequence[Dict[str, Any]], val: Sequence[T0Branch], seed: int) -> Dict[str, Dict[str, float]]:
    """Real vs generated radius CV along a branch per class of the real median radius (first draw, no rejection)."""
    rng = np.random.default_rng(seed + 11)
    real_cv, gen_cv, radius = [], [], []
    for rec, b in zip(model.sample(list(requests), rng), val):
        r_real, r_gen = np.exp(b.log_radius), np.exp(rec["log_radius"])
        real_cv.append(float(r_real.std() / r_real.mean()))
        gen_cv.append(float(r_gen.std() / r_gen.mean()))
        radius.append(float(np.median(r_real)))
    real_cv, gen_cv, radius = np.asarray(real_cv), np.asarray(gen_cv), np.asarray(radius)
    out = {}
    for name, lo, hi in RADIUS_CLASSES:
        m = (radius >= lo) & (radius < hi)
        if m.sum() >= 3:
            out[name] = {"n": int(m.sum()), "real_median": float(np.median(real_cv[m])), "real_p90": float(np.percentile(real_cv[m], 90)),
                         "gen_median": float(np.median(gen_cv[m])), "gen_p90": float(np.percentile(gen_cv[m], 90))}
    return out


def _raw_tortuosity_by_length(model: T0Model, requests: Sequence[Dict[str, Any]], val: Sequence[T0Branch], seed: int) -> Dict[str, Dict[str, float]]:
    """Real vs generated tortuosity per length class; generated = one draw per request WITHOUT the rejection step
    (a structural excess of winding grows with the length when the heading diffuses freely)."""
    rng = np.random.default_rng(seed + 7)
    real_t, gen_t, lengths = [], [], []
    for rec, b in zip(model.sample(list(requests), rng), val):
        real_t.append(_real_functionals(b)["tortuosity"])
        gen_t.append(functionals(rec["kappa1"], rec["kappa2"], rec["log_radius"], rec["step"])["tortuosity"])
        lengths.append(b.length)
    real_t, gen_t, lengths = np.asarray(real_t), np.asarray(gen_t), np.asarray(lengths)
    out = {}
    for name, lo, hi in LENGTH_CLASSES:
        m = (lengths >= lo) & (lengths < hi)
        if m.sum() >= 3:
            out[name] = {"n": int(m.sum()), "real_median": float(np.median(real_t[m])), "real_p90": float(np.percentile(real_t[m], 90)),
                         "gen_median": float(np.median(gen_t[m])), "gen_p90": float(np.percentile(gen_t[m], 90))}
    return out


def _real_functionals(b: T0Branch) -> Dict[str, float]:
    return functionals(b.kappa1, b.kappa2, b.log_radius, b.step)


def fit_t0(
    train: Sequence[T0Branch],
    val: Sequence[T0Branch],
    val_conditions: Sequence[Dict[str, Any]],
    val_neurons: Sequence[str],
    *,
    encoder: DesignEncoder,
    mean_models: Sequence[str] = ("parabola", "gam"),
    shrinkage_std: float = 0.3,
    max_iter: int = 200,
    length_bounds=(5000.0, 255500.0),
    direction: str = "multiscale",
    radius_residual: str = "multiscale",
    train_conditions: Sequence[Dict[str, Any]] = None,
    seed: int = 0,
    log=functools.partial(print, flush=True),
) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)
    log(f"[t0] {len(train)} train / {len(val)} validation branches: calibrating the validity checks ...")
    limits = TrunkLimits.from_data([np.hypot(b.kappa1, b.kappa2) for b in train], [_real_functionals(b)["tortuosity"] for b in train])
    # the real branches must pass the checks (section 7): report the rejection rate on the validation set
    real_reject = []
    for b in val:
        g = geometry_from_features(b.kappa1, b.kappa2, b.log_radius, b.step)
        real_reject.append(not check_trunk(g["gamma"], g["radius"], g["kappa1"], g["kappa2"], b.step, limits)["passed"])
    real_f = [_real_functionals(b) for b in val]
    # noise floor: two halves of the validation set split by neuron (one validation neuron: by branch, optimistic)
    neurons = np.array(sorted(set(val_neurons)))
    if len(neurons) >= 2:
        half = set(rng.permutation(neurons)[: len(neurons) // 2])
        in_a = np.array([n in half for n in val_neurons])
        floor_by = "neuron"
    else:
        in_a = rng.permutation(len(val_neurons)) < len(val_neurons) // 2
        floor_by = "branch"
    floor = {f: compare([x[f] for x, m in zip(real_f, in_a) if m], [x[f] for x, m in zip(real_f, in_a) if not m]) for f in FUNCTIONALS}
    results: Dict[str, Any] = {}
    models: Dict[str, T0Model] = {}
    requests = [dict(c, length_nm=b.length, step_nm=b.step) for c, b in zip(val_conditions, val)]
    for mean in mean_models:
        log(f"[t0] variant {mean}: fitting ...")
        model = T0Model(mean, length_bounds=length_bounds, direction=direction, radius_residual=radius_residual).fit(train, encoder=encoder, shrinkage_std=shrinkage_std, max_iter=max_iter, progress=log)
        log(f"[t0] variant {mean}: generating and checking the validation set ...")
        records, summary = generate_trunks(model, requests, limits, seed=seed, model_id=f"t0-{mean}")
        gen_f = [functionals(r["kappa1"], r["kappa2"], np.log(r["radius"]), r["step"], r["gamma"]) for r in records if r is not None]
        metrics = {f: compare([x[f] for x in gen_f], [x[f] for x in real_f]) for f in FUNCTIONALS}
        # is the difference of the medians larger than chance? neurons are the unit of resampling (significance.py)
        gen_neurons = [n for r, n in zip(records, val_neurons) if r is not None]
        for f in FUNCTIONALS:
            metrics[f]["ci"] = cluster_bootstrap_diff([x[f] for x in gen_f], gen_neurons, [x[f] for x in real_f], list(val_neurons), n_boot=500, seed=seed)
            metrics[f]["ks_p_naive"] = ks_pvalue([x[f] for x in gen_f], [x[f] for x in real_f])
        # tortuosity of the FIRST draw of every request (before rejection), real vs generated by length class
        raw = _raw_tortuosity_by_length(model, requests, val, seed)
        raw_cv = _raw_radius_cv_by_radius(model, requests, val, seed)
        # the same on (at most 500) TRAIN branches: matching there but not on validation = the validation neurons differ
        if train_conditions is not None:
            idx = np.random.default_rng(seed + 13).permutation(len(train))[:500]
            tr_req = [dict(train_conditions[i], length_nm=train[i].length, step_nm=train[i].step) for i in idx]
            raw_cv_train = _raw_radius_cv_by_radius(model, tr_req, [train[i] for i in idx], seed)
        else:
            raw_cv_train = {}
        results[mean] = {"generation": summary, "functionals": metrics, "tortuosity_by_length": raw, "radius_cv_by_radius": raw_cv, "radius_cv_by_radius_train": raw_cv_train,
                         "mean_w1_rel": float(np.nanmean([m["w1_rel"] for m in metrics.values()]))}
        models[mean] = model
        log(f"[t0] {mean:9s} acceptance {summary['acceptance_rate']:.3f}, rejected {summary.get('rejection_reasons', {})}, "
            f"mean W1/std over functionals {results[mean]['mean_w1_rel']:.3f}")
        for f, m in metrics.items():
            ci = m["ci"]
            ci_text = (f"  diff {ci['diff']:+.3g} [95 % CI {ci['lo']:+.3g}, {ci['hi']:+.3g}]{' *' if ci['significant'] else ''}; KS p (naive) {m['ks_p_naive']:.2g}" if np.isfinite(ci["diff"]) else "")
            log(f"       {f:24s} median generated {m['median_a']:.4g}  real {m['median_b']:.4g}  W1/std {m['w1_rel']:.2f}  (noise floor {floor[f]['w1_rel']:.2f}){ci_text}")
        for name, v in raw.items():
            log(f"       tortuosity, length {name:12s} n {v['n']:3d}: real median {v['real_median']:.3f} p90 {v['real_p90']:.3f} | "
                f"generated (no rejection) median {v['gen_median']:.3f} p90 {v['gen_p90']:.3f}")
        for part, table in (("val", raw_cv), ("train", raw_cv_train)):
            for name, v in table.items():
                log(f"       radius CV {part:5s}, {name:12s} n {v['n']:3d}: real median {v['real_median']:.3f} p90 {v['real_p90']:.3f} | "
                    f"generated median {v['gen_median']:.3f} p90 {v['gen_p90']:.3f}")
    chosen = min(results, key=lambda k: results[k]["mean_w1_rel"])
    report = {
        "n_train": len(train), "n_val": len(val), "limits": limits.to_dict(),
        "real_val_rejection_rate": float(np.mean(real_reject)) if real_reject else None,
        "noise_floor": floor, "noise_floor_by": floor_by, "noise_floor_mean_w1_rel": float(np.nanmean([m["w1_rel"] for m in floor.values()])),
        "variants": results, "chosen": chosen, "direction": models[chosen].direction.to_dict() if models[chosen].direction is not None else None,
        "radius_residual": models[chosen].radius.to_dict() if models[chosen].radius is not None else None,
        "pooled_ou": {"length_scales_nm": models[chosen].params_for("__pooled__").length_scales.tolist()},
    }
    return {"models": models, "chosen": chosen, "limits": limits, "report": report}
