"""Generation procedure with rejection (t-module-technical-spec.md, section 8).

Any model with ``sample(conditions, rng) -> [{"kappa1", "kappa2", "log_radius", "step", "length", "group"}]`` can be
plugged in. Each sample is turned into a geometry (frame integration, canonical zero angle) and passes the same
checks; a rejected request is generated again (new random draw) up to ``max_attempts`` times. The acceptance rate
a_T = accepted / attempts is a reported property of the model: the distribution of ACCEPTED samples differs from
what the model produces, so both are evaluated.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .constraints import TrunkLimits, check_trunk
from .representation import geometry_from_features


def build_record(raw: Dict[str, Any], limits: TrunkLimits) -> Dict[str, Any]:
    geom = geometry_from_features(raw["kappa1"], raw["kappa2"], raw["log_radius"], raw["step"])
    check = check_trunk(geom["gamma"], geom["radius"], geom["kappa1"], geom["kappa2"], raw["step"], limits)
    return {
        "s": geom["s"], "gamma": geom["gamma"], "radius": geom["radius"],
        "tangent": geom["tangent"], "normal": geom["e1"], "binormal": geom["e2"],
        "kappa1": geom["kappa1"], "kappa2": geom["kappa2"],
        "length": geom["length"], "n_nodes": len(geom["s"]), "step": float(raw["step"]),
        "frame_reference": geom["frame_reference"], "group": raw.get("group"), "checks": check,
    }


def generate_trunks(
    model: Any,
    conditions: Sequence[Dict[str, Any]],
    limits: TrunkLimits,
    *,
    seed: int = 0,
    max_attempts: int = 20,
    model_id: str = "t0",
) -> Tuple[List[Optional[Dict[str, Any]]], Dict[str, Any]]:
    """Returns (records aligned with ``conditions``; ``None`` where all attempts were rejected, summary)."""
    rng = np.random.default_rng(seed)
    records: List[Optional[Dict[str, Any]]] = [None] * len(conditions)
    pending = list(range(len(conditions)))
    attempts = np.zeros(len(conditions), dtype=int)
    reasons: Counter = Counter()
    for _ in range(max_attempts):
        if not pending:
            break
        # a request without a fixed length keeps drawing a NEW length on every attempt
        raws = model.sample([conditions[i] for i in pending], rng)
        still = []
        for i, raw in zip(pending, raws):
            attempts[i] += 1
            record = build_record(raw, limits)
            if record["checks"]["passed"]:
                record.update(seed=seed, model_id=model_id, conditions=dict(conditions[i]), attempts=int(attempts[i]))
                records[i] = record
            else:
                reasons.update(record["checks"]["reasons"])
                still.append(i)
        pending = still
    accepted = sum(r is not None for r in records)
    summary = {
        "n_requests": len(conditions), "n_accepted": accepted, "n_failed": len(conditions) - accepted,
        "n_attempts": int(attempts.sum()), "acceptance_rate": float(accepted / max(int(attempts.sum()), 1)),
        "rejection_reasons": dict(reasons), "model_id": model_id, "seed": seed, "limits": limits.to_dict(),
    }
    return records, summary
