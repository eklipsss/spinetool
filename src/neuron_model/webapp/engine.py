"""Generation of one branch for the web application: T0 -> spine count -> positions -> spines -> assembled mesh.

The flow mirrors ``scripts/neuron_model/generate_tp.py`` (T + P) and ``scripts/neuron_model/assemble_branches.py`` (S +
assembly) for a single request; nothing is re-implemented, the engine only wires the existing modules together:

1. ``generate_trunks`` (T0 with the geometric checks of ``limits.json``; a trunk rejected ``max_attempts`` times is an error);
2. ``covariates_from_trunk`` + ``CountModel.sample`` (one generated branch = its own neuron: the neuron effect is drawn);
3. ``generate_placements`` (positions by the inverse CDF, uniform angles, d_min);
4. a ``SpineSource`` gives the pool of spine meshes for the request: ``LibrarySpineSource`` takes the most specific key
   of the bundle's spine library (``<species>/<cell type>/<detailed type>`` -> ... -> ``any``) and reports the key used;
5. ``assemble_branch`` (base size matched to the trunk radius when the bundle has a ``ContextSelector``).

Reproducibility: the four stages draw from independent streams of ``SeedSequence(seed)``, so the same request + seed +
bundle gives the same branch, bit for bit (the T0 / P0 sampling and the assembly are deterministic given their rng).
Units: nm.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Protocol, Tuple

import numpy as np

from ..branch_assembly import AssemblyResult, SpinePool, assemble_branch, densify, rotation, trunk_mesh
from ..branch_assembly.selection import SpineChooser
from ..placement_generation.generate import covariates_from_trunk, generate_placements
from ..placement_generation.models.count import build_count_data
from ..trunk_generation.conditions import group_key
from ..trunk_generation.generate import generate_trunks
from . import metrics as M
from .bundle import ModelBundle, library_key_chain

REQUEST_FIELDS = ("species", "cell_type_binary", "compartment", "spine_cell_type", "length_nm", "soma_distance_start_nm", "is_terminal", "seed")
FIELD_NAMES_RU = {"species": "вид", "cell_type_binary": "тип клетки", "compartment": "компартмент", "spine_cell_type": "подробный тип клетки шипиков",
                  "length_nm": "длина", "soma_distance_start_nm": "расстояние до сомы", "is_terminal": "концевая ветвь", "seed": "зерно генератора"}


class RequestError(ValueError):
    """Invalid request: ``errors`` = [{"field", "message"}] (messages in Russian, for the page)."""

    def __init__(self, errors: List[Dict[str, str]]) -> None:
        super().__init__("; ".join(f"{e['field']}: {e['message']}" for e in errors))
        self.errors = errors


class GenerationError(RuntimeError):
    """A valid request the models could not turn into a branch (e.g. every trunk draw failed the geometric checks)."""


# ---------------------------------------------------------------------------
# spine sources
# ---------------------------------------------------------------------------


@dataclass
class SpineSelection:
    """What a spine source gives the assembly for one request."""

    pool: SpinePool
    chooser: SpineChooser
    source: str                       # "library" | "live"
    key: str                          # library key actually used
    requested_keys: List[str]         # candidate keys, most specific first
    metrics: Dict[str, Dict[str, Optional[float]]]  # pool key -> precomputed morphometrics of that spine

    @property
    def fallback(self) -> bool:
        return self.key != self.requested_keys[0]


class SpineSource(Protocol):
    def select(self, request: Mapping[str, Any]) -> SpineSelection: ...


class LibrarySpineSource:
    """Spines from the pre-generated library of the bundle (CPU, fast): pool + chooser are built once per key."""

    def __init__(self, bundle: ModelBundle) -> None:
        self.bundle = bundle
        self._cache: Dict[str, Tuple[SpinePool, SpineChooser, Dict[str, Dict[str, Optional[float]]]]] = {}

    def _prepared(self, key: str):
        if key not in self._cache:  # a race only builds the same objects twice: harmless
            lib = self.bundle.library(key)
            pool = lib.pool(self.bundle.assembly.base_band_nm)
            selector = self.bundle.context_selector
            if selector is not None and not np.isfinite([sp.base_radius_nm for sp in pool.spines]).any():
                selector = None  # no base area in this library: uniform choice (SpineChooser would refuse)
            self._cache[key] = (pool, SpineChooser(pool, selector), {sp.key: lib.spine_metrics(i) for i, sp in enumerate(pool.spines)})
        return self._cache[key]

    def select(self, request: Mapping[str, Any]) -> SpineSelection:
        chain = library_key_chain(request.get("species"), request.get("cell_type_binary"), request.get("spine_cell_type"))
        available = set(self.bundle.library_keys)
        key = next((k for k in chain if k in available), None)
        if key is None:
            raise GenerationError(f"spine library has none of the keys {chain} (bundle keys {sorted(available)})")
        pool, chooser, metrics = self._prepared(key)
        return SpineSelection(pool, chooser, "library", key, chain, metrics)


class LiveSpineSource:
    """Extension point (not implemented): spines generated on request by a conditional generator on a GPU.

    The intended implementation keeps a loaded conditional MoGen (``spine_generation/models/spine_mogen``, condition from
    ``spine_generation/conditioning.py``: species, cell type, compartment), samples ``n`` point clouds for the request's
    condition with a seed derived from the request seed, reconstructs meshes (normals -> Screened Poisson, the same
    calibrated backend as ``generate_mogen.py``), and returns them as a ``SpinePool`` (``SpinePool.from_meshes``) with
    morphometrics from ``spine_morphometrics.compute_scalar_metrics``. Generation of a few hundred spines takes minutes on
    a GPU, so a public deployment would queue such requests instead of answering them synchronously.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError("live spine generation is not implemented: use LibrarySpineSource (pre-generated spines)")

    def select(self, request: Mapping[str, Any]) -> SpineSelection:  # pragma: no cover - stub
        raise NotImplementedError


# ---------------------------------------------------------------------------
# result
# ---------------------------------------------------------------------------


@dataclass
class GenerationResult:
    result_id: str
    bundle_id: str
    request: Dict[str, Any]
    trunk: Dict[str, Any]
    placement: Dict[str, Any]
    assembly: AssemblyResult
    trunk_mesh: Tuple[np.ndarray, np.ndarray]
    spines_local: Dict[int, Tuple[np.ndarray, np.ndarray]]   # placement index -> library mesh (local spine frame)
    spines_global: Dict[int, Tuple[np.ndarray, np.ndarray]]  # placement index -> the same mesh placed on the branch (before the union)
    library: Dict[str, Any]
    metrics: Dict[str, Any]
    warnings: List[str] = field(default_factory=list)
    timings: Dict[str, float] = field(default_factory=dict)

    @property
    def branch_mesh(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.assembly.vertices, self.assembly.faces

    def summary(self) -> Dict[str, Any]:
        """JSON-ready description for the page (no meshes): request, library used, metric tables, spine rows."""
        m = self.metrics
        return {
            "id": self.result_id, "bundle_id": self.bundle_id, "request": self.request, "library": self.library,
            "status": self.assembly.branch.get("status"), "status_reason": self.assembly.branch.get("reason"), "warnings": self.warnings,
            "trunk": M.present(m["trunk"], M.TRUNK_INFO), "placement": M.present(m["placement"], M.PLACEMENT_INFO),
            "failure_reasons": [{"reason": k, "name": M.FAILURE_REASONS_RU.get(k, k), "count": v} for k, v in m["placement"].get("failure_reasons", {}).items()],
            "branch_mesh": M.present(m["branch_mesh"], M.BRANCH_MESH_INFO),
            "spine_columns": [{"key": k, **v} for k, v in M.SPINE_COLUMNS.items()], "spines": m["spines"],
            "reference": m.get("reference", {"available": False}), "timings_s": self.timings,
        }


def canonical_request_id(bundle_id: str, request: Mapping[str, Any]) -> str:
    """Stable id of (bundle, normalised request): the same parameters always map to the same result."""
    text = json.dumps({"bundle": bundle_id, **{k: request.get(k) for k in REQUEST_FIELDS}}, sort_keys=True)
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def placed_vertices(dense_trunk, spine, s: float, theta: float, delta_nm: float) -> np.ndarray:
    """Vertices of a pool spine as placed by ``assemble_branch`` (``place``: p = R^T (p_loc - c) + a - delta n)."""
    f = dense_trunk.surface_frame(np.array([s]), np.array([theta]))
    a, n, t = f["anchor"][0], f["normal"][0], f["tangent"][0]
    return (spine.vertices - spine.center) @ rotation(t, n) + a - float(delta_nm) * n


# ---------------------------------------------------------------------------
# generator
# ---------------------------------------------------------------------------


class BranchGenerator:
    def __init__(self, bundle: ModelBundle, spine_source: Optional[SpineSource] = None) -> None:
        self.bundle = bundle
        self.spine_source = spine_source or LibrarySpineSource(bundle)

    # ---- request ------------------------------------------------------------------------------------------------------
    def normalize(self, request: Mapping[str, Any]) -> Dict[str, Any]:
        """Defaults filled in, types coerced, ranges checked; raises ``RequestError`` with every problem at once."""
        man = self.bundle.manifest
        cats, lim = man["categories"], man["limits"]
        q = dict(man["defaults"])
        q.update({k: v for k, v in dict(request).items() if k in REQUEST_FIELDS})
        errors: List[Dict[str, str]] = []
        unknown = sorted(set(request) - set(REQUEST_FIELDS))
        if unknown:
            errors.append({"field": ",".join(unknown), "message": f"неизвестные параметры (допустимы: {', '.join(REQUEST_FIELDS)})"})
        for name in ("species", "cell_type_binary", "compartment"):
            if q.get(name) not in cats[name]:
                errors.append({"field": name, "message": f"значение {q.get(name)!r} недопустимо; допустимые: {', '.join(cats[name])}"})
        if q.get("spine_cell_type") in ("", "auto"):
            q["spine_cell_type"] = None
        if q.get("spine_cell_type") is not None and q["spine_cell_type"] not in cats.get("spine_cell_type", []):
            errors.append({"field": "spine_cell_type", "message": f"значение {q['spine_cell_type']!r} недопустимо; допустимые: {', '.join(cats.get('spine_cell_type', [])) or 'нет'}"})

        def number(name: str, lo: float, hi: float, optional: bool = False) -> None:
            v = q.get(name)
            if v is None and optional:
                return
            try:
                v = float(v)
            except (TypeError, ValueError):
                errors.append({"field": name, "message": f"нужно число, получено {v!r}"})
                return
            if not math.isfinite(v) or not lo <= v <= hi:
                errors.append({"field": name, "message": f"{v:g} нм вне допустимого диапазона [{lo:g}, {hi:g}] нм"})
                return
            q[name] = v

        number("length_nm", float(lim["min_length_nm"]), float(lim["max_length_nm"]))
        number("soma_distance_start_nm", 0.0, float(lim["max_soma_distance_nm"]), optional=True)
        if not isinstance(q.get("is_terminal"), bool):
            if q.get("is_terminal") in (0, 1, "true", "false"):
                q["is_terminal"] = q["is_terminal"] in (1, "true")
            else:
                errors.append({"field": "is_terminal", "message": f"нужно true / false, получено {q.get('is_terminal')!r}"})
        seed = q.get("seed")
        if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) and not (isinstance(seed, str) and seed.isdigit()):
            errors.append({"field": "seed", "message": f"нужно целое число, получено {seed!r}"})
        elif not 0 <= int(seed) <= int(lim["seed_max"]):
            errors.append({"field": "seed", "message": f"должно быть в диапазоне [0, {lim['seed_max']}]"})
        else:
            q["seed"] = int(seed)
        if errors:
            raise RequestError(errors)
        return {k: q.get(k) for k in REQUEST_FIELDS}

    def request_warnings(self, q: Mapping[str, Any]) -> List[str]:
        man = self.bundle.manifest
        out = []
        groups = man.get("trained_groups") or []
        g = group_key(q["species"], q["cell_type_binary"], q["compartment"])
        if groups and g not in groups:
            out.append(f"сочетание условий {g} не было отдельной группой при обучении модели ствола: параметры взяты общие (объединённые по всем группам)")
        ct = q.get("spine_cell_type")
        binary_of = man.get("spine_cell_type_binary", {})
        if ct and binary_of.get(ct) not in (None, q["cell_type_binary"]):
            out.append(f"подробный тип {ct} относится к типу «{binary_of[ct]}», а ствол генерируется для «{q['cell_type_binary']}»")
        return out

    # ---- generation ---------------------------------------------------------------------------------------------------
    def generate(self, request: Mapping[str, Any]) -> GenerationResult:
        q = self.normalize(request)
        warnings = self.request_warnings(q)
        b = self.bundle
        timings: Dict[str, float] = {}
        started = time.perf_counter()
        seed_trunk, seed_count, seed_place, seed_assembly = (int(x) for x in np.random.SeedSequence(q["seed"]).generate_state(4))

        t_req = {"species": q["species"], "cell_type_binary": q["cell_type_binary"], "compartment": q["compartment"], "length_nm": q["length_nm"],
                 "step_nm": b.manifest["defaults"].get("step_nm"), "soma_distance_start_nm": q["soma_distance_start_nm"], "is_terminal": q["is_terminal"]}
        trunks, t_summary = generate_trunks(b.t0_model, [t_req], b.trunk_limits, seed=seed_trunk, model_id=f"t0-{b.manifest['sources'].get('t0_model')}")
        trunk = trunks[0]
        if trunk is None:
            raise GenerationError(f"ствол не прошёл геометрические проверки ни в одной из {t_summary['n_attempts']} попыток "
                                  f"(причины: {t_summary['rejection_reasons']}); попробуйте другое зерно или длину")
        timings["trunk"] = time.perf_counter() - started

        mark = time.perf_counter()
        p0 = b.p0
        cov = [covariates_from_trunk(trunk, t_req, "gen0")]
        data = build_count_data(p0["design"], cov, p0["count_model"].exposure)
        n_requested = int(p0["count_model"].sample(data, np.random.default_rng(seed_count))[0])
        n_used = n_requested
        max_spines = int(b.manifest["limits"].get("max_spines", 0) or 0)
        if max_spines and n_requested > max_spines:
            n_used = max_spines
            warnings.append(f"модель P выбрала {n_requested} шипиков, поставлено не больше {max_spines} (ограничение сервера)")
        placement = generate_placements([trunk], [n_used], p0["position_model"], d_min_nm=p0["d_min_nm"],
                                        soma_start_nm=[q["soma_distance_start_nm"]], seed=seed_place)[0]
        if not placement["d_min_satisfied"]:
            warnings.append("положения шипиков не удовлетворили d_min за отведённое число попыток")
        timings["placement"] = time.perf_counter() - mark

        mark = time.perf_counter()
        sel = self.spine_source.select(q)
        if sel.fallback:
            warnings.append(f"библиотеки шипиков «{sel.requested_keys[0]}» нет, использована «{sel.key}»")
        cfg = b.assembly
        res = assemble_branch(trunk, placement["s"], placement["theta"], sel.pool, cfg, np.random.default_rng(seed_assembly), chooser=sel.chooser)
        if res.vertices is None:
            raise GenerationError(f"сборка ветви не удалась: {res.branch.get('reason')}")
        timings["assembly"] = time.perf_counter() - mark

        mark = time.perf_counter()
        dense = densify(trunk, cfg.trunk_step_nm)
        tv, tf = trunk_mesh(dense, cfg.trunk_n_theta)
        by_key = {sp.key: sp for sp in sel.pool.spines}
        local, placed = {}, {}
        for row in res.spines:
            if row.get("status") != "placed":
                continue
            sp = by_key[row["spine_key"]]
            local[int(row["spine_index"])] = (sp.vertices, sp.faces)
            placed[int(row["spine_index"])] = (placed_vertices(dense, sp, row["s"], row["theta"], row["delta_nm"]), sp.faces)
        rows = M.spine_rows(res.spines, placement, float(trunk["length"]), sel.metrics)
        trunk_values = M.trunk_metrics(trunk)
        placement_values = M.placement_metrics(trunk, placement, rows, n_requested, p0["d_min_nm"])
        metrics = {"trunk": trunk_values, "placement": placement_values,
                   "branch_mesh": {k: res.branch.get(k) for k in M.BRANCH_MESH_INFO}, "spines": rows}
        metrics["reference"] = M.reference_rows(b.reference.get("trunk_placement"), q, {**trunk_values, **placement_values})
        if res.branch.get("status") != "ok":
            warnings.append(f"сетка ветви с замечаниями: {res.branch.get('reason')}")
        timings["metrics"] = time.perf_counter() - mark
        timings["total"] = time.perf_counter() - started
        library = {"source": sel.source, "requested_keys": sel.requested_keys, "used_key": sel.key, "fallback": sel.fallback,
                   "pool_size": len(sel.pool), "selection": "context" if sel.chooser.selector is not None else "uniform"}
        return GenerationResult(canonical_request_id(b.bundle_id, q), b.bundle_id, q, trunk, placement, res, (tv, tf), local, placed,
                                library, metrics, warnings, {k: round(v, 3) for k, v in timings.items()})
