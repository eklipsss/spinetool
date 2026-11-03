"""Model bundle of the web application: one self-contained folder, nothing is read from ``runs/`` at serve time.

Layout of ``models/webapp/<bundle_id>/`` (units: nm)::

    manifest.json            bundle id, creation time, units, source runs, categories and allowed values, trained
                             condition groups, defaults, limits, assembly options, spine library summary
    t0_model.json            T0Model (``T0Model.load``) of the chosen mean variant
    limits.json              TrunkLimits of the T0 run (geometric checks of a generated trunk)
    p0_bundle.json           P0 (``load_p0_bundle``): count design + count model + position model + d_min
    context_selector.json    optional: ContextSelector (base size of the spine given the trunk radius, real spines)
    spine_library/<key>/spines.npz
                             packed spine meshes of one condition (local spine frame) + attachment region and
                             precomputed morphometrics of every spine
    reference/trunk_placement.json
                             optional: quantiles of real trunk / placement metrics per condition group

Spine library keys go from the most to the least specific: ``<species>/<cell_type_binary>/<cell_type>``,
``<species>/<cell_type_binary>``, ``<species>``, ``any``. A request takes the most specific key that exists (the engine
reports which one). ``spines.npz``: ``vertices`` float32 [sum V, 3], ``faces`` int32 [sum F, 3], ``v_offsets`` /
``f_offsets`` [n + 1], ``center`` [n, 3] (attachment centre found by ``DownFacingCapFinder``), ``cap_area_nm2``,
``loop_area_nm2``, ``base_radius_nm``, ``attachment_method``, ``source`` (path relative to the source folder) and
``metric_<name>`` for every scalar morphometric (NaN when not computed). Vertices are rounded to float32 BEFORE the
attachment region and the morphometrics are computed, so the stored numbers describe exactly the stored meshes.
"""

from __future__ import annotations

import datetime as _dt
import functools
import json
import math
import shutil
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

import numpy as np

from ..branch_assembly import AssemblyConfig, PoolSpine, SpinePool, find_mesh_files
from ..branch_assembly.selection import ContextSelector, base_radius
from ..spine_morphometrics import SCALAR_METRICS
from ..trunk_generation.conditions import CATEGORIES

FORMAT_VERSION = 1
ANY_KEY = "any"
SEED_MAX = 2**31 - 1
DEFAULT_LIMITS = {"max_length_nm": 100_000.0, "min_length_nm": 5_000.0, "max_soma_distance_nm": 600_000.0, "max_spines": 400}


def library_key_parts(key: str) -> Dict[str, Optional[str]]:
    """``mouse/excitatory/BPC`` -> {species, cell_type_binary, cell_type}; ``any`` -> all None. Raises on an invalid key."""
    if key == ANY_KEY:
        return {"species": None, "cell_type_binary": None, "cell_type": None}
    parts = key.split("/")
    if not 1 <= len(parts) <= 3 or any(not p for p in parts):
        raise ValueError(f"spine library key {key!r}: expected 'any', '<species>', '<species>/<cell_type_binary>' or '<species>/<cell_type_binary>/<cell_type>'")
    if parts[0] not in CATEGORIES["species"]:
        raise ValueError(f"spine library key {key!r}: unknown species {parts[0]!r} (allowed {CATEGORIES['species']})")
    if len(parts) >= 2 and parts[1] not in CATEGORIES["cell_type_binary"]:
        raise ValueError(f"spine library key {key!r}: unknown binary cell type {parts[1]!r} (allowed {CATEGORIES['cell_type_binary']})")
    return {"species": parts[0], "cell_type_binary": parts[1] if len(parts) >= 2 else None, "cell_type": parts[2] if len(parts) == 3 else None}


def library_key_chain(species: Optional[str], cell_type_binary: Optional[str], cell_type: Optional[str]) -> List[str]:
    """Candidate library keys of a request, most specific first (missing parts are skipped)."""
    chain = []
    if species:
        if cell_type_binary:
            if cell_type:
                chain.append(f"{species}/{cell_type_binary}/{cell_type}")
            chain.append(f"{species}/{cell_type_binary}")
        chain.append(species)
    chain.append(ANY_KEY)
    return chain


# ---------------------------------------------------------------------------
# spine library
# ---------------------------------------------------------------------------


@dataclass
class SpineLibrary:
    """Spines of one library key, unpacked (local frame, nm)."""

    key: str
    vertices: List[np.ndarray]
    faces: List[np.ndarray]
    center: np.ndarray
    cap_area_nm2: np.ndarray
    loop_area_nm2: np.ndarray
    base_radius_nm: np.ndarray
    attachment_method: List[str]
    source: List[str]
    metrics: Dict[str, np.ndarray] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.vertices)

    def pool(self, base_band_nm: float) -> SpinePool:
        """The assembly pool. Pool keys are ``<library key>#<spine id>`` (the engine maps a placed spine back to its row)."""
        spines = []
        for i, (v, f) in enumerate(zip(self.vertices, self.faces)):
            c = self.center[i]
            spines.append(PoolSpine(key=f"{self.key}#{i}", vertices=v, faces=f, center=c, height=float(v[:, 1].max() - c[1]),
                                    base_mask=v[:, 1] <= c[1] + base_band_nm, attachment_method=self.attachment_method[i],
                                    base_radius_nm=float(self.base_radius_nm[i])))
        return SpinePool(spines)

    def spine_metrics(self, i: int) -> Dict[str, Optional[float]]:
        out = {}
        for name, values in self.metrics.items():
            v = float(values[i])
            out[name] = v if math.isfinite(v) else None
        return out

    # ---- persistence ----------------------------------------------------------------------------------------------
    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        nv = np.array([len(v) for v in self.vertices], dtype=np.int64)
        nf = np.array([len(f) for f in self.faces], dtype=np.int64)
        arrays = {
            "vertices": np.concatenate(self.vertices).astype(np.float32) if len(self) else np.zeros((0, 3), np.float32),
            "faces": np.concatenate(self.faces).astype(np.int32) if len(self) else np.zeros((0, 3), np.int32),
            "v_offsets": np.r_[0, np.cumsum(nv)].astype(np.int64), "f_offsets": np.r_[0, np.cumsum(nf)].astype(np.int64),
            "center": np.asarray(self.center, dtype=np.float64).reshape(-1, 3), "cap_area_nm2": np.asarray(self.cap_area_nm2, dtype=np.float64),
            "loop_area_nm2": np.asarray(self.loop_area_nm2, dtype=np.float64), "base_radius_nm": np.asarray(self.base_radius_nm, dtype=np.float64),
            "attachment_method": np.asarray(self.attachment_method, dtype=str), "source": np.asarray(self.source, dtype=str),
        }
        arrays.update({f"metric_{k}": np.asarray(v, dtype=np.float64) for k, v in self.metrics.items()})
        np.savez_compressed(path, **arrays)

    @classmethod
    def load(cls, key: str, path: Path) -> "SpineLibrary":
        with np.load(path, allow_pickle=False) as d:
            vo, fo = d["v_offsets"], d["f_offsets"]
            v_all, f_all = d["vertices"].astype(np.float64), d["faces"].astype(np.int64)
            n = len(vo) - 1
            return cls(key=key, vertices=[v_all[vo[i]:vo[i + 1]] for i in range(n)], faces=[f_all[fo[i]:fo[i + 1]] for i in range(n)],
                       center=d["center"], cap_area_nm2=d["cap_area_nm2"], loop_area_nm2=d["loop_area_nm2"], base_radius_nm=d["base_radius_nm"],
                       attachment_method=[str(x) for x in d["attachment_method"]], source=[str(x) for x in d["source"]],
                       metrics={k[len("metric_"):]: d[k] for k in d.files if k.startswith("metric_")})


def build_spine_library(key: str, files: Sequence[Path], root: Optional[Path] = None, *, compute_morphometrics: bool = True,
                        log: Callable[[str], None] = lambda _: None) -> tuple:
    """Spine meshes -> (SpineLibrary, rejected rows). The same acceptance as the assembly pool (``SpinePool.from_meshes``:
    closed, consistent winding, positive volume, attachment region found; largest component of a multi-component mesh)."""
    import trimesh

    from ..spine_generation.evaluation.attachment import DownFacingCapFinder
    from ..spine_morphometrics import AttachmentRegion, compute_scalar_metrics

    finder = DownFacingCapFinder()
    lib = SpineLibrary(key, [], [], np.zeros((0, 3)), np.zeros(0), np.zeros(0), np.zeros(0), [], [], {m: np.zeros(0) for m in SCALAR_METRICS})
    centers, caps, loops, bases, metric_rows = [], [], [], [], []
    rejected: List[Dict[str, Any]] = []
    for k, path in enumerate(files):
        rel = str(Path(path).relative_to(root)) if root is not None else str(path)
        try:
            mesh = trimesh.load(path, process=True, force="mesh")
        except Exception:  # unreadable file -> rejected, as in SpinePool.from_paths
            mesh = None
        accepted = SpinePool.from_meshes([(rel, mesh)], finder=finder)
        if not len(accepted):
            rejected.extend(accepted.rejected)
            continue
        sp = accepted.spines[0]
        # round to the stored precision first: the region and the metrics then describe the stored mesh exactly
        stored = trimesh.Trimesh(sp.vertices.astype(np.float32).astype(np.float64), sp.faces, process=False)
        region = finder(stored)
        if region is None:
            rejected.append({"key": rel, "reason": "no_attachment_region_after_rounding"})
            continue
        values = {m: None for m in SCALAR_METRICS}
        if compute_morphometrics:
            try:
                values.update(compute_scalar_metrics(stored, AttachmentRegion(np.asarray(region.center, float), region.cap_area, region.loop_area, region.method)))
            except Exception as exc:  # one failing metric run must not drop the spine: its metrics stay NaN
                log(f"[webapp-bundle] {key}: morphometrics failed for {rel}: {exc!r}")
        lib.vertices.append(np.asarray(stored.vertices))
        lib.faces.append(np.asarray(stored.faces, dtype=np.int64))
        lib.attachment_method.append(str(region.method))
        lib.source.append(rel)
        centers.append(np.asarray(region.center, dtype=float))
        caps.append(np.nan if region.cap_area is None else float(region.cap_area))
        loops.append(np.nan if getattr(region, "loop_area", None) is None else float(region.loop_area))
        bases.append(base_radius(getattr(region, "loop_area", None), getattr(region, "cap_area", None)))
        metric_rows.append(values)
        if (k + 1) % 50 == 0:
            log(f"[webapp-bundle] {key}: {k + 1} / {len(files)} meshes")
    lib.center = np.asarray(centers, dtype=float).reshape(-1, 3)
    lib.cap_area_nm2, lib.loop_area_nm2, lib.base_radius_nm = (np.asarray(x, dtype=float) for x in (caps, loops, bases))
    lib.metrics = {m: np.array([np.nan if r.get(m) is None else float(r[m]) for r in metric_rows], dtype=float) for m in SCALAR_METRICS}
    return lib, rejected


# ---------------------------------------------------------------------------
# bundle
# ---------------------------------------------------------------------------


@dataclass
class ModelBundle:
    root: Path
    manifest: Dict[str, Any]
    t0_model: Any
    trunk_limits: Any
    p0: Dict[str, Any]
    context_selector: Optional[ContextSelector]
    assembly: AssemblyConfig
    reference: Dict[str, Any] = field(default_factory=dict)
    _libraries: Dict[str, SpineLibrary] = field(default_factory=dict, repr=False)
    _lock: Any = field(default_factory=threading.Lock, repr=False)

    @property
    def bundle_id(self) -> str:
        return str(self.manifest["bundle_id"])

    @property
    def library_keys(self) -> List[str]:
        return list(self.manifest.get("spine_library", {}))

    def library(self, key: str) -> SpineLibrary:
        """Spines of one key, read on first use and kept (a public server keeps every key it has served in memory)."""
        if key not in self.manifest.get("spine_library", {}):
            raise KeyError(key)
        with self._lock:  # generation runs in a thread pool: read each file once
            if key not in self._libraries:
                self._libraries[key] = SpineLibrary.load(key, self.root / "spine_library" / key / "spines.npz")
            return self._libraries[key]

    @classmethod
    def load(cls, path: Path) -> "ModelBundle":
        from ..placement_generation.fit import load_p0_bundle
        from ..trunk_generation.constraints import TrunkLimits
        from ..trunk_generation.models.t0_statistical import T0Model

        root = Path(path)
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        if int(manifest.get("format_version", 0)) != FORMAT_VERSION:
            raise ValueError(f"{root}: bundle format {manifest.get('format_version')} (this code reads {FORMAT_VERSION})")
        selector_path = root / "context_selector.json"
        selector = ContextSelector.from_dict(json.loads(selector_path.read_text(encoding="utf-8"))) if selector_path.exists() else None
        reference = {}
        for p in sorted((root / "reference").glob("*.json")) if (root / "reference").exists() else []:
            reference[p.stem] = json.loads(p.read_text(encoding="utf-8"))
        return cls(root=root, manifest=manifest, t0_model=T0Model.load(root / "t0_model.json"),
                   trunk_limits=TrunkLimits(**json.loads((root / "limits.json").read_text(encoding="utf-8"))),
                   p0=load_p0_bundle(root / "p0_bundle.json"), context_selector=selector,
                   assembly=AssemblyConfig.from_mapping(manifest.get("assembly")), reference=reference)


def _categories_from_groups(groups: Sequence[str]) -> Dict[str, List[str]]:
    """Allowed values of the trunk conditions: those of the trained T0 groups (species|cell type|compartment), else all."""
    out = {}
    for k, name in enumerate(CATEGORIES):
        seen = [g.split("|")[k] for g in groups if len(g.split("|")) == 3]
        values = [c for c in CATEGORIES[name] if c in seen]
        out[name] = values or list(CATEGORIES[name])
    return out


def build_bundle(
    out_root: Path,
    *,
    t0_run: Path,
    p0_run: Path,
    spines: Mapping[str, Path],
    bundle_id: Optional[str] = None,
    t0_model: Optional[str] = None,
    context_spines: Any = None,
    branches_reference: Any = None,
    assembly: Optional[Mapping[str, Any]] = None,
    compute_morphometrics: bool = True,
    max_spines_per_key: int = 0,
    limits: Optional[Mapping[str, Any]] = None,
    defaults: Optional[Mapping[str, Any]] = None,
    all_categories: bool = False,
    log: Callable[[str], None] = functools.partial(print, flush=True),
) -> Path:
    """Write ``<out_root>/<bundle_id>/`` from a T0 run, a P0 run and spine mesh folders per library key.

    ``context_spines``: ``branch_spines.parquet`` (path or DataFrame) of REAL spines -> ``context_selector.json``;
    ``branches_reference``: ``branches.parquet`` of the branch preprocessing (path, folder with it, or DataFrame) ->
    ``reference/trunk_placement.json``. ``spines`` keys: see the module docstring; an ``any`` key is required (the last fallback).
    The trunk conditions offered to the user are those of the trained T0 groups; ``all_categories`` offers every category
    (an untrained combination then gets the pooled T0 parameters, and the engine warns about it)."""
    import pandas as pd

    from .metrics import reference_from_branches

    t0_run, p0_run = Path(t0_run), Path(p0_run)
    if ANY_KEY not in spines:
        raise ValueError(f"the spine library needs an '{ANY_KEY}' key (the fallback of every request); got {sorted(spines)}")
    for key in spines:
        library_key_parts(key)
    chosen = t0_model or json.loads((t0_run / "report.json").read_text(encoding="utf-8"))["chosen"]
    bundle_id = bundle_id or f"{_dt.datetime.now():%Y-%m-%d_%H%M%S}_branch-generator"
    root = Path(out_root) / bundle_id
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"{root} exists and is not empty: choose another --bundle-id")
    root.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(t0_run / f"t0_{chosen}.json", root / "t0_model.json")
    shutil.copyfile(t0_run / "limits.json", root / "limits.json")
    shutil.copyfile(p0_run / "p0_bundle.json", root / "p0_bundle.json")
    t0 = json.loads((root / "t0_model.json").read_text(encoding="utf-8"))
    p0 = json.loads((root / "p0_bundle.json").read_text(encoding="utf-8"))

    selection = "uniform"
    if context_spines is not None:
        frame = context_spines if isinstance(context_spines, pd.DataFrame) else pd.read_parquet(context_spines)
        selector = ContextSelector.fit(frame)
        (root / "context_selector.json").write_text(json.dumps(selector.to_dict(), indent=2), encoding="utf-8")
        selection = "context"
        log(f"[webapp-bundle] context selection: log b = {selector.a:.3f} + {selector.c:.3f} log(r / 300 nm) +- {selector.s:.3f} ({selector.info['n']} real spines)")

    cfg = AssemblyConfig.from_mapping(dict(assembly or {}))
    library_info: Dict[str, Any] = {}
    cell_types: Dict[str, str] = {}
    for key, folder in spines.items():
        files = find_mesh_files(Path(folder), limit=max_spines_per_key or None)
        if not files:
            raise ValueError(f"no spine meshes under {folder} (key {key})")
        lib, rejected = build_spine_library(key, files, Path(folder), compute_morphometrics=compute_morphometrics, log=log)
        if not len(lib):
            raise ValueError(f"key {key}: none of {len(files)} meshes is usable ({rejected[:3]} ...)")
        lib.save(root / "spine_library" / key / "spines.npz")
        reasons: Dict[str, int] = {}
        for r in rejected:
            reasons[r["reason"]] = reasons.get(r["reason"], 0) + 1
        library_info[key] = {"n_spines": len(lib), "n_rejected": len(rejected), "rejected_reasons": reasons, "source": str(folder),
                             "morphometrics": bool(compute_morphometrics), "median_base_radius_nm": float(np.nanmedian(lib.base_radius_nm))}
        parts = library_key_parts(key)
        if parts["cell_type"]:
            cell_types[parts["cell_type"]] = parts["cell_type_binary"]
        log(f"[webapp-bundle] library {key}: {len(lib)} spines ({len(rejected)} rejected {reasons})")

    reference_groups = None
    if branches_reference is not None:
        if isinstance(branches_reference, pd.DataFrame):
            table = branches_reference
        else:
            path = Path(branches_reference)
            table = pd.read_parquet(path / "branches.parquet" if path.is_dir() else path)
        ref = reference_from_branches(table)
        (root / "reference").mkdir(exist_ok=True)
        (root / "reference" / "trunk_placement.json").write_text(json.dumps(ref, indent=2, default=float), encoding="utf-8")
        reference_groups = sorted(ref["groups"])
        log(f"[webapp-bundle] reference: {len(reference_groups)} condition groups from {len(table)} branches")

    groups = list(t0.get("fit_info", {}).get("groups", []))
    lim = dict(DEFAULT_LIMITS, **dict(limits or {}))
    length_bounds = [float(b) for b in t0.get("length_bounds", [5000.0, 255500.0])]
    lim["min_length_nm"] = max(float(lim["min_length_nm"]), length_bounds[0])
    lim["max_length_nm"] = min(float(lim["max_length_nm"]), length_bounds[1])
    categories = {k: list(v) for k, v in CATEGORIES.items()} if all_categories else _categories_from_groups(groups)
    default_request = {"species": categories["species"][0], "cell_type_binary": categories["cell_type_binary"][0],
                       "compartment": categories["compartment"][0], "spine_cell_type": None,
                       "length_nm": float(min(max(20_000.0, lim["min_length_nm"]), lim["max_length_nm"])),
                       "soma_distance_start_nm": 50_000.0, "is_terminal": False, "seed": 1}
    default_request.update(dict(defaults or {}))
    manifest = {
        "bundle_id": bundle_id, "format_version": FORMAT_VERSION, "created": _dt.datetime.now().isoformat(timespec="seconds"), "units": "nm",
        "sources": {"t0_run": str(t0_run), "t0_model": chosen, "p0_run": str(p0_run), "spines": {k: str(v) for k, v in spines.items()},
                    "context_spines": None if context_spines is None or isinstance(context_spines, pd.DataFrame) else str(context_spines),
                    "branches_reference": None if branches_reference is None or isinstance(branches_reference, pd.DataFrame) else str(branches_reference)},
        "models": {"t0": {"mean": t0.get("mean", {}).get("kind"), "direction": t0.get("direction_mode"), "radius_residual": t0.get("radius_mode"),
                          "uses_soma_distance": "soma_log_mean" in t0.get("encoder", {}), "length_bounds_nm": length_bounds},
                   "p0": {"count_family": p0["count_model"]["family"], "exposure": p0["count_model"]["exposure"],
                          "neuron_effect": bool(p0["count_model"]["random_effect"]), "d_min_nm": float(p0["d_min_nm"])}},
        "categories": {**categories, "spine_cell_type": sorted(cell_types)},
        "spine_cell_type_binary": cell_types,
        "trained_groups": groups,
        "defaults": default_request,
        "limits": {**lim, "seed_max": SEED_MAX},
        "assembly": cfg.__dict__,
        "spine_selection": selection,
        "spine_library": library_info,
        "reference_groups": reference_groups,
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
    log(f"[webapp-bundle] -> {root}")
    return root
