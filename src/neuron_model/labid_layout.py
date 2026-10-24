"""LabID -> the raw layout of Minnie65, so that the S preprocessing (stages 1-9) runs on LabID unchanged.

LabID (``<labid>/<group>/<dendrite>/``): ``surface_mesh.off`` (shaft + spines), ``pure_surface_mesh.off`` (shaft
only, closed) and ``spine_<n>.off`` - ALREADY SEALED spine meshes cut from ``surface_mesh``, in micrometres, all in
one coordinate system. The S preprocessing expects raw OPEN spine meshes whose hole at the base is the attachment
loop (stage 1 seals them itself and finds the loop and the cap). So every LabID spine is "unsealed":

- the lab's seal = the faces of the spine mesh that are NOT faces of ``surface_mesh`` (the spine vertices are a subset
  of the surface vertices; checked on the local data: one connected patch per spine, centre within tens of nm of the
  shaft surface). They are removed; what is left is the spine surface exactly as it was on the dendrite, with one
  boundary loop where it met the shaft - the same kind of input as a raw Minnie65 spine. Stage 1 then seals it with
  the project's own algorithm, so the cap, its area, the loop and the local frame are defined as for Minnie65.
- a seal in several patches (two holes touching at a vertex, seen once in 17 local spines) is opened as well:
  stage 1 resolves several loops / a figure-eight (patch fallback) or marks the spine needs_review;
- spines whose seal cannot be found (no new face, or every face new) are written unchanged and flagged in the
  conversion table; stage 1 will mark them (no boundary loop -> attachment not found).

Written per LabID folder (``<out>/labid/<branch id>/limb_000/branch_000/``; every folder is a separate branch and its
own grouping unit, ``metadata.labid_unit_id``): ``branch_mesh.off`` (= surface_mesh),
``shaft_mesh.off`` (= pure_surface_mesh), ``branch_skeleton.npy`` ((N, 2, 3) segments of the longest path of the CGAL
skeleton of the closed shaft: an axis line, used by stage 1 for scoring candidate loops, stage 2 for the
false-spine distance profile, stage 3 for the tangent axis), ``spines/spine_<n>.off`` (open, nm), ``_converted.json``
(source folder, group, the numbers of the folder name as metadata, scales). The S preprocessing AND the LabID branch
preprocessing read this tree: units and scales are applied only here.

Scales: ``unit_nm`` (mesh unit -> nm, default 1000: micrometres) and ``z_scale`` (factor on z per group, key ``all``
for every group; OFF by default) for meshes built with a wrong voxel size. They change the geometry but no config
hash downstream, so ``<out>/labid/conversion_info.json`` records them and a conversion into a tree made with other
scales is refused: a new scale = a new output folder (and new S / branch outputs).
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

UNIT_NM = 1000.0


def _scaled(mesh, z_factor: float = 1.0, unit_nm: float = UNIT_NM):
    import trimesh

    v = np.asarray(mesh.vertices, dtype=float) * unit_nm
    if z_factor != 1.0:
        v[:, 2] *= z_factor
    return trimesh.Trimesh(v, np.asarray(mesh.faces, dtype=np.int64), process=False)


def seal_faces(spine, surface_faces: set, tree) -> np.ndarray:
    """Boolean mask of the spine faces that are not faces of the surface mesh (the lab's seal)."""
    _, idx = tree.query(spine.vertices)
    mapped = np.sort(idx[spine.faces], axis=1)
    return np.fromiter((tuple(r) not in surface_faces for r in mapped), dtype=bool, count=len(mapped))


def unseal(spine, surface_faces: set, tree) -> Dict[str, Any]:
    """Open the spine at its base: drop the seal faces. Returns the mesh to write and a description."""
    import trimesh

    new = seal_faces(spine, surface_faces, tree)
    info: Dict[str, Any] = {"n_faces": int(len(new)), "n_seal_faces": int(new.sum())}
    if not new.any() or new.all():
        info["unseal_status"] = "seal_not_found"
        return {"mesh": spine, **info}
    seal = trimesh.Trimesh(spine.vertices, spine.faces[new], process=False)
    info["n_seal_patches"] = int(len(seal.split(only_watertight=False)))
    info["seal_area_nm2"] = float(spine.area_faces[new].sum())
    opened = trimesh.Trimesh(spine.vertices, spine.faces[~new], process=False)
    opened.remove_unreferenced_vertices()  # vertex coordinates unchanged (still a subset of the surface mesh)
    from .spine_geometry import boundary_loops

    info["n_boundary_loops"] = int(len(boundary_loops(opened)))
    # several patches (e.g. two holes touching at a vertex, sealed in two pieces) are opened too: stage 1 handles
    # several loops / a figure-eight (patch fallback) or marks the spine needs_review
    info["unseal_status"] = "ok" if info["n_boundary_loops"] == 1 and info["n_seal_patches"] == 1 else "several_loops"
    return {"mesh": opened, **info}


def skeleton_segments(shaft, timeout: float = 300.0) -> np.ndarray:
    from .spine_geometry import cgal_skeleton_segments_from_trimesh_isolated, longest_path_polyline

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "shaft.off"
        shaft.export(path)
        polyline = longest_path_polyline(cgal_skeleton_segments_from_trimesh_isolated(path, timeout=timeout))
    return np.stack([polyline[:-1], polyline[1:]], axis=1)


def discover_raw(labid_root: Path, units: Optional[List[str]] = None) -> pd.DataFrame:
    """One row per LabID folder with a shaft mesh: group, dendrite (folder), path, neuron_id (= branch id), name numbers."""
    from .metadata import labid_name_numbers, labid_unit_id

    rows = []
    for group_dir in sorted(p for p in Path(labid_root).iterdir() if p.is_dir()):
        for d in sorted(p for p in group_dir.iterdir() if p.is_dir()):
            if (d / "pure_surface_mesh.off").exists():
                n1, n2 = labid_name_numbers(d.name)
                rows.append({"group": group_dir.name, "dendrite": d.name, "path": str(d), "neuron_id": labid_unit_id(group_dir.name, d.name),
                             "name_number_1": n1, "name_number_2": n2})
    frame = pd.DataFrame(rows, columns=["group", "dendrite", "path", "neuron_id", "name_number_1", "name_number_2"])
    dup = frame["neuron_id"].duplicated(keep=False)
    if dup.any():
        raise ValueError(f"LabID folder names collide after sanitising: {frame.loc[dup, ['group', 'dendrite']].values.tolist()}")
    if units:
        frame = frame[frame["neuron_id"].isin(set(units))]
    return frame.reset_index(drop=True)


def _z_of(z_scale: Optional[Dict[str, float]], group: str) -> float:
    z_scale = z_scale or {}
    return float(z_scale.get(group, z_scale.get("all", 1.0)))


def convert_dendrite(item: Dict[str, Any], out_dataset_root: Path, *, unit_nm: float = UNIT_NM, z_scale: Optional[Dict[str, float]] = None,
                     overwrite: bool = False) -> List[Dict[str, Any]]:
    """One LabID folder -> ``<out_dataset_root>/<branch id>/limb_000/branch_000/``. Returns one row per spine."""
    from scipy.spatial import cKDTree

    from .spine_geometry import load_trimesh

    src = Path(item["path"])
    z = _z_of(z_scale, item["group"])
    branch_dir = out_dataset_root / item["neuron_id"] / "limb_000" / "branch_000"
    marker = branch_dir / "_converted.json"
    if marker.exists() and not overwrite:
        return json.loads(marker.read_text(encoding="utf-8"))["spines"]
    (branch_dir / "spines").mkdir(parents=True, exist_ok=True)
    surface = _scaled(load_trimesh(src / "surface_mesh.off", process=False), z, unit_nm)
    shaft = _scaled(load_trimesh(src / "pure_surface_mesh.off", process=False), z, unit_nm)
    surface.export(branch_dir / "branch_mesh.off")
    shaft.export(branch_dir / "shaft_mesh.off")
    np.save(branch_dir / "branch_skeleton.npy", skeleton_segments(shaft))
    faces = set(map(tuple, np.sort(np.asarray(surface.faces), axis=1)))
    tree = cKDTree(np.asarray(surface.vertices))
    rows = []
    for path in sorted(src.glob("spine_*.off"), key=lambda p: int("".join(ch for ch in p.stem if ch.isdigit()) or 0)):
        spine_id = "".join(ch for ch in path.stem if ch.isdigit())
        res = unseal(_scaled(load_trimesh(path, process=False), z, unit_nm), faces, tree)
        res.pop("mesh").export(branch_dir / "spines" / f"spine_{spine_id}.off")
        rows.append({"neuron_id": item["neuron_id"], "limb_id": "000", "branch_id": "000", "spine_id": spine_id,
                     "source_dir": f"{item['group']}/{item['dendrite']}", "group": item["group"], "unit_nm": unit_nm, "z_scale": z, **res})
    meta = {"source": str(src), "source_dir": f"{item['group']}/{item['dendrite']}", "group": item["group"], "dendrite": item["dendrite"],
            "name_number_1": item.get("name_number_1"), "name_number_2": item.get("name_number_2"), "unit_nm": unit_nm, "z_scale": z}
    marker.write_text(json.dumps({**meta, "spines": rows}, indent=1, default=float), encoding="utf-8")
    return rows


def convert(labid_root: Path, out_root: Path, *, units: Optional[List[str]] = None, unit_nm: float = UNIT_NM, z_scale: Optional[Dict[str, float]] = None,
            overwrite: bool = False, log=print) -> pd.DataFrame:
    """Every folder of ``labid_root`` (``<raw>/labid``) -> ``<out_root>/labid/...``; writes ``labid_conversion.parquet``."""
    items = discover_raw(labid_root, units).to_dict("records")
    out_dataset_root = Path(out_root) / "labid"
    out_dataset_root.mkdir(parents=True, exist_ok=True)
    info_path = out_dataset_root / "conversion_info.json"
    info = {"unit_nm": float(unit_nm), "z_scale": {k: float(v) for k, v in sorted((z_scale or {}).items())}, "labid_root": str(labid_root)}
    if info_path.exists():
        old = json.loads(info_path.read_text(encoding="utf-8"))
        if (old.get("unit_nm"), old.get("z_scale")) != (info["unit_nm"], info["z_scale"]):
            raise ValueError(f"{out_dataset_root} was converted with unit_nm={old.get('unit_nm')}, z_scale={old.get('z_scale')}; "
                             f"now unit_nm={info['unit_nm']}, z_scale={info['z_scale']}: use a NEW output folder (S and branch outputs too)")
    info_path.write_text(json.dumps(info, indent=2), encoding="utf-8")
    rows: List[Dict[str, Any]] = []
    failed = []
    for k, item in enumerate(items):
        try:
            rows.extend(convert_dendrite(item, out_dataset_root, unit_nm=unit_nm, z_scale=z_scale, overwrite=overwrite))
        except Exception as exc:  # one bad folder must not stop the conversion
            failed.append({"source_dir": f"{item['group']}/{item['dendrite']}", "error": repr(exc)[:300]})
        if log and (k + 1) % 20 == 0:
            log(f"[labid] {k + 1}/{len(items)} branches")
    table = pd.DataFrame(rows)
    table.to_parquet(out_dataset_root / "labid_conversion.parquet", index=False)
    if failed:
        pd.DataFrame(failed).to_parquet(out_dataset_root / "labid_conversion_failed.parquet", index=False)
    if log:
        counts = table["unseal_status"].value_counts().to_dict() if len(table) else {}
        log(f"[labid] {len(items)} branches ({len(failed)} failed), {len(table)} spines -> {out_dataset_root} "
            f"(unit_nm {unit_nm}, z_scale {info['z_scale'] or 'none'}); unseal: {counts}")
    return table
