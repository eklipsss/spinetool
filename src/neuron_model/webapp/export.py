"""Download of a generated branch: one zip archive with every mesh and table (units: nm).

Archive layout::

    branch.off                    assembled closed surface (trunk + spines, one boolean union)
    trunk.off                     trunk tube alone (closed, flat caps)
    spines_local/spine_XXX.off    every placed spine in its LOCAL frame (origin = attachment centre, +y into the spine)
    spines_global/spine_XXX.off   the same spine placed on the branch (branch coordinates, before the union)
    metrics.json                  trunk, placement and branch-mesh metrics (nm) + library used + warnings
    spines.csv                    one row per placement (placed or dropped), morphometrics in nm / nm^2 / nm^3
    trunk_centerline.csv          s, x, y, z, r of the trunk nodes (nm)
    request.json                  every parameter + seed + bundle id: the exact reproduction
    README.txt                    units, frames, how to reproduce

``XXX`` = placement index (the row ``index`` of ``spines.csv``); dropped placements have no mesh.
"""

from __future__ import annotations

import csv
import io
import json
import math
import zipfile
from typing import Any, Dict, Tuple

import numpy as np

README = """Сгенерированная ветвь дендрита с шипиками
==========================================

Единицы: все координаты и длины в нанометрах (нм), площади в нм², объёмы в нм³.

Файлы
-----
branch.off                    собранная замкнутая поверхность ветви (ствол + шипики, одно булево объединение)
trunk.off                     только ствол (трубка с плоскими торцами)
spines_local/spine_XXX.off    каждый поставленный шипик в ЛОКАЛЬНОЙ системе координат шипика:
                              начало — центр основания, ось +y — от ствола в тело шипика, ось x — вдоль ветви
spines_global/spine_XXX.off   тот же шипик, поставленный на ветвь (координаты ветви, до объединения со стволом)
metrics.json                  метрики ствола, расстановки шипиков и сетки ветви (нм), использованная библиотека шипиков
spines.csv                    строка на каждое положение шипика (поставленный или нет), морфометрики шипика
trunk_centerline.csv          узлы центральной линии ствола: s, x, y, z, радиус r (нм)
request.json                  все параметры генерации, зерно и идентификатор набора моделей

XXX — номер положения (столбец index в spines.csv). Положения со статусом dropped сетки не имеют:
ни один шипик библиотеки не встал на ствол (причина в столбце failure_reasons).

Как воспроизвести
-----------------
Тот же request.json на сервере с тем же набором моделей (bundle_id) даёт ту же ветвь:
POST /api/generate с телом request.json (без поля bundle_id), либо в Python:

    from src.neuron_model.webapp import BranchGenerator, ModelBundle
    gen = BranchGenerator(ModelBundle.load("models/webapp/<bundle_id>"))
    result = gen.generate(request)

Ограничения
-----------
Ствол — статистическая модель T0, число и положения шипиков — модель P0, шипики взяты из заранее
сгенерированной библиотеки для выбранных условий (подбор по радиусу ствола). Стык шипика со стволом
не перестраивается: сетка объединения сохраняет треугольники, вырезанные булевой операцией.
"""


def off_text(vertices: np.ndarray, faces: np.ndarray) -> str:
    """OFF file content (trimesh writer, 10 significant digits: sub-nm on any branch coordinate)."""
    import trimesh
    from trimesh.exchange.off import export_off

    return export_off(trimesh.Trimesh(np.asarray(vertices, dtype=float), np.asarray(faces, dtype=np.int64), process=False), digits=10)


def _csv(rows, columns) -> str:
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=columns, extrasaction="ignore", lineterminator="\n")
    writer.writeheader()
    for r in rows:
        writer.writerow({k: ("" if r.get(k) is None else r.get(k)) for k in columns})
    return buf.getvalue()


def spines_csv(result) -> str:
    rows = []
    for r in result.metrics["spines"]:
        row = dict(r)
        anchor = row.pop("anchor_nm", None) or [None] * 3
        row.update(anchor_x_nm=anchor[0], anchor_y_nm=anchor[1], anchor_z_nm=anchor[2])
        rows.append(row)
    columns = list(rows[0]) if rows else ["index", "status"]
    return _csv(rows, columns)


def centerline_csv(trunk: Dict[str, Any]) -> str:
    s, g, r = (np.asarray(trunk[k], dtype=float) for k in ("s", "gamma", "radius"))
    rows = [{"s_nm": s[i], "x_nm": g[i, 0], "y_nm": g[i, 1], "z_nm": g[i, 2], "r_nm": r[i]} for i in range(len(s))]
    return _csv(rows, ["s_nm", "x_nm", "y_nm", "z_nm", "r_nm"])


def metrics_document(result) -> Dict[str, Any]:
    m = result.metrics
    return {"bundle_id": result.bundle_id, "result_id": result.result_id, "units": "nm", "request": result.request, "library": result.library,
            "warnings": result.warnings, "trunk": m["trunk"], "placement": m["placement"], "branch_mesh": m["branch_mesh"],
            "assembly_status": result.assembly.branch.get("status"), "assembly_reason": result.assembly.branch.get("reason"),
            "reference": m.get("reference"), "timings_s": result.timings}


def request_document(result) -> Dict[str, Any]:
    return {"bundle_id": result.bundle_id, **result.request}


def build_zip(result) -> bytes:
    """The archive of a ``GenerationResult`` as bytes (the server streams it, a script can write it to a file)."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr("branch.off", off_text(*result.branch_mesh))
        z.writestr("trunk.off", off_text(*result.trunk_mesh))
        for frame, meshes in (("local", result.spines_local), ("global", result.spines_global)):
            for k, (v, f) in sorted(meshes.items()):
                z.writestr(f"spines_{frame}/spine_{k:03d}.off", off_text(v, f))
        z.writestr("metrics.json", dumps(metrics_document(result)))
        z.writestr("spines.csv", spines_csv(result))
        z.writestr("trunk_centerline.csv", centerline_csv(result.trunk))
        z.writestr("request.json", dumps(request_document(result)))
        z.writestr("README.txt", README)
    return buf.getvalue()


def json_safe(value: Any) -> Any:
    """NaN / inf -> None everywhere (a browser's JSON.parse rejects NaN), numpy -> python."""
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def dumps(value: Any) -> str:
    return json.dumps(json_safe(value), indent=2, ensure_ascii=False, allow_nan=False, default=str)


def mesh_of(result, what: str, index: int = -1, frame: str = "local") -> Tuple[np.ndarray, np.ndarray]:
    """(vertices, faces) of ``branch`` / ``trunk`` / ``spine`` (``index`` = placement index, ``frame`` local | global)."""
    if what == "branch":
        return result.branch_mesh
    if what == "trunk":
        return result.trunk_mesh
    if frame not in ("local", "global"):
        raise ValueError(f"frame must be 'local' or 'global', got {frame!r}")
    meshes = result.spines_local if frame == "local" else result.spines_global
    if index not in meshes:
        raise KeyError(index)
    return meshes[index]
