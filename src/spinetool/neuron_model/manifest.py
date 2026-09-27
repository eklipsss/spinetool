from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import pandas as pd


def manifest_path_for(output_path: Path) -> Path:
    return output_path.with_suffix(output_path.suffix + ".manifest.json")


def write_table_manifest(
    output_path: Path,
    frame: pd.DataFrame,
    *,
    artifact_name: str,
    processing_version: str,
    config_hash: str,
    input_paths: Optional[Mapping[str, Any]] = None,
    status_column: Optional[str] = None,
    extra: Optional[Mapping[str, Any]] = None,
) -> Path:
    status_counts: Dict[str, int] = {}
    if status_column and status_column in frame.columns:
        status_counts = {
            str(key): int(value)
            for key, value in frame[status_column].value_counts(dropna=False).to_dict().items()
        }

    payload: Dict[str, Any] = {
        "artifact_name": artifact_name,
        "output_path": str(output_path),
        "created_at_unix": time.time(),
        "processing_version": processing_version,
        "config_hash": config_hash,
        "n_rows": int(len(frame)),
        "n_columns": int(len(frame.columns)),
        "columns": [str(column) for column in frame.columns],
        "status_column": status_column,
        "status_counts": status_counts,
        "input_paths": _stringify(input_paths or {}),
    }
    if extra:
        payload["extra"] = _stringify(extra)

    path = manifest_path_for(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp")
    with tmp_path.open("w", encoding="utf-8") as fd:
        json.dump(payload, fd, ensure_ascii=False, indent=2)
        fd.write("\n")
    tmp_path.replace(path)
    return path


def _stringify(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _stringify(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_stringify(item) for item in value]
    return value

