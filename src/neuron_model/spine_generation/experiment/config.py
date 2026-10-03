"""YAML experiment configs with inheritance and CLI overrides.

A config file may list parent files under ``defaults`` (paths relative to the
file itself); parents are deep-merged first, in order, then the file's own
keys on top. ``overrides`` are ``"dotted.key=value"`` strings whose value is
parsed as YAML (``lr=1e-4``, ``model.use_normals=false``, ``sizes=[2048,4096]``).
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence

import yaml

REPO_ROOT = Path(__file__).resolve().parents[4]


class _ConfigLoader(yaml.SafeLoader):
    """SafeLoader that also reads ``1e-4`` / ``5E3`` as floats.

    PyYAML follows YAML 1.1, where a float needs a dot (``1.0e-4``); plain
    ``lr: 1e-4`` would otherwise silently become the *string* "1e-4".
    """


_ConfigLoader.add_implicit_resolver(
    "tag:yaml.org,2002:float",
    re.compile(
        r"""^(?:[-+]?(?:[0-9][0-9_]*)\.[0-9_]*(?:[eE][-+]?[0-9]+)?
        |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
        |\.[0-9_]+(?:[eE][-+][0-9]+)?
        |[-+]?\.(?:inf|Inf|INF)
        |\.(?:nan|NaN|NAN))$""",
        re.X,
    ),
    list("-+0123456789."),
)


def _yaml_load(text: str) -> Any:
    return yaml.load(text, Loader=_ConfigLoader)

DEFAULTS_KEY = "defaults"


def deep_merge(base: Mapping[str, Any], update: Mapping[str, Any]) -> Dict[str, Any]:
    """Recursively merge ``update`` into a copy of ``base`` (dicts merge, everything else replaces)."""
    result = copy.deepcopy(dict(base))
    for key, value in update.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _load_with_defaults(path: Path, stack: Sequence[Path]) -> Dict[str, Any]:
    path = path.resolve()
    if path in stack:
        chain = " -> ".join(str(p) for p in (*stack, path))
        raise ValueError(f"Cyclic config defaults: {chain}")
    raw = _yaml_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"Config root must be a mapping: {path}")
    parents = raw.pop(DEFAULTS_KEY, None) or []
    if isinstance(parents, str):
        parents = [parents]
    merged: Dict[str, Any] = {}
    for parent in parents:
        merged = deep_merge(merged, _load_with_defaults(path.parent / parent, (*stack, path)))
    return deep_merge(merged, raw)


def set_dotted(config: Dict[str, Any], dotted_key: str, value: Any) -> None:
    keys = dotted_key.split(".")
    node = config
    for key in keys[:-1]:
        child = node.get(key)
        if child is None:
            child = node[key] = {}
        elif not isinstance(child, dict):
            raise KeyError(f"Cannot set '{dotted_key}': '{key}' is not a mapping")
        node = child
    node[keys[-1]] = value


def get_dotted(config: Mapping[str, Any], dotted_key: str, default: Any = None) -> Any:
    node: Any = config
    for key in dotted_key.split("."):
        if not isinstance(node, Mapping) or key not in node:
            return default
        node = node[key]
    return node


def apply_overrides(config: Dict[str, Any], overrides: Iterable[str]) -> Dict[str, Any]:
    result = copy.deepcopy(config)
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Override must look like key=value, got: {item!r}")
        key, raw_value = item.split("=", 1)
        set_dotted(result, key.strip(), _yaml_load(raw_value))
    return result


def load_config(path: Path, overrides: Optional[Iterable[str]] = None) -> Dict[str, Any]:
    config = _load_with_defaults(Path(path), ())
    return apply_overrides(config, overrides or [])


def save_config(config: Mapping[str, Any], path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(yaml.safe_dump(_plain(config), sort_keys=False, allow_unicode=True), encoding="utf-8")
    tmp.replace(path)


def config_hash(config: Mapping[str, Any]) -> str:
    """Stable short hash of a config (key order independent)."""
    payload = json.dumps(_plain(config), sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def resolve_path(value: Any, base: Path = REPO_ROOT) -> Optional[Path]:
    """Config paths are relative to the repository root unless absolute."""
    if value is None:
        return None
    path = Path(str(value)).expanduser()
    return path if path.is_absolute() else (base / path)


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    return value
