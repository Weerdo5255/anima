"""Save and load agent state without pickle.

State dicts are nested Python structures with NumPy arrays inside.  Arrays
go into a compressed ``.npz`` archive; everything else is stored as JSON in
the same archive.  Loading never executes code, so a brain file is safe to
share.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

_ARRAY = "__ndarray__"


def _flatten(obj: Any, arrays: dict[str, np.ndarray], path: str) -> Any:
    if isinstance(obj, np.ndarray):
        key = f"a{len(arrays)}"
        arrays[key] = obj
        return {_ARRAY: key}
    if isinstance(obj, dict):
        return {str(k): _flatten(v, arrays, f"{path}/{k}") for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_flatten(v, arrays, f"{path}/{i}") for i, v in enumerate(obj)]
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def _unflatten(obj: Any, arrays: Any) -> Any:
    if isinstance(obj, dict):
        if set(obj) == {_ARRAY}:
            return np.array(arrays[obj[_ARRAY]])
        return {k: _unflatten(v, arrays) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_unflatten(v, arrays) for v in obj]
    return obj


def save_state(state: dict[str, Any], path: str | Path) -> Path:
    path = Path(path)
    if path.suffix != ".npz":
        path = path.with_suffix(path.suffix + ".npz") if path.suffix else path.with_suffix(".npz")
    arrays: dict[str, np.ndarray] = {}
    meta = _flatten(state, arrays, "")
    tmp = path.with_name(path.name + ".tmp.npz")
    np.savez_compressed(tmp, __meta__=np.array(json.dumps(meta)), **arrays)
    tmp.replace(path)  # atomic: a crash mid-save never corrupts the old brain
    return path


def load_state(path: str | Path) -> dict[str, Any]:
    with np.load(Path(path), allow_pickle=False) as data:
        meta = json.loads(str(data["__meta__"]))
        return _unflatten(meta, data)
