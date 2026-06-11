from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel


def json_ready(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_ready(item) for item in value]
    if hasattr(value, "to_dict"):
        try:
            return value.to_dict()
        except Exception:
            return repr(value)
    return value


def sha256_json(value: Any) -> str:
    payload = json.dumps(json_ready(value), sort_keys=True, ensure_ascii=True, default=repr)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))
