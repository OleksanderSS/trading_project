from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from dean_os.utils import json_ready


def load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_json_or_csv(path: str | Path) -> Any:
    resolved = Path(path)
    if resolved.suffix.lower() == ".json":
        return load_json(resolved)
    if resolved.suffix.lower() == ".csv":
        import pandas as pd

        frame = pd.read_csv(resolved)
        if frame.empty:
            return {}
        return frame.iloc[-1].to_dict()
    raise ValueError(f"Unsupported artifact type: {resolved.suffix}. Use .json or .csv.")


def run_id(prefix: str) -> str:
    stamp = datetime.now(UTC).isoformat().replace(":", "").replace("-", "").replace(".", "_")
    return f"{prefix}_{stamp}"


def write_json(path: str | Path, payload: Any) -> Path:
    resolved = Path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(json.dumps(json_ready(payload), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return resolved


def save_latest_json(output: str | Path | None, output_dir: str | Path, payload: dict[str, Any]) -> dict[str, Any]:
    if output:
        output_path = write_json(output, payload)
        payload.setdefault("saved_paths", {})["json"] = str(output_path)
        return payload

    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    run_path = root / f"{payload.get('run_id', run_id('report'))}.json"
    latest_path = root / "latest.json"
    payload.setdefault("saved_paths", {})
    payload["saved_paths"].update({"json": str(run_path), "latest_json": str(latest_path)})
    rendered = json.dumps(json_ready(payload), indent=2, ensure_ascii=False) + "\n"
    run_path.write_text(rendered, encoding="utf-8")
    latest_path.write_text(rendered, encoding="utf-8")
    return payload


def print_json(payload: Any) -> None:
    print(json.dumps(json_ready(payload), indent=2, ensure_ascii=False))

