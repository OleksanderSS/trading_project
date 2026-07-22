from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any
from uuid import uuid4

from dean_os.schemas import utc_now_iso
from dean_os.utils import json_ready


class EventLog:
    """Append-only JSONL event log for agent runs and review actions."""

    def __init__(self, log_path: str | Path = "logs/dean_os/events.jsonl"):
        self.log_path = Path(log_path)

    def write(
        self,
        event_type: str,
        source: str,
        payload: dict[str, Any] | None = None,
        run_id: str | None = None,
    ) -> str:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        entry = {
            "event_id": uuid4().hex,
            "event_type": event_type,
            "source": source,
            "run_id": run_id,
            "timestamp": utc_now_iso(),
            "payload": json_ready(payload or {}),
        }
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, sort_keys=True, ensure_ascii=True) + "\n")
        return entry["event_id"]

    def read(self, limit: int | None = None, event_type: str | None = None) -> list[dict[str, Any]]:
        if not self.log_path.exists():
            return []
        events = []
        with self.log_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                event = json.loads(line)
                if event_type and event.get("event_type") != event_type:
                    continue
                events.append(event)
        if limit is not None:
            return events[-limit:]
        return events

    def summary(self) -> dict[str, Any]:
        events = self.read()
        event_counts = Counter(event.get("event_type", "unknown") for event in events)
        source_counts = Counter(event.get("source", "unknown") for event in events)
        return {
            "log_path": str(self.log_path),
            "event_count": len(events),
            "event_counts": dict(sorted(event_counts.items())),
            "source_counts": dict(sorted(source_counts.items())),
            "latest_event": events[-1] if events else None,
        }
