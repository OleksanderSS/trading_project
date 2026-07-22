from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

from dean_os.schemas import utc_now_iso
from dean_os.utils import json_ready


class ArtifactPathError(ValueError):
    """Raised when an artifact path escapes its trusted output root."""


class ReviewArtifactWriter:
    """Small safe writer for DEAN-OS review artifacts.

    The writer creates run-id scoped JSON/Markdown files and stable latest pointers.
    It is intentionally local-filesystem only and does not write learning memory,
    production config, or broker/execution state.
    """

    def __init__(self, output_dir: str | Path):
        self.output_dir = Path(output_dir)

    def write(
        self,
        *,
        payload: dict[str, Any],
        markdown: str,
        run_id: str | None = None,
    ) -> dict[str, str]:
        safe_dir = self._safe_output_dir()
        safe_dir.mkdir(parents=True, exist_ok=True)

        resolved_run_id = _safe_run_id(run_id or str(payload.get("run_id") or _run_id("review_artifact")))
        json_path = self._safe_child(safe_dir, f"{resolved_run_id}.json")
        md_path = self._safe_child(safe_dir, f"{resolved_run_id}.md")
        latest_json = self._safe_child(safe_dir, "latest.json")
        latest_md = self._safe_child(safe_dir, "latest.md")

        saved_paths = {
            "json": str(json_path),
            "markdown": str(md_path),
            "latest_json": str(latest_json),
            "latest_markdown": str(latest_md),
        }

        enriched_payload = dict(payload)
        enriched_payload.setdefault("run_id", resolved_run_id)
        enriched_payload.setdefault("created_at", utc_now_iso())
        enriched_payload["saved_paths"] = saved_paths
        enriched_payload.setdefault("artifact_safety", {})
        enriched_payload["artifact_safety"].update(
            {
                "review_artifact": True,
                "atomic_write": True,
                "learning_write_performed": False,
                "production_config_write_performed": False,
                "broker_access_performed": False,
                "live_execution_performed": False,
            }
        )

        rendered_json = json.dumps(json_ready(enriched_payload), indent=2, ensure_ascii=False) + "\n"
        self.atomic_write_text(json_path, rendered_json)
        self.atomic_write_text(latest_json, rendered_json)
        self.atomic_write_text(md_path, markdown)
        self.atomic_write_text(latest_md, markdown)

        return saved_paths

    def _safe_output_dir(self) -> Path:
        path = self.output_dir
        if not path.is_absolute():
            path = Path.cwd() / path
        resolved = path.resolve()

        # Keep default behavior permissive enough for local reports while still
        # rejecting traversal patterns.
        if ".." in self.output_dir.parts:
            raise ArtifactPathError(f"Output directory contains traversal: {self.output_dir}")
        return resolved

    def _safe_child(self, base: Path, child_name: str) -> Path:
        if "/" in child_name or "\\" in child_name or ".." in child_name:
            raise ArtifactPathError(f"Unsafe artifact name: {child_name}")
        child = (base / child_name).resolve()
        try:
            child.relative_to(base.resolve())
        except ValueError as exc:
            raise ArtifactPathError(f"Artifact path escapes output dir: {child}") from exc
        return child

    @staticmethod
    def atomic_write_text(path: Path, text: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(text)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp_name, path)
        finally:
            tmp_path = Path(tmp_name)
            if tmp_path.exists():
                tmp_path.unlink()


def _run_id(prefix: str) -> str:
    return f"{prefix}_{utc_now_iso().replace(':', '').replace('+', 'Z')}"


def _safe_run_id(value: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in {"_", "-", "."} else "_" for ch in value)
    return cleaned.strip("._-") or _run_id("review_artifact")
