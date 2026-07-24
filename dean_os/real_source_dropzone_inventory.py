from __future__ import annotations

from pathlib import Path
from typing import Any

from dean_os.artifact_writer import ReviewArtifactWriter
from dean_os.material_loaders import SUPPORTED_EXTENSIONS
from dean_os.schemas import utc_now_iso

DEFAULT_DROPZONE = "docs/research"


class RealSourceDropzoneInventory:
    """Review-only inventory for operator-supplied research files."""

    def __init__(self, output_dir: str | Path = "reports/dean_os/real_source_dropzone_inventory_current"):
        self.output_dir = Path(output_dir)

    def build(
        self,
        dropzone: str | Path = DEFAULT_DROPZONE,
        *,
        recursive: bool = False,
        save: bool = True,
    ) -> dict[str, Any]:
        root = Path(dropzone)
        supported_files, ignored_files, unsupported_files = _scan_dropzone(root, recursive=recursive)
        status = "ready_for_operator_source_review" if supported_files else "empty_dropzone"
        payload = {
            "run_id": _run_id("real_source_dropzone_inventory"),
            "created_at": utc_now_iso(),
            "mode": "real_source_dropzone_inventory",
            "input": {
                "dropzone": str(root),
                "recursive": recursive,
            },
            "summary": {
                "dropzone_status": status,
                "supported_file_count": len(supported_files),
                "ignored_file_count": len(ignored_files),
                "unsupported_file_count": len(unsupported_files),
                "can_build_normalized_packet": bool(supported_files),
                "can_execute_extraction_now": False,
                "can_promote_to_evidence": False,
                "can_write_learning_memory": False,
                "can_trade": False,
            },
            "supported_files": supported_files,
            "ignored_files": ignored_files,
            "unsupported_files": unsupported_files,
            "commands": _commands(supported_files, root),
            "safety_flags": {
                "dropzone_inventory_only": True,
                "file_content_read": False,
                "live_fetch_allowed": False,
                "external_api_call_allowed": False,
                "claim_extraction_execution_allowed_now": False,
                "event_extraction_execution_allowed_now": False,
                "entity_resolution_execution_allowed_now": False,
                "evidence_promotion_allowed": False,
                "learning_write_allowed": False,
                "trade_signal_allowed": False,
                "trading_allowed": False,
            },
            "recommendations": _recommendations(supported_files, unsupported_files),
        }
        if save:
            saved_paths = ReviewArtifactWriter(self.output_dir).write(
                payload=payload,
                markdown=render_real_source_dropzone_inventory_markdown(payload),
                run_id=payload["run_id"],
            )
            payload["saved_paths"] = saved_paths
        return payload


def render_real_source_dropzone_inventory_markdown(payload: dict[str, Any]) -> str:
    summary = payload.get("summary", {})
    lines = [
        "# DEAN-OS Real Source Dropzone Inventory",
        "",
        f"- Run ID: `{payload.get('run_id')}`",
        f"- Dropzone: `{payload.get('input', {}).get('dropzone')}`",
        f"- Dropzone status: `{summary.get('dropzone_status')}`",
        f"- Supported files: {summary.get('supported_file_count')}",
        f"- Ignored files: {summary.get('ignored_file_count')}",
        f"- Unsupported files: {summary.get('unsupported_file_count')}",
        f"- Can build normalized packet: {summary.get('can_build_normalized_packet')}",
        f"- Can trade: {summary.get('can_trade')}",
        "",
        "## Supported Files",
        "",
    ]
    supported_files = payload.get("supported_files", [])
    if supported_files:
        for item in supported_files[:20]:
            lines.append(f"- `{item['path']}` ({item['extension']}, {item['size_bytes']} bytes)")
    else:
        lines.append("- none")
    lines.extend(["", "## Next Commands", ""])
    for item in payload.get("commands", []):
        lines.append(f"- `{item['command_id']}`")
        lines.append(f"```powershell\n{item['command']}\n```")
    lines.extend(["", "## Boundary", ""])
    lines.extend(
        [
            "- This inventory reads file metadata only.",
            "- It does not normalize, extract claims/events/entities, promote evidence, write learning memory, recommend, or trade.",
        ]
    )
    lines.extend(["", "## Recommendations", ""])
    lines.extend(f"- {item}" for item in payload.get("recommendations", []))
    return "\n".join(lines).strip() + "\n"


def _scan_dropzone(root: Path, *, recursive: bool) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    if not root.exists():
        return [], [], [{"path": str(root), "reason": "dropzone_missing"}]
    pattern = "**/*" if recursive else "*"
    supported_files: list[dict[str, Any]] = []
    ignored_files: list[dict[str, Any]] = []
    unsupported_files: list[dict[str, Any]] = []
    for path in sorted(root.glob(pattern)):
        if not path.is_file():
            continue
        if _is_ignored(path):
            ignored_files.append(_file_record(path, reason="dropzone_admin_or_hidden_file"))
            continue
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            unsupported_files.append(_file_record(path, reason="unsupported_extension"))
            continue
        supported_files.append(_file_record(path, source_type_hint=_source_type_hint(path)))
    return supported_files, ignored_files, unsupported_files


def _file_record(path: Path, **extra: Any) -> dict[str, Any]:
    stat = path.stat()
    record = {
        "path": str(path),
        "name": path.name,
        "extension": path.suffix.lower(),
        "size_bytes": stat.st_size,
        "last_modified": utc_now_iso_from_timestamp(stat.st_mtime),
    }
    record.update(extra)
    return record


def utc_now_iso_from_timestamp(timestamp: float) -> str:
    from datetime import UTC, datetime

    return datetime.fromtimestamp(timestamp, UTC).isoformat()


def _is_ignored(path: Path) -> bool:
    name = path.name.lower()
    return name.startswith(".") or name in {"readme.md", "readme.txt"}


def _source_type_hint(path: Path) -> str:
    lower = path.name.lower()
    if "10-k" in lower or "10-q" in lower or "filing" in lower:
        return "filing"
    if "transcript" in lower or "earnings-call" in lower:
        return "transcript"
    if "news" in lower:
        return "news"
    if "report" in lower or path.suffix.lower() in {".pdf", ".docx"}:
        return "report"
    return "article"


def _commands(supported_files: list[dict[str, Any]], root: Path) -> list[dict[str, str]]:
    if not supported_files:
        return [
            {
                "command_id": "add_operator_source_file",
                "command": f"Place a supported research file under {root}",
            }
        ]
    first = supported_files[0]
    source_type = first.get("source_type_hint", "article")
    build_command = (
        f"python run_agent_real_source_normalized_packet.py {first['path']} "
        f"--source-type {source_type} "
        "--output-dir reports\\dean_os\\real_source_normalized_packet_current"
    )
    validation_command = (
        "python run_review_only_real_source_normalized_packet_validation_gate.py "
        "--input-json reports\\dean_os\\real_source_normalized_packet_current\\latest.json "
        "--output-dir reports\\dean_os\\real_source_normalized_packet_validation_gate_current"
    )
    return [
        {"command_id": "build_first_supported_normalized_packet", "command": build_command},
        {"command_id": "validate_normalized_packet", "command": validation_command},
    ]


def _recommendations(supported_files: list[dict[str, Any]], unsupported_files: list[dict[str, Any]]) -> list[str]:
    if not supported_files:
        return ["Add one operator-supplied research file before running real-source normalization."]
    recommendations = ["Review the first supported file's normalized packet before adding more source volume."]
    if unsupported_files:
        recommendations.append("Move unsupported files out of the dropzone or convert them to a supported format.")
    return recommendations


def _run_id(prefix: str) -> str:
    return f"{prefix}_{utc_now_iso().replace(':', '').replace('+', 'Z')}"
