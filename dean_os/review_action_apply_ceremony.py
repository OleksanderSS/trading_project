from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from dean_os.review_actions import ReviewActionStore
from dean_os.schemas import ReviewActionRecord, utc_now_iso
from dean_os.utils import json_ready

DEFAULT_DRY_RUN_PATH = "reports/dean_os/review_action_dry_run/latest.json"


class ReviewActionApplyCeremony:
    """Explicit one-action write gate after a review action dry-run."""

    def __init__(self, output_dir: str | Path = "reports/dean_os/review_action_apply_ceremony"):
        self.output_dir = Path(output_dir)

    def apply(
        self,
        dry_run_path: str | Path = DEFAULT_DRY_RUN_PATH,
        review_actions_path: str | Path = "data/dean_os/review_actions.sqlite",
        operations_path: str | Path = "data/dean_os/operation_queue.sqlite",
        event_log_path: str | Path | None = "logs/dean_os/events.jsonl",
        apply_review_action: bool = False,
        save: bool = True,
    ) -> dict[str, Any]:
        dry_run = _load_json(Path(dry_run_path))
        preview = dry_run.get("would_record_review_action", {})
        validation = _validate_apply_request(
            dry_run=dry_run,
            preview=preview,
            review_actions_path=Path(review_actions_path),
            apply_review_action=apply_review_action,
        )
        recorded_action = None
        if validation["can_apply"]:
            recorded_action = self._record_action(
                preview=preview,
                review_actions_path=review_actions_path,
                operations_path=operations_path,
                event_log_path=event_log_path,
            )
        payload = {
            "run_id": _run_id("review_action_apply_ceremony"),
            "created_at": utc_now_iso(),
            "mode": "review_action_apply_ceremony",
            "inputs": {
                "dry_run_path": str(dry_run_path),
                "review_actions_path": str(review_actions_path),
                "operations_path": str(operations_path),
                "event_log_path": str(event_log_path) if event_log_path else None,
                "apply_review_action": apply_review_action,
            },
            "summary": {
                "source_type": preview.get("source_type"),
                "source_id": preview.get("source_id"),
                "intent": dry_run.get("summary", {}).get("intent"),
                "action_type": preview.get("action_type"),
                "apply_status": validation["status"],
                "can_apply": validation["can_apply"],
                "review_action_write_performed": recorded_action is not None,
                "recorded_action_id": recorded_action.get("action_id") if recorded_action else None,
                "learning_write_performed": False,
                "proposal_enqueue_performed": False,
                "config_write_performed": False,
                "pipeline_run_performed": False,
                "broker_access_performed": False,
            },
            "validation": validation,
            "dry_run_summary": dry_run.get("summary", {}),
            "recorded_action": recorded_action,
            "commands": _commands(dry_run, review_actions_path),
            "operator_notes": _operator_notes(),
            "recommendations": _recommendations(validation, preview),
        }
        if save:
            self.save(payload)
        return payload

    def save(self, payload: dict[str, Any]) -> tuple[Path, Path]:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        json_path = self.output_dir / f"{payload['run_id']}.json"
        md_path = self.output_dir / f"{payload['run_id']}.md"
        latest_json = self.output_dir / "latest.json"
        latest_md = self.output_dir / "latest.md"
        payload["saved_paths"] = {
            "json": str(json_path),
            "markdown": str(md_path),
            "latest_json": str(latest_json),
            "latest_markdown": str(latest_md),
        }
        rendered_json = json.dumps(json_ready(payload), indent=2, ensure_ascii=False) + "\n"
        rendered_md = render_review_action_apply_ceremony_markdown(payload)
        json_path.write_text(rendered_json, encoding="utf-8")
        latest_json.write_text(rendered_json, encoding="utf-8")
        md_path.write_text(rendered_md, encoding="utf-8")
        latest_md.write_text(rendered_md, encoding="utf-8")
        return json_path, md_path

    def _record_action(
        self,
        preview: dict[str, Any],
        review_actions_path: str | Path,
        operations_path: str | Path,
        event_log_path: str | Path | None,
    ) -> dict[str, Any]:
        store = ReviewActionStore(
            review_actions_path,
            operations_path=operations_path,
            event_log_path=event_log_path,
        )
        action_type = preview.get("action_type")
        if action_type == "mark_reviewed":
            action = store.mark_reviewed(
                source_type=preview["source_type"],
                source_id=preview["source_id"],
                notes=preview.get("notes", ""),
                reviewer=preview.get("reviewer", "human"),
            )
        elif action_type == "needs_more_data":
            action = store.needs_more_data(
                source_type=preview["source_type"],
                source_id=preview["source_id"],
                data_request=preview.get("payload", {}).get("data_request", ""),
                notes=preview.get("notes", ""),
                reviewer=preview.get("reviewer", "human"),
            )
        else:
            raise ValueError(f"Unsupported review action type: {action_type}")
        return action.model_dump(mode="json")


def render_review_action_apply_ceremony_markdown(payload: dict[str, Any]) -> str:
    summary = payload.get("summary", {})
    validation = payload.get("validation", {})
    lines = [
        "# DEAN-OS Review Action Apply Ceremony",
        "",
        f"- Run ID: `{payload.get('run_id')}`",
        f"- Source: `{summary.get('source_type')}:{summary.get('source_id')}`",
        f"- Action type: `{summary.get('action_type')}`",
        f"- Apply status: `{summary.get('apply_status')}`",
        f"- Review action write performed: {summary.get('review_action_write_performed')}",
        f"- Recorded action ID: `{summary.get('recorded_action_id')}`",
        "",
        "## Validation",
        "",
    ]
    lines.extend(f"- {reason}" for reason in validation.get("reasons", []))
    lines.extend(["", "## Commands", ""])
    for key, command in payload.get("commands", {}).items():
        if command:
            lines.append(f"- {key}: `{command}`")
    lines.extend(["", "## Recommendations", ""])
    lines.extend(f"- {item}" for item in payload.get("recommendations", []))
    return "\n".join(lines).strip() + "\n"


def _validate_apply_request(
    dry_run: dict[str, Any],
    preview: dict[str, Any],
    review_actions_path: Path,
    apply_review_action: bool,
) -> dict[str, Any]:
    base_reasons = _schema_reasons(dry_run, preview)
    if base_reasons:
        return _validation("blocked_invalid_dry_run", False, base_reasons, [])

    dry_run_status = dry_run.get("summary", {}).get("dry_run_status")
    can_record = bool(dry_run.get("summary", {}).get("can_record_review_action"))
    if not can_record:
        return _validation(
            "blocked_dry_run_not_recordable",
            False,
            [f"Dry-run is not recordable: {dry_run_status}."],
            [],
        )

    existing = _existing_active_actions(
        review_actions_path=review_actions_path,
        source_type=preview["source_type"],
        source_id=preview["source_id"],
    )
    duplicate_reasons = _duplicate_reasons(preview["action_type"], existing)
    if duplicate_reasons:
        return _validation("blocked_existing_review_action", False, duplicate_reasons, existing)

    if not apply_review_action:
        return _validation(
            "blocked_apply_flag_required",
            False,
            ["Pass --apply-review-action to record this already validated review action."],
            existing,
        )

    return _validation(
        "applied",
        True,
        ["Dry-run is recordable, no active duplicate action was found, and apply flag is explicit."],
        existing,
    )


def _schema_reasons(dry_run: dict[str, Any], preview: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    if dry_run.get("mode") != "review_action_dry_run":
        reasons.append("Input JSON is not a review_action_dry_run artifact.")
    if preview.get("dry_run") is not True:
        reasons.append("Dry-run artifact does not contain a dry_run preview action.")
    for field in ("source_type", "source_id", "action_type"):
        if not preview.get(field):
            reasons.append(f"Preview action is missing {field}.")
    if preview.get("action_type") not in {"mark_reviewed", "needs_more_data"}:
        reasons.append(f"Unsupported preview action type: {preview.get('action_type')}.")
    if preview.get("action_type") == "needs_more_data" and not preview.get("payload", {}).get("data_request"):
        reasons.append("Needs-more-data action requires a data_request.")
    return reasons


def _existing_active_actions(
    review_actions_path: Path,
    source_type: str,
    source_id: str,
) -> list[dict[str, Any]]:
    if not review_actions_path.exists():
        return []
    with sqlite3.connect(f"file:{review_actions_path}?mode=ro", uri=True) as conn:
        conn.row_factory = sqlite3.Row
        try:
            rows = conn.execute(
                """
                SELECT payload FROM review_actions
                WHERE source_type = ? AND source_id = ? AND status != 'voided'
                ORDER BY rowid
                """,
                (source_type, source_id),
            ).fetchall()
        except sqlite3.OperationalError:
            return []
    actions: list[dict[str, Any]] = []
    for row in rows:
        try:
            actions.append(ReviewActionRecord(**json.loads(row["payload"])).model_dump(mode="json"))
        except Exception:
            actions.append({"payload_error": "could_not_decode_review_action"})
    return actions


def _duplicate_reasons(action_type: str, existing: list[dict[str, Any]]) -> list[str]:
    reasons: list[str] = []
    existing_types = {item.get("action_type") for item in existing}
    if action_type in existing_types:
        reasons.append(f"An active {action_type} action already exists for this source.")
    if action_type == "mark_reviewed" and "needs_more_data" in existing_types:
        reasons.append("An active needs-more-data action exists; resolve or void it before marking reviewed.")
    return reasons


def _validation(status: str, can_apply: bool, reasons: list[str], existing_actions: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "status": status,
        "can_apply": can_apply,
        "reasons": reasons,
        "existing_active_action_count": len(existing_actions),
        "existing_active_actions": existing_actions,
    }


def _commands(dry_run: dict[str, Any], review_actions_path: str | Path) -> dict[str, str | None]:
    source = dry_run.get("would_record_review_action", {})
    source_type = source.get("source_type")
    source_id = source.get("source_id")
    return {
        "list_review_actions": (
            f"python run_agent_review_actions.py --store {review_actions_path} "
            f"list --source-type {source_type}"
            if source_type
            else None
        ),
        "bridge_dry_run_after_action": dry_run.get("commands", {}).get("bridge_dry_run_after_action"),
        "bridge_apply_after_review_only_if_dry_run_passes": dry_run.get("commands", {}).get(
            "bridge_apply_after_review_only_if_dry_run_passes"
        ),
        "source_filter_hint": f"source_id={source_id}" if source_id else None,
    }


def _recommendations(validation: dict[str, Any], preview: dict[str, Any]) -> list[str]:
    if validation["status"] == "applied":
        if preview.get("action_type") == "mark_reviewed":
            return [
                "Rerun the analyst learning bridge in dry-run mode before any learning apply.",
                "Do not treat the review action as a trade signal or calibration approval.",
                "Only apply learning if the bridge dry-run passes with no unresolved blockers.",
            ]
        return [
            "Keep this source blocked until the requested data is added or the action is voided.",
            "Rebuild the evidence pack, profile run, inbox, and decision packet after improving coverage.",
            "Do not apply learning from this source while needs-more-data remains active.",
        ]
    return [
        "No review action was recorded.",
        "Use the validation reason to decide whether to rerun dry-run, void an old action, or improve source data.",
        "Do not proceed to learning promotion from this artifact.",
    ]


def _operator_notes() -> list[str]:
    return [
        "This ceremony can write exactly one review action and nothing else.",
        "It never writes learning records, proposals, config, pipeline outputs, or broker actions.",
        "The learning bridge apply command is shown only as a later gated step.",
    ]


def _load_json(path: Path) -> dict[str, Any]:
    from dean_os.dean_paths import DeanPaths

    payload = DeanPaths.load_json(path)
    return payload


def _run_id(prefix: str) -> str:
    return f"{prefix}_{utc_now_iso().replace(':', '').replace('-', '').replace('.', '_')}"
