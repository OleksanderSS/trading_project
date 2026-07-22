from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Literal

from dean_os.schemas import utc_now_iso
from dean_os.utils import json_ready

ReviewIntent = Literal["mark_reviewed", "needs_more_data"]
DEFAULT_PACKET_PATH = "reports/dean_os/review_decision_packet/latest.json"


class ReviewActionDryRun:
    """Read-only preview for recording a review action from a decision packet."""

    def __init__(self, output_dir: str | Path = "reports/dean_os/review_action_dry_run"):
        self.output_dir = Path(output_dir)

    def build(
        self,
        packet_path: str | Path = DEFAULT_PACKET_PATH,
        intent: ReviewIntent = "needs_more_data",
        reviewer: str = "human",
        review_notes: str = "",
        data_request: str = "Add stronger citations or missing source coverage before learning promotion.",
        acknowledge_warnings: bool = False,
        save: bool = True,
    ) -> dict[str, Any]:
        packet = _load_json(Path(packet_path))
        validation = _validate_intent(packet, intent, acknowledge_warnings)
        command_context = _command_context(packet)
        review_action_preview = _review_action_preview(
            packet=packet,
            intent=intent,
            reviewer=reviewer,
            review_notes=review_notes,
            data_request=data_request,
        )
        commands = _commands(
            intent=intent,
            command_context=command_context,
            review_notes=review_notes,
            data_request=data_request,
        )
        payload = {
            "run_id": _run_id("review_action_dry_run"),
            "created_at": utc_now_iso(),
            "mode": "review_action_dry_run",
            "inputs": {
                "packet_path": str(packet_path),
                "intent": intent,
                "reviewer": reviewer,
                "review_notes": review_notes,
                "data_request": data_request,
                "acknowledge_warnings": acknowledge_warnings,
            },
            "summary": {
                "source_id": packet.get("summary", {}).get("source_id"),
                "profile": packet.get("summary", {}).get("profile"),
                "packet_status": packet.get("summary", {}).get("packet_status"),
                "intent": intent,
                "dry_run_status": validation["status"],
                "can_record_review_action": validation["can_record"],
                "review_action_write_performed": False,
                "learning_write_performed": False,
                "proposal_enqueue_performed": False,
                "config_write_performed": False,
                "pipeline_run_performed": False,
                "broker_access_performed": False,
            },
            "validation": validation,
            "would_record_review_action": review_action_preview,
            "commands": commands,
            "bridge_expectation": _bridge_expectation(intent, validation["can_record"]),
            "operator_notes": _operator_notes(),
            "recommendations": _recommendations(intent, validation),
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
        rendered_md = render_review_action_dry_run_markdown(payload)
        json_path.write_text(rendered_json, encoding="utf-8")
        latest_json.write_text(rendered_json, encoding="utf-8")
        md_path.write_text(rendered_md, encoding="utf-8")
        latest_md.write_text(rendered_md, encoding="utf-8")
        return json_path, md_path


def render_review_action_dry_run_markdown(payload: dict[str, Any]) -> str:
    summary = payload.get("summary", {})
    validation = payload.get("validation", {})
    lines = [
        "# DEAN-OS Review Action Dry Run",
        "",
        f"- Run ID: `{payload.get('run_id')}`",
        f"- Source ID: `{summary.get('source_id')}`",
        f"- Intent: `{summary.get('intent')}`",
        f"- Dry-run status: `{summary.get('dry_run_status')}`",
        f"- Can record review action: {summary.get('can_record_review_action')}",
        f"- Review action write performed: {summary.get('review_action_write_performed')}",
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


def _validate_intent(packet: dict[str, Any], intent: ReviewIntent, acknowledge_warnings: bool) -> dict[str, Any]:
    packet_status = packet.get("summary", {}).get("packet_status")
    checks = packet.get("review_checks", [])
    fail_codes = {check.get("code") for check in checks if check.get("status") == "fail"}
    warning_codes = {check.get("code") for check in checks if check.get("status") == "warn"}
    reasons: list[str] = []

    if "already_reviewed" in fail_codes:
        return _validation("blocked_already_reviewed", False, ["Source already has a mark-reviewed action."], warning_codes)
    if intent == "needs_more_data" and "open_needs_more_data" in fail_codes:
        return _validation("blocked_open_needs_more_data", False, ["Source already has an open needs-more-data action."], warning_codes)

    if intent == "mark_reviewed":
        if fail_codes:
            reasons.append(f"Packet has failing checks: {', '.join(sorted(fail_codes))}.")
            return _validation("blocked_packet_failed_checks", False, reasons, warning_codes)
        if packet_status == "reviewable":
            return _validation("allowed", True, ["Packet is reviewable and has no failing checks."], warning_codes)
        if packet_status == "manual_review_with_warnings":
            if acknowledge_warnings:
                reasons.append("Packet has warnings, but acknowledge_warnings=true allows a mark-reviewed dry-run preview.")
                return _validation("allowed_with_warning_ack", True, reasons, warning_codes)
            reasons.append("Packet has warnings; acknowledge warnings before previewing mark-reviewed as recordable.")
            return _validation("blocked_warning_ack_required", False, reasons, warning_codes)
        reasons.append(f"Packet status {packet_status} does not support mark-reviewed.")
        return _validation("blocked_packet_not_reviewable", False, reasons, warning_codes)

    if intent == "needs_more_data":
        if packet_status in {"reviewable", "manual_review_with_warnings", "needs_more_data_recommended"}:
            reasons.append("Needs-more-data can be previewed for this packet without weakening review gates.")
            return _validation("allowed", True, reasons, warning_codes)
        reasons.append(f"Packet status {packet_status} does not support a needs-more-data action.")
        return _validation("blocked_packet_not_actionable", False, reasons, warning_codes)

    return _validation("blocked_unknown_intent", False, [f"Unsupported intent: {intent}"], warning_codes)


def _validation(status: str, can_record: bool, reasons: list[str], warning_codes: set[str]) -> dict[str, Any]:
    return {
        "status": status,
        "can_record": can_record,
        "warning_codes": sorted(code for code in warning_codes if code),
        "reasons": reasons,
    }


def _review_action_preview(
    packet: dict[str, Any],
    intent: ReviewIntent,
    reviewer: str,
    review_notes: str,
    data_request: str,
) -> dict[str, Any]:
    source = packet.get("source", {})
    action_type = "mark_reviewed" if intent == "mark_reviewed" else "needs_more_data"
    payload = {"data_request": data_request} if intent == "needs_more_data" else {}
    return {
        "dry_run": True,
        "action_id": "DRY_RUN_REVIEW_ACTION_ID",
        "source_type": source.get("source_type", "agent_lab_report"),
        "source_id": source.get("source_id") or packet.get("summary", {}).get("source_id"),
        "action_type": action_type,
        "status": "active",
        "reviewer": reviewer,
        "notes": review_notes,
        "payload": payload,
    }


def _command_context(packet: dict[str, Any]) -> dict[str, str | None]:
    source = packet.get("source", {})
    commands = source.get("suggested_commands", {})
    sample = commands.get("mark_reviewed_preview") or commands.get("needs_more_data_preview") or ""
    return {
        "profile_run_json": _extract_arg(sample, "--profile-run-json"),
        "learning_store": _extract_arg(sample, "--learning-store"),
        "review_actions_store": _extract_arg(sample, "--review-actions-store"),
        "operations_store": _extract_arg(sample, "--operations-store"),
        "source_id": source.get("source_id") or packet.get("summary", {}).get("source_id"),
    }


def _commands(
    intent: ReviewIntent,
    command_context: dict[str, str | None],
    review_notes: str,
    data_request: str,
) -> dict[str, str | None]:
    profile = command_context.get("profile_run_json")
    learning = command_context.get("learning_store")
    review = command_context.get("review_actions_store")
    operations = command_context.get("operations_store")
    if not profile or not learning or not review:
        return {
            "review_action_command_preview": None,
            "bridge_dry_run_after_action": None,
            "reason": "Could not derive profile/learning/review store paths from packet suggested commands.",
        }
    base = (
        f"python run_agent_review_approved_learning.py --profile-run-json {profile} "
        f"--learning-store {learning} --review-actions-store {review}"
    )
    if operations:
        base = f"{base} --operations-store {operations}"
    if intent == "mark_reviewed":
        action_command = f'{base} --mark-reviewed --review-notes "{_quote_value(review_notes)}"'
    else:
        action_command = f'{base} --needs-more-data "{_quote_value(data_request)}" --review-notes "{_quote_value(review_notes)}"'
    bridge_command = (
        f"python run_agent_analyst_learning_bridge.py --profile-run-json {profile} "
        f"--learning-store {learning} --review-actions-store {review}"
    )
    return {
        "review_action_command_preview": action_command,
        "bridge_dry_run_after_action": bridge_command,
        "bridge_apply_after_review_only_if_dry_run_passes": f"{bridge_command} --apply",
    }


def _bridge_expectation(intent: ReviewIntent, can_record: bool) -> dict[str, str]:
    if not can_record:
        return {
            "status": "not_applicable",
            "expected": "No bridge change should be expected because the review action is blocked in dry-run.",
        }
    if intent == "mark_reviewed":
        return {
            "status": "would_unlock_review_gate",
            "expected": "After recording mark-reviewed, bridge dry-run may become promotable if no other blockers exist.",
        }
    return {
        "status": "would_keep_learning_blocked",
        "expected": "After recording needs-more-data, bridge dry-run should remain blocked until the data gap is resolved or voided.",
    }


def _recommendations(intent: ReviewIntent, validation: dict[str, Any]) -> list[str]:
    if not validation["can_record"]:
        return [
            "Do not record the selected review action yet.",
            "Resolve the validation reason or choose a safer intent such as needs_more_data.",
            "Rebuild the decision packet after improving evidence or acknowledging warnings.",
        ]
    if intent == "mark_reviewed":
        return [
            "This is only a dry-run preview; no review action was recorded.",
            "If you record mark-reviewed, rerun the learning bridge in dry-run mode before any apply.",
            "Do not treat mark-reviewed as a trade signal or agent-weight promotion.",
        ]
    return [
        "This is only a dry-run preview; no needs-more-data action was recorded.",
        "If recorded, the source should stay blocked until stronger evidence is provided or the action is voided.",
        "Prefer needs-more-data when packet warnings are material.",
    ]


def _operator_notes() -> list[str]:
    return [
        "This report never writes review actions.",
        "Use it to sanity-check the operator intent before running a real review command.",
        "The apply command is shown only as a later gated step, not as an instruction to run now.",
    ]


def _extract_arg(command: str, flag: str) -> str | None:
    match = re.search(rf"{re.escape(flag)}\s+([^\s]+)", command)
    return match.group(1) if match else None


def _quote_value(value: str) -> str:
    return str(value or "").replace('"', "'")


def _load_json(path: Path) -> dict[str, Any]:
    from dean_os.draft.dean_os_agent_system_v7.dean_os.dean_paths import DeanPaths

    payload = DeanPaths.load_json(path)
    return payload


def _run_id(prefix: str) -> str:
    return f"{prefix}_{utc_now_iso().replace(':', '').replace('-', '').replace('.', '_')}"
