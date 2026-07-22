from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from dean_os.draft.dean_os_agent_system_v7.dean_os.analyst_learning_promotion_bridge import AnalystLearningPromotionBridge
from dean_os.schemas import AgentLearningRecord, ReviewActionRecord, utc_now_iso
from dean_os.utils import json_ready

DEFAULT_BRIDGE_DRY_RUN_PATH = "reports/dean_os/analyst_learning_bridge/latest.json"


class AnalystLearningApplyCeremony:
    """Explicit learning-record write gate after a reviewed bridge dry-run."""

    def __init__(self, output_dir: str | Path = "reports/dean_os/analyst_learning_apply_ceremony"):
        self.output_dir = Path(output_dir)

    def apply(
        self,
        bridge_dry_run_path: str | Path = DEFAULT_BRIDGE_DRY_RUN_PATH,
        learning_path: str | Path | None = None,
        review_actions_path: str | Path | None = None,
        operations_path: str | Path | None = None,
        apply_learning: bool = False,
        save: bool = True,
    ) -> dict[str, Any]:
        bridge = _load_json(Path(bridge_dry_run_path))
        resolved = _resolved_paths(
            bridge=bridge,
            learning_path=learning_path,
            review_actions_path=review_actions_path,
            operations_path=operations_path,
        )
        validation = _validate_apply_request(bridge, resolved, apply_learning)
        apply_report = None
        promoted_records: list[dict[str, Any]] = []
        if validation["can_apply"]:
            apply_report = self._run_bridge_apply(bridge, resolved)
            promoted_records = apply_report.get("promoted_records", [])
            validation = _validate_apply_result(validation, bridge, apply_report)
        payload = {
            "run_id": _run_id("analyst_learning_apply_ceremony"),
            "created_at": utc_now_iso(),
            "mode": "analyst_learning_apply_ceremony",
            "inputs": {
                "bridge_dry_run_path": str(bridge_dry_run_path),
                "learning_path": str(resolved["learning_path"]),
                "review_actions_path": str(resolved["review_actions_path"]),
                "operations_path": str(resolved["operations_path"]),
                "apply_learning": apply_learning,
            },
            "summary": {
                "apply_status": validation["status"],
                "can_apply": validation["can_apply"],
                "source_count": len(bridge.get("sources", [])),
                "candidate_count": bridge.get("promotion_gate", {}).get("candidate_count", 0),
                "promotable_count": bridge.get("promotion_gate", {}).get("promotable_count", 0),
                "blocked_count": bridge.get("promotion_gate", {}).get("blocked_count", 0),
                "learning_write_performed": bool(promoted_records),
                "promoted_count": len(promoted_records),
                "review_action_write_performed": False,
                "proposal_enqueue_performed": False,
                "config_write_performed": False,
                "pipeline_run_performed": False,
                "broker_access_performed": False,
            },
            "validation": validation,
            "bridge_dry_run_summary": bridge.get("promotion_gate", {}),
            "sources": _source_summary(bridge),
            "promoted_records": promoted_records,
            "apply_report_summary": apply_report.get("promotion_gate", {}) if apply_report else {},
            "commands": _commands(resolved),
            "operator_notes": _operator_notes(),
            "recommendations": _recommendations(validation, promoted_records),
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
        rendered_md = render_analyst_learning_apply_ceremony_markdown(payload)
        json_path.write_text(rendered_json, encoding="utf-8")
        latest_json.write_text(rendered_json, encoding="utf-8")
        md_path.write_text(rendered_md, encoding="utf-8")
        latest_md.write_text(rendered_md, encoding="utf-8")
        return json_path, md_path

    def _run_bridge_apply(self, bridge: dict[str, Any], resolved: dict[str, str]) -> dict[str, Any]:
        bridge_output_dir = self.output_dir / "bridge_apply"
        inputs = bridge.get("inputs", {})
        return AnalystLearningPromotionBridge(output_dir=bridge_output_dir).run(
            profile_run_path=inputs.get("profile_run_path"),
            agent_lab_report_path=inputs.get("agent_lab_report_path"),
            learning_path=resolved["learning_path"],
            review_actions_path=resolved["review_actions_path"],
            operations_path=resolved["operations_path"],
            require_review=bool(inputs.get("require_review", True)),
            apply=True,
            allow_weak_notes=bool(inputs.get("allow_weak_notes", False)),
            allow_duplicates=bool(inputs.get("allow_duplicates", False)),
            default_horizon_days=int(inputs.get("default_horizon_days", 365) or 365),
            save=True,
        )


def render_analyst_learning_apply_ceremony_markdown(payload: dict[str, Any]) -> str:
    summary = payload.get("summary", {})
    validation = payload.get("validation", {})
    lines = [
        "# DEAN-OS Analyst Learning Apply Ceremony",
        "",
        f"- Run ID: `{payload.get('run_id')}`",
        f"- Apply status: `{summary.get('apply_status')}`",
        f"- Learning write performed: {summary.get('learning_write_performed')}",
        f"- Promoted records: {summary.get('promoted_count')}",
        f"- Candidates: {summary.get('candidate_count')} | promotable={summary.get('promotable_count')} | blocked={summary.get('blocked_count')}",
        "",
        "## Validation",
        "",
    ]
    lines.extend(f"- {reason}" for reason in validation.get("reasons", []))
    lines.extend(["", "## Sources", ""])
    for source in payload.get("sources", []):
        lines.append(
            f"- `{source.get('source_id')}` profile={source.get('profile')} "
            f"reviewed={source.get('reviewed')} promotable={source.get('promotable_count')}"
        )
    lines.extend(["", "## Commands", ""])
    for key, command in payload.get("commands", {}).items():
        if command:
            lines.append(f"- {key}: `{command}`")
    lines.extend(["", "## Recommendations", ""])
    lines.extend(f"- {item}" for item in payload.get("recommendations", []))
    return "\n".join(lines).strip() + "\n"


def _validate_apply_request(bridge: dict[str, Any], resolved: dict[str, str], apply_learning: bool) -> dict[str, Any]:
    schema_reasons = _schema_reasons(bridge)
    if schema_reasons:
        return _validation("blocked_invalid_bridge_dry_run", False, schema_reasons, {})

    gate = bridge.get("promotion_gate", {})
    if gate.get("status") != "dry_run_ready":
        return _validation(
            "blocked_bridge_not_ready",
            False,
            [f"Bridge dry-run status is {gate.get('status')}, not dry_run_ready."],
            {},
        )
    if int(gate.get("blocked_count", 0) or 0) > 0:
        return _validation("blocked_candidates_present", False, ["Bridge dry-run still has blocked candidates."], {})
    if int(gate.get("promotable_count", 0) or 0) <= 0:
        return _validation("blocked_no_promotable_candidates", False, ["Bridge dry-run has no promotable candidates."], {})

    review_check = _review_actions_check(bridge, Path(resolved["review_actions_path"]))
    if not review_check["ok"]:
        return _validation("blocked_review_actions_not_active", False, review_check["reasons"], review_check)

    duplicate_check = _duplicate_note_check(bridge, Path(resolved["learning_path"]))
    if not duplicate_check["ok"]:
        return _validation("blocked_duplicate_learning_records", False, duplicate_check["reasons"], duplicate_check)

    if not apply_learning:
        return _validation(
            "blocked_apply_flag_required",
            False,
            ["Pass --apply-learning to write pending analyst learning records from this validated dry-run."],
            {"review_actions": review_check, "duplicates": duplicate_check},
        )

    return _validation(
        "ready_to_apply",
        True,
        ["Bridge dry-run is ready, review actions are active, no duplicate note ids were found, and apply flag is explicit."],
        {"review_actions": review_check, "duplicates": duplicate_check},
    )


def _validate_apply_result(validation: dict[str, Any], bridge: dict[str, Any], apply_report: dict[str, Any]) -> dict[str, Any]:
    expected = int(bridge.get("promotion_gate", {}).get("promotable_count", 0) or 0)
    actual = int(apply_report.get("promotion_gate", {}).get("promoted_count", 0) or 0)
    reasons = list(validation.get("reasons", []))
    if apply_report.get("promotion_gate", {}).get("status") != "applied":
        reasons.append(f"Bridge apply returned status {apply_report.get('promotion_gate', {}).get('status')}.")
        return _validation("blocked_apply_result_not_applied", False, reasons, validation.get("details", {}))
    if actual != expected:
        reasons.append(f"Promoted count {actual} did not match dry-run promotable count {expected}.")
        return _validation("applied_with_count_mismatch", False, reasons, validation.get("details", {}))
    reasons.append(f"Promoted {actual} pending learning record(s).")
    return _validation("applied", False, reasons, validation.get("details", {}))


def _schema_reasons(bridge: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    if bridge.get("mode") != "analyst_learning_promotion_bridge":
        reasons.append("Input JSON is not an analyst_learning_promotion_bridge artifact.")
    if bridge.get("inputs", {}).get("apply") is not False:
        reasons.append("Input bridge artifact must be a dry-run with apply=false.")
    if not bridge.get("sources"):
        reasons.append("Bridge dry-run contains no sources.")
    if not bridge.get("inputs", {}).get("profile_run_path") and not bridge.get("inputs", {}).get("agent_lab_report_path"):
        reasons.append("Bridge dry-run is missing profile_run_path or agent_lab_report_path.")
    if not bridge.get("inputs", {}).get("learning_path"):
        reasons.append("Bridge dry-run is missing learning_path.")
    if not bridge.get("inputs", {}).get("review_actions_path"):
        reasons.append("Bridge dry-run is missing review_actions_path.")
    return reasons


def _review_actions_check(bridge: dict[str, Any], review_actions_path: Path) -> dict[str, Any]:
    if not review_actions_path.exists():
        return {"ok": False, "reasons": [f"Review actions store does not exist: {review_actions_path}"], "sources": []}
    actions = _read_review_actions(review_actions_path)
    reasons: list[str] = []
    source_checks: list[dict[str, Any]] = []
    for source in bridge.get("sources", []):
        source_id = source.get("source_id")
        expected_ids = set(source.get("review", {}).get("review_action_ids", []))
        active = [
            action
            for action in actions
            if action.get("source_type") == "agent_lab_report"
            and action.get("source_id") == source_id
            and action.get("status") != "voided"
        ]
        mark_reviewed = [action for action in active if action.get("action_type") == "mark_reviewed"]
        needs_more_data = [action for action in active if action.get("action_type") == "needs_more_data"]
        active_ids = {action.get("action_id") for action in active if action.get("action_id")}
        if not mark_reviewed:
            reasons.append(f"Source {source_id} has no active mark_reviewed action.")
        if needs_more_data:
            reasons.append(f"Source {source_id} still has an active needs_more_data action.")
        if expected_ids and not expected_ids.issubset(active_ids):
            missing = ", ".join(sorted(expected_ids - active_ids))
            reasons.append(f"Source {source_id} is missing expected review action id(s): {missing}.")
        source_checks.append(
            {
                "source_id": source_id,
                "expected_review_action_ids": sorted(expected_ids),
                "active_review_action_ids": sorted(active_ids),
                "mark_reviewed_count": len(mark_reviewed),
                "needs_more_data_count": len(needs_more_data),
            }
        )
    return {"ok": not reasons, "reasons": reasons, "sources": source_checks}


def _duplicate_note_check(bridge: dict[str, Any], learning_path: Path) -> dict[str, Any]:
    candidate_note_ids = [
        candidate.get("note_id")
        for source in bridge.get("sources", [])
        for candidate in source.get("candidates", [])
        if candidate.get("can_promote") and candidate.get("note_id")
    ]
    existing_note_ids = _read_learning_note_ids(learning_path)
    duplicates = sorted(set(candidate_note_ids).intersection(existing_note_ids))
    reasons = [f"Learning store already has promoted note_id(s): {', '.join(duplicates)}."] if duplicates else []
    return {
        "ok": not duplicates,
        "reasons": reasons,
        "candidate_note_ids": sorted(set(candidate_note_ids)),
        "existing_duplicate_note_ids": duplicates,
        "learning_store_exists": learning_path.exists(),
    }


def _read_review_actions(path: Path) -> list[dict[str, Any]]:
    with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute("SELECT payload FROM review_actions ORDER BY rowid").fetchall()
    actions: list[dict[str, Any]] = []
    for row in rows:
        try:
            actions.append(ReviewActionRecord(**json.loads(row["payload"])).model_dump(mode="json"))
        except Exception:
            continue
    return actions


def _read_learning_note_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as conn:
        conn.row_factory = sqlite3.Row
        try:
            rows = conn.execute("SELECT payload FROM learning_records ORDER BY rowid").fetchall()
        except sqlite3.OperationalError:
            return set()
    note_ids: set[str] = set()
    for row in rows:
        try:
            record = AgentLearningRecord(**json.loads(row["payload"]))
        except Exception:
            continue
        note_ids.add(record.note_id)
    return note_ids


def _resolved_paths(
    bridge: dict[str, Any],
    learning_path: str | Path | None,
    review_actions_path: str | Path | None,
    operations_path: str | Path | None,
) -> dict[str, str]:
    inputs = bridge.get("inputs", {})
    return {
        "learning_path": str(learning_path or inputs.get("learning_path") or "data/dean_os/agent_learning.sqlite"),
        "review_actions_path": str(review_actions_path or inputs.get("review_actions_path") or "data/dean_os/review_actions.sqlite"),
        "operations_path": str(operations_path or inputs.get("operations_path") or "data/dean_os/operation_queue.sqlite"),
    }


def _source_summary(bridge: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "source_type": source.get("source_type"),
            "source_id": source.get("source_id"),
            "profile": source.get("profile"),
            "reviewed": source.get("review", {}).get("reviewed"),
            "needs_more_data": source.get("review", {}).get("needs_more_data"),
            "review_action_ids": source.get("review", {}).get("review_action_ids", []),
            "candidate_count": source.get("candidate_count"),
            "promotable_count": source.get("promotable_count"),
            "promoted_count": source.get("promoted_count"),
        }
        for source in bridge.get("sources", [])
    ]


def _commands(resolved: dict[str, str]) -> dict[str, str]:
    return {
        "list_learning_records": f"python run_agent_learning.py list --store {resolved['learning_path']}",
        "next_outcome_check": (
            "python run_agent_analyst_outcome_loop.py "
            f"--learning-store {resolved['learning_path']} --dry-run"
        ),
    }


def _recommendations(validation: dict[str, Any], promoted_records: list[dict[str, Any]]) -> list[str]:
    if validation["status"] == "applied":
        return [
            f"Pending learning now contains {len(promoted_records)} new analyst record(s).",
            "Do not calibrate agent weights until outcomes mature and outcome evaluation passes.",
            "Run outcome evaluation only when the horizon has enough newer price data.",
        ]
    return [
        "No learning records were written.",
        "Resolve validation blockers or pass --apply-learning only after manually accepting the dry-run.",
        "Do not bypass the bridge dry-run with direct learning writes.",
    ]


def _operator_notes() -> list[str]:
    return [
        "This ceremony can write pending analyst learning records and nothing else.",
        "It never records review actions, enqueues proposals, changes config, runs the pipeline, or accesses a broker.",
        "Learning records are not calibration changes; they need later outcome evaluation.",
    ]


def _validation(status: str, can_apply: bool, reasons: list[str], details: dict[str, Any]) -> dict[str, Any]:
    return {"status": status, "can_apply": can_apply, "reasons": reasons, "details": details}


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON artifact must be an object: {path}")
    return payload


def _run_id(prefix: str) -> str:
    return f"{prefix}_{utc_now_iso().replace(':', '').replace('-', '').replace('.', '_')}"
