from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from dean_os.schemas import utc_now_iso
from dean_os.utils import json_ready

DEFAULT_REVIEW_ACTION_PATH = "reports/dean_os/review_action_apply_ceremony/latest.json"
DEFAULT_DECISION_PACKET_PATH = "reports/dean_os/review_decision_packet/latest.json"


class EvidenceGapResolutionPlan:
    """Read-only plan for resolving an active needs-more-data review action."""

    def __init__(self, output_dir: str | Path = "reports/dean_os/evidence_gap_resolution_plan"):
        self.output_dir = Path(output_dir)

    def build(
        self,
        review_action_path: str | Path = DEFAULT_REVIEW_ACTION_PATH,
        decision_packet_path: str | Path | None = DEFAULT_DECISION_PACKET_PATH,
        evidence_pack_path: str | Path | None = None,
        source_routing_path: str | Path | None = None,
        min_documents_per_missing_ticker: int = 2,
        min_date_span_days: int = 30,
        suggested_max_rows_per_table: int = 200,
        save: bool = True,
    ) -> dict[str, Any]:
        action_artifact = _load_optional_json(review_action_path)
        review_action = _extract_review_action(action_artifact)
        decision_packet = _load_optional_json(decision_packet_path)
        resolved_evidence_path = _resolve_evidence_pack_path(evidence_pack_path, decision_packet)
        evidence_pack = _load_optional_json(resolved_evidence_path)
        resolved_source_routing_path = _resolve_source_routing_path(source_routing_path, evidence_pack)
        source_routing = _load_optional_json(resolved_source_routing_path)

        coverage = evidence_pack.get("coverage", {}) if evidence_pack else {}
        requested_tickers = _requested_tickers(review_action, decision_packet, evidence_pack)
        missing_tickers = _missing_tickers(requested_tickers, coverage, review_action)
        tasks = _resolution_tasks(
            review_action=review_action,
            decision_packet=decision_packet,
            evidence_pack=evidence_pack,
            source_routing=source_routing,
            requested_tickers=requested_tickers,
            missing_tickers=missing_tickers,
            min_documents_per_missing_ticker=min_documents_per_missing_ticker,
            min_date_span_days=min_date_span_days,
            suggested_max_rows_per_table=suggested_max_rows_per_table,
        )
        validation = _validation(review_action, tasks)
        payload = {
            "run_id": _run_id("evidence_gap_resolution_plan"),
            "created_at": utc_now_iso(),
            "mode": "evidence_gap_resolution_plan",
            "inputs": {
                "review_action_path": str(review_action_path),
                "decision_packet_path": str(decision_packet_path) if decision_packet_path else None,
                "evidence_pack_path": str(resolved_evidence_path) if resolved_evidence_path else None,
                "source_routing_path": str(resolved_source_routing_path) if resolved_source_routing_path else None,
                "min_documents_per_missing_ticker": min_documents_per_missing_ticker,
                "min_date_span_days": min_date_span_days,
                "suggested_max_rows_per_table": suggested_max_rows_per_table,
            },
            "summary": {
                "source_type": review_action.get("source_type"),
                "source_id": review_action.get("source_id"),
                "action_type": review_action.get("action_type"),
                "plan_status": validation["status"],
                "can_resume_learning": False,
                "task_count": len(tasks),
                "missing_ticker_count": len(missing_tickers),
                "missing_tickers": missing_tickers,
                "current_data_quality": coverage.get("data_quality"),
                "current_document_count": int(coverage.get("document_count", 0) or 0),
                "review_action_write_performed": False,
                "learning_write_performed": False,
                "proposal_enqueue_performed": False,
                "config_write_performed": False,
                "pipeline_run_performed": False,
                "broker_access_performed": False,
            },
            "validation": validation,
            "active_review_action": review_action,
            "decision_packet_summary": decision_packet.get("summary", {}) if decision_packet else {},
            "current_coverage": _coverage_summary(coverage),
            "source_routing_summary": _source_routing_summary(source_routing),
            "resolution_tasks": tasks,
            "acceptance_criteria": _acceptance_criteria(requested_tickers, min_documents_per_missing_ticker),
            "commands": _commands(evidence_pack, requested_tickers, resolved_source_routing_path, review_action),
            "operator_notes": _operator_notes(),
            "recommendations": _recommendations(validation, missing_tickers, tasks),
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
        rendered_md = render_evidence_gap_resolution_plan_markdown(payload)
        json_path.write_text(rendered_json, encoding="utf-8")
        latest_json.write_text(rendered_json, encoding="utf-8")
        md_path.write_text(rendered_md, encoding="utf-8")
        latest_md.write_text(rendered_md, encoding="utf-8")
        return json_path, md_path


def render_evidence_gap_resolution_plan_markdown(payload: dict[str, Any]) -> str:
    summary = payload.get("summary", {})
    coverage = payload.get("current_coverage", {})
    lines = [
        "# DEAN-OS Evidence Gap Resolution Plan",
        "",
        f"- Run ID: `{payload.get('run_id')}`",
        f"- Source: `{summary.get('source_type')}:{summary.get('source_id')}`",
        f"- Plan status: `{summary.get('plan_status')}`",
        f"- Current quality: `{summary.get('current_data_quality')}`",
        f"- Missing tickers: {', '.join(summary.get('missing_tickers', [])) or 'none'}",
        f"- Source types: {coverage.get('source_types', {})}",
        f"- Task count: {summary.get('task_count')}",
        "",
        "## Resolution Tasks",
        "",
    ]
    for task in payload.get("resolution_tasks", []):
        lines.append(f"- `{task.get('priority')}` {task.get('task_id')}: {task.get('description')}")
    lines.extend(["", "## Acceptance Criteria", ""])
    lines.extend(f"- {item}" for item in payload.get("acceptance_criteria", []))
    lines.extend(["", "## Commands", ""])
    for key, command in payload.get("commands", {}).items():
        if command:
            lines.append(f"- {key}: `{command}`")
    lines.extend(["", "## Recommendations", ""])
    lines.extend(f"- {item}" for item in payload.get("recommendations", []))
    return "\n".join(lines).strip() + "\n"


def _extract_review_action(artifact: dict[str, Any]) -> dict[str, Any]:
    if not artifact:
        return {}
    if artifact.get("mode") == "review_action_apply_ceremony":
        return artifact.get("recorded_action") or {}
    if artifact.get("action_type") in {"mark_reviewed", "needs_more_data"}:
        return artifact
    if isinstance(artifact.get("action"), dict):
        return artifact["action"]
    return {}


def _validation(review_action: dict[str, Any], tasks: list[dict[str, Any]]) -> dict[str, Any]:
    reasons: list[str] = []
    if not review_action:
        reasons.append("No recorded review action was found.")
        return {"status": "blocked_no_review_action", "can_plan": False, "reasons": reasons}
    if review_action.get("action_type") != "needs_more_data":
        reasons.append(f"Review action is {review_action.get('action_type')}, not needs_more_data.")
        return {"status": "blocked_not_needs_more_data", "can_plan": False, "reasons": reasons}
    if review_action.get("status") not in {None, "recorded", "active"}:
        reasons.append(f"Review action status is {review_action.get('status')}; only active/recorded actions are actionable.")
        return {"status": "blocked_review_action_not_active", "can_plan": False, "reasons": reasons}
    if not tasks:
        reasons.append("No concrete evidence gaps were detected; rebuild the decision packet before changing review state.")
        return {"status": "no_gap_detected_recheck_review", "can_plan": True, "reasons": reasons}
    reasons.append("Active needs-more-data action has concrete source or coverage gaps to resolve.")
    return {"status": "ready_to_collect", "can_plan": True, "reasons": reasons}


def _resolution_tasks(
    review_action: dict[str, Any],
    decision_packet: dict[str, Any],
    evidence_pack: dict[str, Any],
    source_routing: dict[str, Any],
    requested_tickers: list[str],
    missing_tickers: list[str],
    min_documents_per_missing_ticker: int,
    min_date_span_days: int,
    suggested_max_rows_per_table: int,
) -> list[dict[str, Any]]:
    if review_action.get("action_type") != "needs_more_data":
        return []
    coverage = evidence_pack.get("coverage", {}) if evidence_pack else {}
    tasks: list[dict[str, Any]] = []
    current_max_rows = int(evidence_pack.get("inputs", {}).get("max_rows_per_table", 0) or 0) if evidence_pack else 0
    dropped = evidence_pack.get("dropped", []) if evidence_pack else []
    if any(item.get("reason") == "max_rows_per_table" for item in dropped):
        tasks.append(
            _task(
                task_id="increase_table_row_window",
                priority="high",
                description=(
                    "Existing cached tables were truncated before full ticker coverage was checked; rerun evidence pack "
                    f"with at least {max(suggested_max_rows_per_table, current_max_rows)} rows per table."
                ),
                target={"current_max_rows_per_table": current_max_rows, "suggested_max_rows_per_table": suggested_max_rows_per_table},
                evidence=[{"source": "evidence_pack.dropped", "value": dropped}],
            )
        )
    for ticker in missing_tickers:
        tasks.append(
            _task(
                task_id=f"add_{ticker.lower()}_ticker_sources",
                priority="high",
                description=(
                    f"Add at least {min_documents_per_missing_ticker} ticker-specific source items for {ticker}, "
                    "preferably one news/catalyst item and one research/report/filing-style item."
                ),
                target={
                    "ticker": ticker,
                    "minimum_documents": min_documents_per_missing_ticker,
                    "preferred_source_types": ["news", "report", "filing", "transcript"],
                },
                evidence=[{"source": "coverage.missing_requested_tickers", "value": missing_tickers}],
            )
        )
    if coverage.get("data_quality") in {None, "weak", "partial"}:
        tasks.append(
            _task(
                task_id="raise_evidence_quality",
                priority="medium",
                description="Raise evidence quality before review approval by broadening ticker coverage and keeping at least two source types.",
                target={
                    "current_quality": coverage.get("data_quality"),
                    "source_types_present": sorted(coverage.get("by_source_type", {})),
                },
                evidence=[{"source": "decision_packet.review_checks", "value": decision_packet.get("review_checks", [])}],
            )
        )
    if not source_routing:
        tasks.append(
            _task(
                task_id="build_source_routing_snapshot",
                priority="medium",
                description="Run SourceRoutingAgent so materials and collector feeds are mapped before the next evidence-pack build.",
                target={"expected_artifact": "reports/dean_os/source_routing/latest.json"},
                evidence=[{"source": "evidence_pack.source_routing", "value": evidence_pack.get("source_routing", {}) if evidence_pack else {}}],
            )
        )
    date_span = _date_span_days(coverage.get("date_range", {}))
    if date_span is not None and date_span < min_date_span_days:
        tasks.append(
            _task(
                task_id="extend_evidence_date_span",
                priority="medium",
                description=(
                    f"Current evidence date span is about {date_span} day(s); extend the source window toward "
                    f"{min_date_span_days}+ days before treating the thesis as stable."
                ),
                target={"current_span_days": date_span, "minimum_span_days": min_date_span_days},
                evidence=[{"source": "coverage.date_range", "value": coverage.get("date_range", {})}],
            )
        )
    if requested_tickers and not missing_tickers and coverage.get("data_quality") not in {"strong", "clean"}:
        tasks.append(
            _task(
                task_id="rebuild_packet_after_quality_upgrade",
                priority="low",
                description="Ticker coverage appears complete, but quality is still not strong; rebuild the decision packet after source cleanup.",
                target={"requested_tickers": requested_tickers},
                evidence=[{"source": "coverage", "value": _coverage_summary(coverage)}],
            )
        )
    return tasks


def _task(task_id: str, priority: str, description: str, target: dict[str, Any], evidence: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "task_id": task_id,
        "priority": priority,
        "description": description,
        "target": target,
        "evidence": evidence,
        "status": "open",
    }


def _requested_tickers(review_action: dict[str, Any], decision_packet: dict[str, Any], evidence_pack: dict[str, Any]) -> list[str]:
    values: list[str] = []
    values.extend(evidence_pack.get("inputs", {}).get("tickers", []) if evidence_pack else [])
    values.extend(decision_packet.get("evidence_pack", {}).get("tickers", []) if decision_packet else [])
    values.extend(decision_packet.get("evidence_pack", {}).get("missing_requested_tickers", []) if decision_packet else [])
    data_request = str(review_action.get("payload", {}).get("data_request", ""))
    values.extend(_tickers_from_text(data_request))
    return _normalize_tickers(values)


def _missing_tickers(requested_tickers: list[str], coverage: dict[str, Any], review_action: dict[str, Any]) -> list[str]:
    explicit_missing = _normalize_tickers(coverage.get("missing_requested_tickers", []))
    if explicit_missing:
        return explicit_missing
    covered = set(_normalize_tickers(coverage.get("tickers", [])))
    missing = [ticker for ticker in requested_tickers if ticker not in covered]
    if missing:
        return missing
    return [ticker for ticker in _tickers_from_text(str(review_action.get("payload", {}).get("data_request", ""))) if ticker not in covered]


def _commands(
    evidence_pack: dict[str, Any],
    requested_tickers: list[str],
    source_routing_path: str | Path | None,
    review_action: dict[str, Any],
) -> dict[str, str | None]:
    inputs = evidence_pack.get("inputs", {}) if evidence_pack else {}
    tickers = requested_tickers or inputs.get("tickers", [])
    max_rows = max(int(inputs.get("max_rows_per_table", 0) or 0), 200)
    evidence_parts = ["python run_agent_analyst_evidence_pack.py"]
    evidence_parts.extend(_flag_list("--materials", inputs.get("materials_paths", [])))
    evidence_parts.extend(_flag_list("--news-data", inputs.get("news_data_paths", [])))
    evidence_parts.extend(_flag_list("--macro-data", inputs.get("macro_data_paths", [])))
    evidence_parts.extend(_flag_list("--tickers", tickers))
    evidence_parts.extend(_flag_list("--sectors", inputs.get("sectors", [])))
    evidence_parts.extend(_flag_list("--tags", inputs.get("tags", [])))
    if source_routing_path:
        evidence_parts.extend(["--source-routing-json", _quote(str(source_routing_path))])
    evidence_parts.extend(["--max-rows-per-table", str(max_rows)])
    evidence_parts.extend(["--output-dir", "reports/dean_os/analyst_evidence_pack_refreshed"])
    refreshed_pack = "reports/dean_os/analyst_evidence_pack_refreshed/latest.json"
    refreshed_profiles = "reports/dean_os/analyst_profiles_refreshed"
    refreshed_inbox = "reports/dean_os/analyst_review_inbox_refreshed"
    refreshed_packet = "reports/dean_os/review_decision_packet_refreshed"
    return {
        "source_routing_snapshot": "python run_agent_source_routing.py MATERIALS_PATH --collector-inventory reports/dean_os/collector_inventory/latest.json",
        "rebuild_evidence_pack_after_sources_added": " ".join(evidence_parts),
        "rerun_profiles_after_pack": (
            f"python run_agent_analyst_profiles.py {refreshed_pack} "
            f"--tickers {' '.join(tickers) if tickers else 'TICKER_HERE'} "
            f"--output-dir {refreshed_profiles}"
        ),
        "rebuild_review_inbox_after_profiles": (
            f"python run_agent_analyst_review_inbox.py --profile-run-json {refreshed_profiles}/latest.json "
            "--review-actions-store data/dean_os/review_actions.sqlite "
            f"--output-dir {refreshed_inbox}"
        ),
        "rebuild_decision_packet_after_inbox": (
            f"python run_agent_review_decision_packet.py --inbox-json {refreshed_inbox}/latest.json "
            f"--output-dir {refreshed_packet}"
        ),
    }


def _flag_list(flag: str, values: list[Any]) -> list[str]:
    cleaned = [str(value) for value in values if str(value).strip()]
    if not cleaned:
        return []
    return [flag, *[_quote(value) for value in cleaned]]


def _quote(value: str) -> str:
    if re.search(r"\s", value):
        return f'"{value.replace(chr(34), chr(39))}"'
    return value


def _coverage_summary(coverage: dict[str, Any]) -> dict[str, Any]:
    return {
        "document_count": int(coverage.get("document_count", 0) or 0),
        "data_quality": coverage.get("data_quality"),
        "source_types": coverage.get("by_source_type", {}),
        "tickers": coverage.get("tickers", []),
        "missing_requested_tickers": coverage.get("missing_requested_tickers", []),
        "date_range": coverage.get("date_range", {}),
        "warning_count": int(coverage.get("warning_count", 0) or 0),
        "dropped_count": int(coverage.get("dropped_count", 0) or 0),
    }


def _source_routing_summary(source_routing: dict[str, Any]) -> dict[str, Any]:
    routing = source_routing.get("source_routing", source_routing) if source_routing else {}
    return {
        "available": bool(routing),
        "summary": routing.get("summary", {}) if routing else {},
        "analyst_inputs": routing.get("analyst_inputs", {}) if routing else {},
        "recommendations": routing.get("recommendations", []) if routing else [],
        "warnings": routing.get("warnings", []) if routing else [],
    }


def _acceptance_criteria(requested_tickers: list[str], min_documents_per_missing_ticker: int) -> list[str]:
    tickers_text = ", ".join(requested_tickers) if requested_tickers else "the requested ticker set"
    return [
        f"Evidence pack coverage should include {tickers_text} with no missing_requested_tickers.",
        f"Each previously missing ticker should have at least {min_documents_per_missing_ticker} ticker-specific source items.",
        "Decision packet should be rebuilt and should not contain material ticker-coverage warnings.",
        "Any existing needs-more-data action should remain active until the improved packet is manually reviewed or the action is voided.",
        "Learning bridge must be rerun in dry-run mode before any learning apply.",
    ]


def _recommendations(validation: dict[str, Any], missing_tickers: list[str], tasks: list[dict[str, Any]]) -> list[str]:
    if not validation.get("can_plan"):
        return ["Do not collect or apply from this artifact until the review action input is corrected."]
    if missing_tickers:
        return [
            "Fix ticker coverage before approving this analyst source.",
            "Prefer rebuilding the evidence pack with a larger cached-table row window before adding manual materials.",
            "After sources are added, rebuild profiles, inbox, and decision packet instead of voiding review warnings by hand.",
        ]
    if tasks:
        return [
            "Resolve the open tasks, then rebuild the evidence and review artifacts.",
            "Do not resume learning until the decision packet is clean enough for manual approval.",
        ]
    return ["No concrete gap was detected; rebuild the decision packet and inspect whether the needs-more-data action should be voided."]


def _operator_notes() -> list[str]:
    return [
        "This plan is read-only and never fetches data.",
        "It does not record review actions, write learning records, enqueue proposals, change config, run the pipeline, or access a broker.",
        "It turns needs-more-data into source acquisition tasks and follow-up rebuild commands.",
    ]


def _resolve_evidence_pack_path(explicit_path: str | Path | None, decision_packet: dict[str, Any]) -> str | Path | None:
    if explicit_path:
        return explicit_path
    return decision_packet.get("source", {}).get("evidence_pack_path") if decision_packet else None


def _resolve_source_routing_path(explicit_path: str | Path | None, evidence_pack: dict[str, Any]) -> str | Path | None:
    if explicit_path:
        return explicit_path
    return evidence_pack.get("inputs", {}).get("source_routing_path") if evidence_pack else None


def _date_span_days(date_range: dict[str, Any]) -> int | None:
    start = _parse_datetime(date_range.get("start"))
    end = _parse_datetime(date_range.get("end"))
    if not start or not end:
        return None
    return max((end - start).days, 0)


def _parse_datetime(value: Any) -> datetime | None:
    if not value:
        return None
    if isinstance(value, datetime):
        return value.astimezone(UTC) if value.tzinfo else value.replace(tzinfo=UTC)
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed.astimezone(UTC) if parsed.tzinfo else parsed.replace(tzinfo=UTC)


def _tickers_from_text(text: str) -> list[str]:
    return _normalize_tickers(re.findall(r"\b[A-Z]{2,5}\b", text or ""))


def _normalize_tickers(values: list[Any]) -> list[str]:
    return sorted({str(value).strip().upper() for value in values if str(value).strip()})


def _load_optional_json(path: str | Path | None) -> dict[str, Any]:
    if not path:
        return {}
    resolved = Path(path)
    if not resolved.exists():
        return {}
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _run_id(prefix: str) -> str:
    return f"{prefix}_{utc_now_iso().replace(':', '').replace('-', '').replace('.', '_')}"
