from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from dean_os.analyst_outcome_evaluation_loop import ANALYST_LEARNING_FLAG
from dean_os.learning import LearningStore
from dean_os.outcome_evaluation import OutcomeEvaluationRunner
from dean_os.schemas import AgentLearningRecord, utc_now_iso
from dean_os.utils import json_ready


class OutcomeReadinessGate:
    """Read-only gate before analyst outcome evaluation."""

    def __init__(self, output_dir: str | Path = "reports/dean_os/outcome_readiness_gate"):
        self.output_dir = Path(output_dir)

    def build(
        self,
        learning_path: str | Path = "data/dean_os/agent_learning.sqlite",
        memory_path: str | Path = "data/dean_os/recommendation_memory.sqlite",
        market_data_path: str | Path | None = None,
        latest_processed_prices: str | None = None,
        tickers: list[str] | None = None,
        as_of: str | None = None,
        close_col: str = "close",
        datetime_col: str = "datetime",
        neutral_band: float = 0.01,
        limit: int | None = None,
        profile: str | None = None,
        agent_names: list[str] | None = None,
        include_non_analyst_records: bool = False,
        historical_diagnostic: bool = False,
        save: bool = True,
    ) -> dict[str, Any]:
        records = _pending_records(
            learning_path=learning_path,
            profile=profile,
            agent_names=agent_names or [],
            include_non_analyst_records=include_non_analyst_records,
            limit=limit,
        )
        evaluation = _dry_run_evaluation(
            learning_path=learning_path,
            market_data_path=market_data_path,
            latest_processed_prices=latest_processed_prices,
            tickers=tickers or [],
            as_of=as_of,
            close_col=close_col,
            datetime_col=datetime_col,
            neutral_band=neutral_band,
            limit=limit,
            profile=profile,
            agent_names=agent_names or [],
            include_non_analyst_records=include_non_analyst_records,
            historical_diagnostic=historical_diagnostic,
        )
        gate = _readiness_gate(records, evaluation, historical_diagnostic)
        payload = {
            "run_id": _run_id("outcome_readiness_gate"),
            "created_at": utc_now_iso(),
            "mode": "outcome_readiness_gate",
            "inputs": {
                "learning_path": str(learning_path),
                "memory_path": str(memory_path),
                "market_data_path": str(market_data_path) if market_data_path else None,
                "latest_processed_prices": latest_processed_prices,
                "tickers": [ticker.upper() for ticker in tickers or []],
                "as_of": as_of,
                "close_col": close_col,
                "datetime_col": datetime_col,
                "neutral_band": neutral_band,
                "limit": limit,
                "profile": profile,
                "agent_names": agent_names or [],
                "include_non_analyst_records": include_non_analyst_records,
                "historical_diagnostic": historical_diagnostic,
            },
            "summary": {
                "readiness_status": gate["status"],
                "can_run_outcome_dry_run": gate["can_run_outcome_dry_run"],
                "can_apply_outcomes": False,
                "pending_record_count": len(records),
                "evaluable_count": evaluation.get("evaluable_count", 0),
                "updated_count": 0,
                "status_counts": evaluation.get("status_counts", {}),
                "outcome_write_performed": False,
                "learning_write_performed": False,
                "review_action_write_performed": False,
                "proposal_enqueue_performed": False,
                "config_write_performed": False,
                "pipeline_run_performed": False,
                "broker_access_performed": False,
            },
            "readiness_gate": gate,
            "pending_records": _record_summaries(records),
            "profile_readiness": _profile_readiness(records, evaluation),
            "dry_run_outcome_evaluation": evaluation,
            "commands": _commands(
                learning_path=learning_path,
                memory_path=memory_path,
                market_data_path=market_data_path,
                latest_processed_prices=latest_processed_prices,
                tickers=tickers or [],
                as_of=as_of,
                close_col=close_col,
                datetime_col=datetime_col,
                neutral_band=neutral_band,
                profile=profile,
                agent_names=agent_names or [],
                include_non_analyst_records=include_non_analyst_records,
                historical_diagnostic=historical_diagnostic,
                gate=gate,
            ),
            "operator_notes": _operator_notes(),
            "recommendations": _recommendations(gate, evaluation),
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
        rendered_md = render_outcome_readiness_gate_markdown(payload)
        json_path.write_text(rendered_json, encoding="utf-8")
        latest_json.write_text(rendered_json, encoding="utf-8")
        md_path.write_text(rendered_md, encoding="utf-8")
        latest_md.write_text(rendered_md, encoding="utf-8")
        return json_path, md_path


def render_outcome_readiness_gate_markdown(payload: dict[str, Any]) -> str:
    summary = payload.get("summary", {})
    gate = payload.get("readiness_gate", {})
    lines = [
        "# DEAN-OS Outcome Readiness Gate",
        "",
        f"- Run ID: `{payload.get('run_id')}`",
        f"- Status: `{summary.get('readiness_status')}`",
        f"- Pending records: {summary.get('pending_record_count')}",
        f"- Evaluable: {summary.get('evaluable_count')}",
        f"- Status counts: `{summary.get('status_counts', {})}`",
        f"- Can run outcome dry-run: {summary.get('can_run_outcome_dry_run')}",
        "",
        "## Reasons",
        "",
    ]
    lines.extend(f"- {reason}" for reason in gate.get("reasons", []))
    lines.extend(["", "## Profile Readiness", ""])
    for profile, profile_summary in payload.get("profile_readiness", {}).items():
        lines.append(
            f"- `{profile}` pending={profile_summary.get('pending_count')} "
            f"evaluable={profile_summary.get('evaluable_count')} statuses={profile_summary.get('status_counts')}"
        )
    lines.extend(["", "## Commands", ""])
    for key, command in payload.get("commands", {}).items():
        if command:
            lines.append(f"- {key}: `{command}`")
    lines.extend(["", "## Recommendations", ""])
    lines.extend(f"- {item}" for item in payload.get("recommendations", []))
    return "\n".join(lines).strip() + "\n"


def _pending_records(
    learning_path: str | Path,
    profile: str | None,
    agent_names: list[str],
    include_non_analyst_records: bool,
    limit: int | None,
) -> list[AgentLearningRecord]:
    store = LearningStore(learning_path)
    records = [record for record in store.list_records() if record.outcome_label is None]
    if not include_non_analyst_records:
        records = [record for record in records if record.metadata.get(ANALYST_LEARNING_FLAG)]
    if profile:
        records = [record for record in records if record.metadata.get("profile") == profile]
    allowed_agents = {agent for agent in agent_names if agent}
    if allowed_agents:
        records = [record for record in records if record.agent_name in allowed_agents]
    return records[:limit] if limit is not None else records


def _dry_run_evaluation(
    learning_path: str | Path,
    market_data_path: str | Path | None,
    latest_processed_prices: str | None,
    tickers: list[str],
    as_of: str | None,
    close_col: str,
    datetime_col: str,
    neutral_band: float,
    limit: int | None,
    profile: str | None,
    agent_names: list[str],
    include_non_analyst_records: bool,
    historical_diagnostic: bool,
) -> dict[str, Any]:
    try:
        return OutcomeEvaluationRunner(learning_path).evaluate(
            market_data_path=market_data_path,
            latest_processed_prices=latest_processed_prices,
            tickers=[ticker.upper() for ticker in tickers],
            as_of=as_of,
            close_col=close_col,
            datetime_col=datetime_col,
            allow_early=historical_diagnostic,
            apply_updates=False,
            neutral_band=neutral_band,
            limit=limit,
            agent_names=agent_names,
            metadata_filters={"profile": profile} if profile else {},
            require_metadata_flag=None if include_non_analyst_records else ANALYST_LEARNING_FLAG,
        )
    except Exception as exc:
        return {
            "status": "evaluation_unavailable",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "pending_record_count": 0,
            "updated_count": 0,
            "evaluable_count": 0,
            "status_counts": {},
            "evaluations": [],
            "recommendations": ["Provide readable local market data before outcome readiness can be checked."],
        }


def _readiness_gate(records: list[AgentLearningRecord], evaluation: dict[str, Any], historical_diagnostic: bool) -> dict[str, Any]:
    reasons: list[str] = []
    if not records:
        return _gate("no_pending_records", False, ["No pending analyst learning records were found."], evaluation)
    if evaluation.get("status") == "evaluation_unavailable":
        return _gate(
            "blocked_missing_market_data",
            False,
            [f"Outcome evaluation dry-run is unavailable: {evaluation.get('error_type')}: {evaluation.get('error')}"],
            evaluation,
        )
    counts = evaluation.get("status_counts", {})
    if evaluation.get("evaluable_count", 0) > 0:
        if historical_diagnostic:
            reasons.append("Historical diagnostic has evaluable records, but should not be applied as production learning truth.")
            return _gate("historical_diagnostic_ready", True, reasons, evaluation)
        reasons.append("At least one pending analyst learning record is evaluable in dry-run mode.")
        return _gate("ready_for_outcome_dry_run", True, reasons, evaluation)
    if counts.get("not_due"):
        reasons.append("Records exist, but their configured horizons have not elapsed.")
        return _gate("waiting_for_horizon", False, reasons, evaluation)
    if counts.get("no_price_after_created_at"):
        reasons.append("Market data ends before the learning records were created.")
        return _gate("blocked_need_newer_prices", False, reasons, evaluation)
    if counts.get("missing_price_window") or counts.get("missing_tickers"):
        reasons.append("Market data or ticker metadata is incomplete for outcome evaluation.")
        return _gate("blocked_missing_inputs", False, reasons, evaluation)
    reasons.append("No evaluable outcome records were found.")
    return _gate("no_evaluable_records", False, reasons, evaluation)


def _gate(status: str, can_run: bool, reasons: list[str], evaluation: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": status,
        "can_run_outcome_dry_run": can_run,
        "can_apply_outcomes": False,
        "reasons": reasons,
        "evaluable_count": evaluation.get("evaluable_count", 0),
        "status_counts": evaluation.get("status_counts", {}),
    }


def _profile_readiness(records: list[AgentLearningRecord], evaluation: dict[str, Any]) -> dict[str, dict[str, Any]]:
    by_profile: dict[str, list[AgentLearningRecord]] = defaultdict(list)
    for record in records:
        by_profile[str(record.metadata.get("profile") or record.agent_name or "unknown")].append(record)
    eval_by_record = {item.get("record_id"): item for item in evaluation.get("evaluations", [])}
    result: dict[str, dict[str, Any]] = {}
    for profile, items in sorted(by_profile.items()):
        statuses = Counter(eval_by_record.get(record.record_id, {}).get("status", "not_checked") for record in items)
        result[profile] = {
            "pending_count": len(items),
            "evaluable_count": statuses.get("evaluable", 0),
            "status_counts": dict(sorted(statuses.items())),
            "record_ids": [record.record_id for record in items],
        }
    return result


def _record_summaries(records: list[AgentLearningRecord]) -> list[dict[str, Any]]:
    return [
        {
            "record_id": record.record_id,
            "agent_name": record.agent_name,
            "note_id": record.note_id,
            "expected_direction": record.expected_direction,
            "horizon_days": record.horizon_days,
            "created_at": record.created_at,
            "profile": record.metadata.get("profile"),
            "tickers": record.metadata.get("tickers", []),
            "context_tags": record.metadata.get("context_tags", []),
        }
        for record in records
    ]


def _commands(
    learning_path: str | Path,
    memory_path: str | Path,
    market_data_path: str | Path | None,
    latest_processed_prices: str | None,
    tickers: list[str],
    as_of: str | None,
    close_col: str,
    datetime_col: str,
    neutral_band: float,
    profile: str | None,
    agent_names: list[str],
    include_non_analyst_records: bool,
    historical_diagnostic: bool,
    gate: dict[str, Any],
) -> dict[str, str | None]:
    base = [
        "python run_agent_analyst_outcome_loop.py",
        "--learning-store",
        str(learning_path),
        "--memory-store",
        str(memory_path),
    ]
    if market_data_path:
        base.extend(["--market-data-path", str(market_data_path)])
    if latest_processed_prices:
        base.extend(["--latest-processed-prices", latest_processed_prices])
    if tickers:
        base.extend(["--tickers", *[ticker.upper() for ticker in tickers]])
    if as_of:
        base.extend(["--as-of", as_of])
    if close_col != "close":
        base.extend(["--close-col", close_col])
    if datetime_col != "datetime":
        base.extend(["--datetime-col", datetime_col])
    if neutral_band != 0.01:
        base.extend(["--neutral-band", str(neutral_band)])
    if profile:
        base.extend(["--profile", profile])
    if agent_names:
        base.extend(["--agent-names", *agent_names])
    if include_non_analyst_records:
        base.append("--include-non-analyst-records")
    if historical_diagnostic:
        base.append("--historical-diagnostic")
    dry_run = " ".join(base)
    apply_command = f"{dry_run} --apply" if gate.get("status") == "ready_for_outcome_dry_run" else None
    return {
        "outcome_dry_run": dry_run if gate.get("can_run_outcome_dry_run") else None,
        "outcome_apply_after_dry_run_review": apply_command,
        "list_pending_learning_records": f"python run_agent_learning.py --store {learning_path} list",
    }


def _recommendations(gate: dict[str, Any], evaluation: dict[str, Any]) -> list[str]:
    status = gate["status"]
    if status == "ready_for_outcome_dry_run":
        return [
            "Run analyst outcome evaluation in dry-run mode and inspect ticker windows before any apply.",
            "Only apply outcomes if the dry-run uses valid future prices and no diagnostic shortcut.",
            "Do not calibrate weights until outcome evaluation has been applied and reviewed.",
        ]
    if status == "historical_diagnostic_ready":
        return [
            "Use this only as a diagnostic mechanics test, not as production learning truth.",
            "Do not apply diagnostic outcomes unless explicitly accepted as a separate experiment.",
        ]
    if status == "waiting_for_horizon":
        return ["Do not evaluate yet; wait until record horizons have elapsed or run a clearly labeled diagnostic."]
    if status == "blocked_need_newer_prices":
        return ["Load or collect market prices after the learning record creation date before outcome evaluation."]
    if status == "blocked_missing_market_data":
        return evaluation.get("recommendations", []) or ["Provide local market data before outcome readiness can be checked."]
    if status == "no_pending_records":
        return ["Promote reviewed analyst notes into pending learning records before outcome readiness checks."]
    return ["Resolve missing ticker/price inputs, then rerun the readiness gate."]


def _operator_notes() -> list[str]:
    return [
        "This gate is read-only and never updates outcomes.",
        "It never writes learning records, review actions, proposals, config, pipeline outputs, or broker actions.",
        "It decides whether outcome evaluation is meaningful before the outcome loop is run.",
    ]


def _run_id(prefix: str) -> str:
    return f"{prefix}_{utc_now_iso().replace(':', '').replace('-', '').replace('.', '_')}"
