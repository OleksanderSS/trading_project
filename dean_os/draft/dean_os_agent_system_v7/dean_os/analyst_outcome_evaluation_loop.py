from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

from dean_os.draft.dean_os_agent_system_v7.dean_os.context_performance import AgentPerformanceByContext
from dean_os.draft.dean_os_agent_system_v7.dean_os.learning import LearningStore
from dean_os.draft.dean_os_agent_system_v7.dean_os.outcome_evaluation import OutcomeEvaluationRunner
from dean_os.schemas import AgentLearningRecord, utc_now_iso
from dean_os.utils import json_ready

ANALYST_LEARNING_FLAG = "analyst_learning_bridge"


class AnalystOutcomeEvaluationLoop:
    """Review-friendly outcome evaluation for promoted analyst theses.

    This loop evaluates only analyst learning records by default. It reports
    outcomes in dry-run mode first, and writes results only when apply=True.
    """

    def __init__(self, output_dir: str | Path = "reports/dean_os/analyst_outcome_evaluation"):
        self.output_dir = Path(output_dir)

    def run(
        self,
        learning_path: str | Path = "data/dean_os/agent_learning.sqlite",
        memory_path: str | Path = "data/dean_os/recommendation_memory.sqlite",
        market_data_path: str | Path | None = None,
        latest_processed_prices: str | None = None,
        tickers: list[str] | None = None,
        as_of: str | None = None,
        close_col: str = "close",
        datetime_col: str = "datetime",
        apply: bool = False,
        allow_early: bool = False,
        historical_diagnostic: bool = False,
        allow_diagnostic_apply: bool = False,
        neutral_band: float = 0.01,
        limit: int | None = None,
        profile: str | None = None,
        agent_names: list[str] | None = None,
        include_non_analyst_records: bool = False,
        save: bool = True,
    ) -> dict[str, Any]:
        if historical_diagnostic and apply and not allow_diagnostic_apply:
            raise ValueError("historical_diagnostic is dry-run by default; pass allow_diagnostic_apply=True to write outcomes.")

        run_id = _run_id("analyst_outcome_evaluation")
        metadata_filters = {"profile": profile} if profile else {}
        result = OutcomeEvaluationRunner(learning_path).evaluate(
            market_data_path=market_data_path,
            latest_processed_prices=latest_processed_prices,
            tickers=[ticker.upper() for ticker in tickers or []],
            as_of=as_of,
            close_col=close_col,
            datetime_col=datetime_col,
            allow_early=allow_early or historical_diagnostic,
            apply_updates=apply,
            neutral_band=neutral_band,
            limit=limit,
            agent_names=agent_names or [],
            metadata_filters=metadata_filters,
            require_metadata_flag=None if include_non_analyst_records else ANALYST_LEARNING_FLAG,
        )
        if apply and result.get("updated_count", 0):
            _attach_evaluation_audit(
                learning_path=learning_path,
                evaluations=result.get("evaluations", []),
                run_id=run_id,
                market_data_path=result.get("market_data_path"),
                as_of=result.get("as_of"),
                historical_diagnostic=historical_diagnostic,
                neutral_band=neutral_band,
            )
        profile_outcomes = _profile_outcomes(learning_path, include_non_analyst_records=include_non_analyst_records)
        context_summary = AgentPerformanceByContext(learning_path, memory_path).build_summary()
        payload = {
            "run_id": run_id,
            "created_at": utc_now_iso(),
            "mode": "analyst_outcome_evaluation_loop",
            "inputs": {
                "learning_path": str(learning_path),
                "memory_path": str(memory_path),
                "market_data_path": str(market_data_path) if market_data_path else None,
                "latest_processed_prices": latest_processed_prices,
                "tickers": [ticker.upper() for ticker in tickers or []],
                "as_of": as_of,
                "close_col": close_col,
                "datetime_col": datetime_col,
                "apply": apply,
                "allow_early": allow_early,
                "historical_diagnostic": historical_diagnostic,
                "allow_diagnostic_apply": allow_diagnostic_apply,
                "neutral_band": neutral_band,
                "limit": limit,
                "profile": profile,
                "agent_names": agent_names or [],
                "include_non_analyst_records": include_non_analyst_records,
            },
            "evaluation_gate": _evaluation_gate(result, apply, historical_diagnostic),
            "outcome_evaluation": result,
            "profile_outcomes": profile_outcomes,
            "context_performance": context_summary,
            "recommendations": _recommendations(result, profile_outcomes, apply, historical_diagnostic),
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
        rendered_md = render_analyst_outcome_evaluation_markdown(payload)
        json_path.write_text(rendered_json, encoding="utf-8")
        latest_json.write_text(rendered_json, encoding="utf-8")
        md_path.write_text(rendered_md, encoding="utf-8")
        latest_md.write_text(rendered_md, encoding="utf-8")
        return json_path, md_path


def render_analyst_outcome_evaluation_markdown(payload: dict[str, Any]) -> str:
    gate = payload.get("evaluation_gate", {})
    result = payload.get("outcome_evaluation", {})
    lines = [
        "# DEAN-OS Analyst Outcome Evaluation Loop",
        "",
        f"- Run ID: `{payload.get('run_id')}`",
        f"- Status: `{gate.get('status')}`",
        f"- Apply: {payload.get('inputs', {}).get('apply')}",
        f"- Pending checked: {result.get('pending_record_count', 0)}",
        f"- Evaluable: {result.get('evaluable_count', 0)}",
        f"- Updated: {result.get('updated_count', 0)}",
        f"- Status counts: `{result.get('status_counts', {})}`",
        "",
        "## Profile Outcomes",
        "",
    ]
    for profile, summary in payload.get("profile_outcomes", {}).items():
        lines.append(
            f"- `{profile}` records={summary.get('record_count')} completed={summary.get('completed_count')} "
            f"pending={summary.get('pending_count')} hit_rate={summary.get('hit_rate')}"
        )
    lines.extend(["", "## Recommendations", ""])
    lines.extend(f"- {item}" for item in payload.get("recommendations", []))
    return "\n".join(lines).strip() + "\n"


def _attach_evaluation_audit(
    learning_path: str | Path,
    evaluations: list[dict[str, Any]],
    run_id: str,
    market_data_path: str | None,
    as_of: str | None,
    historical_diagnostic: bool,
    neutral_band: float,
) -> None:
    store = LearningStore(learning_path)
    for item in evaluations:
        if item.get("status") != "updated":
            continue
        record = store.get_record(item["record_id"])
        if record is None:
            continue
        record.metadata["analyst_outcome_evaluation_loop"] = {
            "run_id": run_id,
            "market_data_path": market_data_path,
            "as_of": as_of,
            "historical_diagnostic": historical_diagnostic,
            "neutral_band": neutral_band,
            "realized_return": item.get("realized_return"),
            "outcome_label": item.get("outcome_label"),
            "target_at": item.get("target_at"),
        }
        store.add_record(record)


def _profile_outcomes(
    learning_path: str | Path,
    include_non_analyst_records: bool,
) -> dict[str, dict[str, Any]]:
    records = LearningStore(learning_path).list_records()
    if not include_non_analyst_records:
        records = [record for record in records if record.metadata.get(ANALYST_LEARNING_FLAG)]
    buckets: dict[str, list[AgentLearningRecord]] = defaultdict(list)
    for record in records:
        profile = str(record.metadata.get("profile") or record.agent_name or "unknown")
        buckets[profile].append(record)
    return {profile: _summarize_records(items) for profile, items in sorted(buckets.items())}


def _summarize_records(records: list[AgentLearningRecord]) -> dict[str, Any]:
    completed = [record for record in records if record.outcome_label is not None]
    pending = len(records) - len(completed)
    counts = Counter(record.outcome_label or "pending" for record in records)
    returns = [record.realized_return for record in completed if record.realized_return is not None]
    completed_count = len(completed)
    hit_count = counts.get("hit", 0)
    miss_count = counts.get("miss", 0)
    return {
        "record_count": len(records),
        "completed_count": completed_count,
        "pending_count": pending,
        "outcome_counts": dict(sorted(counts.items())),
        "hit_rate": hit_count / completed_count if completed_count else None,
        "miss_rate": miss_count / completed_count if completed_count else None,
        "avg_realized_return": mean(returns) if returns else None,
        "recommendation": _profile_recommendation(completed_count, hit_count, miss_count, pending),
    }


def _profile_recommendation(completed_count: int, hit_count: int, miss_count: int, pending_count: int) -> str:
    if not completed_count:
        return "Keep collecting outcomes before changing profile confidence."
    if miss_count > hit_count:
        return "Use as a weak-context warning; require stronger evidence before similar theses."
    if hit_count > miss_count and completed_count >= 3:
        return "Candidate strength signal only; require more out-of-sample evidence before changing defaults."
    if pending_count:
        return "Mixed or early evidence; wait for more pending outcomes."
    return "No strong profile adjustment signal yet."


def _evaluation_gate(
    result: dict[str, Any],
    apply: bool,
    historical_diagnostic: bool,
) -> dict[str, Any]:
    counts = result.get("status_counts", {})
    if result.get("pending_record_count", 0) == 0:
        status = "no_pending_records"
    elif result.get("updated_count", 0):
        status = "applied"
    elif historical_diagnostic and result.get("evaluable_count", 0):
        status = "historical_diagnostic_ready"
    elif result.get("evaluable_count", 0):
        status = "dry_run_ready"
    elif counts.get("not_due"):
        status = "waiting_for_horizon"
    elif counts.get("no_price_after_created_at"):
        status = "blocked_need_newer_prices"
    elif counts.get("missing_price_window") or counts.get("missing_tickers"):
        status = "blocked_missing_inputs"
    else:
        status = "no_updates"
    return {
        "status": status,
        "apply_requested": apply,
        "historical_diagnostic": historical_diagnostic,
        "can_apply": result.get("evaluable_count", 0) > 0 and not historical_diagnostic,
        "updated_count": result.get("updated_count", 0),
        "evaluable_count": result.get("evaluable_count", 0),
        "status_counts": counts,
    }


def _recommendations(
    result: dict[str, Any],
    profile_outcomes: dict[str, dict[str, Any]],
    apply: bool,
    historical_diagnostic: bool,
) -> list[str]:
    gate = _evaluation_gate(result, apply, historical_diagnostic)
    status = gate["status"]
    if status == "no_pending_records":
        return ["No pending analyst learning records were found. Promote reviewed notes before evaluating outcomes."]
    if status == "blocked_need_newer_prices":
        return ["Load market prices after the learning record creation date or wait for the thesis horizon."]
    if status == "waiting_for_horizon":
        return ["Do not apply yet; the configured thesis horizon has not elapsed."]
    if status == "historical_diagnostic_ready":
        return ["Historical diagnostic is evaluable. Treat it as a test of mechanics, not production learning truth."]
    if status == "dry_run_ready":
        return ["Dry-run outcomes are evaluable. Inspect ticker windows, then rerun with --apply if the evaluation is valid."]
    if status == "applied":
        recommendations = ["Outcomes were written. Rebuild context/profile scorecards before changing analyst weights."]
        weak_profiles = [
            profile
            for profile, summary in profile_outcomes.items()
            if (summary.get("miss_rate") or 0.0) > (summary.get("hit_rate") or 0.0)
        ]
        if weak_profiles:
            recommendations.append(f"Weak profile signal detected for: {', '.join(weak_profiles)}.")
        return recommendations
    return result.get("recommendations", []) or ["No outcome action available."]


def _run_id(prefix: str) -> str:
    return f"{prefix}_{utc_now_iso().replace(':', '').replace('-', '').replace('.', '_')}"
