from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path
from typing import Any

from dean_os.base import BaseAgent
from dean_os.paper_trading import PaperTradeStore
from dean_os.schemas import MarketContext, PaperTradeRecord, PipelineActionProposal, PipelineReport

DECISION_DIARY_COLUMNS = {
    "agent_id",
    "decision_timestamp",
    "ticker",
    "decision_type",
    "reasoning",
    "market_context",
    "context_fingerprint",
    "outcome",
    "profit_loss",
}

MODELING_DIARY_COLUMNS = {
    "timestamp",
    "ticker",
    "tf",
    "target",
    "model_name",
    "context_fingerprint",
    "is_champion",
}


class DiaryBridgeAgent(BaseAgent):
    """Inspects whether DEAN paper outcomes can be bridged into the pipeline diary."""

    version = "0.1.0"
    branch = "pipeline"

    async def run(self, context: MarketContext) -> PipelineReport:
        bridge = inspect_diary_bridge(
            experience_diary_path=self.config.get("experience_diary_path", "logs/experience_diary.csv"),
            paper_store_path=self.config.get("paper_store_path", "data/dean_os/paper_trades.sqlite"),
            candidate_limit=int(self.config.get("candidate_limit", 20)),
        )
        context.metadata["diary_bridge"] = bridge

        proposal = build_diary_bridge_proposal(bridge)
        if proposal:
            context.action_proposals.append(proposal)

        verdict = _verdict(bridge)
        return PipelineReport(
            agent_name=self.name,
            agent_version=self.version,
            verdict=verdict,
            confidence=0.82,
            data_quality_score=_quality_score(bridge),
            signal_strength=0.0,
            reasons=bridge["reasons"],
            risks=bridge["risks"],
            blind_spots=[
                "DiaryBridgeAgent creates review evidence only; it does not write to the pipeline experience diary."
            ],
            evidence=[
                self.evidence("file", bridge["pipeline_diary"]["path"], "experience_diary_path", bridge["pipeline_diary"]["path"]),
                self.evidence("metric", "diary_bridge", "status", bridge["status"]),
                self.evidence("metric", "diary_bridge", "bridge_candidate_count", bridge["paper_records"]["bridge_candidate_count"]),
                self.evidence("metric", "diary_bridge", "skipped_by_reason", bridge["paper_records"]["skipped_by_reason"]),
                self.evidence("metric", "diary_bridge", "schema_kind", bridge["pipeline_diary"]["schema_kind"]),
            ],
            input_hash=self.context_hash(context),
            metrics_snapshot=bridge,
        )


def inspect_diary_bridge(
    experience_diary_path: str | Path = "logs/experience_diary.csv",
    paper_store_path: str | Path = "data/dean_os/paper_trades.sqlite",
    candidate_limit: int = 20,
) -> dict[str, Any]:
    diary = _read_diary_summary(Path(experience_diary_path))
    records = PaperTradeStore(paper_store_path).list_records()
    candidates: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for record in records:
        candidate, reason = _bridge_candidate(record)
        if candidate:
            candidates.append(candidate)
        else:
            skipped.append(_skip_payload(record, reason))

    skipped_counts = Counter(item["reason"] for item in skipped)
    status, reasons, risks = _classify(diary, candidates)
    return {
        "status": status,
        "pipeline_diary": diary,
        "paper_records": {
            "paper_store_path": str(paper_store_path),
            "record_count": len(records),
            "evaluated_count": sum(1 for record in records if record.status == "evaluated"),
            "bridge_candidate_count": len(candidates),
            "skipped_count": len(skipped),
            "skipped_by_reason": dict(sorted(skipped_counts.items())),
            "bridge_candidates": candidates[:candidate_limit],
            "skipped_examples": skipped[:candidate_limit],
        },
        "reasons": reasons,
        "risks": risks,
        "recommendations": _recommendations(status, diary, candidates),
    }


def build_diary_bridge_proposal(bridge: dict[str, Any]) -> PipelineActionProposal | None:
    status = bridge["status"]
    if status == "schema_mismatch":
        return PipelineActionProposal(
            agent_name="diary_bridge",
            action_type="validate",
            target="pipeline_experience_diary_schema",
            reason="Pipeline diary schema is not compatible with DEAN paper outcome bridge.",
            command_preview="Review logs/experience_diary.csv schema before bridging paper outcomes.",
            expected_effect="Prevent incompatible paper outcomes from corrupting pipeline model-memory records.",
            risks=["No automatic diary write should happen until the target diary schema is confirmed."],
            evidence=[],
        )
    if status == "bridge_proposal_ready":
        return PipelineActionProposal(
            agent_name="diary_bridge",
            action_type="report",
            target="pipeline_experience_diary_bridge",
            reason="Evaluated DEAN paper records can be reviewed for possible diary bridge.",
            command_preview="Review diary_bridge.bridge_candidates; no automatic write is performed.",
            expected_effect="Create a human-review package for mapping DEAN paper outcomes to pipeline diary memory.",
            risks=[
                "Thesis-style paper decisions may not map cleanly to concrete BUY/SELL/HOLD diary records.",
                "Human review is required before any pipeline diary write.",
            ],
            evidence=[],
        )
    return None


def _read_diary_summary(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {
            "path": str(path),
            "available": False,
            "row_count": 0,
            "columns": [],
            "schema_kind": "missing",
            "compatible_decision_schema": False,
            "warnings": ["Pipeline experience diary file does not exist."],
        }
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = list(reader.fieldnames or [])
        rows = list(reader)
    column_set = set(columns)
    compatible = DECISION_DIARY_COLUMNS.issubset(column_set)
    modeling_schema = MODELING_DIARY_COLUMNS.issubset(column_set)
    schema_kind = "decision_outcome" if compatible else "modeling_champion" if modeling_schema else "unknown_csv"
    warnings: list[str] = []
    if modeling_schema and not compatible:
        warnings.append("Diary CSV looks like a modeling/champion log, not a trade outcome diary.")
    if not compatible:
        missing = sorted(DECISION_DIARY_COLUMNS - column_set)
        warnings.append(f"Missing decision-diary columns: {', '.join(missing)}.")
    return {
        "path": str(path),
        "available": True,
        "row_count": len(rows),
        "columns": columns,
        "schema_kind": schema_kind,
        "compatible_decision_schema": compatible,
        "warnings": warnings,
        "ticker_counts": _value_counts(rows, "ticker"),
        "outcome_counts": _value_counts(rows, "outcome"),
        "decision_type_counts": _value_counts(rows, "decision_type"),
        "agent_counts": _value_counts(rows, "agent_id") or _value_counts(rows, "model_name"),
    }


def _bridge_candidate(record: PaperTradeRecord) -> tuple[dict[str, Any] | None, str]:
    if record.status != "evaluated":
        return None, f"status_{record.status}"
    if record.realized_return is None or record.outcome_label is None:
        return None, "missing_outcome"
    if record.action in {"watchlist", "no_trade"} or record.expected_direction == "neutral":
        return None, "not_concrete_trade_decision"
    if len(record.tickers) != 1:
        return None, "requires_single_ticker"

    ticker_result = _first_ticker_result(record)
    return {
        "paper_trade_id": record.trade_id,
        "agent_id": record.agent_name,
        "ticker": record.tickers[0],
        "decision_type": "BUY" if record.expected_direction == "bullish" else "SELL",
        "reasoning": f"DEAN paper decision {record.trade_id}: {record.thesis}",
        "market_context": {
            "source_type": record.source_type,
            "source_id": record.source_id,
            "context_tags": record.context_tags,
            "regime_tags": record.regime_tags,
            "confidence": record.confidence,
            "horizon_days": record.horizon_days,
            "outcome_label": record.outcome_label,
        },
        "context_fingerprint": record.metadata.get("context_fingerprint")
        or "|".join([*record.context_tags, *record.regime_tags])
        or "dean_os_unknown_context",
        "model_prediction": record.realized_return,
        "model_confidence": record.confidence,
        "entry_price": ticker_result.get("start_price"),
        "exit_price": ticker_result.get("end_price"),
        "outcome": _diary_outcome(record.outcome_label),
        "profit_loss": record.realized_return,
        "decision_timestamp": record.created_at,
        "outcome_at": record.outcome_at,
    }, "bridge_candidate"


def _first_ticker_result(record: PaperTradeRecord) -> dict[str, Any]:
    results = record.metadata.get("ticker_results", [])
    if isinstance(results, list):
        for item in results:
            if isinstance(item, dict) and item.get("status") == "ok":
                return item
    return {}


def _skip_payload(record: PaperTradeRecord, reason: str) -> dict[str, Any]:
    return {
        "paper_trade_id": record.trade_id,
        "status": record.status,
        "action": record.action,
        "expected_direction": record.expected_direction,
        "tickers": record.tickers,
        "reason": reason,
    }


def _diary_outcome(outcome_label: str) -> str:
    return {
        "hit": "profitable",
        "miss": "unprofitable",
        "inconclusive": "break_even",
    }.get(outcome_label, "pending")


def _classify(diary: dict[str, Any], candidates: list[dict[str, Any]]) -> tuple[str, list[str], list[str]]:
    if not diary["available"]:
        return (
            "needs_diary",
            ["Pipeline experience diary is not available."],
            ["Pipeline model-memory bridge cannot be reviewed until a diary target exists."],
        )
    if not diary["compatible_decision_schema"]:
        return (
            "schema_mismatch",
            ["Pipeline diary exists, but its schema is not compatible with trade outcome records."],
            ["Do not write DEAN paper outcomes into this diary until the schema/target table is clarified."],
        )
    if candidates:
        return (
            "bridge_proposal_ready",
            [f"{len(candidates)} evaluated paper record(s) can be reviewed for diary bridging."],
            ["Bridge candidates are review evidence only; automatic writes remain disabled."],
        )
    return (
        "needs_evaluated_paper_records",
        ["No evaluated concrete single-ticker paper decisions are ready for diary bridging."],
        ["Pipeline diary learning cannot improve from DEAN paper records until outcomes exist."],
    )


def _recommendations(status: str, diary: dict[str, Any], candidates: list[dict[str, Any]]) -> list[str]:
    if status == "schema_mismatch":
        return [
            "Choose the target diary contract before writing: CSV modeling diary, DuckDB experience_diary table, or a new DEAN bridge table.",
            "Keep DEAN paper outcomes in review/proposal mode until the target schema has decision_type/outcome/profit_loss fields.",
        ]
    if status == "bridge_proposal_ready":
        return [
            "Review bridge_candidates and decide whether to create a manual diary-write migration.",
            "Require one-to-one ticker mapping and explicit outcome evidence before any write.",
        ]
    if status == "needs_evaluated_paper_records":
        return ["Evaluate paper records after their horizon and fresh price coverage, then rerun DiaryBridgeAgent."]
    if not diary["available"]:
        return ["Create or locate the actual pipeline experience diary target before bridging."]
    return ["No diary bridge action is ready."]


def _value_counts(rows: list[dict[str, str]], key: str) -> dict[str, int]:
    if not rows or key not in rows[0]:
        return {}
    counts = Counter(row.get(key) or "missing" for row in rows)
    return {str(item): int(count) for item, count in sorted(counts.items())}


def _verdict(bridge: dict[str, Any]) -> str:
    if bridge["status"] in {"schema_mismatch", "bridge_proposal_ready"}:
        return "caution"
    return "needs_more_data"


def _quality_score(bridge: dict[str, Any]) -> float:
    if bridge["status"] == "bridge_proposal_ready":
        return 0.75
    if bridge["pipeline_diary"]["available"]:
        return 0.45
    return 0.25
