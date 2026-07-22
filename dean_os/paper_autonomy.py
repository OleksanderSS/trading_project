from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from dean_os.agents.chief_review import ChiefReviewAgent
from dean_os.agents.diary_bridge import DiaryBridgeAgent
from dean_os.agents.market_data_freshness import MarketDataFreshnessAgent
from dean_os.agents.paper_portfolio import PaperPortfolioAgent
from dean_os.agents.regime import RegimeAgent
from dean_os.event_log import EventLog
from dean_os.review import AgentReviewBuilder
from dean_os.schemas import MarketContext, PipelineReport, utc_now_iso
from dean_os.utils import json_ready


class PaperAutonomyRunner:
    """Runs the supervised paper-autonomy loop without executing trades."""

    def __init__(
        self,
        output_dir: str | Path = "reports/dean_os/paper_autonomy",
        event_log_path: str | Path = "logs/dean_os/events.jsonl",
        decision_log_path: str | Path = "logs/dean_os/decisions.jsonl",
        experience_diary_path: str | Path = "logs/experience_diary.csv",
    ):
        self.output_dir = Path(output_dir)
        self.event_log_path = Path(event_log_path)
        self.decision_log_path = Path(decision_log_path)
        self.experience_diary_path = Path(experience_diary_path)

    async def run(
        self,
        tickers: list[str] | None = None,
        timeframe: str = "1d",
        market_data_path: str | Path | None = None,
        latest_processed_prices: str = "1d",
        as_of: str | None = None,
        max_age_hours: float = 72.0,
        paper_store_path: str | Path = "data/dean_os/paper_trades.sqlite",
        initial_cash: float = 100_000.0,
        position_size_pct: float = 0.05,
        include_watchlist: bool = False,
        review_snapshot_path: str | Path | None = None,
        max_drawdown_limit: float = 0.10,
        save: bool = True,
    ) -> dict[str, Any]:
        tickers = [ticker.upper() for ticker in tickers or [] if str(ticker).strip()]
        context = MarketContext(tickers=tickers, timeframes=[timeframe], timeframe=timeframe)
        reports: list[PipelineReport] = []

        freshness_report = await MarketDataFreshnessAgent(
            name="market_data_freshness",
            config={
                "market_data_path": str(market_data_path) if market_data_path else None,
                "latest_processed_prices": latest_processed_prices,
                "tickers": tickers,
                "as_of": as_of,
                "max_age_hours": max_age_hours,
            },
        ).run(context)
        reports.append(freshness_report)

        regime_report = await RegimeAgent(
            name="regime",
            config={
                "market_data_path": str(market_data_path) if market_data_path else None,
                "latest_processed_prices": latest_processed_prices,
                "ticker": tickers[0] if tickers else None,
                "engine": "fallback",
            },
        ).run(context)
        reports.append(regime_report)

        review_snapshot = _load_or_build_review_snapshot(review_snapshot_path)
        context.metadata["review_snapshot"] = review_snapshot

        chief_report = await ChiefReviewAgent(
            name="chief_review",
            config={"autonomy_mode": "paper_supervised"},
        ).run(context)
        reports.append(chief_report)

        portfolio_report = await PaperPortfolioAgent(
            name="paper_portfolio",
            config={
                "store_path": str(paper_store_path),
                "market_data_path": str(market_data_path) if market_data_path else None,
                "latest_processed_prices": latest_processed_prices,
                "tickers": tickers,
                "as_of": as_of,
                "initial_cash": initial_cash,
                "position_size_pct": position_size_pct,
                "include_watchlist": include_watchlist,
                "max_drawdown_limit": max_drawdown_limit,
            },
        ).run(context)
        reports.append(portfolio_report)

        diary_bridge_report = await DiaryBridgeAgent(
            name="diary_bridge",
            config={
                "experience_diary_path": str(self.experience_diary_path),
                "paper_store_path": str(paper_store_path),
            },
        ).run(context)
        reports.append(diary_bridge_report)

        journals = build_journal_summary(
            event_log_path=self.event_log_path,
            decision_log_path=self.decision_log_path,
            experience_diary_path=self.experience_diary_path,
        )
        payload = {
            "run_id": _run_id(),
            "created_at": utc_now_iso(),
            "mode": "paper_supervised",
            "inputs": {
                "tickers": tickers,
                "timeframe": timeframe,
                "market_data_path": str(market_data_path) if market_data_path else None,
                "latest_processed_prices": latest_processed_prices,
                "paper_store_path": str(paper_store_path),
                "review_snapshot_path": str(review_snapshot_path) if review_snapshot_path else None,
            },
            "decision": _classify_autonomy(context=context, reports=reports, journals=journals),
            "reports": [report.model_dump(mode="json") for report in reports],
            "data_freshness": context.metadata.get("data_freshness", {}),
            "regime_context": context.metadata.get("regime_context", {}),
            "chief_review": context.metadata.get("chief_review", {}),
            "paper_portfolio": context.metadata.get("paper_portfolio", {}),
            "diary_bridge": context.metadata.get("diary_bridge", {}),
            "action_proposals": [proposal.model_dump(mode="json") for proposal in context.action_proposals],
            "journals": journals,
        }
        payload["recommendations"] = _recommendations(payload)
        if save:
            json_path, md_path = self.save(payload)
            payload["saved_paths"] = {"json": str(json_path), "markdown": str(md_path)}
            EventLog(self.event_log_path).write(
                "paper_autonomy_run_completed",
                "paper_autonomy",
                {
                    "run_id": payload["run_id"],
                    "decision": payload["decision"]["status"],
                    "json_path": str(json_path),
                    "tickers": tickers,
                },
                run_id=payload["run_id"],
            )
        return payload

    def save(self, payload: dict[str, Any]) -> tuple[Path, Path]:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        run_id = payload["run_id"]
        json_path = self.output_dir / f"{run_id}.json"
        md_path = self.output_dir / f"{run_id}.md"
        latest_json = self.output_dir / "latest.json"
        latest_md = self.output_dir / "latest.md"
        rendered_json = json.dumps(json_ready(payload), indent=2, ensure_ascii=False)
        rendered_md = render_paper_autonomy_markdown(payload)
        json_path.write_text(rendered_json + "\n", encoding="utf-8")
        md_path.write_text(rendered_md, encoding="utf-8")
        latest_json.write_text(rendered_json + "\n", encoding="utf-8")
        latest_md.write_text(rendered_md, encoding="utf-8")
        return json_path, md_path


def build_journal_summary(
    event_log_path: str | Path = "logs/dean_os/events.jsonl",
    decision_log_path: str | Path = "logs/dean_os/decisions.jsonl",
    experience_diary_path: str | Path = "logs/experience_diary.csv",
) -> dict[str, Any]:
    return {
        "dean_events": EventLog(event_log_path).summary(),
        "dean_decisions": _decision_log_summary(Path(decision_log_path)),
        "pipeline_experience_diary": _experience_diary_summary(Path(experience_diary_path)),
    }


def render_paper_autonomy_markdown(payload: dict[str, Any]) -> str:
    decision = payload["decision"]
    freshness = payload.get("data_freshness", {}).get("market_prices", {})
    regime = payload.get("regime_context", {})
    portfolio = payload.get("paper_portfolio", {}).get("summary", {})
    diary_bridge = payload.get("diary_bridge", {})
    journals = payload.get("journals", {})
    lines = [
        "# DEAN-OS Paper Autonomy",
        "",
        f"- Run ID: `{payload['run_id']}`",
        f"- Status: `{decision['status']}`",
        f"- Reason: {decision['reason']}",
        f"- Tickers: {', '.join(payload['inputs']['tickers']) or 'none'}",
        "",
        "## Market",
        "",
        f"- Freshness status: {freshness.get('status')}",
        f"- Latest timestamp: {freshness.get('latest_timestamp')}",
        f"- Regime: {regime.get('regime', 'UNKNOWN')}",
        f"- Regime tags: {', '.join(regime.get('context_tags', [])) or 'none'}",
        "",
        "## Paper Portfolio",
        "",
        f"- Records: {portfolio.get('record_count', 0)}",
        f"- Positions: {portfolio.get('position_count', 0)}",
        f"- Total return: {portfolio.get('total_return', 0)}",
        f"- Max drawdown: {portfolio.get('max_drawdown', 0)}",
        f"- Skipped: {portfolio.get('skipped_by_status', {})}",
        "",
        "## Diary Bridge",
        "",
        f"- Status: {diary_bridge.get('status')}",
        f"- Schema: {diary_bridge.get('pipeline_diary', {}).get('schema_kind')}",
        f"- Bridge candidates: {diary_bridge.get('paper_records', {}).get('bridge_candidate_count', 0)}",
        "",
        "## Journals",
        "",
        f"- DEAN events: {journals.get('dean_events', {}).get('event_count', 0)}",
        f"- DEAN decisions: {journals.get('dean_decisions', {}).get('decision_count', 0)}",
        f"- Pipeline diary rows: {journals.get('pipeline_experience_diary', {}).get('row_count', 0)}",
        "",
        "## Recommendations",
        "",
    ]
    for recommendation in payload.get("recommendations", []):
        lines.append(f"- {recommendation}")
    return "\n".join(lines).strip() + "\n"


def _load_or_build_review_snapshot(review_snapshot_path: str | Path | None) -> dict[str, Any]:
    from dean_os.dean_paths import DeanPaths

    if review_snapshot_path:
        try:
            return DeanPaths.load_json(review_snapshot_path)
        except Exception as exc:
            return {"available": False, "error": f"Review snapshot error: {exc}"}
    try:
        return AgentReviewBuilder().build()
    except Exception as exc:
        return {"available": False, "error": f"Could not build review snapshot: {type(exc).__name__}: {exc}"}


def _decision_log_summary(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"path": str(path), "available": False, "decision_count": 0, "latest_decision": None}
    decisions: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                decisions.append(json.loads(line))
    return {
        "path": str(path),
        "available": True,
        "decision_count": len(decisions),
        "final_decision_counts": dict(sorted(Counter(item.get("final_decision", "unknown") for item in decisions).items())),
        "latest_decision": decisions[-1] if decisions else None,
    }


def _experience_diary_summary(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"path": str(path), "available": False, "row_count": 0, "format": "missing"}
    try:
        import pandas as pd

        frame = pd.read_csv(path)
    except Exception as exc:
        return {
            "path": str(path),
            "available": True,
            "row_count": None,
            "format": "csv",
            "error": f"Could not read diary CSV: {type(exc).__name__}: {exc}",
        }
    summary: dict[str, Any] = {
        "path": str(path),
        "available": True,
        "format": "csv",
        "row_count": int(len(frame)),
        "columns": list(frame.columns),
    }
    for column in ("outcome", "decision_type", "ticker", "model_name", "agent_id", "is_champion"):
        if column in frame.columns:
            summary[f"{column}_counts"] = {
                str(key): int(value) for key, value in sorted(frame[column].fillna("missing").value_counts().to_dict().items())
            }
    return summary


def _classify_autonomy(
    context: MarketContext,
    reports: list[PipelineReport],
    journals: dict[str, Any],
) -> dict[str, Any]:
    blocked_reports = [report.agent_name for report in reports if report.verdict == "blocked"]
    if blocked_reports:
        return {"status": "blocked", "reason": f"Blocked by: {', '.join(blocked_reports)}", "allow_new_paper_decision": False}

    freshness = context.metadata.get("data_freshness", {}).get("market_prices", {})
    if freshness.get("stale") or freshness.get("status") in {"stale", "unavailable"}:
        return {"status": "needs_market_data", "reason": "Market data freshness is not sufficient for new paper autonomy.", "allow_new_paper_decision": False}

    portfolio = context.metadata.get("paper_portfolio", {})
    skipped = portfolio.get("summary", {}).get("skipped_by_status", {})
    if skipped.get("as_of_before_record") or skipped.get("no_price_after_created_at"):
        return {"status": "needs_market_data", "reason": "Paper decisions are newer than local market data.", "allow_new_paper_decision": False}

    chief = context.metadata.get("chief_review", {})
    if chief.get("decision") == "needs_more_data":
        return {"status": "needs_more_data", "reason": "Chief review requires more evidence before expanding paper autonomy.", "allow_new_paper_decision": False}

    diary_bridge = context.metadata.get("diary_bridge", {})
    if diary_bridge.get("status") == "schema_mismatch":
        return {
            "status": "observe_only",
            "reason": "Paper diagnostics can continue, but diary bridge schema is not ready for learning writes.",
            "allow_new_paper_decision": False,
        }

    diary = journals.get("pipeline_experience_diary", {})
    if diary.get("available") and diary.get("row_count") == 0:
        return {
            "status": "observe_only",
            "reason": "Safe paper loop can run, but the pipeline experience diary has no rows yet.",
            "allow_new_paper_decision": False,
        }

    return {
        "status": "paper_observe",
        "reason": "Safe paper-autonomy diagnostics can continue under human review.",
        "allow_new_paper_decision": False,
    }


def _recommendations(payload: dict[str, Any]) -> list[str]:
    recommendations: list[str] = []
    decision = payload["decision"]["status"]
    if decision == "needs_market_data":
        recommendations.append("Refresh local market prices before evaluating or expanding paper autonomy.")
    if payload.get("journals", {}).get("pipeline_experience_diary", {}).get("row_count") == 0:
        recommendations.append("Keep the pipeline experience diary as a separate historical model-memory source; bridge it into reports before writing to it from DEAN-OS.")
    if payload.get("journals", {}).get("dean_decisions", {}).get("decision_count") == 0:
        recommendations.append("DecisionLogger has no consensus decisions yet; run DEAN-OS with logging only after safe gates are stable.")
    if payload.get("paper_portfolio", {}).get("summary", {}).get("position_count", 0) == 0:
        recommendations.append("Paper portfolio has no simulated positions; do not interpret hit rate or PnL until local prices cover paper decision dates.")
    if payload.get("diary_bridge", {}).get("status") == "schema_mismatch":
        recommendations.append("Resolve diary bridge schema before allowing DEAN paper outcomes to influence pipeline model memory.")
    if not recommendations:
        recommendations.append("Review this paper-autonomy report before approving any proposal or recording new paper decisions.")
    return recommendations


def _run_id() -> str:
    return "paper_autonomy_" + utc_now_iso().replace(":", "").replace("-", "").replace(".", "_")
