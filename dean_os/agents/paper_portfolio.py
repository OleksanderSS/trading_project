from __future__ import annotations

from pathlib import Path
from typing import Any

from dean_os.base import BaseAgent
from dean_os.paper_portfolio import PaperPortfolioSimulator
from dean_os.schemas import MarketContext, PipelineReport
from dean_os.utils import clamp


class PaperPortfolioAgent(BaseAgent):
    """Runs a paper-only portfolio simulation from logged paper decisions."""

    version = "0.1.0"
    branch = "pipeline"

    async def run(self, context: MarketContext) -> PipelineReport:
        try:
            simulation = PaperPortfolioSimulator(
                self.config.get("store_path", "data/dean_os/paper_trades.sqlite")
            ).simulate(
                market_data_path=self.config.get("market_data_path"),
                latest_processed_prices=self.config.get("latest_processed_prices", "1d"),
                tickers=self.config.get("tickers") or context.tickers,
                as_of=self.config.get("as_of"),
                initial_cash=float(self.config.get("initial_cash", 100_000.0)),
                position_size_pct=float(self.config.get("position_size_pct", 0.05)),
                include_watchlist=bool(self.config.get("include_watchlist", False)),
                watchlist_position_size_pct=float(self.config.get("watchlist_position_size_pct", 0.0)),
                confidence_weighting=bool(self.config.get("confidence_weighting", False)),
                slippage_bps=float(self.config.get("slippage_bps", 5.0)),
                commission_bps=float(self.config.get("commission_bps", 1.0)),
                close_col=self.config.get("close_col", "close"),
                datetime_col=self.config.get("datetime_col", "datetime"),
                statuses=list(self.config.get("statuses", ["pending", "evaluated"])),
                limit=self.config.get("limit"),
            )
        except (FileNotFoundError, ValueError) as exc:
            if (
                isinstance(exc, ValueError)
                and "Provide market_data_path" not in str(exc)
            ):
                raise
            return PipelineReport(
                agent_name=self.name,
                agent_version=self.version,
                verdict="needs_more_data",
                confidence=0.0,
                data_quality_score=0.0,
                reasons=[
                    "No market price data available for portfolio "
                    f"simulation: {exc}"
                ],
                risks=["Paper portfolio cannot be simulated without market price data."],
                blind_spots=["Agent autonomy cannot be calibrated without price coverage."],
                evidence=[],
                input_hash=self.context_hash(context),
                metrics_snapshot={"simulation_skipped": True},
            )
        context.metadata["paper_portfolio"] = simulation

        summary = simulation["summary"]
        verdict, reasons, risks, quality_score = _classify_simulation(
            simulation=simulation,
            max_drawdown_limit=float(self.config.get("max_drawdown_limit", 0.10)),
        )
        return PipelineReport(
            agent_name=self.name,
            agent_version=self.version,
            verdict=verdict,
            confidence=0.84,
            data_quality_score=quality_score,
            signal_strength=clamp(float(summary["total_return"]) / 0.10, -1.0, 1.0),
            reasons=reasons,
            risks=risks,
            blind_spots=[
                "PaperPortfolioAgent simulates logged decisions only; it does not infer missing order size, stops, or live broker execution quality."
            ],
            evidence=[
                self.evidence("file", str(Path(simulation["paper_trade_store"])), "paper_trade_store", simulation["paper_trade_store"]),
                self.evidence("file", simulation["market_data_path"], "market_data_path", simulation["market_data_path"]),
                self.evidence("metric", "paper_portfolio", "position_count", summary["position_count"]),
                self.evidence("metric", "paper_portfolio", "total_return", summary["total_return"]),
                self.evidence("metric", "paper_portfolio", "max_drawdown", summary["max_drawdown"]),
                self.evidence("metric", "paper_portfolio", "skipped_by_status", summary["skipped_by_status"]),
            ],
            input_hash=self.context_hash(context),
            metrics_snapshot=simulation,
        )


def _classify_simulation(
    simulation: dict[str, Any],
    max_drawdown_limit: float,
) -> tuple[str, list[str], list[str], float]:
    summary = simulation["summary"]
    if summary["record_count"] == 0:
        return (
            "needs_more_data",
            ["No paper records are available for portfolio simulation."],
            ["Agent autonomy cannot be calibrated until paper decisions are logged and tested."],
            0.25,
        )
    if summary["position_count"] == 0:
        return (
            "needs_more_data",
            ["Paper records exist, but no paper positions could be opened from current local price coverage."],
            ["Recent paper decisions may remain untestable until market data is refreshed."],
            0.35,
        )
    if float(summary["max_drawdown"]) > max_drawdown_limit:
        return (
            "caution",
            [f"Paper portfolio max drawdown {summary['max_drawdown']:.3f} exceeds limit {max_drawdown_limit:.3f}."],
            ["Review sizing, regime filters, and thesis evidence before expanding paper autonomy."],
            0.7,
        )
    if summary["skipped_count"]:
        return (
            "caution",
            ["Paper portfolio simulation ran, but some records were skipped."],
            ["Skipped records can bias portfolio diagnostics if missing data is systematic."],
            0.75,
        )
    return (
        "clear",
        ["Paper portfolio simulation completed within configured drawdown limits."],
        ["Paper results are simulated diagnostics only and do not authorize live execution."],
        0.88,
    )
