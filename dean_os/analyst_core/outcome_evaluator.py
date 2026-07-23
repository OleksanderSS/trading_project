"""OutcomeEvaluator — compares SectorReport forecasts against actual prices.

Evaluates ticker recommendations at fixed horizons (1, 5, 20, 60, 120 days)
and computes hit rates, direction accuracy, and calibration metrics.

Usage:
    evaluator = OutcomeEvaluator(price_data_path="path/to/prices_1d.parquet")
    result = evaluator.evaluate(report, as_of="2026-07-01")
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from dean_os.analyst_core.schemas import OUTCOME_HORIZONS
from dean_os.analyst_core.sector_analyst import SectorReport


@dataclass
class TickerOutcome:
    """Outcome for a single ticker at a single horizon."""
    ticker: str
    horizon_days: int
    entry_price: float
    exit_price: float | None
    actual_return: float | None
    predicted_direction: str  # "bullish", "bearish", "neutral"
    direction_correct: bool | None  # None if outcome unavailable
    entry_date: str
    exit_date: str | None = None


@dataclass
class HorizonOutcome:
    """Aggregated outcomes for a single horizon."""
    horizon_days: int
    tickers_evaluated: int
    direction_accuracy: float | None  # fraction correct
    mean_return: float | None
    bullish_mean_return: float | None
    bearish_mean_return: float | None
    neutral_mean_return: float | None
    ticker_outcomes: list[TickerOutcome] = field(default_factory=list)


@dataclass
class EvaluationResult:
    """Full evaluation result for a SectorReport."""
    report_domain_id: str
    report_as_of: str
    evaluation_date: str
    tickers_in_basket: int
    horizons: list[HorizonOutcome]
    overall_direction_accuracy: float | None
    summary: dict[str, Any] = field(default_factory=dict)


class OutcomeEvaluator:
    """Evaluates SectorReport forecasts against actual price data.

    Args:
        price_data_path: Path to daily price parquet file.
            Expected columns: datetime, ticker, close, open, high, low, volume.
    """

    def __init__(self, price_data_path: str | Path):
        self.price_data_path = Path(price_data_path)
        self._prices: pd.DataFrame | None = None

    @property
    def prices(self) -> pd.DataFrame:
        """Lazy-load price data."""
        if self._prices is None:
            self._prices = pd.read_parquet(self.price_data_path)
            # Normalize datetime
            if "datetime" in self._prices.columns:
                self._prices["date"] = pd.to_datetime(self._prices["datetime"]).dt.normalize()
            elif "date" in self._prices.columns:
                self._prices["date"] = pd.to_datetime(self._prices["date"]).dt.normalize()
            # Sort for efficient lookups
            self._prices = self._prices.sort_values(["ticker", "date"]).reset_index(drop=True)
        return self._prices

    def evaluate(
        self,
        report: SectorReport,
        *,
        as_of: str,
        horizons: list[int] | None = None,
    ) -> EvaluationResult:
        """Evaluate a SectorReport's ticker recommendations.

        Args:
            report: SectorReport with ticker basket recommendations.
            as_of: Point-in-time for the report (ISO format).
            horizons: Horizons to evaluate. Defaults to OUTCOME_HORIZONS.

        Returns:
            EvaluationResult with outcomes at each horizon.
        """
        horizons = horizons or OUTCOME_HORIZONS
        as_of_date = pd.Timestamp(as_of).normalize().tz_localize(None)

        # Get ticker recommendations
        recommendations = self._extract_recommendations(report)

        # Evaluate each horizon
        horizon_outcomes: list[HorizonOutcome] = []
        all_accuracies: list[float] = []

        for h in horizons:
            outcome = self._evaluate_horizon(
                recommendations=recommendations,
                as_of_date=as_of_date,
                horizon_days=h,
            )
            horizon_outcomes.append(outcome)
            if outcome.direction_accuracy is not None:
                all_accuracies.append(outcome.direction_accuracy)

        # Overall accuracy
        overall_accuracy = (
            sum(all_accuracies) / len(all_accuracies) if all_accuracies else None
        )

        return EvaluationResult(
            report_domain_id=report.domain_id,
            report_as_of=report.as_of,
            evaluation_date=pd.Timestamp.now().isoformat(),
            tickers_in_basket=len(recommendations),
            horizons=horizon_outcomes,
            overall_direction_accuracy=overall_accuracy,
            summary=self._build_summary(horizon_outcomes, overall_accuracy),
        )

    def _extract_recommendations(
        self, report: SectorReport
    ) -> dict[str, str]:
        """Extract ticker -> direction mapping from report.

        Returns dict mapping ticker to predicted direction
        ("bullish", "bearish", "neutral").
        """
        recs: dict[str, str] = {}
        basket = report.ticker_basket

        for candidate in basket.candidates:
            ticker = candidate.ticker
            direction = candidate.expected_direction or "neutral"
            # Normalize direction
            if direction in ("up", "bullish", "positive"):
                direction = "bullish"
            elif direction in ("down", "bearish", "negative"):
                direction = "bearish"
            else:
                direction = "neutral"
            recs[ticker] = direction

        return recs

    def _evaluate_horizon(
        self,
        recommendations: dict[str, str],
        as_of_date: pd.Timestamp,
        horizon_days: int,
    ) -> HorizonOutcome:
        """Evaluate recommendations at a specific horizon."""
        outcomes: list[TickerOutcome] = []
        exit_date = as_of_date + pd.Timedelta(days=horizon_days)

        for ticker, direction in recommendations.items():
            # Get entry price (first available T+1 price to avoid look-ahead)
            entry_row = self._get_price_after(ticker, as_of_date)
            if entry_row is None:
                continue

            # Prefer open price for entry, fallback to close
            entry_price = float(entry_row.get("open", entry_row["close"]))
            entry_date = str(entry_row["date"])

            # Get exit price (closest to exit_date)
            exit_row = self._get_price_at_or_before(ticker, exit_date)
            if exit_row is None:
                continue

            exit_price = float(exit_row["close"])
            exit_date_str = str(exit_row["date"])

            # Calculate return
            if entry_price > 0:
                actual_return = (exit_price - entry_price) / entry_price
            else:
                actual_return = None

            # Direction correctness
            if actual_return is not None:
                if actual_return > 0.005:  # > 0.5%
                    actual_direction = "bullish"
                elif actual_return < -0.005:  # < -0.5%
                    actual_direction = "bearish"
                else:
                    actual_direction = "neutral"

                if direction == "neutral":
                    direction_correct = (actual_direction == "neutral")
                else:
                    direction_correct = direction == actual_direction
            else:
                direction_correct = None

            outcomes.append(TickerOutcome(
                ticker=ticker,
                horizon_days=horizon_days,
                entry_price=entry_price,
                exit_price=exit_price,
                actual_return=actual_return,
                predicted_direction=direction,
                direction_correct=direction_correct,
                entry_date=entry_date,
                exit_date=exit_date_str,
            ))

        # Aggregate
        if not outcomes:
            return HorizonOutcome(
                horizon_days=horizon_days,
                tickers_evaluated=0,
                direction_accuracy=None,
                mean_return=None,
                bullish_mean_return=None,
                bearish_mean_return=None,
                neutral_mean_return=None,
            )

        # Direction accuracy
        correct = [o for o in outcomes if o.direction_correct is not None]
        accuracy = (
            sum(1 for o in correct if o.direction_correct) / len(correct)
            if correct
            else None
        )

        # Mean returns by direction
        returns = [o.actual_return for o in outcomes if o.actual_return is not None]
        bullish_returns = [
            o.actual_return for o in outcomes
            if o.actual_return is not None and o.predicted_direction == "bullish"
        ]
        bearish_returns = [
            o.actual_return for o in outcomes
            if o.actual_return is not None and o.predicted_direction == "bearish"
        ]
        neutral_returns = [
            o.actual_return for o in outcomes
            if o.actual_return is not None and o.predicted_direction == "neutral"
        ]

        return HorizonOutcome(
            horizon_days=horizon_days,
            tickers_evaluated=len(outcomes),
            direction_accuracy=accuracy,
            mean_return=sum(returns) / len(returns) if returns else None,
            bullish_mean_return=sum(bullish_returns) / len(bullish_returns) if bullish_returns else None,
            bearish_mean_return=sum(bearish_returns) / len(bearish_returns) if bearish_returns else None,
            neutral_mean_return=sum(neutral_returns) / len(neutral_returns) if neutral_returns else None,
            ticker_outcomes=outcomes,
        )

    def _get_price_at_or_before(
        self, ticker: str, target_date: pd.Timestamp
    ) -> pd.Series | None:
        """Get price row for ticker at or before target_date."""
        ticker_prices = self.prices[self.prices["ticker"] == ticker]
        if ticker_prices.empty:
            return None

        # Find prices on or before target_date
        valid = ticker_prices[ticker_prices["date"] <= target_date]
        if valid.empty:
            return None

        # Return the most recent
        return valid.iloc[-1]

    def _get_price_after(
        self, ticker: str, target_date: pd.Timestamp
    ) -> pd.Series | None:
        """Get the first available price row for ticker strictly after target_date."""
        ticker_prices = self.prices[self.prices["ticker"] == ticker]
        if ticker_prices.empty:
            return None

        valid = ticker_prices[ticker_prices["date"] > target_date]
        if valid.empty:
            return None

        return valid.iloc[0]

    def _build_summary(
        self,
        horizons: list[HorizonOutcome],
        overall_accuracy: float | None,
    ) -> dict[str, Any]:
        """Build summary statistics."""
        summary: dict[str, Any] = {
            "horizons_evaluated": len(horizons),
            "overall_direction_accuracy": overall_accuracy,
        }

        for h in horizons:
            key = f"horizon_{h.horizon_days}d"
            summary[key] = {
                "tickers_evaluated": h.tickers_evaluated,
                "direction_accuracy": h.direction_accuracy,
                "mean_return": h.mean_return,
            }

        return summary


def render_evaluation_markdown(result: EvaluationResult) -> str:
    """Render evaluation result as markdown."""
    lines: list[str] = []

    lines.append("# Outcome Evaluation")
    lines.append(f"**Domain:** {result.report_domain_id}")
    lines.append(f"**Report as of:** {result.report_as_of}")
    lines.append(f"**Evaluation date:** {result.evaluation_date}")
    lines.append(f"**Tickers in basket:** {result.tickers_in_basket}")
    lines.append("")

    if result.overall_direction_accuracy is not None:
        lines.append(f"**Overall direction accuracy:** {result.overall_direction_accuracy:.1%}")
    else:
        lines.append("**Overall direction accuracy:** N/A")
    lines.append("")

    lines.append("## Horizon Results")
    lines.append("")
    lines.append("| Horizon | Tickers | Accuracy | Mean Return | Bullish | Bearish | Neutral |")
    lines.append("|---------|---------|----------|-------------|---------|---------|---------|")

    for h in result.horizons:
        acc = f"{h.direction_accuracy:.1%}" if h.direction_accuracy is not None else "N/A"
        mr = f"{h.mean_return:+.2%}" if h.mean_return is not None else "N/A"
        br = f"{h.bullish_mean_return:+.2%}" if h.bullish_mean_return is not None else "N/A"
        ber = f"{h.bearish_mean_return:+.2%}" if h.bearish_mean_return is not None else "N/A"
        nr = f"{h.neutral_mean_return:+.2%}" if h.neutral_mean_return is not None else "N/A"
        lines.append(f"| {h.horizon_days}d | {h.tickers_evaluated} | {acc} | {mr} | {br} | {ber} | {nr} |")

    lines.append("")

    # Per-ticker detail for first horizon with outcomes
    for h in result.horizons:
        if h.ticker_outcomes:
            lines.append(f"## Ticker Details ({h.horizon_days}d horizon)")
            lines.append("")
            lines.append("| Ticker | Direction | Entry | Exit | Return | Correct |")
            lines.append("|--------|-----------|-------|------|--------|---------|")
            for o in h.ticker_outcomes:
                ret = f"{o.actual_return:+.2%}" if o.actual_return is not None else "N/A"
                correct = "Y" if o.direction_correct else ("N/A" if o.direction_correct is None else "N")
                lines.append(
                    f"| {o.ticker} | {o.predicted_direction} | "
                    f"${o.entry_price:.2f} | ${o.exit_price:.2f} | {ret} | {correct} |"
                )
            lines.append("")
            break  # Only show first horizon detail

    return "\n".join(lines)
