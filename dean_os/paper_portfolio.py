from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from dean_os.market_data_api import parse_datetime, prepare_market_frame, read_market_frame
from dean_os.outcome_evaluation import (
    _frame_latest_datetime,
    _resolve_market_data_path,
)
from dean_os.paper_trading import PaperTradeStore
from dean_os.schemas import PaperTradeRecord
from dean_os.utils import clamp


class PaperPortfolioSimulator:
    """Turns logged paper decisions into a deterministic portfolio simulation."""

    def __init__(self, store_path: str | Path = "data/dean_os/paper_trades.sqlite"):
        self.store_path = Path(store_path)

    def simulate(
        self,
        market_data_path: str | Path | None = None,
        latest_processed_prices: str | None = "1d",
        tickers: list[str] | None = None,
        as_of: str | None = None,
        initial_cash: float = 100_000.0,
        position_size_pct: float = 0.05,
        include_watchlist: bool = False,
        watchlist_position_size_pct: float = 0.0,
        confidence_weighting: bool = False,
        slippage_bps: float = 5.0,
        commission_bps: float = 1.0,
        close_col: str = "close",
        datetime_col: str = "datetime",
        statuses: list[str] | None = None,
        limit: int | None = None,
        max_gross_exposure: float = 1.0,
        max_net_exposure: float = 1.0,
    ) -> dict[str, Any]:
        try:
            import pandas as pd
        except Exception as exc:
            raise RuntimeError(f"pandas is required for paper portfolio simulation: {exc}") from exc

        resolved_path = _resolve_market_data_path(market_data_path, latest_processed_prices)
        if resolved_path is None:
            raise ValueError("Provide market_data_path or --latest-processed-prices.")
        if not resolved_path.exists():
            raise FileNotFoundError(f"Market data file does not exist: {resolved_path}")

        frame = prepare_market_frame(
            pd=pd,
            frame=read_market_frame(pd, resolved_path),
            close_col=close_col,
            datetime_col=datetime_col,
        )
        if frame.empty:
            raise ValueError("No usable market rows after parsing close/datetime columns.")

        as_of_dt = parse_datetime(as_of) if as_of else _frame_latest_datetime(frame)
        store = PaperTradeStore(self.store_path)
        records = _select_records(store=store, statuses=statuses)
        if limit is not None:
            records = records[:limit]

        requested_tickers = [ticker.upper() for ticker in tickers or [] if str(ticker).strip()]
        cost_rate = (float(slippage_bps) + float(commission_bps)) / 10_000.0
        positions: list[dict[str, Any]] = []
        skipped: list[dict[str, Any]] = []

        # Track aggregate exposure for enforcement
        aggregate_gross_exposure = 0.0
        aggregate_net_exposure = 0.0

        for record in records:
            simulated = _simulate_record(
                frame=frame,
                record=record,
                requested_tickers=requested_tickers,
                as_of=as_of_dt,
                initial_cash=float(initial_cash),
                position_size_pct=float(position_size_pct),
                include_watchlist=include_watchlist,
                watchlist_position_size_pct=float(watchlist_position_size_pct),
                confidence_weighting=confidence_weighting,
                cost_rate=cost_rate,
                max_gross_exposure=float(max_gross_exposure),
                max_net_exposure=float(max_net_exposure),
                current_gross_exposure=aggregate_gross_exposure,
                current_net_exposure=aggregate_net_exposure,
            )

            # Update aggregate exposure with positions that passed validation
            for position in simulated["positions"]:
                aggregate_gross_exposure += abs(float(position["notional"]))
                aggregate_net_exposure += float(position["notional"]) * float(position["side_multiplier"])

            positions.extend(simulated["positions"])
            skipped.extend(simulated["skipped"])

        equity_curve = _build_equity_curve(
            frame=frame,
            positions=positions,
            initial_cash=float(initial_cash),
            cost_rate=cost_rate,
        )
        summary = _summary(
            records=records,
            positions=positions,
            skipped=skipped,
            equity_curve=equity_curve,
            initial_cash=float(initial_cash),
        )
        return {
            "status": "simulated" if positions else "no_positions",
            "paper_trade_store": str(self.store_path),
            "market_data_path": str(resolved_path),
            "latest_processed_prices": latest_processed_prices,
            "as_of": as_of_dt.isoformat(),
            "assumptions": {
                "initial_cash": float(initial_cash),
                "position_size_pct": float(position_size_pct),
                "include_watchlist": include_watchlist,
                "watchlist_position_size_pct": float(watchlist_position_size_pct),
                "confidence_weighting": confidence_weighting,
                "slippage_bps": float(slippage_bps),
                "commission_bps": float(commission_bps),
                "round_trip_cost_bps": round(cost_rate * 20_000.0, 6),
                "max_gross_exposure": float(max_gross_exposure),
                "max_net_exposure": float(max_net_exposure),
            },
            "record_count": len(records),
            "positions": positions,
            "skipped": skipped,
            "equity_curve": equity_curve,
            "summary": summary,
            "recommendations": _recommendations(summary),
        }


def _select_records(store: PaperTradeStore, statuses: list[str] | None) -> list[PaperTradeRecord]:
    selected_statuses = set(statuses or ["pending", "evaluated"])
    return [record for record in store.list_records() if record.status in selected_statuses]


def _simulate_record(
    frame: Any,
    record: PaperTradeRecord,
    requested_tickers: list[str],
    as_of: datetime,
    initial_cash: float,
    position_size_pct: float,
    include_watchlist: bool,
    watchlist_position_size_pct: float,
    confidence_weighting: bool,
    cost_rate: float,
    max_gross_exposure: float = 1.0,
    max_net_exposure: float = 1.0,
    current_gross_exposure: float = 0.0,
    current_net_exposure: float = 0.0,
) -> dict[str, list[dict[str, Any]]]:
    base_payload = _record_payload(record)
    record_tickers = _record_tickers(record, requested_tickers)
    if not record_tickers:
        return {
            "positions": [],
            "skipped": [{**base_payload, "status": "missing_tickers", "reason": "Paper record has no tickers."}],
        }
    if record.action == "no_trade":
        return {
            "positions": [],
            "skipped": [{**base_payload, "tickers": record_tickers, "status": "no_trade", "reason": "Record explicitly chose no_trade."}],
        }
    if record.action == "watchlist" and not include_watchlist:
        return {
            "positions": [],
            "skipped": [
                {
                    **base_payload,
                    "tickers": record_tickers,
                    "status": "watchlist_only",
                    "reason": "Watchlist records are not opened unless include_watchlist is enabled.",
                }
            ],
        }

    side = _side(record)
    if side == 0:
        return {
            "positions": [],
            "skipped": [{**base_payload, "tickers": record_tickers, "status": "neutral_direction", "reason": "Neutral records do not open positions."}],
        }

    start_at = parse_datetime(record.created_at)
    due_at = start_at + timedelta(days=record.horizon_days)
    target_at = min(as_of, due_at)
    if target_at < start_at:
        return {
            "positions": [],
            "skipped": [{**base_payload, "tickers": record_tickers, "status": "as_of_before_record", "reason": "as_of is earlier than record creation."}],
        }

    size_pct = watchlist_position_size_pct if record.action == "watchlist" else position_size_pct
    confidence_multiplier = clamp(record.confidence, 0.0, 1.0) if confidence_weighting else 1.0
    notional_per_ticker = initial_cash * size_pct * confidence_multiplier / max(len(record_tickers), 1)
    if notional_per_ticker <= 0:
        return {
            "positions": [],
            "skipped": [
                {
                    **base_payload,
                    "tickers": record_tickers,
                    "status": "zero_notional",
                    "reason": "Configured position sizing produced zero notional.",
                }
            ],
        }

    # Check if adding this position would exceed exposure limits
    new_gross_exposure = current_gross_exposure + (abs(notional_per_ticker) * len(record_tickers))
    new_net_exposure = current_net_exposure + (notional_per_ticker * side * len(record_tickers))

    if new_gross_exposure > initial_cash * max_gross_exposure:
        return {
            "positions": [],
            "skipped": [
                {
                    **base_payload,
                    "tickers": record_tickers,
                    "status": "exceeds_gross_exposure_limit",
                    "reason": f"Position would exceed gross exposure limit: {new_gross_exposure:.2f} > {initial_cash * max_gross_exposure:.2f}",
                }
            ],
        }

    if abs(new_net_exposure) > initial_cash * max_net_exposure:
        return {
            "positions": [],
            "skipped": [
                {
                    **base_payload,
                    "tickers": record_tickers,
                    "status": "exceeds_net_exposure_limit",
                    "reason": f"Position would exceed net exposure limit: {abs(new_net_exposure):.2f} > {initial_cash * max_net_exposure:.2f}",
                }
            ],
        }

    positions: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for ticker in record_tickers:
        ticker_frame = frame[frame["_dean_ticker"] == ticker.upper()]
        if ticker_frame.empty:
            skipped.append({**base_payload, "ticker": ticker.upper(), "status": "missing_ticker", "reason": "Ticker not found in market data."})
            continue
        entry_candidates = ticker_frame[ticker_frame["_dean_datetime"] >= start_at]
        if entry_candidates.empty:
            skipped.append(
                {
                    **base_payload,
                    "ticker": ticker.upper(),
                    "status": "no_price_after_created_at",
                    "reason": "Market data ends before this record can open a paper position.",
                    "latest_price_at": ticker_frame["_dean_datetime"].max().isoformat(),
                }
            )
            continue
        entry_row = entry_candidates.iloc[0]
        entry_at = entry_row["_dean_datetime"].to_pydatetime()
        if target_at < entry_at:
            skipped.append(
                {
                    **base_payload,
                    "ticker": ticker.upper(),
                    "status": "no_price_before_as_of",
                    "reason": "The first usable entry price is after as_of.",
                    "entry_at": entry_at.isoformat(),
                }
            )
            continue
        exit_candidates = ticker_frame[(ticker_frame["_dean_datetime"] <= target_at) & (ticker_frame["_dean_datetime"] >= entry_at)]
        if exit_candidates.empty:
            skipped.append(
                {
                    **base_payload,
                    "ticker": ticker.upper(),
                    "status": "missing_exit_price",
                    "reason": "No usable price exists between entry and target.",
                    "entry_at": entry_at.isoformat(),
                    "target_at": target_at.isoformat(),
                }
            )
            continue
        exit_row = exit_candidates.iloc[-1]
        entry_price = float(entry_row["_dean_close"])
        exit_price = float(exit_row["_dean_close"])
        if entry_price == 0:
            skipped.append({**base_payload, "ticker": ticker.upper(), "status": "zero_entry_price", "reason": "Entry price is zero."})
            continue

        gross_return = side * (exit_price / entry_price - 1.0)
        net_return = gross_return - 2 * cost_rate
        pnl = notional_per_ticker * net_return
        exit_at = exit_row["_dean_datetime"].to_pydatetime()
        positions.append(
            {
                **base_payload,
                "ticker": ticker.upper(),
                "status": "closed" if as_of >= due_at else "open",
                "side": "long" if side > 0 else "short",
                "side_multiplier": side,
                "notional": round(notional_per_ticker, 6),
                "entry_at": entry_at.isoformat(),
                "exit_at": exit_at.isoformat(),
                "due_at": due_at.isoformat(),
                "entry_price": entry_price,
                "exit_price": exit_price,
                "gross_return": round(gross_return, 8),
                "net_return": round(net_return, 8),
                "pnl": round(pnl, 6),
                "cost_return": round(2 * cost_rate, 8),
                "holding_days": round((exit_at - entry_at).total_seconds() / 86_400.0, 4),
            }
        )
    return {"positions": positions, "skipped": skipped}


def _build_equity_curve(
    frame: Any,
    positions: list[dict[str, Any]],
    initial_cash: float,
    cost_rate: float,
) -> list[dict[str, Any]]:
    if not positions:
        return []

    intervals = [(parse_datetime(position["entry_at"]), parse_datetime(position["exit_at"])) for position in positions]
    first_at = min(start for start, _ in intervals)
    last_at = max(end for _, end in intervals)
    relevant_frame = frame[(frame["_dean_datetime"] >= first_at) & (frame["_dean_datetime"] <= last_at)]
    timestamps = [value.to_pydatetime() for value in relevant_frame["_dean_datetime"].drop_duplicates().sort_values()]
    if first_at not in timestamps:
        timestamps.insert(0, first_at)
    if last_at not in timestamps:
        timestamps.append(last_at)

    curve: list[dict[str, Any]] = []
    for timestamp in sorted(set(timestamps)):
        pnl = 0.0
        gross_exposure = 0.0
        net_exposure = 0.0
        active_count = 0
        for position in positions:
            entry_at = parse_datetime(position["entry_at"])
            exit_at = parse_datetime(position["exit_at"])
            if timestamp < entry_at:
                continue
            if timestamp >= exit_at:
                pnl += float(position["pnl"])
                continue
            mark = _marked_position_pnl(frame=frame, position=position, timestamp=timestamp, cost_rate=cost_rate)
            if mark is None:
                continue
            pnl += mark
            gross_exposure += abs(float(position["notional"]))
            net_exposure += float(position["notional"]) * float(position["side_multiplier"])
            active_count += 1

        equity = initial_cash + pnl
        curve.append(
            {
                "timestamp": timestamp.isoformat(),
                "equity": round(equity, 6),
                "pnl": round(pnl, 6),
                "gross_exposure": round(gross_exposure, 6),
                "net_exposure": round(net_exposure, 6),
                "active_position_count": active_count,
            }
        )
    return curve


def _marked_position_pnl(frame: Any, position: dict[str, Any], timestamp: datetime, cost_rate: float) -> float | None:
    ticker_frame = frame[
        (frame["_dean_ticker"] == position["ticker"])
        & (frame["_dean_datetime"] >= parse_datetime(position["entry_at"]))
        & (frame["_dean_datetime"] <= timestamp)
    ]
    if ticker_frame.empty:
        return None
    mark_price = float(ticker_frame.iloc[-1]["_dean_close"])
    entry_price = float(position["entry_price"])
    if entry_price == 0:
        return None
    gross_return = float(position["side_multiplier"]) * (mark_price / entry_price - 1.0)
    return float(position["notional"]) * (gross_return - 2 * cost_rate)


def _summary(
    records: list[PaperTradeRecord],
    positions: list[dict[str, Any]],
    skipped: list[dict[str, Any]],
    equity_curve: list[dict[str, Any]],
    initial_cash: float,
) -> dict[str, Any]:
    final_equity = equity_curve[-1]["equity"] if equity_curve else initial_cash
    total_pnl = final_equity - initial_cash
    total_return = total_pnl / initial_cash if initial_cash else 0.0
    max_drawdown = _max_drawdown([float(point["equity"]) for point in equity_curve])
    position_counts = Counter(position["status"] for position in positions)
    skipped_counts = Counter(item["status"] for item in skipped)
    return {
        "record_count": len(records),
        "position_count": len(positions),
        "open_position_count": position_counts.get("open", 0),
        "closed_position_count": position_counts.get("closed", 0),
        "skipped_count": len(skipped),
        "skipped_by_status": dict(sorted(skipped_counts.items())),
        "final_equity": round(final_equity, 6),
        "total_pnl": round(total_pnl, 6),
        "total_return": round(total_return, 8),
        "max_drawdown": round(max_drawdown, 8),
        "win_rate": _win_rate(positions),
        "pnl_by_agent": _sum_by(positions, "agent_name"),
        "pnl_by_action": _sum_by(positions, "action"),
        "pnl_by_regime_tag": _sum_by_tag(positions, "regime_tags"),
        "pnl_by_context_tag": _sum_by_tag(positions, "context_tags"),
    }


def _record_payload(record: PaperTradeRecord) -> dict[str, Any]:
    return {
        "trade_id": record.trade_id,
        "source_type": record.source_type,
        "source_id": record.source_id,
        "agent_name": record.agent_name,
        "action": record.action,
        "expected_direction": record.expected_direction,
        "horizon_days": record.horizon_days,
        "created_at": record.created_at,
        "confidence": record.confidence,
        "context_tags": record.context_tags,
        "regime_tags": record.regime_tags,
    }


def _record_tickers(record: PaperTradeRecord, requested_tickers: list[str]) -> list[str]:
    record_tickers = [str(ticker).upper() for ticker in record.tickers if str(ticker).strip()]
    if record_tickers and requested_tickers:
        return sorted(set(record_tickers).intersection(requested_tickers))
    return record_tickers or requested_tickers


def _side(record: PaperTradeRecord) -> int:
    if record.action == "candidate_short" or record.expected_direction == "bearish":
        return -1
    if record.expected_direction == "bullish":
        return 1
    return 0


def _max_drawdown(equity_values: list[float]) -> float:
    if not equity_values:
        return 0.0
    peak = equity_values[0]
    max_dd = 0.0
    for equity in equity_values:
        peak = max(peak, equity)
        if peak:
            max_dd = max(max_dd, (peak - equity) / peak)
    return max_dd


def _win_rate(positions: list[dict[str, Any]]) -> float | None:
    if not positions:
        return None
    wins = sum(1 for position in positions if float(position["net_return"]) > 0)
    return round(wins / len(positions), 6)


def _sum_by(positions: list[dict[str, Any]], key: str) -> dict[str, float]:
    totals: defaultdict[str, float] = defaultdict(float)
    for position in positions:
        totals[str(position.get(key, ""))] += float(position.get("pnl", 0.0))
    return {key: round(value, 6) for key, value in sorted(totals.items()) if key}


def _sum_by_tag(positions: list[dict[str, Any]], key: str) -> dict[str, float]:
    totals: defaultdict[str, float] = defaultdict(float)
    for position in positions:
        tags = position.get(key) or []
        for tag in tags:
            totals[str(tag)] += float(position.get("pnl", 0.0))
    return {tag: round(value, 6) for tag, value in sorted(totals.items())}


def _recommendations(summary: dict[str, Any]) -> list[str]:
    recommendations: list[str] = []
    if summary["record_count"] == 0:
        return ["No paper records found; record paper decisions before portfolio simulation."]
    if summary["position_count"] == 0:
        recommendations.append("No positions could be opened from current paper records and local price coverage.")
    if summary["skipped_by_status"].get("no_price_after_created_at"):
        recommendations.append("Load newer local market data before testing recently created paper decisions.")
    if summary["skipped_by_status"].get("as_of_before_record"):
        recommendations.append("Local market data as_of is earlier than at least one paper decision; refresh prices before simulation.")
    if summary["skipped_by_status"].get("no_price_before_as_of"):
        recommendations.append("The first usable entry price is after as_of for at least one record; rerun with newer data.")
    if summary["skipped_by_status"].get("watchlist_only"):
        recommendations.append("Use --include-watchlist only when watchlist ideas should be treated as paper positions.")
    if summary["skipped_by_status"].get("exceeds_gross_exposure_limit"):
        recommendations.append("Some positions were skipped due to gross exposure limits; consider increasing max_gross_exposure or reducing position_size_pct.")
    if summary["skipped_by_status"].get("exceeds_net_exposure_limit"):
        recommendations.append("Some positions were skipped due to net exposure limits; consider increasing max_net_exposure or reducing position_size_pct.")
    if summary["open_position_count"]:
        recommendations.append("Some paper positions are still open; treat PnL as mark-to-market, not final outcome.")
    if summary["max_drawdown"] > 0.1:
        recommendations.append("Paper drawdown exceeded 10%; review sizing, regime filters, and thesis quality before expanding autonomy.")
    if not recommendations:
        recommendations.append("Paper portfolio simulation completed; review PnL by agent/action/context before changing any process.")
    return recommendations
