from __future__ import annotations

from collections import Counter
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from dean_os.learning import LearningStore, classify_outcome
from dean_os.schemas import AgentLearningRecord


class OutcomeEvaluationRunner:
    """Evaluates pending learning records against local price data."""

    def __init__(self, learning_path: str | Path = "data/dean_os/agent_learning.sqlite"):
        self.learning_path = Path(learning_path)

    def evaluate(
        self,
        market_data_path: str | Path | None = None,
        latest_processed_prices: str | None = None,
        tickers: list[str] | None = None,
        as_of: str | None = None,
        close_col: str = "close",
        datetime_col: str = "datetime",
        allow_early: bool = False,
        apply_updates: bool = False,
        neutral_band: float = 0.01,
        limit: int | None = None,
    ) -> dict[str, Any]:
        try:
            import pandas as pd
        except Exception as exc:
            raise RuntimeError(f"pandas is required for outcome evaluation: {exc}") from exc

        resolved_path = _resolve_market_data_path(market_data_path, latest_processed_prices)
        if resolved_path is None:
            raise ValueError("Provide market_data_path or --latest-processed-prices.")
        if not resolved_path.exists():
            raise FileNotFoundError(f"Market data file does not exist: {resolved_path}")

        frame = _prepare_market_frame(
            pd=pd,
            frame=_read_market_frame(pd, resolved_path),
            close_col=close_col,
            datetime_col=datetime_col,
        )
        as_of_dt = _parse_datetime(as_of) if as_of else _frame_latest_datetime(frame)
        store = LearningStore(self.learning_path)
        records = [record for record in store.list_records() if record.outcome_label is None]
        if limit is not None:
            records = records[:limit]

        evaluations = [
            self._evaluate_record(
                store=store,
                record=record,
                frame=frame,
                tickers=tickers or [],
                as_of=as_of_dt,
                allow_early=allow_early,
                apply_updates=apply_updates,
                neutral_band=neutral_band,
            )
            for record in records
        ]

        status_counts = Counter(item["status"] for item in evaluations)
        return {
            "learning_store": str(self.learning_path),
            "market_data_path": str(resolved_path),
            "as_of": as_of_dt.isoformat(),
            "allow_early": allow_early,
            "apply_updates": apply_updates,
            "pending_record_count": len(records),
            "updated_count": status_counts.get("updated", 0),
            "evaluable_count": status_counts.get("evaluable", 0),
            "status_counts": dict(sorted(status_counts.items())),
            "evaluations": evaluations,
            "recommendations": _recommendations(status_counts),
        }

    def _evaluate_record(
        self,
        store: LearningStore,
        record: AgentLearningRecord,
        frame: Any,
        tickers: list[str],
        as_of: datetime,
        allow_early: bool,
        apply_updates: bool,
        neutral_band: float,
    ) -> dict[str, Any]:
        record_tickers = _record_tickers(record, tickers)
        base_payload = {
            "record_id": record.record_id,
            "agent_name": record.agent_name,
            "topic": record.metadata.get("topic", ""),
            "expected_direction": record.expected_direction,
            "horizon_days": record.horizon_days,
            "created_at": record.created_at,
            "tickers": record_tickers,
            "context_tags": record.metadata.get("context_tags", []),
            "regime_tags": record.metadata.get("regime_tags", []),
        }
        if not record_tickers:
            return {**base_payload, "status": "missing_tickers", "reason": "Learning record has no tickers."}

        start_at = _parse_datetime(record.created_at)
        due_at = start_at + timedelta(days=record.horizon_days)
        latest_price_at = _frame_latest_datetime(frame)
        if latest_price_at < start_at:
            return {
                **base_payload,
                "status": "no_price_after_created_at",
                "reason": "Market data ends before the learning record was created.",
                "due_at": due_at.isoformat(),
                "latest_price_at": latest_price_at.isoformat(),
            }
        if not allow_early and as_of < due_at:
            return {
                **base_payload,
                "status": "not_due",
                "reason": "Learning record horizon has not elapsed.",
                "due_at": due_at.isoformat(),
                "latest_price_at": latest_price_at.isoformat(),
            }

        target_at = as_of if allow_early else min(as_of, due_at)
        ticker_results = [
            _ticker_return(frame=frame, ticker=ticker, start_at=start_at, target_at=target_at)
            for ticker in record_tickers
        ]
        valid_results = [item for item in ticker_results if item["status"] == "ok"]
        if not valid_results:
            return {
                **base_payload,
                "status": "missing_price_window",
                "reason": "Could not find usable start/end prices for record tickers.",
                "due_at": due_at.isoformat(),
                "target_at": target_at.isoformat(),
                "ticker_results": ticker_results,
            }

        realized_return = sum(item["realized_return"] for item in valid_results) / len(valid_results)
        outcome_label = classify_outcome(
            record.expected_direction,
            realized_return,
            neutral_band=neutral_band,
        )
        status = "evaluable"
        if apply_updates:
            store.update_outcome(
                record.record_id,
                realized_return=realized_return,
                outcome_at=target_at.isoformat(),
                neutral_band=neutral_band,
            )
            status = "updated"

        return {
            **base_payload,
            "status": status,
            "due_at": due_at.isoformat(),
            "target_at": target_at.isoformat(),
            "realized_return": realized_return,
            "outcome_label": outcome_label,
            "ticker_results": ticker_results,
        }


def _resolve_market_data_path(raw_path: str | Path | None, latest_interval: str | None) -> Path | None:
    if raw_path:
        return Path(raw_path)
    if not latest_interval:
        return None
    candidates = sorted(
        Path("data/processed").glob(f"prices_{latest_interval}_*.parquet"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def _read_market_frame(pd: Any, path: Path) -> Any:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported market data file type: {path.suffix}. Use .csv or .parquet.")


def _prepare_market_frame(pd: Any, frame: Any, close_col: str, datetime_col: str) -> Any:
    close_col = _resolve_column(frame, close_col)
    datetime_col = _resolve_column(frame, datetime_col)
    if close_col not in frame.columns:
        raise ValueError(f"Missing close column: {close_col}")
    if datetime_col not in frame.columns:
        raise ValueError(f"Missing datetime column: {datetime_col}")
    prepared = frame.copy()
    prepared["_dean_close"] = pd.to_numeric(prepared[close_col], errors="coerce")
    prepared["_dean_datetime"] = pd.to_datetime(prepared[datetime_col], utc=True, errors="coerce")
    ticker_col = _first_existing_column(prepared, ["ticker", "symbol", "Ticker", "Symbol"])
    prepared["_dean_ticker"] = prepared[ticker_col].astype(str).str.upper() if ticker_col else ""
    prepared = prepared.dropna(subset=["_dean_close", "_dean_datetime"])
    return prepared.sort_values("_dean_datetime")


def _ticker_return(frame: Any, ticker: str, start_at: datetime, target_at: datetime) -> dict[str, Any]:
    ticker_frame = frame[frame["_dean_ticker"] == ticker.upper()]
    if ticker_frame.empty:
        return {"ticker": ticker.upper(), "status": "missing_ticker"}
    start_candidates = ticker_frame[ticker_frame["_dean_datetime"] >= start_at]
    end_candidates = ticker_frame[(ticker_frame["_dean_datetime"] <= target_at) & (ticker_frame["_dean_datetime"] >= start_at)]
    if start_candidates.empty:
        return {"ticker": ticker.upper(), "status": "missing_start_price"}
    if end_candidates.empty:
        return {"ticker": ticker.upper(), "status": "missing_end_price"}
    start_row = start_candidates.iloc[0]
    end_row = end_candidates.iloc[-1]
    start_price = float(start_row["_dean_close"])
    end_price = float(end_row["_dean_close"])
    if start_price == 0:
        return {"ticker": ticker.upper(), "status": "zero_start_price"}
    return {
        "ticker": ticker.upper(),
        "status": "ok",
        "start_at": start_row["_dean_datetime"].isoformat(),
        "end_at": end_row["_dean_datetime"].isoformat(),
        "start_price": start_price,
        "end_price": end_price,
        "realized_return": end_price / start_price - 1.0,
    }


def _record_tickers(record: AgentLearningRecord, fallback_tickers: list[str]) -> list[str]:
    metadata_tickers = [str(ticker).upper() for ticker in record.metadata.get("tickers", []) if str(ticker).strip()]
    fallback = [str(ticker).upper() for ticker in fallback_tickers if str(ticker).strip()]
    if metadata_tickers and fallback:
        return sorted(set(metadata_tickers).intersection(fallback))
    return metadata_tickers or fallback


def _frame_latest_datetime(frame: Any) -> datetime:
    value = frame["_dean_datetime"].max()
    return value.to_pydatetime()


def _parse_datetime(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _resolve_column(frame: Any, requested: str) -> str:
    if requested in frame.columns:
        return requested
    lowered = {str(column).lower(): column for column in frame.columns}
    return lowered.get(requested.lower(), requested)


def _first_existing_column(frame: Any, candidates: list[str]) -> str | None:
    lowered = {str(column).lower(): column for column in frame.columns}
    for candidate in candidates:
        if candidate in frame.columns:
            return candidate
        if candidate.lower() in lowered:
            return lowered[candidate.lower()]
    return None


def _recommendations(status_counts: Counter) -> list[str]:
    if not status_counts:
        return ["No pending learning records found."]
    recommendations = []
    if status_counts.get("no_price_after_created_at"):
        recommendations.append("Load newer market data before evaluating recently created learning records.")
    if status_counts.get("not_due"):
        recommendations.append("Keep not_due records pending until their configured horizon elapses.")
    if status_counts.get("missing_tickers"):
        recommendations.append("Ensure future learning records include tickers in metadata.")
    if status_counts.get("evaluable"):
        recommendations.append("Review evaluable dry-run outcomes, then rerun with --apply if they are acceptable.")
    if status_counts.get("updated"):
        recommendations.append("Run context performance again to refresh hit/miss guardrails.")
    if not recommendations:
        recommendations.append("No actionable outcome updates detected.")
    return recommendations
