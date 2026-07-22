from __future__ import annotations

import json
import sqlite3
from collections import Counter
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import timedelta
from pathlib import Path
from typing import Any

from dean_os.draft.dean_os_agent_system_v7.dean_os.learning import classify_outcome
from dean_os.draft.dean_os_agent_system_v7.dean_os.market_data_api import parse_datetime, prepare_market_frame, read_market_frame
from dean_os.draft.dean_os_agent_system_v7.dean_os.outcome_evaluation import (
    _frame_latest_datetime,
    _resolve_market_data_path,
    _ticker_return,
)
from dean_os.draft.dean_os_agent_system_v7.dean_os.regime_context import normalize_context_tags
from dean_os.schemas import PaperTradeRecord


class PaperTradeStore:
    """Durable log of autonomous paper decisions and later outcomes."""

    def __init__(self, db_path: str | Path = "data/dean_os/paper_trades.sqlite"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def add_record(self, record: PaperTradeRecord) -> str:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO paper_trades
                (trade_id, source_type, source_id, agent_name, action, tickers, expected_direction,
                 horizon_days, thesis, confidence, context_tags, regime_tags, status, created_at,
                 outcome_at, realized_return, outcome_label, metadata, payload)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.trade_id,
                    record.source_type,
                    record.source_id,
                    record.agent_name,
                    record.action,
                    json.dumps(record.tickers, ensure_ascii=True),
                    record.expected_direction,
                    record.horizon_days,
                    record.thesis,
                    record.confidence,
                    json.dumps(record.context_tags, ensure_ascii=True),
                    json.dumps(record.regime_tags, ensure_ascii=True),
                    record.status,
                    record.created_at,
                    record.outcome_at,
                    record.realized_return,
                    record.outcome_label,
                    json.dumps(record.metadata, ensure_ascii=True),
                    json.dumps(record.model_dump(mode="json"), ensure_ascii=True),
                ),
            )
        return record.trade_id

    def get_record(self, trade_id: str) -> PaperTradeRecord | None:
        with self._connect() as conn:
            row = conn.execute("SELECT payload FROM paper_trades WHERE trade_id = ?", (trade_id,)).fetchone()
        if row is None:
            return None
        return PaperTradeRecord(**json.loads(row["payload"]))

    def list_records(self, status: str | None = None, agent_name: str | None = None) -> list[PaperTradeRecord]:
        clauses = []
        params: list[Any] = []
        if status:
            clauses.append("status = ?")
            params.append(status)
        if agent_name:
            clauses.append("agent_name = ?")
            params.append(agent_name)
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        with self._connect() as conn:
            rows = conn.execute(f"SELECT payload FROM paper_trades{where} ORDER BY rowid", tuple(params)).fetchall()
        return [PaperTradeRecord(**json.loads(row["payload"])) for row in rows]

    def update_outcome(
        self,
        trade_id: str,
        realized_return: float,
        outcome_at: str,
        neutral_band: float = 0.01,
        metadata: dict[str, Any] | None = None,
    ) -> PaperTradeRecord:
        record = self.get_record(trade_id)
        if record is None:
            raise KeyError(f"Paper trade record not found: {trade_id}")
        record.realized_return = realized_return
        record.outcome_label = classify_outcome(record.expected_direction, realized_return, neutral_band=neutral_band)
        record.outcome_at = outcome_at
        record.status = "evaluated"
        record.metadata = {**record.metadata, **(metadata or {})}
        self.add_record(record)
        return record

    def void_record(self, trade_id: str, reason: str) -> PaperTradeRecord:
        record = self.get_record(trade_id)
        if record is None:
            raise KeyError(f"Paper trade record not found: {trade_id}")
        record.status = "voided"
        record.metadata = {**record.metadata, "void_reason": reason}
        self.add_record(record)
        return record

    def summary(self) -> dict[str, Any]:
        records = self.list_records()
        evaluated = [record for record in records if record.outcome_label is not None]
        hits = sum(1 for record in evaluated if record.outcome_label == "hit")
        misses = sum(1 for record in evaluated if record.outcome_label == "miss")
        return {
            "record_count": len(records),
            "pending_count": sum(1 for record in records if record.status == "pending"),
            "evaluated_count": len(evaluated),
            "voided_count": sum(1 for record in records if record.status == "voided"),
            "hit_rate": hits / len(evaluated) if evaluated else None,
            "miss_rate": misses / len(evaluated) if evaluated else None,
            "records_by_status": dict(sorted(Counter(record.status for record in records).items())),
            "records_by_action": dict(sorted(Counter(record.action for record in records).items())),
            "records_by_agent": dict(sorted(Counter(record.agent_name for record in records).items())),
            "records_by_regime_tag": dict(
                sorted(Counter(tag for record in records for tag in record.regime_tags).items())
            ),
        }

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS paper_trades (
                    trade_id TEXT PRIMARY KEY,
                    source_type TEXT NOT NULL,
                    source_id TEXT NOT NULL,
                    agent_name TEXT NOT NULL,
                    action TEXT NOT NULL,
                    tickers TEXT NOT NULL,
                    expected_direction TEXT NOT NULL,
                    horizon_days INTEGER NOT NULL,
                    thesis TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    context_tags TEXT NOT NULL,
                    regime_tags TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    outcome_at TEXT,
                    realized_return REAL,
                    outcome_label TEXT,
                    metadata TEXT NOT NULL,
                    payload TEXT NOT NULL
                )
                """
            )

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()


class PaperTradeEvaluationRunner:
    """Evaluates pending paper decisions against local OHLCV data."""

    def __init__(self, store_path: str | Path = "data/dean_os/paper_trades.sqlite"):
        self.store_path = Path(store_path)

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
            raise RuntimeError(f"pandas is required for paper trade evaluation: {exc}") from exc

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
        as_of_dt = parse_datetime(as_of) if as_of else _frame_latest_datetime(frame)
        store = PaperTradeStore(self.store_path)
        records = store.list_records(status="pending")
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
            "paper_trade_store": str(self.store_path),
            "market_data_path": str(resolved_path),
            "as_of": as_of_dt.isoformat(),
            "allow_early": allow_early,
            "apply_updates": apply_updates,
            "pending_record_count": len(records),
            "updated_count": status_counts.get("updated", 0),
            "evaluable_count": status_counts.get("evaluable", 0),
            "status_counts": dict(sorted(status_counts.items())),
            "evaluations": evaluations,
            "summary_after": store.summary() if apply_updates else None,
            "recommendations": _paper_recommendations(status_counts),
        }

    def _evaluate_record(
        self,
        store: PaperTradeStore,
        record: PaperTradeRecord,
        frame: Any,
        tickers: list[str],
        as_of,
        allow_early: bool,
        apply_updates: bool,
        neutral_band: float,
    ) -> dict[str, Any]:
        record_tickers = _record_tickers(record, tickers)
        base_payload = {
            "trade_id": record.trade_id,
            "agent_name": record.agent_name,
            "source_type": record.source_type,
            "source_id": record.source_id,
            "action": record.action,
            "expected_direction": record.expected_direction,
            "horizon_days": record.horizon_days,
            "created_at": record.created_at,
            "tickers": record_tickers,
            "context_tags": record.context_tags,
            "regime_tags": record.regime_tags,
        }
        if not record_tickers:
            return {**base_payload, "status": "missing_tickers", "reason": "Paper record has no tickers."}

        start_at = parse_datetime(record.created_at)
        due_at = start_at + timedelta(days=record.horizon_days)
        latest_price_at = _frame_latest_datetime(frame)
        if latest_price_at < start_at:
            return {
                **base_payload,
                "status": "no_price_after_created_at",
                "reason": "Market data ends before the paper record was created.",
                "due_at": due_at.isoformat(),
                "latest_price_at": latest_price_at.isoformat(),
            }
        if not allow_early and as_of < due_at:
            return {
                **base_payload,
                "status": "not_due",
                "reason": "Paper record horizon has not elapsed.",
                "due_at": due_at.isoformat(),
                "latest_price_at": latest_price_at.isoformat(),
            }

        target_at = as_of if allow_early else min(as_of, due_at)
        ticker_results = [_ticker_return(frame=frame, ticker=ticker, start_at=start_at, target_at=target_at) for ticker in record_tickers]
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
        outcome_label = classify_outcome(record.expected_direction, realized_return, neutral_band=neutral_band)
        status = "evaluable"
        if apply_updates:
            store.update_outcome(
                record.trade_id,
                realized_return=realized_return,
                outcome_at=target_at.isoformat(),
                neutral_band=neutral_band,
                metadata={"ticker_results": ticker_results},
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


def create_paper_trade_record(
    action: str,
    tickers: list[str],
    expected_direction: str | None = None,
    source_type: str = "manual",
    source_id: str = "",
    agent_name: str = "chief_review",
    horizon_days: int = 30,
    thesis: str = "",
    confidence: float = 0.0,
    context_tags: list[str] | None = None,
    regime_tags: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
) -> PaperTradeRecord:
    return PaperTradeRecord(
        source_type=source_type,
        source_id=source_id,
        agent_name=agent_name,
        action=action,
        tickers=[ticker.upper() for ticker in tickers if str(ticker).strip()],
        expected_direction=expected_direction or direction_from_action(action),
        horizon_days=horizon_days,
        thesis=thesis,
        confidence=confidence,
        context_tags=normalize_context_tags(context_tags or []),
        regime_tags=normalize_context_tags(regime_tags or []),
        metadata=metadata or {},
    )


def direction_from_action(action: str) -> str:
    if action in {"candidate_long", "paper_trade_only", "watchlist"}:
        return "bullish"
    if action == "candidate_short":
        return "bearish"
    return "neutral"


def _record_tickers(record: PaperTradeRecord, fallback_tickers: list[str]) -> list[str]:
    metadata_tickers = [str(ticker).upper() for ticker in record.tickers if str(ticker).strip()]
    fallback = [str(ticker).upper() for ticker in fallback_tickers if str(ticker).strip()]
    if metadata_tickers and fallback:
        return sorted(set(metadata_tickers).intersection(fallback))
    return metadata_tickers or fallback


def _paper_recommendations(status_counts: Counter) -> list[str]:
    if not status_counts:
        return ["No pending paper records found."]
    recommendations = []
    if status_counts.get("no_price_after_created_at"):
        recommendations.append("Load newer market data before evaluating recently created paper decisions.")
    if status_counts.get("not_due"):
        recommendations.append("Keep pending records until their paper horizon elapses, or use --allow-early for diagnostics only.")
    if status_counts.get("missing_price_window"):
        recommendations.append("Check ticker/date coverage in the local OHLCV file.")
    if status_counts.get("evaluable"):
        recommendations.append("Review evaluable paper outcomes, then rerun with --apply after approval.")
    return recommendations or ["Paper trade evaluation completed."]
