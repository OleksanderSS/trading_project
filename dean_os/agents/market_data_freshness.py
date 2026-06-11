from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from dean_os.base import BaseAgent
from dean_os.schemas import MarketContext, PipelineReport
from dean_os.utils import clamp


class MarketDataFreshnessAgent(BaseAgent):
    """Checks whether local market price data is fresh enough for agent evaluation."""

    version = "0.1.0"
    branch = "pipeline"

    async def run(self, context: MarketContext) -> PipelineReport:
        max_age_hours = float(self.config.get("max_age_hours", 72.0))
        latest_interval = self.config.get("latest_processed_prices", "1d")
        market_data_path = self.config.get("market_data_path")
        tickers = self.config.get("tickers") or context.tickers
        as_of = _parse_datetime(self.config.get("as_of")) if self.config.get("as_of") else datetime.now(UTC)

        metrics = inspect_market_data_freshness(
            market_data_path=market_data_path,
            latest_processed_prices=latest_interval,
            tickers=tickers,
            as_of=as_of,
            max_age_hours=max_age_hours,
            close_col=self.config.get("close_col", "close"),
            datetime_col=self.config.get("datetime_col", "datetime"),
        )
        context.metadata.setdefault("data_freshness", {})["market_prices"] = metrics

        stale = bool(metrics.get("stale"))
        missing_tickers = metrics.get("missing_tickers", [])
        if metrics.get("status") == "unavailable":
            verdict = "caution"
            reasons = [metrics.get("reason", "Market price data unavailable.")]
            risks = ["Outcome evaluation and regime detection may be blocked until market data is refreshed."]
            signal_strength = -0.4
            quality_score = 0.25
        elif stale:
            verdict = "caution"
            reasons = [f"Market price data is stale: age {metrics.get('age_hours')}h > {max_age_hours}h"]
            if missing_tickers:
                reasons.append(f"Missing ticker rows: {', '.join(missing_tickers)}")
            risks = ["Pending learning records may remain unevaluable; regime context may be stale."]
            signal_strength = -0.3
            quality_score = 0.45
        else:
            verdict = "clear"
            reasons = ["Market price data freshness checks passed."]
            risks = []
            signal_strength = 0.3
            quality_score = 0.9

        return PipelineReport(
            agent_name=self.name,
            agent_version=self.version,
            verdict=verdict,
            confidence=0.85,
            data_quality_score=quality_score,
            signal_strength=signal_strength,
            reasons=reasons,
            risks=risks,
            blind_spots=["Freshness does not prove data correctness, adjusted-price quality, or publication-time integrity."],
            evidence=[
                self.evidence("file", str(metrics.get("market_data_path")), "market_data_path", metrics.get("market_data_path")),
                self.evidence("metric", "market_data_freshness", "status", metrics.get("status")),
                self.evidence("metric", "market_data_freshness", "age_hours", metrics.get("age_hours")),
                self.evidence("metric", "market_data_freshness", "missing_tickers", missing_tickers),
            ],
            input_hash=self.context_hash(context),
            metrics_snapshot=metrics,
        )


def inspect_market_data_freshness(
    market_data_path: str | Path | None = None,
    latest_processed_prices: str | None = "1d",
    tickers: list[str] | None = None,
    as_of: datetime | None = None,
    max_age_hours: float = 72.0,
    close_col: str = "close",
    datetime_col: str = "datetime",
) -> dict[str, Any]:
    as_of = as_of or datetime.now(UTC)
    path = _resolve_market_data_path(market_data_path, latest_processed_prices)
    if path is None:
        return {
            "status": "unavailable",
            "stale": True,
            "reason": "No market data path found.",
            "max_age_hours": max_age_hours,
        }
    if not path.exists():
        return {
            "status": "unavailable",
            "stale": True,
            "reason": f"Market data file does not exist: {path}",
            "market_data_path": str(path),
            "max_age_hours": max_age_hours,
        }

    try:
        import pandas as pd

        frame = _read_market_frame(pd, path)
        prepared = _prepare_frame(pd, frame, close_col=close_col, datetime_col=datetime_col)
    except Exception as exc:
        return {
            "status": "unavailable",
            "stale": True,
            "reason": f"Could not inspect market data: {type(exc).__name__}: {exc}",
            "market_data_path": str(path),
            "max_age_hours": max_age_hours,
        }

    latest_timestamp = prepared["_dean_datetime"].max().to_pydatetime()
    age_hours = (as_of - latest_timestamp).total_seconds() / 3600
    ticker_rows = _ticker_row_counts(prepared)
    requested_tickers = [ticker.upper() for ticker in tickers or [] if str(ticker).strip()]
    missing_tickers = [ticker for ticker in requested_tickers if ticker not in ticker_rows]
    stale = age_hours > max_age_hours or bool(missing_tickers)
    per_ticker_latest = _per_ticker_latest(prepared, requested_tickers or sorted(ticker_rows))

    return {
        "status": "stale" if stale else "fresh",
        "stale": stale,
        "market_data_path": str(path),
        "latest_processed_prices": latest_processed_prices,
        "as_of": as_of.isoformat(),
        "latest_timestamp": latest_timestamp.isoformat(),
        "age_hours": round(age_hours, 3),
        "max_age_hours": max_age_hours,
        "requested_tickers": requested_tickers,
        "available_ticker_count": len(ticker_rows),
        "missing_tickers": missing_tickers,
        "row_count": int(len(prepared)),
        "per_ticker_latest": per_ticker_latest,
        "freshness_score": clamp(1.0 - max(age_hours, 0.0) / max(max_age_hours, 1.0), 0.0, 1.0),
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


def _prepare_frame(pd: Any, frame: Any, close_col: str, datetime_col: str) -> Any:
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
    if prepared.empty:
        raise ValueError("No usable market rows after parsing close/datetime columns.")
    return prepared.sort_values("_dean_datetime")


def _ticker_row_counts(frame: Any) -> dict[str, int]:
    if "_dean_ticker" not in frame.columns:
        return {}
    counts = frame["_dean_ticker"].value_counts().to_dict()
    return {str(ticker): int(count) for ticker, count in counts.items() if str(ticker)}


def _per_ticker_latest(frame: Any, tickers: list[str]) -> dict[str, dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    for ticker in tickers:
        ticker_frame = frame[frame["_dean_ticker"] == ticker.upper()]
        if ticker_frame.empty:
            continue
        row = ticker_frame.iloc[-1]
        latest[ticker.upper()] = {
            "latest_timestamp": row["_dean_datetime"].isoformat(),
            "latest_close": float(row["_dean_close"]),
            "row_count": int(len(ticker_frame)),
        }
    return latest


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


def _parse_datetime(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)
