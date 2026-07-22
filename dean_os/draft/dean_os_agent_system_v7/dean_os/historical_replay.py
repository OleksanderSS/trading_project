from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from dean_os.base import AnalyticalAgent
from dean_os.draft.dean_os_agent_system_v7.dean_os.learning import classify_outcome
from dean_os.draft.dean_os_agent_system_v7.dean_os.outcome_evaluation import (
    _parse_datetime,
    _prepare_market_frame,
    _read_market_frame,
    _ticker_return,
)
from dean_os.draft.dean_os_agent_system_v7.dean_os.regime_context import RegimeContextBuilder, normalize_context_tags
from dean_os.schemas import AnalyticalReport, MarketContext, ResearchNote, utc_now_iso
from dean_os.utils import clamp, json_ready

LEAKY_COLUMN_TOKENS = (
    "target",
    "future",
    "_after_",
    "after_",
    "outcome",
    "realized",
    "label",
    "prediction",
    "predicted",
)


@dataclass(frozen=True)
class ReplayDataGuardResult:
    safe_frame: Any
    removed_columns: list[str]
    retained_columns: list[str]

    def summary(self) -> dict[str, Any]:
        return {
            "removed_column_count": len(self.removed_columns),
            "removed_columns": self.removed_columns,
            "retained_column_count": len(self.retained_columns),
            "retained_columns_sample": self.retained_columns[:30],
        }


class HistoricalReplayAnalyst(AnalyticalAgent):
    """Deterministic replay analyst for old-data thesis checks.

    This is intentionally not an LLM. It gives us a repeatable baseline before
    we let more expressive agents into the replay exam.
    """

    version = "0.1.0"

    async def run(self, context: MarketContext) -> AnalyticalReport:
        replay = context.metadata.get("historical_replay", {})
        rankings = replay.get("rankings", [])
        top = rankings[0] if rankings else None
        if not top:
            return AnalyticalReport(
                agent_name=self.name,
                agent_version=self.version,
                verdict="needs_more_data",
                confidence=0.2,
                data_quality_score=0.2,
                signal_strength=0.0,
                ticker=None,
                asset_or_sector="historical_replay",
                horizon_years=_horizon_years(replay.get("horizon_days", 30)),
                thesis="No replay candidate could be formed from the historical snapshot.",
                data_quality="weak",
                position_bias="insufficient_data",
                reasons=["Historical snapshot has no usable ticker metrics."],
                risks=["Replay analyst cannot evaluate missing price coverage."],
                blind_spots=["No LLM, filings, or full fundamentals are used in this baseline replay analyst."],
                evidence=[],
                input_hash=self.context_hash(context),
            )

        direction = _direction_from_score(float(top["score"]))
        verdict = "bullish" if direction == "bullish" else "bearish" if direction == "bearish" else "neutral"
        confidence = clamp(0.35 + abs(float(top["score"])) * 0.9, 0.0, 0.88)
        data_quality = _data_quality_from_coverage(replay.get("coverage", {}), replay.get("news_summary", {}))
        signal_strength = float(top["score"]) if direction != "neutral" else 0.0
        thesis = _build_replay_thesis(top, direction, replay)

        return AnalyticalReport(
            agent_name=self.name,
            agent_version=self.version,
            verdict=verdict,
            confidence=confidence,
            data_quality_score={"strong": 0.82, "partial": 0.58, "weak": 0.28}[data_quality],
            signal_strength=clamp(signal_strength, -1.0, 1.0),
            ticker=top["ticker"],
            asset_or_sector="historical_replay",
            horizon_years=_horizon_years(replay.get("horizon_days", 30)),
            thesis=thesis,
            data_quality=data_quality,
            position_bias=direction if direction in {"bullish", "bearish", "neutral"} else "insufficient_data",
            catalysts=_replay_catalysts(top, replay),
            tailwinds=top.get("tailwinds", []),
            headwinds=top.get("headwinds", []),
            watchlist_score=clamp(abs(float(top["score"])) + 0.25, 0.0, 1.0),
            reasons=[
                f"Top replay score: {top['ticker']}={float(top['score']):.3f}",
                f"Lookback return: {float(top.get('lookback_return', 0.0)):.3f}",
                f"Relative return vs benchmark: {float(top.get('relative_return', 0.0)):.3f}",
            ],
            risks=[
                "HistoricalReplayAnalyst is a deterministic baseline, not an investment decision maker.",
                "Replay evaluation must keep future prices hidden from thesis formation.",
            ],
            blind_spots=[
                "No live broker, no heavy pipeline run, no LLM reasoning, and no complete fundamentals in this replay layer.",
            ],
            evidence=[
                self.evidence("metric", "historical_replay.rankings", "top_candidate", top),
                self.evidence("metric", "historical_replay", "as_of", replay.get("as_of")),
                self.evidence("metric", "historical_replay", "lookback_days", replay.get("lookback_days")),
                self.evidence("metric", "historical_replay", "horizon_days", replay.get("horizon_days")),
            ],
            input_hash=self.context_hash(context),
        )


class HistoricalReplayRunner:
    """Runs an old-data replay without starting the trading pipeline."""

    def __init__(self, output_dir: str | Path = "reports/dean_os/historical_replay"):
        self.output_dir = Path(output_dir)

    async def run(
        self,
        price_data_path: str | Path,
        tickers: list[str],
        as_of: str,
        lookback_days: int = 180,
        horizon_days: int = 30,
        news_data_path: str | Path | None = None,
        macro_data_path: str | Path | None = None,
        benchmark_ticker: str = "SPY",
        close_col: str = "close",
        datetime_col: str = "datetime",
        neutral_band: float = 0.01,
        max_news_items: int = 80,
        normalize_daily_bars: bool = False,
    ) -> dict[str, Any]:
        try:
            import pandas as pd
        except Exception as exc:
            raise RuntimeError(f"pandas is required for historical replay: {exc}") from exc

        as_of_dt = _parse_datetime(as_of)
        lookback_start = as_of_dt - timedelta(days=lookback_days)
        horizon_at = as_of_dt + timedelta(days=horizon_days)
        price_path = Path(price_data_path)
        if not price_path.exists():
            raise FileNotFoundError(f"Price data file does not exist: {price_path}")

        raw_prices = _read_market_frame(pd, price_path)
        price_guard = guard_replay_frame(raw_prices, required_columns=[datetime_col, "ticker", close_col])
        full_prices = _prepare_market_frame(
            pd=pd,
            frame=price_guard.safe_frame,
            close_col=close_col,
            datetime_col=datetime_col,
        )
        daily_normalization = {"applied": False}
        if normalize_daily_bars:
            full_prices, daily_normalization = _normalize_daily_bars(pd, full_prices)
        requested_tickers = [ticker.upper() for ticker in tickers if str(ticker).strip()]
        snapshot_prices = _filter_price_snapshot(full_prices, requested_tickers, lookback_start, as_of_dt)
        rankings = _rank_tickers(snapshot_prices, requested_tickers, benchmark_ticker=benchmark_ticker)
        coverage = _coverage_summary(
            snapshot_prices,
            full_prices,
            requested_tickers,
            lookback_start,
            as_of_dt,
            horizon_at,
        )
        news_payload = _load_news_snapshot(
            pd=pd,
            path=Path(news_data_path) if news_data_path else None,
            tickers=requested_tickers,
            start_at=lookback_start,
            as_of=as_of_dt,
            max_items=max_news_items,
        )
        macro_payload = _load_macro_snapshot(
            pd=pd,
            path=Path(macro_data_path) if macro_data_path else None,
            start_at=lookback_start,
            as_of=as_of_dt,
        )
        regime = _build_regime(snapshot_prices, benchmark_ticker, requested_tickers)
        context = MarketContext(
            tickers=requested_tickers,
            timeframe="1d",
            timeframes=["1d"],
            dataframes={"prices": snapshot_prices},
            news=news_payload["items"],
            macro=macro_payload["latest_by_series"],
            metadata={
                "historical_replay": {
                    "as_of": as_of_dt.isoformat(),
                    "lookback_start": lookback_start.isoformat(),
                    "lookback_days": lookback_days,
                    "horizon_days": horizon_days,
                    "horizon_at": horizon_at.isoformat(),
                    "price_data_path": str(price_path),
                    "news_data_path": str(news_data_path) if news_data_path else None,
                    "macro_data_path": str(macro_data_path) if macro_data_path else None,
                    "benchmark_ticker": benchmark_ticker.upper(),
                    "daily_normalization": daily_normalization,
                    "coverage": coverage,
                    "rankings": rankings,
                    "news_summary": news_payload["summary"],
                    "macro_summary": macro_payload["summary"],
                    "data_guard": {
                        "prices": price_guard.summary(),
                        "news": news_payload["data_guard"],
                        "macro": macro_payload["data_guard"],
                    },
                    "regime_context": json_ready(regime),
                }
            },
        )
        analyst = HistoricalReplayAnalyst(name="historical_replay", config={})
        report = await analyst.run(context)
        note = _note_from_report(report, context)
        evaluation = _evaluate_replay_report(
            report=report,
            full_prices=full_prices,
            as_of=as_of_dt,
            horizon_at=horizon_at,
            neutral_band=neutral_band,
        )
        payload = {
            "run_id": "historical_replay_" + utc_now_iso().replace(":", "").replace("-", "").replace(".", "_"),
            "created_at": utc_now_iso(),
            "mode": "historical_replay",
            "inputs": {
                "price_data_path": str(price_path),
                "news_data_path": str(news_data_path) if news_data_path else None,
                "macro_data_path": str(macro_data_path) if macro_data_path else None,
                "tickers": requested_tickers,
                "as_of": as_of_dt.isoformat(),
                "lookback_days": lookback_days,
                "horizon_days": horizon_days,
                "benchmark_ticker": benchmark_ticker.upper(),
                "normalize_daily_bars": normalize_daily_bars,
            },
            "decision": _decision_from_report(report),
            "report": report.model_dump(mode="json"),
            "research_note": note.model_dump(mode="json"),
            "historical_replay": context.metadata["historical_replay"],
            "evaluation": evaluation,
            "recommendations": _replay_recommendations(context.metadata["historical_replay"], evaluation),
        }
        saved_paths = self.save_report(payload)
        payload["saved_paths"] = {key: str(value) for key, value in saved_paths.items()}
        return payload

    def save_report(self, payload: dict[str, Any]) -> dict[str, Path]:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        run_id = payload["run_id"]
        json_path = self.output_dir / f"{run_id}.json"
        md_path = self.output_dir / f"{run_id}.md"
        latest_json = self.output_dir / "latest.json"
        latest_md = self.output_dir / "latest.md"
        paths = {"json": json_path, "markdown": md_path, "latest_json": latest_json, "latest_markdown": latest_md}
        payload["saved_paths"] = {key: str(value) for key, value in paths.items()}
        rendered_json = json.dumps(json_ready(payload), indent=2, ensure_ascii=False)
        rendered_md = render_historical_replay_markdown(payload)
        json_path.write_text(rendered_json, encoding="utf-8")
        latest_json.write_text(rendered_json, encoding="utf-8")
        md_path.write_text(rendered_md, encoding="utf-8")
        latest_md.write_text(rendered_md, encoding="utf-8")
        return paths


def guard_replay_frame(frame: Any, required_columns: list[str] | None = None) -> ReplayDataGuardResult:
    required = {column.lower() for column in required_columns or []}
    removed: list[str] = []
    retained: list[str] = []
    for column in frame.columns:
        lower = str(column).lower()
        if lower in required or not _is_leaky_column(lower):
            retained.append(column)
        else:
            removed.append(column)
    return ReplayDataGuardResult(safe_frame=frame.loc[:, retained].copy(), removed_columns=[str(c) for c in removed], retained_columns=[str(c) for c in retained])


def render_historical_replay_markdown(payload: dict[str, Any]) -> str:
    replay = payload.get("historical_replay", {})
    evaluation = payload.get("evaluation", {})
    report = payload.get("report", {})
    decision = payload.get("decision", {})
    lines = [
        "# DEAN-OS Historical Replay",
        "",
        f"- Run ID: `{payload.get('run_id')}`",
        f"- As of: `{payload.get('inputs', {}).get('as_of')}`",
        f"- Lookback days: {payload.get('inputs', {}).get('lookback_days')}",
        f"- Horizon days: {payload.get('inputs', {}).get('horizon_days')}",
        f"- Decision: `{decision.get('action')}`",
        f"- Ticker: `{decision.get('ticker')}`",
        f"- Confidence: {decision.get('confidence')}",
        "",
        "## Thesis",
        "",
        report.get("thesis", ""),
        "",
        "## Evaluation",
        "",
        f"- Status: `{evaluation.get('status')}`",
        f"- Outcome: `{evaluation.get('outcome_label')}`",
        f"- Realized return: {evaluation.get('realized_return')}",
        "",
        "## Top Rankings",
        "",
    ]
    for item in replay.get("rankings", [])[:10]:
        lines.append(f"- `{item.get('ticker')}` score={item.get('score')} lookback={item.get('lookback_return')} relative={item.get('relative_return')}")
    lines.extend(["", "## Guardrails", ""])
    guard = replay.get("data_guard", {})
    for name, summary in guard.items():
        lines.append(f"- `{name}` removed columns: {summary.get('removed_column_count', 0)}")
    lines.extend(["", "## Recommendations", ""])
    for recommendation in payload.get("recommendations", []):
        lines.append(f"- {recommendation}")
    return "\n".join(lines).strip() + "\n"


def _is_leaky_column(lower_name: str) -> bool:
    return any(token in lower_name for token in LEAKY_COLUMN_TOKENS)


def _filter_price_snapshot(frame: Any, tickers: list[str], start_at: datetime, as_of: datetime) -> Any:
    filtered = frame[(frame["_dean_datetime"] >= start_at) & (frame["_dean_datetime"] <= as_of)]
    if tickers:
        filtered = filtered[filtered["_dean_ticker"].isin(tickers)]
    return filtered.copy()


def _rank_tickers(frame: Any, tickers: list[str], benchmark_ticker: str = "SPY") -> list[dict[str, Any]]:
    benchmark_return = _ticker_lookback_return(frame, benchmark_ticker)
    rankings = []
    for ticker in tickers:
        ticker_frame = frame[frame["_dean_ticker"] == ticker.upper()].sort_values("_dean_datetime")
        if len(ticker_frame) < 5:
            continue
        close = ticker_frame["_dean_close"]
        lookback_return = _series_return(close)
        return_20 = _window_return(close, min(20, len(close)))
        return_60 = _window_return(close, min(60, len(close)))
        volatility = float(close.pct_change(fill_method=None).dropna().std() or 0.0)
        drawdown = _max_drawdown(close)
        volume_ratio = _volume_ratio(ticker_frame)
        relative_return = lookback_return - benchmark_return if benchmark_return is not None and ticker != benchmark_ticker.upper() else 0.0
        score = _ticker_score(
            lookback_return=lookback_return,
            return_20=return_20,
            return_60=return_60,
            volatility=volatility,
            drawdown=drawdown,
            relative_return=relative_return,
            volume_ratio=volume_ratio,
        )
        tailwinds, headwinds = _score_reasons(lookback_return, return_20, return_60, relative_return, drawdown, volume_ratio)
        rankings.append(
            {
                "ticker": ticker.upper(),
                "score": round(score, 6),
                "lookback_return": round(lookback_return, 6),
                "return_20": round(return_20, 6),
                "return_60": round(return_60, 6),
                "relative_return": round(relative_return, 6),
                "volatility": round(volatility, 6),
                "max_drawdown": round(drawdown, 6),
                "volume_ratio": round(volume_ratio, 6) if volume_ratio is not None else None,
                "row_count": int(len(ticker_frame)),
                "start_at": ticker_frame["_dean_datetime"].iloc[0].isoformat(),
                "end_at": ticker_frame["_dean_datetime"].iloc[-1].isoformat(),
                "tailwinds": tailwinds,
                "headwinds": headwinds,
            }
        )
    return sorted(rankings, key=lambda item: item["score"], reverse=True)


def _coverage_summary(frame: Any, full_frame: Any, tickers: list[str], start_at: datetime, as_of: datetime, horizon_at: datetime) -> dict[str, Any]:
    per_ticker = {}
    for ticker in tickers:
        ticker_frame = frame[frame["_dean_ticker"] == ticker.upper()]
        future_frame = full_frame[
            (full_frame["_dean_ticker"] == ticker.upper())
            & (full_frame["_dean_datetime"] >= as_of)
            & (full_frame["_dean_datetime"] <= horizon_at)
        ]
        per_ticker[ticker.upper()] = {
            "snapshot_rows": int(len(ticker_frame)),
            "future_rows_for_evaluation": int(len(future_frame)),
            "snapshot_start": ticker_frame["_dean_datetime"].min().isoformat() if len(ticker_frame) else None,
            "snapshot_end": ticker_frame["_dean_datetime"].max().isoformat() if len(ticker_frame) else None,
            "future_end": future_frame["_dean_datetime"].max().isoformat() if len(future_frame) else None,
        }
    return {
        "requested_ticker_count": len(tickers),
        "snapshot_row_count": int(len(frame)),
        "lookback_start": start_at.isoformat(),
        "as_of": as_of.isoformat(),
        "horizon_at": horizon_at.isoformat(),
        "per_ticker": per_ticker,
        "price_quality": _price_quality_summary(frame, tickers),
    }


def _load_news_snapshot(pd: Any, path: Path | None, tickers: list[str], start_at: datetime, as_of: datetime, max_items: int) -> dict[str, Any]:
    if path is None:
        return {"items": [], "summary": {"available": False, "row_count": 0}, "data_guard": {"removed_column_count": 0, "removed_columns": []}}
    if not path.exists():
        return {"items": [], "summary": {"available": False, "row_count": 0, "error": f"Missing news path: {path}"}, "data_guard": {"removed_column_count": 0, "removed_columns": []}}
    raw = _read_table(pd, path)
    guard = guard_replay_frame(raw)
    frame = guard.safe_frame
    date_col = _first_column(frame, ["published_date", "publishedAt", "timestamp", "date", "datetime", "news_timestamp"])
    if date_col is None:
        return {"items": [], "summary": {"available": True, "row_count": len(frame), "error": "No news datetime column found."}, "data_guard": guard.summary()}
    frame = frame.copy()
    frame["_dean_news_datetime"] = pd.to_datetime(frame[date_col], utc=True, errors="coerce")
    frame = frame[(frame["_dean_news_datetime"] >= start_at) & (frame["_dean_news_datetime"] <= as_of)]
    frame = frame.sort_values("_dean_news_datetime", ascending=False)
    items = [_news_record(row) for _, row in frame.head(max_items).iterrows()]
    ticker_hits = {ticker: _count_news_ticker_hits(frame, ticker) for ticker in tickers}
    source_col = _first_column(frame, ["source", "news_source"])
    source_counts = frame[source_col].astype(str).value_counts().head(10).to_dict() if source_col else {}
    return {
        "items": items,
        "summary": {
            "available": True,
            "path": str(path),
            "row_count": int(len(frame)),
            "date_column": str(date_col),
            "start": frame["_dean_news_datetime"].min().isoformat() if len(frame) else None,
            "end": frame["_dean_news_datetime"].max().isoformat() if len(frame) else None,
            "ticker_hits": ticker_hits,
            "source_counts": source_counts,
        },
        "data_guard": guard.summary(),
    }


def _load_macro_snapshot(pd: Any, path: Path | None, start_at: datetime, as_of: datetime) -> dict[str, Any]:
    if path is None:
        return {"latest_by_series": {}, "summary": {"available": False, "row_count": 0}, "data_guard": {"removed_column_count": 0, "removed_columns": []}}
    if not path.exists():
        return {"latest_by_series": {}, "summary": {"available": False, "row_count": 0, "error": f"Missing macro path: {path}"}, "data_guard": {"removed_column_count": 0, "removed_columns": []}}
    raw = _read_table(pd, path)
    guard = guard_replay_frame(raw)
    frame = guard.safe_frame
    date_col = _first_column(frame, ["date", "datetime", "timestamp", "realtime_start"])
    if date_col is None:
        return {"latest_by_series": {}, "summary": {"available": True, "row_count": len(frame), "error": "No macro datetime column found."}, "data_guard": guard.summary()}
    frame = frame.copy()
    frame["_dean_macro_datetime"] = pd.to_datetime(frame[date_col], utc=True, errors="coerce")
    frame = frame[(frame["_dean_macro_datetime"] >= start_at) & (frame["_dean_macro_datetime"] <= as_of)]
    series_col = _first_column(frame, ["series_id", "series", "indicator"])
    latest_by_series: dict[str, Any] = {}
    if series_col and "value" in frame.columns:
        for series, group in frame.sort_values("_dean_macro_datetime").groupby(series_col):
            row = group.iloc[-1]
            latest_by_series[str(series)] = {
                "value": row.get("value"),
                "date": row["_dean_macro_datetime"].isoformat(),
            }
    return {
        "latest_by_series": latest_by_series,
        "summary": {
            "available": True,
            "path": str(path),
            "row_count": int(len(frame)),
            "date_column": str(date_col),
            "series_count": len(latest_by_series),
            "series": sorted(latest_by_series.keys()),
        },
        "data_guard": guard.summary(),
    }


def _build_regime(frame: Any, benchmark_ticker: str, tickers: list[str]):
    ticker = benchmark_ticker.upper() if benchmark_ticker.upper() in tickers else (tickers[0] if tickers else "")
    ticker_frame = frame[frame["_dean_ticker"] == ticker].sort_values("_dean_datetime")
    if ticker_frame.empty:
        return RegimeContextBuilder().from_analyzer_result({"regime": "UNKNOWN"}, source="historical_replay")
    return RegimeContextBuilder().from_price_frame(ticker_frame, close_col="_dean_close", volume_col="volume" if "volume" in ticker_frame.columns else None)


def _note_from_report(report: AnalyticalReport, context: MarketContext) -> ResearchNote:
    replay = context.metadata.get("historical_replay", {})
    return ResearchNote(
        agent_name=report.agent_name,
        topic=f"historical_replay:{report.ticker or 'none'}:{replay.get('as_of')}",
        thesis=report.thesis,
        patterns=normalize_context_tags(["historical_replay", report.asset_or_sector or "", report.position_bias]),
        catalysts=report.catalysts,
        tailwinds=report.tailwinds,
        headwinds=report.headwinds,
        tickers=[report.ticker] if report.ticker else [],
        horizon_days=int(replay.get("horizon_days", 30)),
        confidence=report.confidence,
        data_quality=report.data_quality,
        evidence=report.evidence,
        risks=report.risks,
        blind_spots=report.blind_spots,
    )


def _evaluate_replay_report(report: AnalyticalReport, full_prices: Any, as_of: datetime, horizon_at: datetime, neutral_band: float) -> dict[str, Any]:
    if not report.ticker or report.position_bias not in {"bullish", "bearish", "neutral"}:
        return {"status": "no_candidate", "reason": "Replay report did not produce an evaluable ticker/direction."}
    target = _ticker_return(full_prices, ticker=report.ticker, start_at=as_of, target_at=horizon_at)
    if target.get("status") != "ok":
        return {"status": "missing_price_window", "ticker": report.ticker, "ticker_result": target}
    expected_direction = report.position_bias
    realized_return = float(target["realized_return"])
    return {
        "status": "evaluated",
        "ticker": report.ticker,
        "expected_direction": expected_direction,
        "start_at": target["start_at"],
        "target_at": target["end_at"],
        "requested_target_at": horizon_at.isoformat(),
        "start_price": target["start_price"],
        "end_price": target["end_price"],
        "realized_return": realized_return,
        "outcome_label": classify_outcome(expected_direction, realized_return, neutral_band=neutral_band),
        "neutral_band": neutral_band,
    }


def _decision_from_report(report: AnalyticalReport) -> dict[str, Any]:
    if report.position_bias == "bullish" and report.confidence >= 0.45:
        action = "candidate_long"
    elif report.position_bias == "bearish" and report.confidence >= 0.45:
        action = "candidate_short"
    elif report.position_bias == "neutral":
        action = "watchlist"
    else:
        action = "needs_more_data"
    return {
        "action": action,
        "ticker": report.ticker,
        "expected_direction": report.position_bias,
        "confidence": report.confidence,
        "reason": report.thesis,
        "paper_trade_created": False,
    }


def _read_table(pd: Any, path: Path) -> Any:
    from dean_os.draft.dean_os_agent_system_v7.dean_os.dean_paths import DeanPaths

    try:
        return DeanPaths.load_data_file(path)
    except Exception as exc:
        raise ValueError(f"Failed to load table from {path}: {exc}")


def _first_column(frame: Any, candidates: list[str]) -> str | None:
    lowered = {str(column).lower(): column for column in frame.columns}
    for candidate in candidates:
        if candidate in frame.columns:
            return candidate
        if candidate.lower() in lowered:
            return lowered[candidate.lower()]
    return None


def _news_record(row: Any) -> dict[str, Any]:
    return {
        "title": str(row.get("title") or row.get("headline") or ""),
        "source": str(row.get("source") or row.get("news_source") or ""),
        "published_at": row.get("_dean_news_datetime").isoformat() if row.get("_dean_news_datetime") is not None else None,
        "sentiment": _safe_float(row.get("sentiment") or row.get("news_sentiment"), default=0.0),
        "content": str(row.get("content") or row.get("description") or "")[:500],
        "url": row.get("link") or row.get("url"),
    }


def _count_news_ticker_hits(frame: Any, ticker: str) -> int:
    text_cols = [col for col in ["title", "headline", "content", "description", "search_term"] if col in frame.columns]
    if not text_cols:
        return 0
    ticker_lower = ticker.lower()
    count = 0
    for col in text_cols:
        count += int(frame[col].fillna("").astype(str).str.lower().str.contains(ticker_lower, regex=False).sum())
    return count


def _ticker_lookback_return(frame: Any, ticker: str) -> float | None:
    ticker_frame = frame[frame["_dean_ticker"] == ticker.upper()].sort_values("_dean_datetime")
    if len(ticker_frame) < 2:
        return None
    return _series_return(ticker_frame["_dean_close"])


def _series_return(series: Any) -> float:
    if len(series) < 2:
        return 0.0
    start = float(series.iloc[0])
    end = float(series.iloc[-1])
    return end / start - 1.0 if start else 0.0


def _window_return(series: Any, window: int) -> float:
    if len(series) < 2:
        return 0.0
    window = min(max(window, 2), len(series))
    start = float(series.iloc[-window])
    end = float(series.iloc[-1])
    return end / start - 1.0 if start else 0.0


def _max_drawdown(series: Any) -> float:
    if len(series) < 2:
        return 0.0
    running_max = series.cummax()
    drawdown = series / running_max - 1.0
    return float(drawdown.min())


def _volume_ratio(frame: Any) -> float | None:
    if "volume" not in frame.columns or len(frame) < 20:
        return None
    volume = frame["volume"].astype(float)
    baseline = float(volume.tail(60).mean())
    recent = float(volume.tail(10).mean())
    return recent / baseline if baseline else None


def _ticker_score(
    lookback_return: float,
    return_20: float,
    return_60: float,
    volatility: float,
    drawdown: float,
    relative_return: float,
    volume_ratio: float | None,
) -> float:
    score = 0.0
    score += clamp(lookback_return, -0.4, 0.4) * 0.55
    score += clamp(return_60, -0.25, 0.25) * 0.45
    score += clamp(return_20, -0.18, 0.18) * 0.35
    score += clamp(relative_return, -0.3, 0.3) * 0.45
    score += clamp(drawdown, -0.4, 0.0) * 0.25
    score -= clamp(volatility, 0.0, 0.08) * 0.4
    if volume_ratio is not None and return_20 > 0:
        score += clamp(volume_ratio - 1.0, -0.3, 0.8) * 0.08
    return clamp(score, -1.0, 1.0)


def _score_reasons(
    lookback_return: float,
    return_20: float,
    return_60: float,
    relative_return: float,
    drawdown: float,
    volume_ratio: float | None,
) -> tuple[list[str], list[str]]:
    tailwinds: list[str] = []
    headwinds: list[str] = []
    if lookback_return > 0.05:
        tailwinds.append("positive lookback return")
    if return_20 > 0.03:
        tailwinds.append("positive 20-period momentum")
    if return_60 > 0.05:
        tailwinds.append("positive 60-period momentum")
    if relative_return > 0.03:
        tailwinds.append("outperforming benchmark")
    if volume_ratio is not None and volume_ratio > 1.2 and return_20 > 0:
        tailwinds.append("rising volume with positive momentum")
    if lookback_return < -0.05:
        headwinds.append("negative lookback return")
    if return_20 < -0.03:
        headwinds.append("negative 20-period momentum")
    if drawdown < -0.12:
        headwinds.append("meaningful drawdown")
    if relative_return < -0.03:
        headwinds.append("underperforming benchmark")
    return tailwinds, headwinds


def _direction_from_score(score: float) -> str:
    if score >= 0.08:
        return "bullish"
    if score <= -0.08:
        return "bearish"
    return "neutral"


def _data_quality_from_coverage(coverage: dict[str, Any], news_summary: dict[str, Any]) -> str:
    rows = int(coverage.get("snapshot_row_count", 0))
    has_news = bool(news_summary.get("row_count", 0))
    if coverage.get("price_quality", {}).get("warnings"):
        return "partial" if rows >= 60 else "weak"
    if rows >= 300 and has_news:
        return "strong"
    if rows >= 60:
        return "partial"
    return "weak"


def _build_replay_thesis(top: dict[str, Any], direction: str, replay: dict[str, Any]) -> str:
    ticker = top["ticker"]
    horizon = int(replay.get("horizon_days", 30))
    if direction == "bullish":
        return f"{ticker} is the strongest replay candidate for the next {horizon} days based on momentum, relative strength, and available historical context."
    if direction == "bearish":
        return f"{ticker} is the weakest replay candidate for the next {horizon} days; historical context favors caution or short-bias review."
    return f"{ticker} is a watchlist-only replay candidate; evidence is not directional enough for a strong thesis."


def _replay_catalysts(top: dict[str, Any], replay: dict[str, Any]) -> list[str]:
    catalysts = list(top.get("tailwinds", []))
    news_hits = replay.get("news_summary", {}).get("ticker_hits", {}).get(top["ticker"], 0)
    if news_hits:
        catalysts.append(f"{news_hits} ticker-specific news references in the replay window")
    return catalysts


def _replay_recommendations(replay: dict[str, Any], evaluation: dict[str, Any]) -> list[str]:
    recommendations = [
        "Treat this as an agent reasoning exam, not paper trading and not a live recommendation.",
        "Keep future/target columns out of the replay snapshot before adding LLM or FinBERT agents.",
    ]
    guard = replay.get("data_guard", {}).get("prices", {})
    if guard.get("removed_column_count", 0):
        recommendations.append("Replay guard removed leakage columns; review removed_columns if using enriched datasets.")
    price_warnings = replay.get("coverage", {}).get("price_quality", {}).get("warnings", [])
    if price_warnings:
        recommendations.append("Review price-quality warnings before trusting replay hit/miss results.")
    if evaluation.get("status") == "evaluated":
        recommendations.append("Store this replay outcome in learning memory only after reviewing the thesis and data window.")
    if evaluation.get("status") == "missing_price_window":
        recommendations.append("Pick an earlier as_of date or a shorter horizon so future prices exist for evaluation.")
    return recommendations


def _horizon_years(horizon_days: int | None) -> float:
    return float(horizon_days or 30) / 365.0


def _price_quality_summary(frame: Any, tickers: list[str]) -> dict[str, Any]:
    if frame.empty:
        return {"warnings": ["Historical replay snapshot has no price rows."]}
    working = frame.copy()
    working["_dean_date"] = working["_dean_datetime"].dt.date
    rows_per_day = working.groupby(["_dean_ticker", "_dean_date"]).size()
    duplicate_timestamp_count = int(working.duplicated(subset=["_dean_ticker", "_dean_datetime"]).sum())
    interval_counts = working["interval"].astype(str).value_counts().to_dict() if "interval" in working.columns else {}
    max_rows_per_ticker_day = int(rows_per_day.max()) if len(rows_per_day) else 0
    multi_row_day_count = int((rows_per_day > 1).sum()) if len(rows_per_day) else 0
    benchmark_return = _ticker_lookback_return(frame, "SPY") if "SPY" in [ticker.upper() for ticker in tickers] else None
    warnings: list[str] = []
    if duplicate_timestamp_count:
        warnings.append(f"Duplicate ticker/datetime rows detected: {duplicate_timestamp_count}.")
    if interval_counts.get("1d") and max_rows_per_ticker_day > 1:
        warnings.append("Rows are labelled 1d but multiple rows per ticker/day exist; normalize daily bars before relying on replay scores.")
    if benchmark_return is not None and abs(benchmark_return) > 0.5:
        warnings.append(f"Benchmark SPY lookback return is extreme ({benchmark_return:.3f}); review splits, interval mixing, or price normalization.")
    return {
        "duplicate_ticker_datetime_count": duplicate_timestamp_count,
        "max_rows_per_ticker_day": max_rows_per_ticker_day,
        "multi_row_ticker_day_count": multi_row_day_count,
        "interval_counts": interval_counts,
        "benchmark_spy_lookback_return": benchmark_return,
        "warnings": warnings,
    }


def _normalize_daily_bars(pd: Any, frame: Any) -> tuple[Any, dict[str, Any]]:
    if frame.empty:
        return frame, {"applied": True, "input_rows": 0, "output_rows": 0, "warnings": ["No rows to normalize."]}
    working = frame.copy()
    working["_dean_date"] = working["_dean_datetime"].dt.date
    sort_cols = ["_dean_ticker", "_dean_datetime"]
    working = working.sort_values(sort_cols)
    aggregations: dict[str, Any] = {
        "_dean_datetime": "max",
        "_dean_close": "last",
    }
    for column, method in [("ticker", "last"), ("symbol", "last"), ("Ticker", "last"), ("interval", "last"), ("hash", "last")]:
        if column in working.columns:
            aggregations[column] = method
    if "open" in working.columns:
        aggregations["open"] = "first"
    if "high" in working.columns:
        aggregations["high"] = "max"
    if "low" in working.columns:
        aggregations["low"] = "min"
    if "close" in working.columns:
        aggregations["close"] = "last"
    if "volume" in working.columns:
        aggregations["volume"] = "sum"
    grouped = working.groupby(["_dean_ticker", "_dean_date"], as_index=False).agg(aggregations)
    grouped["_dean_datetime"] = pd.to_datetime(grouped["_dean_datetime"], utc=True, errors="coerce")
    grouped["_dean_close"] = pd.to_numeric(grouped["_dean_close"], errors="coerce")
    if "interval" in grouped.columns:
        grouped["interval"] = "1d_normalized"
    grouped = grouped.dropna(subset=["_dean_datetime", "_dean_close"]).sort_values("_dean_datetime")
    return grouped, {
        "applied": True,
        "input_rows": int(len(frame)),
        "output_rows": int(len(grouped)),
        "collapsed_rows": int(len(frame) - len(grouped)),
        "method": "group by ticker/date; open=first, high=max, low=min, close=last, volume=sum",
        "warnings": [],
    }


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default
