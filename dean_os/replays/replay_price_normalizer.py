from __future__ import annotations

import asyncio
import json
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from dean_os.historical_replay import (
    HistoricalReplayRunner,
    _price_quality_summary,
    guard_replay_frame,
)
from dean_os.market_data_api import prepare_market_frame, read_market_frame
from dean_os.schemas import utc_now_iso
from dean_os.utils import json_ready


class ReplayPriceNormalizer:
    """Creates a reusable normalized daily OHLCV artifact for replay tests.

    This runner is deliberately data-only. It does not create paper trades,
    update learning memory, or start the trading pipeline.
    """

    def __init__(
        self,
        output_dir: str | Path = "reports/dean_os/replay_price_normalizer",
        artifact_dir: str | Path = "data/dean_os/replay_prices",
    ):
        self.output_dir = Path(output_dir)
        self.artifact_dir = Path(artifact_dir)

    async def run(
        self,
        price_data_path: str | Path,
        tickers: list[str] | None = None,
        output_path: str | Path | None = None,
        close_col: str = "close",
        datetime_col: str = "datetime",
        compare_replay: bool = False,
        as_of: str | None = None,
        lookback_days: int = 180,
        horizon_days: int = 60,
        news_data_path: str | Path | None = None,
        macro_data_path: str | Path | None = None,
        benchmark_ticker: str = "SPY",
        neutral_band: float = 0.01,
    ) -> dict[str, Any]:
        payload = self.normalize(
            price_data_path=price_data_path,
            tickers=tickers,
            output_path=output_path,
            close_col=close_col,
            datetime_col=datetime_col,
            benchmark_ticker=benchmark_ticker,
        )
        if compare_replay:
            payload["replay_comparison"] = await self.compare_replay(
                raw_price_path=price_data_path,
                normalized_price_path=payload["artifact"]["path"],
                tickers=payload["inputs"]["tickers"],
                as_of=as_of,
                lookback_days=lookback_days,
                horizon_days=horizon_days,
                news_data_path=news_data_path,
                macro_data_path=macro_data_path,
                benchmark_ticker=benchmark_ticker,
                close_col="close",
                datetime_col="datetime",
                neutral_band=neutral_band,
            )
            _apply_replay_comparison_gate(payload)
            payload["recommendations"] = _recommendations(payload)
            self.save_report(payload)
        return payload

    def normalize(
        self,
        price_data_path: str | Path,
        tickers: list[str] | None = None,
        output_path: str | Path | None = None,
        close_col: str = "close",
        datetime_col: str = "datetime",
        benchmark_ticker: str = "SPY",
    ) -> dict[str, Any]:
        try:
            import pandas as pd
        except Exception as exc:
            raise RuntimeError(f"pandas is required for replay price normalization: {exc}") from exc

        price_path = Path(price_data_path)
        if not price_path.exists():
            raise FileNotFoundError(f"Price data file does not exist: {price_path}")

        raw_frame = read_market_frame(pd, price_path)
        requested_tickers = _normalize_tickers(tickers)
        guard = guard_replay_frame(
            raw_frame,
            required_columns=[datetime_col, close_col, "ticker", "symbol", "Ticker", "Symbol"],
        )
        prepared = prepare_market_frame(
            pd=pd,
            frame=guard.safe_frame,
            close_col=close_col,
            datetime_col=datetime_col,
        )
        if requested_tickers:
            prepared = prepared[prepared["_dean_ticker"].isin(requested_tickers)].copy()
        else:
            requested_tickers = sorted(ticker for ticker in prepared["_dean_ticker"].dropna().unique() if ticker)

        normalized_internal = _normalize_daily_ohlcv(pd, prepared)
        artifact_frame = _to_artifact_frame(normalized_internal)
        artifact_path, artifact_warnings = self._write_artifact(pd, artifact_frame, output_path)
        raw_quality = _quality_for_frame(prepared, requested_tickers or [benchmark_ticker.upper()])
        normalized_quality = _quality_for_frame(normalized_internal, requested_tickers or [benchmark_ticker.upper()])
        quality = {
            "raw": raw_quality,
            "normalized": normalized_quality,
            "improvement": _quality_improvement(raw_quality, normalized_quality),
            "warnings": list(normalized_quality.get("warnings", [])) + artifact_warnings,
        }
        learning_gate = _learning_gate(quality)
        payload = {
            "run_id": "replay_price_normalizer_" + utc_now_iso().replace(":", "").replace("-", "").replace(".", "_"),
            "created_at": utc_now_iso(),
            "mode": "replay_price_normalization",
            "inputs": {
                "price_data_path": str(price_path),
                "tickers": requested_tickers,
                "close_col": close_col,
                "datetime_col": datetime_col,
                "benchmark_ticker": benchmark_ticker.upper(),
            },
            "artifact": {
                "path": str(artifact_path),
                "format": artifact_path.suffix.lower().lstrip("."),
                "row_count": int(len(artifact_frame)),
                "ticker_count": int(artifact_frame["ticker"].nunique()) if not artifact_frame.empty else 0,
                "start": artifact_frame["datetime"].min().isoformat() if not artifact_frame.empty else None,
                "end": artifact_frame["datetime"].max().isoformat() if not artifact_frame.empty else None,
                "columns": list(artifact_frame.columns),
            },
            "normalization": {
                "method": "group by ticker/date; open=first, high=max, low=min, close=last, volume=sum",
                "input_rows": int(len(prepared)),
                "output_rows": int(len(artifact_frame)),
                "collapsed_rows": int(len(prepared) - len(artifact_frame)),
                "assumptions": [
                    "Rows are sorted by ticker and timestamp before aggregation.",
                    "If open/high/low columns are missing, close is used as the fallback price series.",
                    "The resulting artifact is for replay/evaluation only; it is not a trading signal.",
                ],
            },
            "data_guard": {"prices": guard.summary()},
            "quality": quality,
            "learning_gate": learning_gate,
            "recommendations": _recommendations({"quality": quality, "learning_gate": learning_gate}),
        }
        self.save_report(payload)
        return payload

    async def compare_replay(
        self,
        raw_price_path: str | Path,
        normalized_price_path: str | Path,
        tickers: list[str],
        as_of: str | None,
        lookback_days: int,
        horizon_days: int,
        news_data_path: str | Path | None = None,
        macro_data_path: str | Path | None = None,
        benchmark_ticker: str = "SPY",
        close_col: str = "close",
        datetime_col: str = "datetime",
        neutral_band: float = 0.01,
    ) -> dict[str, Any]:
        if not as_of:
            return {
                "status": "skipped",
                "reason": "Provide --as-of with --compare-replay to compare raw vs normalized replay outcomes.",
            }
        if not tickers:
            return {"status": "skipped", "reason": "Replay comparison needs at least one ticker."}

        comparison_dir = self.output_dir / "replay_comparison"
        raw_runner = HistoricalReplayRunner(output_dir=comparison_dir / "raw")
        normalized_runner = HistoricalReplayRunner(output_dir=comparison_dir / "normalized")
        raw = await raw_runner.run(
            price_data_path=raw_price_path,
            tickers=tickers,
            as_of=as_of,
            lookback_days=lookback_days,
            horizon_days=horizon_days,
            news_data_path=news_data_path,
            macro_data_path=macro_data_path,
            benchmark_ticker=benchmark_ticker,
            close_col=close_col,
            datetime_col=datetime_col,
            neutral_band=neutral_band,
            normalize_daily_bars=False,
        )
        normalized = await normalized_runner.run(
            price_data_path=normalized_price_path,
            tickers=tickers,
            as_of=as_of,
            lookback_days=lookback_days,
            horizon_days=horizon_days,
            news_data_path=news_data_path,
            macro_data_path=macro_data_path,
            benchmark_ticker=benchmark_ticker,
            close_col="close",
            datetime_col="datetime",
            neutral_band=neutral_band,
            normalize_daily_bars=False,
        )
        raw_summary = _replay_summary(raw)
        normalized_summary = _replay_summary(normalized)
        return {
            "status": "compared",
            "raw": raw_summary,
            "normalized": normalized_summary,
            "same_decision_ticker": raw_summary.get("ticker") == normalized_summary.get("ticker"),
            "same_action": raw_summary.get("action") == normalized_summary.get("action"),
            "same_outcome_label": raw_summary.get("outcome_label") == normalized_summary.get("outcome_label"),
            "interpretation": _comparison_interpretation(raw_summary, normalized_summary),
            "saved_paths": {
                "raw": raw.get("saved_paths", {}),
                "normalized": normalized.get("saved_paths", {}),
            },
        }

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
        rendered_md = render_replay_price_normalizer_markdown(payload)
        json_path.write_text(rendered_json, encoding="utf-8")
        latest_json.write_text(rendered_json, encoding="utf-8")
        md_path.write_text(rendered_md, encoding="utf-8")
        latest_md.write_text(rendered_md, encoding="utf-8")
        return paths

    def _write_artifact(self, pd: Any, frame: Any, output_path: str | Path | None) -> tuple[Path, list[str]]:
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        artifact_path = Path(output_path) if output_path else self.artifact_dir / _default_artifact_name()
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        suffix = artifact_path.suffix.lower()
        warnings: list[str] = []
        if suffix == ".csv":
            frame.to_csv(artifact_path, index=False)
            return artifact_path, warnings
        if suffix in {".parquet", ".pq"}:
            try:
                frame.to_parquet(artifact_path, index=False)
                return artifact_path, warnings
            except Exception as exc:
                fallback = artifact_path.with_suffix(".csv")
                frame.to_csv(fallback, index=False)
                warnings.append(f"Could not write parquet artifact ({type(exc).__name__}: {exc}); wrote CSV fallback.")
                return fallback, warnings
        raise ValueError(f"Unsupported artifact file type: {artifact_path.suffix}. Use .csv or .parquet.")


def run_sync(**kwargs: Any) -> dict[str, Any]:
    """Small helper for scripts that want a synchronous call boundary."""

    return asyncio.run(ReplayPriceNormalizer().run(**kwargs))


def render_replay_price_normalizer_markdown(payload: dict[str, Any]) -> str:
    artifact = payload.get("artifact", {})
    quality = payload.get("quality", {})
    gate = payload.get("learning_gate", {})
    comparison = payload.get("replay_comparison", {})
    lines = [
        "# DEAN-OS Replay Price Normalizer",
        "",
        f"- Run ID: `{payload.get('run_id')}`",
        f"- Artifact: `{artifact.get('path')}`",
        f"- Rows: {artifact.get('row_count')}",
        f"- Tickers: {artifact.get('ticker_count')}",
        f"- Learning gate: `{gate.get('status')}`",
        "",
        "## Quality",
        "",
        f"- Raw duplicate ticker/datetime rows: {quality.get('raw', {}).get('duplicate_ticker_datetime_count')}",
        f"- Raw max rows per ticker/day: {quality.get('raw', {}).get('max_rows_per_ticker_day')}",
        f"- Normalized duplicate ticker/datetime rows: {quality.get('normalized', {}).get('duplicate_ticker_datetime_count')}",
        f"- Normalized max rows per ticker/day: {quality.get('normalized', {}).get('max_rows_per_ticker_day')}",
        "",
        "## Warnings",
        "",
    ]
    warnings = quality.get("warnings", [])
    if warnings:
        lines.extend(f"- {warning}" for warning in warnings)
    else:
        lines.append("- No normalized price-quality warnings.")
    if comparison:
        lines.extend(
            [
                "",
                "## Replay Comparison",
                "",
                f"- Status: `{comparison.get('status')}`",
                f"- Same ticker: {comparison.get('same_decision_ticker')}",
                f"- Same action: {comparison.get('same_action')}",
                f"- Same outcome: {comparison.get('same_outcome_label')}",
                f"- Interpretation: {comparison.get('interpretation')}",
            ]
        )
    lines.extend(["", "## Recommendations", ""])
    lines.extend(f"- {item}" for item in payload.get("recommendations", []))
    return "\n".join(lines).strip() + "\n"


def _normalize_daily_ohlcv(pd: Any, frame: Any) -> Any:
    if frame.empty:
        return _empty_internal_frame(pd)
    working = frame.copy()
    working["_dean_date"] = working["_dean_datetime"].dt.date
    working = working.sort_values(["_dean_ticker", "_dean_datetime"])
    rows: list[dict[str, Any]] = []
    for (ticker, date_value), group in working.groupby(["_dean_ticker", "_dean_date"], sort=True):
        close = group["_dean_close"].astype(float)
        row = {
            "_dean_ticker": str(ticker).upper(),
            "_dean_date": date_value,
            "_dean_datetime": group["_dean_datetime"].iloc[-1],
            "_dean_close": float(close.iloc[-1]),
            "ticker": str(ticker).upper(),
            "datetime": group["_dean_datetime"].iloc[-1],
            "date": str(date_value),
            "open": _first_numeric(group, "open", fallback=close.iloc[0]),
            "high": _max_numeric(group, "high", fallback=close.max()),
            "low": _min_numeric(group, "low", fallback=close.min()),
            "close": float(close.iloc[-1]),
            "volume": _sum_numeric(group, "volume"),
            "interval": "1d_normalized",
            "source_row_count": int(len(group)),
            "first_source_datetime": group["_dean_datetime"].iloc[0],
            "last_source_datetime": group["_dean_datetime"].iloc[-1],
            "source_intervals": _source_intervals(group),
        }
        rows.append(row)
    normalized = pd.DataFrame(rows)
    if normalized.empty:
        return _empty_internal_frame(pd)
    normalized["_dean_datetime"] = pd.to_datetime(normalized["_dean_datetime"], utc=True, errors="coerce")
    normalized["_dean_close"] = pd.to_numeric(normalized["_dean_close"], errors="coerce")
    return normalized.dropna(subset=["_dean_datetime", "_dean_close"]).sort_values(["_dean_ticker", "_dean_datetime"])


def _to_artifact_frame(frame: Any) -> Any:
    columns = [
        "ticker",
        "datetime",
        "date",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "interval",
        "source_row_count",
        "first_source_datetime",
        "last_source_datetime",
        "source_intervals",
    ]
    existing = [column for column in columns if column in frame.columns]
    return frame.loc[:, existing].copy()


def _empty_internal_frame(pd: Any) -> Any:
    return pd.DataFrame(
        columns=[
            "_dean_ticker",
            "_dean_date",
            "_dean_datetime",
            "_dean_close",
            "ticker",
            "datetime",
            "date",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "interval",
            "source_row_count",
            "first_source_datetime",
            "last_source_datetime",
            "source_intervals",
        ]
    )


def _quality_for_frame(frame: Any, tickers: list[str]) -> dict[str, Any]:
    if frame.empty:
        return {
            "warnings": ["Price artifact has no rows."],
            "duplicate_ticker_datetime_count": 0,
            "max_rows_per_ticker_day": 0,
            "multi_row_ticker_day_count": 0,
            "interval_counts": {},
        }
    return _price_quality_summary(frame, tickers)


def _quality_improvement(raw_quality: dict[str, Any], normalized_quality: dict[str, Any]) -> dict[str, Any]:
    return {
        "duplicate_ticker_datetime_delta": int(normalized_quality.get("duplicate_ticker_datetime_count", 0))
        - int(raw_quality.get("duplicate_ticker_datetime_count", 0)),
        "max_rows_per_ticker_day_delta": int(normalized_quality.get("max_rows_per_ticker_day", 0))
        - int(raw_quality.get("max_rows_per_ticker_day", 0)),
        "multi_row_ticker_day_delta": int(normalized_quality.get("multi_row_ticker_day_count", 0))
        - int(raw_quality.get("multi_row_ticker_day_count", 0)),
    }


def _learning_gate(quality: dict[str, Any]) -> dict[str, Any]:
    warnings = quality.get("warnings", [])
    if warnings:
        return {
            "status": "blocked",
            "can_write_learning_memory": False,
            "reason": "Normalized price-quality warnings remain; replay outcomes must stay diagnostic.",
            "warnings": warnings,
        }
    return {
        "status": "clear",
        "can_write_learning_memory": False,
        "reason": "Price artifact is clean, but learning-memory writes still require a separate human-reviewed bridge.",
        "warnings": [],
    }


def _apply_replay_comparison_gate(payload: dict[str, Any]) -> None:
    comparison = payload.get("replay_comparison", {})
    normalized_warnings = comparison.get("normalized", {}).get("price_warnings", [])
    if not normalized_warnings:
        return
    quality = payload.setdefault("quality", {})
    quality["comparison_window_warnings"] = normalized_warnings
    existing_warnings = list(quality.get("warnings", []))
    merged_warnings = _unique_strings([*existing_warnings, *normalized_warnings])
    quality["warnings"] = merged_warnings
    payload["learning_gate"] = {
        "status": "blocked",
        "can_write_learning_memory": False,
        "reason": "The normalized artifact is clean globally, but the requested replay window still has price-quality warnings.",
        "warnings": merged_warnings,
    }


def _recommendations(payload: dict[str, Any]) -> list[str]:
    gate = payload.get("learning_gate", {})
    quality = payload.get("quality", {})
    recommendations = [
        "Use the normalized artifact for historical replay instead of raw cached daily-like rows.",
        "Keep this step data-only: do not create paper trades or update learning memory from normalization alone.",
    ]
    if gate.get("status") == "blocked":
        recommendations.append("Do not write replay outcomes into learning memory until normalized price-quality warnings are resolved.")
    else:
        recommendations.append("Next safe step: run batch historical replay across multiple as_of dates using this artifact.")
    if quality.get("raw", {}).get("warnings") and not quality.get("normalized", {}).get("warnings"):
        recommendations.append("Raw data warnings were resolved by normalization; keep the report with the artifact as provenance.")
    comparison = payload.get("replay_comparison", {})
    if comparison.get("status") == "compared" and not comparison.get("same_outcome_label"):
        recommendations.append("Raw and normalized replay outcomes differ; treat older raw replay results as invalid for learning.")
    return recommendations


def _normalize_tickers(tickers: Iterable[str] | None) -> list[str]:
    return sorted({str(ticker).strip().upper() for ticker in tickers or [] if str(ticker).strip()})


def _default_artifact_name() -> str:
    stamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    return f"replay_prices_1d_normalized_{stamp}.parquet"


def _first_numeric(group: Any, column: str, fallback: Any) -> float:
    if column in group.columns:
        series = group[column].dropna()
        if not series.empty:
            return float(series.iloc[0])
    return float(fallback)


def _max_numeric(group: Any, column: str, fallback: Any) -> float:
    if column in group.columns:
        series = group[column].dropna()
        if not series.empty:
            return float(series.max())
    return float(fallback)


def _min_numeric(group: Any, column: str, fallback: Any) -> float:
    if column in group.columns:
        series = group[column].dropna()
        if not series.empty:
            return float(series.min())
    return float(fallback)


def _sum_numeric(group: Any, column: str) -> float | None:
    if column not in group.columns:
        return None
    series = group[column].dropna()
    if series.empty:
        return None
    return float(series.sum())


def _source_intervals(group: Any) -> str:
    if "interval" not in group.columns:
        return ""
    values = sorted({str(value) for value in group["interval"].dropna().unique() if str(value)})
    return ",".join(values)


def _replay_summary(payload: dict[str, Any]) -> dict[str, Any]:
    decision = payload.get("decision", {})
    evaluation = payload.get("evaluation", {})
    quality = payload.get("historical_replay", {}).get("coverage", {}).get("price_quality", {})
    return {
        "action": decision.get("action"),
        "ticker": decision.get("ticker"),
        "expected_direction": decision.get("expected_direction"),
        "confidence": decision.get("confidence"),
        "evaluation_status": evaluation.get("status"),
        "outcome_label": evaluation.get("outcome_label"),
        "realized_return": evaluation.get("realized_return"),
        "price_warnings": quality.get("warnings", []),
    }


def _comparison_interpretation(raw: dict[str, Any], normalized: dict[str, Any]) -> str:
    if raw.get("outcome_label") != normalized.get("outcome_label"):
        return "Raw and normalized replay outcomes differ; raw replay should not be used as learning truth."
    if raw.get("ticker") != normalized.get("ticker") or raw.get("action") != normalized.get("action"):
        return "Replay thesis changed after normalization; review rankings before trusting either run."
    return "Raw and normalized replay summaries match at the decision/outcome level."


def _unique_strings(values: Iterable[Any]) -> list[str]:
    unique: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value)
        if text not in seen:
            seen.add(text)
            unique.append(text)
    return unique
