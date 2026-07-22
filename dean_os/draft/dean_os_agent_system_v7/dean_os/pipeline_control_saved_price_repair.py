from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

from dean_os.draft.dean_os_agent_system_v7.dean_os.artifact_writer import ReviewArtifactWriter
from dean_os.schemas import utc_now_iso
from dean_os.utils import json_ready


class PipelineControlSavedPriceRepair:
    """Build non-destructive clean and resampled candidates from eligible saved 15m tails."""

    def __init__(
        self,
        output_dir: str | Path = "reports/dean_os/pipeline_control_saved_price_repair_current",
    ):
        self.output_dir = Path(output_dir)

    def build(
        self,
        *,
        coverage_json: str | Path,
        required_model_rows: int = 180,
        min_daily_source_bars: int = 24,
        save: bool = True,
    ) -> dict[str, Any]:
        coverage_path = Path(coverage_json)
        coverage = json.loads(coverage_path.read_text(encoding="utf-8"))
        contexts = [
            context
            for context in coverage.get("eligible_contexts", [])
            if context.get("timeframe") == "15m"
        ]
        if not contexts:
            raise ValueError("Coverage contains no eligible 15m contexts.")
        source_paths = {
            str(context.get("source_path"))
            for context in contexts
            if context.get("source_path")
        }
        if len(source_paths) != 1:
            raise ValueError("Repair requires one unambiguous saved 15m source artifact.")
        source_path = Path(next(iter(source_paths)))
        source = _load_table(source_path)
        clean_15m, source_stats = _build_clean_15m(source, contexts)
        contamination = _cross_ticker_identity_groups(clean_15m)
        if contamination:
            raise ValueError(
                f"Eligible tails still contain {contamination} cross-ticker identical OHLCV groups."
            )

        repaired_60m = _resample_prices(
            clean_15m,
            timeframe="60m",
            rule="60min",
            min_source_bars=4,
            offset="30min",
        )
        repaired_1d = _resample_prices(
            clean_15m,
            timeframe="1d",
            rule="1D",
            min_source_bars=min_daily_source_bars,
            offset=None,
        )
        run_id = _run_id("pipeline_control_saved_price_repair")
        artifact_dir = self.output_dir / run_id / "artifacts"
        artifact_paths = {}
        if save:
            artifact_dir.mkdir(parents=True, exist_ok=True)
            artifact_paths = {
                "prices_15m_clean": artifact_dir / "prices_15m_clean.parquet",
                "prices_60m_resampled": artifact_dir / "prices_60m_resampled.parquet",
                "prices_1d_resampled": artifact_dir / "prices_1d_resampled.parquet",
            }
            clean_15m.to_parquet(artifact_paths["prices_15m_clean"], index=False)
            repaired_60m.to_parquet(artifact_paths["prices_60m_resampled"], index=False)
            repaired_1d.to_parquet(artifact_paths["prices_1d_resampled"], index=False)

        coverage_by_timeframe = {
            "15m": _timeframe_coverage(clean_15m, required_model_rows),
            "60m": _timeframe_coverage(repaired_60m, required_model_rows),
            "1d": _timeframe_coverage(repaired_1d, required_model_rows),
        }
        summary = {
            "repair_status": "non_destructive_price_candidates_ready",
            "source_ticker_count": int(clean_15m["ticker"].nunique()),
            "clean_15m_row_count": len(clean_15m),
            "resampled_60m_row_count": len(repaired_60m),
            "resampled_1d_row_count": len(repaired_1d),
            "cross_ticker_identity_groups": contamination,
            "timeframes_ready_for_required_rows": [
                timeframe
                for timeframe, item in coverage_by_timeframe.items()
                if item["all_tickers_meet_required_rows"]
            ],
            "timeframes_still_short": [
                timeframe
                for timeframe, item in coverage_by_timeframe.items()
                if not item["all_tickers_meet_required_rows"]
            ],
            "can_replace_source_artifacts_automatically": False,
            "can_write_database": False,
            "can_train": False,
            "can_trade": False,
        }
        payload = {
            "run_id": run_id,
            "created_at": utc_now_iso(),
            "mode": "pipeline_control_saved_price_repair",
            "inputs": {
                "coverage_json": str(coverage_path),
                "source_path": str(source_path),
                "required_model_rows": required_model_rows,
                "min_daily_source_bars": min_daily_source_bars,
            },
            "summary": summary,
            "source_provenance": {
                "path": str(source_path),
                "sha256": _sha256_file(source_path),
                "synthetic": False,
                "source_evidence_class": "saved_observed_15m_market_data",
            },
            "source_stats": source_stats,
            "coverage_by_timeframe": coverage_by_timeframe,
            "artifacts": {
                key: {
                    "path": str(path),
                    "sha256": _sha256_file(path),
                    "synthetic": False,
                    "derived_from_observed_bars": True,
                }
                for key, path in artifact_paths.items()
            },
            "repair_rules": {
                "15m": "Use only per-ticker eligible tails declared by saved-data coverage.",
                "60m": "Aggregate four observed 15m bars per UTC hour bin anchored at minute 30.",
                "1d": (
                    f"Aggregate observed 15m bars by UTC date and require at least "
                    f"{min_daily_source_bars} source bars."
                ),
                "ohlcv": "open=first, high=max, low=min, close=last, volume=sum.",
            },
            "operator_next_steps": _next_steps(coverage_by_timeframe),
            "explicit_non_actions": [
                "Original parquet files were not modified.",
                "The DuckDB database was not opened or written.",
                "No missing price was synthesized or interpolated.",
                "No collector, model training, tuning, recommendation, order, or trade ran.",
            ],
        }
        if save:
            payload["saved_paths"] = ReviewArtifactWriter(self.output_dir).write(
                payload=payload,
                markdown=render_saved_price_repair_markdown(payload),
                run_id=run_id,
            )
        return json_ready(payload)


def render_saved_price_repair_markdown(payload: dict[str, Any]) -> str:
    summary = payload.get("summary", {})
    lines = [
        "# Pipeline Control Saved Price Repair",
        "",
        f"- Status: `{summary.get('repair_status')}`",
        f"- Tickers: {summary.get('source_ticker_count')}",
        f"- Clean 15m rows: {summary.get('clean_15m_row_count')}",
        f"- Resampled 60m rows: {summary.get('resampled_60m_row_count')}",
        f"- Resampled 1d rows: {summary.get('resampled_1d_row_count')}",
        f"- Ready timeframes: {summary.get('timeframes_ready_for_required_rows')}",
        f"- Still short: {summary.get('timeframes_still_short')}",
        f"- Can replace sources automatically: {summary.get('can_replace_source_artifacts_automatically')}",
        f"- Can trade: {summary.get('can_trade')}",
        "",
        "## Coverage",
        "",
    ]
    for timeframe, item in payload.get("coverage_by_timeframe", {}).items():
        lines.append(
            f"- `{timeframe}`: min_rows={item.get('minimum_ticker_rows')} "
            f"max_rows={item.get('maximum_ticker_rows')} "
            f"all_ready={item.get('all_tickers_meet_required_rows')}"
        )
    lines.extend(["", "## Next Steps", ""])
    lines.extend(f"- {item}" for item in payload.get("operator_next_steps", []))
    lines.extend(["", "## Explicit Non-Actions", ""])
    lines.extend(f"- {item}" for item in payload.get("explicit_non_actions", []))
    return "\n".join(lines).strip() + "\n"


def _build_clean_15m(
    source: pd.DataFrame,
    contexts: list[dict[str, Any]],
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    required = {"datetime", "ticker", "interval", "open", "high", "low", "close", "volume"}
    missing = sorted(required - set(source.columns))
    if missing:
        raise ValueError(f"15m source is missing required columns: {', '.join(missing)}.")
    frame = source.copy()
    frame["datetime"] = pd.to_datetime(frame["datetime"], errors="coerce", utc=True)
    frame["ticker"] = frame["ticker"].astype(str).str.upper()
    frame["interval"] = frame["interval"].astype(str).str.lower()
    pieces = []
    stats = []
    for context in contexts:
        ticker = str(context["ticker"]).upper()
        start = pd.to_datetime(context.get("effective_start"), utc=True)
        ticker_frame = frame.loc[
            frame["ticker"].eq(ticker)
            & frame["interval"].eq("15m")
            & frame["datetime"].ge(start)
        ].copy()
        ticker_frame = ticker_frame.dropna(subset=["datetime"]).sort_values(
            "datetime"
        ).drop_duplicates("datetime", keep="last")
        for column in ("open", "high", "low", "close", "volume"):
            ticker_frame[column] = pd.to_numeric(ticker_frame[column], errors="coerce")
        ticker_frame = ticker_frame.dropna(
            subset=["open", "high", "low", "close", "volume"]
        )
        returns = ticker_frame["close"].pct_change(fill_method=None).abs().dropna()
        max_return = float(returns.max()) if not returns.empty else None
        deltas = ticker_frame["datetime"].diff().dt.total_seconds().div(60).dropna()
        cadence_ratio = float((deltas == 15.0).mean()) if not deltas.empty else 0.0
        if max_return is not None and max_return > 0.25:
            raise ValueError(f"Eligible tail for {ticker} still has max return {max_return:.6f}.")
        if cadence_ratio < 0.75:
            raise ValueError(f"Eligible tail for {ticker} has 15m cadence ratio {cadence_ratio:.3f}.")
        pieces.append(ticker_frame)
        stats.append(
            {
                "ticker": ticker,
                "row_count": len(ticker_frame),
                "start": _iso(ticker_frame["datetime"].min()),
                "end": _iso(ticker_frame["datetime"].max()),
                "max_abs_return": max_return,
                "cadence_ratio": cadence_ratio,
            }
        )
    clean = pd.concat(pieces, ignore_index=True)
    clean = clean.sort_values(["ticker", "datetime"]).reset_index(drop=True)
    clean["interval"] = "15m"
    return clean, stats


def _resample_prices(
    source: pd.DataFrame,
    *,
    timeframe: str,
    rule: str,
    min_source_bars: int,
    offset: str | None,
) -> pd.DataFrame:
    results = []
    for ticker, group in source.groupby("ticker", sort=True):
        indexed = group.sort_values("datetime").set_index("datetime")
        resampler = indexed.resample(
            rule,
            origin="start_day",
            offset=offset,
            label="left",
            closed="left",
        )
        aggregated = resampler.agg(
            open=("open", "first"),
            high=("high", "max"),
            low=("low", "min"),
            close=("close", "last"),
            volume=("volume", "sum"),
            source_bar_count=("close", "count"),
        )
        aggregated = aggregated.loc[
            aggregated["source_bar_count"].ge(min_source_bars)
        ].dropna(subset=["open", "high", "low", "close"])
        aggregated = aggregated.reset_index()
        aggregated["ticker"] = ticker
        aggregated["interval"] = timeframe
        results.append(aggregated)
    if not results:
        return pd.DataFrame(
            columns=[
                "datetime",
                "ticker",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "interval",
                "source_bar_count",
                "hash",
            ]
        )
    combined = pd.concat(results, ignore_index=True)
    combined = combined[
        [
            "datetime",
            "ticker",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "interval",
            "source_bar_count",
        ]
    ].sort_values(["ticker", "datetime"]).reset_index(drop=True)
    identity = (
        combined["datetime"].astype(str)
        + "|"
        + combined["ticker"].astype(str)
        + "|"
        + combined["interval"].astype(str)
    )
    combined["hash"] = identity.map(
        lambda value: hashlib.sha256(value.encode("utf-8")).hexdigest()
    )
    return combined


def _cross_ticker_identity_groups(frame: pd.DataFrame) -> int:
    identity = ["datetime", "open", "high", "low", "close", "volume"]
    duplicates = frame.loc[frame.duplicated(identity, keep=False)]
    if duplicates.empty:
        return 0
    groups = duplicates.groupby(identity, dropna=False)["ticker"].nunique()
    return int((groups > 1).sum())


def _timeframe_coverage(frame: pd.DataFrame, required_rows: int) -> dict[str, Any]:
    counts = (
        frame.groupby("ticker").size().sort_index().to_dict()
        if not frame.empty and "ticker" in frame.columns
        else {}
    )
    deficits = {
        ticker: max(0, required_rows - int(count))
        for ticker, count in counts.items()
    }
    return {
        "ticker_count": len(counts),
        "row_count": len(frame),
        "rows_by_ticker": {str(ticker): int(count) for ticker, count in counts.items()},
        "row_deficit_by_ticker": deficits,
        "minimum_ticker_rows": min(counts.values()) if counts else 0,
        "maximum_ticker_rows": max(counts.values()) if counts else 0,
        "required_model_rows": required_rows,
        "all_tickers_meet_required_rows": bool(counts)
        and all(int(count) >= required_rows for count in counts.values()),
    }


def _next_steps(coverage: dict[str, dict[str, Any]]) -> list[str]:
    steps = []
    for timeframe in ("60m", "1d"):
        item = coverage[timeframe]
        if item["all_tickers_meet_required_rows"]:
            steps.append(
                f"{timeframe} has enough derived rows for a new coverage review; it is not auto-promoted."
            )
        else:
            maximum_deficit = max(item["row_deficit_by_ticker"].values(), default=0)
            steps.append(
                f"Collect or recover more real history for {timeframe}; largest per-ticker "
                f"deficit is {maximum_deficit} rows."
            )
    steps.append(
        "Refresh macro observations with publication/vintage timestamps before expecting variable macro features."
    )
    return steps


def _load_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Price repair source must be parquet or CSV: {path}")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _iso(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None
    return pd.Timestamp(value).isoformat()


def _run_id(prefix: str) -> str:
    return f"{prefix}_{utc_now_iso().replace(':', '').replace('-', '').replace('.', '')}"
