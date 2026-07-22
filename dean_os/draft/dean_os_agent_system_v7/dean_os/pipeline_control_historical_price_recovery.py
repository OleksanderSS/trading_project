from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import pandas as pd

from dean_os.draft.dean_os_agent_system_v7.dean_os.artifact_writer import ReviewArtifactWriter
from dean_os.draft.dean_os_agent_system_v7.dean_os.agents.pipeline_control.pipeline_control_saved_price_repair import (
    _cross_ticker_identity_groups,
    _resample_prices,
    _sha256_file,
    _timeframe_coverage,
)
from dean_os.schemas import utc_now_iso
from dean_os.utils import json_ready


class PipelineControlHistoricalPriceRecovery:
    """Validate and partition observed price history without promoting or training it."""

    def __init__(
        self,
        output_dir: str | Path = (
            "reports/dean_os/pipeline_control_historical_price_recovery_current"
        ),
    ):
        self.output_dir = Path(output_dir)

    def build(
        self,
        *,
        historical_15m_path: str | Path,
        current_15m_path: str | Path,
        historical_1d_path: str | Path,
        required_development_rows: int = 180,
        minimum_past_evaluation_rows: int = 60,
        min_daily_source_bars: int = 24,
        save: bool = True,
    ) -> dict[str, Any]:
        paths = {
            "historical_15m": Path(historical_15m_path),
            "current_15m": Path(current_15m_path),
            "historical_1d": Path(historical_1d_path),
        }
        historical_15m, historical_15m_stats = _load_observed_prices(
            paths["historical_15m"],
            expected_interval="15m",
        )
        current_15m, current_15m_stats = _load_observed_prices(
            paths["current_15m"],
            expected_interval="15m",
        )
        historical_1d, historical_1d_stats = _load_observed_prices(
            paths["historical_1d"],
            expected_interval="1d",
        )
        _require_matching_tickers(
            historical_15m=historical_15m,
            current_15m=current_15m,
            historical_1d=historical_1d,
        )
        if historical_15m["datetime"].max() >= current_15m["datetime"].min():
            raise ValueError(
                "Historical development 15m data must end before the current "
                "past-evaluation partition starts."
            )

        historical_60m = _resample_prices(
            historical_15m,
            timeframe="60m",
            rule="60min",
            min_source_bars=4,
            offset="30min",
        )
        current_60m = _resample_prices(
            current_15m,
            timeframe="60m",
            rule="60min",
            min_source_bars=4,
            offset="30min",
        )
        historical_1d_derived = _resample_prices(
            historical_15m,
            timeframe="1d",
            rule="1D",
            min_source_bars=min_daily_source_bars,
            offset=None,
        )
        current_1d_derived = _resample_prices(
            current_15m,
            timeframe="1d",
            rule="1D",
            min_source_bars=min_daily_source_bars,
            offset=None,
        )
        daily_consistency = _daily_consistency(
            direct=historical_1d,
            derived=historical_1d_derived,
        )
        if not daily_consistency["consistent"]:
            raise ValueError(
                "Direct daily observations are inconsistent with overlapping "
                "daily bars derived from the historical 15m source."
            )

        development = {
            "15m": _timeframe_coverage(
                historical_15m,
                required_development_rows,
            ),
            "60m": _timeframe_coverage(
                historical_60m,
                required_development_rows,
            ),
            "1d": _timeframe_coverage(
                historical_1d,
                required_development_rows,
            ),
        }
        past_evaluation = {
            "15m": _timeframe_coverage(
                current_15m,
                minimum_past_evaluation_rows,
            ),
            "60m": _timeframe_coverage(
                current_60m,
                minimum_past_evaluation_rows,
            ),
            "1d": _timeframe_coverage(
                current_1d_derived,
                minimum_past_evaluation_rows,
            ),
        }

        run_id = _run_id("pipeline_control_historical_price_recovery")
        artifact_dir = self.output_dir / run_id / "artifacts"
        frames = {
            "development_15m": historical_15m,
            "development_60m": historical_60m,
            "development_1d": historical_1d,
            "past_evaluation_15m": current_15m,
            "past_evaluation_60m": current_60m,
            "past_evaluation_1d_context_tail": current_1d_derived,
        }
        artifact_paths: dict[str, Path] = {}
        if save:
            artifact_dir.mkdir(parents=True, exist_ok=True)
            for name, frame in frames.items():
                path = artifact_dir / f"{name}.parquet"
                frame.to_parquet(path, index=False)
                artifact_paths[name] = path

        development_ready = all(
            item["all_tickers_meet_required_rows"] for item in development.values()
        )
        intraday_evaluation_ready = all(
            past_evaluation[timeframe]["all_tickers_meet_required_rows"]
            for timeframe in ("15m", "60m")
        )
        summary = {
            "recovery_status": "historical_context_partitions_ready",
            "ticker_count": int(historical_15m["ticker"].nunique()),
            "development_timeframes_ready": [
                timeframe
                for timeframe, item in development.items()
                if item["all_tickers_meet_required_rows"]
            ],
            "past_evaluation_timeframes_ready": [
                timeframe
                for timeframe, item in past_evaluation.items()
                if item["all_tickers_meet_required_rows"]
            ],
            "ready_for_bounded_offline_intraday_evaluation": (
                development_ready and intraday_evaluation_ready
            ),
            "ready_for_1d_past_evaluation": past_evaluation["1d"][
                "all_tickers_meet_required_rows"
            ],
            "daily_overlap_consistent": daily_consistency["consistent"],
            "can_merge_partitions_automatically": False,
            "can_train_automatically": False,
            "can_write_database": False,
            "can_trade": False,
        }
        payload = {
            "run_id": run_id,
            "created_at": utc_now_iso(),
            "mode": "pipeline_control_historical_price_recovery",
            "summary": summary,
            "inputs": {
                **{key: str(path) for key, path in paths.items()},
                "required_development_rows": required_development_rows,
                "minimum_past_evaluation_rows": minimum_past_evaluation_rows,
                "min_daily_source_bars": min_daily_source_bars,
            },
            "source_lineage": {
                key: {
                    "path": str(path),
                    "sha256": _sha256_file(path),
                    "format": "parquet",
                    "synthetic": False,
                    "observed": True,
                }
                for key, path in paths.items()
            },
            "source_quality": {
                "historical_15m": historical_15m_stats,
                "current_15m": current_15m_stats,
                "historical_1d": historical_1d_stats,
                "daily_overlap_consistency": daily_consistency,
            },
            "coverage": {
                "development": development,
                "past_evaluation": past_evaluation,
            },
            "artifacts": {
                name: {
                    "path": str(path),
                    "sha256": _sha256_file(path),
                    "row_count": len(frames[name]),
                    "synthetic": False,
                }
                for name, path in artifact_paths.items()
            },
            "context_and_target_contract": {
                "15m": {
                    "role": "short_horizon_state_and_prediction_frame",
                    "one_hour_target_shift_bars": 4,
                },
                "60m": {
                    "role": "hourly trend_and_regime_context",
                    "one_hour_target_shift_bars": 1,
                },
                "1d": {
                    "role": "slow_regime_and_historical_context",
                    "one_day_target_shift_bars": 1,
                },
                "asof_join_direction": "backward_only",
                "future_context_allowed": False,
                "targets_may_cross_partition_boundary": False,
                "development_and_past_evaluation_may_be_concatenated": False,
                "new_forward_holdout_required_after_model_selection": True,
            },
            "operator_next_steps": [
                "Use development partitions for walk-forward train/validation only.",
                (
                    "Use the current partition as past evaluation evidence; it is not a "
                    "virgin locked holdout because earlier diagnostics already inspected it."
                ),
                (
                    "Implement timeframe-aware target shifts and backward-only context joins "
                    "before any multi-timeframe model comparison."
                ),
                "Accumulate a new forward holdout after model and feature selection are frozen.",
            ],
            "explicit_non_actions": [
                "No pickle-disguised-as-parquet source was loaded.",
                "No source artifact or database was modified.",
                "No missing observation was synthesized, interpolated, or forward-filled.",
                "No partitions were concatenated across the observed time gap.",
                "No model training, tuning, recommendation, order, or trade ran.",
            ],
        }
        if save:
            payload["saved_paths"] = ReviewArtifactWriter(self.output_dir).write(
                payload=payload,
                markdown=render_historical_price_recovery_markdown(payload),
                run_id=run_id,
            )
        return json_ready(payload)


def render_historical_price_recovery_markdown(payload: dict[str, Any]) -> str:
    summary = payload["summary"]
    lines = [
        "# Pipeline Control Historical Price Recovery",
        "",
        f"- Status: `{summary['recovery_status']}`",
        f"- Tickers: {summary['ticker_count']}",
        (
            "- Development timeframes ready: "
            f"{summary['development_timeframes_ready']}"
        ),
        (
            "- Past-evaluation timeframes ready: "
            f"{summary['past_evaluation_timeframes_ready']}"
        ),
        (
            "- Bounded offline intraday evaluation ready: "
            f"{summary['ready_for_bounded_offline_intraday_evaluation']}"
        ),
        f"- 1d past evaluation ready: {summary['ready_for_1d_past_evaluation']}",
        f"- Can train automatically: {summary['can_train_automatically']}",
        f"- Can trade: {summary['can_trade']}",
        "",
        "## Coverage",
        "",
    ]
    for partition, coverage in payload["coverage"].items():
        for timeframe, item in coverage.items():
            lines.append(
                f"- `{partition}/{timeframe}`: rows={item['row_count']} "
                f"min_per_ticker={item['minimum_ticker_rows']} "
                f"ready={item['all_tickers_meet_required_rows']}"
            )
    lines.extend(["", "## Next Steps", ""])
    lines.extend(f"- {item}" for item in payload["operator_next_steps"])
    lines.extend(["", "## Explicit Non-Actions", ""])
    lines.extend(f"- {item}" for item in payload["explicit_non_actions"])
    return "\n".join(lines).strip() + "\n"


def _load_observed_prices(
    path: Path,
    *,
    expected_interval: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    _require_real_parquet(path)
    frame = pd.read_parquet(path)
    required = {
        "datetime",
        "ticker",
        "interval",
        "open",
        "high",
        "low",
        "close",
        "volume",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{path} is missing required columns: {', '.join(missing)}.")
    frame = frame[list(required)].copy()
    frame["datetime"] = pd.to_datetime(frame["datetime"], errors="coerce", utc=True)
    frame["ticker"] = frame["ticker"].astype(str).str.upper()
    frame["interval"] = frame["interval"].astype(str).str.lower()
    for column in ("open", "high", "low", "close", "volume"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if frame[list(required)].isna().any().any():
        raise ValueError(f"{path} contains null or unparseable required values.")
    intervals = sorted(frame["interval"].unique())
    if intervals != [expected_interval]:
        raise ValueError(
            f"{path} declares intervals {intervals}; expected only {expected_interval}."
        )
    duplicate_count = int(frame.duplicated(["ticker", "datetime"]).sum())
    if duplicate_count:
        raise ValueError(f"{path} contains {duplicate_count} duplicate row identities.")
    if expected_interval == "1d":
        same_day = frame.assign(
            _date=frame["datetime"].dt.normalize()
        ).duplicated(["ticker", "_date"])
        if same_day.any():
            raise ValueError(f"{path} contains multiple 1d rows for the same ticker/date.")
    valid_ohlcv = (
        frame["low"].le(frame[["open", "close"]].min(axis=1))
        & frame["high"].ge(frame[["open", "close"]].max(axis=1))
        & frame["volume"].ge(0)
    )
    if not valid_ohlcv.all():
        raise ValueError(f"{path} contains {int((~valid_ohlcv).sum())} invalid OHLCV rows.")
    cross_ticker_groups = _cross_ticker_identity_groups(frame)
    if cross_ticker_groups:
        raise ValueError(
            f"{path} contains {cross_ticker_groups} cross-ticker identical OHLCV groups."
        )
    ordered = frame.sort_values(["ticker", "datetime"]).reset_index(drop=True)
    returns = ordered.groupby("ticker")["close"].pct_change(fill_method=None).abs()
    max_abs_return = float(returns.max()) if returns.notna().any() else None
    return_limit = 0.25 if expected_interval == "15m" else 0.75
    if max_abs_return is not None and max_abs_return > return_limit:
        raise ValueError(
            f"{path} has max absolute return {max_abs_return:.6f}; "
            f"limit is {return_limit:.2f}."
        )
    cadence_ratio = None
    if expected_interval == "15m":
        deltas = ordered.groupby("ticker")["datetime"].diff().dt.total_seconds().div(60)
        cadence_ratio = float(deltas.dropna().eq(15.0).mean())
        if cadence_ratio < 0.75:
            raise ValueError(
                f"{path} has 15m cadence ratio {cadence_ratio:.3f}; minimum is 0.75."
            )
    canonical = ordered[
        ["datetime", "ticker", "open", "high", "low", "close", "volume", "interval"]
    ].copy()
    identity = (
        canonical["datetime"].astype(str)
        + "|"
        + canonical["ticker"]
        + "|"
        + canonical["interval"]
    )
    canonical["hash"] = identity.map(
        lambda value: hashlib.sha256(value.encode("utf-8")).hexdigest()
    )
    counts = canonical.groupby("ticker").size()
    stats = {
        "row_count": len(canonical),
        "ticker_count": int(canonical["ticker"].nunique()),
        "minimum_ticker_rows": int(counts.min()),
        "maximum_ticker_rows": int(counts.max()),
        "start": canonical["datetime"].min().isoformat(),
        "end": canonical["datetime"].max().isoformat(),
        "max_abs_return": max_abs_return,
        "cadence_ratio": cadence_ratio,
        "duplicate_row_identities": duplicate_count,
        "cross_ticker_identity_groups": cross_ticker_groups,
    }
    return canonical, stats


def _require_real_parquet(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("rb") as handle:
        prefix = handle.read(4)
        handle.seek(-4, 2)
        suffix = handle.read(4)
    if prefix != b"PAR1" or suffix != b"PAR1":
        raise ValueError(
            f"{path} is not a real Parquet file. Refusing to deserialize by extension."
        )


def _require_matching_tickers(**frames: pd.DataFrame) -> None:
    ticker_sets = {
        name: set(frame["ticker"].unique())
        for name, frame in frames.items()
    }
    expected = next(iter(ticker_sets.values()))
    mismatched = {
        name: sorted(tickers)
        for name, tickers in ticker_sets.items()
        if tickers != expected
    }
    if mismatched:
        raise ValueError(f"Price sources do not cover the same ticker set: {mismatched}.")


def _daily_consistency(
    *,
    direct: pd.DataFrame,
    derived: pd.DataFrame,
) -> dict[str, Any]:
    joined = direct[["ticker", "datetime", "close"]].merge(
        derived[["ticker", "datetime", "close"]],
        on=["ticker", "datetime"],
        suffixes=("_direct", "_derived"),
    )
    if joined.empty:
        return {
            "overlap_row_count": 0,
            "median_close_relative_error": None,
            "p95_close_relative_error": None,
            "max_close_relative_error": None,
            "consistent": False,
        }
    error = (
        joined["close_direct"].sub(joined["close_derived"]).abs()
        / joined["close_direct"].abs().clip(lower=1e-12)
    )
    p95 = float(error.quantile(0.95))
    maximum = float(error.max())
    return {
        "overlap_row_count": len(joined),
        "overlap_ticker_count": int(joined["ticker"].nunique()),
        "median_close_relative_error": float(error.median()),
        "p95_close_relative_error": p95,
        "max_close_relative_error": maximum,
        "consistent": p95 <= 0.01 and maximum <= 0.03,
        "consistency_rule": "p95 close error <= 1% and max close error <= 3%",
    }


def _run_id(prefix: str) -> str:
    return f"{prefix}_{utc_now_iso().replace(':', '').replace('-', '').replace('.', '')}"
