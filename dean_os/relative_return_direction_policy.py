from __future__ import annotations

import hashlib
from datetime import UTC, datetime, time
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo


RELATIVE_RETURN_DIRECTION_CONTRACT = (
    "dean_relative_return_direction_contract_v1"
)


def calibrate_relative_return_direction_contract(
    *,
    price_paths: list[str | Path],
    members: list[str],
    benchmark: str,
    calibration_cutoff_at: str,
    horizon_days: int,
    expected_direction: str,
    minimum_sample_count: int = 60,
    minimum_neutral_band: float = 0.01,
    robust_scale_multiplier: float = 1.0,
) -> dict[str, Any]:
    """Calibrate a symmetric, pre-outcome materiality band.

    Every historical window must finish before the cutoff. The band measures
    ordinary basket-versus-benchmark dispersion and is never fitted to the
    realized outcome of the hypothesis being assessed.
    """
    import numpy as np
    import pandas as pd

    direction = str(expected_direction).strip().lower()
    if direction not in {"positive", "negative"}:
        raise ValueError("expected_direction must be positive or negative")
    horizon = int(horizon_days)
    if horizon < 1:
        raise ValueError("horizon_days must be positive")
    sample_floor = int(minimum_sample_count)
    if sample_floor < 20:
        raise ValueError("minimum_sample_count must be at least 20")
    absolute_floor = float(minimum_neutral_band)
    scale_multiplier = float(robust_scale_multiplier)
    if absolute_floor <= 0 or scale_multiplier <= 0:
        raise ValueError("neutral-band parameters must be positive")

    normalized_members = list(
        dict.fromkeys(str(item).strip().upper() for item in members if str(item).strip())
    )
    normalized_benchmark = str(benchmark).strip().upper()
    if not normalized_members or not normalized_benchmark:
        raise ValueError("members and benchmark are required")
    if normalized_benchmark in normalized_members:
        raise ValueError("benchmark cannot also be a basket member")

    cutoff = pd.Timestamp(calibration_cutoff_at)
    cutoff = cutoff.tz_localize("UTC") if cutoff.tzinfo is None else cutoff.tz_convert("UTC")
    frame, bindings = _load_prices([Path(item) for item in price_paths])
    required = normalized_members + [normalized_benchmark]
    available = set(frame["ticker"].unique()) if not frame.empty else set()
    missing_tickers = [ticker for ticker in required if ticker not in available]
    samples: list[dict[str, Any]] = []
    if not missing_tickers:
        pivot = (
            frame[frame["ticker"].isin(required)]
            .pivot_table(index="datetime", columns="ticker", values="close", aggfunc="last")
            .dropna(subset=required)
            .sort_index()
        )
        sessions = pivot.index
        for baseline in sessions:
            due = baseline + pd.Timedelta(days=horizon)
            candidates = sessions[sessions >= due]
            if not len(candidates):
                continue
            checkpoint = candidates[0]
            # Strict anti-lookahead rule: the whole calibration window must be
            # complete before the event/cutoff, not merely its baseline.
            if _us_session_close_utc(checkpoint) >= cutoff:
                continue
            member_returns = (
                pivot.loc[checkpoint, normalized_members]
                / pivot.loc[baseline, normalized_members]
                - 1.0
            )
            basket_return = float(member_returns.mean())
            benchmark_return = float(
                pivot.loc[checkpoint, normalized_benchmark]
                / pivot.loc[baseline, normalized_benchmark]
                - 1.0
            )
            relative = (1.0 + basket_return) / (1.0 + benchmark_return) - 1.0
            samples.append(
                {
                    "baseline_session": baseline.date().isoformat(),
                    "checkpoint_session": checkpoint.date().isoformat(),
                    "relative_total_return": relative,
                }
            )

    values = np.asarray(
        [item["relative_total_return"] for item in samples], dtype=float
    )
    status = (
        "calibrated_pre_outcome_direction_contract"
        if len(values) >= sample_floor and not missing_tickers
        else "insufficient_pre_outcome_history"
    )
    median = float(np.median(values)) if len(values) else None
    mad = (
        float(np.median(np.abs(values - float(median))))
        if len(values) and median is not None
        else None
    )
    robust_scale = float(1.4826 * mad) if mad is not None else None
    neutral_band = (
        max(absolute_floor, scale_multiplier * robust_scale)
        if status == "calibrated_pre_outcome_direction_contract"
        and robust_scale is not None
        else None
    )
    blockers: list[str] = []
    if missing_tickers:
        blockers.append("missing_price_tickers:" + ",".join(missing_tickers))
    if len(values) < sample_floor:
        blockers.append(f"historical_sample_below_floor:{len(values)}/{sample_floor}")

    classification = _classification_rules(direction, neutral_band)
    return {
        "contract": RELATIVE_RETURN_DIRECTION_CONTRACT,
        "status": status,
        "metric": "basket_relative_total_return",
        "expected_direction": direction,
        "horizon_days": horizon,
        "basket": {
            "members": normalized_members,
            "weighting": "equal_weight_at_baseline_no_intra_window_rebalance",
            "benchmark": normalized_benchmark,
            "relative_return_formula": "(1 + basket_return) / (1 + benchmark_return) - 1",
        },
        "calibration": {
            "method": "zero_centered_symmetric_band_from_pre_cutoff_robust_mad_scale",
            "calibration_cutoff_at": cutoff.isoformat(),
            "anti_lookahead_rule": "every historical US session close (16:00 America/New_York, DST-aware) is strictly before calibration_cutoff_at",
            "calendar_horizon_days": horizon,
            "historical_sample_count": int(len(values)),
            "minimum_sample_count": sample_floor,
            "minimum_neutral_band": absolute_floor,
            "robust_scale_multiplier": scale_multiplier,
            "historical_median_relative_return": median,
            "historical_mad": mad,
            "historical_robust_scale": robust_scale,
            "historical_minimum": float(values.min()) if len(values) else None,
            "historical_maximum": float(values.max()) if len(values) else None,
            "first_baseline_session": samples[0]["baseline_session"] if samples else None,
            "last_checkpoint_session": samples[-1]["checkpoint_session"] if samples else None,
        },
        "neutral_band_absolute_return": neutral_band,
        "classification_rules": classification,
        "data_lineage": bindings,
        "blockers": blockers,
        "governance": {
            "forecast_quality_is_direction_neutral": True,
            "negative_forecast_can_be_a_successful_forecast": True,
            "retroactive_binding_to_existing_reviewed_hypothesis_allowed": False,
            "bind_before_new_hypothesis_registration": True,
            "automatic_trading_allowed": False,
        },
    }


def classify_relative_total_return(
    realized_relative_total_return: float,
    direction_contract: dict[str, Any],
) -> dict[str, Any]:
    if direction_contract.get("contract") != RELATIVE_RETURN_DIRECTION_CONTRACT:
        raise ValueError("unsupported relative-return direction contract")
    if direction_contract.get("status") != "calibrated_pre_outcome_direction_contract":
        return {
            "classification": "unresolved",
            "reason": "direction_contract_not_calibrated",
        }
    direction = str(direction_contract.get("expected_direction") or "").lower()
    band = float(direction_contract.get("neutral_band_absolute_return") or 0.0)
    if direction not in {"positive", "negative"} or band <= 0:
        raise ValueError("invalid direction contract")
    realized = float(realized_relative_total_return)
    if abs(realized) < band:
        label = "neutral"
    elif direction == "negative":
        label = "support" if realized <= -band else "contradict"
    else:
        label = "support" if realized >= band else "contradict"
    return {
        "classification": label,
        "expected_direction": direction,
        "realized_relative_total_return": realized,
        "neutral_band_absolute_return": band,
        "distance_in_band_units": realized / band,
        "forecast_quality_is_direction_neutral": True,
    }


def validate_relative_return_direction_contract(
    contract: dict[str, Any], *, primary_horizon_days: int
) -> None:
    if contract.get("contract") != RELATIVE_RETURN_DIRECTION_CONTRACT:
        raise ValueError("relative-return direction contract mismatch")
    if contract.get("status") != "calibrated_pre_outcome_direction_contract":
        raise ValueError("relative-return direction contract is not calibrated")
    if int(contract.get("horizon_days") or 0) != int(primary_horizon_days):
        raise ValueError("relative-return direction horizon mismatch")
    if contract.get("expected_direction") not in {"positive", "negative"}:
        raise ValueError("relative-return expected direction invalid")
    if float(contract.get("neutral_band_absolute_return") or 0.0) <= 0:
        raise ValueError("relative-return neutral band invalid")
    if contract.get("blockers"):
        raise ValueError("relative-return direction contract has blockers")


def _classification_rules(direction: str, band: float | None) -> dict[str, Any]:
    if band is None:
        return {
            "support": None,
            "neutral": None,
            "contradict": None,
            "status": "pending_calibration",
        }
    if direction == "negative":
        support = f"realized_relative_total_return <= -{band:.12g}"
        contradict = f"realized_relative_total_return >= {band:.12g}"
    else:
        support = f"realized_relative_total_return >= {band:.12g}"
        contradict = f"realized_relative_total_return <= -{band:.12g}"
    return {
        "support": support,
        "neutral": f"abs(realized_relative_total_return) < {band:.12g}",
        "contradict": contradict,
        "boundary_rule": "support_and_contradict_are_inclusive; neutral_is_strict",
        "status": "ready",
    }


def _us_session_close_utc(session: Any):
    """Return the availability time of a daily US session observation."""
    import pandas as pd

    session_date = pd.Timestamp(session).date()
    close_local = datetime.combine(
        session_date,
        time(hour=16),
        tzinfo=ZoneInfo("America/New_York"),
    )
    return pd.Timestamp(close_local.astimezone(UTC))


def _load_prices(paths: list[Path]):
    import pandas as pd

    frames = []
    bindings = []
    for path in paths:
        if not path.is_file():
            continue
        raw = pd.read_parquet(path) if path.suffix.lower() == ".parquet" else pd.read_csv(path)
        ticker_col = next((c for c in raw.columns if str(c).lower() in {"ticker", "symbol"}), None)
        time_col = next((c for c in raw.columns if str(c).lower() in {"datetime", "timestamp", "date"}), None)
        close_col = next((c for c in raw.columns if str(c).lower() in {"adjusted_close", "adj_close", "close"}), None)
        if not all((ticker_col, time_col, close_col)):
            continue
        frame = raw[[ticker_col, time_col, close_col]].copy()
        frame.columns = ["ticker", "datetime", "close"]
        frame["ticker"] = frame["ticker"].astype(str).str.upper()
        frame["datetime"] = pd.to_datetime(frame["datetime"], utc=True, errors="coerce")
        frame["close"] = pd.to_numeric(frame["close"], errors="coerce")
        frames.append(frame.dropna())
        bindings.append(
            {
                "path": str(path),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "selected_price_column": str(close_col),
            }
        )
    if not frames:
        return pd.DataFrame(columns=["ticker", "datetime", "close"]), bindings
    combined = pd.concat(frames, ignore_index=True)
    combined = combined.sort_values("datetime").drop_duplicates(
        ["ticker", "datetime"], keep="last"
    )
    return combined, bindings


__all__ = [
    "RELATIVE_RETURN_DIRECTION_CONTRACT",
    "calibrate_relative_return_direction_contract",
    "classify_relative_total_return",
    "validate_relative_return_direction_contract",
]
