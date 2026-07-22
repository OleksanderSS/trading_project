from __future__ import annotations

__all__ = [
    'DEFAULT_BENCHMARK',
    'DEFAULT_SECTOR_TICKERS',
    'SAVED_SECTOR_MARKET_EVIDENCE_CONTRACT',
    'SUPPORTED_REPAIR_MODE',
    'SavedSectorMarketEvidenceProducer',
    'load_verified_sector_market_context_fragment',
    'render_sector_market_evidence_markdown',
]

import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from statistics import median, pstdev
from typing import Any

import pandas as pd

from dean_os.analysts.profiles import get_domain_profile
from dean_os.draft.dean_os_agent_system_v7.dean_os.artifact_writer import ReviewArtifactWriter
from dean_os.draft.dean_os_agent_system_v7.dean_os.context_evidence_provenance import parse_timezone_aware
from dean_os.schemas import utc_now_iso
from dean_os.draft.dean_os_agent_system_v7.dean_os.structured_context_provenance import audit_structured_context

SAVED_SECTOR_MARKET_EVIDENCE_CONTRACT = (
    "dean_saved_sector_market_evidence_producer_v1"
)
SUPPORTED_REPAIR_MODE = "pipeline_control_saved_price_repair"
_DOMAIN_ID = "semiconductor_ai_infrastructure"
DEFAULT_SECTOR_TICKERS = tuple(get_domain_profile(_DOMAIN_ID).ticker_universe_hint)
DEFAULT_BENCHMARK = "QQQ"


class SavedSectorMarketEvidenceProducer:
    """Build review-only sector market confirmation from repaired prices."""

    def __init__(
        self,
        output_dir: str | Path = (
            "reports/dean_os/saved_sector_market_evidence_producer"
        ),
    ):
        self.output_dir = Path(output_dir)

    def build(
        self,
        *,
        repair_artifact_path: str | Path,
        as_of: str,
        sector_tickers: list[str] | None = None,
        benchmark: str = DEFAULT_BENCHMARK,
        lookback_sessions: int = 20,
        min_source_bars_per_day: int = 24,
        max_staleness_days: int = 7,
        save: bool = True,
    ) -> dict[str, Any]:
        as_of_dt = parse_timezone_aware(as_of)
        if as_of_dt is None:
            raise ValueError(
                "sector market producer as_of must be a timezone-aware "
                "ISO-8601 timestamp"
            )
        if lookback_sessions < 2:
            raise ValueError("lookback_sessions must be at least 2")

        requested = sorted(
            {
                str(ticker).upper().strip()
                for ticker in (
                    sector_tickers or list(DEFAULT_SECTOR_TICKERS)
                )
                if str(ticker).strip()
            }
        )
        benchmark = str(benchmark).upper().strip()
        run_id = _run_id()
        input_path = Path(repair_artifact_path)
        payload: dict[str, Any] = {
            "run_id": run_id,
            "created_at": utc_now_iso(),
            "mode": "saved_sector_market_evidence_producer",
            "producer_contract": SAVED_SECTOR_MARKET_EVIDENCE_CONTRACT,
            "inputs": {
                "repair_artifact_path": str(input_path),
                "as_of": as_of_dt.isoformat(),
                "sector_tickers": requested,
                "benchmark": benchmark,
                "lookback_sessions": lookback_sessions,
                "min_source_bars_per_day": min_source_bars_per_day,
                "max_staleness_days": max_staleness_days,
            },
        }

        repair_payload, repair_source, reasons = _load_repair_artifact(
            input_path
        )
        payload["source_artifact"] = repair_source
        if reasons:
            payload.update(_blocked_payload(reasons))
            return self._finish(payload, save=save)

        lineage, lineage_reasons = _verify_lineage(repair_payload)
        payload["lineage"] = lineage
        if lineage_reasons:
            payload.update(_blocked_payload(lineage_reasons))
            return self._finish(payload, save=save)

        repair_created_at = parse_timezone_aware(
            repair_payload.get("created_at")
        )
        if repair_created_at is None:
            reasons.append("repair_artifact_created_at_invalid")
        elif repair_created_at > as_of_dt:
            reasons.append("repair_artifact_available_after_as_of")

        daily_path = Path(lineage["daily_artifact"]["path"])
        try:
            frame = pd.read_parquet(daily_path)
        except (OSError, ValueError, ImportError) as exc:
            payload["load_error"] = str(exc)
            reasons.append("daily_price_artifact_unreadable")
            payload.update(_blocked_payload(reasons))
            return self._finish(payload, save=save)

        normalized = _normalize_market_window(
            frame,
            sector_tickers=requested,
            benchmark=benchmark,
            as_of=as_of_dt,
            lookback_sessions=lookback_sessions,
            min_source_bars_per_day=min_source_bars_per_day,
            max_staleness_days=max_staleness_days,
        )
        payload["data_quality"] = normalized["data_quality"]
        reasons.extend(normalized["blocking_reasons"])
        if reasons:
            payload.update(_blocked_payload(reasons))
            payload["exclusions"] = normalized["exclusions"]
            return self._finish(payload, save=save)

        metrics = _build_metrics(
            normalized=normalized,
            sector_tickers=requested,
            benchmark=benchmark,
            available_at=repair_created_at.isoformat(),
            source_locator=str(daily_path),
            source_sha256=lineage["daily_artifact"]["sha256"],
        )
        sector_data = {
            item["name"]: {
                "value": item["value"],
                "unit": item["unit"],
                "period": item["period"],
                "available_at": item["available_at"],
                "source_locator": item["source_locator"],
                "metadata": {
                    "evidence_type": item["evidence_type"],
                    "required_lane_eligible": item[
                        "required_lane_eligible"
                    ],
                    "stance_hint": item["stance_hint"],
                    "source_artifact_sha256": item[
                        "source_artifact_sha256"
                    ],
                    "sector_tickers": requested,
                    "benchmark": benchmark,
                    "lookback_sessions": lookback_sessions,
                },
            }
            for item in metrics
        }
        audit = audit_structured_context(
            fundamentals={},
            macro={},
            sector_data=sector_data,
            as_of=as_of_dt.isoformat(),
        )
        accepted = audit["accepted_context"]["sector_data"]
        status = (
            "sector_market_evidence_ready"
            if accepted
            and audit["accepted_count"] == len(metrics)
            and audit["excluded_count"] == 0
            else "blocked_structured_sector_market_evidence"
        )
        payload.update(
            {
                "status": status,
                "summary": {
                    "sector_ticker_count": len(requested),
                    "sector_ticker_coverage_ratio": normalized[
                        "data_quality"
                    ]["sector_ticker_coverage_ratio"],
                    "common_session_count": normalized["data_quality"][
                        "common_session_count"
                    ],
                    "lookback_sessions": lookback_sessions,
                    "accepted_metric_count": audit["accepted_count"],
                    "accepted_fingerprint": audit[
                        "accepted_fingerprint"
                    ],
                    "required_market_confirmation_ready": (
                        status == "sector_market_evidence_ready"
                    ),
                    "can_enter_market_context_review": (
                        status == "sector_market_evidence_ready"
                    ),
                    "can_train": False,
                    "can_influence_ticker_prediction": False,
                    "can_trade": False,
                },
                "metrics": metrics,
                "exclusions": normalized["exclusions"],
                "structured_context_audit": {
                    key: value
                    for key, value in audit.items()
                    if key
                    not in {
                        "accepted_context",
                        "accepted_observations",
                    }
                },
                "market_context_fragment": {
                    "as_of": as_of_dt.isoformat(),
                    "sector_data": accepted,
                    "metadata": {
                        "saved_sector_market_producer_run_id": run_id,
                        "saved_sector_market_source_sha256": lineage[
                            "daily_artifact"
                        ]["sha256"],
                        "saved_sector_market_accepted_fingerprint": audit[
                            "accepted_fingerprint"
                        ],
                        "sector_tickers": requested,
                        "benchmark": benchmark,
                        "required_market_confirmation_ready": (
                            status == "sector_market_evidence_ready"
                        ),
                    },
                },
                "integration_boundary": {
                    "review_only": True,
                    "market_confirmation_only": True,
                    "sector_thesis_not_created_by_this_producer": True,
                    "ticker_forecast_created": False,
                    "pipeline_feature_promotion_allowed": False,
                    "training_allowed": False,
                    "automatic_trading_allowed": False,
                },
                "safety": _safety(),
            }
        )
        return self._finish(payload, save=save)

    def _finish(
        self,
        payload: dict[str, Any],
        *,
        save: bool,
    ) -> dict[str, Any]:
        payload.setdefault("safety", _safety())
        if save:
            payload["saved_paths"] = ReviewArtifactWriter(
                self.output_dir
            ).write(
                payload=payload,
                markdown=render_sector_market_evidence_markdown(
                    payload
                ),
                run_id=payload["run_id"],
            )
        return payload


def load_verified_sector_market_context_fragment(
    artifact_path: str | Path,
    *,
    expected_as_of: str | None = None,
) -> dict[str, Any]:
    path = Path(artifact_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        payload.get("producer_contract")
        != SAVED_SECTOR_MARKET_EVIDENCE_CONTRACT
    ):
        raise ValueError("unsupported sector market evidence contract")
    if payload.get("status") != "sector_market_evidence_ready":
        raise ValueError("sector market evidence artifact is not ready")
    summary = payload.get("summary", {})
    safety = payload.get("safety", {})
    if (
        summary.get("required_market_confirmation_ready") is not True
        or summary.get("can_trade") is not False
        or summary.get("can_train") is not False
        or safety.get("review_only") is not True
        or safety.get("pipeline_run_performed") is not False
        or safety.get("live_execution_performed") is not False
    ):
        raise ValueError("sector market evidence safety boundary invalid")

    fragment = payload.get("market_context_fragment")
    if not isinstance(fragment, dict):
        raise ValueError("sector market context fragment missing")
    fragment_as_of = parse_timezone_aware(fragment.get("as_of"))
    if fragment_as_of is None:
        raise ValueError("sector market context as_of invalid")
    if expected_as_of is not None:
        expected = parse_timezone_aware(expected_as_of)
        if expected is None or expected != fragment_as_of:
            raise ValueError("sector market context expected as_of mismatch")

    source = payload.get("source_artifact", {})
    source_path = Path(str(source.get("path") or ""))
    if (
        not source_path.exists()
        or _sha256_file(source_path) != source.get("sha256")
    ):
        raise ValueError("sector market repair artifact hash mismatch")
    repair_payload = json.loads(source_path.read_text(encoding="utf-8"))
    lineage, reasons = _verify_lineage(repair_payload)
    if reasons:
        raise ValueError("sector market source lineage invalid")
    expected_lineage = payload.get("lineage", {})
    for key in ("raw_source", "daily_artifact"):
        if lineage.get(key) != expected_lineage.get(key):
            raise ValueError("sector market lineage changed")

    sector_data = fragment.get("sector_data")
    if not isinstance(sector_data, dict):
        raise ValueError("sector market context payload invalid")
    audit = audit_structured_context(
        fundamentals={},
        macro={},
        sector_data=sector_data,
        as_of=fragment_as_of.isoformat(),
    )
    if (
        audit["excluded_count"] != 0
        or audit["accepted_count"]
        != summary.get("accepted_metric_count")
        or audit["accepted_fingerprint"]
        != summary.get("accepted_fingerprint")
    ):
        raise ValueError("sector market context fingerprint mismatch")
    return {
        "as_of": fragment_as_of.isoformat(),
        "sector_data": audit["accepted_context"]["sector_data"],
        "metadata": {
            **dict(fragment.get("metadata", {})),
            "saved_sector_market_artifact_path": str(path),
            "saved_sector_market_artifact_sha256": _sha256_file(path),
            "saved_sector_market_verified": True,
        },
    }


def _load_repair_artifact(
    path: Path,
) -> tuple[dict[str, Any], dict[str, Any], list[str]]:
    if not path.exists():
        return {}, {"path": str(path), "exists": False}, [
            "repair_artifact_missing"
        ]
    try:
        initial = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}, {"path": str(path), "exists": True}, [
            "repair_artifact_unreadable"
        ]
    exact_path = Path(
        str(initial.get("saved_paths", {}).get("json") or path)
    )
    if exact_path.exists():
        try:
            exact = json.loads(exact_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            exact = {}
    else:
        exact = {}
    reasons: list[str] = []
    if not exact:
        reasons.append("immutable_repair_artifact_missing_or_unreadable")
    elif exact.get("run_id") != initial.get("run_id"):
        reasons.append("repair_latest_pointer_run_mismatch")
    source = {
        "path": str(exact_path),
        "exists": exact_path.exists(),
        "sha256": (
            _sha256_file(exact_path) if exact_path.exists() else None
        ),
        "run_id": exact.get("run_id") if exact else None,
        "mode": exact.get("mode") if exact else None,
    }
    return exact, source, reasons


def _verify_lineage(
    repair_payload: dict[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    reasons: list[str] = []
    summary = repair_payload.get("summary", {})
    artifact_safety = repair_payload.get("artifact_safety", {})
    if repair_payload.get("mode") != SUPPORTED_REPAIR_MODE:
        reasons.append("unsupported_repair_artifact_mode")
    if (
        summary.get("repair_status")
        != "non_destructive_price_candidates_ready"
    ):
        reasons.append("repair_artifact_not_ready")
    if (
        summary.get("can_train") is not False
        or summary.get("can_trade") is not False
    ):
        reasons.append("repair_safety_boundary_invalid")
    if (
        artifact_safety.get("learning_write_performed") is not False
        or artifact_safety.get("live_execution_performed") is not False
    ):
        reasons.append("repair_artifact_safety_invalid")
    if summary.get("cross_ticker_identity_groups") != 0:
        reasons.append("cross_ticker_price_identity_detected")

    raw = repair_payload.get("source_provenance", {})
    daily = repair_payload.get("artifacts", {}).get(
        "prices_1d_resampled", {}
    )
    raw_path = Path(str(raw.get("path") or ""))
    daily_path = Path(str(daily.get("path") or ""))
    if (
        not raw_path.exists()
        or not raw.get("sha256")
        or _sha256_file(raw_path) != raw.get("sha256")
    ):
        reasons.append("raw_price_source_hash_mismatch")
    if (
        not daily_path.exists()
        or not daily.get("sha256")
        or _sha256_file(daily_path) != daily.get("sha256")
    ):
        reasons.append("daily_price_artifact_hash_mismatch")
    if (
        daily.get("synthetic") is not False
        or daily.get("derived_from_observed_bars") is not True
    ):
        reasons.append("daily_price_artifact_lineage_invalid")
    return {
        "raw_source": {
            "path": str(raw_path),
            "sha256": raw.get("sha256"),
            "synthetic": raw.get("synthetic"),
        },
        "daily_artifact": {
            "path": str(daily_path),
            "sha256": daily.get("sha256"),
            "synthetic": daily.get("synthetic"),
            "derived_from_observed_bars": daily.get(
                "derived_from_observed_bars"
            ),
        },
    }, sorted(set(reasons))


def _normalize_market_window(
    frame: pd.DataFrame,
    *,
    sector_tickers: list[str],
    benchmark: str,
    as_of: Any,
    lookback_sessions: int,
    min_source_bars_per_day: int,
    max_staleness_days: int,
) -> dict[str, Any]:
    required_columns = {
        "datetime",
        "ticker",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "interval",
        "source_bar_count",
    }
    missing_columns = sorted(required_columns - set(frame.columns))
    reasons: list[str] = []
    exclusions: list[dict[str, Any]] = []
    if missing_columns:
        return {
            "blocking_reasons": ["daily_price_schema_missing_columns"],
            "exclusions": [
                {
                    "status": "excluded",
                    "reasons": ["daily_price_schema_missing_columns"],
                    "missing_columns": missing_columns,
                }
            ],
            "data_quality": {
                "missing_columns": missing_columns,
                "sector_ticker_coverage_ratio": 0.0,
                "common_session_count": 0,
            },
        }

    data = frame.copy()
    data["ticker"] = data["ticker"].astype(str).str.upper()
    data["datetime"] = pd.to_datetime(
        data["datetime"], utc=True, errors="coerce"
    )
    wanted = set(sector_tickers) | {benchmark}
    data = data[data["ticker"].isin(wanted)].copy()
    present = sorted(set(data["ticker"]))
    missing_tickers = sorted(wanted - set(present))
    missing_sector = sorted(set(sector_tickers) - set(present))
    if missing_tickers:
        reasons.append("required_market_ticker_coverage_incomplete")
    duplicated = data.duplicated(["ticker", "datetime"], keep=False)
    if duplicated.any():
        reasons.append("duplicate_ticker_session_rows")
    if data["datetime"].isna().any():
        reasons.append("invalid_market_timestamp")
    if (data["datetime"] > pd.Timestamp(as_of)).any():
        reasons.append("market_observation_after_as_of")
    if not data["interval"].eq("1d").all():
        reasons.append("non_daily_interval_in_daily_artifact")

    numeric_columns = [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "source_bar_count",
    ]
    for column in numeric_columns:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    finite = data[numeric_columns].apply(
        lambda column: column.map(
            lambda value: (
                value is not None
                and not pd.isna(value)
                and math.isfinite(float(value))
            )
        )
    )
    if not finite.all(axis=None):
        reasons.append("non_finite_market_value")
    if (
        (data[["open", "high", "low", "close"]] <= 0).any(axis=None)
        or (data["volume"] < 0).any()
    ):
        reasons.append("invalid_ohlcv_value")
    if (
        (data["high"] < data[["open", "close", "low"]].max(axis=1)).any()
        or (data["low"] > data[["open", "close", "high"]].min(axis=1)).any()
    ):
        reasons.append("invalid_ohlc_ordering")
    if (data["source_bar_count"] < min_source_bars_per_day).any():
        reasons.append("daily_row_source_bar_count_below_minimum")

    dates_by_ticker = {
        ticker: set(group["datetime"].tolist())
        for ticker, group in data.groupby("ticker")
    }
    common_dates = (
        sorted(
            set.intersection(
                *(dates_by_ticker[ticker] for ticker in sorted(wanted))
            )
        )
        if not missing_tickers and wanted
        else []
    )
    required_common = lookback_sessions + 1
    if len(common_dates) < required_common:
        reasons.append("insufficient_common_market_sessions")
    selected_dates = common_dates[-required_common:]
    last_observation = max(common_dates) if common_dates else None
    staleness_days = (
        (pd.Timestamp(as_of) - last_observation).total_seconds() / 86400
        if last_observation is not None
        else None
    )
    if staleness_days is None or staleness_days > max_staleness_days:
        reasons.append("sector_market_evidence_stale")

    selected = data[data["datetime"].isin(selected_dates)].copy()
    selected = selected.sort_values(["ticker", "datetime"])
    rows_by_ticker = {
        ticker: int(count)
        for ticker, count in selected.groupby("ticker").size().items()
    }
    for reason in sorted(set(reasons)):
        exclusions.append({"status": "excluded", "reasons": [reason]})
    coverage_ratio = (
        (len(sector_tickers) - len(missing_sector))
        / len(sector_tickers)
        if sector_tickers
        else 0.0
    )
    return {
        "blocking_reasons": sorted(set(reasons)),
        "exclusions": exclusions,
        "selected": selected,
        "selected_dates": selected_dates,
        "data_quality": {
            "input_row_count": len(frame),
            "selected_universe_row_count": len(data),
            "selected_window_row_count": len(selected),
            "requested_sector_tickers": sector_tickers,
            "benchmark": benchmark,
            "present_tickers": present,
            "missing_tickers": missing_tickers,
            "sector_ticker_coverage_ratio": coverage_ratio,
            "common_session_count": len(common_dates),
            "selected_common_session_count": len(selected_dates),
            "rows_by_ticker": rows_by_ticker,
            "first_selected_session": (
                selected_dates[0].isoformat() if selected_dates else None
            ),
            "last_selected_session": (
                selected_dates[-1].isoformat() if selected_dates else None
            ),
            "staleness_days": staleness_days,
            "minimum_source_bars_per_day": (
                int(data["source_bar_count"].min())
                if not data.empty
                else None
            ),
        },
    }


def _build_metrics(
    *,
    normalized: dict[str, Any],
    sector_tickers: list[str],
    benchmark: str,
    available_at: str,
    source_locator: str,
    source_sha256: str,
) -> list[dict[str, Any]]:
    selected = normalized["selected"]
    selected_dates = normalized["selected_dates"]
    returns: dict[str, float] = {}
    for ticker, group in selected.groupby("ticker"):
        ordered = group.sort_values("datetime")
        returns[ticker] = float(
            ordered["close"].iloc[-1] / ordered["close"].iloc[0] - 1.0
        )
    sector_returns = [returns[ticker] for ticker in sector_tickers]
    benchmark_return = returns[benchmark]
    median_return = median(sector_returns)
    breadth = sum(value > 0 for value in sector_returns) / len(
        sector_returns
    )
    excess = median(
        value - benchmark_return for value in sector_returns
    )
    dispersion = pstdev(sector_returns)
    period = (
        f"{selected_dates[0].date().isoformat()}/"
        f"{selected_dates[-1].date().isoformat()}"
    )

    def item(
        name: str,
        value: float,
        unit: str,
        stance_hint: str,
        *,
        eligible: bool,
    ) -> dict[str, Any]:
        return {
            "name": name,
            "value": float(value),
            "unit": unit,
            "period": period,
            "available_at": available_at,
            "source_locator": source_locator,
            "source_artifact_sha256": source_sha256,
            "evidence_type": "market_confirmation",
            "required_lane_eligible": eligible,
            "stance_hint": stance_hint,
        }

    metrics = [
        item(
            "sector_median_return_20_session",
            median_return * 100.0,
            "percent",
            _direction(median_return),
            eligible=True,
        ),
        item(
            "sector_positive_breadth",
            breadth,
            "ratio",
            _direction(breadth - 0.5),
            eligible=True,
        ),
        item(
            "sector_median_excess_return_vs_qqq",
            excess * 100.0,
            "percentage_points",
            _direction(excess),
            eligible=True,
        ),
        item(
            "sector_return_dispersion_20_session",
            dispersion * 100.0,
            "percent",
            "unknown",
            eligible=False,
        ),
        item(
            "sector_ticker_coverage",
            normalized["data_quality"]["sector_ticker_coverage_ratio"],
            "ratio",
            "unknown",
            eligible=False,
        ),
        item(
            "sector_common_session_count",
            normalized["data_quality"]["common_session_count"],
            "count",
            "unknown",
            eligible=False,
        ),
    ]
    for ticker in sector_tickers:
        metrics.append(
            item(
                f"{ticker.lower()}_return_20_session",
                returns[ticker] * 100.0,
                "percent",
                _direction(returns[ticker]),
                eligible=False,
            )
        )
    metrics.append(
        item(
            f"{benchmark.lower()}_return_20_session",
            benchmark_return * 100.0,
            "percent",
            _direction(benchmark_return),
            eligible=False,
        )
    )
    return metrics


def _direction(value: float, tolerance: float = 1e-12) -> str:
    if value > tolerance:
        return "positive"
    if value < -tolerance:
        return "negative"
    return "neutral"


def _blocked_payload(reasons: list[str]) -> dict[str, Any]:
    reason_counts = dict(Counter(sorted(set(reasons))))
    return {
        "status": "blocked_sector_market_evidence",
        "summary": {
            "reason_counts": reason_counts,
            "required_market_confirmation_ready": False,
            "can_enter_market_context_review": False,
            "can_train": False,
            "can_influence_ticker_prediction": False,
            "can_trade": False,
        },
        "market_context_fragment": {
            "sector_data": {},
        },
        "integration_boundary": {
            "review_only": True,
            "fail_closed": True,
            "market_confirmation_only": True,
            "training_allowed": False,
            "automatic_trading_allowed": False,
        },
        "safety": _safety(),
    }


def _safety() -> dict[str, Any]:
    return {
        "review_only": True,
        "network_access_performed": False,
        "collector_run_performed": False,
        "pipeline_run_performed": False,
        "training_run_performed": False,
        "tuning_run_performed": False,
        "learning_write_performed": False,
        "production_config_write_performed": False,
        "broker_access_performed": False,
        "live_execution_performed": False,
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run_id() -> str:
    return (
        "saved_sector_market_evidence_"
        + utc_now_iso().replace(":", "").replace("+", "Z")
    )


def render_sector_market_evidence_markdown(
    payload: dict[str, Any],
) -> str:
    summary = payload.get("summary", {})
    lines = [
        "# Saved Sector Market Evidence",
        "",
        f"- Status: `{payload.get('status')}`",
        f"- As of: `{payload.get('inputs', {}).get('as_of')}`",
        (
            "- Sector tickers: `"
            + ", ".join(payload.get("inputs", {}).get(
                "sector_tickers", []
            ))
            + "`"
        ),
        (
            "- Benchmark: `"
            + str(payload.get("inputs", {}).get("benchmark"))
            + "`"
        ),
        (
            "- Common sessions: `"
            + str(summary.get("common_session_count", 0))
            + "`"
        ),
        (
            "- Required market confirmation ready: `"
            + str(
                summary.get("required_market_confirmation_ready", False)
            ).lower()
            + "`"
        ),
        "",
        "This artifact is review-only sector market context. It cannot train, "
        "tune, predict a ticker, or trade.",
        "",
    ]
    return "\n".join(lines)
