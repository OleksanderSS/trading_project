from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from dean_os.artifact_writer import ReviewArtifactWriter
from dean_os.relative_return_direction_policy import classify_relative_total_return
from dean_os.schemas import utc_now_iso
from dean_os.world_model.world_model_replay_registration import (
    WORLD_MODEL_REPLAY_REGISTRATION_CONTRACT,
)


HISTORICAL_REPLAY_OUTCOME_REVIEW_CONTRACT = (
    "dean_historical_replay_outcome_review_v1"
)


class HistoricalReplayOutcomeReview:
    """Audit matured replay checkpoints without inventing missing outcomes."""

    def __init__(
        self,
        output_dir: str | Path = "reports/dean_os/historical_replay_outcome_review_current",
    ) -> None:
        self.output_dir = Path(output_dir)

    def build(
        self,
        *,
        review_gate_json: str | Path,
        registration_json: str | Path,
        price_paths: list[str | Path] | None = None,
        pipeline_paths: list[str | Path] | None = None,
        task_ids: list[str] | None = None,
        save: bool = True,
    ) -> dict[str, Any]:
        gate_path = Path(review_gate_json)
        registration_path = Path(registration_json)
        gate = _load(gate_path)
        registration = _load(registration_path)
        _verify(gate_path, gate, registration)
        reviews = {
            str(item.get("hypothesis_id")): item
            for item in gate.get("hypothesis_review", []) or []
            if item.get("hypothesis_id")
        }
        discovered = [
            Path(item)
            for item in (
                price_paths if price_paths is not None else _default_price_paths()
            )
        ]
        pipeline_artifacts = [
            Path(item)
            for item in (
                pipeline_paths
                if pipeline_paths is not None
                else _default_pipeline_paths()
            )
        ]
        price_inventory = _price_inventory(discovered)
        price_series = _price_series(discovered)
        pipeline_inventory = _pipeline_inventory(pipeline_artifacts)
        task_reviews: list[dict[str, Any]] = []
        outcomes: list[dict[str, Any]] = []
        plan_by_id = {
            str(item.get("task_id")): item
            for item in registration.get("registration_plan", []) or []
            if item.get("task_id")
        }
        if task_ids is None:
            selected_tasks = list(
                registration.get("deferred_historical_tasks", []) or []
            )
            review_scope = "deferred_historical_tasks"
        else:
            requested = list(dict.fromkeys(str(item) for item in task_ids))
            missing_task_ids = [item for item in requested if item not in plan_by_id]
            if missing_task_ids:
                raise ValueError(
                    "requested task IDs are absent from registration plan: "
                    + ", ".join(missing_task_ids)
                )
            selected_tasks = [plan_by_id[item] for item in requested]
            review_scope = "explicit_matured_task_ids"
        for task in selected_tasks:
            task_id = str(task.get("task_id") or "")
            plan = plan_by_id.get(task_id, {})
            hypothesis_id = str(plan.get("hypothesis_id") or "")
            review = reviews.get(hypothesis_id, {})
            measurement = dict(review.get("measurement_spec") or {})
            primary_horizon = _positive_int(measurement.get("primary_horizon_days"))
            horizon = _positive_int(plan.get("horizon_days"))
            required = _required_evidence(review)
            coverage = _coverage(required, price_inventory)
            price_observation = _checkpoint_price_observation(
                required=required,
                measurement=measurement,
                price_series=price_series,
                event_anchor_at=plan.get("event_anchor_at") or plan.get("as_of"),
                due_at=plan.get("due_at") or task.get("due_at"),
            )
            pipeline_context = _pipeline_context(required, pipeline_inventory)
            missing = list(required["non_price_requirements"])
            missing.extend(coverage["missing_price_requirements"])
            missing.extend(price_observation.get("missing_requirements") or [])
            checkpoint_role = (
                "primary_hypothesis_outcome"
                if horizon is not None and horizon == primary_horizon
                else "intermediate_event_response_checkpoint"
            )
            status = (
                "unobservable_primary_outcome_missing_point_in_time_evidence"
                if checkpoint_role == "primary_hypothesis_outcome" and missing
                else "unresolved_intermediate_checkpoint_missing_point_in_time_evidence"
                if missing
                else "ready_for_manual_causal_outcome_review"
            )
            item = {
                "task_id": task_id,
                "hypothesis_id": hypothesis_id,
                "horizon_days": horizon,
                "due_at": plan.get("due_at") or task.get("due_at"),
                "checkpoint_role": checkpoint_role,
                "review_status": status,
                "required_evidence": required,
                "price_coverage": coverage,
                "price_observation": price_observation,
                "pipeline_context": pipeline_context,
                "missing_point_in_time_evidence": list(dict.fromkeys(missing)),
                "result_label": (
                    "unobservable"
                    if checkpoint_role == "primary_hypothesis_outcome" and missing
                    else "unresolved"
                ),
                "outcome_scoring_performed": False,
            }
            task_reviews.append(item)
            if checkpoint_role == "primary_hypothesis_outcome":
                outcomes.append(
                    {
                        "outcome_id": "historical_outcome_" + hashlib.sha256(
                            task_id.encode("utf-8")
                        ).hexdigest()[:24],
                        "hypothesis_id": hypothesis_id,
                        "task_id": task_id,
                        "horizon_days": horizon,
                        "horizon_family": review.get("horizon_family")
                        or "event_response_fixed_v1",
                        "result_label": (
                            "unobservable" if missing else "inconclusive"
                        ),
                        "observable": not missing,
                        "coverage_status": (
                            "insufficient" if missing else "ready_for_causal_review"
                        ),
                        "data_quality_status": (
                            "incomplete" if missing else "verified_inputs_present"
                        ),
                        "observations": [price_observation]
                        if price_observation.get("status") != "not_applicable"
                        else [],
                        "missing_point_in_time_evidence": list(
                            dict.fromkeys(missing)
                        ),
                        "alternative_explanations": [
                            "The hypothesis may be correct or false; the current local evidence cannot distinguish those states."
                        ],
                        "automatic_outcome_scoring_allowed": False,
                        "human_causal_attribution_required": True,
                    }
                )
        payload: dict[str, Any] = {
            "run_id": "historical_replay_outcome_review_"
            + utc_now_iso().replace(":", "").replace("+00:00", "Z"),
            "created_at": utc_now_iso(),
            "mode": "historical_replay_outcome_review",
            "contract": HISTORICAL_REPLAY_OUTCOME_REVIEW_CONTRACT,
            "inputs": {
                "review_gate": _binding(gate_path),
                "registration": _binding(registration_path),
                "price_paths": [_binding(path) for path in discovered if path.is_file()],
                "pipeline_paths": [
                    _binding(path) for path in pipeline_artifacts if path.is_file()
                ],
                "review_scope": review_scope,
                "selected_task_ids": [
                    str(item.get("task_id")) for item in selected_tasks
                ],
            },
            "summary": {
                "historical_task_count": len(task_reviews),
                "reviewed_task_count": len(task_reviews),
                "primary_outcome_count": len(outcomes),
                "unobservable_primary_outcome_count": sum(
                    item.get("result_label") == "unobservable" for item in outcomes
                ),
                "intermediate_unresolved_count": sum(
                    item.get("checkpoint_role")
                    == "intermediate_event_response_checkpoint"
                    and item.get("result_label") == "unresolved"
                    for item in task_reviews
                ),
                "outcome_scoring_performed": False,
                "learning_memory_write_performed": False,
                "can_trade": False,
            },
            "price_inventory": price_inventory,
            "pipeline_inventory": pipeline_inventory,
            "checkpoint_reviews": task_reviews,
            "outcomes": outcomes,
            "recommended_actions": _recommended_actions(task_reviews),
            "safety": {
                "review_only": True,
                "missing_evidence_is_never_confirmation_or_falsification": True,
                "outcome_scoring_performed": False,
                "learning_memory_write_performed": False,
                "production_rule_update_performed": False,
                "broker_access_performed": False,
                "can_trade": False,
            },
        }
        if save:
            payload["saved_paths"] = ReviewArtifactWriter(self.output_dir).write(
                payload=payload,
                markdown=_markdown(payload),
                run_id=payload["run_id"],
            )
        return payload


def _required_evidence(review: dict[str, Any]) -> dict[str, Any]:
    measurement = dict(review.get("measurement_spec") or {})
    context = dict(measurement.get("measurement_context") or {})
    metrics = [str(item) for item in measurement.get("target_metrics") or []]
    price_tickers: list[str] = []
    minimum_price_coverage = 1
    basket = context.get("capital_equipment_basket") or {}
    if basket:
        price_tickers.extend(str(item).upper() for item in basket.get("members") or [])
        if basket.get("benchmark"):
            price_tickers.append(str(basket["benchmark"]).upper())
        minimum_price_coverage = int(basket.get("minimum_checkpoint_coverage") or 1)
        if basket.get("benchmark"):
            minimum_price_coverage += 1
    elif any("amat" in item.lower() for item in metrics):
        price_tickers.append("AMAT")
    non_price: list[str] = []
    if any("consensus" in item.lower() for item in metrics):
        non_price.append("point_in_time_consensus_estimate_baseline_and_checkpoint")
    if any("capex_plan_revision" in item.lower() for item in metrics):
        non_price.append("point_in_time_public_capex_plan_checkpoint_updates")
    if (
        any("relative" in item.lower() and "return" in item.lower() for item in metrics)
        and not basket
    ):
        non_price.append("predeclared_price_benchmark_for_relative_return")
    if not price_tickers and any("return" in item.lower() for item in metrics):
        non_price.append("predeclared_price_universe_and_benchmark")
    return {
        "target_metrics": metrics,
        "price_tickers": list(dict.fromkeys(price_tickers)),
        "minimum_price_ticker_coverage": minimum_price_coverage,
        "non_price_requirements": non_price,
    }


def _price_inventory(paths: list[Path]) -> dict[str, Any]:
    artifacts: list[dict[str, Any]] = []
    ticker_union: set[str] = set()
    for path in paths:
        if not path.is_file():
            continue
        try:
            import pandas as pd

            frame = (
                pd.read_parquet(path)
                if path.suffix.lower() == ".parquet"
                else pd.read_csv(path)
            )
            ticker_col = next(
                (col for col in frame.columns if str(col).lower() in {"ticker", "symbol"}),
                None,
            )
            time_col = next(
                (
                    col
                    for col in frame.columns
                    if "date" in str(col).lower() or "time" in str(col).lower()
                ),
                None,
            )
            tickers = (
                sorted({str(item).upper() for item in frame[ticker_col].dropna()})
                if ticker_col
                else []
            )
            ticker_union.update(tickers)
            artifacts.append(
                {
                    "path": str(path),
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "row_count": len(frame),
                    "tickers": tickers,
                    "minimum_timestamp": (
                        str(frame[time_col].min()) if time_col else None
                    ),
                    "maximum_timestamp": (
                        str(frame[time_col].max()) if time_col else None
                    ),
                    "read_status": "readable",
                }
            )
        except Exception as exc:
            artifacts.append(
                {"path": str(path), "read_status": "unreadable", "error": str(exc)}
            )
    return {"artifacts": artifacts, "available_tickers": sorted(ticker_union)}


def _price_series(paths: list[Path]) -> dict[str, list[dict[str, Any]]]:
    """Load only dated closes; inventory presence alone is not checkpoint evidence."""
    import pandas as pd

    rows: dict[str, dict[str, float]] = {}
    for path in paths:
        if not path.is_file():
            continue
        try:
            frame = (
                pd.read_parquet(path)
                if path.suffix.lower() == ".parquet"
                else pd.read_csv(path)
            )
            ticker_col = next(
                (col for col in frame.columns if str(col).lower() in {"ticker", "symbol"}),
                None,
            )
            time_col = next(
                (
                    col
                    for col in frame.columns
                    if str(col).lower() in {"datetime", "timestamp", "date"}
                ),
                None,
            )
            close_col = next(
                (
                    col
                    for col in frame.columns
                    if str(col).lower() in {"adjusted_close", "adj_close", "close"}
                ),
                None,
            )
            if not all((ticker_col, time_col, close_col)):
                continue
            selected = frame[[ticker_col, time_col, close_col]].copy()
            selected[time_col] = pd.to_datetime(selected[time_col], utc=True, errors="coerce")
            selected[close_col] = pd.to_numeric(selected[close_col], errors="coerce")
            selected = selected.dropna()
            for ticker, timestamp, close in selected.itertuples(index=False, name=None):
                key = str(ticker).upper()
                session = timestamp.date().isoformat()
                rows.setdefault(key, {})[session] = float(close)
        except Exception:
            continue
    return {
        ticker: [
            {"session": session, "close": close}
            for session, close in sorted(values.items())
        ]
        for ticker, values in rows.items()
    }


def _checkpoint_price_observation(
    *,
    required: dict[str, Any],
    measurement: dict[str, Any],
    price_series: dict[str, list[dict[str, Any]]],
    event_anchor_at: Any,
    due_at: Any,
) -> dict[str, Any]:
    requested = list(required.get("price_tickers") or [])
    if not requested:
        return {"status": "not_applicable", "missing_requirements": []}
    try:
        import pandas as pd

        anchor = pd.Timestamp(event_anchor_at)
        due = pd.Timestamp(due_at)
        anchor = anchor.tz_localize("UTC") if anchor.tzinfo is None else anchor.tz_convert("UTC")
        due = due.tz_localize("UTC") if due.tzinfo is None else due.tz_convert("UTC")
    except Exception:
        return {
            "status": "checkpoint_window_unavailable",
            "missing_requirements": ["valid_event_anchor_and_checkpoint_due_time"],
        }

    # The clean daily artifact is session-dated at midnight. During US daylight
    # saving time, a regular close is 20:00 UTC. Events after that close may use
    # the same date as baseline; due times after it require the next session.
    close_hour_utc = 20
    anchor_date = anchor.date().isoformat()
    due_date = due.date().isoformat()

    def window(ticker: str) -> dict[str, Any] | None:
        values = price_series.get(ticker) or []
        baseline_candidates = [
            item
            for item in values
            if item["session"] < anchor_date
            or (item["session"] == anchor_date and anchor.hour >= close_hour_utc)
        ]
        checkpoint_candidates = [
            item
            for item in values
            if item["session"] > due_date
            or (item["session"] == due_date and due.hour < close_hour_utc)
        ]
        if not baseline_candidates or not checkpoint_candidates:
            return None
        baseline = baseline_candidates[-1]
        checkpoint = checkpoint_candidates[0]
        if not baseline["close"]:
            return None
        return {
            "ticker": ticker,
            "baseline_session": baseline["session"],
            "baseline_close": baseline["close"],
            "checkpoint_session": checkpoint["session"],
            "checkpoint_close": checkpoint["close"],
            "price_return": checkpoint["close"] / baseline["close"] - 1.0,
        }

    windows = [item for ticker in requested if (item := window(ticker))]
    minimum = int(required.get("minimum_price_ticker_coverage") or 0)
    missing: list[str] = []
    if len(windows) < minimum:
        missing.append(
            f"verified_price_checkpoint_window_below_floor:{len(windows)}/{minimum}"
        )
    context = dict(measurement.get("measurement_context") or {})
    basket = dict(context.get("capital_equipment_basket") or {})
    result: dict[str, Any] = {
        "status": "checkpoint_price_window_observed" if not missing else "checkpoint_window_incomplete",
        "event_anchor_at": str(event_anchor_at),
        "checkpoint_due_at": str(due_at),
        "session_resolution": "daily_session_proxy_us_close_20_utc_summer",
        "return_quality": "price-return input; adjustment semantics must be read from the bound source lineage",
        "ticker_windows": windows,
        "missing_requirements": missing,
        "automatic_hypothesis_scoring_allowed": False,
    }
    if basket:
        members = {str(item).upper() for item in basket.get("members") or []}
        benchmark = str(basket.get("benchmark") or "").upper()
        member_windows = [item for item in windows if item["ticker"] in members]
        benchmark_window = next(
            (item for item in windows if item["ticker"] == benchmark), None
        )
        member_floor = int(basket.get("minimum_checkpoint_coverage") or 1)
        if len(member_windows) >= member_floor and benchmark_window:
            basket_return = sum(item["price_return"] for item in member_windows) / len(member_windows)
            benchmark_return = benchmark_window["price_return"]
            active_spread = basket_return - benchmark_return
            relative_total_return = (
                (1.0 + basket_return) / (1.0 + benchmark_return) - 1.0
            )
            result["relative_return_observation"] = {
                "definition": "equal_weight_initial_value_basket_relative_wealth_ratio_to_predeclared_benchmark",
                "member_count": len(member_windows),
                "basket_weighting": "equal_weight_at_baseline_no_intra_window_rebalance",
                "basket_price_return": basket_return,
                "benchmark": benchmark,
                "benchmark_price_return": benchmark_return,
                "active_return_spread_percentage_points": active_spread,
                "relative_total_return": relative_total_return,
                "relative_price_return": relative_total_return,
                "formula": "(1 + basket_return) / (1 + benchmark_return) - 1",
                "observed_sign": "negative" if relative_total_return < 0 else "positive" if relative_total_return > 0 else "flat",
                "market_performance_interpretation": (
                    "basket_outperformed_benchmark"
                    if relative_total_return > 0
                    else "basket_underperformed_benchmark"
                    if relative_total_return < 0
                    else "basket_matched_benchmark"
                ),
                "forecast_quality_interpretation": "not inherently good or bad; score against the predeclared claim direction at the primary horizon",
            }
            direction_contract = measurement.get(
                "relative_return_direction_contract"
            )
            if isinstance(direction_contract, dict):
                result["relative_return_observation"]["claim_relation"] = (
                    classify_relative_total_return(
                        relative_total_return, direction_contract
                    )
                )
            else:
                result["relative_return_observation"]["claim_relation"] = {
                    "classification": "not_scored",
                    "reason": "predeclared_relative_return_direction_contract_missing",
                }
        else:
            requirement = "predeclared_basket_and_benchmark_checkpoint_window"
            result["missing_requirements"].append(requirement)
            result["status"] = "checkpoint_window_incomplete"
    else:
        result["relative_return_observation"] = {
            "status": "not_computed",
            "reason": "benchmark_not_predeclared_in_measurement_spec",
        }
    result["missing_requirements"] = list(dict.fromkeys(result["missing_requirements"]))
    return result


def _pipeline_inventory(paths: list[Path]) -> dict[str, Any]:
    artifacts: list[dict[str, Any]] = []
    ticker_union: set[str] = set()
    feature_union: set[str] = set()
    for path in paths:
        if not path.is_file():
            continue
        try:
            import pandas as pd
            import pyarrow.parquet as pq

            columns = [str(item) for item in pq.ParquetFile(path).schema.names]
            ticker_col = next(
                (col for col in columns if col.lower() in {"ticker", "symbol"}),
                None,
            )
            time_col = next(
                (
                    col
                    for col in columns
                    if col.lower() in {"datetime", "timestamp", "date"}
                ),
                None,
            ) or next(
                (
                    col
                    for col in columns
                    if col.lower().endswith(("_datetime", "_timestamp", "_date"))
                ),
                None,
            )
            selected = [item for item in (ticker_col, time_col) if item]
            frame = pd.read_parquet(path, columns=selected) if selected else None
            tickers = (
                sorted({str(item).upper() for item in frame[ticker_col].dropna()})
                if frame is not None and ticker_col
                else []
            )
            relevant = sorted(
                {
                    col
                    for col in columns
                    if any(
                        term in col.lower()
                        for term in (
                            "return",
                            "price",
                            "close",
                            "news",
                            "sentiment",
                            "context",
                            "capex",
                            "consensus",
                            "estimate",
                        )
                    )
                }
            )
            ticker_union.update(tickers)
            feature_union.update(relevant)
            artifacts.append(
                {
                    "path": str(path),
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "row_count": len(frame) if frame is not None else None,
                    "column_count": len(columns),
                    "tickers": tickers,
                    "relevant_context_features": relevant,
                    "minimum_timestamp": (
                        str(frame[time_col].min())
                        if frame is not None and time_col
                        else None
                    ),
                    "maximum_timestamp": (
                        str(frame[time_col].max())
                        if frame is not None and time_col
                        else None
                    ),
                    "read_status": "readable_with_lineage",
                }
            )
        except Exception as exc:
            artifacts.append(
                {"path": str(path), "read_status": "unreadable", "error": str(exc)}
            )
    return {
        "artifacts": artifacts,
        "available_tickers": sorted(ticker_union),
        "relevant_context_features": sorted(feature_union),
    }


def _pipeline_context(
    required: dict[str, Any], inventory: dict[str, Any]
) -> dict[str, Any]:
    requested = set(required.get("price_tickers") or [])
    available = set(inventory.get("available_tickers") or [])
    overlap = sorted(requested & available)
    features = list(inventory.get("relevant_context_features") or [])
    metrics = [str(item).lower() for item in required.get("target_metrics") or []]
    direct_metric_features = sorted(
        {
            feature
            for feature in features
            if any(
                metric.replace("_", " ") in feature.lower().replace("_", " ")
                for metric in metrics
            )
        }
    )
    status = (
        "direct_target_evidence_candidate_requires_checkpoint_validation"
        if requested and requested <= available and direct_metric_features
        else "partial_target_universe_secondary_context_only"
        if overlap
        else "sector_or_market_secondary_context_only"
    )
    return {
        "status": status,
        "requested_target_tickers_present": overlap,
        "direct_target_metric_features": direct_metric_features,
        "available_context_feature_count": len(features),
        "usable_context_feature_families": sorted(
            {
                family
                for family in ("return", "price", "news", "sentiment", "context")
                if any(family in feature.lower() for feature in features)
            }
        ),
        "can_replace_missing_primary_outcome_evidence": False,
        "allowed_use": "regime_confounder_and_relative_context_review_only",
    }


def _coverage(required: dict[str, Any], inventory: dict[str, Any]) -> dict[str, Any]:
    requested = list(required.get("price_tickers") or [])
    available = set(inventory.get("available_tickers") or [])
    present = [item for item in requested if item in available]
    missing = [item for item in requested if item not in available]
    minimum = int(required.get("minimum_price_ticker_coverage") or 0)
    requirements: list[str] = []
    if requested and len(present) < minimum:
        requirements.append(
            "verified_price_coverage_below_floor:"
            + f"{len(present)}/{minimum};missing={','.join(missing)}"
        )
    return {
        "requested_tickers": requested,
        "present_tickers": present,
        "missing_tickers": missing,
        "minimum_ticker_coverage": minimum,
        "coverage_floor_met": not requirements,
        "missing_price_requirements": requirements,
    }


def _recommended_actions(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    missing = sorted(
        {
            requirement
            for item in items
            for requirement in item.get("missing_point_in_time_evidence") or []
        }
    )
    return [
        {
            "priority": index + 1,
            "requirement": requirement,
            "action": "collect_or_reconstruct_verified_point_in_time_evidence_then_rerun",
            "automatic_outcome_label_allowed": False,
        }
        for index, requirement in enumerate(missing)
    ]


def _verify(gate_path: Path, gate: dict[str, Any], registration: dict[str, Any]) -> None:
    if registration.get("contract") != WORLD_MODEL_REPLAY_REGISTRATION_CONTRACT:
        raise ValueError("unsupported replay registration contract")
    source_gate = registration.get("source_gate") or {}
    if source_gate.get("run_id") != gate.get("run_id"):
        raise ValueError("registration points to a different review gate")
    if source_gate.get("sha256") != hashlib.sha256(gate_path.read_bytes()).hexdigest():
        raise ValueError("review gate changed after registration")


def _default_price_paths() -> list[Path]:
    candidates: list[Path] = []
    clean = Path("data/dean_os/clean_market_snapshots")
    processed = Path("data/processed")
    if clean.is_dir():
        candidates.extend(sorted(clean.glob("*.parquet"), key=lambda p: p.stat().st_mtime)[-1:])
    if processed.is_dir():
        candidates.extend(
            sorted(processed.glob("prices_1d_*.parquet"), key=lambda p: p.stat().st_mtime)[-1:]
        )
    return candidates


def _default_pipeline_paths() -> list[Path]:
    return [
        path
        for path in (
            Path("data/colab/accumulated/main_database/features.parquet"),
            Path("data/colab/regenerated/semiconductor_clean_1d_stage23/features.parquet"),
        )
        if path.is_file()
    ]


def _binding(path: Path) -> dict[str, Any]:
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"artifact must be a JSON object: {path}")
    return payload


def _positive_int(value: Any) -> int | None:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _markdown(payload: dict[str, Any]) -> str:
    summary = payload["summary"]
    lines = [
        "# DEAN-OS Historical Replay Outcome Review",
        "",
        f"- Historical tasks: {summary['historical_task_count']}",
        f"- Primary outcomes: {summary['primary_outcome_count']}",
        f"- Unobservable primary outcomes: {summary['unobservable_primary_outcome_count']}",
        f"- Intermediate unresolved checkpoints: {summary['intermediate_unresolved_count']}",
        "- Outcome scoring performed: false",
        "- Can trade: false",
        "",
    ]
    for item in payload.get("checkpoint_reviews", []):
        lines.extend(
            [
                f"## `{item.get('task_id')}`",
                "",
                f"- Role: `{item.get('checkpoint_role')}`",
                f"- Status: `{item.get('review_status')}`",
                f"- Result: `{item.get('result_label')}`",
                f"- Missing: {', '.join(item.get('missing_point_in_time_evidence') or []) or 'none'}",
                f"- Pipeline context: `{(item.get('pipeline_context') or {}).get('status')}`",
                f"- Pipeline target overlap: {', '.join((item.get('pipeline_context') or {}).get('requested_target_tickers_present') or []) or 'none'}",
                f"- Price window: `{(item.get('price_observation') or {}).get('status')}`",
                f"- Relative return: `{((item.get('price_observation') or {}).get('relative_return_observation') or {}).get('relative_price_return', 'not scored')}`",
                "",
            ]
        )
    lines.extend(
        [
            "## Boundary",
            "",
            "Missing evidence is an observability failure, not confirmation or falsification of the hypothesis.",
            "",
        ]
    )
    return "\n".join(lines)


__all__ = [
    "HISTORICAL_REPLAY_OUTCOME_REVIEW_CONTRACT",
    "HistoricalReplayOutcomeReview",
]
