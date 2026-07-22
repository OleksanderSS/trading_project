from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from dean_os.base import BaseAgent
from dean_os.schemas import MarketContext, PipelineReport


class ContextSynthesisAgent(BaseAgent):
    """Compares Stage5 and Stage7 context contracts without deciding."""

    version = "0.1.0"
    branch = "pipeline"

    def check_prerequisites(self, context: MarketContext) -> bool:
        if not super().check_prerequisites(context):
            return False
        if not self.config.get("require_predecessor_data", True):
            return True
        run_phases = {
            str(item)
            for item in self.config.get(
                "run_phases",
                ["pre_trade"],
            )
        }
        prediction = context.metadata.get(
            "stage5_prediction_review"
        )
        regime = context.metadata.get("stage7_regime_review")
        return (
            context.phase in run_phases
            and isinstance(prediction, dict)
            and prediction.get("schema_version")
            == "dean_stage5_prediction_review_v1"
            and prediction.get("status")
            in {
                "stage5_prediction_review_ready",
                "stage5_prediction_review_partial",
            }
            and isinstance(regime, dict)
            and regime.get("schema_version")
            == "dean_stage7_regime_review_v1"
            and regime.get("status")
            == "stage7_regime_contexts_recorded"
        )

    async def run(self, context: MarketContext) -> PipelineReport:
        synthesis = build_context_synthesis(
            context,
            max_as_of_skew_minutes=float(
                self.config.get("max_as_of_skew_minutes", 60.0)
            ),
            min_prediction_confidence=float(
                self.config.get(
                    "min_prediction_confidence",
                    0.5,
                )
            ),
            min_anomaly_score=float(
                self.config.get("min_anomaly_score", 0.8)
            ),
        )
        context.metadata["context_synthesis"] = synthesis
        verdict = (
            "clear"
            if synthesis["status"] == "context_synthesis_ready"
            else "caution"
        )
        reasons = [
            "Stage5 prediction and Stage7 regime contracts were "
            f"compared for {synthesis['ticker']}/"
            f"{synthesis['timeframe']}; status="
            f"{synthesis['status']}. Shadow review only."
        ]
        risks = [
            item["message"]
            for item in synthesis["conflicts"]
        ]
        return PipelineReport(
            agent_name=self.name,
            agent_version=self.version,
            verdict=verdict,
            confidence=synthesis["review_confidence"],
            data_quality_score=0.75 if verdict == "clear" else 0.45,
            signal_strength=0.0,
            reasons=reasons,
            risks=risks,
            blind_spots=[
                "This agent checks context compatibility only; it does "
                "not infer forecast direction, validate a model, approve "
                "a thesis, or evaluate a realized outcome."
            ],
            evidence=[
                self.evidence(
                    "metric",
                    "context_synthesis",
                    "prediction_context_count",
                    synthesis["prediction_context_count"],
                ),
                self.evidence(
                    "metric",
                    "context_synthesis",
                    "regime_context_count",
                    synthesis["regime_context_count"],
                ),
                self.evidence(
                    "metric",
                    "context_synthesis",
                    "conflict_codes",
                    [
                        item["code"]
                        for item in synthesis["conflicts"]
                    ],
                ),
            ],
            input_hash=self.context_hash(context),
            metrics_snapshot=synthesis,
        )


def build_context_synthesis(
    context: MarketContext,
    *,
    max_as_of_skew_minutes: float = 60.0,
    min_prediction_confidence: float = 0.5,
    min_anomaly_score: float = 0.8,
) -> dict[str, Any]:
    prediction_review = _mapping(
        context.metadata.get("stage5_prediction_review")
    )
    regime_review = _mapping(
        context.metadata.get("stage7_regime_review")
    )
    if not regime_review:
        regime_review = _fallback_from_regime_context(context)
    specialist_review = _mapping(
        context.metadata.get("specialist_context_review")
    )
    ticker = (
        str(context.tickers[0]).upper()
        if len(context.tickers) == 1
        else None
    )
    timeframe = (
        str(context.timeframe).lower()
        if context.timeframe
        else (
            str(context.timeframes[0]).lower()
            if len(context.timeframes) == 1
            else None
        )
    )
    conflicts: list[dict[str, Any]] = []
    if ticker is None or timeframe is None:
        conflicts.append(
            _conflict(
                "context_identity_ambiguous",
                "MarketContext must identify exactly one ticker and "
                "one timeframe for synthesis.",
            )
        )
    predictions = [
        item
        for item in prediction_review.get("contexts", [])
        if isinstance(item, dict)
        and _same_context(item, ticker, timeframe)
    ]
    regimes = [
        item
        for item in regime_review.get("contexts", [])
        if isinstance(item, dict)
        and _same_context(item, ticker, timeframe)
    ]
    if not predictions:
        conflicts.append(
            _conflict(
                "prediction_context_missing",
                "No Stage5 prediction context exactly matches the "
                "requested ticker/timeframe.",
            )
        )
    if len(regimes) != 1:
        conflicts.append(
            _conflict(
                "regime_context_not_unique",
                "Exactly one Stage7 regime context must match the "
                "requested ticker/timeframe.",
                observed=len(regimes),
            )
        )
    regime = regimes[0] if len(regimes) == 1 else {}
    prediction_assessments = []
    for item in predictions:
        assessment, item_conflicts = _assess_prediction(
            item,
            regime,
            max_as_of_skew_minutes=max_as_of_skew_minutes,
            min_prediction_confidence=min_prediction_confidence,
            min_anomaly_score=min_anomaly_score,
        )
        prediction_assessments.append(assessment)
        conflicts.extend(item_conflicts)
    specialist_assessment, specialist_conflicts = (
        _assess_specialist_context(
            specialist_review,
            ticker=ticker,
            timeframe=timeframe,
        )
    )
    conflicts.extend(specialist_conflicts)
    conflict_codes = {item["code"] for item in conflicts}
    if not conflicts:
        status = "context_synthesis_ready"
    elif conflict_codes <= {
        "as_of_missing",
        "prediction_confidence_low",
        "prediction_anomaly_caution",
        "specialist_sector_context_only",
        "specialist_timeframe_unaligned",
        "specialist_evidence_age_exceeded",
        "specialist_manual_review_pending",
    }:
        status = "context_synthesis_caution"
    else:
        status = "context_synthesis_incompatible"
    confidence_values = [
        value
        for value in [
            _unit_value(regime.get("confidence")),
            *[
                _unit_value(
                    _mapping(item.get("prediction")).get(
                        "confidence"
                    )
                )
                for item in predictions
            ],
        ]
        if value is not None
    ]
    return {
        "schema_version": "dean_pipeline_context_synthesis_v1",
        "status": status,
        "ticker": ticker,
        "timeframe": timeframe,
        "prediction_context_count": len(predictions),
        "regime_context_count": len(regimes),
        "regime": {
            "regime": regime.get("regime"),
            "confidence": regime.get("confidence"),
            "as_of": regime.get("as_of"),
            "context_key": regime.get("context_key"),
        },
        "prediction_assessments": prediction_assessments,
        "specialist_assessment": specialist_assessment,
        "conflicts": conflicts,
        "review_confidence": (
            round(min(confidence_values), 4)
            if confidence_values
            else 0.0
        ),
        "supporting_review_only": True,
        "directional_synthesis_performed": False,
        "sector_context_promoted_to_ticker": False,
        "is_model_evaluation": False,
        "is_realized_outcome": False,
        "decision_influence": False,
        "can_promote_model": False,
        "can_write_learning_memory": False,
        "can_create_recommendation": False,
        "can_trade": False,
    }


def _assess_prediction(
    item: dict[str, Any],
    regime: dict[str, Any],
    *,
    max_as_of_skew_minutes: float,
    min_prediction_confidence: float,
    min_anomaly_score: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    conflicts: list[dict[str, Any]] = []
    context_key = item.get("context_key")
    if item.get("lineage_status") != "complete":
        conflicts.append(
            _conflict(
                "prediction_lineage_incomplete",
                "Stage5 prediction lineage is incomplete.",
                context_key=context_key,
                missing=item.get("missing_lineage_fields", []),
            )
        )
    review_issues = list(item.get("review_issues") or [])
    if review_issues:
        conflicts.append(
            _conflict(
                "prediction_review_issues",
                "Stage5 prediction review contains unresolved issues.",
                context_key=context_key,
                issues=review_issues,
            )
        )
    prediction = _mapping(item.get("prediction"))
    confidence = _unit_value(prediction.get("confidence"))
    anomaly = _unit_value(prediction.get("anomaly_score"))
    if confidence is not None and confidence < min_prediction_confidence:
        conflicts.append(
            _conflict(
                "prediction_confidence_low",
                "Prediction confidence is below the review floor.",
                context_key=context_key,
                observed=confidence,
                threshold=min_prediction_confidence,
            )
        )
    if anomaly is not None and anomaly < min_anomaly_score:
        conflicts.append(
            _conflict(
                "prediction_anomaly_caution",
                "Prediction anomaly score is below the review floor.",
                context_key=context_key,
                observed=anomaly,
                threshold=min_anomaly_score,
            )
        )
    prediction_as_of = _parse_timestamp(prediction.get("as_of"))
    regime_as_of = _parse_timestamp(regime.get("as_of"))
    skew_minutes = None
    if prediction_as_of is None or regime_as_of is None:
        conflicts.append(
            _conflict(
                "as_of_missing",
                "Prediction/regime as-of compatibility cannot be "
                "verified because at least one timestamp is missing.",
                context_key=context_key,
            )
        )
        freshness_status = "unverifiable"
    else:
        skew_minutes = abs(
            (prediction_as_of - regime_as_of).total_seconds()
        ) / 60.0
        if skew_minutes > max_as_of_skew_minutes:
            conflicts.append(
                _conflict(
                    "as_of_skew_exceeded",
                    "Prediction and regime as-of timestamps exceed the "
                    "allowed skew.",
                    context_key=context_key,
                    observed=round(skew_minutes, 4),
                    threshold=max_as_of_skew_minutes,
                )
            )
            freshness_status = "incompatible"
        else:
            freshness_status = "compatible"
    return {
        "context_key": context_key,
        "model_context_id": item.get("model_context_id"),
        "target_name": item.get("target_name"),
        "model_type": item.get("model_type"),
        "selected_primary_model": item.get(
            "selected_primary_model"
        ),
        "context_fingerprint": item.get("context_fingerprint"),
        "prediction_value": prediction.get("value"),
        "confidence": confidence,
        "anomaly_score": anomaly,
        "prediction_as_of": prediction.get("as_of"),
        "regime_as_of": regime.get("as_of"),
        "as_of_skew_minutes": (
            round(skew_minutes, 4)
            if skew_minutes is not None
            else None
        ),
        "freshness_status": freshness_status,
        "directional_comparison_performed": False,
    }, conflicts


def _assess_specialist_context(
    review: dict[str, Any],
    *,
    ticker: str | None,
    timeframe: str | None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if not review:
        return {
            "status": "not_supplied",
            "evidence_scope": None,
            "eligible_for_exact_pipeline_context": False,
            "directional_use_allowed": False,
        }, []
    conflicts: list[dict[str, Any]] = []
    if review.get("schema_version") != (
        "dean_specialist_context_review_v1"
    ):
        conflicts.append(
            _conflict(
                "invalid_specialist_contract",
                "Specialist context does not use the canonical schema.",
            )
        )
    requested = _mapping(review.get("requested_context"))
    review_ticker = str(
        requested.get("ticker") or ""
    ).upper()
    review_timeframe = str(
        requested.get("timeframe") or ""
    ).lower()
    if ticker and review_ticker != ticker:
        conflicts.append(
            _conflict(
                "specialist_ticker_mismatch",
                "Specialist context ticker does not match "
                "MarketContext.",
                expected=ticker,
                observed=review_ticker or None,
            )
        )
    ticker_scope = _mapping(review.get("ticker_scope"))
    evidence_scope = ticker_scope.get("evidence_scope")
    if evidence_scope == "sector_context_only":
        conflicts.append(
            _conflict(
                "specialist_sector_context_only",
                "Specialist evidence is sector-only and remains "
                "ineligible as ticker evidence.",
            )
        )
    timeframe_alignment = _mapping(
        review.get("timeframe_alignment")
    )
    if timeframe_alignment.get("status") != "aligned":
        conflicts.append(
            _conflict(
                "specialist_timeframe_unaligned",
                "Specialist evidence has no proven timeframe alignment.",
                expected=timeframe,
                observed=review_timeframe or None,
            )
        )
    point_in_time = _mapping(review.get("point_in_time"))
    point_status = point_in_time.get("status")
    if point_status == "future_evidence_conflict":
        conflicts.append(
            _conflict(
                "specialist_future_evidence",
                "Specialist evidence is later than the pipeline "
                "context.",
            )
        )
    elif point_status in {
        "older_than_review_window",
        "unverifiable_missing_as_of",
    }:
        conflicts.append(
            _conflict(
                "specialist_evidence_age_exceeded",
                "Specialist evidence freshness is stale or "
                "unverifiable for this context.",
                status=point_status,
            )
        )
    safety = _mapping(review.get("safety"))
    if safety.get("manual_review_required", True):
        conflicts.append(
            _conflict(
                "specialist_manual_review_pending",
                "Specialist ticker evidence still requires manual "
                "review.",
            )
        )
    return {
        "status": review.get("status"),
        "ticker": review_ticker or None,
        "timeframe": review_timeframe or None,
        "domain_id": _mapping(review.get("domain_scope")).get(
            "domain_id"
        ),
        "sector": _mapping(review.get("domain_scope")).get(
            "sector"
        ),
        "evidence_scope": evidence_scope,
        "point_in_time_status": point_status,
        "timeframe_alignment_status": (
            timeframe_alignment.get("status")
        ),
        "eligible_for_exact_pipeline_context": bool(
            safety.get("eligible_for_exact_pipeline_context", False)
        ),
        "eligible_as_approved_ticker_thesis": bool(
            ticker_scope.get(
                "eligible_as_approved_ticker_thesis",
                False,
            )
        ),
        "directional_use_allowed": False,
        "source_packet_fingerprint": review.get(
            "packet_fingerprint"
        ),
    }, conflicts


def _same_context(
    item: dict[str, Any],
    ticker: str | None,
    timeframe: str | None,
) -> bool:
    if ticker is None or timeframe is None:
        return False
    return (
        str(item.get("ticker") or "").upper() == ticker
        and str(item.get("timeframe") or "").lower() == timeframe
    )


def _fallback_from_regime_context(
    context: MarketContext,
) -> dict[str, Any]:
    regime = _mapping(context.metadata.get("regime_context"))
    if not regime.get("regime"):
        return {}
    ticker = (
        str(context.tickers[0]).upper()
        if len(context.tickers) == 1
        else None
    )
    timeframe = (
        str(context.timeframe).lower()
        if context.timeframe
        else (
            str(context.timeframes[0]).lower()
            if len(context.timeframes) == 1
            else None
        )
    )
    return {
        "schema_version": "dean_stage7_regime_review_v1",
        "status": "stage7_regime_contexts_recorded",
        "contexts": [
            {
                "ticker": ticker,
                "timeframe": timeframe,
                "regime": regime.get("regime"),
                "confidence": regime.get("confidence"),
                "as_of": context.as_of,
                "context_tags": regime.get("context_tags", []),
                "metrics": regime.get("metrics", {}),
            }
        ],
    }


def _parse_timestamp(value: Any) -> datetime | None:
    if value is None:
        return None
    try:
        parsed = datetime.fromisoformat(
            str(value).replace("Z", "+00:00")
        )
    except (TypeError, ValueError):
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _unit_value(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return parsed if 0.0 <= parsed <= 1.0 else None


def _mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _conflict(
    code: str,
    message: str,
    **details: Any,
) -> dict[str, Any]:
    return {
        "code": code,
        "message": message,
        "details": details,
    }
