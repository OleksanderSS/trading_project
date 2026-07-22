from __future__ import annotations

import asyncio

from dean_os.agents.context_synthesis import (
    ContextSynthesisAgent,
    build_context_synthesis,
)
from dean_os.consensus import ConsensusEngine
from dean_os.schemas import MarketContext, PipelineReport


def _prediction_context(
    *,
    ticker: str = "NVDA",
    timeframe: str = "15m",
    context_key: str = "prediction_nvda",
    as_of: str | None = "2026-06-29T12:14:00+00:00",
    confidence: float = 0.72,
    anomaly_score: float = 0.91,
) -> dict:
    return {
        "context_key": context_key,
        "ticker": ticker,
        "timeframe": timeframe,
        "model_context_id": "ctx_nvda",
        "target_name": "target_intraday_up_15m",
        "model_type": "random_forest",
        "context_fingerprint": "fingerprint_nvda",
        "selected_primary_model": "random_forest",
        "lineage_status": "complete",
        "missing_lineage_fields": [],
        "review_issues": [],
        "prediction": {
            "value": 0.64,
            "confidence": confidence,
            "anomaly_score": anomaly_score,
            "as_of": as_of,
        },
    }


def _prediction_review(*contexts: dict) -> dict:
    return {
        "schema_version": "dean_stage5_prediction_review_v1",
        "status": "stage5_prediction_review_ready",
        "contexts": list(contexts),
    }


def _regime_review(
    *,
    ticker: str = "NVDA",
    timeframe: str = "15m",
    as_of: str | None = "2026-06-29T12:15:00+00:00",
) -> dict:
    return {
        "schema_version": "dean_stage7_regime_review_v1",
        "status": "stage7_regime_contexts_recorded",
        "contexts": [
            {
                "context_key": "ticker=NVDA|interval=15m",
                "ticker": ticker,
                "timeframe": timeframe,
                "regime": "TRENDING_UP",
                "confidence": 0.8,
                "as_of": as_of,
            }
        ],
    }


def _context(
    prediction_review: dict,
    regime_review: dict,
    *,
    ticker: str = "NVDA",
    timeframe: str = "15m",
) -> MarketContext:
    return MarketContext(
        phase="pre_trade",
        tickers=[ticker],
        timeframe=timeframe,
        metadata={
            "stage5_prediction_review": prediction_review,
            "stage7_regime_review": regime_review,
        },
    )


def _agent() -> ContextSynthesisAgent:
    return ContextSynthesisAgent(
        name="context_synthesis",
        config={
            "shadow_mode": True,
            "run_phases": ["pre_trade"],
            "max_as_of_skew_minutes": 60,
            "min_prediction_confidence": 0.5,
            "min_anomaly_score": 0.8,
        },
    )


def test_context_synthesis_matches_exact_context_and_freshness():
    context = _context(
        _prediction_review(_prediction_context()),
        _regime_review(),
    )

    result = build_context_synthesis(context)

    assert result["status"] == "context_synthesis_ready"
    assert result["prediction_context_count"] == 1
    assert result["regime_context_count"] == 1
    assert result["prediction_assessments"][0][
        "freshness_status"
    ] == "compatible"
    assert result["prediction_assessments"][0][
        "as_of_skew_minutes"
    ] == 1.0
    assert result["directional_synthesis_performed"] is False
    assert result["decision_influence"] is False
    assert result["can_trade"] is False


def test_context_synthesis_does_not_take_first_other_ticker():
    context = _context(
        _prediction_review(
            _prediction_context(ticker="MSFT")
        ),
        _regime_review(ticker="MSFT"),
        ticker="AMD",
    )

    result = build_context_synthesis(context)

    assert result["status"] == "context_synthesis_incompatible"
    assert result["prediction_context_count"] == 0
    assert result["regime_context_count"] == 0
    assert {item["code"] for item in result["conflicts"]} == {
        "prediction_context_missing",
        "regime_context_not_unique",
    }


def test_context_synthesis_records_as_of_skew_without_direction():
    context = _context(
        _prediction_review(
            _prediction_context(
                as_of="2026-06-29T09:00:00+00:00"
            )
        ),
        _regime_review(
            as_of="2026-06-29T12:15:00+00:00"
        ),
    )

    result = build_context_synthesis(
        context,
        max_as_of_skew_minutes=60,
    )

    assert result["status"] == "context_synthesis_incompatible"
    assert result["conflicts"][0]["code"] == "as_of_skew_exceeded"
    assert result["prediction_assessments"][0][
        "freshness_status"
    ] == "incompatible"
    assert result["prediction_assessments"][0][
        "directional_comparison_performed"
    ] is False


def test_context_synthesis_marks_missing_as_of_and_low_quality_caution():
    context = _context(
        _prediction_review(
            _prediction_context(
                as_of=None,
                confidence=0.4,
                anomaly_score=0.7,
            )
        ),
        _regime_review(as_of=None),
    )

    result = build_context_synthesis(context)

    assert result["status"] == "context_synthesis_caution"
    assert {item["code"] for item in result["conflicts"]} == {
        "prediction_confidence_low",
        "prediction_anomaly_caution",
        "as_of_missing",
    }


def test_context_synthesis_agent_is_pretrade_shadow_only():
    context = _context(
        _prediction_review(_prediction_context()),
        _regime_review(),
    )
    agent = _agent()

    context.phase = "pre_pipeline"
    assert agent.check_prerequisites(context) is False
    context.phase = "pre_trade"
    assert agent.check_prerequisites(context) is True
    report = asyncio.run(agent.run(context))

    assert report.verdict == "clear"
    assert report.signal_strength == 0.0
    assert report.metrics_snapshot["decision_influence"] is False
    assert context.metadata["context_synthesis"]["can_trade"] is False


def test_context_synthesis_shadow_report_cannot_change_consensus():
    context = _context(
        _prediction_review(
            _prediction_context(
                as_of="2026-06-29T09:00:00+00:00"
            )
        ),
        _regime_review(),
    )
    shadow_report = asyncio.run(_agent().run(context))
    risk = PipelineReport(
        agent_name="risk",
        agent_version="test",
        verdict="clear",
        confidence=0.8,
        data_quality_score=0.8,
        signal_strength=0.3,
    )
    engine = ConsensusEngine()
    pipeline_result = {"model_score": 0.4, "timeframe": "15m"}

    baseline = engine.combine([risk], pipeline_result, [])
    with_shadow = engine.combine(
        [risk, shadow_report],
        pipeline_result,
        [],
    )

    assert shadow_report.verdict == "caution"
    assert with_shadow.final_score == baseline.final_score
    assert with_shadow.decision == baseline.decision
    assert with_shadow.confidence == baseline.confidence


def test_context_synthesis_keeps_sector_only_specialist_separate():
    context = _context(
        _prediction_review(_prediction_context()),
        _regime_review(),
    )
    context.metadata["specialist_context_review"] = {
        "schema_version": "dean_specialist_context_review_v1",
        "status": "specialist_context_review_only_with_limits",
        "requested_context": {
            "ticker": "NVDA",
            "timeframe": "15m",
        },
        "domain_scope": {
            "domain_id": "semiconductor_ai_infrastructure",
            "sector": "semiconductor",
        },
        "ticker_scope": {
            "evidence_scope": "sector_context_only",
            "eligible_as_approved_ticker_thesis": False,
        },
        "point_in_time": {
            "status": "point_in_time_compatible"
        },
        "timeframe_alignment": {
            "status": "unverified_source_timeframe_not_declared"
        },
        "safety": {
            "manual_review_required": True,
            "eligible_for_exact_pipeline_context": False,
        },
    }

    result = build_context_synthesis(context)

    assert result["status"] == "context_synthesis_caution"
    assert result["specialist_assessment"]["evidence_scope"] == (
        "sector_context_only"
    )
    assert result["specialist_assessment"][
        "eligible_as_approved_ticker_thesis"
    ] is False
    assert result["sector_context_promoted_to_ticker"] is False
    assert {
        item["code"] for item in result["conflicts"]
    } >= {
        "specialist_sector_context_only",
        "specialist_timeframe_unaligned",
        "specialist_manual_review_pending",
    }


def test_context_synthesis_rejects_specialist_ticker_mismatch():
    context = _context(
        _prediction_review(_prediction_context()),
        _regime_review(),
    )
    context.metadata["specialist_context_review"] = {
        "schema_version": "dean_specialist_context_review_v1",
        "status": "specialist_context_review_only_with_limits",
        "requested_context": {
            "ticker": "AMD",
            "timeframe": "15m",
        },
        "ticker_scope": {
            "evidence_scope": "direct_ticker_review_candidate",
        },
        "point_in_time": {
            "status": "point_in_time_compatible"
        },
        "timeframe_alignment": {"status": "aligned"},
        "safety": {
            "manual_review_required": False,
            "eligible_for_exact_pipeline_context": True,
        },
    }

    result = build_context_synthesis(context)

    assert result["status"] == "context_synthesis_incompatible"
    assert "specialist_ticker_mismatch" in {
        item["code"] for item in result["conflicts"]
    }
