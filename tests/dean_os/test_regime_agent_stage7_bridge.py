from __future__ import annotations

import asyncio

from dean_os.agents.regime import RegimeAgent, build_regime_snapshot
from dean_os.consensus import ConsensusEngine
from dean_os.schemas import MarketContext, PipelineReport


def _stage7_review() -> dict:
    return {
        "schema_version": "dean_stage7_regime_review_v1",
        "status": "stage7_regime_contexts_recorded",
        "context_count": 2,
        "contexts": [
            {
                "context_key": "ticker=NVDA|interval=15m",
                "ticker": "NVDA",
                "timeframe": "15m",
                "identity_status": "exact_context_key",
                "regime": "TRENDING_UP",
                "confidence": 0.8,
                "metrics": {"trend_strength": 0.4},
                "supporting_review_only": True,
                "decision_influence": False,
                "can_promote_model": False,
                "can_trade": False,
            },
            {
                "context_key": "ticker=MSFT|interval=15m",
                "ticker": "MSFT",
                "timeframe": "15m",
                "identity_status": "exact_context_key",
                "regime": "RANGING",
                "confidence": 0.6,
                "metrics": {"trend_strength": 0.03},
                "supporting_review_only": True,
                "decision_influence": False,
                "can_promote_model": False,
                "can_trade": False,
            },
        ],
        "evidence_class": "supporting_analysis_not_locked_evidence",
        "can_clear_locked_evidence": False,
        "can_promote_model": False,
        "can_trade": False,
    }


def _agent() -> RegimeAgent:
    return RegimeAgent(
        name="regime",
        config={
            "require_stage7_regime_review": True,
            "shadow_mode": True,
            "run_phases": ["pre_trade"],
        },
    )


def test_regime_agent_selects_exact_stage7_ticker_timeframe_partition():
    context = MarketContext(
        phase="pre_trade",
        tickers=["NVDA"],
        timeframe="15m",
        metadata={"stage7_regime_review": _stage7_review()},
    )

    report = asyncio.run(_agent().run(context))

    assert report.verdict == "clear"
    assert report.signal_strength == 0.0
    assert report.metrics_snapshot["regime"] == "TRENDING_UP"
    assert report.metrics_snapshot["metrics"]["ticker"] == "NVDA"
    assert report.metrics_snapshot["metrics"]["timeframe"] == "15m"
    assert report.metrics_snapshot["decision_influence"] is False
    assert report.metrics_snapshot["can_write_learning_memory"] is False
    assert report.metrics_snapshot["can_trade"] is False


def test_regime_agent_does_not_fallback_to_another_partition():
    context = MarketContext(
        phase="pre_trade",
        tickers=["AMD"],
        timeframe="15m",
        metadata={"stage7_regime_review": _stage7_review()},
    )

    report = asyncio.run(_agent().run(context))

    assert report.verdict == "caution"
    assert report.metrics_snapshot["regime"] == "UNKNOWN"
    assert "No unique Stage 7 regime context" in report.risks[0]
    assert report.metrics_snapshot["decision_influence"] is False


def test_strict_stage7_mode_never_uses_dataframe_or_file_fallback():
    context = MarketContext(
        tickers=["NVDA"],
        timeframe="15m",
        metadata={
            "regime_context": {
                "regime": "BULL_MARKET",
                "confidence": 1.0,
                "source": "stale_legacy_context",
            }
        },
    )

    snapshot = build_regime_snapshot(
        context=context,
        ticker="NVDA",
        latest_processed_prices="1d",
        require_stage7_review=True,
    )

    assert snapshot.regime == "UNKNOWN"
    assert snapshot.source == "stage7_market_regime_review"
    assert "fallbacks were not used" in snapshot.warnings[0]


def test_regime_agent_loads_only_pretrade_with_stage7_review():
    agent = _agent()
    context = MarketContext(
        phase="pre_pipeline",
        tickers=["NVDA"],
        timeframe="15m",
        metadata={"stage7_regime_review": _stage7_review()},
    )

    assert agent.check_prerequisites(context) is False
    context.phase = "pre_trade"
    assert agent.check_prerequisites(context) is True


def test_shadow_regime_report_does_not_change_consensus():
    risk_report = PipelineReport(
        agent_name="risk",
        agent_version="test",
        verdict="clear",
        confidence=0.7,
        data_quality_score=0.8,
        signal_strength=0.4,
    )
    shadow_regime = PipelineReport(
        agent_name="regime",
        agent_version="test",
        verdict="caution",
        confidence=1.0,
        data_quality_score=1.0,
        signal_strength=-1.0,
        reasons=["visible supporting context"],
        metrics_snapshot={
            "decision_influence": False,
            "supporting_review_only": True,
        },
    )
    engine = ConsensusEngine()
    pipeline_result = {"model_score": 0.3, "timeframe": "15m"}

    baseline = engine.combine([risk_report], pipeline_result, [])
    with_shadow = engine.combine(
        [risk_report, shadow_regime],
        pipeline_result,
        [],
    )

    assert with_shadow.final_score == baseline.final_score
    assert with_shadow.decision == baseline.decision
    assert with_shadow.confidence == baseline.confidence
    assert "visible supporting context" in with_shadow.reasons
