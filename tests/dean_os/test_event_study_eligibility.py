from __future__ import annotations

from dean_os.event_study_eligibility import (
    EventStudyDesign,
    EventStudyEligibilityGate,
    EventStudyReadiness,
)


def _design() -> EventStudyDesign:
    return EventStudyDesign(
        event_id="policy-1",
        event_timestamp="2026-07-01T14:00:00+00:00",
        asset_ids=["NVDA"],
        benchmark_id="SOXX",
        timeframe="15m",
        event_time_alignment="market_session_exact",
    )


def test_event_study_blocks_before_data_and_time_alignment_are_verified() -> None:
    result = EventStudyEligibilityGate().evaluate(
        _design(), EventStudyReadiness()
    )

    assert result.status == "blocked"
    assert result.can_estimate_abnormal_returns is False
    assert result.causal_attribution_allowed is False
    assert "release_timestamp_not_verified" in result.blockers


def test_confounders_allow_measurement_but_not_clean_attribution() -> None:
    readiness = EventStudyReadiness(
        release_timestamp_verified=True,
        market_data_hash="market-sha",
        benchmark_data_hash="benchmark-sha",
        available_estimation_observations=100,
        event_window_complete=True,
        benchmark_window_complete=True,
        liquidity_evidence_available=True,
        volatility_evidence_available=True,
        unresolved_confounding_events=["earnings-call-same-window"],
        anticipation_risk="low",
    )

    result = EventStudyEligibilityGate().evaluate(_design(), readiness)

    assert result.status == "descriptive_only"
    assert result.can_estimate_abnormal_returns is True
    assert result.can_compute_car is True
    assert result.causal_attribution_allowed is False


def test_clean_inputs_enable_abnormal_return_estimation_not_causal_claim() -> None:
    readiness = EventStudyReadiness(
        release_timestamp_verified=True,
        market_data_hash="market-sha",
        benchmark_data_hash="benchmark-sha",
        available_estimation_observations=100,
        event_window_complete=True,
        benchmark_window_complete=True,
        liquidity_evidence_available=True,
        volatility_evidence_available=True,
        anticipation_risk="low",
    )

    result = EventStudyEligibilityGate().evaluate(_design(), readiness)

    assert result.status == "eligible_for_abnormal_return_estimation"
    assert result.can_estimate_abnormal_returns is True
    assert result.causal_attribution_allowed is False
