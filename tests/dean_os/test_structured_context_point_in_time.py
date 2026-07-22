from __future__ import annotations

import asyncio

import pytest

from dean_os.agents.domain_research import (
    MacroPolicyAgent,
    ValueScreeningAgent,
)
from dean_os.analysts.context_adapter import (
    MarketContextEvidenceAdapter,
)
from dean_os.schemas import MarketContext
from dean_os.structured_context_provenance import (
    STRUCTURED_CONTEXT_CONTRACT,
    apply_market_context_structured_boundary,
    audit_structured_context,
)


AS_OF = "2026-06-30T12:00:00+00:00"
AVAILABLE_AT = "2026-06-29T16:00:00+00:00"


def _fundamentals(*, available_at: str = AVAILABLE_AT):
    return {
        "AMD": {
            "metrics": {
                "pe": {
                    "value": 10.0,
                    "unit": "ratio",
                    "period": "FY2025",
                },
                "fcf_yield": {
                    "value": 0.08,
                    "unit": "ratio",
                    "period": "FY2025",
                },
                "roe": {
                    "value": 0.15,
                    "unit": "ratio",
                    "period": "FY2025",
                },
            },
            "available_at": available_at,
            "source_url": (
                "https://example.test/filings/amd-fy2025"
            ),
        }
    }


def _macro(*, available_at: str = AVAILABLE_AT):
    return {
        "cpi": {
            "value": 2.8,
            "unit": "percent_yoy",
            "period": "2026-05",
            "available_at": available_at,
            "source_url": "https://example.test/macro/cpi-2026-05",
        }
    }


def _sector():
    return {
        "semiconductor_demand": {
            "value": "expanding",
            "unit": "qualitative_state",
            "period": "2026-Q2",
            "available_at": AVAILABLE_AT,
            "source_url": (
                "https://example.test/sector/semiconductor-2026-q2"
            ),
        }
    }


def _clean_gate_metadata(*, fingerprint: str | None = None):
    if fingerprint is None:
        fingerprint = audit_structured_context(
            fundamentals=_fundamentals(),
            macro={},
            sector_data={},
            as_of=AS_OF,
        )["accepted_fingerprint"]
    return {
        "gate_attached": True,
        "readiness_status": "fundamental_input_ready_for_manual_review",
        "can_feed_value_screening_after_manual_review": True,
        "structured_accepted_fingerprint": fingerprint,
    }


def test_structured_audit_accepts_semantic_point_in_time_observations():
    audit = audit_structured_context(
        fundamentals=_fundamentals(),
        macro=_macro(),
        sector_data=_sector(),
        as_of=AS_OF,
    )

    assert audit["contract"] == STRUCTURED_CONTEXT_CONTRACT
    assert audit["status"] == "point_in_time_ready"
    assert audit["accepted_count"] == 5
    assert audit["family_counts"] == {
        "fundamental": 3,
        "macro": 1,
        "sector": 1,
    }
    assert audit["accepted_context"]["fundamentals"]["AMD"]["pe"] == 10.0
    observation = next(
        item
        for item in audit["accepted_observations"]
        if item["family"] == "macro"
    )
    assert observation["unit"] == "percent_yoy"
    assert observation["period"] == "2026-05"
    assert observation["available_at"] == AVAILABLE_AT
    assert observation["source_locator"].startswith("https://")


def test_structured_audit_fails_closed_on_semantics_and_future_data():
    audit = audit_structured_context(
        fundamentals={
            "AMD": {
                "pe": 10.0,
                "available_at": "2026-07-01T00:00:00+00:00",
            }
        },
        macro={"cpi": {"value": 2.8}},
        sector_data={},
        as_of=AS_OF,
    )

    assert audit["accepted_count"] == 0
    assert audit["status"] == (
        "blocked_no_point_in_time_structured_context"
    )
    assert audit["reason_counts"][
        "structured_availability_after_as_of"
    ] == 1
    assert audit["reason_counts"]["structured_unit_missing"] == 2
    assert audit["reason_counts"]["structured_period_missing"] == 2
    assert audit["reason_counts"][
        "structured_source_locator_missing"
    ] == 2


def test_structured_audit_requires_timezone_aware_cutoff():
    with pytest.raises(ValueError):
        audit_structured_context(
            fundamentals=_fundamentals(),
            macro={},
            sector_data={},
            as_of="2026-06-30T12:00:00",
        )


def test_normalized_context_can_be_reaudited_without_losing_provenance():
    first = audit_structured_context(
        fundamentals=_fundamentals(),
        macro=_macro(),
        sector_data=_sector(),
        as_of=AS_OF,
    )
    accepted = first["accepted_context"]
    second = audit_structured_context(
        fundamentals=accepted["fundamentals"],
        macro=accepted["macro"],
        sector_data=accepted["sector_data"],
        as_of=AS_OF,
    )

    assert second["accepted_count"] == first["accepted_count"]
    assert (
        second["accepted_fingerprint"]
        == first["accepted_fingerprint"]
    )


def test_structured_semantics_survive_normalization_and_reaudit():
    sector = {
        "market_breadth": {
            "value": 0.75,
            "unit": "ratio",
            "period": "2026-06",
            "available_at": AVAILABLE_AT,
            "source_url": "https://example.test/sector/breadth",
            "metadata": {
                "evidence_type": "market_confirmation",
                "required_lane_eligible": True,
                "stance_hint": "positive",
            },
        }
    }
    first = audit_structured_context(
        fundamentals={},
        macro={},
        sector_data=sector,
        as_of=AS_OF,
    )
    second = audit_structured_context(
        fundamentals={},
        macro={},
        sector_data=first["accepted_context"]["sector_data"],
        as_of=AS_OF,
    )

    observation = second["accepted_observations"][0]
    assert observation["evidence_type"] == "market_confirmation"
    assert observation["required_lane_eligible"] is True
    assert observation["stance_hint"] == "positive"
    assert (
        second["accepted_fingerprint"]
        == first["accepted_fingerprint"]
    )


def test_context_adapter_emits_one_semantic_item_per_observation():
    context = MarketContext(
        as_of=AS_OF,
        tickers=["AMD"],
        fundamentals=_fundamentals(),
        macro=_macro(),
        sector_data=_sector(),
    )

    packet = MarketContextEvidenceAdapter("macro_policy").adapt(
        context,
        as_of=AS_OF,
    )

    assert packet["summary"]["evidence_count"] == 5
    macro_item = next(
        item for item in packet["evidence"] if item.directness == "macro"
    )
    assert macro_item.point_in_time["unit"] == "percent_yoy"
    assert macro_item.point_in_time["period"] == "2026-05"
    assert (
        macro_item.provenance["contract"]
        == STRUCTURED_CONTEXT_CONTRACT
    )


def test_direct_macro_agent_cannot_count_unbounded_structured_data():
    context = MarketContext(
        macro={"cpi": 2.8},
    )

    report = asyncio.run(
        MacroPolicyAgent(name="macro_policy", config={}).run(context)
    )

    assert context.macro == {}
    assert report.verdict == "needs_more_data"
    audit = context.metadata[
        "structured_context_point_in_time_audit"
    ]
    assert audit["status"] == "blocked_context_as_of_missing"


def test_value_screening_requires_semantic_contract_and_clean_gate():
    context = MarketContext(
        as_of=AS_OF,
        tickers=["AMD"],
        fundamentals=_fundamentals(),
        metadata={
            "fundamental_input_readiness_gate": _clean_gate_metadata()
        },
    )

    report = asyncio.run(
        ValueScreeningAgent(
            name="value_screening",
            config={},
        ).run(context)
    )

    assert report.verdict == "undervalued"
    assert report.ticker == "AMD"
    assert report.valuation_gap == (
        "best_value_score=1.00; average_value_score=1.00"
    )


def test_value_screening_rejects_raw_numbers_even_with_clean_gate():
    context = MarketContext(
        as_of=AS_OF,
        tickers=["AMD"],
        fundamentals={"AMD": {"pe": 10.0}},
        metadata={
            "fundamental_input_readiness_gate": _clean_gate_metadata()
        },
    )

    report = asyncio.run(
        ValueScreeningAgent(
            name="value_screening",
            config={},
        ).run(context)
    )

    assert report.verdict == "needs_more_data"
    assert report.position_bias == "insufficient_data"
    assert report.valuation_gap is None
    assert "point-in-time semantic contract" in report.thesis


def test_value_screening_rejects_gate_for_different_metric_payload():
    context = MarketContext(
        as_of=AS_OF,
        tickers=["AMD"],
        fundamentals=_fundamentals(),
        metadata={
            "fundamental_input_readiness_gate": (
                _clean_gate_metadata(fingerprint="different-input")
            )
        },
    )

    report = asyncio.run(
        ValueScreeningAgent(
            name="value_screening",
            config={},
        ).run(context)
    )

    assert report.verdict == "needs_more_data"
    assert report.valuation_gap is None
    assert any(
        "fingerprint_mismatch" in reason
        for reason in report.reasons
    )


def test_boundary_application_is_fail_closed_and_records_audit():
    context = MarketContext(
        as_of=AS_OF,
        fundamentals=_fundamentals(),
        macro={"cpi": {"value": 2.8}},
    )

    audit = apply_market_context_structured_boundary(context)

    assert audit["accepted_count"] == 3
    assert context.fundamentals["AMD"]["pe"] == 10.0
    assert context.macro == {}
    assert context.metadata[
        "structured_context_point_in_time_audit"
    ]["excluded_count"] == 1
