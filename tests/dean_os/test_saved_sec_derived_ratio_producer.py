from __future__ import annotations

from pathlib import Path

import pytest

from dean_os.analysts._producers.sec.ratios import (
    SavedSECDerivedRatioProducer,
    _comparison_lanes,
    _derive_ratios,
    load_verified_derived_ratio_context_fragment,
)


AS_OF = "2026-06-30T21:00:00+00:00"


def _fact(
    ticker: str,
    metric: str,
    value: float,
    *,
    period: str = "2025-12-28/2026-03-28",
    unit: str = "USD",
    form: str = "10-Q",
    fiscal_period: str | None = "Q1",
) -> dict:
    period_type = "duration"
    start, end = period.split("/")
    return {
        "ticker": ticker,
        "metric_name": metric,
        "value": value,
        "unit": unit,
        "period": period,
        "period_type": period_type,
        "period_start": start,
        "period_end": end,
        "form": form,
        "fiscal_period": fiscal_period,
        "available_at": "2026-05-20T12:00:00+00:00",
        "accession_number": f"{ticker}-accession",
        "fact_sha256": f"{ticker}-{metric}-{period}",
    }


def test_ratio_derivation_keeps_quarterly_and_annual_lanes_separate():
    facts = []
    for ticker, operating, revenue in (
        ("AMD", 15.0, 100.0),
        ("INTC", -10.0, 100.0),
    ):
        facts.extend(
            [
                _fact(ticker, "operating_income", operating),
                _fact(ticker, "revenue", revenue),
            ]
        )
    annual = "2025-01-01/2025-12-31"
    facts.extend(
        [
            _fact(
                "TSM",
                "operating_income",
                40.0,
                period=annual,
                unit="TWD",
                form="20-F",
                fiscal_period=None,
            ),
            _fact(
                "TSM",
                "revenue",
                100.0,
                period=annual,
                unit="TWD",
                form="20-F",
                fiscal_period=None,
            ),
        ]
    )

    normalized = _derive_ratios(facts)
    operating = [
        item
        for item in normalized["ratios"]
        if item["ratio_name"] == "operating_margin"
    ]
    lanes = _comparison_lanes(operating)

    assert {item["ticker"]: item["value"] for item in operating} == {
        "AMD": 0.15,
        "INTC": -0.1,
        "TSM": 0.4,
    }
    assert lanes[0]["comparison_period_class"] == "annual"
    assert lanes[0]["tickers"] == ["TSM"]
    assert lanes[1]["comparison_period_class"] == "quarterly_Q1"
    assert lanes[1]["tickers"] == ["AMD", "INTC"]


def test_ratio_derivation_blocks_period_mismatch():
    facts = [
        _fact("AMD", "operating_income", 15.0),
        _fact(
            "AMD",
            "revenue",
            100.0,
            period="2025-01-01/2025-12-31",
        ),
    ]

    normalized = _derive_ratios(facts)

    assert not normalized["ratios"]
    assert normalized["reason_counts"]["ratio_source_period_mismatch"] == 1


def test_current_real_merged_facts_build_verified_ratio_lanes(tmp_path):
    source = Path(
        "reports/dean_os/"
        "saved_sec_fundamental_evidence_merger_current/latest.json"
    )
    if not source.exists():
        pytest.skip("current merged fundamental artifact is absent")
    output = tmp_path / "output"
    payload = SavedSECDerivedRatioProducer(
        output_dir=output
    ).build(
        merged_fundamental_artifact_path=source,
        as_of=AS_OF,
    )

    assert payload["status"] == "derived_ratio_evidence_ready_with_gaps"
    assert payload["summary"]["derived_ratio_count"] == 21
    assert payload["summary"]["multi_ticker_comparison_lane_count"] == 5
    assert payload["summary"]["full_cohort_comparison_lane_count"] == 0
    assert (
        payload["summary"]["can_claim_full_cohort_comparability"]
        is False
    )
    fragment = load_verified_derived_ratio_context_fragment(
        output / "latest.json",
        expected_as_of=AS_OF,
    )
    assert fragment["metadata"]["saved_sec_derived_ratio_verified"]
    assert set(fragment["fundamentals"]) == {
        "AMD",
        "INTC",
        "NVDA",
        "TSM",
    }
