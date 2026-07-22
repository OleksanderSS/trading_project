from __future__ import annotations

from pathlib import Path

import pytest

from dean_os.analysts._producers.sec.merger import (
    SavedSECFundamentalEvidenceMerger,
    _merge_facts,
    load_verified_merged_fundamental_context_fragment,
)


def test_merger_blocks_conflicting_same_ticker_metric():
    base = {
        "ticker": "AMD",
        "metric_name": "revenue",
        "unit": "USD",
        "period": "2026-01-01/2026-03-31",
        "accepted_at": "2026-05-01T00:00:00+00:00",
        "accession_number": "a",
        "fact_sha256": "one",
    }
    result = _merge_facts(
        [
            {**base, "value": 1.0},
            {**base, "value": 2.0, "fact_sha256": "two"},
        ]
    )

    assert result["facts"] == []
    assert result["conflicting_fact_count"] == 2
    assert result["reason_counts"][
        "conflicting_verified_fundamental_facts"
    ] == 1


def test_current_real_fundamental_sources_merge_with_explicit_gaps():
    company = Path(
        "reports/dean_os/saved_sec_companyfacts_producer_current/latest.json"
    )
    inline = Path(
        "reports/dean_os/saved_sec_inline_xbrl_producer_current/latest.json"
    )
    nvda = Path(
        "reports/dean_os/"
        "nvda_saved_sec_companyfacts_producer_current/latest.json"
    )
    if not company.exists() or not inline.exists() or not nvda.exists():
        pytest.skip("current real SEC artifacts are absent")

    payload = SavedSECFundamentalEvidenceMerger().build(
        companyfacts_artifact_path=company,
        additional_companyfacts_artifact_paths=[nvda],
        inline_xbrl_artifact_paths=[inline],
        save=False,
    )

    assert payload["summary"]["accepted_fact_count"] >= 33
    assert payload["summary"]["accepted_fact_tickers"] == [
        "AMD",
        "INTC",
        "NVDA",
        "TSM",
    ]
    assert payload["summary"]["missing_tickers"] == []
    inventory_tickers = sorted(
        item["ticker"]
        for item in payload["facts"]
        if item["metric_name"] == "inventory"
    )
    assert inventory_tickers == ["AMD", "INTC", "NVDA", "TSM"]
    assert payload["summary"]["ticker_coverage_ratio"] == 1.0
    assert payload["summary"][
        "cross_ticker_comparability_status"
    ] == "partial_or_period_unit_mismatch"
    assert payload["summary"][
        "can_claim_complete_sector_fundamentals"
    ] is False


def test_current_merged_fragment_reverifies_all_sources():
    artifact = Path(
        "reports/dean_os/"
        "saved_sec_fundamental_evidence_merger_current/latest.json"
    )
    if not artifact.exists():
        pytest.skip("current merged SEC artifact is absent")

    fragment = load_verified_merged_fundamental_context_fragment(
        artifact,
        expected_as_of="2026-06-30T21:00:00+00:00",
    )

    assert sorted(fragment["fundamentals"]) == [
        "AMD",
        "INTC",
        "NVDA",
        "TSM",
    ]
    assert fragment["metadata"][
        "saved_sec_fundamental_merger_verified"
    ] is True
    assert fragment["metadata"]["missing_tickers"] == []
