import json

from dean_os.filing_order_evidence import FilingOrderEvidenceBuilder


def _write(path, payload):
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_extracts_latest_point_in_time_rpo_as_partial_proxy(tmp_path):
    source = _write(tmp_path / "facts.json", {
        "entityName": "Issuer",
        "facts": {"us-gaap": {"RevenueRemainingPerformanceObligation": {
            "units": {"USD": [
                {"end": "2026-01-01", "val": 100, "accn": "old", "form": "10-K", "filed": "2026-02-01"},
                {"end": "2026-04-01", "val": 120, "accn": "new", "form": "10-Q", "filed": "2026-05-01"},
                {"end": "2026-07-01", "val": 999, "accn": "future", "form": "10-Q", "filed": "2026-08-01"},
            ]}
        }}},
    })
    payload = FilingOrderEvidenceBuilder(tmp_path / "out").build(
        {"NVDA": source}, as_of="2026-06-01T00:00:00+00:00", save=False
    )
    row = payload["observations"][0]
    assert row["value"] == 120.0
    assert row["semantic_role"] == "contracted_revenue_proxy_not_full_order_backlog"
    assert payload["summary"]["full_backlog_observation_count"] == 0
    assert row["automatic_gap_closure_allowed"] is False
    assert row["current_gap_support_eligible"] is True


def test_purchase_obligation_does_not_become_backlog(tmp_path):
    source = _write(tmp_path / "facts.json", {
        "entityName": "Issuer", "facts": {"us-gaap": {
            "PurchaseObligation": {"units": {"USD": [{"val": 500, "filed": "2026-01-01"}]}}
        }}
    })
    payload = FilingOrderEvidenceBuilder().build(
        {"AMD": source}, as_of="2026-06-01T00:00:00+00:00", save=False
    )
    assert payload["observations"] == []
    assert payload["exclusions"][0]["reason"] == "rpo_concept_absent"


def test_old_rpo_is_historical_context_not_current_gap_support(tmp_path):
    source = _write(tmp_path / "facts.json", {
        "entityName": "Issuer", "facts": {"us-gaap": {
            "RevenueRemainingPerformanceObligation": {"units": {"USD": [
                {"end": "2019-12-31", "val": 100, "accn": "old", "form": "10-K", "filed": "2020-01-31"}
            ]}}
        }}
    })
    payload = FilingOrderEvidenceBuilder().build(
        {"INTC": source}, as_of="2026-07-12T00:00:00+00:00", max_age_days=730, save=False
    )
    assert payload["observations"][0]["current_gap_support_eligible"] is False
    assert payload["summary"]["historical_context_only_count"] == 1
