from dean_os.industry_operational_metrics import IndustryOperationalMetricsBuilder


AS_OF = "2026-07-01T12:00:00+00:00"
HASH = "a" * 64


def _record(**changes):
    record = {
        "record_id": "util_1",
        "entity": "Example Foundry",
        "metric_name": "capacity_utilization",
        "value": 82.5,
        "unit": "percent",
        "period": "2026-Q2",
        "available_at": "2026-06-30T08:00:00+00:00",
        "source_locator": "file:///industry/q2.json#utilization",
        "source_sha256": HASH,
        "value_kind": "actual",
    }
    record.update(changes)
    return record


def test_accepts_point_in_time_metric_and_keeps_review_boundary():
    payload = IndustryOperationalMetricsBuilder().build(
        [_record()], as_of=AS_OF, domain_id="semiconductor", save=False
    )

    assert payload["summary"]["accepted_count"] == 1
    assert payload["accepted_records"][0]["value"] == 82.5
    assert payload["summary"]["can_close_gap_automatically"] is False
    assert payload["integration_boundary"]["stage5_feature_write_allowed"] is False


def test_quarantines_future_missing_unit_and_prose():
    payload = IndustryOperationalMetricsBuilder().build(
        [
            _record(record_id="future", available_at="2026-07-02T00:00:00+00:00"),
            _record(record_id="no_unit", unit=""),
            "utilization is improving",
        ],
        as_of=AS_OF,
        domain_id="semiconductor",
        save=False,
    )

    assert payload["summary"]["accepted_count"] == 0
    assert payload["summary"]["quarantined_count"] == 3
    assert payload["summary"]["reason_counts"]["available_at_after_as_of"] == 1
    assert payload["summary"]["reason_counts"]["unit_missing"] == 1
    assert payload["summary"]["reason_counts"]["record_not_structured"] == 1


def test_guidance_is_not_counted_as_actual():
    payload = IndustryOperationalMetricsBuilder().build(
        [_record(value_kind="guidance")], as_of=AS_OF, domain_id="semiconductor", save=False
    )

    assert payload["summary"]["actual_count"] == 0
    assert payload["summary"]["non_actual_count"] == 1
    assert payload["accepted_records"][0]["value_kind"] == "guidance"


def test_revision_preserves_and_supersedes_original():
    revised = _record(
        record_id="util_2",
        value=84.0,
        revision_status="revised",
        supersedes_record_id="util_1",
    )
    payload = IndustryOperationalMetricsBuilder().build(
        [_record(), revised], as_of=AS_OF, domain_id="semiconductor", save=False
    )

    assert payload["summary"]["accepted_count"] == 2
    assert payload["summary"]["superseded_count"] == 1
    original = next(row for row in payload["accepted_records"] if row["record_id"] == "util_1")
    assert original["lifecycle_status"] == "superseded"
    assert original["superseded_by_record_id"] == "util_2"


def test_rejects_implicit_fraction_for_percent_metric():
    payload = IndustryOperationalMetricsBuilder().build(
        [_record(value=0.825, unit="ratio")], as_of=AS_OF, domain_id="semiconductor", save=False
    )

    assert payload["summary"]["accepted_count"] == 0
    assert payload["summary"]["reason_counts"]["percent_metric_requires_explicit_percent_unit"] == 1
