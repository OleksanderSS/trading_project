from dean_os.expectation_evidence import CONTRACT, validate_expectation_evidence


AS_OF = "2026-07-01T12:00:00+00:00"
HASH = "a" * 64


def _payload():
    return {
        "contract": CONTRACT,
        "expectation_type": "analyst_consensus",
        "actual": {
            "value": 1.2, "unit": "USD/share",
            "available_at": "2026-06-30T12:00:00+00:00",
            "source_locator": "filing://issuer/result", "source_sha256": HASH,
        },
        "expected": {
            "value": 1.0, "unit": "USD/share",
            "available_at": "2026-06-29T12:00:00+00:00",
            "source_locator": "consensus://snapshot", "source_sha256": "b" * 64,
        },
        "expectation_std": 0.1,
    }


def test_accepts_hash_bound_point_in_time_pair():
    result = validate_expectation_evidence(_payload(), as_of=AS_OF)
    assert result["quantitative_gap_allowed"] is True
    assert result["accepted"][0]["expectation_type"] == "analyst_consensus"


def test_rejects_consensus_published_after_actual():
    payload = _payload()
    payload["expected"]["available_at"] = "2026-06-30T13:00:00+00:00"
    result = validate_expectation_evidence(payload, as_of=AS_OF)
    assert result["quantitative_gap_allowed"] is False
    assert "expectation_available_after_actual" in result["reasons"]


def test_rejects_unit_mismatch_and_missing_hash():
    payload = _payload()
    payload["expected"]["unit"] = "percent"
    payload["expected"]["source_sha256"] = ""
    result = validate_expectation_evidence(payload, as_of=AS_OF)
    assert result["quantitative_gap_allowed"] is False
    assert "actual_expected_unit_mismatch" in result["reasons"]
    assert "expected_source_sha256_invalid" in result["reasons"]
