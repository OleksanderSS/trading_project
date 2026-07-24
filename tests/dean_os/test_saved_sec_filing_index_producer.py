from __future__ import annotations

import hashlib
import json
from pathlib import Path

import duckdb
import pytest

from dean_os.analysts._producers.sec.filing_index import (
    SavedSECFilingIndexProducer,
    verify_saved_sec_filing_index,
)


AS_OF = "2026-07-01T00:00:00+03:00"


def _create_database(path: Path, rows: list[tuple]) -> Path:
    connection = duckdb.connect(str(path))
    connection.execute(
        """
        CREATE TABLE sec_filings (
            accessionNumber VARCHAR,
            filingDate VARCHAR,
            reportDate VARCHAR,
            acceptanceDateTime VARCHAR,
            form VARCHAR,
            isXBRL BIGINT,
            isInlineXBRL BIGINT,
            primaryDocument VARCHAR,
            ticker VARCHAR,
            cik VARCHAR,
            hash VARCHAR
        )
        """
    )
    connection.executemany(
        "INSERT INTO sec_filings VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        rows,
    )
    connection.close()
    return path


def _filing_row(
    *,
    accession: str = "0000002488-26-000076",
    accepted_at: str = "2026-05-05T22:06:27.000Z",
    stored_hash: str | None = None,
) -> tuple:
    cik = "0000002488"
    expected_hash = hashlib.sha256(
        f"{accession}{cik}".encode()
    ).hexdigest()
    return (
        accession,
        "2026-05-06",
        "2026-03-28",
        accepted_at,
        "10-Q",
        1,
        1,
        "amd-20260328.htm",
        "AMD",
        cik,
        stored_hash or expected_hash,
    )


def test_saved_sec_index_builds_verified_metadata_only_record(tmp_path):
    database = _create_database(
        tmp_path / "filings.duckdb",
        [_filing_row()],
    )

    payload = SavedSECFilingIndexProducer(
        tmp_path / "reports"
    ).build(
        tickers=["amd"],
        as_of=AS_OF,
        database_path=database,
        save=True,
    )

    assert payload["status"] == "filing_index_ready"
    assert payload["summary"]["accepted_filing_count"] == 1
    assert payload["summary"]["xbrl_filing_count"] == 1
    assert payload["summary"]["ticker_coverage_status"] == "complete"
    assert payload["summary"]["missing_tickers"] == []
    assert payload["summary"]["can_create_fundamental_metrics"] is False
    assert payload["summary"]["can_feed_value_screening"] is False
    filing = payload["filings"][0]
    assert filing["content_status"] == (
        "metadata_only_primary_document_not_stored"
    )
    assert filing["source_locator"] == (
        "https://www.sec.gov/Archives/edgar/data/2488/"
        "000000248826000076/amd-20260328.htm"
    )
    verified = verify_saved_sec_filing_index(
        payload["saved_paths"]["latest_json"],
        expected_as_of=AS_OF,
    )
    assert verified["verified"] is True
    assert verified["fundamental_metrics_available"] is False


def test_saved_sec_index_excludes_future_acceptance(tmp_path):
    database = _create_database(
        tmp_path / "filings.duckdb",
        [_filing_row(accepted_at="2026-07-02T12:00:00Z")],
    )

    payload = SavedSECFilingIndexProducer().build(
        tickers=["AMD"],
        as_of=AS_OF,
        database_path=database,
        save=False,
    )

    assert payload["status"] == "blocked_no_admissible_filings"
    assert payload["summary"]["reason_counts"][
        "sec_filing_accepted_after_as_of"
    ] == 1


def test_saved_sec_index_reports_partial_cohort_coverage(tmp_path):
    database = _create_database(
        tmp_path / "filings.duckdb",
        [_filing_row()],
    )

    payload = SavedSECFilingIndexProducer().build(
        tickers=["AMD", "NVDA"],
        as_of=AS_OF,
        database_path=database,
        save=False,
    )

    assert payload["summary"]["ticker_coverage_status"] == "partial"
    assert payload["summary"]["ticker_coverage_ratio"] == 0.5
    assert payload["summary"]["represented_tickers"] == ["AMD"]
    assert payload["summary"]["missing_tickers"] == ["NVDA"]


def test_saved_sec_index_rejects_collector_hash_mismatch(tmp_path):
    database = _create_database(
        tmp_path / "filings.duckdb",
        [_filing_row(stored_hash="wrong")],
    )

    payload = SavedSECFilingIndexProducer().build(
        tickers=["AMD"],
        as_of=AS_OF,
        database_path=database,
        save=False,
    )

    assert payload["summary"]["accepted_filing_count"] == 0
    assert payload["summary"]["reason_counts"][
        "sec_collector_hash_mismatch"
    ] == 1


def test_saved_sec_index_verifier_rejects_artifact_tampering(
    tmp_path,
):
    database = _create_database(
        tmp_path / "filings.duckdb",
        [_filing_row()],
    )
    payload = SavedSECFilingIndexProducer(
        tmp_path / "reports"
    ).build(
        tickers=["AMD"],
        as_of=AS_OF,
        database_path=database,
        save=True,
    )
    artifact = json.loads(
        Path(payload["saved_paths"]["latest_json"]).read_text(
            encoding="utf-8"
        )
    )
    artifact["filings"][0]["form"] = "10-K"
    tampered = tmp_path / "tampered.json"
    tampered.write_text(json.dumps(artifact), encoding="utf-8")

    with pytest.raises(ValueError, match="artifact payload mismatch"):
        verify_saved_sec_filing_index(
            tampered,
            expected_as_of=AS_OF,
        )


def test_saved_sec_index_supports_hash_bound_offline_reverification(
    tmp_path,
):
    database = _create_database(
        tmp_path / "filings.duckdb",
        [_filing_row()],
    )
    payload = SavedSECFilingIndexProducer(
        tmp_path / "reports"
    ).build(
        tickers=["AMD"],
        as_of=AS_OF,
        database_path=database,
        save=True,
    )
    database.unlink()

    verified = verify_saved_sec_filing_index(
        payload["saved_paths"]["latest_json"],
        expected_as_of=AS_OF,
        verify_source_database=False,
    )

    assert verified["verified"] is True
    assert verified["verification_mode"] == "hash_bound_artifact_only"
    with pytest.raises(ValueError, match="source database is missing"):
        verify_saved_sec_filing_index(
            payload["saved_paths"]["latest_json"],
            expected_as_of=AS_OF,
        )


def test_current_database_has_verified_amd_periodic_filing():
    database = Path("data/trading_data.duckdb")
    if not database.exists():
        pytest.skip("current trading DuckDB is absent")

    payload = SavedSECFilingIndexProducer().build(
        tickers=["AMD"],
        forms=["10-Q", "10-K"],
        as_of=AS_OF,
        database_path=database,
        save=False,
    )

    assert payload["status"] == "filing_index_ready"
    # The real database keeps growing as new data is ingested -- this was
    # exactly 10191 when the test was written; assert a floor instead of an
    # exact snapshot so real, expected growth doesn't fail the test.
    assert payload["summary"]["database_row_count"] >= 10191
    assert payload["summary"]["accepted_filing_count"] == 1
    assert payload["filings"][0]["ticker"] == "AMD"
    assert payload["filings"][0]["form"] == "10-Q"
    assert payload["summary"]["can_request_filing_content"] is True
    assert payload["summary"]["can_create_fundamental_metrics"] is False
