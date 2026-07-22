from __future__ import annotations

import hashlib
import json
from pathlib import Path

import duckdb
import pytest
import yaml

from dean_os.analysts._producers.sec.filing_index import (
    SavedSECFilingIndexProducer,
)
from dean_os.analysts._producers.sec.inline_xbrl import (
    SavedSECInlineXBRLProducer,
    load_verified_inline_xbrl_context_fragment,
)


AS_OF = "2026-07-01T00:00:00+00:00"
CIK = "0001046179"
ACCESSION = "0001628280-26-025362"
SOURCE_URL = (
    "https://www.sec.gov/Archives/edgar/data/1046179/"
    "000162828026025362/tsm-20251231.htm"
)


def _filing_index(tmp_path: Path) -> dict:
    database = tmp_path / "source.duckdb"
    connection = duckdb.connect(str(database))
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
    row_hash = hashlib.sha256(
        f"{ACCESSION}{CIK}".encode("utf-8")
    ).hexdigest()
    connection.execute(
        "INSERT INTO sec_filings VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [
            ACCESSION,
            "2026-04-16",
            "2025-12-31",
            "2026-04-16T10:05:49Z",
            "20-F",
            1,
            1,
            "tsm-20251231.htm",
            "TSM",
            CIK,
            row_hash,
        ],
    )
    connection.close()
    return SavedSECFilingIndexProducer(
        tmp_path / "filing_index"
    ).build(
        tickers=["TSM"],
        as_of=AS_OF,
        database_path=database,
        save=True,
    )


def _inline_html() -> bytes:
    return f"""
    <html>
      <body>
        <xbrli:context id="annual">
          <xbrli:entity>
            <xbrli:identifier scheme="http://www.sec.gov/CIK">{CIK}</xbrli:identifier>
          </xbrli:entity>
          <xbrli:period>
            <xbrli:startDate>2025-01-01</xbrli:startDate>
            <xbrli:endDate>2025-12-31</xbrli:endDate>
          </xbrli:period>
        </xbrli:context>
        <xbrli:context id="dimensional">
          <xbrli:entity>
            <xbrli:identifier scheme="http://www.sec.gov/CIK">{CIK}</xbrli:identifier>
            <xbrli:segment>
              <xbrldi:explicitMember dimension="tsm:SegmentAxis">tsm:Other</xbrldi:explicitMember>
            </xbrli:segment>
          </xbrli:entity>
          <xbrli:period>
            <xbrli:startDate>2025-01-01</xbrli:startDate>
            <xbrli:endDate>2025-12-31</xbrli:endDate>
          </xbrli:period>
        </xbrli:context>
        <xbrli:unit id="twd"><xbrli:measure>iso4217:TWD</xbrli:measure></xbrli:unit>
        <xbrli:unit id="usd"><xbrli:measure>iso4217:USD</xbrli:measure></xbrli:unit>
        <ix:nonFraction name="ifrs-full:RevenueFromContractsWithCustomers"
          contextRef="annual" unitRef="twd" scale="6">3,809,054.3</ix:nonFraction>
        <ix:nonFraction name="ifrs-full:RevenueFromContractsWithCustomers"
          contextRef="annual" unitRef="usd" scale="6">121,423.5</ix:nonFraction>
        <ix:nonFraction name="ifrs-full:RevenueFromContractsWithCustomers"
          contextRef="dimensional" unitRef="twd" scale="6">100.0</ix:nonFraction>
      </body>
    </html>
    """.encode("utf-8")


def _snapshot(tmp_path: Path) -> tuple[Path, Path]:
    raw = _inline_html()
    sha = hashlib.sha256(raw).hexdigest()
    source = tmp_path / "raw" / f"{sha}.htm"
    source.parent.mkdir(parents=True)
    source.write_bytes(raw)
    payload = {
        "run_id": "snapshot",
        "snapshot_contract": "dean_sec_primary_document_snapshot_v1",
        "status": "primary_document_snapshots_ready",
        "snapshots": [
            {
                "ticker": "TSM",
                "cik": CIK,
                "form": "20-F",
                "report_date": "2025-12-31",
                "accepted_at": "2026-04-16T10:05:49+00:00",
                "accession_number": ACCESSION,
                "primary_document": "tsm-20251231.htm",
                "source_url": SOURCE_URL,
                "sha256": sha,
                "immutable_path": str(source),
            }
        ],
        "safety": {
            "official_sec_get_requests_only": True,
            "pipeline_run_performed": False,
            "live_execution_performed": False,
            "can_trade": False,
        },
    }
    artifact = tmp_path / "snapshot.json"
    artifact.write_text(json.dumps(payload), encoding="utf-8")
    return artifact, source


def _registry(tmp_path: Path, include_reporting_unit=True) -> Path:
    payload = {
        "registry_version": "test",
        "review_status": "test",
        "metrics": {
            "revenue": {
                "statement_role": "income_statement",
                "period_type": "duration",
                "accepted_units": ["USD", "TWD"],
                "concepts": [
                    {
                        "taxonomy": "ifrs-full",
                        "concept": "RevenueFromContractsWithCustomers",
                    }
                ],
            }
        },
    }
    if include_reporting_unit:
        payload["issuer_reporting_units"] = {CIK: "TWD"}
    path = tmp_path / "registry.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    return path


def test_inline_xbrl_selects_registered_reporting_unit_and_consolidated_context(
    tmp_path,
):
    filing_index = _filing_index(tmp_path)
    snapshot, _ = _snapshot(tmp_path)
    registry = _registry(tmp_path)

    payload = SavedSECInlineXBRLProducer(
        tmp_path / "producer"
    ).build(
        primary_snapshot_path=snapshot,
        filing_index_path=filing_index["saved_paths"][
            "latest_json"
        ],
        registry_path=registry,
        as_of=AS_OF,
        save=True,
    )

    assert payload["status"] == "inline_xbrl_facts_ready"
    assert payload["summary"]["accepted_fact_count"] == 1
    fact = payload["facts"][0]
    assert fact["value"] == 3_809_054_300_000
    assert fact["unit"] == "TWD"
    assert fact["period"] == "2025-01-01/2025-12-31"
    assert fact["reporting_currency_basis"] == (
        "hashed_issuer_reporting_unit_registry"
    )
    fragment = load_verified_inline_xbrl_context_fragment(
        payload["saved_paths"]["latest_json"],
        expected_as_of=AS_OF,
    )
    assert fragment["fundamentals"]["TSM"]["revenue"] == (
        3_809_054_300_000.0
    )
    assert fragment["metadata"][
        "filing_index_verification_mode"
    ] == "hash_bound_artifact_only"


def test_inline_xbrl_blocks_two_currencies_without_reporting_unit(
    tmp_path,
):
    filing_index = _filing_index(tmp_path)
    snapshot, _ = _snapshot(tmp_path)
    registry = _registry(
        tmp_path,
        include_reporting_unit=False,
    )

    payload = SavedSECInlineXBRLProducer().build(
        primary_snapshot_path=snapshot,
        filing_index_path=filing_index["saved_paths"][
            "latest_json"
        ],
        registry_path=registry,
        save=False,
    )

    assert payload["status"] == (
        "blocked_no_admissible_inline_xbrl_facts"
    )
    assert payload["summary"]["reason_counts"][
        "inline_xbrl_metric_ambiguous"
    ] == 1


def test_inline_xbrl_loader_rechecks_primary_document_hash(
    tmp_path,
):
    filing_index = _filing_index(tmp_path)
    snapshot, source = _snapshot(tmp_path)
    registry = _registry(tmp_path)
    payload = SavedSECInlineXBRLProducer(
        tmp_path / "producer"
    ).build(
        primary_snapshot_path=snapshot,
        filing_index_path=filing_index["saved_paths"][
            "latest_json"
        ],
        registry_path=registry,
        save=True,
    )
    source.write_bytes(source.read_bytes() + b"\n")

    with pytest.raises(ValueError, match="primary source hash"):
        load_verified_inline_xbrl_context_fragment(
            payload["saved_paths"]["latest_json"],
            expected_as_of=AS_OF,
        )
