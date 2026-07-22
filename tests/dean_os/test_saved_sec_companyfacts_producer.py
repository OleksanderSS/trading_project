from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path

import duckdb
import pytest

from dean_os.agent_lab import AgentLabRunner
from dean_os.fundamental_input_readiness_gate import (
    FundamentalInputReadinessGate,
)
from dean_os.analysts._producers.sec.companyfacts import (
    SavedSECCompanyFactsProducer,
    fetch_companyfacts_snapshots,
    fetch_primary_filing_snapshots,
    fetch_sec_submissions_snapshots,
    load_verified_fundamental_context_fragment,
)
from dean_os.analysts._producers.sec.filing_index import (
    SavedSECFilingIndexProducer,
)


AS_OF = "2026-07-01T00:00:00+00:00"
CIK = "0000002488"
ACCESSION = "0000002488-26-000076"


def _filing_index(tmp_path: Path, tickers=None) -> dict:
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
            "2026-04-24",
            "2026-03-28",
            "2026-04-23T22:49:27Z",
            "10-Q",
            1,
            1,
            "amd-20260328.htm",
            "AMD",
            CIK,
            row_hash,
        ],
    )
    connection.close()
    return SavedSECFilingIndexProducer(
        tmp_path / "filing_index"
    ).build(
        tickers=tickers or ["AMD"],
        as_of=AS_OF,
        database_path=database,
        save=True,
    )


def _companyfacts(
    *,
    revenue_value: float = 7_400_000_000,
    revenue_entries: list[dict] | None = None,
) -> dict:
    revenue = revenue_entries or [
        {
            "start": "2025-12-28",
            "end": "2026-03-28",
            "val": revenue_value,
            "accn": ACCESSION,
            "fy": 2026,
            "fp": "Q1",
            "form": "10-Q",
            "filed": "2026-04-24",
            "frame": "CY2026Q1",
        },
        {
            "start": "2024-12-29",
            "end": "2025-03-29",
            "val": 5_800_000_000,
            "accn": "0000002488-25-000050",
            "fy": 2025,
            "fp": "Q1",
            "form": "10-Q",
            "filed": "2025-05-02",
            "frame": "CY2025Q1",
        },
    ]
    return {
        "cik": int(CIK),
        "entityName": "Advanced Micro Devices, Inc.",
        "facts": {
            "us-gaap": {
                "RevenueFromContractWithCustomerExcludingAssessedTax": {
                    "label": "Revenue",
                    "units": {"USD": revenue},
                },
                "Assets": {
                    "label": "Assets",
                    "units": {
                        "USD": [
                            {
                                "end": "2026-03-28",
                                "val": 72_000_000_000,
                                "accn": ACCESSION,
                                "fy": 2026,
                                "fp": "Q1",
                                "form": "10-Q",
                                "filed": "2026-04-24",
                                "frame": "CY2026Q1I",
                            }
                        ]
                    },
                },
                "InventoryNet": {
                    "label": "Inventory, net",
                    "units": {
                        "USD": [
                            {
                                "end": "2026-03-28",
                                "val": 7_100_000_000,
                                "accn": ACCESSION,
                                "fy": 2026,
                                "fp": "Q1",
                                "form": "10-Q",
                                "filed": "2026-04-24",
                                "frame": "CY2026Q1I",
                            }
                        ]
                    },
                },
            }
        },
    }


def _write_companyfacts(
    tmp_path: Path,
    payload: dict | None = None,
) -> Path:
    source = tmp_path / "raw" / f"CIK{CIK}" / "latest.json"
    source.parent.mkdir(parents=True)
    source.write_text(
        json.dumps(payload or _companyfacts()),
        encoding="utf-8",
    )
    return source


def test_companyfacts_producer_binds_facts_to_verified_accession(
    tmp_path,
):
    filing_index = _filing_index(
        tmp_path,
        tickers=["AMD", "NVDA"],
    )
    _write_companyfacts(tmp_path)

    payload = SavedSECCompanyFactsProducer().build(
        filing_index_path=filing_index["saved_paths"][
            "latest_json"
        ],
        source_dir=tmp_path / "raw",
        as_of=AS_OF,
        save=False,
    )

    assert payload["status"] == "fundamental_facts_ready_with_gaps"
    assert payload["summary"]["accepted_fact_tickers"] == ["AMD"]
    assert payload["summary"]["missing_filing_tickers"] == ["NVDA"]
    assert payload["summary"][
        "can_claim_complete_sector_fundamentals"
    ] is False
    facts = {
        item["metric_name"]: item for item in payload["facts"]
    }
    assert facts["revenue"]["value"] == 7_400_000_000
    assert facts["revenue"]["period"] == (
        "2025-12-28/2026-03-28"
    )
    assert facts["revenue"]["available_at"] == (
        "2026-04-23T22:49:27+00:00"
    )
    assert facts["assets"]["period"] == "2026-03-28"
    assert facts["inventory"]["value"] == 7_100_000_000
    assert facts["inventory"]["period"] == "2026-03-28"
    assert payload["summary"]["accepted_fact_count"] == 3


def test_companyfacts_producer_rejects_ytd_duration_for_quarter(
    tmp_path,
):
    filing_index = _filing_index(tmp_path)
    _write_companyfacts(
        tmp_path,
        _companyfacts(
            revenue_entries=[
                {
                    "start": "2025-06-29",
                    "end": "2026-03-28",
                    "val": 20_000_000_000,
                    "accn": ACCESSION,
                    "fy": 2026,
                    "fp": "Q1",
                    "form": "10-Q",
                    "filed": "2026-04-24",
                }
            ]
        ),
    )

    payload = SavedSECCompanyFactsProducer().build(
        filing_index_path=filing_index["saved_paths"][
            "latest_json"
        ],
        source_dir=tmp_path / "raw",
        save=False,
    )

    assert "revenue" not in {
        item["metric_name"] for item in payload["facts"]
    }
    assert payload["summary"]["reason_counts"][
        "sec_companyfacts_duration_not_filing_period"
    ] == 1


def test_single_ticker_source_artifact_never_claims_sector_complete(
    tmp_path,
):
    filing_index = _filing_index(tmp_path)
    _write_companyfacts(tmp_path)

    payload = SavedSECCompanyFactsProducer().build(
        filing_index_path=filing_index["saved_paths"][
            "latest_json"
        ],
        source_dir=tmp_path / "raw",
        save=False,
    )

    assert payload["summary"]["requested_scope_complete"] is True
    assert payload["summary"][
        "can_claim_complete_sector_fundamentals"
    ] is False


def test_companyfacts_producer_rejects_ambiguous_same_accession(
    tmp_path,
):
    filing_index = _filing_index(tmp_path)
    base = {
        "start": "2025-12-28",
        "end": "2026-03-28",
        "accn": ACCESSION,
        "fy": 2026,
        "fp": "Q1",
        "form": "10-Q",
        "filed": "2026-04-24",
    }
    _write_companyfacts(
        tmp_path,
        _companyfacts(
            revenue_entries=[
                {**base, "val": 7_400_000_000},
                {**base, "val": 7_500_000_000},
            ]
        ),
    )

    payload = SavedSECCompanyFactsProducer().build(
        filing_index_path=filing_index["saved_paths"][
            "latest_json"
        ],
        source_dir=tmp_path / "raw",
        save=False,
    )

    assert payload["summary"]["reason_counts"][
        "sec_companyfacts_metric_ambiguous"
    ] == 1
    assert "revenue" not in {
        item["metric_name"] for item in payload["facts"]
    }


def test_verified_companyfacts_fragment_rechecks_source_hash(
    tmp_path,
):
    filing_index = _filing_index(tmp_path)
    source = _write_companyfacts(tmp_path)
    payload = SavedSECCompanyFactsProducer(
        tmp_path / "producer"
    ).build(
        filing_index_path=filing_index["saved_paths"][
            "latest_json"
        ],
        source_dir=tmp_path / "raw",
        save=True,
    )

    fragment = load_verified_fundamental_context_fragment(
        payload["saved_paths"]["latest_json"],
        expected_as_of=AS_OF,
    )

    assert fragment["metadata"][
        "saved_sec_companyfacts_verified"
    ] is True
    assert fragment["fundamentals"]["AMD"]["revenue"] == (
        7_400_000_000
    )
    source.write_text(
        source.read_text(encoding="utf-8") + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="source artifact hash"):
        load_verified_fundamental_context_fragment(
            payload["saved_paths"]["latest_json"],
            expected_as_of=AS_OF,
        )


def test_verified_companyfacts_fragment_does_not_reopen_live_database(
    tmp_path,
):
    filing_index = _filing_index(tmp_path)
    _write_companyfacts(tmp_path)
    payload = SavedSECCompanyFactsProducer(
        tmp_path / "producer"
    ).build(
        filing_index_path=filing_index["saved_paths"][
            "latest_json"
        ],
        source_dir=tmp_path / "raw",
        save=True,
    )
    Path(
        filing_index["inputs"]["database_path"]
    ).unlink()

    fragment = load_verified_fundamental_context_fragment(
        payload["saved_paths"]["latest_json"],
        expected_as_of=AS_OF,
    )

    assert fragment["metadata"][
        "filing_index_verification_mode"
    ] == "hash_bound_artifact_only"
    assert fragment["fundamentals"]["AMD"]["revenue"] == (
        7_400_000_000
    )


def test_verified_companyfacts_enters_agent_lab_without_fake_value_score(
    tmp_path,
):
    filing_index = _filing_index(tmp_path)
    _write_companyfacts(tmp_path)
    producer = SavedSECCompanyFactsProducer(
        tmp_path / "producer"
    ).build(
        filing_index_path=filing_index["saved_paths"][
            "latest_json"
        ],
        source_dir=tmp_path / "raw",
        save=True,
    )
    gate = FundamentalInputReadinessGate(
        tmp_path / "gate"
    ).build(
        fundamentals_json=producer["saved_paths"]["latest_json"],
        as_of=AS_OF,
        save=False,
    )
    fragment = load_verified_fundamental_context_fragment(
        producer["saved_paths"]["latest_json"],
        expected_as_of=AS_OF,
    )

    report = asyncio.run(
        AgentLabRunner(
            corpus_path=tmp_path / "corpus.sqlite",
            learning_path=tmp_path / "learning.sqlite",
            output_dir=tmp_path / "agent_lab",
            memory_path=tmp_path / "memory.sqlite",
            log_path=None,
        ).run(
            documents=[],
            tickers=["AMD"],
            fundamentals=fragment["fundamentals"],
            fundamental_gate=gate,
            fundamental_provenance=fragment["metadata"],
            as_of=AS_OF,
            include_financial_nlp=False,
            include_synthesis=False,
            create_learning_records=False,
            include_operations_proposals=False,
        )
    )

    assert report.summary["fundamental_evidence_provenance"][
        "saved_sec_companyfacts_verified"
    ] is True
    value_report = next(
        item
        for item in report.reports
        if item.agent_name == "value_screening"
    )
    assert value_report.verdict == "needs_more_data"
    assert value_report.position_bias == "insufficient_data"
    assert "required value-screening ratios" in value_report.thesis


class _FakeResponse:
    def __init__(self, payload: bytes):
        self.payload = payload
        self.headers = {}

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self):
        return self.payload


def test_companyfacts_snapshot_saves_immutable_and_latest(
    tmp_path,
):
    filing_index = _filing_index(tmp_path)
    raw = json.dumps(_companyfacts()).encode("utf-8")
    requests = []

    def opener(request, timeout):
        requests.append((request, timeout))
        return _FakeResponse(raw)

    payload = fetch_companyfacts_snapshots(
        filing_index_path=filing_index["saved_paths"][
            "latest_json"
        ],
        output_dir=tmp_path / "snapshots",
        user_agent="DEAN-OS local research contact",
        request_delay_seconds=0,
        opener=opener,
    )

    assert payload["status"] == "companyfacts_snapshots_ready"
    assert payload["summary"]["saved_snapshot_count"] == 1
    snapshot = payload["snapshots"][0]
    assert Path(snapshot["immutable_path"]).read_bytes() == raw
    assert Path(snapshot["latest_path"]).read_bytes() == raw
    assert requests[0][0].get_header("User-agent") == (
        "DEAN-OS local research contact"
    )
    assert payload["safety"]["network_access_performed"] is True
    assert payload["safety"]["can_trade"] is False


def test_primary_filing_snapshot_is_bound_to_verified_index(
    tmp_path,
):
    filing_index = _filing_index(tmp_path)
    raw = b"<html><body><ix:nonfraction>1</ix:nonfraction></body></html>"

    def opener(request, timeout):
        assert ACCESSION.replace("-", "") in request.full_url
        assert timeout == 30.0
        return _FakeResponse(raw)

    payload = fetch_primary_filing_snapshots(
        filing_index_path=filing_index["saved_paths"][
            "latest_json"
        ],
        output_dir=tmp_path / "filings",
        user_agent="DEAN-OS local research contact",
        request_delay_seconds=0,
        opener=opener,
    )

    assert payload["status"] == "primary_document_snapshots_ready"
    snapshot = payload["snapshots"][0]
    assert snapshot["accession_number"] == ACCESSION
    assert Path(snapshot["immutable_path"]).read_bytes() == raw
    assert Path(snapshot["latest_path"]).read_bytes() == raw


def test_submissions_snapshot_uses_configured_cik_and_checks_ticker(
    tmp_path,
):
    assets = tmp_path / "assets.yaml"
    assets.write_text(
        "assets:\n  details:\n    NVDA:\n      cik: 1045810\n",
        encoding="utf-8",
    )
    raw = json.dumps(
        {
            "cik": "0001045810",
            "tickers": ["NVDA"],
            "filings": {"recent": {}},
        }
    ).encode("utf-8")

    def opener(request, timeout):
        assert request.full_url.endswith("CIK0001045810.json")
        assert timeout == 30.0
        return _FakeResponse(raw)

    payload = fetch_sec_submissions_snapshots(
        tickers=["NVDA"],
        output_dir=tmp_path / "submissions",
        user_agent="DEAN-OS local research contact",
        assets_config_path=assets,
        request_delay_seconds=0,
        opener=opener,
    )

    assert payload["status"] == "sec_submissions_snapshots_ready"
    snapshot = payload["snapshots"][0]
    assert snapshot["ticker"] == "NVDA"
    assert snapshot["cik"] == "0001045810"
    assert Path(snapshot["immutable_path"]).read_bytes() == raw
