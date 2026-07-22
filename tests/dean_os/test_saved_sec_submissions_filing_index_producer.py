from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from dean_os.analysts._producers.sec.filing_index import (
    verify_sec_filing_index,
)
from dean_os.analysts._producers.sec.submissions_index import (
    SavedSECSubmissionsFilingIndexProducer,
    verify_saved_sec_submissions_filing_index,
)


AS_OF = "2026-06-30T21:00:00+00:00"
CIK = "0001045810"


def _snapshot(tmp_path: Path) -> tuple[Path, Path]:
    recent = {
        "accessionNumber": [
            "0001045810-26-000052",
            "0001045810-26-000021",
            "0001045810-26-000099",
        ],
        "filingDate": [
            "2026-05-20",
            "2026-02-25",
            "2026-07-02",
        ],
        "reportDate": [
            "2026-04-26",
            "2026-01-25",
            "2026-06-28",
        ],
        "acceptanceDateTime": [
            "2026-05-20T20:35:52.000Z",
            "2026-02-25T21:42:19.000Z",
            "2026-07-02T20:00:00.000Z",
        ],
        "form": ["10-Q", "10-K", "10-Q"],
        "isXBRL": [1, 1, 1],
        "isInlineXBRL": [1, 1, 1],
        "primaryDocument": [
            "nvda-20260426.htm",
            "nvda-20260125.htm",
            "nvda-20260628.htm",
        ],
    }
    raw = json.dumps(
        {
            "cik": CIK,
            "tickers": ["NVDA"],
            "filings": {"recent": recent},
        }
    ).encode("utf-8")
    sha = hashlib.sha256(raw).hexdigest()
    source = tmp_path / "raw" / f"{sha}.json"
    source.parent.mkdir()
    source.write_bytes(raw)
    artifact = tmp_path / "snapshot.json"
    artifact.write_text(
        json.dumps(
            {
                "snapshot_contract": "dean_sec_submissions_snapshot_v1",
                "status": "sec_submissions_snapshots_ready",
                "snapshots": [
                    {
                        "ticker": "NVDA",
                        "cik": CIK,
                        "source_url": (
                            "https://data.sec.gov/submissions/"
                            f"CIK{CIK}.json"
                        ),
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
        ),
        encoding="utf-8",
    )
    return artifact, source


def test_submissions_index_selects_latest_admissible_periodic_filing(
    tmp_path,
):
    snapshot, _ = _snapshot(tmp_path)
    payload = SavedSECSubmissionsFilingIndexProducer(
        tmp_path / "producer"
    ).build(
        submissions_snapshot_path=snapshot,
        tickers=["NVDA"],
        as_of=AS_OF,
        save=True,
    )

    assert payload["status"] == (
        "submissions_filing_index_ready_with_exclusions"
    )
    assert payload["summary"]["periodic_candidate_count"] == 2
    assert payload["summary"][
        "not_selected_older_periodic_count"
    ] == 1
    assert payload["summary"]["accepted_filing_count"] == 1
    filing = payload["filings"][0]
    assert filing["form"] == "10-Q"
    assert filing["report_date"] == "2026-04-26"
    assert filing["accession_number"] == "0001045810-26-000052"
    assert payload["summary"]["reason_counts"][
        "sec_filing_accepted_after_as_of"
    ] == 1
    verified = verify_sec_filing_index(
        payload["saved_paths"]["latest_json"],
        expected_as_of=AS_OF,
    )
    assert verified["filings"][0]["record_sha256"] == (
        filing["record_sha256"]
    )


def test_submissions_index_loader_rechecks_raw_hash(tmp_path):
    snapshot, source = _snapshot(tmp_path)
    payload = SavedSECSubmissionsFilingIndexProducer(
        tmp_path / "producer"
    ).build(
        submissions_snapshot_path=snapshot,
        tickers=["NVDA"],
        as_of=AS_OF,
        save=True,
    )
    source.write_bytes(source.read_bytes() + b"\n")

    with pytest.raises(ValueError, match="source hash"):
        verify_saved_sec_submissions_filing_index(
            payload["saved_paths"]["latest_json"],
            expected_as_of=AS_OF,
        )


def test_current_nvda_submissions_selects_expected_10q():
    snapshot = Path(
        "reports/dean_os/sec_submissions_snapshot_current/latest.json"
    )
    if not snapshot.exists():
        pytest.skip("current NVDA submissions snapshot is absent")

    payload = SavedSECSubmissionsFilingIndexProducer().build(
        submissions_snapshot_path=snapshot,
        tickers=["NVDA"],
        as_of=AS_OF,
        save=False,
    )

    assert payload["summary"]["accepted_filing_count"] == 1
    assert payload["filings"][0]["accession_number"] == (
        "0001045810-26-000052"
    )
    assert payload["filings"][0]["report_date"] == "2026-04-26"
