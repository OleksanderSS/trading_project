from __future__ import annotations

import json

import pandas as pd

from dean_os.evidence_timestamp_audit import EvidenceTimestampAudit


def test_evidence_timestamp_audit_allows_ready_sources(tmp_path):
    news_path = tmp_path / "news.csv"
    pd.DataFrame(
        [
            {"ticker": "AAPL", "published_at": "2026-01-02T00:00:00+00:00", "title": "AI demand"},
            {"ticker": "AAPL", "published_at": "2026-01-03T00:00:00+00:00", "title": "Pricing power"},
            {"ticker": "AMD", "published_at": "2026-01-04T00:00:00+00:00", "title": "Supply chain"},
        ]
    ).to_csv(news_path, index=False)

    payload = EvidenceTimestampAudit(tmp_path / "reports").run(
        news_data_paths=[news_path],
        as_of="2026-01-05T00:00:00+00:00",
        start_at="2026-01-01T00:00:00+00:00",
        collapse_min_rows=3,
        save=False,
    )

    assert payload["summary"]["audit_status"] == "timestamp_ready"
    assert payload["summary"]["can_run_historical_research_replay"] is True
    assert payload["source_audits"][0]["primary_timestamp"]["column"] == "published_at"


def test_evidence_timestamp_audit_records_filter_requirement_for_future_raw_rows(tmp_path):
    news_path = tmp_path / "news.csv"
    pd.DataFrame(
        [
            {"ticker": "AAPL", "published_at": "2026-01-02T00:00:00+00:00", "title": "Visible"},
            {"ticker": "AAPL", "published_at": "2026-01-08T00:00:00+00:00", "title": "Future"},
        ]
    ).to_csv(news_path, index=False)

    payload = EvidenceTimestampAudit(tmp_path / "reports").run(
        news_data_paths=[news_path],
        as_of="2026-01-05T00:00:00+00:00",
        save=False,
    )

    assert payload["summary"]["audit_status"] == "timestamp_ready"
    assert payload["source_audits"][0]["status"] == "timestamp_ready"
    assert payload["source_audits"][0]["primary_timestamp"]["after_as_of_count"] == 1
    assert "must be filtered out" in payload["source_audits"][0]["notes"][0]


def test_evidence_timestamp_audit_flags_collapsed_evidence_pack(tmp_path):
    news_path = tmp_path / "news.csv"
    pack_path = tmp_path / "pack.json"
    pd.DataFrame(
        [
            {"ticker": f"T{i}", "published_at": "2026-01-05T00:00:00+00:00", "title": f"Batch row {i}"}
            for i in range(12)
        ]
    ).to_csv(news_path, index=False)
    pack_path.write_text(
        json.dumps(
            {
                "coverage": {
                    "document_count": 12,
                    "date_range": {
                        "start": "2026-01-05T00:00:00+00:00",
                        "end": "2026-01-05T00:00:00+00:00",
                    },
                }
            }
        ),
        encoding="utf-8",
    )

    payload = EvidenceTimestampAudit(tmp_path / "reports").run(
        news_data_paths=[news_path],
        evidence_pack_path=pack_path,
        as_of="2026-01-05T00:00:00+00:00",
        collapse_min_rows=10,
        save=False,
    )

    assert payload["summary"]["audit_status"] == "timestamp_suspicious"
    assert payload["source_audits"][0]["status"] == "timestamp_suspicious"
    assert payload["evidence_pack_audit"]["status"] == "timestamp_suspicious"
    assert payload["summary"]["can_run_historical_research_replay"] is False


def test_evidence_timestamp_audit_blocks_evidence_pack_future_leak(tmp_path):
    news_path = tmp_path / "news.csv"
    pack_path = tmp_path / "pack.json"
    pd.DataFrame(
        [
            {"ticker": "AAPL", "published_at": "2026-01-02T00:00:00+00:00", "title": "Visible"},
            {"ticker": "AAPL", "published_at": "2026-01-08T00:00:00+00:00", "title": "Future raw row"},
        ]
    ).to_csv(news_path, index=False)
    pack_path.write_text(
        json.dumps(
            {
                "coverage": {
                    "document_count": 2,
                    "date_range": {
                        "start": "2026-01-02T00:00:00+00:00",
                        "end": "2026-01-08T00:00:00+00:00",
                    },
                }
            }
        ),
        encoding="utf-8",
    )

    payload = EvidenceTimestampAudit(tmp_path / "reports").run(
        news_data_paths=[news_path],
        evidence_pack_path=pack_path,
        as_of="2026-01-05T00:00:00+00:00",
        save=False,
    )

    assert payload["source_audits"][0]["status"] == "timestamp_ready"
    assert payload["evidence_pack_audit"]["status"] == "timestamp_blocked"
    assert payload["summary"]["audit_status"] == "timestamp_blocked"
