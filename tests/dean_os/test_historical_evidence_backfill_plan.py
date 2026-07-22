from __future__ import annotations

import json

import pandas as pd

from dean_os.historical_evidence_backfill_plan import HistoricalEvidenceBackfillPlan


def _write_json(path, payload):
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_historical_evidence_backfill_plan_creates_tasks_for_weak_runs(tmp_path):
    readiness = _write_json(
        tmp_path / "readiness.json",
        {
            "mode": "replay_calibration_readiness_gate",
            "summary": {"readiness_status": "need_evidence_backfill", "next_action": "backfill_research_evidence"},
            "gate": {"passed_checks": ["price_quality", "replay_sample", "research_sample"]},
        },
    )
    research = _write_json(
        tmp_path / "research.json",
        {
            "mode": "historical_research_replay_batch",
            "inputs": {"tickers": ["AMD", "NVDA"], "price_data_path": "prices.parquet"},
            "runs": [
                {
                    "as_of": "2026-01-15T00:00:00+00:00",
                    "evidence_document_count": 1,
                    "evidence_data_quality": "weak",
                    "evidence_missing_tickers": ["NVDA"],
                    "evidence_tickers": ["AMD"],
                    "research_stance": "insufficient_data",
                }
            ],
        },
    )

    payload = HistoricalEvidenceBackfillPlan(tmp_path / "reports").build(
        readiness_report_path=readiness,
        research_batch_path=research,
        tickers=["AMD", "NVDA"],
        save=False,
    )

    assert payload["summary"]["backfill_status"] == "backfill_required"
    assert payload["summary"]["weak_run_count"] == 1
    assert payload["coverage_gaps"]["missing_tickers"] == ["NVDA"]
    assert any(task["task_id"] == "backfill_historical_news_evidence" for task in payload["backfill_tasks"])
    assert payload["safety"]["collector_run_performed"] is False


def test_historical_evidence_backfill_plan_audits_source_rows_and_ticker_hits(tmp_path):
    readiness = _write_json(tmp_path / "readiness.json", {"summary": {"readiness_status": "need_evidence_backfill"}})
    research = _write_json(
        tmp_path / "research.json",
        {
            "inputs": {"tickers": ["AMD", "NVDA"], "price_data_path": "prices.parquet"},
            "runs": [
                {
                    "as_of": "2026-01-15T00:00:00+00:00",
                    "evidence_document_count": 0,
                    "evidence_data_quality": "weak",
                    "evidence_missing_tickers": ["AMD", "NVDA"],
                }
            ],
        },
    )
    news = tmp_path / "news.csv"
    pd.DataFrame(
        [
            {"published_date": "2025-12-20T00:00:00Z", "title": "AMD data center update", "content": "AMD momentum"},
            {"published_date": "2025-12-21T00:00:00Z", "title": "Market note", "content": "No ticker"},
        ]
    ).to_csv(news, index=False)

    payload = HistoricalEvidenceBackfillPlan(tmp_path / "reports").build(
        readiness_report_path=readiness,
        research_batch_path=research,
        news_data_paths=[news],
        tickers=["AMD", "NVDA"],
        save=False,
    )

    audit = payload["source_audits"]["news"][0]
    assert audit["status"] == "inspected"
    assert audit["windows_with_rows"] == 1
    assert audit["ticker_hits"]["AMD"] >= 1
    assert audit["ticker_hits"]["NVDA"] == 0


def test_historical_evidence_backfill_plan_reports_ready_when_no_weak_runs(tmp_path):
    research = _write_json(
        tmp_path / "research.json",
        {
            "inputs": {"tickers": ["AMD"]},
            "runs": [
                {
                    "as_of": "2026-01-15T00:00:00+00:00",
                    "evidence_document_count": 12,
                    "evidence_data_quality": "strong",
                    "evidence_missing_tickers": [],
                    "evidence_tickers": ["AMD"],
                }
            ],
        },
    )

    payload = HistoricalEvidenceBackfillPlan(tmp_path / "reports").build(
        readiness_report_path=None,
        research_batch_path=research,
        save=False,
    )

    assert payload["summary"]["backfill_status"] == "evidence_ready"
    assert payload["backfill_tasks"] == []
    assert payload["recommendations"][0].startswith("Evidence coverage is ready")
