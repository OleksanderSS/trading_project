from __future__ import annotations

import json

from dean_os.research_replay_directionality_diagnostic import ResearchReplayDirectionalityDiagnostic


def _write_json(path, payload):
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _batch(path, runs):
    return _write_json(
        path,
        {
            "mode": "historical_research_replay_batch",
            "inputs": {
                "price_data_path": "prices.parquet",
                "tickers": ["AMD", "NVDA"],
                "horizon_days": [30],
                "lookback_days": 180,
                "news_data_paths": ["news.parquet"],
                "macro_data_paths": ["macro.parquet"],
                "materials_paths": [],
                "tags": ["historical_replay"],
            },
            "summary": {"outcome_counts": {"hit": 2}},
            "runs": runs,
        },
    )


def _run(**overrides):
    payload = {
        "run_id": "run-1",
        "as_of": "2026-03-18T00:00:00+00:00",
        "horizon_days": 30,
        "research_stance": "mixed",
        "research_expected_direction": "neutral",
        "research_confidence": 0.9,
        "research_price_agreement": "research_inconclusive",
        "exam_verdict": "price_only_candidate_not_research_confirmed",
        "evidence_document_count": 40,
        "evidence_data_quality": "strong",
        "evidence_tickers": ["AMD", "NVDA"],
        "evidence_missing_tickers": [],
        "ticker_specificity": "basket_or_sector",
        "price_action": "candidate_long",
        "price_ticker": "AMD",
        "price_expected_direction": "bullish",
        "outcome_label": "hit",
        "realized_return": 0.1,
    }
    payload.update(overrides)
    return payload


def test_directionality_diagnostic_flags_strong_neutral_runs(tmp_path):
    research = _batch(tmp_path / "research.json", [_run(), _run(run_id="run-2", as_of="2026-03-25T00:00:00+00:00")])

    payload = ResearchReplayDirectionalityDiagnostic(tmp_path / "reports").build(research_batch_path=research, save=False)

    assert payload["summary"]["diagnostic_status"] == "diagnose_base_analyst_rules"
    assert payload["summary"]["strong_inconclusive_run_count"] == 2
    assert payload["issue_counts"]["strong_evidence_still_inconclusive"] == 2
    assert any(task["task_id"] == "inspect_base_analyst_directionality_rules" for task in payload["diagnostic_tasks"])
    assert payload["safety"]["learning_write_performed"] is False


def test_directionality_diagnostic_keeps_evidence_blocker_visible(tmp_path):
    research = _batch(
        tmp_path / "research.json",
        [
            _run(
                evidence_document_count=3,
                evidence_data_quality="partial",
                evidence_missing_tickers=["AAPL"],
                evidence_tickers=["AMD"],
                price_ticker="AAPL",
            )
        ],
    )

    payload = ResearchReplayDirectionalityDiagnostic(tmp_path / "reports").build(research_batch_path=research, save=False)

    assert payload["summary"]["diagnostic_status"] == "evidence_and_directionality_blocked"
    assert payload["summary"]["missing_tickers"] == ["AAPL"]
    assert payload["run_diagnostics"][0]["primary_diagnosis"] == "missing_ticker_evidence"
    assert any(task["task_id"] == "backfill_missing_ticker_evidence" for task in payload["diagnostic_tasks"])


def test_directionality_diagnostic_reports_ready_when_directional(tmp_path):
    research = _batch(
        tmp_path / "research.json",
        [
            _run(
                research_stance="bullish",
                research_expected_direction="bullish",
                research_price_agreement="confirmed",
                ticker_specificity="ticker_specific",
            )
        ],
    )

    payload = ResearchReplayDirectionalityDiagnostic(tmp_path / "reports").build(research_batch_path=research, save=False)

    assert payload["summary"]["diagnostic_status"] == "directionality_ready"
    assert payload["summary"]["directional_run_count"] == 1
    assert payload["issue_counts"] == {}
    assert payload["commands"]["strong_evidence_replay_batch"] is not None
