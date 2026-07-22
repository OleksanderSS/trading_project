from __future__ import annotations

import json

from dean_os.ticker_specific_attribution_audit import TickerSpecificAttributionAudit


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _make_fixture(tmp_path, *, price_ticker="AMD", direct_docs=3, selected_note_tickers=None):
    selected_note_tickers = selected_note_tickers or [price_ticker]
    evidence_path = tmp_path / "evidence_pack" / "latest.json"
    documents = [
        {"title": f"{price_ticker} direct doc {index}", "tickers": [price_ticker], "source_type": "news"}
        for index in range(direct_docs)
    ]
    documents.append({"title": "NVDA sector doc", "tickers": ["NVDA"], "source_type": "news"})
    _write_json(
        evidence_path,
        {
            "coverage": {
                "document_count": len(documents),
                "by_ticker": {price_ticker: direct_docs, "NVDA": 1},
                "missing_requested_tickers": [],
            },
            "documents": documents,
        },
    )
    run_path = tmp_path / "run.json"
    _write_json(
        run_path,
        {
            "run_id": "run-1",
            "inputs": {"as_of": "2026-03-18T00:00:00+00:00"},
            "research_exam": {
                "selected_note_agent": "evidence_synthesis",
                "research_stance": "constructive",
                "research_expected_direction": "bullish",
                "research_price_agreement": "aligned",
                "exam_verdict": "aligned_hit",
                "ticker_specificity": "single_ticker" if len(selected_note_tickers) == 1 else "basket_or_sector",
            },
            "evidence_pack": {"saved_paths": {"latest_json": str(evidence_path)}, "coverage": {}},
            "agent_lab": {
                "research_notes": [
                    {
                        "agent_name": "evidence_synthesis",
                        "tickers": selected_note_tickers,
                        "patterns": ["ai_compute_cycle"],
                        "tailwinds": ["ai_compute_cycle"],
                        "headwinds": [],
                        "citation_count": len(documents),
                    }
                ]
            },
            "price_replay": {"decision": {"ticker": price_ticker}},
        },
    )
    batch_path = tmp_path / "batch.json"
    _write_json(
        batch_path,
        {
            "summary": {"total_runs": 1},
            "inputs": {"price_data_path": "prices.parquet", "tickers": ["AMD", "NVDA"], "horizon_days": [30]},
            "runs": [
                {
                    "run_id": "run-1",
                    "as_of": "2026-03-18T00:00:00+00:00",
                    "horizon_days": 30,
                    "price_ticker": price_ticker,
                    "research_stance": "constructive",
                    "research_expected_direction": "bullish",
                    "saved_paths": {"json": str(run_path)},
                }
            ],
        },
    )
    return batch_path


def test_ticker_specific_attribution_ready_when_direct_and_note_specific(tmp_path):
    batch = _make_fixture(tmp_path, price_ticker="AMD", direct_docs=3, selected_note_tickers=["AMD"])

    payload = TickerSpecificAttributionAudit(tmp_path / "reports").build(research_batch_path=batch, save=False)

    assert payload["summary"]["attribution_status"] == "ticker_attribution_ready"
    assert payload["summary"]["ticker_ready_run_count"] == 1
    assert payload["run_audits"][0]["attribution_status"] == "ticker_specific_ready"
    assert payload["safety"]["learning_write_performed"] is False


def test_ticker_specific_attribution_blocks_weak_direct_docs(tmp_path):
    batch = _make_fixture(tmp_path, price_ticker="TSM", direct_docs=1, selected_note_tickers=["TSM"])

    payload = TickerSpecificAttributionAudit(tmp_path / "reports").build(research_batch_path=batch, save=False)

    assert payload["summary"]["attribution_status"] == "blocked_weak_ticker_evidence"
    assert payload["run_audits"][0]["attribution_status"] == "blocked_weak_direct_evidence"
    assert "weak_direct_price_ticker_documents" in payload["run_audits"][0]["issues"]
    assert any(task["task_id"] == "backfill_direct_price_ticker_documents" for task in payload["attribution_tasks"])


def test_ticker_specific_attribution_blocks_basket_selected_note(tmp_path):
    batch = _make_fixture(tmp_path, price_ticker="AMD", direct_docs=5, selected_note_tickers=["AAPL", "AMD", "NVDA", "TSM"])

    payload = TickerSpecificAttributionAudit(tmp_path / "reports").build(research_batch_path=batch, save=False)

    assert payload["summary"]["attribution_status"] == "blocked_basket_attribution"
    assert payload["summary"]["basket_note_run_count"] == 1
    assert payload["run_audits"][0]["attribution_status"] == "needs_ticker_specific_attribution"
    assert any(task["task_id"] == "improve_ticker_specific_note_selection" for task in payload["attribution_tasks"])


def test_ticker_specific_attribution_uses_applied_focused_overlay(tmp_path):
    batch = _make_fixture(tmp_path, price_ticker="AMD", direct_docs=5, selected_note_tickers=["AAPL", "AMD", "NVDA", "TSM"])
    batch_payload = json.loads(batch.read_text(encoding="utf-8"))
    run_path = tmp_path / "run.json"
    run_payload = json.loads(run_path.read_text(encoding="utf-8"))
    run_payload["research_exam"] = {
        **run_payload["research_exam"],
        "selected_note_agent": "ticker_focused_research_note_builder",
        "ticker_specificity": "single_ticker",
        "focused_overlay_applied": True,
        "focused_overlay_status": "focused_overlay_ready",
    }
    run_payload["focused_research_exam_overlay"] = {
        "overlay_status": "focused_overlay_ready",
        "focused_note": {"note_id": "note-amd", "price_ticker": "AMD", "citation_count": 5},
    }
    _write_json(run_path, run_payload)
    _write_json(batch, batch_payload)

    payload = TickerSpecificAttributionAudit(tmp_path / "reports").build(research_batch_path=batch, save=False)

    assert payload["summary"]["attribution_status"] == "ticker_attribution_ready"
    assert payload["summary"]["ticker_ready_run_count"] == 1
    audit = payload["run_audits"][0]
    assert audit["selected_note_source"] == "focused_overlay"
    assert audit["selected_note_tickers"] == ["AMD"]
    assert audit["focused_overlay_applied"] is True
    assert audit["attribution_status"] == "ticker_specific_ready"
