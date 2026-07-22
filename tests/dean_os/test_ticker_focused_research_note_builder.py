from __future__ import annotations

import json

from dean_os.ticker_focused_research_note_builder import TickerFocusedResearchNoteBuilder


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _make_fixture(tmp_path, *, price_ticker="AMD", direct_docs=3):
    evidence_path = tmp_path / "evidence_pack" / "latest.json"
    documents = [
        {
            "document_id": f"{price_ticker.lower()}-{index}",
            "title": f"{price_ticker} direct catalyst {index}",
            "text": f"{price_ticker} direct evidence text {index}",
            "source_type": "news" if index % 2 == 0 else "report",
            "tickers": [price_ticker],
            "sectors": ["semiconductor"],
            "published_at": "2026-03-01T00:00:00+00:00",
        }
        for index in range(direct_docs)
    ]
    documents.append(
        {
            "document_id": "nvda-sector",
            "title": "NVDA sector catalyst",
            "text": "Sector context",
            "source_type": "news",
            "tickers": ["NVDA"],
        }
    )
    _write_json(evidence_path, {"documents": documents, "coverage": {"document_count": len(documents)}})

    run_path = tmp_path / "run.json"
    _write_json(
        run_path,
        {
            "run_id": "run-1",
            "inputs": {"as_of": "2026-03-18T00:00:00+00:00", "horizon_days": 30},
            "research_exam": {
                "selected_note_agent": "evidence_synthesis",
                "research_stance": "constructive",
                "research_expected_direction": "bullish",
                "ticker_specificity": "basket_or_sector",
                "exam_verdict": "aligned_hit",
            },
            "evidence_pack": {"saved_paths": {"latest_json": str(evidence_path)}},
            "agent_lab": {
                "research_notes": [
                    {
                        "agent_name": "evidence_synthesis",
                        "thesis": "Broad AI basket thesis.",
                        "tickers": ["AMD", "NVDA", "TSM"],
                        "patterns": ["ai_compute_cycle"],
                        "tailwinds": ["accelerator demand"],
                        "headwinds": [],
                        "risks": ["Basket note is broad."],
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
                    "saved_paths": {"json": str(run_path)},
                }
            ],
        },
    )
    return batch_path


def test_ticker_focused_note_ready_with_direct_documents(tmp_path):
    batch = _make_fixture(tmp_path, price_ticker="AMD", direct_docs=5)

    payload = TickerFocusedResearchNoteBuilder(tmp_path / "reports").build(research_batch_path=batch, save=False)

    assert payload["summary"]["builder_status"] == "focused_notes_ready"
    assert payload["summary"]["ready_note_count"] == 1
    note = payload["focused_notes"][0]
    assert note["note_status"] == "focused_note_ready"
    assert note["tickers"] == ["AMD"]
    assert note["citation_count"] == 5
    assert note["data_quality"] == "strong"
    assert payload["safety"]["learning_write_performed"] is False


def test_ticker_focused_note_blocks_weak_direct_evidence(tmp_path):
    batch = _make_fixture(tmp_path, price_ticker="TSM", direct_docs=1)

    payload = TickerFocusedResearchNoteBuilder(tmp_path / "reports").build(research_batch_path=batch, save=False)

    assert payload["summary"]["builder_status"] == "blocked_no_ticker_focused_notes"
    assert payload["focused_notes"][0]["note_status"] == "blocked_weak_direct_evidence"
    assert "weak_direct_price_ticker_documents" in payload["focused_notes"][0]["issues"]
    assert any(task["task_id"] == "backfill_direct_price_ticker_documents" for task in payload["tasks"])


def test_ticker_focused_note_partial_batch_when_some_windows_ready(tmp_path):
    ready_batch = _make_fixture(tmp_path / "ready", price_ticker="AMD", direct_docs=4)
    weak_batch = _make_fixture(tmp_path / "weak", price_ticker="TSM", direct_docs=1)
    ready_payload = json.loads(ready_batch.read_text(encoding="utf-8"))
    weak_payload = json.loads(weak_batch.read_text(encoding="utf-8"))
    batch_path = tmp_path / "combined" / "batch.json"
    _write_json(
        batch_path,
        {
            "summary": {"total_runs": 2},
            "inputs": ready_payload["inputs"],
            "runs": [ready_payload["runs"][0], weak_payload["runs"][0]],
        },
    )

    payload = TickerFocusedResearchNoteBuilder(tmp_path / "reports").build(research_batch_path=batch_path, save=False)

    assert payload["summary"]["builder_status"] == "partial_focused_notes_ready"
    assert payload["summary"]["ready_note_count"] == 1
    assert payload["summary"]["weak_direct_evidence_count"] == 1
