from __future__ import annotations

import json

from dean_os.ticker_focused_replay_exam_bridge import TickerFocusedReplayExamBridge


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _batch_run(run_id="run-1", as_of="2026-03-18T00:00:00+00:00", ticker="AMD", original_direction="bullish"):
    return {
        "run_id": run_id,
        "as_of": as_of,
        "horizon_days": 30,
        "price_ticker": ticker,
        "price_expected_direction": "bullish",
        "outcome_label": "hit",
        "realized_return": 0.12,
        "evaluation_status": "evaluated",
        "quality_warnings": [],
        "research_stance": "constructive" if original_direction == "bullish" else "mixed",
        "research_expected_direction": original_direction,
        "research_confidence": 0.9,
        "research_data_quality": "strong",
        "ticker_specificity": "basket_or_sector",
        "research_price_agreement": "aligned" if original_direction == "bullish" else "research_inconclusive",
        "exam_verdict": "aligned_hit" if original_direction == "bullish" else "price_only_candidate_not_research_confirmed",
        "learning_gate_status": "review_required",
    }


def _focused_note(run_id="run-1", as_of="2026-03-18T00:00:00+00:00", ticker="AMD", status="focused_note_ready", direction="bullish"):
    return {
        "note_id": f"note-{run_id}",
        "agent_name": "ticker_focused_research_note_builder",
        "run_id": run_id,
        "as_of": as_of,
        "price_ticker": ticker,
        "note_status": status,
        "research_stance": "constructive" if direction == "bullish" else "mixed",
        "expected_direction": direction,
        "confidence": 0.82,
        "data_quality": "partial",
        "direct_document_count": 5 if status == "focused_note_ready" else 1,
        "citation_count": 5 if status == "focused_note_ready" else 1,
        "issues": [] if status == "focused_note_ready" else ["weak_direct_price_ticker_documents"],
        "limitations": ["focused_note_is_candidate_only_until_runner_integration"],
        "thesis": "Focused ticker thesis.",
    }


def _make_files(tmp_path, runs, notes):
    batch_path = tmp_path / "batch.json"
    notes_path = tmp_path / "focused_notes.json"
    _write_json(batch_path, {"summary": {"total_runs": len(runs)}, "inputs": {"price_data_path": "prices.parquet"}, "runs": runs})
    _write_json(notes_path, {"summary": {"run_count": len(notes)}, "inputs": {"research_batch_path": str(batch_path)}, "focused_notes": notes})
    return batch_path, notes_path


def test_bridge_builds_ready_overlay_from_focused_note(tmp_path):
    batch_path, notes_path = _make_files(tmp_path, [_batch_run()], [_focused_note()])

    payload = TickerFocusedReplayExamBridge(tmp_path / "reports").build(batch_path, notes_path, save=False)

    assert payload["summary"]["bridge_status"] == "focused_overlay_ready"
    assert payload["summary"]["overlay_ready_count"] == 1
    overlay = payload["run_overlays"][0]
    assert overlay["overlay_status"] == "focused_overlay_ready"
    assert overlay["focused_exam"]["ticker_specificity"] == "single_ticker"
    assert overlay["focused_exam"]["exam_verdict"] == "aligned_hit"
    assert overlay["comparison"]["specificity_improved"] is True
    assert payload["safety"]["learning_write_performed"] is False


def test_bridge_blocks_weak_focused_note_without_forcing_signal(tmp_path):
    batch_path, notes_path = _make_files(tmp_path, [_batch_run(ticker="TSM")], [_focused_note(ticker="TSM", status="blocked_weak_direct_evidence")])

    payload = TickerFocusedReplayExamBridge(tmp_path / "reports").build(batch_path, notes_path, save=False)

    assert payload["summary"]["bridge_status"] == "blocked_no_focused_overlay"
    overlay = payload["run_overlays"][0]
    assert overlay["overlay_status"] == "blocked_focused_note_not_ready"
    assert overlay["focused_exam"]["research_expected_direction"] == "neutral"
    assert overlay["focused_exam"]["exam_verdict"] == "focused_note_blocked"
    assert overlay["focused_exam"]["learning_gate"]["status"] == "blocked_focused_note"


def test_bridge_reports_partial_overlay_when_some_runs_ready(tmp_path):
    runs = [
        _batch_run(run_id="run-1", ticker="AMD"),
        _batch_run(run_id="run-2", as_of="2026-03-11T00:00:00+00:00", ticker="TSM"),
    ]
    notes = [
        _focused_note(run_id="run-1", ticker="AMD"),
        _focused_note(run_id="run-2", as_of="2026-03-11T00:00:00+00:00", ticker="TSM", status="blocked_weak_direct_evidence"),
    ]
    batch_path, notes_path = _make_files(tmp_path, runs, notes)

    payload = TickerFocusedReplayExamBridge(tmp_path / "reports").build(batch_path, notes_path, save=False)

    assert payload["summary"]["bridge_status"] == "partial_focused_overlay_ready"
    assert payload["summary"]["overlay_ready_count"] == 1
    assert payload["summary"]["blocked_overlay_count"] == 1
    assert any(task["task_id"] == "keep_blocked_windows_out_of_calibration" for task in payload["tasks"])
