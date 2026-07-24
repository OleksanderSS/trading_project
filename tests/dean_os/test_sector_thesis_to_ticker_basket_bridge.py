from __future__ import annotations

import json
import hashlib

import pandas as pd

from dean_os.analysts._producers.news import (
    SavedSemiconductorNewsEvidenceProducer,
)
from dean_os.analysts._producers.ticker import (
    SavedTickerSpecificEvidenceProducer,
)
from dean_os.sector_thesis_to_ticker_basket_bridge import SectorThesisToTickerBasketBridge


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _run(ticker, status="focused_overlay_ready", direction="bullish", outcome="hit"):
    return {
        "as_of": "2026-03-18T00:00:00+00:00",
        "horizon_days": 30,
        "price_ticker": ticker,
        "research_stance": "constructive" if direction == "bullish" else "mixed",
        "research_expected_direction": direction,
        "ticker_specificity": "single_ticker" if status == "focused_overlay_ready" else "none",
        "exam_verdict": "aligned_hit" if direction == "bullish" and status == "focused_overlay_ready" else "focused_note_blocked",
        "focused_overlay_status": status,
        "focused_overlay_applied": True,
        "evaluation_status": "evaluated",
        "outcome_label": outcome,
        "realized_return": 0.12,
        "quality_status": "clear",
    }


def _current_thesis_review():
    return {
        "run_id": "thesis_review_current_fixture",
        "mode": "domain_analyst_thesis_review_packet",
        "summary": {
            "packet_status": (
                "domain_thesis_review_ready_with_cautions"
            ),
            "domain_id": "semiconductor_ai_infrastructure",
        },
        "thesis_snapshot": {
            "domain_id": "semiconductor_ai_infrastructure",
            "stance": "mixed",
            "expected_direction": "mixed",
            "confidence": 0.52,
            "thesis": "Semiconductor sector evidence is mixed.",
        },
        "analytical_review": {
            "scope_decision": "sector_thesis_only",
            "ticker_decision": "no_direct_ticker_thesis",
            "confidence_interpretation": (
                "Evidence-quality heuristic only."
            ),
            "evidence_balance": {
                "raw_context_item_count": 152,
                "required_lane_count": 5,
                "satisfied_required_lane_count": 5,
            },
            "quality_cautions": ["short market window"],
        },
        "linked_artifact_verification": {
            "all_hashes_match": True,
            "status": "verified",
        },
        "ticker_bridge_boundary": {
            "ticker_candidates": [
                {
                    "ticker": ticker,
                    "candidate_status": "basket_candidate",
                    "ticker_specific_evidence_count": 0,
                    "required_missing_evidence": [
                        "ticker_specific_evidence"
                    ],
                }
                for ticker in ("NVDA", "AMD", "INTC", "TSM")
            ]
        },
    }


def _pipeline_case():
    return {
        "run_id": "pipeline_case_fixture",
        "mode": "pipeline_model_case_packet",
        "summary": {
            "case_status": "evaluation_block_case_ready",
            "case_id": "pipeline_model_case:fixture",
            "case_scope": "ticker_model_evaluation_only",
            "case_classification": (
                "negative_evaluation_block_case"
            ),
            "eligible_as_domain_evidence": False,
            "result_label": (
                "failed_validation_and_feature_stability"
            ),
            "review_disposition": (
                "retain_as_negative_review_case_wait_for_new_forward_data"
            ),
            "blocked_metric_planes": [
                "validation",
                "feature_stability",
            ],
            "can_use_as_forecast_outcome": False,
            "can_trade": False,
        },
        "case": {
            "evaluated_at": "2026-06-24T19:30:00+00:00",
            "lineage": {
                "ticker": "AMD",
                "model": "random_forest",
                "target_name": "target_intraday_up_15m",
                "timeframe": "15m",
                "context_fingerprint": "a" * 64,
            },
        },
    }


def _prediction_review(*, sector_overlay=False):
    context = {
        "context_key": "AMD_target_return_5d_random_forest",
        "ticker": "AMD",
        "model_context_id": (
            "AMD_target_return_5d_random_forest"
        ),
        "target_name": "target_return_5d",
        "model_type": "random_forest",
        "timeframe": None,
        "context_fingerprint": "normal",
        "selected_primary_model": "random_forest",
        "lineage_status": "incomplete",
        "missing_lineage_fields": ["timeframe"],
        "review_issues": [
            "prediction_as_of_missing",
            "context_fingerprint_placeholder_or_pattern",
        ],
        "prediction": {"as_of": None},
    }
    return {
        "run_id": "prediction_review_fixture",
        "mode": "pipeline_prediction_review_packet",
        "schema_version": "dean_stage5_prediction_review_v1",
        "status": "stage5_prediction_review_partial",
        "source_contract": "pipeline_stage5_prediction_results",
        "source_artifact": {
            "path": "stage_5_results.json",
            "available": True,
            "sha256": "b" * 64,
            "immutable_binding_ready": True,
        },
        "sector_context_review": {
            "available": sector_overlay,
        },
        "source_context_count": 1,
        "excluded_by_scope_count": 0,
        "context_count": 1,
        "complete_context_count": 0,
        "review_issue_counts": {
            "prediction_as_of_missing": 1,
            "context_fingerprint_placeholder_or_pattern": 1,
        },
        "missing_lineage_field_counts": {"timeframe": 1},
        "contexts": [context],
        "packet_fingerprint": "c" * 64,
        "safety": {
            "supporting_review_only": True,
            "decision_influence": False,
            "can_promote_model": False,
            "can_trade": False,
        },
    }


def _feature_timeframe_audit(features_path):
    feature_sha = hashlib.sha256(features_path.read_bytes()).hexdigest()
    return {
        "run_id": "feature_timeframe_audit_fixture",
        "mode": "pipeline_feature_timeframe_audit",
        "schema_version": (
            "dean_pipeline_feature_timeframe_audit_v1"
        ),
        "status": (
            "pipeline_feature_timeframe_audit_blocked_mismatch"
        ),
        "inputs": {
            "features_path": str(features_path),
            "features_sha256": feature_sha,
        },
        "summary": {
            "timeframe_mismatch_ticker_count": 1,
            "timeframe_mismatch_tickers": ["AMD"],
            "can_use_for_stage4": False,
            "can_use_for_stage5": False,
            "can_trade": False,
        },
        "ticker_timeframe_reports": [
            {
                "ticker": "AMD",
                "status": "timeframe_cadence_mismatch",
                "row_count": 12,
                "datetime_timezone_aware": False,
                "lineage": {
                    "declared_timeframe": "1d",
                    "observed_timeframe": "15m",
                    "resolved_timeframe": None,
                },
            }
        ],
        "stage5_candidate_binding": {
            "sha256": "b" * 64,
            "relationship_status": (
                "co_located_same_batch_candidate_not_hash_bound"
            ),
            "can_assert_feature_parentage": False,
        },
        "safety": {
            "read_only": True,
            "training_performed": False,
            "can_promote_model": False,
            "can_trade": False,
        },
    }


def test_sector_bridge_maps_sector_thesis_to_direct_and_blocked_candidates(tmp_path):
    batch_path = tmp_path / "batch.json"
    _write_json(
        batch_path,
        {
            "summary": {
                "total_runs": 3,
                "research_stance_counts": {"constructive": 2, "insufficient_data": 1},
                "weak_evidence_runs": 1,
                "research_inconclusive_runs": 1,
                "hit_rate": 0.8,
            },
            "runs": [
                _run("AMD"),
                _run("TSM", status="blocked_focused_note_not_ready", direction="neutral", outcome="miss"),
                _run("NVDA"),
            ],
        },
    )

    payload = SectorThesisToTickerBasketBridge(tmp_path / "reports").build(
        research_batch_path=batch_path,
        domain_profile="semiconductor_ai_infrastructure",
        sector="semiconductor",
        save=False,
    )

    assert payload["summary"]["bridge_status"] == "partial_basket_ready"
    assert payload["summary"]["direct_ticker_thesis_ready_count"] == 2
    assert payload["summary"]["blocked_candidate_count"] == 1
    statuses = {candidate["ticker"]: candidate["candidate_status"] for candidate in payload["ticker_candidates"]}
    assert statuses["AMD"] == "direct_ticker_thesis_ready"
    assert statuses["NVDA"] == "direct_ticker_thesis_ready"
    assert statuses["TSM"] == "blocked_missing_ticker_evidence"
    assert payload["safety"]["learning_write_performed"] is False


def test_sector_bridge_keeps_neutral_ready_ticker_as_context(tmp_path):
    batch_path = tmp_path / "batch.json"
    _write_json(
        batch_path,
        {
            "summary": {"total_runs": 1, "research_stance_counts": {"mixed": 1}},
            "runs": [_run("AMD", direction="neutral")],
        },
    )

    payload = SectorThesisToTickerBasketBridge(tmp_path / "reports").build(batch_path, save=False)

    candidate = payload["ticker_candidates"][0]
    assert candidate["candidate_status"] == "ticker_context_ready"
    assert candidate["allocation_guidance"] == "watchlist_context_only"
    assert payload["summary"]["bridge_status"] == "sector_context_only"


def test_sector_bridge_documents_domain_contract(tmp_path):
    batch_path = tmp_path / "batch.json"
    _write_json(batch_path, {"summary": {}, "runs": []})

    payload = SectorThesisToTickerBasketBridge(tmp_path / "reports").build(batch_path, save=False)

    contract = payload["domain_analyst_contract"]
    assert "sector_thesis" in contract["required_outputs"]
    assert "ticker_candidate_map" in contract["required_outputs"]
    assert payload["summary"]["bridge_status"] == "no_runs"


def test_sector_bridge_marks_basket_partial_when_ready_candidate_has_blocked_windows(tmp_path):
    batch_path = tmp_path / "batch.json"
    _write_json(
        batch_path,
        {
            "summary": {"total_runs": 2, "research_stance_counts": {"constructive": 1, "insufficient_data": 1}},
            "runs": [
                _run("TSM"),
                _run("TSM", status="blocked_focused_note_not_ready", direction="neutral", outcome="miss"),
            ],
        },
    )

    payload = SectorThesisToTickerBasketBridge(tmp_path / "reports").build(batch_path, save=False)

    assert payload["summary"]["bridge_status"] == "partial_basket_ready"
    assert payload["summary"]["direct_ticker_thesis_ready_count"] == 1
    assert payload["summary"]["evidence_limited_direct_candidate_count"] == 1
    assert payload["ticker_candidates"][0]["candidate_status"] == "direct_ticker_thesis_ready"
    assert "some_windows_blocked_by_weak_direct_evidence" in payload["ticker_candidates"][0]["limitations"]


def test_current_bridge_keeps_sector_context_out_of_exact_ticker_forecast(
    tmp_path,
):
    thesis_path = _write_json(
        tmp_path / "thesis_review.json",
        _current_thesis_review(),
    )
    pipeline_case_path = _write_json(
        tmp_path / "pipeline_case.json",
        _pipeline_case(),
    )

    payload = SectorThesisToTickerBasketBridge(
        tmp_path / "reports"
    ).build_from_current_review(
        domain_thesis_review_path=thesis_path,
        pipeline_case_paths=[pipeline_case_path],
        save=False,
    )

    assert payload["bridge_contract"] == (
        "dean_sector_context_to_exact_ticker_pipeline_v2"
    )
    assert payload["summary"]["bridge_status"] == (
        "ticker_pipeline_inputs_incomplete"
    )
    assert payload["summary"][
        "direct_ticker_thesis_ready_count"
    ] == 0
    assert payload["summary"]["blocked_candidate_count"] == 4
    assert payload["summary"]["exact_pipeline_case_count"] == 1
    assert payload["summary"]["negative_pipeline_case_count"] == 1
    assert payload["summary"]["can_create_ticker_forecast"] is False
    candidates = {
        item["ticker"]: item
        for item in payload["ticker_candidates"]
    }
    assert candidates["AMD"]["sector_context"][
        "can_influence_ticker_direction"
    ] is False
    assert candidates["AMD"]["exact_pipeline_contexts"][0][
        "case_classification"
    ] == "negative_evaluation_block_case"
    assert (
        "new_forward_development_data_after_blocked_case_window"
        in candidates["AMD"]["required_next_inputs"]
    )
    assert candidates["NVDA"]["exact_pipeline_contexts"] == []
    assert payload["safety"][
        "sector_context_promoted_to_ticker_evidence"
    ] is False
    assert len(
        payload["inputs"]["domain_thesis_review_sha256"]
    ) == 64


def test_current_bridge_quarantines_real_incomplete_stage5_review(
    tmp_path,
):
    thesis_path = _write_json(
        tmp_path / "thesis_review.json",
        _current_thesis_review(),
    )
    prediction_path = _write_json(
        tmp_path / "prediction_review.json",
        _prediction_review(),
    )

    payload = SectorThesisToTickerBasketBridge(
        tmp_path / "reports"
    ).build_from_current_review(
        domain_thesis_review_path=thesis_path,
        prediction_review_path=prediction_path,
        save=False,
    )

    candidates = {
        item["ticker"]: item
        for item in payload["ticker_candidates"]
    }
    amd_review = candidates["AMD"]["stage5_prediction_review"]
    assert amd_review["status"] == "prediction_review_quarantined"
    assert amd_review["context_count"] == 1
    assert amd_review["complete_context_count"] == 0
    assert amd_review["prediction_values_exposed"] is False
    assert (
        "repair_stage4_stage5_lineage_and_regenerate_"
        "immutable_prediction_review"
        in candidates["AMD"]["required_next_inputs"]
    )
    assert (
        "trustworthy_stage5_prediction_review_exact_identity"
        not in candidates["AMD"]["required_next_inputs"]
    )
    assert candidates["NVDA"]["stage5_prediction_review"][
        "status"
    ] == "not_present_for_ticker"
    assert payload["summary"]["prediction_context_count"] == 1
    assert payload["summary"][
        "quarantined_prediction_context_count"
    ] == 1
    assert payload["safety"][
        "quarantined_prediction_promoted"
    ] is False
    task_ids = {item["task_id"] for item in payload["tasks"]}
    assert "repair_stage4_stage5_lineage_and_regenerate" in task_ids


def test_current_bridge_exposes_feature_timeframe_mismatch(
    tmp_path,
):
    thesis_path = _write_json(
        tmp_path / "thesis_review.json",
        _current_thesis_review(),
    )
    prediction_path = _write_json(
        tmp_path / "prediction_review.json",
        _prediction_review(),
    )
    features_path = tmp_path / "features.parquet"
    pd.DataFrame(
        {
            "ticker": ["AMD"],
            "datetime": [pd.Timestamp("2026-06-01")],
        }
    ).to_parquet(features_path, index=False)
    audit_path = _write_json(
        tmp_path / "feature_timeframe_audit.json",
        _feature_timeframe_audit(features_path),
    )

    payload = SectorThesisToTickerBasketBridge(
        tmp_path / "reports"
    ).build_from_current_review(
        domain_thesis_review_path=thesis_path,
        prediction_review_path=prediction_path,
        feature_timeframe_audit_path=audit_path,
        save=False,
    )

    candidates = {
        item["ticker"]: item
        for item in payload["ticker_candidates"]
    }
    amd_audit = candidates["AMD"]["feature_timeframe_audit"]
    assert amd_audit["status"] == "timeframe_cadence_mismatch"
    assert amd_audit["declared_timeframe"] == "1d"
    assert amd_audit["observed_timeframe"] == "15m"
    assert (
        "regenerate_stage2_stage3_with_cadence_validated_"
        "timeframe_before_stage4_stage5"
        in candidates["AMD"]["required_next_inputs"]
    )
    assert payload["summary"][
        "timeframe_mismatch_tickers"
    ] == ["AMD"]
    assert payload["safety"]["timeframe_mismatch_overridden"] is False


def test_current_bridge_rejects_prediction_review_with_sector_overlay(
    tmp_path,
):
    thesis_path = _write_json(
        tmp_path / "thesis_review.json",
        _current_thesis_review(),
    )
    prediction_path = _write_json(
        tmp_path / "prediction_review.json",
        _prediction_review(sector_overlay=True),
    )

    try:
        SectorThesisToTickerBasketBridge(
            tmp_path / "reports"
        ).build_from_current_review(
            domain_thesis_review_path=thesis_path,
            prediction_review_path=prediction_path,
            save=False,
        )
    except ValueError as exc:
        assert "circular artifact lineage" in str(exc)
    else:
        raise AssertionError(
            "Prediction review with sector overlay was accepted"
        )


def test_current_bridge_rejects_pipeline_case_as_domain_evidence(
    tmp_path,
):
    thesis_path = _write_json(
        tmp_path / "thesis_review.json",
        _current_thesis_review(),
    )
    unsafe_case = _pipeline_case()
    unsafe_case["summary"]["eligible_as_domain_evidence"] = True
    pipeline_case_path = _write_json(
        tmp_path / "pipeline_case.json",
        unsafe_case,
    )

    try:
        SectorThesisToTickerBasketBridge(
            tmp_path / "reports"
        ).build_from_current_review(
            domain_thesis_review_path=thesis_path,
            pipeline_case_paths=[pipeline_case_path],
            save=False,
        )
    except ValueError as exc:
        assert "ticker-only review boundary" in str(exc)
    else:
        raise AssertionError("Unsafe pipeline case was accepted")


def test_current_bridge_accepts_ticker_evidence_but_keeps_forecast_blocked(
    tmp_path,
):
    thesis_path = _write_json(
        tmp_path / "thesis_review.json",
        _current_thesis_review(),
    )
    pipeline_case_path = _write_json(
        tmp_path / "pipeline_case.json",
        _pipeline_case(),
    )
    news_source = tmp_path / "news.parquet"
    pd.DataFrame(
        [
            {
                "title": (
                    "AMD shares soar after strong AI demand "
                    "sales forecast"
                ),
                "description": None,
                "published_date": (
                    "2026-05-05T20:19:47+00:00"
                ),
                "publishedAt": None,
                "link": "https://bloomberg.test/amd",
                "url": None,
                "source": "Bloomberg.com",
            },
            {
                "title": (
                    "AMD revenue above expectations on strong "
                    "AI demand"
                ),
                "description": None,
                "published_date": (
                    "2026-05-05T22:28:48+00:00"
                ),
                "publishedAt": None,
                "link": "https://reuters.test/amd",
                "url": None,
                "source": "Reuters",
            },
        ]
    ).to_parquet(news_source, index=False)
    news_output = tmp_path / "news_output"
    SavedSemiconductorNewsEvidenceProducer(
        news_output
    ).build(
        source_path=news_source,
        as_of="2026-06-19T00:00:00+00:00",
    )
    ticker_output = tmp_path / "ticker_output"
    SavedTickerSpecificEvidenceProducer(
        ticker_output
    ).build(
        news_artifact_path=news_output / "latest.json",
        as_of="2026-06-19T00:00:00+00:00",
    )

    payload = SectorThesisToTickerBasketBridge(
        tmp_path / "reports"
    ).build_from_current_review(
        domain_thesis_review_path=thesis_path,
        pipeline_case_paths=[pipeline_case_path],
        ticker_evidence_path=ticker_output / "latest.json",
        save=False,
    )

    candidates = {
        item["ticker"]: item
        for item in payload["ticker_candidates"]
    }
    assert candidates["AMD"]["candidate_status"] == (
        "ticker_evidence_ready_pipeline_blocked"
    )
    # 2 news items x up to 2 evidence_type lanes each (market_confirmation +
    # sector_demand, since "AI demand"/"sales forecast" phrasing satisfies
    # both independently) = 4 records, not 2 -- this only became reachable
    # once semiconductor_issuer_identity_registry.yaml was filled in with the
    # domain's full ticker universe; before that this test failed earlier,
    # on a missing-registry-entry error, and never reached this assertion.
    assert candidates["AMD"]["ticker_specific_evidence"][
        "eligible_record_count"
    ] == 4
    assert (
        "ticker_specific_directional_evidence"
        not in candidates["AMD"]["required_next_inputs"]
    )
    assert candidates["NVDA"]["candidate_status"] == (
        "blocked_missing_ticker_evidence"
    )
    assert payload["summary"][
        "ticker_evidence_ready_pipeline_blocked_count"
    ] == 1
    assert payload["summary"][
        "direct_ticker_thesis_ready_count"
    ] == 0
    assert payload["summary"]["can_create_ticker_forecast"] is False
