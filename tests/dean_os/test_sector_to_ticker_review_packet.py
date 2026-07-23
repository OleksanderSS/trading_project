from __future__ import annotations

import json

from dean_os.packets.sector_to_ticker_review_packet import DomainSpecialistReviewPacket, SectorToTickerReviewPacket


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _run(ticker, signal_level="direct_ticker_thesis", direction="bullish", stance="constructive"):
    return {
        "as_of": "2026-03-18T00:00:00+00:00",
        "horizon_days": 30,
        "price_ticker": ticker,
        "sector_signal_level": "sector_context",
        "ticker_signal_level": signal_level,
        "research_stance": stance,
        "research_expected_direction": direction,
        "ticker_specificity": "single_ticker" if signal_level == "direct_ticker_thesis" else "none",
        "exam_verdict": "aligned_hit" if signal_level == "direct_ticker_thesis" else "focused_note_blocked",
        "focused_overlay_status": "focused_overlay_ready" if signal_level == "direct_ticker_thesis" else "blocked_focused_note_not_ready",
        "focused_overlay_applied": True,
        "outcome_label": "hit",
        "realized_return": 0.12,
    }


def _candidate(ticker, status="direct_ticker_thesis_ready", blocked_runs=0, neutral_ready_runs=0):
    return {
        "ticker": ticker,
        "candidate_status": status,
        "runs": 2 if blocked_runs else 1,
        "overlay_ready_runs": 1,
        "blocked_runs": blocked_runs,
        "directional_ready_runs": 1 if status == "direct_ticker_thesis_ready" else 0,
        "neutral_ready_runs": neutral_ready_runs,
        "dominant_focused_stance": "constructive",
        "focused_direction_counts": {"bullish": 1} if status == "direct_ticker_thesis_ready" else {},
        "exam_verdict_counts": {"aligned_hit": 1},
        "outcome_counts": {"hit": 1},
        "hit_rate_context": 1.0,
        "average_realized_return_context": 0.12,
        "supporting_as_of": ["2026-03-18T00:00:00+00:00"] if status == "direct_ticker_thesis_ready" else [],
        "blocked_as_of": ["2026-03-11T00:00:00+00:00"] if blocked_runs else [],
        "allocation_guidance": "eligible_for_reviewed_ticker_candidate",
        "limitations": ["some_windows_blocked_by_weak_direct_evidence"] if blocked_runs else [],
    }


def _bridge_payload(candidates=None, runs=None, safety=None, summary=None):
    candidates = candidates if candidates is not None else [_candidate("AMD")]
    runs = runs if runs is not None else [_run("AMD")]
    return {
        "run_id": "sector_thesis_to_ticker_basket_test",
        "mode": "sector_thesis_to_ticker_basket_bridge",
        "summary": {
            "bridge_status": "ticker_basket_ready_for_review",
            "can_create_ticker_basket_review": True,
            "can_change_analyst_weights": False,
            "can_write_learning_memory": False,
            **(summary or {}),
        },
        "domain_analyst_contract": {
            "profile_id": "semiconductor_ai_infrastructure",
            "sector": "semiconductor",
            "rule": "A sector thesis may propose a basket or candidate list, but it is not a ticker thesis until direct ticker evidence supports the company.",
        },
        "sector_thesis": {
            "domain_profile": "semiconductor_ai_infrastructure",
            "sector": "semiconductor",
            "thesis_level": "sector_thesis",
            "sector_stance": "evidence_limited",
            "thesis": "Sector context only until ticker-specific evidence is reviewed.",
            "source_run_count": len(runs),
            "research_stance_counts": {"constructive": 1},
            "exam_verdict_counts": {"aligned_hit": 1},
            "evidence_quality_counts": {"strong": 1},
            "warnings": [],
        },
        "ticker_candidates": candidates,
        "mapping_runs": runs,
        "safety": {
            "read_only": True,
            "data_mutation_performed": False,
            "collector_run_performed": False,
            "network_access_performed": False,
            "pipeline_run_performed": False,
            "learning_write_performed": False,
            "operation_proposal_created": False,
            "config_write_performed": False,
            "broker_access_performed": False,
            **(safety or {}),
        },
    }


def _source_gate_payload(
    status="source_evidence_ready_with_warnings",
    can_domain=True,
    warning_count=1,
    fail_count=0,
    can_extract=False,
    can_trade=False,
):
    return {
        "run_id": "source_evidence_validation_gate_test",
        "mode": "source_evidence_validation_gate",
        "summary": {
            "gate_status": status,
            "recommended_action": "manual_domain_review_with_source_warnings",
            "artifact_type": "analyst_evidence_pack",
            "document_count": 2,
            "content_unit_count": 2,
            "can_enter_domain_research": can_domain,
            "can_promote_to_evidence": False,
            "can_extract_claims_events_entities": can_extract,
            "can_trade": can_trade,
        },
        "decision_guidance": {
            "pass_count": 5,
            "warning_count": warning_count,
            "fail_count": fail_count,
            "reasons": ["doc1 has no published_at timestamp."] if warning_count else ["Source artifact is blocked."] if fail_count else [],
        },
        "candidate_routing_indexes": {
            "entities": ["AMD", "TSM"],
            "sectors": ["semiconductor"],
            "topics": ["ai_capex_cycle"],
        },
    }


def test_sector_to_ticker_review_packet_separates_ready_and_limited_candidates(tmp_path):
    bridge_path = _write_json(
        tmp_path / "bridge.json",
        _bridge_payload(
            candidates=[_candidate("AMD"), _candidate("TSM", blocked_runs=1)],
            runs=[
                _run("AMD"),
                _run("TSM"),
                _run("TSM", signal_level="blocked_missing_ticker_evidence", direction="neutral", stance="insufficient_data"),
            ],
            summary={"bridge_status": "partial_basket_ready"},
        ),
    )

    payload = SectorToTickerReviewPacket(tmp_path / "reports").build(bridge_path=bridge_path, save=False)

    assert payload["summary"]["packet_status"] == "review_ready_with_limitations"
    assert payload["summary"]["review_ready_count"] == 1
    assert payload["summary"]["review_ready_with_limits_count"] == 1
    assert payload["summary"]["can_enter_manual_sector_to_ticker_review"] is True
    review_statuses = {item["ticker"]: item["review_status"] for item in payload["ticker_review_map"]}
    assert review_statuses["AMD"] == "review_ready"
    assert review_statuses["TSM"] == "review_ready_with_evidence_limits"
    assert payload["sector_thesis"]["review_boundary"].startswith("This is a sector thesis")
    assert payload["summary"]["can_write_learning_memory"] is False
    assert payload["summary"]["can_trade"] is False
    assert payload["blocked_or_limited_candidates"][0]["ticker"] == "TSM"


def test_sector_to_ticker_review_packet_blocks_without_direct_ticker_evidence(tmp_path):
    candidate = _candidate(
        "TSM",
        status="blocked_missing_ticker_evidence",
        blocked_runs=1,
    )
    candidate["sector_context"] = {
        "allowed_use": "supporting_context_only",
        "can_influence_ticker_direction": False,
    }
    candidate["exact_pipeline_contexts"] = [
        {
            "model": "random_forest",
            "target_name": "target_intraday_up_15m",
            "timeframe": "15m",
            "case_classification": (
                "negative_evaluation_block_case"
            ),
            "blocked_metric_planes": ["validation"],
        }
    ]
    candidate["required_next_inputs"] = [
        "ticker_specific_directional_evidence"
    ]
    candidate["can_create_ticker_forecast"] = False
    bridge_path = _write_json(
        tmp_path / "bridge.json",
        _bridge_payload(
            candidates=[candidate],
            runs=[_run("TSM", signal_level="blocked_missing_ticker_evidence", direction="neutral", stance="insufficient_data")],
            summary={"bridge_status": "sector_context_only"},
        ),
    )

    payload = SectorToTickerReviewPacket(tmp_path / "reports").build(bridge_path=bridge_path, save=False)

    assert payload["summary"]["packet_status"] == (
        "review_ready_with_limitations"
    )
    assert payload["summary"]["can_enter_manual_sector_to_ticker_review"] is True
    assert any(check["code"] == "no_direct_ticker_candidates" for check in payload["review_checks"])
    assert payload["ticker_review_map"][0]["allowed_use"] == "review_context_only"
    assert payload["summary"]["can_create_ticker_forecast"] is False
    assert payload["ticker_review_map"][0][
        "sector_context"
    ]["can_influence_ticker_direction"] is False
    assert payload["ticker_review_map"][0][
        "exact_pipeline_contexts"
    ][0]["case_classification"] == (
        "negative_evaluation_block_case"
    )
    assert payload["ticker_review_map"][0][
        "can_create_ticker_forecast"
    ] is False


def test_sector_to_ticker_review_packet_fails_on_non_read_only_bridge(tmp_path):
    bridge_path = _write_json(
        tmp_path / "bridge.json",
        _bridge_payload(
            safety={"learning_write_performed": True},
            summary={"can_write_learning_memory": True},
        ),
    )

    payload = SectorToTickerReviewPacket(tmp_path / "reports").build(bridge_path=bridge_path, save=False)

    assert payload["summary"]["packet_status"] == "blocked"
    check_codes = {check["code"] for check in payload["review_checks"] if check["status"] == "fail"}
    assert "bridge_safety_violation" in check_codes
    assert "learning_or_weight_change_enabled" in check_codes


def test_sector_to_ticker_review_separates_ticker_evidence_from_forecast(
    tmp_path,
):
    candidate = _candidate(
        "AMD",
        status="ticker_evidence_ready_pipeline_blocked",
        blocked_runs=1,
    )
    candidate["directional_ready_runs"] = 0
    candidate["ticker_specific_evidence"] = {
        "status": "company_mechanism_corroborated",
        "eligible_record_count": 3,
        "corroborated_lane_count": 1,
        "can_create_ticker_forecast": False,
    }
    candidate["required_next_inputs"] = [
        "trustworthy_stage5_prediction_review_exact_identity",
        "realized_outcome_calibration_for_target_and_horizon",
    ]
    candidate["stage5_prediction_review"] = {
        "status": "prediction_review_quarantined",
        "context_count": 98,
        "complete_context_count": 0,
        "quarantined_context_count": 98,
        "can_create_ticker_forecast": False,
    }
    candidate["feature_timeframe_audit"] = {
        "status": "timeframe_cadence_mismatch",
        "declared_timeframe": "1d",
        "observed_timeframe": "15m",
        "datetime_timezone_aware": False,
        "can_override_timeframe": False,
    }
    candidate["can_create_ticker_forecast"] = False
    bridge_path = _write_json(
        tmp_path / "bridge.json",
        _bridge_payload(
            candidates=[candidate],
            runs=[
                _run(
                    "AMD",
                    signal_level="blocked_pipeline_model_case",
                    direction="neutral",
                    stance="not_applicable",
                )
            ],
            summary={
                "bridge_status": (
                    "ticker_pipeline_inputs_incomplete"
                )
            },
        ),
    )

    payload = SectorToTickerReviewPacket(
        tmp_path / "reports"
    ).build(bridge_path=bridge_path, save=False)

    item = payload["ticker_review_map"][0]
    assert item["stage5_prediction_review"]["status"] == (
        "prediction_review_quarantined"
    )
    assert (
        "real_stage5_output_exists_but_is_quarantined"
        in item["risk_and_counter_thesis_flags"]
    )
    assert (
        "feature_timeframe_conflicts_with_observed_cadence"
        in item["risk_and_counter_thesis_flags"]
    )
    assert payload["summary"]["packet_status"] == (
        "review_ready_with_limitations"
    )
    assert payload["summary"][
        "ticker_evidence_ready_pipeline_blocked_count"
    ] == 1
    assert item["review_status"] == (
        "ticker_evidence_ready_pipeline_blocked"
    )
    assert item["allowed_use"] == (
        "manual_review_of_ticker_evidence_not_forecast"
    )
    assert item["can_create_ticker_forecast"] is False
    assert (
        "company_mechanism_corroborated_but_pipeline_not_ready"
        in item["risk_and_counter_thesis_flags"]
    )


def test_sector_to_ticker_review_packet_saves_markdown_with_guardrails(tmp_path):
    bridge_path = _write_json(tmp_path / "bridge.json", _bridge_payload())

    payload = SectorToTickerReviewPacket(tmp_path / "reports").build(bridge_path=bridge_path)

    latest_json = tmp_path / "reports" / "latest.json"
    latest_md = tmp_path / "reports" / "latest.md"
    assert latest_json.exists()
    assert latest_md.exists()
    markdown = latest_md.read_text(encoding="utf-8")
    assert "Can trade: False" in markdown
    assert "No learning memory write is performed." in markdown
    assert "No recommendation, rating, price target, or position size is created." in markdown
    assert payload["saved_paths"]["latest_markdown"].endswith("latest.md")


def test_domain_specialist_review_packet_is_domain_first_with_ticker_bridge_section(tmp_path):
    bridge_path = _write_json(
        tmp_path / "bridge.json",
        _bridge_payload(
            candidates=[_candidate("AMD"), _candidate("TSM", blocked_runs=1)],
            runs=[
                _run("AMD"),
                _run("TSM"),
                _run("TSM", signal_level="blocked_missing_ticker_evidence", direction="neutral", stance="insufficient_data"),
            ],
            summary={"bridge_status": "partial_basket_ready"},
        ),
    )

    payload = DomainSpecialistReviewPacket(tmp_path / "domain_reports").build(bridge_path=bridge_path, save=False)

    assert payload["mode"] == "domain_specialist_review_packet"
    assert payload["summary"]["packet_status"] == "domain_review_ready_with_limitations"
    assert payload["summary"]["can_enter_manual_domain_review"] is True
    assert payload["summary"]["can_enter_ticker_candidate_review"] is True
    assert payload["summary"]["can_standardize_domain_template"] is False
    assert payload["domain_first_contract"]["primary_axis"] == "domain_or_sector_research"
    assert payload["domain_first_contract"]["ticker_axis"] == "derived_bridge_only"
    assert payload["domain_thesis"]["allowed_use"] == "manual_domain_review_and_context_mapping"
    assert payload["claims_events_entities"]["extraction_status"] == "not_extracted_in_this_packet"
    assert "AMD" in payload["claims_events_entities"]["candidate_entities"]
    assert "sector_to_ticker_bridge_review" in payload
    assert payload["sector_to_ticker_bridge_review"]["ticker_review_map"][0]["ticker"] == "AMD"
    assert payload["summary"]["can_write_learning_memory"] is False
    assert payload["summary"]["can_trade"] is False


def test_domain_specialist_review_can_proceed_when_ticker_bridge_has_no_direct_evidence(tmp_path):
    bridge_path = _write_json(
        tmp_path / "bridge.json",
        _bridge_payload(
            candidates=[_candidate("TSM", status="blocked_missing_ticker_evidence", blocked_runs=1)],
            runs=[_run("TSM", signal_level="blocked_missing_ticker_evidence", direction="neutral", stance="insufficient_data")],
            summary={"bridge_status": "sector_context_only"},
        ),
    )

    payload = DomainSpecialistReviewPacket(tmp_path / "domain_reports").build(bridge_path=bridge_path, save=False)

    assert payload["summary"]["packet_status"] == "domain_review_ready_with_limitations"
    assert payload["summary"]["can_enter_manual_domain_review"] is True
    assert payload["summary"]["can_enter_ticker_candidate_review"] is False
    assert any(check["code"] == "ticker_bridge_has_no_direct_candidates" for check in payload["review_checks"])
    assert payload["sector_to_ticker_bridge_review"]["ticker_review_map"][0]["allowed_use"] == "review_context_only"


def test_domain_specialist_review_packet_saves_domain_first_markdown(tmp_path):
    bridge_path = _write_json(tmp_path / "bridge.json", _bridge_payload())

    payload = DomainSpecialistReviewPacket(tmp_path / "domain_reports").build(bridge_path=bridge_path)

    latest_md = tmp_path / "domain_reports" / "latest.md"
    assert latest_md.exists()
    markdown = latest_md.read_text(encoding="utf-8")
    assert "# DEAN-OS Domain Specialist Review Packet" in markdown
    assert "## Domain Thesis" in markdown
    assert "## Sector To Ticker Bridge" in markdown
    assert "Can trade: False" in markdown
    assert "No learning memory write is performed." in markdown
    assert payload["saved_paths"]["latest_markdown"].endswith("latest.md")


def test_domain_specialist_review_packet_attaches_source_gate_context(tmp_path):
    bridge_path = _write_json(tmp_path / "bridge.json", _bridge_payload())
    source_gate_path = _write_json(tmp_path / "source_gate.json", _source_gate_payload())

    payload = DomainSpecialistReviewPacket(tmp_path / "domain_reports").build(
        bridge_path=bridge_path,
        source_gate_path=source_gate_path,
        save=False,
    )

    assert payload["inputs"]["source_gate_run_id"] == "source_evidence_validation_gate_test"
    assert payload["summary"]["packet_status"] == "domain_review_ready_with_limitations"
    assert payload["summary"]["recommended_review_action"] == "manual_domain_review_with_source_limitations"
    assert payload["summary"]["source_gate_status"] == "source_evidence_ready_with_warnings"
    assert payload["summary"]["source_gate_warning_count"] == 1
    assert payload["source_evidence_context"]["source_gate_attached"] is True
    assert payload["source_evidence_context"]["source_document_count"] == 2
    assert payload["source_evidence_context"]["source_validation_counts"] == {"pass": 5, "warn": 1, "fail": 0}
    assert payload["source_evidence_context"]["source_candidate_entities"] == ["AMD", "TSM"]
    assert any(check["code"] == "source_gate_warnings_present" for check in payload["review_checks"])


def test_domain_specialist_review_packet_blocks_when_source_gate_blocks_domain_research(tmp_path):
    bridge_path = _write_json(tmp_path / "bridge.json", _bridge_payload())
    source_gate_path = _write_json(
        tmp_path / "source_gate.json",
        _source_gate_payload(status="blocked_source_evidence", can_domain=False, warning_count=0, fail_count=1),
    )

    payload = DomainSpecialistReviewPacket(tmp_path / "domain_reports").build(
        bridge_path=bridge_path,
        source_gate_path=source_gate_path,
        save=False,
    )

    assert payload["summary"]["packet_status"] == "blocked"
    assert payload["summary"]["can_enter_manual_domain_review"] is False
    failed_codes = {check["code"] for check in payload["review_checks"] if check["status"] == "fail"}
    assert "source_gate_blocks_domain_research" in failed_codes
