from __future__ import annotations

import json

from dean_os.hypothesis_evidence_gap_review import HypothesisEvidenceGapReview


def _write(path, payload):
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_inventory_and_capex_are_partial_not_resolved(tmp_path):
    safety = {"review_only": True, "live_execution_performed": False}
    analyst = _write(
        tmp_path / "analyst.json",
        {
            "created_at": "2026-07-01T00:00:00+00:00",
            "contract": "dean_domain_analyst_review_run_v1",
            "safety": safety,
            "agent_report": {
                "metrics_snapshot": {
                    "hypotheses": [
                        {
                            "hypothesis_id": "h_supply",
                            "hypothesis": "Supply constraints will persist for 180 days",
                            "horizons_to_check": [30, 90, 180],
                        }
                    ],
                    "evidence_gaps": [
                        {
                            "gap_id": "g_inventory",
                            "description": "Inventory levels across supply chain",
                            "priority": "medium",
                            "expected_source_type": "industry_report",
                        },
                        {
                            "gap_id": "g_capex",
                            "description": "Capex breakdown: maintenance vs. growth",
                            "priority": "high",
                            "expected_source_type": "company_filing",
                        },
                    ],
                }
            },
        },
    )
    fundamental = _write(
        tmp_path / "fundamental.json",
        {
            "created_at": "2026-07-01T00:00:00+00:00",
            "safety": safety,
            "facts": [
                {
                    "ticker": "NVDA",
                    "metric_name": "inventory",
                    "value": 10,
                    "unit": "USD",
                    "period": "2026-03-31",
                    "available_at": "2026-05-01T00:00:00+00:00",
                    "fact_sha256": "a" * 64,
                },
                {
                    "ticker": "TSM",
                    "metric_name": "capital_expenditure",
                    "value": 20,
                    "unit": "TWD",
                    "period": "2025",
                    "available_at": "2026-04-01T00:00:00+00:00",
                    "fact_sha256": "b" * 64,
                },
            ],
        },
    )

    payload = HypothesisEvidenceGapReview(tmp_path / "out").build(
        analyst_review_path=analyst,
        fundamental_artifact_path=fundamental,
        as_of="2026-07-02T00:00:00+00:00",
        save=False,
    )

    statuses = {
        item["gap_id"]: item["resolution_status"]
        for item in payload["gap_reviews"]
    }
    assert statuses == {
        "g_inventory": "partial_supported",
        "g_capex": "partial_supported",
    }
    assert payload["summary"]["fully_resolved_gap_count"] == 0
    assert payload["replay_task_candidates"][0]["registration_allowed"] is False


def test_operational_actual_supports_gap_but_does_not_close_it(tmp_path):
    safety = {"review_only": True, "live_execution_performed": False}
    analyst = _write(tmp_path / "analyst.json", {
        "created_at": "2026-07-01T00:00:00+00:00",
        "contract": "dean_domain_analyst_review_run_v1",
        "safety": safety,
        "agent_report": {"metrics_snapshot": {
            "hypotheses": [],
            "evidence_gaps": [{
                "gap_id": "g_util", "description": "Production capacity and utilization",
                "priority": "high", "expected_source_type": "industry_data",
            }],
        }},
    })
    fundamental = _write(tmp_path / "fundamental.json", {
        "created_at": "2026-07-01T00:00:00+00:00", "safety": safety, "facts": []
    })
    operational = _write(tmp_path / "operational.json", {
        "created_at": "2026-07-01T00:00:00+00:00",
        "contract": "dean_industry_operational_metrics_v1",
        "safety": {"review_only": True},
        "accepted_records": [{
            "record_id": "util_1", "entity": "Foundry", "metric_name": "capacity_utilization",
            "value": 82.5, "unit": "percent", "period": "2026-Q2",
            "available_at": "2026-06-30T00:00:00+00:00", "value_kind": "actual",
            "lifecycle_status": "active", "source_locator": "file:///source",
            "source_sha256": "a" * 64, "observation_sha256": "b" * 64,
        }],
    })

    payload = HypothesisEvidenceGapReview(tmp_path / "out").build(
        analyst_review_path=analyst,
        fundamental_artifact_path=fundamental,
        operational_metrics_path=operational,
        as_of="2026-07-02T00:00:00+00:00",
        save=False,
    )

    review = payload["gap_reviews"][0]
    assert review["resolution_status"] == "partial_supported"
    assert review["supporting_evidence"][0]["evidence_role"] == "observed_metric"
    assert review["automatic_closure_allowed"] is False


def test_operational_guidance_is_context_not_observed_support(tmp_path):
    safety = {"review_only": True, "live_execution_performed": False}
    analyst = _write(tmp_path / "analyst.json", {
        "created_at": "2026-07-01T00:00:00+00:00", "contract": "dean_domain_analyst_review_run_v1",
        "safety": safety, "agent_report": {"metrics_snapshot": {"hypotheses": [], "evidence_gaps": [{
            "gap_id": "g_orders", "description": "Equipment order series", "priority": "high",
            "expected_source_type": "industry_data",
        }]}}
    })
    fundamental = _write(tmp_path / "fundamental.json", {
        "created_at": "2026-07-01T00:00:00+00:00", "safety": safety, "facts": []
    })
    operational = _write(tmp_path / "operational.json", {
        "created_at": "2026-07-01T00:00:00+00:00", "contract": "dean_industry_operational_metrics_v1",
        "safety": {"review_only": True}, "accepted_records": [{
            "record_id": "orders_g", "entity": "Supplier", "metric_name": "equipment_orders",
            "value": 10.0, "unit": "count", "period": "2026-Q3",
            "available_at": "2026-06-30T00:00:00+00:00", "value_kind": "guidance",
            "lifecycle_status": "active", "source_locator": "file:///source",
            "source_sha256": "a" * 64, "observation_sha256": "b" * 64,
        }]
    })
    payload = HypothesisEvidenceGapReview(tmp_path / "out").build(
        analyst_review_path=analyst, fundamental_artifact_path=fundamental,
        operational_metrics_path=operational, as_of="2026-07-02T00:00:00+00:00", save=False,
    )

    review = payload["gap_reviews"][0]
    assert review["resolution_status"] == "context_only_not_resolved"
    assert review["supporting_evidence"][0]["evidence_role"] == "forward_or_estimated_context"


def test_filing_rpo_is_partial_backlog_proxy_not_gap_closure(tmp_path):
    safety = {"review_only": True, "live_execution_performed": False}
    analyst = _write(tmp_path / "analyst.json", {
        "created_at": "2026-07-01T00:00:00+00:00", "contract": "dean_domain_analyst_review_run_v1",
        "safety": safety, "agent_report": {"metrics_snapshot": {"hypotheses": [], "evidence_gaps": [{
            "gap_id": "g_backlog", "description": "Actual order backlog data vs. narrative claims",
            "priority": "high", "expected_source_type": "company_filing",
        }]}}
    })
    fundamental = _write(tmp_path / "fundamental.json", {
        "created_at": "2026-07-01T00:00:00+00:00", "safety": safety, "facts": []
    })
    filing = _write(tmp_path / "filing.json", {
        "created_at": "2026-07-01T00:00:00+00:00", "contract": "dean_filing_order_evidence_v1",
        "safety": {"review_only": True}, "observations": [{
            "ticker": "NVDA", "metric_name": "revenue_remaining_performance_obligation",
            "value": 2.6e9, "unit": "USD", "period": "2026-04-26",
            "available_at": "2026-05-20T23:59:59.999999+00:00", "accession_number": "accn",
            "source_locator": "facts.json#rpo", "source_sha256": "a" * 64,
            "observation_sha256": "b" * 64,
            "semantic_role": "contracted_revenue_proxy_not_full_order_backlog",
            "gap_support_role": "partial_support_only",
        }]
    })
    payload = HypothesisEvidenceGapReview(tmp_path / "out").build(
        analyst_review_path=analyst, fundamental_artifact_path=fundamental,
        filing_order_evidence_path=filing, as_of="2026-07-02T00:00:00+00:00", save=False,
    )
    review = payload["gap_reviews"][0]
    assert review["resolution_status"] == "partial_supported"
    assert review["supporting_evidence"][0]["evidence_role"] == "contracted_revenue_proxy_not_full_order_backlog"
    assert review["automatic_closure_allowed"] is False
