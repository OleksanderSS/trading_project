from __future__ import annotations

import asyncio
import hashlib
import json

import pandas as pd

from dean_os.agents.risk import RiskAgent
from dean_os.agents.model_performance import (
    ModelPerformanceAgent,
    inspect_model_performance,
    inspect_real_metric_evidence_chain,
)
from dean_os.pipeline_adapter import HybridPipelineAdapter
from dean_os.schemas import MarketContext
from src.pipeline.stages.prediction.output_contract import (
    build_model_output_contract,
)


class _LocalOrchestrator:
    def __init__(self, result):
        self.result = result

    async def run_local_pipeline(self, **kwargs):
        return self.result


def _run_adapter(frame: pd.DataFrame, **extra_results):
    result = {
        "status": "completed",
        "results": {
            "features_df": frame,
            **extra_results,
        },
    }
    context = MarketContext(tickers=["NVDA"], timeframe="1d")
    adapter = HybridPipelineAdapter(
        mode="local",
        orchestrator=_LocalOrchestrator(result),
    )
    normalized = asyncio.run(adapter(context))
    return context, normalized


def test_pipeline_keeps_raw_macro_table_out_of_structured_evidence():
    macro_frame = pd.DataFrame(
        {
            "date": ["2026-05-01"],
            "series": ["cpi"],
            "value": [2.8],
        }
    )
    result = {
        "status": "completed",
        "as_of": "2026-06-30T12:00:00+00:00",
        "results": {"macro_data": macro_frame},
    }
    context = MarketContext(tickers=["AMD"], timeframe="1d")
    adapter = HybridPipelineAdapter(
        mode="local",
        orchestrator=_LocalOrchestrator(result),
    )

    asyncio.run(adapter(context))

    assert context.dataframes["macro"] is macro_frame
    assert context.macro == {}
    inventory = context.metadata["raw_macro_frame_inventory"]
    assert inventory["rows"] == 1
    assert inventory["evidence_status"] == (
        "raw_pipeline_table_not_structured_context_evidence"
    )


def test_pipeline_quarantines_future_caller_supplied_macro():
    context = MarketContext(
        tickers=["AMD"],
        timeframe="1d",
        macro={
            "cpi": {
                "value": 2.8,
                "unit": "percent_yoy",
                "period": "2026-05",
                "available_at": "2026-07-01T00:00:00+00:00",
                "source_url": (
                    "https://example.test/macro/cpi-2026-05"
                ),
            }
        },
    )
    adapter = HybridPipelineAdapter(
        mode="local",
        orchestrator=_LocalOrchestrator(
            {
                "status": "completed",
                "as_of": "2026-06-30T12:00:00+00:00",
            }
        ),
    )

    asyncio.run(adapter(context))

    assert context.macro == {}
    audit = context.metadata[
        "structured_context_point_in_time_audit"
    ]
    assert audit["reason_counts"][
        "structured_availability_after_as_of"
    ] == 1


def test_adapter_prefers_realized_returns_over_supervised_target_labels():
    frame = pd.DataFrame({
        "return": [0.01, -0.02],
        "target_return_1d": [0.8, 0.9],
    })

    context, _ = _run_adapter(frame)

    assert context.returns.tolist() == [0.01, -0.02]
    assert context.metadata.get("returns_offline_only") is not True


def test_pretrade_risk_blocks_target_labels_used_as_offline_returns():
    frame = pd.DataFrame({"target_return_1d": [0.01, -0.02]})
    context, _ = _run_adapter(frame)
    context.phase = "pre_trade"
    risk = RiskAgent(
        name="risk",
        config={"veto_level": "hard", "error_behavior": "block"},
    )

    report = asyncio.run(risk.run(context))

    assert context.metadata["returns_offline_only"] is True
    assert report.verdict == "blocked"
    assert "Supervised target labels" in report.reasons[0]


def test_adapter_exposes_small_canonical_dean_review_contract():
    frame = pd.DataFrame({"return": [0.01, -0.02]})
    context, normalized = _run_adapter(
        frame,
        prediction_results={
            "NVDA|1d|target_up_1d": {
                "ticker": "NVDA",
                "model_context_id": "ctx_nvda_1d",
                "target_name": "target_up_1d",
                "model_type": "random_forest",
                "timeframe": "1d",
                    "context_fingerprint": "fingerprint_nvda",
                    "selected_primary_model": "random_forest",
                    "model_output_contract": (
                        build_model_output_contract(
                            target_name="target_up_1d",
                            target_type="classification",
                            model_count=1,
                            contextual_adjustment_applied=True,
                            nlp_adjustment_applied=False,
                            target_scaler_applied=False,
                            classification_predict_semantics=(
                                "class_label"
                            ),
                        )
                    ),
                    "predictions": [0.61],
                "raw_forecast": 0.58,
                "predictions_by_model": {
                    "random_forest": 0.61
                },
                "confidence": 0.73,
                "anomaly_score": 0.91,
                "last_price": 151.25,
                "timestamp": "2026-06-28T12:00:00+00:00",
            }
        },
        pipeline_control_metric_artifact_manifests=[
            "data/results/stage4_manifest.json"
        ],
        evaluation_summary={
            "pipeline_control_evaluation_metric_artifacts": {
                "manifest": "data/results/stage7_manifest.json"
            },
            "analysis": {
                "_analysis_coverage": {
                    "status": "stage7_context_partitioned_analysis_recorded",
                    "context_count": 2,
                    "context_coverage": {
                        "ticker=NVDA|interval=1d": {
                            "analysis_contract_hash": "contract-v1"
                        },
                        "ticker=MSFT|interval=1d": {
                            "analysis_contract_hash": "contract-v1"
                        },
                    },
                    "executed_analyzers": [
                        "critical_signals",
                        "market_regime",
                    ],
                    "failed_analyzers": [],
                    "disabled_analyzers": ["news_impact"],
                    "evidence_class": (
                        "supporting_analysis_not_locked_evidence"
                    ),
                },
                "_stage7_analysis_contract": {
                    "price_context_partitioned": True,
                    "price_context_count": 2,
                    "price_data_source": "derived_from_features_data",
                },
                "analysis_by_context": {
                    "ticker=NVDA|interval=1d": {
                        "market_regime": {
                            "status": "completed",
                            "regime": "TRENDING_UP",
                            "confidence": 0.82,
                            "trend_strength": 0.44,
                            "timestamp": "2026-06-28T12:00:00+00:00",
                            "supporting_review_only": True,
                        }
                    },
                    "ticker=MSFT|interval=1d": {
                        "market_regime": {
                            "status": "completed",
                            "regime": "RANGING",
                            "confidence": 0.65,
                            "trend_strength": 0.05,
                            "supporting_review_only": True,
                        }
                    },
                },
            },
            "learning_review_candidate": {
                "status": "proposal_only_pending_dean_os_review"
            },
        },
        execution_status="stage_6_not_requested",
        execution_boundary={
            "effective_mode": "review_only",
            "portfolio_mutated": False,
        },
    )

    contract = normalized["dean_os_review_contract"]

    assert contract["schema_version"] == "dean_pipeline_review_contract_v1"
    assert contract["stage4_metric_artifact_manifests"] == [
        "data/results/stage4_manifest.json"
    ]
    assert contract["stage7_metric_artifacts"]["manifest"].endswith(
        "stage7_manifest.json"
    )
    assert contract["execution_status"] == "stage_6_not_requested"
    assert contract["learning_review_status"] == (
        "proposal_only_pending_dean_os_review"
    )
    analyzer_review = contract["stage7_analyzer_review"]
    assert analyzer_review["context_count"] == 2
    assert analyzer_review["context_partitioned"] is True
    assert analyzer_review["executed_analyzers"] == [
        "critical_signals",
        "market_regime",
    ]
    assert analyzer_review["can_promote_model"] is False
    assert analyzer_review["can_trade"] is False
    prediction_review = contract["stage5_prediction_review"]
    assert prediction_review["schema_version"] == (
        "dean_stage5_prediction_review_v1"
    )
    assert prediction_review["status"] == (
        "stage5_prediction_review_ready"
    )
    assert prediction_review["context_count"] == 1
    assert prediction_review["contexts"][0]["ticker"] == "NVDA"
    assert prediction_review["contexts"][0]["target_name"] == (
        "target_up_1d"
    )
    assert prediction_review["contexts"][0]["prediction"]["value"] == 0.61
    assert prediction_review["contexts"][0]["decision_influence"] is False
    assert prediction_review["safety"]["is_model_evaluation"] is False
    assert prediction_review["safety"]["can_trade"] is False
    regime_review = contract["stage7_regime_review"]
    assert regime_review["schema_version"] == (
        "dean_stage7_regime_review_v1"
    )
    assert regime_review["status"] == (
        "stage7_regime_contexts_recorded"
    )
    assert regime_review["analysis_contract_hash"] == "contract-v1"
    assert regime_review["context_count"] == 2
    assert regime_review["contexts"][0]["ticker"] == "MSFT"
    assert regime_review["contexts"][1]["ticker"] == "NVDA"
    assert regime_review["contexts"][1]["timeframe"] == "1d"
    assert regime_review["contexts"][1]["decision_influence"] is False
    assert regime_review["contexts"][1]["can_trade"] is False
    assert contract["can_trade"] is False
    assert context.metadata["pipeline_review_contract"] == contract
    assert context.metadata["stage7_regime_review"] == regime_review
    assert context.metadata["stage5_prediction_review"] == (
        prediction_review
    )


def test_model_performance_agent_references_analyzer_review_without_using_it_as_evidence():
    context = MarketContext(tickers=["NVDA"], timeframe="1d")
    context.pipeline_result = {
        "evaluation_summary": {
            "metrics": {
                "validation_score": 0.70,
                "sharpe": 1.0,
                "max_drawdown": 0.10,
                "sample_count": 100,
            },
            "timestamp": "2026-06-28T12:00:00+00:00",
        }
    }
    context.metadata["pipeline_review_contract"] = {
        "stage7_analyzer_review": {
            "status": "stage7_analyzer_coverage_recorded",
            "evidence_class": "supporting_analysis_not_locked_evidence",
            "can_promote_model": False,
            "can_trade": False,
        }
    }
    agent = ModelPerformanceAgent(
        name="model_performance",
        config={
            "as_of": "2026-06-28T13:00:00+00:00",
            "max_age_hours": 24,
        },
    )

    report = asyncio.run(agent.run(context))

    assert report.verdict == "caution"
    assert report.metrics_snapshot["stage7_analyzer_review"]["status"] == (
        "stage7_analyzer_coverage_recorded"
    )
    assert report.metrics_snapshot["stage7_analyzer_review"]["can_trade"] is False
    assert report.metrics_snapshot["source_contract"] == (
        "pipeline_stage7_evaluation_summary_metrics"
    )
    assert "pipeline_metrics_not_locked_model_evidence" in (
        report.metrics_snapshot["threshold_failures"]
    )


def test_model_performance_rejects_arbitrary_nested_analyzer_scores():
    result = inspect_model_performance(
        pipeline_result={
            "analysis": {
                "market_regime": {
                    "score": 0.99,
                    "row_count": 1000,
                },
                "timestamp": "2026-06-28T12:00:00+00:00",
            }
        },
    )

    assert result["status"] == "unavailable"
    assert result["verdict"] == "caution"
    assert result["metrics"] == {}
    assert "no canonical evaluation_summary.metrics" in result["reason"]


def test_model_performance_requires_complete_canonical_metric_set():
    result = inspect_model_performance(
        pipeline_result={
            "evaluation_summary": {
                "metrics": {
                    "sharpe": 1.0,
                    "max_drawdown": 0.1,
                },
                "timestamp": "2026-06-28T12:00:00+00:00",
            }
        },
        as_of=pd.Timestamp("2026-06-28T13:00:00Z").to_pydatetime(),
    )

    assert result["verdict"] == "caution"
    assert "missing_validation_score" in result["threshold_failures"]
    assert "missing_sample_count" in result["threshold_failures"]


def test_model_performance_accepts_verified_locked_artifact_and_uses_evaluation_window(
    tmp_path,
):
    artifact = tmp_path / "locked_model_evaluation.json"
    artifact.write_text(
        json.dumps({
            "artifact_class": "locked_model_evaluation",
            "evidence_class": (
                "assembled_from_joined_training_and_stage_7_evaluation_candidates"
            ),
            "metrics": {
                "validation_score": 0.70,
                "sharpe": 1.0,
                "max_drawdown": 0.10,
                "sample_count": 100,
            },
            "joined_lineage": {
                "ticker": "NVDA",
                "model": "random_forest",
                "target_name": "target_up_1d",
                "timeframe": "1d",
                "context_fingerprint": "ctx_locked",
                "evaluation_window": {
                    "training": {
                        "start": "2026-06-01T00:00:00+00:00",
                        "end": "2026-06-20T00:00:00+00:00",
                    },
                    "evaluation": {
                        "start": "2026-06-01T00:00:00+00:00",
                        "end": "2026-06-20T00:00:00+00:00",
                    },
                },
            },
            "join_contract": {
                "join_status": "same_window_lineage_proven"
            },
            "created_at": "2026-06-28T12:00:00+00:00",
        }),
        encoding="utf-8",
    )

    result = inspect_model_performance(
        performance_path=artifact,
        as_of=pd.Timestamp("2026-06-20T12:00:00Z").to_pydatetime(),
        max_age_hours=24,
    )

    assert result["verdict"] == "clear"
    assert result["source_contract"] == (
        "verified_locked_model_evaluation_artifact"
    )
    assert result["evaluated_at"] == "2026-06-20T00:00:00+00:00"
    assert result["evidence_provenance"]["valid"] is True


def test_model_performance_rejects_complete_unlocked_file(tmp_path):
    artifact = tmp_path / "unlocked_metrics.json"
    artifact.write_text(
        json.dumps({
            "metrics": {
                "validation_score": 0.99,
                "sharpe": 3.0,
                "max_drawdown": 0.01,
                "sample_count": 1000,
            }
        }),
        encoding="utf-8",
    )

    result = inspect_model_performance(performance_path=artifact)

    assert result["status"] == "unavailable"
    assert result["verdict"] == "caution"
    assert "not a verified locked model evaluation" in result["reason"]


def test_model_performance_agent_respects_full_evidence_chain_block(tmp_path):
    model_path = tmp_path / "locked_model.json"
    model_path.write_text(
        json.dumps({
            "artifact_class": "locked_model_evaluation",
            "metrics": {
                "validation_score": 0.70,
                "sharpe": 1.0,
                "max_drawdown": 0.10,
                "sample_count": 100,
            },
            "joined_lineage": {
                "ticker": "NVDA",
                "model": "random_forest",
                "target_name": "target_up_1d",
                "timeframe": "1d",
                "context_fingerprint": "ctx_locked",
                "evaluation_window": {
                    "training": {
                        "start": "2026-06-01T00:00:00+00:00",
                        "end": "2026-06-20T00:00:00+00:00",
                    },
                    "evaluation": {
                        "start": "2026-06-01T00:00:00+00:00",
                        "end": "2026-06-20T00:00:00+00:00",
                    },
                },
            },
            "join_contract": {
                "join_status": "same_window_lineage_proven"
            },
        }),
        encoding="utf-8",
    )
    chain_path = tmp_path / "evidence_chain.json"
    chain_path.write_text(
        json.dumps({
            "mode": "pipeline_control_real_metric_evidence_run",
            "inputs": {
                "model_evaluation_json": str(model_path),
                "model_evaluation_sha256": hashlib.sha256(
                    model_path.read_bytes()
                ).hexdigest(),
            },
            "summary": {
                "real_metric_evidence_status": (
                    "real_metric_evidence_blocked_by_metric_planes"
                ),
                "can_use_as_metric_evidence": True,
                "can_clear_current_real_cautions": False,
                "blocked_metric_planes": [
                    "validation",
                    "feature_stability",
                ],
            },
        }),
        encoding="utf-8",
    )
    model_case_path = tmp_path / "pipeline_model_case.json"
    model_case_path.write_text(
        json.dumps({
            "mode": "pipeline_model_case_packet",
            "inputs": {
                "model_evaluation_json": str(model_path),
                "model_evaluation_sha256": hashlib.sha256(
                    model_path.read_bytes()
                ).hexdigest(),
                "real_metric_evidence_json": str(chain_path),
                "real_metric_evidence_sha256": hashlib.sha256(
                    chain_path.read_bytes()
                ).hexdigest(),
            },
            "summary": {
                "case_status": "evaluation_block_case_ready",
                "case_id": "pipeline_model_case:test",
                "case_classification": (
                    "negative_evaluation_block_case"
                ),
                "result_label": (
                    "failed_validation_and_feature_stability"
                ),
                "blocked_metric_planes": [
                    "validation",
                    "feature_stability",
                ],
                "root_cause_categories": [
                    "generalization_gap",
                    "feature_instability",
                ],
            },
        }),
        encoding="utf-8",
    )
    context = MarketContext(tickers=["NVDA"], timeframe="1d")
    agent = ModelPerformanceAgent(
        name="model_performance",
        config={
            "performance_path": str(model_path),
            "evidence_chain_path": str(chain_path),
            "model_case_path": str(model_case_path),
            "as_of": "2026-06-20T12:00:00+00:00",
            "max_age_hours": 24,
        },
    )

    report = asyncio.run(agent.run(context))
    chain = inspect_real_metric_evidence_chain(
        chain_path,
        expected_model_evaluation_path=model_path,
    )

    assert chain["status"] == (
        "real_metric_evidence_blocked_by_metric_planes"
    )
    assert chain["model_evaluation_path_matches"] is True
    assert chain["model_evaluation_sha256_matches"] is True
    assert report.verdict == "caution"
    assert report.metrics_snapshot["signal_strength"] == 0.0
    assert "real_metric_evidence_chain_not_ready" in (
        report.metrics_snapshot["threshold_failures"]
    )
    assert report.metrics_snapshot["pipeline_model_case"]["status"] == (
        "evaluation_block_case_ready"
    )
    assert "pipeline_model_negative_evaluation_case" in (
        report.metrics_snapshot["threshold_failures"]
    )
    assert report.metrics_snapshot["can_trade"] is False
