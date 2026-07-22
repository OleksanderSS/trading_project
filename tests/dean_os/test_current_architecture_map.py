from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from dean_os.current_architecture_map import CurrentArchitectureMap


def test_current_architecture_map_defines_two_branch_review_system(tmp_path):
    payload = CurrentArchitectureMap(tmp_path / "reports").build(save=False)

    assert payload["summary"]["architecture_status"] == "current_architecture_map_ready"
    assert payload["summary"]["active_design"] == "source_first_two_branch_review_system"
    assert payload["summary"]["pipeline_metric_plane_count"] >= 6
    assert payload["summary"]["domain_analyst_control_plane_count"] >= 10
    assert payload["summary"]["domain_profile_count"] >= 1
    assert payload["summary"]["can_clone_domain_profiles_now"] is False
    assert payload["summary"]["can_generate_analyst_research_recommendations_now"] is True
    assert payload["summary"]["can_generate_execution_recommendations_now"] is False
    assert payload["summary"]["can_write_production_config_now"] is False
    assert payload["summary"]["can_trade"] is False
    assert payload["summary"]["pipeline_timeframe_context_status"] == (
        "fail_closed_declared_observed_cadence_and_timezone_gated"
    )
    assert payload["summary"]["pipeline_feature_timeframe_audit_status"] == (
        "legacy_four_of_four_blocked_regenerated_four_of_four_15m_utc_ready"
    )
    assert payload["summary"]["active_bounded_stage23_status"] == (
        "four_ticker_1170_row_15m_utc_hash_bound_review_ready"
    )
    assert payload["summary"]["active_pipeline_target_readiness_status"] == (
        "seven_of_seven_semantic_targets_ready_for_bounded_stage4"
    )
    assert payload["summary"]["active_exact_stage4_review_status"] == (
        "nvda_15m_587_rows_three_folds_hash_bound_validation_contract_"
        "blocked"
    )
    assert payload["summary"]["active_colab_identity_status"] == (
        "ticker_datetime_interval_dedup_and_hash_lineage_enforced"
    )
    assert payload["summary"]["active_stage1_source_cadence_status"] == (
        "saved_15m_usable_saved_60m_and_1d_labels_rejected"
    )
    assert payload["summary"]["active_composite_domain_pipeline_status"] == (
        "real_152_item_semiconductor_smoke_caution_readiness_blocked_no_"
        "decision_influence"
    )
    assert payload["summary"]["active_domain_agent_topology_status"] == (
        "composite_manager_canonical_standalone_analyst_alternative"
    )
    assert payload["summary"]["active_agent_execution_group_status"] == (
        "overlapping_enabled_domain_agents_fail_closed"
    )
    assert payload["summary"]["active_agent_phase_status"] == (
        "registry_run_phases_enforced_before_instantiation"
    )
    assert payload["summary"]["active_scaffold_audit_status"] == (
        "parallel_scaffolds_retained_stateful_and_domain_fanout_default_off"
    )
    assert payload["summary"]["active_registry_activation_status"] == (
        "thirty_seven_registered_sixteen_review_only_enabled_expensive_"
        "domain_composite_and_mutating_agents_default_off"
    )
    assert payload["summary"]["active_runtime_cutoff_status"] == (
        "requested_as_of_must_equal_verified_runtime_cutoff"
    )
    assert payload["summary"]["pipeline_walk_forward_status"] == (
        "three_fold_development_candidate_predictive_and_stability_checks_"
        "passed_but_overfit_and_class_balance_checks_blocked"
    )
    assert payload["summary"]["pipeline_forward_accrual_status"] == "prospective_development_boundary_registered"
    assert payload["summary"]["pipeline_forward_accrual_gate_status"] == "existing_pre_registration_artifact_blocked"
    assert payload["summary"]["pipeline_forward_runner_status"] == "accepts_passing_accrual_gate_only"
    assert payload["summary"]["active_stage4_training_contract_status"] == "nested_split_adapter_fixed_validation_only_selection"
    assert payload["summary"]["active_stage4_evidence_status"] == "partial_training_candidates_enabled"
    assert payload["summary"]["active_stage5_lineage_status"] == "target_model_timeframe_context_propagated"
    assert payload["summary"]["active_stage5_prediction_review_status"] == (
        "per_context_output_contract_required_sector_ticker_context_"
        "supporting_only"
    )
    assert payload["summary"]["active_stage5_prediction_artifact_status"] == (
        "no_trustworthy_saved_current_result_do_not_fabricate"
    )
    assert payload["summary"]["active_final_pipeline_status"] == "stage5_to_stage7_review_default_stage6_explicit"
    assert payload["summary"]["active_stage6_execution_status"] == "review_only_no_paper_no_live_no_memory_write"
    assert payload["summary"]["active_paper_lifecycle_status"] == (
        "hash_bound_receipt_plan_external_result_review_no_executor_run"
    )
    assert payload["summary"]["active_paper_lineage_status"] == (
        "post_dry_review_receipt_plan_external_manifest_result_review_bound"
    )
    assert payload["summary"]["active_stage7_learning_status"] == "proposal_only_no_automatic_adaptation"
    assert payload["summary"]["active_stage7_notification_status"] == "explicit_per_run_authorization_required"
    assert payload["summary"]["active_stage7_analyzer_status"] == "context_partitioned_supporting_review_only"
    assert payload["summary"]["active_analyzer_suite_status"] == "two_enabled_ten_explicitly_staged"
    assert payload["summary"]["active_analyzer_observability_status"] == "executed_skipped_failed_disabled_coverage_recorded"
    assert payload["summary"]["active_analyzer_cache_status"] == "data_and_suite_contract_fingerprinted"
    assert payload["summary"]["active_stage7_agent_bridge_status"] == (
        "per_context_regime_review_consumed_by_shadow_regime_agent"
    )
    assert payload["summary"]["active_regime_agent_status"] == (
        "enabled_pretrade_stage7_only_shadow_no_decision_influence"
    )
    assert payload["summary"]["active_agent_capability_matrix_status"] == (
        "registry_agents_contract_mapped_parallel_scaffold_needs_matrix_"
        "refresh"
    )
    assert payload["summary"]["active_context_synthesis_status"] == (
        "stage5_stage7_exact_context_shadow_compatibility_only"
    )
    assert payload["summary"]["active_context_freshness_status"] == (
        "stage7_price_window_provenance_and_as_of_skew_checked"
    )
    assert payload["summary"]["active_specialist_context_status"] == (
        "sector_direct_ticker_point_in_time_scopes_separated"
    )
    assert payload["summary"][
        "active_amd_specialist_context_status"
    ] == (
        "direct_ticker_manual_review_candidate_stale_unaligned_"
        "not_approved"
    )
    assert payload["summary"]["active_semiconductor_amd_boundary"] == (
        "semiconductor_domain_context_is_not_amd_ticker_evidence"
    )
    assert payload["summary"]["parallel_template_audit_status"] == (
        "source_folder_empty_after_transfer_active_tree_contains_workbench"
    )
    assert payload["summary"]["transferred_workbench_status"] == (
        "runtime_foundation_integrated_legacy_islands_classified"
    )
    assert payload["summary"]["template_harvest_status"] == (
        "eval_unit_period_time_leakage_and_safety_rules_adapted"
    )
    assert payload["summary"][
        "active_prediction_target_semantics_status"
    ] == (
        "canonical_target_period_unit_threshold_positive_class_bound"
    )
    assert payload["summary"]["active_stage5_output_scale_status"] == (
        "explicit_predict_output_contract_directional_use_still_blocked"
    )
    assert payload["summary"]["active_shadow_calibration_status"] == (
        "blocked_zero_of_thirty_cases_all_components"
    )
    assert payload["summary"]["active_shadow_case_index_status"] == (
        "prediction_regime_specialist_synthesis_exact_case_producers_"
        "ready_no_real_cases"
    )
    assert payload["summary"]["active_shadow_common_context_status"] == (
        "diagnostic_counts_must_intersect_on_one_exact_context"
    )
    assert payload["summary"]["active_shadow_diagnostics_status"] == (
        "deterministic_engine_ready_currently_blocked_zero_aligned_"
        "episodes"
    )
    assert payload["summary"][
        "active_shadow_consensus_weight_status"
    ] == "ineligible_no_automatic_weight_change"
    assert payload["summary"]["active_analyst_knowledge_status"] == (
        "strict_point_in_time_provenance_contract_ready_store_empty"
    )
    assert payload["summary"]["active_analyst_knowledge_pipeline_status"] == (
        "review_only_no_stage5_or_consensus_influence"
    )
    assert payload["summary"]["active_context_evidence_status"] == (
        "pipeline_news_quarantined_by_as_of_timestamp_locator_and_"
        "duplicate_contract"
    )
    assert payload["summary"]["active_context_ticker_directness_status"] == (
        "explicit_ticker_metadata_or_cashtag_only_no_plain_text_promotion"
    )
    assert payload["summary"]["active_context_evidence_review_status"] == (
        "review_packet_ready_no_saved_real_context_packet"
    )
    assert payload["summary"]["active_context_direct_agent_status"] == (
        "keyword_and_material_news_paths_share_point_in_time_quarantine"
    )
    assert payload["summary"]["active_agent_lab_as_of_status"] == (
        "explicit_live_or_replay_cutoff_propagated"
    )
    assert payload["summary"][
        "active_research_document_point_in_time_status"
    ] == (
        "publication_ingestion_locator_content_hash_and_replay_basis_"
        "audited"
    )
    assert payload["summary"]["active_structured_context_status"] == (
        "fundamental_macro_sector_observations_require_value_unit_period_"
        "availability_and_locator"
    )
    assert payload["summary"]["active_raw_macro_boundary_status"] == (
        "pipeline_macro_dataframe_inventory_is_not_structured_macro_"
        "evidence"
    )
    assert payload["summary"][
        "active_fundamental_gate_binding_status"
    ] == (
        "gate_and_context_accepted_fingerprints_must_match_before_value_"
        "screening"
    )
    assert payload["summary"]["active_package_import_status"] == (
        "lazy_public_api_no_pipeline_boot_for_small_module_imports"
    )
    assert payload["summary"]["active_saved_macro_producer_status"] == (
        "real_snapshot_470_rows_454_point_in_time_eligible_27_series_ready"
    )
    assert payload["summary"]["active_macro_vintage_status"] == (
        "fred_realtime_start_used_as_conservative_snapshot_availability_"
        "not_claimed_release_time"
    )
    assert payload["summary"]["active_macro_registry_status"] == (
        "twenty_seven_series_mapped_operator_confirmation_pending"
    )
    assert payload["summary"]["active_agent_lab_macro_status"] == (
        "source_registry_as_of_and_fragment_fingerprint_reverified_before_"
        "review"
    )
    assert payload["summary"]["active_real_macro_agent_smoke_status"] == (
        "verified_twenty_seven_series_macro_policy_neutral_no_learning_no_"
        "proposals"
    )
    assert payload["summary"]["active_macro_directionality_status"] == (
        "series_presence_never_implies_policy_easing"
    )
    assert payload["summary"]["active_sec_filing_index_status"] == (
        "duckdb_10191_rows_amd_10q_hash_time_and_locator_verified"
    )
    assert payload["summary"]["active_fundamental_fact_status"] == (
        "twenty_nine_accession_bound_facts_four_tickers_companyfacts_and_"
        "inline_xbrl"
    )
    assert payload["summary"][
        "active_fundamental_sector_coverage_status"
    ] == (
        "four_of_four_source_coverage_raw_period_and_currency_comparison_"
        "blocked"
    )
    assert payload["summary"][
        "active_semiconductor_pipeline_universe_status"
    ] == "nvda_amd_intc_tsm_four_ticker_pipeline_cohort"
    assert payload["summary"][
        "active_semiconductor_research_universe_status"
    ] == (
        "twelve_ticker_value_chain_hint_not_automatic_pipeline_scope"
    )
    assert payload["summary"][
        "active_semiconductor_filing_coverage_status"
    ] == (
        "four_of_four_periodic_sources_after_nvda_submissions_recovery"
    )
    assert payload["summary"]["active_sector_market_evidence_status"] == (
        "four_of_four_tickers_plus_qqq_twenty_two_common_sessions_market_"
        "confirmation_ready"
    )
    assert payload["summary"]["active_semiconductor_runtime_status"] == (
        "verified_fundamental_macro_market_news_policy_vertical_slice_partial_"
        "ready_for_review_sector_only_five_of_five_lanes"
    )
    assert payload["summary"][
        "active_semiconductor_thesis_review_status"
    ] == (
        "runtime_linked_hash_verified_sector_review_ready_with_three_"
        "explicit_cautions"
    )
    assert payload["summary"][
        "active_semiconductor_ticker_thesis_status"
    ] == "zero_direct_ticker_theses_four_basket_candidates"
    assert payload["summary"][
        "active_semiconductor_prospective_case_status"
    ] == (
        "one_pre_outcome_sector_case_registered_for_30_90_180_day_review"
    )
    assert payload["summary"][
        "active_sector_to_ticker_pipeline_bridge_status"
    ] == (
        "amd_ticker_evidence_ready_but_pipeline_blocked_three_missing_"
        "ticker_evidence_zero_forecasts"
    )
    assert payload["summary"][
        "active_ticker_specific_evidence_status"
    ] == (
        "forty_nine_company_candidates_six_strong_amd_one_corroborated_"
        "demand_lane"
    )
    assert payload["summary"][
        "active_stage5_sector_context_overlay_status"
    ] == (
        "exact_ticker_match_attached_zero_directional_or_lineage_influence"
    )
    assert payload["summary"][
        "active_sector_to_ticker_review_status"
    ] == (
        "review_ready_with_limitations_readiness_gap_map_only"
    )
    assert payload["summary"]["active_semiconductor_news_status"] == (
        "18813_rows_9604_usable_9209_orphan_excluded_63_candidates_demand_"
        "capex_supply_ready"
    )
    assert payload["summary"]["active_sec_derived_ratio_status"] == (
        "twenty_one_formula_bound_ratios_five_multi_ticker_lanes_zero_full_"
        "cohort_lanes"
    )
    assert payload["summary"]["active_official_policy_status"] == (
        "bis_may_2026_pdf_hash_bound_bloomberg_corroborated_policy_lane_"
        "ready"
    )
    assert payload["summary"]["active_amd_role_status"] == (
        "single_ticker_single_target_smoke_and_negative_model_case_only"
    )
    assert payload["summary"]["active_tuning_exact_scope_status"] == (
        "one_failure_can_tune_only_matching_ticker_model_target_timeframe_"
        "context"
    )
    assert payload["summary"]["active_pipeline_model_case_scope"] == (
        "ticker_model_evaluation_only_not_domain_evidence"
    )
    assert payload["summary"]["active_model_performance_source_status"] == "canonical_evaluation_summary_only_complete_metric_set_required"
    assert payload["summary"]["active_locked_evidence_inventory_status"] == "verified_pair_available_runner_still_required"
    assert payload["summary"]["active_real_metric_evidence_status"] == "blocked_validation_and_feature_stability"
    assert payload["summary"]["active_model_performance_chain_status"] == "locked_artifact_bound_to_full_evidence_chain"
    assert payload["summary"]["active_dean_orchestrator_status"] == "preflight_pipeline_post_pipeline_pretrade_review"
    assert payload["summary"]["active_consensus_status"] == "watchlist_or_blocked_no_execution_candidates"
    assert payload["summary"]["hard_prerequisite_status"] == "synthetic_block_reports_are_enforced"
    assert payload["summary"]["active_pipeline_adapter_status"] == "canonical_dean_review_contract_attached"
    assert payload["summary"]["active_risk_returns_status"] == "realized_returns_preferred_target_labels_blocked_pretrade"
    assert payload["summary"]["can_execute_paper_simulation_now"] is False

    branches = {branch["branch_id"] for branch in payload["branch_map"]}
    assert "pipeline_metric_control" in branches
    assert "domain_analyst_research" in branches
    assert "review_orchestration" in branches
    review_orchestration = next(branch for branch in payload["branch_map"] if branch["branch_id"] == "review_orchestration")
    assert "dean_os/review_only_automation_run.py" in review_orchestration["current_modules"]
    assert "dean_os/paper_lifecycle_contract.py" in review_orchestration["current_modules"]
    assert "dean_os/paper_simulation_plan.py" in review_orchestration["current_modules"]
    assert "dean_os/paper_simulation_result.py" in review_orchestration["current_modules"]
    assert "dean_os/post_paper_simulation_review.py" in review_orchestration["current_modules"]

    planes = {plane["plane_id"] for plane in payload["pipeline_metric_control_branch"]["metric_planes"]}
    assert {"profitability", "risk", "validation_split", "data_quality_leakage", "replay_repeatability"}.issubset(planes)
    assert "dean_os/pipeline_metric_input_readiness_gate.py" in payload["pipeline_metric_control_branch"]["primary_modules"]
    assert "dean_os/pipeline_control_instance_contract.py" in payload["pipeline_metric_control_branch"]["primary_modules"]
    assert "dean_os/pipeline_control_caution_review_packet.py" in payload["pipeline_metric_control_branch"]["primary_modules"]
    assert "dean_os/pipeline_control_evidence_inventory.py" in payload["pipeline_metric_control_branch"]["primary_modules"]
    assert "dean_os/pipeline_control_metric_artifact_materializer.py" in payload["pipeline_metric_control_branch"]["primary_modules"]
    assert "dean_os/pipeline_control_locked_evaluation_assembler.py" in payload["pipeline_metric_control_branch"]["primary_modules"]
    assert "dean_os/pipeline_control_locked_feature_stability_assembler.py" in payload["pipeline_metric_control_branch"]["primary_modules"]
    assert "dean_os/pipeline_control_bounded_evidence_run.py" in payload["pipeline_metric_control_branch"]["primary_modules"]
    assert "dean_os/pipeline_control_walk_forward_validation_run.py" in payload["pipeline_metric_control_branch"]["primary_modules"]
    assert "dean_os/pipeline_control_forward_data_accrual_plan.py" in payload["pipeline_metric_control_branch"]["primary_modules"]
    assert "dean_os/pipeline_control_forward_data_accrual_gate.py" in payload["pipeline_metric_control_branch"]["primary_modules"]
    assert "src/pipeline/stages/feature_engineering/timeframe_context.py" in payload["pipeline_metric_control_branch"]["primary_modules"]
    assert "src/pipeline/stages/modeling/walk_forward_validation.py" in payload["pipeline_metric_control_branch"]["primary_modules"]
    assert "src/pipeline/stages/stage_4_modeling.py" in payload["pipeline_metric_control_branch"]["primary_modules"]
    assert "src/pipeline/stages/stage_5_prediction.py" in payload["pipeline_metric_control_branch"]["primary_modules"]
    assert "src/pipeline/stages/prediction/output_contract.py" in payload["pipeline_metric_control_branch"]["primary_modules"]
    assert "dean_os/shadow_calibration_case_index.py" in payload["pipeline_metric_control_branch"]["primary_modules"]
    assert "dean_os/shadow_component_case_producer.py" in payload["pipeline_metric_control_branch"]["primary_modules"]
    assert "dean_os/shadow_calibration_diagnostics.py" in payload["pipeline_metric_control_branch"]["primary_modules"]
    assert "src/pipeline/stages/stage_6_trading_execution.py" in payload["pipeline_metric_control_branch"]["primary_modules"]
    assert "src/pipeline/hybrid/final_stages_orchestrator.py" in payload["pipeline_metric_control_branch"]["primary_modules"]
    assert "src/analytics/unified_analytics_engine.py" in payload["pipeline_metric_control_branch"]["primary_modules"]
    assert "src/config/analysis.yaml" in payload["pipeline_metric_control_branch"]["primary_modules"]
    assert "src/training/base_trainer.py" in payload["pipeline_metric_control_branch"]["primary_modules"]
    assert "src/targets/timeframe_contract.py" in payload["pipeline_metric_control_branch"]["primary_modules"]
    assert "src/pipeline/modeling_context.py" in payload["pipeline_metric_control_branch"]["primary_modules"]
    assert "src/pipeline/stages/modeling/pipeline_control_artifacts.py" in payload["pipeline_metric_control_branch"]["primary_modules"]
    assert "src/pipeline/stages/evaluation/pipeline_control_artifacts.py" in payload["pipeline_metric_control_branch"]["primary_modules"]
    assert "dean_os/pipeline_control_metric_fixture_validation.py" in payload["pipeline_metric_control_branch"]["primary_modules"]
    assert "dean_os/pipeline_control_real_metric_evidence_run.py" in payload["pipeline_metric_control_branch"]["primary_modules"]
    assert "src/pipeline/timeframe_lineage.py" in payload["pipeline_metric_control_branch"]["primary_modules"]
    assert "dean_os/pipeline_feature_timeframe_audit.py" in payload["pipeline_metric_control_branch"]["primary_modules"]
    assert "dean_os/pipeline_stage4_exact_context_review.py" in payload[
        "pipeline_metric_control_branch"
    ]["primary_modules"]
    assert "dean_os/agents/pipeline_manager.py" in payload["domain_analyst_branch"]["primary_modules"]
    assert "dean_os/agents/domain_analyst.py" in payload["domain_analyst_branch"]["primary_modules"]
    assert "dean_os/agents/pipeline_readiness.py" in payload["domain_analyst_branch"]["primary_modules"]
    assert payload["domain_analyst_branch"]["execution_topology"]["conflict_rule"].startswith(
        "Never enable composite and standalone"
    )
    assert "dean_os/domain_analyst_profile_policy_packet.py" in payload["domain_analyst_branch"]["primary_modules"]
    assert "dean_os/domain_analyst_pipeline_news_taxonomy.py" in payload["domain_analyst_branch"]["primary_modules"]
    assert "dean_os/domain_analyst_event_interpretation_packet.py" in payload["domain_analyst_branch"]["primary_modules"]
    assert "dean_os/domain_analyst_regime_scenario_packet.py" in payload["domain_analyst_branch"]["primary_modules"]
    assert "dean_os/domain_analyst_thesis_review_packet.py" in payload["domain_analyst_branch"]["primary_modules"]
    assert "dean_os/domain_analyst_forecast_review_packet.py" in payload["domain_analyst_branch"]["primary_modules"]
    assert "dean_os/domain_analyst_template_standardization_packet.py" in payload["domain_analyst_branch"]["primary_modules"]
    assert "dean_os/domain_analyst_template_decision_packet.py" in payload["domain_analyst_branch"]["primary_modules"]
    assert "dean_os/domain_analyst_case_registry_packet.py" in payload["domain_analyst_branch"]["primary_modules"]
    assert "dean_os/domain_analyst_feedback_loop_packet.py" in payload["domain_analyst_branch"]["primary_modules"]
    assert "dean_os/analyst_knowledge/store.py" in payload["domain_analyst_branch"]["primary_modules"]
    assert "dean_os/analyst_knowledge/retriever.py" in payload["domain_analyst_branch"]["primary_modules"]
    assert "dean_os/analyst_knowledge_readiness.py" in payload["domain_analyst_branch"]["primary_modules"]
    assert "dean_os/context_evidence_provenance.py" in payload["domain_analyst_branch"]["primary_modules"]
    assert "dean_os/structured_context_provenance.py" in payload["domain_analyst_branch"]["primary_modules"]
    assert "dean_os/context_evidence_review_packet.py" in payload["domain_analyst_branch"]["primary_modules"]
    assert "dean_os/analysts/context_adapter.py" in payload["domain_analyst_branch"]["primary_modules"]
    assert "dean_os/agents/domain_research.py" in payload["domain_analyst_branch"]["primary_modules"]
    assert "dean_os/agents/research_agents.py" in payload["domain_analyst_branch"]["primary_modules"]
    assert "dean_os/agent_lab.py" in payload["domain_analyst_branch"]["primary_modules"]
    assert "dean_os/fundamental_input_readiness_gate.py" in payload["domain_analyst_branch"]["primary_modules"]
    assert "dean_os/saved_macro_evidence_producer.py" in payload["domain_analyst_branch"]["primary_modules"]
    assert "dean_os/saved_sec_filing_index_producer.py" in payload["domain_analyst_branch"]["primary_modules"]
    assert "dean_os/saved_sec_submissions_filing_index_producer.py" in payload["domain_analyst_branch"]["primary_modules"]
    assert "dean_os/saved_sec_companyfacts_producer.py" in payload["domain_analyst_branch"]["primary_modules"]
    assert "dean_os/saved_sec_inline_xbrl_producer.py" in payload["domain_analyst_branch"]["primary_modules"]
    assert "dean_os/saved_sec_fundamental_evidence_merger.py" in payload["domain_analyst_branch"]["primary_modules"]
    assert "dean_os/saved_sec_derived_ratio_producer.py" in payload["domain_analyst_branch"]["primary_modules"]
    assert "dean_os/saved_sector_market_evidence_producer.py" in payload["domain_analyst_branch"]["primary_modules"]
    assert "dean_os/saved_semiconductor_news_evidence_producer.py" in payload["domain_analyst_branch"]["primary_modules"]
    assert "dean_os/semiconductor_analyst_runtime.py" in payload["domain_analyst_branch"]["primary_modules"]
    assert "dean_os/config/macro_series_registry.yaml" in payload["domain_analyst_branch"]["primary_modules"]
    assert "dean_os/JULY_2026_BUILD_ROADMAP.md" in payload["domain_analyst_branch"]["primary_modules"]
    assert "dean_os/__init__.py" in payload["domain_analyst_branch"]["primary_modules"]
    analyst_planes = {plane["plane_id"] for plane in payload["domain_analyst_branch"]["analyst_control_planes"]}
    assert {"news_event_interpretation", "regime_scenario_context", "causal_attribution", "luck_vs_skill", "feedback_to_learning_candidate", "learning_promotion_readiness"}.issubset(analyst_planes)


def test_current_architecture_map_records_plan_corrections_and_domain_rules(tmp_path):
    payload = CurrentArchitectureMap(tmp_path / "reports").build(save=False)

    corrections = " ".join(payload["corrections_to_user_plan"])
    assert "single automatic optimum" in corrections
    assert "sector/domain thesis first" in corrections
    assert payload["domain_analyst_branch"]["ticker_rule"].startswith("Domain thesis can create exposure candidates")
    assert payload["domain_analyst_branch"]["recommendation_rule"].startswith("Review-only analyst recommendations are allowed")
    assert payload["domain_analyst_branch"]["data_analysis_rule"].startswith("Detailed news/data analysis is allowed")
    assert any(profile["domain_id"] == "semiconductor_ai_infrastructure" for profile in payload["domain_analyst_branch"]["current_profiles"])
    semiconductor = next(profile for profile in payload["domain_analyst_branch"]["current_profiles"] if profile["domain_id"] == "semiconductor_ai_infrastructure")
    assert semiconductor["source_registry_policy_id"] == "default_domain_source_registry_policy_v1"
    assert semiconductor["evidence_scoring_policy_id"] == "default_domain_evidence_scoring_policy_v1"
    assert semiconductor["feedback_label_policy_id"] == "default_domain_feedback_label_policy_v1"
    assert "dean_os/domain_analyst_intake_packet.py" in payload["domain_analyst_branch"]["primary_modules"]
    assert "dean_os/domain_analyst_instance_contract.py" in payload["domain_analyst_branch"]["primary_modules"]
    assert "DomainAnalystIntakePacket" in payload["existing_module_map"]["already_useful"]
    assert "DomainAnalystInstanceContract" in payload["existing_module_map"]["already_useful"]
    assert "DomainAnalystPipelineNewsTaxonomy" in payload["existing_module_map"]["already_useful"]
    assert "DomainAnalystEventInterpretationPacket" in payload["existing_module_map"]["already_useful"]
    assert "DomainAnalystRegimeScenarioPacket" in payload["existing_module_map"]["already_useful"]
    assert "DomainAnalystThesisReviewPacket" in payload["existing_module_map"]["already_useful"]
    assert "DomainAnalystForecastReviewPacket" in payload["existing_module_map"]["already_useful"]
    assert "DomainAnalystTemplateStandardizationPacket" in payload["existing_module_map"]["already_useful"]
    assert "DomainAnalystTemplateDecisionPacket" in payload["existing_module_map"]["already_useful"]
    assert "DomainAnalystProfilePolicyPacket" in payload["existing_module_map"]["already_useful"]
    assert "DomainAnalystCaseRegistryPacket" in payload["existing_module_map"]["already_useful"]
    assert "DomainAnalystFeedbackLoopPacket" in payload["existing_module_map"]["already_useful"]
    assert "AnalystKnowledgePointInTimeRetrieval" in payload["existing_module_map"]["already_useful"]
    assert "AnalystKnowledgeReadiness" in payload["existing_module_map"]["already_useful"]
    assert "HashBoundIsolatedPaperLifecycle" in payload["existing_module_map"]["already_useful"]
    assert "ContextEvidencePointInTimeBoundary" in payload["existing_module_map"]["already_useful"]
    assert "ContextEvidenceReviewPacket" in payload["existing_module_map"]["already_useful"]
    assert "BuildFocusReviewPacket" in payload["existing_module_map"]["already_useful"]
    assert payload["build_focus_control"]["current_tool"] == "BuildFocusReviewPacket"
    assert "SourceEvidenceValidationGate" in payload["existing_module_map"]["already_useful"]
    assert "PipelineMetricInputReadinessGate" in payload["existing_module_map"]["already_useful"]
    assert "PipelineControlInstanceContract" in payload["existing_module_map"]["already_useful"]
    assert "PipelineControlCautionReviewPacket" in payload["existing_module_map"]["already_useful"]
    assert "PipelineControlEvidenceInventory" in payload["existing_module_map"]["already_useful"]
    assert "PipelineControlMetricArtifactMaterializer" in payload["existing_module_map"]["already_useful"]
    assert "PipelineControlLockedEvaluationAssembler" in payload["existing_module_map"]["already_useful"]
    assert "PipelineControlLockedFeatureStabilityAssembler" in payload["existing_module_map"]["already_useful"]
    assert "PipelineControlBoundedEvidenceRun" in payload["existing_module_map"]["already_useful"]
    assert "PipelineControlWalkForwardValidationRun" in payload["existing_module_map"]["already_useful"]
    assert "PipelineControlForwardDataAccrualPlan" in payload["existing_module_map"]["already_useful"]
    assert "PipelineControlForwardDataAccrualGate" in payload["existing_module_map"]["already_useful"]
    assert "PipelineWalkForwardValidationEvaluator" in payload["existing_module_map"]["already_useful"]
    assert "PipelineStage4ExactContextReview" in payload[
        "existing_module_map"
    ]["already_useful"]
    assert "ActiveStage4ValidationOnlyTrainingAdapter" in payload["existing_module_map"]["already_useful"]
    assert "Stage5ModelLineagePropagation" in payload["existing_module_map"]["already_useful"]
    assert "ActiveStage6ReviewOnlyExecutionBoundary" in payload["existing_module_map"]["already_useful"]
    assert "Stage5ToStage7DefaultFinalOrchestration" in payload["existing_module_map"]["already_useful"]
    assert "LiveTraderInitializationBlock" in payload["existing_module_map"]["already_useful"]
    assert "Stage7ProposalOnlyLearningBoundary" in payload["existing_module_map"]["already_useful"]
    assert "Stage7ExplicitNotificationAuthorization" in payload["existing_module_map"]["already_useful"]
    assert "Stage7ContextPartitionedAnalyzerReview" in payload["existing_module_map"]["already_useful"]
    assert "UnifiedAnalyticsCoverageContract" in payload["existing_module_map"]["already_useful"]
    assert "CanonicalStage7AnalyzerConfig" in payload["existing_module_map"]["already_useful"]
    assert "Stage7AnalyzerReviewContractBridge" in payload["existing_module_map"]["already_useful"]
    assert "CanonicalModelPerformanceMetricExtraction" in payload["existing_module_map"]["already_useful"]
    assert "LockedEvidenceProvenanceVerification" in payload["existing_module_map"]["already_useful"]
    assert "ModelPerformanceEvidenceChainBinding" in payload["existing_module_map"]["already_useful"]
    assert "DEANOrchestratorTwoPhaseSafetyReview" in payload["existing_module_map"]["already_useful"]
    assert "ConsensusWatchlistOnlyDefault" in payload["existing_module_map"]["already_useful"]
    assert "SyntheticHardPrerequisiteBlock" in payload["existing_module_map"]["already_useful"]
    assert "CanonicalPipelineReviewContractAdapter" in payload["existing_module_map"]["already_useful"]
    assert "OfflineTargetReturnRiskBlock" in payload["existing_module_map"]["already_useful"]
    assert "PipelineTrainingMetricArtifactCandidates" in payload["existing_module_map"]["already_useful"]
    assert "PipelineEvaluationMetricArtifactCandidates" in payload["existing_module_map"]["already_useful"]
    assert "PipelineControlMetricFixtureValidation" in payload["existing_module_map"]["already_useful"]
    assert "PipelineControlRealMetricEvidenceRun" in payload["existing_module_map"]["already_useful"]
    assert "DeanOSReviewOnlyAutomationRun" in payload["existing_module_map"]["already_useful"]
    commands = {item["command_id"]: item["command"] for item in payload["commands"]}
    assert "run_agent_analyst_knowledge_readiness.py" in commands["analyst_knowledge_readiness"]
    assert "--as-of" in commands["analyst_knowledge_readiness"]
    assert "run_agent_paper_simulation_plan.py" in commands["paper_simulation_plan"]
    assert "run_agent_paper_simulation_result.py" in commands["paper_simulation_result"]
    assert "run_agent_post_paper_simulation_review.py" in commands["post_paper_simulation_review"]
    assert "run_agent_context_evidence_review.py" in commands["context_evidence_review"]
    assert "--as-of" in commands["context_evidence_review"]
    assert "run_agent_domain_analyst_profile_policy_packet.py" in commands["domain_analyst_profile_policy_packet"]
    assert "run_agent_domain_analyst_event_interpretation_packet.py" in commands["domain_analyst_event_interpretation_packet"]
    assert "run_agent_domain_analyst_regime_scenario_packet.py" in commands["domain_analyst_regime_scenario_packet"]
    assert "run_agent_semiconductor_analyst.py" in commands["semiconductor_analyst_runtime"]
    assert "semiconductor_analyst_runtime_current" in commands["domain_analyst_thesis_review_packet"]
    assert "--regime-scenario-json" not in commands["domain_analyst_thesis_review_packet"]
    assert "--regime-scenario-json" in commands["domain_analyst_forecast_review_packet"]
    assert "--domain-template-standardization-json" in commands[
        "domain_analyst_case_registry_packet"
    ]
    assert "run_agent_analyst_core_reasoning_snapshot.py" in commands[
        "analyst_core_reasoning_snapshot"
    ]
    assert "--reasoning-snapshot-json" in commands[
        "domain_analyst_thesis_review_packet"
    ]
    assert "--domain-forecast-review-json" not in commands["domain_analyst_case_registry_packet"]
    assert "--domain-thesis-review-json" in commands[
        "sector_to_exact_ticker_pipeline_bridge"
    ]
    assert "run_agent_saved_ticker_specific_evidence.py" in commands[
        "saved_ticker_specific_evidence"
    ]
    assert "--ticker-evidence-json" in commands[
        "sector_to_exact_ticker_pipeline_bridge"
    ]
    assert "--pipeline-case-json" in commands[
        "sector_to_exact_ticker_pipeline_bridge"
    ]
    assert "--prediction-review-json" in commands[
        "sector_to_exact_ticker_pipeline_bridge"
    ]
    assert "--feature-timeframe-audit-json" in commands[
        "sector_to_exact_ticker_pipeline_bridge"
    ]
    assert "run_agent_pipeline_feature_timeframe_audit.py" in commands[
        "pipeline_feature_timeframe_audit"
    ]
    assert "run_agent_composite_domain_pipeline.py" in commands[
        "composite_semiconductor_domain_pipeline"
    ]
    assert "run_agent_pipeline_stage23_regeneration.py" in commands[
        "bounded_pipeline_stage23_regeneration"
    ]
    assert "run_agent_pipeline_target_readiness_audit.py" in commands[
        "pipeline_target_readiness_audit"
    ]
    assert "run_agent_pipeline_stage4_exact_context_review.py" in commands[
        "pipeline_stage4_exact_context_review"
    ]
    assert "--filter-to-requested-scope" in commands[
        "pipeline_prediction_source_review"
    ]
    assert "run_agent_sector_to_ticker_review_packet.py" in commands[
        "sector_to_ticker_readiness_review"
    ]
    assert "--sector-to-ticker-review-json" in commands[
        "pipeline_prediction_review_packet"
    ]
    assert "run_agent_domain_analyst_feedback_loop_packet.py" in commands["domain_analyst_feedback_loop_packet"]
    assert "--decision pending_review" in commands["domain_analyst_template_decision_packet"]
    assert "run_agent_pipeline_control_evidence_inventory.py" in commands["pipeline_control_evidence_inventory"]
    assert "run_agent_pipeline_control_metric_artifact_materializer.py" in commands["pipeline_control_metric_artifact_materializer"]
    assert "run_agent_pipeline_control_locked_evaluation_assembler.py" in commands["pipeline_control_locked_evaluation_assembler"]
    assert "run_agent_pipeline_control_locked_feature_stability_assembler.py" in commands["pipeline_control_locked_feature_stability_assembler"]
    assert "run_agent_pipeline_control_bounded_evidence.py" in commands["pipeline_control_bounded_evidence_run"]
    assert "run_agent_pipeline_control_walk_forward_validation.py" in commands["pipeline_control_walk_forward_validation"]
    assert "--acknowledge-development-only" in commands["pipeline_control_walk_forward_validation"]
    assert "run_agent_pipeline_control_forward_data_accrual_plan.py" in commands["pipeline_control_forward_data_accrual_plan"]
    assert "--acknowledge-development-refresh-only" in commands["pipeline_control_forward_data_accrual_plan"]
    assert "run_agent_pipeline_control_forward_data_accrual_gate.py" in commands["pipeline_control_forward_data_accrual_gate"]
    assert "<new_immutable_price_artifact.parquet>" in commands["pipeline_control_forward_data_accrual_gate"]
    assert "--forward-accrual-gate-json" in commands["pipeline_control_walk_forward_forward_refresh"]
    assert "run_agent_dean_os_review_automation.py" in commands["dean_os_review_automation"]


def test_current_architecture_map_saves_markdown_and_cli_runs(tmp_path):
    payload = CurrentArchitectureMap(tmp_path / "reports").build()
    markdown = (tmp_path / "reports" / "latest.md").read_text(encoding="utf-8")

    assert "Pipeline Metric Planes" in markdown
    assert "Can trade: False" in markdown
    assert payload["saved_paths"]["latest_json"].endswith("latest.json")

    repo_root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        [
            sys.executable,
            str(repo_root / "run_agent_current_architecture_map.py"),
            "--output-dir",
            str(tmp_path / "cli_reports"),
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=True,
    )

    assert "Architecture: current_architecture_map_ready" in result.stdout
    assert "Can trade: False" in result.stdout
    assert (tmp_path / "cli_reports" / "latest.json").exists()
