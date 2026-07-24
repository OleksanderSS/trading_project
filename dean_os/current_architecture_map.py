from __future__ import annotations

from pathlib import Path
from typing import Any

from dean_os.analysts.profiles import get_domain_profile, list_domain_profiles
from dean_os.artifact_writer import ReviewArtifactWriter
from dean_os.pipeline_control.pipeline_control_surface import DEFAULT_CONSTRAINTS
from dean_os.schemas import utc_now_iso
from dean_os.utils import json_ready

ARCHITECTURE_VERSION = "2026-07-09-parallel-scaffold-safety-v12"


class CurrentArchitectureMap:
    """Review-only map of the active DEAN-OS two-branch architecture."""

    def __init__(self, output_dir: str | Path = "reports/dean_os/current_architecture_map"):
        self.output_dir = Path(output_dir)

    def build(self, save: bool = True) -> dict[str, Any]:
        domain_profiles = _domain_profiles()
        metric_planes = _pipeline_metric_planes()
        analyst_planes = _domain_analyst_control_planes()
        payload = {
            "run_id": _run_id("current_architecture_map"),
            "created_at": utc_now_iso(),
            "mode": "current_architecture_map",
            "architecture_version": ARCHITECTURE_VERSION,
            "summary": {
                "architecture_status": "current_architecture_map_ready",
                "active_design": "source_first_two_branch_review_system",
                "branch_count": 4,
                "pipeline_metric_plane_count": len(metric_planes),
                "domain_analyst_control_plane_count": len(analyst_planes),
                "domain_profile_count": len(domain_profiles),
                "recommended_action": (
                    "keep_stage5_blocked_register_new_forward_data_and_"
                    "add_stage3_ticker_shard_cache_before_any_model_variant"
                ),
                "pipeline_timeframe_context_status": "fail_closed_declared_observed_cadence_and_timezone_gated",
                "pipeline_feature_timeframe_audit_status": "legacy_four_of_four_blocked_regenerated_four_of_four_15m_utc_ready",
                "active_bounded_stage23_status": "four_ticker_1170_row_15m_utc_hash_bound_review_ready",
                "active_pipeline_target_readiness_status": "seven_of_seven_semantic_targets_ready_for_bounded_stage4",
                "active_exact_stage4_review_status": "nvda_15m_587_rows_three_folds_hash_bound_validation_contract_blocked",
                "active_exact_stage4_failed_checks": "train_validation_gap_positive_rate_stability_and_majority_baseline",
                "active_stage3_resource_status": "single_ticker_bounded_run_succeeds_four_ticker_600_row_run_exceeds_five_minutes_shard_cache_needed",
                "active_colab_identity_status": "ticker_datetime_interval_dedup_and_hash_lineage_enforced",
                "active_stage1_source_cadence_status": "saved_15m_usable_saved_60m_and_1d_labels_rejected",
                "active_composite_domain_pipeline_status": "real_152_item_semiconductor_smoke_caution_readiness_blocked_no_decision_influence",
                "active_domain_agent_topology_status": "composite_manager_canonical_standalone_analyst_alternative",
                "active_agent_execution_group_status": "overlapping_enabled_domain_agents_fail_closed",
                "active_agent_phase_status": "registry_run_phases_enforced_before_instantiation",
                "active_scaffold_audit_status": "parallel_scaffolds_retained_stateful_and_domain_fanout_default_off",
                "active_registry_activation_status": "thirty_seven_registered_sixteen_review_only_enabled_expensive_domain_composite_and_mutating_agents_default_off",
                "active_standalone_domain_input_status": "timezone_aware_as_of_plus_populated_context_or_verified_runtime_required",
                "active_runtime_cutoff_status": "requested_as_of_must_equal_verified_runtime_cutoff",
                "active_domain_clone_status": "private_profile_bound_to_wrapper_adapter_base_agent_and_lens_config",
                "pipeline_walk_forward_status": "three_fold_development_candidate_predictive_and_stability_checks_passed_but_overfit_and_class_balance_checks_blocked",
                "pipeline_forward_accrual_status": "prospective_development_boundary_registered",
                "pipeline_forward_accrual_gate_status": "existing_pre_registration_artifact_blocked",
                "pipeline_forward_runner_status": "accepts_passing_accrual_gate_only",
                "active_stage4_training_contract_status": "nested_split_adapter_fixed_validation_only_selection",
                "active_stage4_evidence_status": "partial_training_candidates_enabled",
                "active_stage5_lineage_status": "target_model_timeframe_context_propagated",
                "active_stage5_prediction_review_status": "per_context_output_contract_required_sector_ticker_context_supporting_only",
                "active_stage5_sector_context_overlay_status": "exact_ticker_match_attached_zero_directional_or_lineage_influence",
                "active_stage5_prediction_artifact_status": "no_trustworthy_saved_current_result_do_not_fabricate",
                "active_final_pipeline_status": "stage5_to_stage7_review_default_stage6_explicit",
                "active_stage6_execution_status": "review_only_no_paper_no_live_no_memory_write",
                "active_paper_lifecycle_status": "hash_bound_receipt_plan_external_result_review_no_executor_run",
                "active_paper_lineage_status": "post_dry_review_receipt_plan_external_manifest_result_review_bound",
                "active_stage7_learning_status": "proposal_only_no_automatic_adaptation",
                "active_stage7_notification_status": "explicit_per_run_authorization_required",
                "active_stage7_analyzer_status": "context_partitioned_supporting_review_only",
                "active_analyzer_suite_status": "two_enabled_ten_explicitly_staged",
                "active_analyzer_observability_status": "executed_skipped_failed_disabled_coverage_recorded",
                "active_analyzer_cache_status": "data_and_suite_contract_fingerprinted",
                "active_stage7_agent_bridge_status": "per_context_regime_review_consumed_by_shadow_regime_agent",
                "active_regime_agent_status": "enabled_pretrade_stage7_only_shadow_no_decision_influence",
                "active_agent_capability_matrix_status": "registry_agents_contract_mapped_parallel_scaffold_needs_matrix_refresh",
                "active_context_synthesis_status": "stage5_stage7_exact_context_shadow_compatibility_only",
                "active_context_freshness_status": "stage7_price_window_provenance_and_as_of_skew_checked",
                "active_specialist_context_status": "sector_direct_ticker_point_in_time_scopes_separated",
                "active_amd_specialist_context_status": "direct_ticker_manual_review_candidate_stale_unaligned_not_approved",
                "active_semiconductor_amd_boundary": "semiconductor_domain_context_is_not_amd_ticker_evidence",
                "parallel_template_audit_status": "source_folder_empty_after_transfer_active_tree_contains_workbench",
                "transferred_workbench_status": "runtime_foundation_integrated_legacy_islands_classified",
                "template_harvest_status": "eval_unit_period_time_leakage_and_safety_rules_adapted",
                "active_prediction_target_semantics_status": "canonical_target_period_unit_threshold_positive_class_bound",
                "active_stage5_output_scale_status": "explicit_predict_output_contract_directional_use_still_blocked",
                "active_shadow_calibration_status": "blocked_zero_of_thirty_cases_all_components",
                "active_shadow_case_index_status": "prediction_regime_specialist_synthesis_exact_case_producers_ready_no_real_cases",
                "active_shadow_common_context_status": "diagnostic_counts_must_intersect_on_one_exact_context",
                "active_shadow_diagnostics_status": "deterministic_engine_ready_currently_blocked_zero_aligned_episodes",
                "active_shadow_consensus_weight_status": "ineligible_no_automatic_weight_change",
                "active_analyst_knowledge_status": "strict_point_in_time_provenance_contract_ready_store_empty",
                "active_analyst_knowledge_pipeline_status": "review_only_no_stage5_or_consensus_influence",
                "active_context_evidence_status": "pipeline_news_quarantined_by_as_of_timestamp_locator_and_duplicate_contract",
                "active_context_ticker_directness_status": "explicit_ticker_metadata_or_cashtag_only_no_plain_text_promotion",
                "active_context_evidence_review_status": "review_packet_ready_no_saved_real_context_packet",
                "active_context_direct_agent_status": "keyword_and_material_news_paths_share_point_in_time_quarantine",
                "active_agent_lab_as_of_status": "explicit_live_or_replay_cutoff_propagated",
                "active_research_document_point_in_time_status": "publication_ingestion_locator_content_hash_and_replay_basis_audited",
                "active_structured_context_status": "fundamental_macro_sector_observations_require_value_unit_period_availability_and_locator",
                "active_raw_macro_boundary_status": "pipeline_macro_dataframe_inventory_is_not_structured_macro_evidence",
                "active_fundamental_gate_binding_status": "gate_and_context_accepted_fingerprints_must_match_before_value_screening",
                "active_package_import_status": "lazy_public_api_no_pipeline_boot_for_small_module_imports",
                "active_saved_macro_producer_status": "real_snapshot_470_rows_454_point_in_time_eligible_27_series_ready",
                "active_macro_vintage_status": "fred_realtime_start_used_as_conservative_snapshot_availability_not_claimed_release_time",
                "active_macro_registry_status": "twenty_seven_series_mapped_operator_confirmation_pending",
                "active_agent_lab_macro_status": "source_registry_as_of_and_fragment_fingerprint_reverified_before_review",
                "active_july_build_status": "real_producers_then_exact_context_case_outcomes_isolated_paper_and_operations",
                "active_real_macro_agent_smoke_status": "verified_twenty_seven_series_macro_policy_neutral_no_learning_no_proposals",
                "active_macro_directionality_status": "series_presence_never_implies_policy_easing",
                "active_sec_filing_index_status": "duckdb_10191_rows_amd_10q_hash_time_and_locator_verified",
                "active_fundamental_fact_status": "twenty_nine_accession_bound_facts_four_tickers_companyfacts_and_inline_xbrl",
                "active_fundamental_sector_coverage_status": "four_of_four_source_coverage_raw_period_and_currency_comparison_blocked",
                "active_fundamental_agent_lab_status": "merged_verified_fragment_and_matching_gate_connected_raw_statement_facts_return_needs_more_data",
                "active_sec_primary_document_status": "tsm_20f_immutable_sha_bound_inline_xbrl_3353_numeric_facts_parsed",
                "active_fundamental_fingerprint_status": "producer_gate_and_agent_context_numeric_canonicalization_match",
                "active_nvda_filing_recovery_status": "official_submissions_latest_10q_accession_bound_after_local_collector_window_gap",
                "active_single_ticker_sector_claim_status": "single_ticker_source_artifact_can_never_claim_complete_sector_fundamentals",
                "active_financial_template_status": "ratio_and_valuation_templates_deferred_until_verified_facts_exist",
                "active_semiconductor_pipeline_universe_status": "nvda_amd_intc_tsm_four_ticker_pipeline_cohort",
                "active_semiconductor_research_universe_status": "twelve_ticker_value_chain_hint_not_automatic_pipeline_scope",
                "active_semiconductor_filing_coverage_status": "four_of_four_periodic_sources_after_nvda_submissions_recovery",
                "active_sector_market_evidence_status": "four_of_four_tickers_plus_qqq_twenty_two_common_sessions_market_confirmation_ready",
                "active_semiconductor_runtime_status": "verified_fundamental_macro_market_news_policy_vertical_slice_partial_ready_for_review_sector_only_five_of_five_lanes",
                "active_semiconductor_missing_lanes": "none",
                "active_semiconductor_thesis_review_status": "runtime_linked_hash_verified_sector_review_ready_with_three_explicit_cautions",
                "active_analyst_reasoning_snapshot_status": "runtime_hash_bound_152_of_152_classified_62_transmission_channels_4_candidate_hypotheses_14_evidence_gaps",
                "active_analyst_reasoning_module_policy": "classifier_regime_transmission_hypothesis_gap_verified_expectation_analog_scenario_excluded",
                "active_analyst_reasoning_ticker_leakage_status": "zero_directional_ticker_events_explicit_fundamental_attribution_non_directional",
                "active_analyst_reasoning_scenario_status": "not_generated_no_calibrated_scenario_generator",
                "active_domain_template_reasoning_status": "verified_reasoning_embedded_three_self_check_horizons_manual_acceptance_still_required",
                "active_semiconductor_ticker_thesis_status": "zero_direct_ticker_theses_four_basket_candidates",
                "active_semiconductor_prospective_case_status": "one_pre_outcome_sector_case_registered_for_30_90_180_day_review",
                "active_ticker_specific_evidence_status": "forty_nine_company_candidates_six_strong_amd_one_corroborated_demand_lane",
                "active_sector_to_ticker_pipeline_bridge_status": "amd_ticker_evidence_ready_but_pipeline_blocked_three_missing_ticker_evidence_zero_forecasts",
                "active_sector_to_ticker_review_status": "review_ready_with_limitations_readiness_gap_map_only",
                "active_structured_lane_eligibility_status": "only_explicit_required_lane_eligible_structured_evidence_closes_required_lane",
                "active_semiconductor_news_status": "18813_rows_9604_usable_9209_orphan_excluded_63_candidates_demand_capex_supply_ready",
                "active_news_lane_eligibility_status": "keyword_hits_are_candidates_two_independent_strong_sources_required",
                "active_sec_runtime_verification_status": "hash_bound_offline_reverification_no_mutable_duckdb_reopen",
                "active_sec_derived_ratio_status": "twenty_one_formula_bound_ratios_five_multi_ticker_lanes_zero_full_cohort_lanes",
                "active_official_policy_status": "bis_may_2026_pdf_hash_bound_bloomberg_corroborated_policy_lane_ready",
                "active_amd_role_status": "single_ticker_single_target_smoke_and_negative_model_case_only",
                "active_tuning_exact_scope_status": "one_failure_can_tune_only_matching_ticker_model_target_timeframe_context",
                "active_tuning_domain_broadening_status": "sector_or_multi_ticker_scope_inheritance_blocked",
                "active_pipeline_model_case_scope": "ticker_model_evaluation_only_not_domain_evidence",
                "active_model_performance_source_status": "canonical_evaluation_summary_only_complete_metric_set_required",
                "active_locked_evidence_inventory_status": "verified_pair_available_runner_still_required",
                "active_real_metric_evidence_status": "blocked_validation_and_feature_stability",
                "active_model_performance_chain_status": "locked_artifact_bound_to_full_evidence_chain",
                "active_pipeline_model_case_status": "negative_evaluation_block_case_review_ready",
                "active_pipeline_model_case_memory_status": "review_artifact_only_no_learning_write",
                "active_chief_model_case_status": "candidate_scoped_block_unrelated_work_continues",
                "active_review_feedback_taxonomy_status": "shared_taxonomy_with_domain_and_model_case_families",
                "active_pipeline_model_feedback_status": "pending_optional_manual_feedback_no_candidates",
                "active_model_feedback_apply_status": "analyst_learning_apply_loop_explicitly_incompatible",
                "active_dean_orchestrator_status": "preflight_pipeline_post_pipeline_pretrade_review",
                "active_consensus_status": "watchlist_or_blocked_no_execution_candidates",
                "hard_prerequisite_status": "synthetic_block_reports_are_enforced",
                "active_pipeline_adapter_status": "canonical_dean_review_contract_attached",
                "active_risk_returns_status": "realized_returns_preferred_target_labels_blocked_pretrade",
                "can_clone_domain_profiles_now": False,
                "can_run_live_collectors_now": False,
                "can_generate_analyst_research_recommendations_now": True,
                "can_generate_execution_recommendations_now": False,
                "can_execute_paper_simulation_now": False,
                "can_write_learning_memory_now": False,
                "can_write_production_config_now": False,
                "can_generate_recommendations_now": False,
                "can_trade": False,
            },
            "architecture_principles": _architecture_principles(),
            "branch_map": _branch_map(),
            "pipeline_metric_control_branch": {
                "purpose": "Controls whether pipeline experiments are reviewable across metric planes.",
                "agent_pattern": "metrics_guardian_and_proposal_only_tuning",
                "primary_modules": [
                    "dean_os/pipeline_metric_input_readiness_gate.py",
                    "dean_os/pipeline_control_surface.py",
                    "dean_os/pipeline_control_instance_contract.py",
                    "dean_os/pipeline_control_caution_review_packet.py",
                    "dean_os/pipeline_control_evidence_inventory.py",
                    "dean_os/pipeline_control_metric_artifact_materializer.py",
                    "dean_os/pipeline_control_locked_evaluation_assembler.py",
                    "dean_os/pipeline_control_locked_feature_stability_assembler.py",
                    "dean_os/pipeline_control_data_preflight.py",
                    "dean_os/pipeline_control_saved_data_coverage.py",
                    "dean_os/pipeline_control_saved_price_repair.py",
                    "dean_os/pipeline_control_bounded_evidence_run.py",
                    "dean_os/pipeline_control_bounded_evidence_batch.py",
                    "dean_os/pipeline_control_feature_causality_audit.py",
                    "dean_os/pipeline_control_walk_forward_validation_run.py",
                    "dean_os/pipeline_control_forward_data_accrual_plan.py",
                    "dean_os/pipeline_control_forward_data_accrual_gate.py",
                    "dean_os/pipeline_control_train_validation_diagnostic.py",
                    "dean_os/pipeline_control_train_validation_experiment.py",
                    "src/pipeline/stages/feature_engineering/timeframe_context.py",
                    "src/pipeline/timeframe_lineage.py",
                    "src/pipeline/stages/modeling/walk_forward_validation.py",
                    "src/pipeline/stages/stage_4_modeling.py",
                    "src/pipeline/stages/stage_5_prediction.py",
                    "src/pipeline/stages/prediction/output_contract.py",
                    "src/pipeline/stages/stage_6_trading_execution.py",
                    "src/pipeline/stages/stage_7_evaluation.py",
                    "src/pipeline/hybrid/final_stages_orchestrator.py",
                    "src/analytics/unified_analytics_engine.py",
                    "src/config/analysis.yaml",
                    "src/trading/trader.py",
                    "src/training/base_trainer.py",
                    "src/targets/timeframe_contract.py",
                    "src/pipeline/modeling_context.py",
                    "src/models/adapters/data_preparation.py",
                    "src/pipeline/stages/modeling/pipeline_control_artifacts.py",
                    "src/pipeline/stages/evaluation/pipeline_control_artifacts.py",
                    "dean_os/pipeline_control_metric_fixture_validation.py",
                    "dean_os/pipeline_control_real_metric_evidence_run.py",
                    "dean_os/pipeline_model_case_packet.py",
                    "dean_os/pipeline_model_feedback_packet.py",
                    "dean_os/pipeline_prediction_review_packet.py",
                    "dean_os/pipeline_feature_timeframe_audit.py",
                    "run_agent_pipeline_feature_timeframe_audit.py",
                    "dean_os/pipeline_stage23_regeneration.py",
                    "run_agent_pipeline_stage23_regeneration.py",
                    "dean_os/pipeline_target_readiness_audit.py",
                    "run_agent_pipeline_target_readiness_audit.py",
                    "dean_os/pipeline_stage4_exact_context_review.py",
                    "run_agent_pipeline_stage4_exact_context_review.py",
                    "dean_os/prediction_target_semantics.py",
                    "dean_os/shadow_calibration_readiness.py",
                    "dean_os/shadow_calibration_case_index.py",
                    "dean_os/shadow_component_case_producer.py",
                    "dean_os/shadow_calibration_diagnostics.py",
                    "dean_os/config/shadow_calibration_policy.yaml",
                    "dean_os/specialist_context_review_packet.py",
                    "dean_os/review_feedback_taxonomy.py",
                    "dean_os/agent_capability_matrix.py",
                    "dean_os/pipeline_adapter.py",
                    "dean_os/agents/regime.py",
                    "dean_os/agents/context_synthesis.py",
                    "dean_os/consensus.py",
                    "dean_os/agents/tuning.py",
                    "dean_os/replay_calibration_readiness_gate.py",
                    "dean_os/outcome_readiness_gate.py",
                    "dean_os/agents/model_performance.py",
                    "dean_os/agents/data_quality.py",
                    "dean_os/agents/risk.py",
                ],
                "metric_planes": metric_planes,
                "constraints_snapshot": _constraints_snapshot(),
                "boundary": [
                    "Metrics can block or allow reviewed experiments.",
                    "Metrics do not auto-run Optuna, retrain models, write production config, or trade.",
                    "A feasible surface means proposal review can continue, not that a model should be promoted.",
                ],
            },
            "domain_analyst_branch": {
                "purpose": "Builds domain/sector theses from source evidence and knowledge before any ticker thesis.",
                "agent_archetype": "DEANOrchestrator -> PipelineManagerAgent -> SectorPipelineManager -> DomainAnalystRuntime/SectorAnalyst",
                "execution_topology": {
                    "canonical": (
                        "Enable one PipelineManagerAgent per domain execution "
                        "group when artifact discovery, sector analysis, and "
                        "pipeline readiness must run together."
                    ),
                    "alternative": (
                        "Enable DomainAnalystAgent only for an already-populated "
                        "MarketContext research run."
                    ),
                    "conflict_rule": (
                        "Never enable composite and standalone agents for the "
                        "same execution_group and overlapping run phase."
                    ),
                    "decision_boundary": (
                        "Both transferred agents remain review-only with "
                        "decision_influence=false and can_trade=false."
                    ),
                },
                "current_profiles": domain_profiles,
                "scale_rule": "Clone to more domains only after one source-first template is manually accepted.",
                "ticker_rule": "Domain thesis can create exposure candidates; ticker thesis requires direct ticker evidence via bridge.",
                "recommendation_rule": "Review-only analyst recommendations are allowed; execution, buy/sell/hold, sizing, allocation, and order recommendations remain blocked.",
                "data_analysis_rule": "Detailed news/data analysis is allowed as context-sliced event interpretation, pipeline-derived news/crisis taxonomy, optional saved pipeline-context overlay, regime-context vector, scenario outcome graph, mechanism hypotheses, value-chain mapping, watch metrics, evidence gaps, self-check horizons, and review items.",
                "primary_modules": [
                    "dean_os/orchestrator.py",
                    "dean_os/registry.py",
                    "dean_os/config/agent_registry.yaml",
                    "dean_os/agents/pipeline_manager.py",
                    "dean_os/agents/domain_analyst.py",
                    "dean_os/agents/pipeline_readiness.py",
                    "dean_os/analyst_core/pipeline_manager.py",
                    "run_agent_composite_domain_pipeline.py",
                    "dean_os/analysts/base.py",
                    "dean_os/analysts/profiles.py",
                    "dean_os/domain_analyst_profile_policy_packet.py",
                    "dean_os/domain_analyst_pipeline_news_taxonomy.py",
                    "dean_os/domain_analyst_event_interpretation_packet.py",
                    "dean_os/domain_analyst_regime_scenario_packet.py",
                    "dean_os/domain_analyst_intake_packet.py",
                    "dean_os/domain_analyst_instance_contract.py",
                    "dean_os/domain_analyst_thesis_review_packet.py",
                    "dean_os/analyst_core/artifact_evidence_loader.py",
                    "dean_os/analyst_core/lens_contract.py",
                    "dean_os/analyst_core/lens_orchestrator.py",
                    "dean_os/analyst_core/sector_analyst.py",
                    "dean_os/analyst_core/lenses/event_classifier_lens.py",
                    "dean_os/analyst_core/lenses/regime_context_lens.py",
                    "dean_os/analyst_core/lenses/transmission_mapper_lens.py",
                    "dean_os/analyst_core/lenses/hypothesis_ledger_lens.py",
                    "dean_os/analyst_core/lenses/evidence_gap_lens.py",
                    "dean_os/analyst_core_reasoning_snapshot.py",
                    "run_agent_analyst_core_reasoning_snapshot.py",
                    "dean_os/domain_analyst_forecast_review_packet.py",
                    "dean_os/domain_analyst_template_standardization_packet.py",
                    "dean_os/domain_analyst_template_decision_packet.py",
                    "dean_os/domain_analyst_portability_review.py",
                    "dean_os/domain_analyst_case_registry_packet.py",
                    "dean_os/domain_analyst_feedback_loop_packet.py",
                    "dean_os/domain_analyst_vertical_slice_run.py",
                    "dean_os/agents/working_domain_analyst.py",
                    "dean_os/analyst_knowledge/schemas.py",
                    "dean_os/analyst_knowledge/store.py",
                    "dean_os/analyst_knowledge/retriever.py",
                    "dean_os/analyst_knowledge_readiness.py",
                    "dean_os/context_evidence_provenance.py",
                    "dean_os/structured_context_provenance.py",
                    "dean_os/context_evidence_review_packet.py",
                    "dean_os/analysts/context_adapter.py",
                    "dean_os/agents/domain_research.py",
                    "dean_os/agents/research_agents.py",
                    "dean_os/agent_lab.py",
                    "dean_os/fundamental_input_readiness_gate.py",
                    "dean_os/saved_macro_evidence_producer.py",
                    "dean_os/config/macro_series_registry.yaml",
                    "run_agent_saved_macro_evidence_producer.py",
                    "dean_os/saved_sec_filing_index_producer.py",
                    "dean_os/saved_sec_submissions_filing_index_producer.py",
                    "run_agent_saved_sec_filing_index.py",
                    "run_agent_saved_sec_submissions_filing_index.py",
                    "dean_os/saved_sec_companyfacts_producer.py",
                    "dean_os/saved_sec_inline_xbrl_producer.py",
                    "dean_os/saved_sec_fundamental_evidence_merger.py",
                    "dean_os/config/fundamental_metric_registry.yaml",
                    "run_agent_sec_companyfacts_snapshot.py",
                    "run_agent_saved_sec_companyfacts.py",
                    "run_agent_saved_sec_inline_xbrl.py",
                    "run_agent_saved_sec_fundamental_merger.py",
                    "dean_os/saved_sec_derived_ratio_producer.py",
                    "run_agent_saved_sec_derived_ratios.py",
                    "dean_os/saved_sector_market_evidence_producer.py",
                    "run_agent_saved_sector_market_evidence.py",
                    "dean_os/saved_semiconductor_news_evidence_producer.py",
                    "dean_os/config/semiconductor_news_source_registry.yaml",
                    "run_agent_saved_semiconductor_news_evidence.py",
                    "dean_os/saved_ticker_specific_evidence_producer.py",
                    "dean_os/config/semiconductor_issuer_identity_registry.yaml",
                    "run_agent_saved_ticker_specific_evidence.py",
                    "dean_os/saved_official_policy_evidence_producer.py",
                    "dean_os/config/official_policy_evidence_registry.yaml",
                    "run_agent_bis_policy_snapshot.py",
                    "run_agent_saved_official_policy_evidence.py",
                    "dean_os/semiconductor_analyst_runtime.py",
                    "run_agent_semiconductor_analyst.py",
                    "dean_os/JULY_2026_BUILD_ROADMAP.md",
                    "dean_os/__init__.py",
                    "dean_os/sector_thesis_to_ticker_basket_bridge.py",
                    "dean_os/sector_to_ticker_review_packet.py",
                ],
                "analyst_control_planes": analyst_planes,
            },
            "source_and_fundamental_lanes": _source_and_fundamental_lanes(),
            "orchestrator_contract": _orchestrator_contract(),
            "build_focus_control": _build_focus_control(),
            "corrections_to_user_plan": _corrections_to_user_plan(),
            "existing_module_map": _existing_module_map(),
            "review_gates": _review_gates(),
            "next_safe_steps": _next_safe_steps(),
            "explicit_non_actions": _explicit_non_actions(),
            "commands": _commands(),
        }
        if save:
            saved_paths = ReviewArtifactWriter(self.output_dir).write(
                payload=payload,
                markdown=render_current_architecture_map_markdown(payload),
                run_id=payload["run_id"],
            )
            payload["saved_paths"] = saved_paths
        return json_ready(payload)


def render_current_architecture_map_markdown(payload: dict[str, Any]) -> str:
    summary = payload.get("summary", {})
    lines = [
        "# DEAN-OS Current Architecture Map",
        "",
        f"- Run ID: `{payload.get('run_id')}`",
        f"- Version: `{payload.get('architecture_version')}`",
        f"- Architecture: `{summary.get('architecture_status')}`",
        f"- Active design: `{summary.get('active_design')}`",
        f"- Pipeline metric planes: {summary.get('pipeline_metric_plane_count')}",
        f"- Domain profiles: {summary.get('domain_profile_count')}",
        f"- Recommended action: `{summary.get('recommended_action')}`",
        f"- Can clone domain profiles now: {summary.get('can_clone_domain_profiles_now')}",
        f"- Can write production config now: {summary.get('can_write_production_config_now')}",
        f"- Can trade: {summary.get('can_trade')}",
        "",
        "## Branch Map",
        "",
    ]
    for branch in payload.get("branch_map", []):
        lines.append(f"- `{branch.get('branch_id')}`: {branch.get('purpose')}")
        lines.append(f"  - Authority: {branch.get('authority')}")

    lines.extend(["", "## Pipeline Metric Planes", ""])
    for plane in payload.get("pipeline_metric_control_branch", {}).get("metric_planes", []):
        lines.append(f"- `{plane.get('plane_id')}`: {plane.get('purpose')}")
        lines.append(f"  - Current owner: `{plane.get('current_owner')}`")

    lines.extend(["", "## Domain Analyst Branch", ""])
    domain = payload.get("domain_analyst_branch", {})
    lines.append(f"- Archetype: `{domain.get('agent_archetype')}`")
    lines.append(f"- Scale rule: {domain.get('scale_rule')}")
    lines.append(f"- Ticker rule: {domain.get('ticker_rule')}")
    lines.append(f"- Data analysis rule: {domain.get('data_analysis_rule')}")
    for profile in domain.get("current_profiles", []):
        lines.append(f"- `{profile.get('domain_id')}`: {profile.get('display_name')} ({profile.get('ticker_universe_count')} ticker hints)")
    lines.extend(["", "## Domain Analyst Control Planes", ""])
    for plane in domain.get("analyst_control_planes", []):
        lines.append(f"- `{plane.get('plane_id')}`: {plane.get('purpose')}")

    lines.extend(["", "## Corrections To Current Plan", ""])
    for item in payload.get("corrections_to_user_plan", []):
        lines.append(f"- {item}")

    lines.extend(["", "## Orchestrator Contract", ""])
    contract = payload.get("orchestrator_contract", {})
    lines.append(f"- Role: {contract.get('role')}")
    for rule in contract.get("rules", []):
        lines.append(f"- {rule}")

    lines.extend(["", "## Next Safe Steps", ""])
    lines.extend(f"- {item}" for item in payload.get("next_safe_steps", []))

    lines.extend(["", "## Explicit Non-Actions", ""])
    lines.extend(f"- {item}" for item in payload.get("explicit_non_actions", []))
    return "\n".join(lines).strip() + "\n"


def _architecture_principles() -> list[str]:
    return [
        "Deterministic code owns numeric metrics, data splits, leakage checks, replay outcomes, and control surfaces.",
        "Domain analysts own interpretation of source-backed economic, political, historical, and sector context.",
        "Domain analysts may produce review-only research recommendations, scenario priorities, evidence requests, and improvement proposals.",
        "Sector/domain thesis is not a ticker thesis.",
        "Ticker thesis requires direct ticker evidence through a separate bridge.",
        "All branches emit review artifacts; none can approve their own downstream action.",
        "Architecture documents are design context; active pipeline code, causal audits, and current artifacts must be re-checked before each implementation slice.",
        "Multi-timeframe features use completed-bar backward joins, preserve ticker/partition identity, and never import context targets.",
        "Walk-forward development evidence uses purged expanding train/validation folds only and is never eligible as locked test evidence.",
        "Execution recommendations, learning promotion, model promotion, production config writes, and trading are separate gated lifecycles.",
        "A failed locked evaluation is a negative model-review case, not a realized forecast miss or an automatic learning-memory record.",
        "A blocked model candidate stops its own tuning, promotion, recommendation, and trading path without freezing unrelated research, analyzer review, pipeline engineering, or safe forward-data work.",
        "Domain outcome feedback and model evaluation feedback share review/process vocabulary but remain separate semantic families; the directional analyst learning loop cannot consume model cases.",
        "A Stage 5 value produced by predict plus contextual overlays is never called a probability unless an explicit probability-producing path and calibration prove that meaning.",
        "Transferred workbench modules are reusable implementation material, but each module must be classified as active foundation, bounded lifecycle, proposal-only helper, or superseded history before pipeline integration.",
    ]


def _branch_map() -> list[dict[str, Any]]:
    return [
        {
            "branch_id": "source_intake_and_evidence",
            "purpose": "Normalize local/cached/source materials into bounded evidence artifacts.",
            "authority": "read_source_and_write_review_artifacts_only",
            "current_modules": [
                "dean_os/analyst_evidence_pack.py",
                "dean_os/source_evidence_validation_gate.py",
                "dean_os/real_source_normalized_packet.py",
                "dean_os/real_source_dropzone_inventory.py",
            ],
        },
        {
            "branch_id": "pipeline_metric_control",
            "purpose": "Evaluate pipeline health across metric planes before any tuning proposal.",
            "authority": "block_or_allow_reviewed_experiment_proposals",
            "current_modules": [
                "dean_os/pipeline_metric_input_readiness_gate.py",
                "dean_os/pipeline_control_surface.py",
                "dean_os/pipeline_control_instance_contract.py",
                "dean_os/pipeline_control_caution_review_packet.py",
                "dean_os/pipeline_control_evidence_inventory.py",
                "dean_os/pipeline_control_metric_artifact_materializer.py",
                "dean_os/pipeline_control_locked_evaluation_assembler.py",
                "dean_os/pipeline_control_locked_feature_stability_assembler.py",
                "dean_os/pipeline_control_walk_forward_validation_run.py",
                "dean_os/pipeline_control_forward_data_accrual_plan.py",
                "dean_os/pipeline_control_forward_data_accrual_gate.py",
                "src/pipeline/stages/feature_engineering/timeframe_context.py",
                "src/pipeline/stages/modeling/walk_forward_validation.py",
                "src/pipeline/stages/stage_4_modeling.py",
                    "src/pipeline/stages/stage_5_prediction.py",
                    "src/pipeline/hybrid/colab_manager.py",
                    "src/processing/cleaners.py",
                    "src/data/collectors/yf_collector.py",
                    "src/targets/timeframe_contract.py",
                    "src/config/targets.yaml",
                "src/training/base_trainer.py",
                "src/targets/timeframe_contract.py",
                "src/pipeline/modeling_context.py",
                "src/pipeline/stages/modeling/pipeline_control_artifacts.py",
                "src/pipeline/stages/evaluation/pipeline_control_artifacts.py",
                "dean_os/pipeline_control_metric_fixture_validation.py",
                "dean_os/pipeline_control_real_metric_evidence_run.py",
                "dean_os/agents/tuning.py",
            ],
        },
        {
            "branch_id": "domain_analyst_research",
            "purpose": "Produce source-backed domain theses and ticker exposure maps.",
            "authority": "review_only_thesis_and_bridge_candidates",
            "current_modules": [
                "dean_os/analysts/base.py",
                "dean_os/agents/working_domain_analyst.py",
                "dean_os/domain_analyst_forecast_review_packet.py",
                "dean_os/domain_analyst_case_registry_packet.py",
                "dean_os/domain_analyst_event_interpretation_packet.py",
                "dean_os/domain_analyst_regime_scenario_packet.py",
                "dean_os/domain_analyst_feedback_loop_packet.py",
            ],
        },
        {
            "branch_id": "review_orchestration",
            "purpose": "Route branch outputs through review gates and human decisions.",
            "authority": "summarize_gate_and_record_review_decisions",
            "current_modules": [
                "dean_os/review_only_automation_run.py",
                "dean_os/current_system_alignment_review.py",
                "dean_os/build_focus_review_packet.py",
                "dean_os/review_decision_packet.py",
                "dean_os/review_action_dry_run.py",
                "dean_os/review_action_apply_ceremony.py",
                "dean_os/pipeline_model_case_packet.py",
                "dean_os/pipeline_model_feedback_packet.py",
                "dean_os/review_feedback_taxonomy.py",
                "dean_os/review_index.py",
                "dean_os/chief_review_index.py",
                "dean_os/review_decision.py",
                "dean_os/paper_lifecycle_contract.py",
                "dean_os/paper_simulation_plan.py",
                "dean_os/paper_simulation_result.py",
                "dean_os/post_paper_simulation_review.py",
            ],
        },
    ]


def _pipeline_metric_planes() -> list[dict[str, Any]]:
    return [
        _plane("profitability", "PnL, total return, Sharpe, and outcome proxy evidence.", "PipelineControlSurface"),
        _plane("risk", "Drawdown, downside boundary, sizing/risk veto context.", "PipelineControlSurface + RiskAgent"),
        _plane("validation_split", "Train/validation gap, sample count, holdout quality, and data split health.", "PipelineControlSurface"),
        _plane("feature_stability", "Feature concentration, unstable features, and feature lineage.", "PipelineControlSurface"),
        _plane("data_quality_leakage", "Data warnings, leakage flags, timestamp quality, and source freshness.", "DataQualityAgent + timestamp gates"),
        _plane("replay_repeatability", "Clear replay hit rate, blocked replay runs, and sufficient replay samples.", "ReplayCalibrationReadinessGate"),
        _plane("outcome_coverage", "Whether promoted theses have enough future price/outcome coverage to learn from.", "OutcomeReadinessGate"),
        _plane("market_data_freshness", "Whether prices and market inputs are current enough for evaluation.", "MarketDataFreshnessAgent"),
    ]


def _plane(plane_id: str, purpose: str, owner: str) -> dict[str, str]:
    return {
        "plane_id": plane_id,
        "purpose": purpose,
        "current_owner": owner,
        "decision_authority": "can_block_or_allow_reviewed_proposals_only",
    }


def _domain_analyst_control_planes() -> list[dict[str, str]]:
    return [
        _analyst_plane("evidence_coverage", "Required evidence lanes are present before a thesis is treated as reviewable."),
        _analyst_plane("source_quality_timestamp", "Source lineage, timestamps, and local artifacts are visible."),
        _analyst_plane("news_event_interpretation", "News/data become context-sliced event hypotheses, pipeline-derived crisis/news taxonomy, optional saved pipeline-context overlays, mechanisms, counterforces, watch metrics, and evidence gaps.", "DomainAnalystEventInterpretationPacket + PipelineNewsTaxonomyAdapter"),
        _analyst_plane("regime_scenario_context", "News is evaluated against a multi-field regime vector, scenario outcome graph, evidence-gap priorities, historical analog candidates, and self-check horizons.", "DomainAnalystRegimeScenarioPacket"),
        _analyst_plane("thesis_falsifiability", "Theses become explicit expectations with observable future criteria."),
        _analyst_plane("horizon_maturity", "Outcomes are not scored before the stated horizon unless marked diagnostic."),
        _analyst_plane("confidence_calibration", "Confidence is compared against evidence quality and eventual outcomes."),
        _analyst_plane("contradiction_handling", "Contradicting evidence and blind spots remain visible beside support."),
        _analyst_plane("causal_attribution", "Correct direction is separated from correct reasoning."),
        _analyst_plane("luck_vs_skill", "Lucky or wrong-reason hits are not promoted as analyst skill."),
        _analyst_plane("feedback_to_learning_candidate", "Human labels become proposal-only learning candidates, not automatic memory updates.", "DomainAnalystFeedbackLoopPacket"),
        _analyst_plane("ticker_directness_boundary", "Sector/domain theses are not treated as direct ticker forecasts without a bridge."),
        _analyst_plane("learning_promotion_readiness", "Lessons stay proposal-only until human-approved learning promotion."),
    ]


def _analyst_plane(plane_id: str, purpose: str, current_owner: str = "DomainAnalystForecastReviewPacket") -> dict[str, str]:
    return {
        "plane_id": plane_id,
        "purpose": purpose,
        "current_owner": current_owner,
        "decision_authority": "can_block_or_request_human_review_only",
    }


def _constraints_snapshot() -> dict[str, Any]:
    keys = [
        "min_total_return",
        "min_pnl",
        "min_sharpe",
        "max_drawdown",
        "max_train_test_gap",
        "max_leakage_flags",
        "min_clear_replay_hit_rate",
        "min_clear_replay_runs",
    ]
    return {key: DEFAULT_CONSTRAINTS[key] for key in keys}


def _domain_profiles() -> list[dict[str, Any]]:
    profiles = []
    for domain_id in list_domain_profiles():
        profile = get_domain_profile(domain_id)
        profiles.append(
            {
                "domain_id": profile.domain_id,
                "display_name": profile.display_name,
                "horizon_days_default": profile.horizon_days_default,
                "required_evidence_types": list(profile.required_evidence_types),
                "ticker_universe_count": len(profile.ticker_universe_hint),
                "direct_ticker_evidence_rule_count": len(profile.direct_ticker_evidence_rules),
                "source_registry_policy_id": profile.source_registry_policy.get("policy_id"),
                "ingestion_filter_policy_id": profile.ingestion_filter_policy.get("policy_id"),
                "evidence_scoring_policy_id": profile.evidence_scoring_policy.get("policy_id"),
                "review_output_policy_id": profile.review_output_policy.get("policy_id"),
                "feedback_label_policy_id": profile.feedback_label_policy.get("policy_id"),
            }
        )
    return profiles


def _source_and_fundamental_lanes() -> dict[str, Any]:
    return {
        "source_lane": {
            "sequence": [
                "cached_or_operator_sources",
                "AnalystEvidencePack or RealSourceNormalizedPacket",
                "DomainAnalystProfilePolicyPacket",
                "SourceEvidenceValidationGate",
                "DomainAnalystEventInterpretationPacket",
                "DomainAnalystRegimeScenarioPacket",
                "DomainAnalystIntakePacket",
                "DomainAnalystInstanceContract",
                "DomainAnalystThesisReviewPacket",
                "DomainAnalystForecastReviewPacket",
                "DomainAnalystTemplateStandardizationPacket",
                "DomainAnalystTemplateDecisionPacket",
                "DomainAnalystCaseRegistryPacket",
                "DomainAnalystFeedbackLoopPacket",
                "DomainSpecialistReviewPacket",
                "SourceExtractionReviewPacket only after review",
            ],
            "boundary": "Source validation does not extract claims, promote evidence, write learning memory, create execution recommendations, or trade.",
        },
        "fundamental_lane": {
            "sequence": [
                "caller_supplied_fundamentals",
                "StructuredContextProvenance value/unit/period/availability/source quarantine",
                "FundamentalInputReadinessGate with explicit as-of",
                "gate accepted-fingerprint equals context accepted-fingerprint",
                "AgentLab ValueScreeningAgent only when both contracts pass",
            ],
            "boundary": "Fundamental readiness does not extract financial statements, infer missing semantic fields, compute ratios, value companies, create execution recommendations, or trade.",
        },
    }


def _orchestrator_contract() -> dict[str, Any]:
    return {
        "role": "Coordinate branch outputs and enforce gates; never act as a trader.",
        "rules": [
            "Run source validation before domain review.",
            "Run metric-plane checks before tuning proposals.",
            "Route locked evaluation results into a deduplicated model case before Chief Review; never relabel an evaluation block as a forecast miss.",
            "Normalize optional human feedback through the case-family taxonomy and create proposal-only candidates; never infer feedback or approval from a metric failure.",
            "Keep domain thesis, ticker evidence, fundamentals, learning, and trading separate.",
            "Allow hard metric gates to block downstream review when data quality or replay quality fails.",
            "Allow domain analysts to add context, contradictions, and thesis quality, not execution authority.",
            "Record human review decisions before any learning or dry-run lifecycle advances.",
        ],
    }


def _build_focus_control() -> dict[str, Any]:
    return {
        "role": "Prevent implementation loops from deepening a branch after the next action is already known.",
        "productive_deepening_rule": (
            "Continue digging only when the work closes a named blocker, adds a reusable boundary, "
            "or changes the next downstream decision."
        ),
        "stop_or_switch_rule": (
            "Pause or switch branches when the branch is waiting for manual review, future outcome data, "
            "or another branch has concrete blockers."
        ),
        "current_tool": "BuildFocusReviewPacket",
    }


def _corrections_to_user_plan() -> list[str]:
    return [
        "The pipeline-control agent should not search for a single automatic optimum where every metric intersects; it should define a feasible review surface and blockers.",
        "PnL is not enough by itself; data split quality, leakage, replay repeatability, drawdown, feature stability, and outcome coverage are separate planes.",
        "The economic analyst should be cloned only after the base domain analyst contract is stable on one sector/template.",
        "A domain analyst can know economics, history, policy, and sector mechanics, but it must output sector/domain thesis first.",
        "The orchestrator should reconcile gates and branch reports, not merge them into a trade signal.",
    ]


def _existing_module_map() -> dict[str, list[str]]:
    return {
        "already_useful": [
            "PipelineControlSurface",
            "PipelineMetricInputReadinessGate",
            "PipelineControlInstanceContract",
            "PipelineControlCautionReviewPacket",
            "PipelineControlEvidenceInventory",
            "PipelineControlMetricArtifactMaterializer",
            "PipelineControlLockedEvaluationAssembler",
            "PipelineControlLockedFeatureStabilityAssembler",
            "PipelineControlBoundedEvidenceRun",
            "PipelineControlWalkForwardValidationRun",
            "PipelineControlForwardDataAccrualPlan",
            "PipelineControlForwardDataAccrualGate",
            "PipelineWalkForwardValidationEvaluator",
            "Stage4WalkForwardReviewOnlyPath",
            "PipelineStage4ExactContextReview",
            "ActiveStage4ValidationOnlyTrainingAdapter",
            "ActiveStage4PipelineControlArtifactEmission",
            "Stage5ModelLineagePropagation",
            "ActiveStage6ReviewOnlyExecutionBoundary",
            "Stage5ToStage7DefaultFinalOrchestration",
            "LiveTraderInitializationBlock",
            "Stage7ProposalOnlyLearningBoundary",
            "Stage7ExplicitNotificationAuthorization",
            "Stage7ContextPartitionedAnalyzerReview",
            "UnifiedAnalyticsCoverageContract",
            "CanonicalStage7AnalyzerConfig",
            "Stage7AnalyzerReviewContractBridge",
            "CanonicalModelPerformanceMetricExtraction",
            "LockedEvidenceProvenanceVerification",
            "ModelPerformanceEvidenceChainBinding",
            "DEANOrchestratorTwoPhaseSafetyReview",
            "ConsensusWatchlistOnlyDefault",
            "SyntheticHardPrerequisiteBlock",
            "CanonicalPipelineReviewContractAdapter",
            "OfflineTargetReturnRiskBlock",
            "WinnerOnlyChampionPersistence",
            "BackwardTimeframeContextAssembler",
            "TargetTimeframeContract",
            "PipelineModelContextIsolation",
            "PipelineTrainingMetricArtifactCandidates",
            "PipelineEvaluationMetricArtifactCandidates",
            "PipelineControlMetricFixtureValidation",
            "PipelineControlRealMetricEvidenceRun",
            "DeanOSReviewOnlyAutomationRun",
            "TuningAgent proposal-only guardrail",
            "DomainAnalystIntakePacket",
            "DomainAnalystInstanceContract",
            "DomainAnalystPipelineNewsTaxonomy",
            "DomainAnalystEventInterpretationPacket",
            "DomainAnalystRegimeScenarioPacket",
            "VerifiedAnalystCoreLensOrchestrator",
            "HashBoundAnalystCoreReasoningSnapshot",
            "DomainAnalystThesisReviewPacket",
            "DomainAnalystForecastReviewPacket",
            "DomainAnalystTemplateStandardizationPacket",
            "DomainAnalystTemplateDecisionPacket",
            "DomainAnalystProfilePolicyPacket",
            "DomainAnalystPortabilityReview",
            "DomainAnalystCaseRegistryPacket",
            "DomainAnalystFeedbackLoopPacket",
            "DomainAnalystVerticalSliceRun",
            "BaseAnalystAgent and domain profiles",
            "WorkingDomainAnalystAgent",
            "AnalystKnowledgePointInTimeRetrieval",
            "AnalystKnowledgeReadiness",
            "ContextEvidencePointInTimeBoundary",
            "ContextEvidenceReviewPacket",
            "HashBoundIsolatedPaperLifecycle",
            "SourceEvidenceValidationGate",
            "CurrentSystemAlignmentReview",
            "BuildFocusReviewPacket",
            "FundamentalInputReadinessGate",
        ],
        "staged_or_review_only": [
            "SectorToTickerReviewPacket",
            "DomainSpecialistReviewPacket",
            "SourceExtractionReviewPacket",
            "SourceExtractionFixtureReviewGate",
        ],
        "not_next": [
            "live collector execution as default input",
            "autonomous tuning",
            "profile multiplication",
            "live trading",
        ],
    }


def _review_gates() -> list[dict[str, str]]:
    return [
        {"gate_id": "source_evidence_validation", "module": "SourceEvidenceValidationGate", "blocks": "bad source shape or downstream actions enabled"},
        {"gate_id": "analyst_knowledge_point_in_time", "module": "LocalKnowledgeStore + KnowledgeRetriever + AnalystKnowledgeReadiness", "blocks": "knowledge items from entering analyst review when item authoring time, source publication/retrieval time, content hash, locator, allowed use, pack SHA lineage, or as-of timezone is missing or future; knowledge can never substitute for the raw-source gate or directly influence Stage5, consensus, or trading"},
        {"gate_id": "context_evidence_point_in_time", "module": "HybridPipelineAdapter + MarketContextEvidenceAdapter + ContextEvidenceReviewPacket", "blocks": "pipeline/caller news or derived research notes from entering analyst evidence when context as-of, publication timestamp, stable locator, citation time, or provenance is missing/future; plain text ticker mentions cannot create direct-ticker evidence and pipeline_result must use its separate exact-context review contract"},
        {"gate_id": "structured_context_point_in_time", "module": "StructuredContextProvenance + HybridPipelineAdapter + AgentLabRunner + direct domain agents", "blocks": "fundamental, macro, or sector observations when value, explicit unit, period, timezone-aware availability timestamp, stable source locator, or as-of compatibility is missing; raw macro table inventory and document counts remain metadata rather than evidence"},
        {"gate_id": "saved_macro_evidence_producer", "module": "SavedMacroEvidenceProducer + macro_series_registry + verified fragment loader", "blocks": "saved macro rows when schema, series registry, unit, observation period, conservative vintage availability, source locator, source SHA, registry SHA, as-of, or accepted fragment fingerprint is missing, future, changed, or inconsistent; FRED realtime_start is never relabeled as guaranteed original release time"},
        {"gate_id": "saved_sec_filing_index", "module": "SavedSECFilingIndexProducer + read-only DuckDB verification", "blocks": "saved filing metadata when acceptance time, ticker, CIK, accession, form, primary document, collector hash, stable SEC archive locator, as-of eligibility, source row identity, or artifact fingerprint is invalid; metadata-only rows can request content but cannot become fundamental metrics"},
        {"gate_id": "saved_sec_submissions_filing_index", "module": "immutable SEC submissions snapshot + SavedSECSubmissionsFilingIndexProducer + filing-index verifier dispatcher", "blocks": "recovered periodic filings unless snapshot/asset-config/raw hashes, ticker/CIK identity, exact acceptance time, form, report period, accession, primary document, as-of, selected-latest rule, row hash, and index fingerprint agree"},
        {"gate_id": "saved_sec_companyfacts", "module": "SavedSECCompanyFactsProducer + metric registry + verified fragment loader", "blocks": "SEC facts unless the raw snapshot hash, CIK, verified filing accession, form, report end, quarterly or annual duration, unit, acceptance time, registry hash, as-of, fact hash, and accepted fragment fingerprint agree; incomplete ticker coverage or unit mismatch cannot become a sector conclusion"},
        {"gate_id": "saved_sec_inline_xbrl", "module": "immutable primary-document snapshot + SavedSECInlineXBRLProducer", "blocks": "inline-XBRL facts unless primary-document SHA, verified accession, CIK, consolidated non-dimensional context, exact reporting period, registered reporting currency, unit, scale/sign transform, acceptance time, registry hash, and context fingerprint agree"},
        {"gate_id": "saved_sec_fundamental_merger", "module": "SavedSECFundamentalEvidenceMerger + verified Agent Lab loader", "blocks": "conflicting duplicate facts, changed source artifacts, missing tickers, mixed fiscal periods, or mixed currencies from becoming complete sector fundamentals; raw statements never become valuation ratios"},
        {"gate_id": "saved_sec_derived_ratios", "module": "SavedSECDerivedRatioProducer + verified fragment loader", "blocks": "ratios unless numerator and denominator share ticker, unit, exact period, and period type with source-fact hashes; quarterly and annual comparison lanes remain separate"},
        {"gate_id": "saved_sector_market_evidence", "module": "SavedSectorMarketEvidenceProducer + verified fragment loader", "blocks": "market confirmation unless immutable repair/source hashes, four-ticker plus QQQ coverage, daily OHLCV validity, observed source-bar count, common lookback window, freshness, and structured evidence fingerprint agree"},
        {"gate_id": "saved_semiconductor_news_evidence", "module": "SavedSemiconductorNewsEvidenceProducer + source-tier registry + verified fragment loader", "blocks": "keyword-only, orphan, future, stale, locatorless, duplicate, weak-source-only, or singly sourced news from closing demand, capex, supply-chain, or policy lanes"},
        {"gate_id": "saved_ticker_specific_evidence", "module": "SavedTickerSpecificEvidenceProducer + reviewed issuer identity registry + verified fragment loader", "blocks": "plain substring issuer matches, weak or singly sourced company headlines, conflicting directional stance, raw fundamentals, or sector context from becoming eligible ticker-specific mechanism evidence; the packet still cannot create a ticker forecast"},
        {"gate_id": "saved_official_policy_evidence", "module": "immutable BIS snapshot + official policy registry + SavedOfficialPolicyEvidenceProducer", "blocks": "policy evidence unless the official PDF SHA, publication cutoff, registry mapping, BIS host, independent corroborating source, and fragment fingerprint agree"},
        {"gate_id": "semiconductor_required_evidence_lanes", "module": "SemiconductorAnalystRuntime + MarketContextEvidenceAdapter + BaseAnalystAgent", "blocks": "a sector thesis when any required lane lacks explicitly eligible evidence; generic fundamentals and macro remain supporting context, and ticker/model pipeline cases remain excluded"},
        {"gate_id": "verified_analyst_reasoning_snapshot", "module": "ArtifactEvidenceLoader + LensOrchestrator + AnalystCoreReasoningSnapshot", "blocks": "reasoning output when the runtime or linked source hashes change, evidence counts diverge, one evidence item is classified more than once, unverified probability/analog modules enter the verified path, or plain-text/fundamental ticker attribution becomes directional ticker reasoning"},
        {"gate_id": "tuning_exact_model_scope", "module": "ModelPerformanceAgent + TuningAgent", "blocks": "an actionable failure from becoming a tuning proposal unless ticker, model, target_name, timeframe, and context_fingerprint are complete and consistent; config cannot broaden one evaluated ticker into a sector or multi-ticker experiment"},
        {"gate_id": "domain_analyst_instance_contract", "module": "DomainAnalystInstanceContract", "blocks": "template standardization before one analyst instance is coherent"},
        {"gate_id": "domain_analyst_thesis_review", "module": "DomainAnalystThesisReviewPacket", "blocks": "domain template standardization before thesis evidence/risk review is accepted"},
        {"gate_id": "domain_analyst_forecast_review", "module": "DomainAnalystForecastReviewPacket", "blocks": "learning promotion when a thesis lacks frozen expectations, causal outcome taxonomy, or self-improvement boundary"},
        {"gate_id": "domain_analyst_template_standardization", "module": "DomainAnalystTemplateStandardizationPacket", "blocks": "domain scaling or ticker bridge until human template acceptance is recorded separately"},
        {"gate_id": "domain_analyst_template_decision", "module": "DomainAnalystTemplateDecisionPacket", "blocks": "domain cloning until the human decision is recorded as template-process acceptance, not thesis truth"},
        {"gate_id": "domain_analyst_profile_policy", "module": "DomainAnalystProfilePolicyPacket", "blocks": "domain cloning when source, ingestion, scoring, review-output, or feedback policies are missing"},
        {"gate_id": "domain_analyst_event_interpretation", "module": "DomainAnalystEventInterpretationPacket", "blocks": "thesis updates from news/data when context slices, pipeline news/crisis taxonomy, optional saved pipeline-context overlays, event mechanisms, counterforces, evidence gaps, or review boundaries are missing"},
        {"gate_id": "domain_analyst_regime_scenario", "module": "DomainAnalystRegimeScenarioPacket", "blocks": "using news in thesis updates when regime vector, news-vs-regime assessment, scenario graph, evidence gaps, or self-check horizons are missing"},
        {"gate_id": "domain_analyst_portability_review", "module": "DomainAnalystPortabilityReview", "blocks": "profile cloning when required evidence aliases or optional adapter boundaries are unclear"},
        {"gate_id": "domain_analyst_case_registry", "module": "DomainAnalystCaseRegistryPacket", "blocks": "post-outcome thesis rewriting, stale template bindings, or learning promotion that only sees hit/correct cases and ignores misses, pending, or invalid cases; freezes source hash, baseline context, and 30/90/180-day review schedule"},
        {"gate_id": "domain_analyst_feedback_loop", "module": "DomainAnalystFeedbackLoopPacket", "blocks": "learning promotion when human feedback labels are invalid, request execution, or try to apply learning directly"},
        {"gate_id": "domain_specialist_review", "module": "DomainSpecialistReviewPacket", "blocks": "missing domain thesis or source warnings"},
        {"gate_id": "pipeline_metric_input_readiness", "module": "PipelineMetricInputReadinessGate", "blocks": "surface refresh from missing, unreadable, or known-blocked metric inputs"},
        {"gate_id": "pipeline_control_surface", "module": "PipelineControlSurface", "blocks": "metric planes outside constraints"},
        {"gate_id": "pipeline_control_instance_contract", "module": "PipelineControlInstanceContract", "blocks": "pipeline control standardization before saved metric planes are coherent"},
        {"gate_id": "pipeline_control_caution_review", "module": "PipelineControlCautionReviewPacket", "blocks": "using caution planes as if drawdown, validation, and feature stability evidence were already proven"},
        {"gate_id": "pipeline_control_evidence_inventory", "module": "PipelineControlEvidenceInventory", "blocks": "partial pipeline outputs, complete-looking metric shapes without verified locked provenance, selected feature manifests, smoke reports, or clean lineage from being treated as locked metric evidence; a ready pair permits the real runner but never clears cautions by itself"},
        {"gate_id": "pipeline_control_metric_artifact_materializer", "module": "PipelineControlMetricArtifactMaterializer", "blocks": "writing a model-evaluation/feature-stability pair unless saved non-synthetic candidates satisfy the full contracts and matching ticker/model/target/timeframe/context lineage"},
        {"gate_id": "pipeline_control_locked_evaluation_assembler", "module": "PipelineControlLockedEvaluationAssembler", "blocks": "joining training metrics with Stage 7 drawdown unless same-window model/target/context/window lineage is proven"},
        {"gate_id": "pipeline_control_locked_feature_stability_assembler", "module": "PipelineControlLockedFeatureStabilityAssembler", "blocks": "promoting feature importances to stability evidence unless a measured stability signal and model lineage are present"},
        {"gate_id": "pipeline_control_bounded_evidence", "module": "PipelineControlBoundedEvidenceRun", "blocks": "offline model evidence when source quality, purged chronological splits, matching lineage, or locked artifact assembly fails"},
        {"gate_id": "pipeline_control_feature_causality", "module": "PipelineControlFeatureCausalityAudit", "blocks": "locked evidence when adding a future suffix changes earlier numeric feature values or Stage 3 breaks datetime-to-OHLCV identity"},
        {"gate_id": "pipeline_control_walk_forward_validation", "module": "PipelineControlWalkForwardValidationRun", "blocks": "development candidates when purged temporal folds, predictive quality, train-validation gap, feature stability, or positive-rate stability fail; its artifacts can never substitute for locked test evidence"},
        {"gate_id": "stage1_source_cadence", "module": "YFCollector source gate + TimeframeLineage", "blocks": "cache or database writes when requested timeframe conflicts with observed source cadence, exact row identities duplicate, timestamps are unresolved, or intervals are unsupported"},
        {"gate_id": "pipeline_feature_timeframe_lineage", "module": "PipelineFeatureTimeframeAudit + TimeframeLineage", "blocks": "Stage 3, Stage 4, or Stage 5 reuse when declared timeframe conflicts with observed cadence, datetime timezone is unresolved, or the candidate Stage 5 artifact is merely co-located rather than hash-bound to its feature parent"},
        {"gate_id": "colab_batch_exact_identity", "module": "ColabManager", "blocks": "feature/target batch writes or accumulation when ticker, datetime, interval, timezone, cadence, or exact SHA lineage is missing; deduplication includes interval and cannot collapse 15m/60m/1d rows"},
        {"gate_id": "pipeline_target_readiness", "module": "PipelineTargetReadinessAudit + TargetTimeframeContract", "blocks": "Stage 4 when targets are not registry-bound, do not apply to the source timeframe, lack per-ticker non-null coverage, have one-class classification labels, or do not hash-bind to the exact feature batch"},
        {"gate_id": "pipeline_stage4_exact_context_review", "module": "PipelineStage4ExactContextReview + PipelineWalkForwardValidationEvaluator", "blocks": "Stage 5 when exact feature/target/audit hashes, ticker/timeframe/target identity, three purged folds, predictive quality, train-validation gap, feature stability, positive-rate stability, or majority-baseline checks fail; review models are never persisted or promoted"},
        {"gate_id": "active_stage4_training_contract", "module": "ModelingStage + BaseTrainer", "blocks": "normal model selection from receiving nested/incomplete data, consuming the reserved holdout, overwriting candidate models under one champion filename, or emitting complete-looking evidence with unavailable train/risk/importance fields"},
        {"gate_id": "active_stage5_output_contract", "module": "PredictionStage + PredictionTargetSemanticsRegistry + PipelinePredictionReviewPacket", "blocks": "Stage 5 predict outputs from being treated as probabilities or directional evidence when target type, output scale, adjustment path, target identity, or contract validation is missing"},
        {"gate_id": "shadow_calibration_case_index", "module": "ShadowCalibrationCaseIndexBuilder", "blocks": "prediction or agent assessments from becoming calibration cases unless exact ticker/timeframe/target/context, exact realization timestamp, validated output scale, and immutable pipeline/prediction/outcome hashes all match"},
        {"gate_id": "shadow_component_case_production", "module": "ShadowComponentCaseProducer + ShadowCalibrationReadinessPacket", "blocks": "regime, specialist, or synthesis reports from counting toward calibration when their exact context, assessment as-of, component-specific eligibility, immutable source hash, or common diagnostic context is missing"},
        {"gate_id": "shadow_calibration_diagnostics", "module": "ShadowCalibrationDiagnostics", "blocks": "metrics from being computed across unaligned outcome episodes, duplicate component records, insufficient exact-context samples, or undeclared probability/label semantics; unavailable metrics remain explicit rather than approximated"},
        {"gate_id": "active_stage6_execution_boundary", "module": "FinalStagesOrchestrator + TradingExecutionStage + Trader", "blocks": "ordinary prediction/evaluation runs from mutating a virtual portfolio, writing the decision diary, executing paper orders, initializing live mode, or bypassing the separate reviewed paper-simulation workflow"},
        {"gate_id": "isolated_paper_lifecycle_lineage", "module": "ReviewDecisionReceipt + PaperSimulationPlan + PaperSimulationResult + PostPaperSimulationReview", "blocks": "paper lifecycle movement unless one unexpired human receipt is bound to an unchanged post-dry-run review, the plan binds receipt/source SHA and fingerprints, and the recorded result matches one immutable isolated-executor manifest; every layer remains non-live and post-paper output returns only to human review"},
        {"gate_id": "active_stage7_learning_boundary", "module": "EvaluationStage", "blocks": "caller-supplied trading activity from triggering automatic real-time adaptation or external notifications without an explicit per-run authorization"},
        {"gate_id": "active_stage7_analyzer_boundary", "module": "EvaluationStage + UnifiedAnalyticsEngine", "blocks": "mixed ticker/timeframe price series, missing analyzer inputs, individual analyzer failures, disabled modules, or stale suite caches from silently becoming locked evidence or execution authority"},
        {"gate_id": "composite_domain_agent_execution", "module": "AgentRegistry + PipelineManagerAgent + DomainAnalystAgent + PipelineReadiness", "blocks": "duplicate composite and standalone domain analysis in one execution group/phase, agents running outside configured phases, expensive analysis without timezone-aware as-of or source artifacts, and review readiness changing consensus or trade authority"},
        {"gate_id": "dean_orchestrator_two_phase_review", "module": "DEANOrchestrator + AgentRegistry + ConsensusEngine", "blocks": "pipeline execution when preflight hard prerequisites are missing, rechecks data/risk agents after pipeline outputs exist, and maps high scores to watchlist rather than execution candidates"},
        {"gate_id": "pipeline_adapter_review_contract", "module": "HybridPipelineAdapter + RiskAgent", "blocks": "DEAN context from losing Stage 4/7 artifact paths and safety statuses inside arbitrary nesting, prefers realized returns over supervised targets, and blocks target-label returns from pre-trade risk evidence"},
        {"gate_id": "pipeline_prediction_supporting_context", "module": "PipelinePredictionReviewPacket + SectorThesisToTickerBasketBridge + SectorToTickerReviewPacket", "blocks": "incomplete or quarantined Stage 5 outputs, sector stance, company evidence, or mismatched ticker model cases from changing predictions, filling lineage, clearing evaluation, or creating forecast authority; the bridge consumes a base immutable prediction review without sector overlay, and exact model cases align only on ticker/model/target/timeframe/context fingerprint"},
        {"gate_id": "model_performance_metric_source", "module": "ModelPerformanceAgent", "blocks": "arbitrary analyzer scores, row counts, unlocked complete-looking JSON, assembly-time timestamps, or unrelated nested fields from becoming model-performance evidence; a locked artifact is still caution when its full real-metric chain is blocked"},
        {"gate_id": "pipeline_control_forward_data_accrual", "module": "PipelineControlForwardDataAccrualPlan", "blocks": "calling already-seen data new, calling development refresh data a virgin holdout, or registering a boundary from a walk-forward artifact that touched test/past-evaluation rows"},
        {"gate_id": "pipeline_control_forward_data_accrual_intake", "module": "PipelineControlForwardDataAccrualGate", "blocks": "pre-registration files, seen source hashes, target-contaminated price inputs, insufficient post-watermark rows, invalid OHLCV, extreme returns, cadence mismatch, duplicate identities, or cross-ticker OHLCV copies"},
        {"gate_id": "pipeline_control_real_metric_evidence", "module": "PipelineControlRealMetricEvidenceRun", "blocks": "synthetic fixtures or mismatched model/feature lineage from being treated as one locked metric evidence pair"},
        {"gate_id": "pipeline_model_case_review", "module": "PipelineModelCasePacket + ModelPerformanceAgent + ReviewIndexBuilder + ChiefReviewIndexBuilder", "blocks": "stale or mismatched model/feature/chain/readiness snapshots from becoming a case, preserves evaluation blocks without writing learning memory, and scopes the block to the affected model candidate"},
        {"gate_id": "pipeline_model_feedback_review", "module": "PipelineModelFeedbackPacket + ReviewFeedbackTaxonomy", "blocks": "domain forecast labels, unsafe apply/config/tuning requests, stale case bindings, or unsupported incident claims from becoming model learning candidates; all accepted candidates remain proposal-only"},
        {"gate_id": "review_only_automation", "module": "DeanOSReviewOnlyAutomationRun", "blocks": "routine review refresh from silently starting collectors, training, evaluation, replay, tuning, learning writes, recommendations, or trading"},
        {"gate_id": "fundamental_input_readiness", "module": "FundamentalInputReadinessGate + ValueScreeningAgent", "blocks": "bad caller-supplied metrics, missing gate as-of, incomplete point-in-time semantics, or a mismatch between the gate accepted fingerprint and the current context fingerprint"},
        {"gate_id": "review_decision", "module": "ReviewDecisionPacket/ReviewActionDryRun", "blocks": "unreviewed downstream movement"},
        {"gate_id": "build_focus_review", "module": "BuildFocusReviewPacket", "blocks": "unproductive deepening when a branch is waiting for manual review, outcome data, or another branch has concrete blockers"},
    ]


def _next_safe_steps() -> list[str]:
    return [
        "Use this architecture map as the current replacement for stale system_audit_summary.py.",
        "Treat this map and Agents_architecture.md as revisable design context; verify the active Stage 2/3/4 code path before implementing.",
        "Use PipelineManagerAgent as the canonical composite domain path when artifact loading, SectorAnalyst reasoning, and pipeline readiness belong to one run. DomainAnalystAgent is the standalone alternative for a populated MarketContext; never enable both in the same execution group and phase.",
        "Keep transferred domain agents review-only: decision_influence=false, can_trade=false, and readiness evidence separate from sector thesis evidence.",
        "Keep standalone domain analysts, the composite manager, model-performance review, tuning, chief review, paper portfolio, diary, source-routing, and operations proposal agents disabled in the default registry. Enable one bounded workflow explicitly after its inputs are verified.",
        "Use the same execution_group for the standalone and composite implementation of one domain. Different names must never bypass the overlap guard.",
        "A standalone DomainAnalystAgent may run only from a populated MarketContext or a runtime artifact that passes contract, source-hash, domain, safety, and exact as-of verification. Missing or rejected inputs must fail cheaply.",
        "When cloning SectorAnalyst, bind one private overridden profile to the wrapper, evidence adapter, BaseAnalystAgent, and lens configuration; changing only the outer profile creates a misleading partial clone.",
        "Preserve the integrated backward-only timeframe context contract and per-(ticker, interval) Stage 4 isolation. Never default an absent interval to daily or strip timezone information.",
        "Use the bounded regenerated semiconductor batch for the next exact-context Stage 4 check: 1,170 rows across AMD/INTC/NVDA/TSM, explicit 15m, UTC, exact feature/target SHA bindings, and seven of seven target contracts ready. It does not replace the legacy main_database batch.",
        "Run BuildFocusReviewPacket when the next implementation step is ambiguous or a branch feels over-deepened.",
        "Use the hash-bound semiconductor runtime as the active source-first vertical slice: five required sector lanes are covered, but the result remains partial_ready_for_review and sector-only.",
        "Run AnalystKnowledgeReadiness for the requested as-of before using a transferred knowledge pack; only point-in-time eligible items may feed WorkingDomainAnalyst review.",
        "Keep knowledge retrieval as interpretive review context: it cannot satisfy the raw-source gate, become a Stage5 feature, or bypass exact ticker/timeframe/as-of specialist review.",
        "Use the pipeline adapter news quarantine or ContextEvidenceReviewPacket before analytical agents consume MarketContext news; missing as-of, future rows, missing timestamps/locators, and duplicates must remain excluded.",
        "Require explicit ticker metadata or a cashtag for context-news ticker directness. A company name or ticker-like substring in prose remains sector/domain context.",
        "Preserve explicit Agent Lab as-of: historical replay must pass its historical cutoff, while ordinary isolated analysis records its run cutoff. Direct keyword/material news paths use the same quarantine contract.",
        "Audit research documents by publication time, ingestion time, stable locator, content hash, duplicates, and explicit historical-reconstruction basis before material_documents or context evidence uses them.",
        "Require every structured fundamental, macro, or sector observation to declare value, unit, period, availability timestamp, and source locator; keep raw macro frames and document-count inventories outside evidence fields.",
        "Run FundamentalInputReadinessGate with the same explicit as-of and exact metric payload supplied to Agent Lab; ValueScreening requires matching accepted fingerprints.",
        "Keep dean_os package imports lazy so review utilities and provenance tests do not initialize the trading pipeline or optional analytics stack.",
        "Use SavedMacroEvidenceProducer on saved long-form FRED snapshots before macro context enters Agent Lab. Reverify source SHA, registry SHA, as-of, and fragment fingerprint at load time.",
        "Manually confirm the initial 27-series unit registry before treating the mapping as stable production semantics; current output remains review-only.",
        "Follow JULY_2026_BUILD_ROADMAP.md: finish real producers before adding agents, then build one exact-context saved-data case, prospective outcomes, isolated paper execution, and operational recovery.",
        "Preserve semantic macro keys from the registry, but never infer easing, tightening, inflation direction, or a trade stance from series presence alone.",
        "Use the verified saved SEC index, immutable filing HTML, Company Facts, and inline XBRL chain for source-bound facts. Derived ratios are review context only and must preserve exact period, unit, and filing lineage.",
        "Treat AMD artifacts as one technical smoke/model case only. Semiconductor domain analysis uses the value-chain profile; the active pipeline cohort is NVDA/AMD/INTC/TSM and every ticker pipeline result remains separate.",
        "Use the four-ticker semiconductor filing set as source coverage only: AMD, INTC, NVDA, and TSM are represented, while full-cohort period/currency comparability remains blocked.",
        "Require exact evaluation lineage before any tuning proposal. A failure for AMD/random_forest/target_intraday_up_15m/15m cannot tune NVDA, INTC, TSM, or a semiconductor-wide configuration.",
        "Manually review DomainAnalystInstanceContract before treating one domain analyst template as standardized.",
        "Use DomainAnalystThesisReviewPacket directly on the current semiconductor runtime before any sector-to-ticker bridge or domain scaling; it verifies linked source hashes and exposes market, source, policy, ratio, scenario, and quality cautions.",
        "Run AnalystCoreReasoningSnapshot after the semiconductor runtime and before thesis review. The verified path classifies each evidence item exactly once, aggregates regime signals, maps causal transmission channels, and creates explicitly heuristic candidate hypotheses and evidence gaps; expectation-gap probabilities, static analogs, and scenario generation remain excluded.",
        "Do not force a directional forecast from the current mixed sector thesis. Register it prospectively and add an explicit forecast packet only when falsifiable direction and scoring criteria are defined.",
        "Use the rebuilt template-standardization packet as a manual-acceptance candidate only. It now carries the verified reasoning context and 30/90/180-day self-check horizons, but it does not mark the template accepted or enable cloning.",
        "Build DomainAnalystTemplateDecisionPacket to record pending/accept/reject/needs-revision for the reusable analyst process only.",
        "Build DomainAnalystProfilePolicyPacket to verify source, ingestion, scoring, review-output, and feedback policies before clone planning.",
        "Build DomainAnalystEventInterpretationPacket from the real evidence pack, optionally with saved pipeline-context JSON, so news/data analysis keeps context slices, pipeline news/crisis taxonomy, event mechanisms, counterforces, watch metrics, and evidence gaps.",
        "Build DomainAnalystRegimeScenarioPacket after event interpretation so daily news and pipeline context become a regime vector, news-vs-regime checks, scenario graph, evidence gaps, and self-check horizons.",
        "Use DomainAnalystCaseRegistryPacket to freeze the current runtime-linked sector case before outcomes, including thesis and reasoning snapshot hashes, aggregated regime context, transmission channels, candidate hypotheses with invalidation signals, cautions, and 30/90/180-day review horizons. A stale template binding must fail.",
        "Use the current SectorThesisToTickerBasketBridge in runtime-linked mode to attach sector context as supporting-only context to exact ticker/model/target/timeframe identities. Its current result is a readiness-gap map: zero direct ticker theses, one negative AMD model case, and 389 real saved Stage 5 contexts quarantined because none has complete review lineage and semantics.",
        "Treat the real composite semiconductor smoke as wiring proof only: 152 hash-verified sector evidence items reached SectorAnalyst, while pipeline readiness correctly remained blocked and had zero consensus or trade influence.",
        "Use SavedTickerSpecificEvidenceProducer to map only reviewed exact issuer aliases and require two independent strong sources with a consistent mechanism stance. Current saved evidence corroborates AMD AI-demand/guidance only; INTC, NVDA, and TSM still need another independent strong source.",
        "Materialize the saved Stage 5 result twice in an acyclic chain: first a base immutable prediction-source review without sector overlay, then the sector-to-ticker bridge and review, then a final supporting-context overlay. Neither review may change the scalar prediction, fill lineage, clear evaluation, or create forecast authority.",
        "Build DomainAnalystFeedbackLoopPacket so manual review labels become proposal-only learning candidates without writing learning memory.",
        "Use DomainAnalystVerticalSliceRun as the single analyst-branch runner before manual template acceptance.",
        "Run DomainAnalystPortabilityReview before cloning the accepted analyst template into another domain.",
        "Add one operator source file to docs/research before exercising the real-source normalized packet path.",
        "Run PipelineMetricInputReadinessGate before refreshing PipelineControlSurface; no live tuning.",
        "Run PipelineControlSurface only with saved model/replay/feature/data-quality artifacts; no live tuning.",
        "Review PipelineControlInstanceContract before allowing proposal-only tuning experiments.",
        "Run PipelineControlCautionReviewPacket when the instance is review-ready with cautions; do not clear caution planes with audit reports or clean lineage alone.",
        "Run PipelineControlEvidenceInventory before PipelineControlRealMetricEvidenceRun when artifact status is unclear; it must not train, replay, synthesize, or write config.",
        "Run PipelineControlMetricArtifactMaterializer after inventory to normalize a complete locked artifact pair, or produce a gap report without writing fake evidence.",
        "Build PipelineModelCasePacket after the real-metric chain so blocked plane reasons, lineage, source hashes, and proposal-only regression checks reach ModelPerformanceAgent and Chief Review without a learning write.",
        "Keep the current AMD case as ticker/model negative evaluation evidence only, not a semiconductor-domain thesis or forecast miss; wait for accepted forward development data before another iteration of that model candidate while unrelated system work continues.",
        "Leave PipelineModelFeedbackPacket pending until real human feedback is supplied; do not fabricate an operator label or route the case into ReviewApprovedLearningLoop.",
        "Require matching ticker, model, target_name, timeframe, and context_fingerprint lineage across the locked model-evaluation and feature-stability artifacts.",
        "Run PipelineControlLockedEvaluationAssembler only when Stage 4 training candidates and Stage 7 evaluation candidates include same-window model/target/context/window lineage.",
        "Run PipelineControlLockedFeatureStabilityAssembler only when a Stage 4 feature-stability candidate includes feature importances, measured split-distribution stability signal, and model lineage.",
        "Use the training-stage pipeline_control_metric_artifacts_manifest.json as the discovery bridge for future real light-model train/validation/sample/importances/stability/window candidates.",
        "Use the Stage 7 pipeline_control_evaluation_metric_artifacts_manifest.json as supporting drawdown/return/sharpe evidence unless a locked assembler proves single-context same-window lineage.",
        "Use the Stage 4 train/validation split-drift hook for measured feature stability only when all selected features have enough finite split coverage; do not let partial candidates clear risk or stability cautions by themselves.",
        "Use the Stage 4 validation/test split index as training-side evaluation_window lineage only when the split frame exposes a real index; do not invent a window.",
        "Use enriched prediction/Stage 7 lineage fields only when ticker, model, target, timeframe, context fingerprint, and evaluation window are unambiguous; multi-context evaluations remain supporting-only.",
        "Use PipelineControlMetricFixtureValidation only as a synthetic contract check; never as model evidence.",
        "Require PipelineControlFeatureCausalityAudit to pass before rebuilding locked evidence after any Stage 3 feature or row-identity change.",
        "Treat the pre-causality bounded batch, train/validation diagnostic, and feature-selection experiment as superseded evidence; do not compare new variants against their invalid metrics.",
        "Use the corrected four-context bounded batch as a negative baseline only: all contexts remain blocked and no caution is cleared.",
        "Keep the current hash-bound NVDA/15m/target_intraday_up_15m Stage 4 review blocked: 587 exact rows and three temporal folds passed cadence, temporal, predictive-quality, and feature-stability checks, but train-validation gap, positive-rate stability, and majority-baseline checks failed. No model was persisted.",
        "Do not launch model or feature variants on the same walk-forward folds; accumulate genuinely new forward observations before defining a virgin locked holdout.",
        "Treat walk-forward artifacts as development-only supporting evidence in inventory/review automation, never as locked model evaluation or feature-stability inputs.",
        "Use the registered forward-development accrual boundary to require a new immutable source hash acquired after registration and observations strictly after the last used validation timestamp.",
        "Do not call accrued rows a virgin holdout; a passing frozen development candidate and a separate future holdout-registration gate are still required.",
        "The current June 25 price artifact is blocked by the accrual gate because it predates registration and still contains extreme-return and cross-ticker-copy corruption.",
        "Do not invoke Stage 3 or walk-forward until a post-registration immutable artifact clears every forward-data accrual gate check.",
        "When the accrual gate passes, supply its JSON through the walk-forward runner's forward_accrual_gate_json seam; the runner rechecks mode, artifact class, context, SHA, watermark, row count, and development-only flags.",
        "The active normal Stage 4 path now adapts nested prepared data to the trainer contract, uses validation for model selection, keeps the prepared holdout reserved, persists each model separately, and promotes only the actual winner.",
        "Treat active Stage 4 artifacts as partial until train score, native importance, and same-window Stage 7 drawdown exist; Stage 5 now carries target/model/timeframe/context lineage into Stage 7.",
        "Use Stage 5 -> Stage 7 as the normal review/evaluation path. Stage 6 is excluded by default and remains review-only even when explicitly requested.",
        "Use PipelinePredictionReviewPacket to expose Stage 5 prediction lineage, scalar forecast, confidence, and anomaly by exact context; it is supporting context, not evaluation, outcome, recommendation, or trading authority.",
        "The current saved Stage 5 source is real and immutable-bound but not trustworthy for forecast use: 389/389 selected semiconductor contexts are quarantined for missing timeframe, missing prediction as-of, placeholder context fingerprint, and incomplete target/model-output semantics. Preserve it as diagnostic evidence and regenerate through the repaired Stage 4/5 lineage path.",
        "The legacy main_database features.parquet remains blocked for the semiconductor cohort: all four requested tickers declare 1d while observed cadence is 15m, and none has timezone-aware datetimes. Do not patch labels onto that artifact.",
        "The saved Stage 1 artifact also proves that its 60m and 1d labels carry intraday cadence. Only the exact 15m lane was reused; source gates now reject mislabeled cache/database writes.",
        "Colab accumulation must deduplicate by ticker, datetime, and interval. The old ticker/datetime-only rule collapsed multiple timeframes and let the last 1d label overwrite intraday identities.",
        "Do not run Stage 5 from the current NVDA candidate. PipelineManagerAgent now consumes the exact Stage 4 review as separate readiness evidence and must surface stage4_validation_contract_failed without changing the sector thesis.",
        "Shard and cache Stage 3 by exact ticker/timeframe/source hash before another wider sector regeneration. The bounded single-ticker 600-row path completed, while the four-ticker form exceeded five minutes; do not recompute unchanged feature shards.",
        "Do not use active Stage 6 as a paper executor. Paper simulation stays in the receipt -> plan -> isolated external executor -> result review workflow; live Trader initialization is rejected.",
        "Require a timezone-aware receipt expiry and unchanged source/receipt/plan/external-result hashes across the isolated paper lifecycle. Missing external executor evidence must remain blocked, not recorded as completion.",
        "Treat Stage 7 trade outcomes as a proposal-only learning-review candidate. Stage 7 must not adapt weights, risk parameters, learning memory, or production config.",
        "Keep Stage 7 Telegram/Discord delivery disabled unless the individual final-stage request explicitly authorizes evaluation notification.",
        "Treat Stage 7 analyzer output as supporting review context only. Keep ticker/timeframe price histories partitioned and inspect the coverage record before using any analyzer result.",
        "Use Stage7 regime output through the exact ticker/timeframe shadow RegimeAgent bridge only; it has zero consensus, promotion, learning, recommendation, or trading influence.",
        "Use ContextSynthesisAgent only to compare exact Stage 5/Stage 7 identity, lineage quality, anomaly/confidence, and as-of compatibility; directional synthesis and consensus influence remain disabled.",
        "Attach specialist context to synthesis only through an explicit canonical packet in MarketContext metadata; never configure a fixed AMD/semiconductor artifact globally for other tickers.",
        "Keep semiconductor domain thesis, sector basket candidates, direct-ticker manual-review candidates, and approved ticker theses as distinct evidence scopes.",
        "The current AMD specialist packet is a direct-ticker manual-review candidate but is older than the 30-day review window, has no declared 15m alignment, is not approved, and has no decision influence.",
        "Treat the empty dean os1 folder as a completed transfer marker: its workbench modules now live inside active dean_os. Reuse them selectively, keep system_audit_summary historical, and do not overwrite newer modules from the older adjacent zip.",
        "Harvest template-kit ideas selectively: target unit/period/class completeness, zero time leakage, zero unsafe output, zero sector-to-ticker leakage, immutable outcome hashes, and human review before weights.",
        "Bind Stage 5 target names to canonical targets.yaml and TargetTimeframeContract semantics, require the explicit model-output contract in prediction review, and keep directional use blocked until realized-outcome calibration exists.",
        "Use ShadowCalibrationCaseIndexBuilder only with a saved source-bound Stage 5 review and an immutable outcome file containing one exact ticker/timeframe row at the expected realization timestamp; later rows may exist but cannot be selected.",
        "Chain ShadowComponentCaseProducer separately for regime, specialist, and context_synthesis. Regime/synthesis evidence later than prediction_as_of is rejected; specialist evidence must be exact, aligned, point-in-time compatible, and manually cleared.",
        "Count diagnostic readiness per exact ticker/timeframe/target/context fingerprint and require all four component families to meet the threshold on at least one common context.",
        "Compute shadow diagnostics only over common prediction outcome episode IDs. Use verified raw class labels for label metrics, explicit positive-class probabilities for Brier/log-loss, and mark drawdown/conflict/human-disagreement metrics unavailable until their required evidence exists.",
        "Keep all shadow components ineligible for consensus weight until the outcome-bound case index reaches the predeclared diagnostic policy and passes zero-leakage safety checks.",
        "Preserve the Stage 7 price-window start/end/row count so regime freshness can be compared rather than assumed.",
        "Use AgentCapabilityMatrix as observability for registry inputs, effects, activation, and decision influence; it is not another gate.",
        "Enable another analyzer only after its constructor, input, point-in-time provenance, side-effect, and output contracts are tested; disabled catalog entries are not active capabilities.",
        "Run hard-veto pipeline agents twice in DEANOrchestrator: preflight before the pipeline and pre-trade review after pipeline outputs and analytical context exist.",
        "Keep active consensus watchlist-only. High positive/negative scores are review priorities, not candidate_long/candidate_short execution decisions.",
        "Read canonical Stage 4/7 artifact paths and execution/learning statuses from context.metadata.pipeline_review_contract; still pass those artifacts through evidence inventory before promotion review.",
        "Never use target_return_* labels as pre-trade drawdown or VaR inputs. Realized return columns or causal close-price changes take precedence.",
        "ModelPerformanceAgent may reference analyzer coverage as supporting context, but pipeline metrics must come from evaluation_summary.metrics and include validation score, Sharpe, drawdown, sample count, and timestamp before a clear verdict.",
        "Use the evaluation-window end, not artifact assembly time, as the model-performance as-of timestamp; preserve source SHA and verified locked provenance through materialization.",
        "The current verified AMD/15m metric pair is usable as evidence but the real chain remains blocked: train-validation gap 0.3135 exceeds 0.15 and feature stability 0.5987 is below 0.70.",
        "Use PipelineControlRealMetricEvidenceRun when saved locked model-evaluation and feature-stability artifacts are available.",
        "Use DeanOSReviewOnlyAutomationRun for one-command review refresh; it may consume saved artifacts but must not start collectors, training, Stage 7 evaluation, replay, tuning, learning writes, recommendations, or trading.",
        "Standardize one domain analyst template before enabling or cloning additional sector analysts.",
    ]


def _explicit_non_actions() -> list[str]:
    return [
        "No live collector is started.",
        "No autonomous tuning run is started.",
        "No model training, model promotion, or production config write is performed.",
        "No learning memory write is performed.",
        "No execution recommendation, allocation, paper order, broker call, or live trade is generated.",
        "No additional domain analyst is enabled by this map.",
    ]


def _commands() -> list[dict[str, str]]:
    return [
        {
            "command_id": "analyst_knowledge_readiness",
            "command": (
                "python run_agent_analyst_knowledge_readiness.py "
                "--as-of <ISO-8601-timezone-aware-timestamp> "
                "--store-dir data\\dean_os\\analyst_knowledge "
                "--output-dir reports\\dean_os\\"
                "analyst_knowledge_readiness_current"
            ),
        },
        {
            "command_id": "paper_simulation_plan",
            "command": (
                "python run_agent_paper_simulation_plan.py "
                "<explicit_unexpired_paper_only_receipt.json> "
                "--output-dir reports\\dean_os\\"
                "paper_simulation_plans_current"
            ),
        },
        {
            "command_id": "context_evidence_review",
            "command": (
                "python run_agent_context_evidence_review.py "
                "<saved_market_context.json> "
                "--domain-id semiconductor_ai_infrastructure "
                "--as-of <ISO-8601-timezone-aware-timestamp> "
                "--output-dir reports\\dean_os\\"
                "context_evidence_review_current"
            ),
        },
        {
            "command_id": "paper_simulation_result",
            "command": (
                "python run_agent_paper_simulation_result.py "
                "<hash_bound_paper_plan.json> "
                "<immutable_isolated_executor_output.json> "
                "--output-dir reports\\dean_os\\"
                "paper_simulation_results_current"
            ),
        },
        {
            "command_id": "post_paper_simulation_review",
            "command": (
                "python run_agent_post_paper_simulation_review.py "
                "<hash_bound_paper_result.json> "
                "--output-dir reports\\dean_os\\"
                "post_paper_simulation_review_current"
            ),
        },
        {
            "command_id": "domain_analyst_vertical_slice_run",
            "command": (
                "python run_agent_domain_analyst_vertical_slice.py "
                "--evidence-pack-json reports\\dean_os\\analyst_evidence_pack_semiconductor_sector_only_strict_current\\latest.json "
                "--pipeline-context-json <optional_saved_pipeline_context.json> "
                "--output-dir reports\\dean_os\\domain_analyst_vertical_slice_current"
            ),
        },
        {
            "command_id": "domain_analyst_profile_policy_packet",
            "command": (
                "python run_agent_domain_analyst_profile_policy_packet.py "
                "--output-dir reports\\dean_os\\domain_analyst_profile_policy_packet_current"
            ),
        },
        {
            "command_id": "domain_analyst_event_interpretation_packet",
            "command": (
                "python run_agent_domain_analyst_event_interpretation_packet.py "
                "--evidence-pack-json reports\\dean_os\\analyst_evidence_pack_semiconductor_sector_only_strict_current\\latest.json "
                "--pipeline-context-json <optional_saved_pipeline_context.json> "
                "--domain-id semiconductor_ai_infrastructure "
                "--output-dir reports\\dean_os\\domain_analyst_event_interpretation_packet_current"
            ),
        },
        {
            "command_id": "domain_analyst_regime_scenario_packet",
            "command": (
                "python run_agent_domain_analyst_regime_scenario_packet.py "
                "--event-interpretation-json reports\\dean_os\\domain_analyst_event_interpretation_packet_current\\latest.json "
                "--domain-id semiconductor_ai_infrastructure "
                "--output-dir reports\\dean_os\\domain_analyst_regime_scenario_packet_current"
            ),
        },
        {
            "command_id": "domain_analyst_portability_review",
            "command": (
                "python run_agent_domain_analyst_portability_review.py "
                "--vertical-slice-json reports\\dean_os\\domain_analyst_vertical_slice_current\\latest.json "
                "--architecture-map-json reports\\dean_os\\current_architecture_map_current\\latest.json "
                "--output-dir reports\\dean_os\\domain_analyst_portability_review_current"
            ),
        },
        {
            "command_id": "semiconductor_analyst_runtime",
            "command": (
                "python run_agent_semiconductor_analyst.py "
                "--fundamental-artifact reports\\dean_os\\saved_sec_fundamental_evidence_merger_current\\latest.json "
                "--macro-artifact reports\\dean_os\\saved_macro_evidence_producer_current\\latest.json "
                "--sector-market-artifact reports\\dean_os\\saved_sector_market_evidence_producer_current\\latest.json "
                "--news-artifact reports\\dean_os\\saved_semiconductor_news_evidence_producer_current\\latest.json "
                "--official-policy-artifact reports\\dean_os\\saved_official_policy_evidence_producer_current\\latest.json "
                "--derived-ratio-artifact reports\\dean_os\\saved_sec_derived_ratio_producer_current\\latest.json "
                "--as-of <ISO-8601-timezone-aware-timestamp> "
                "--tickers NVDA AMD INTC TSM --horizon-days 180 "
                "--output-dir reports\\dean_os\\semiconductor_analyst_runtime_current"
            ),
        },
        {
            "command_id": "analyst_core_reasoning_snapshot",
            "command": (
                "python run_agent_analyst_core_reasoning_snapshot.py "
                "--runtime-json reports\\dean_os\\semiconductor_analyst_runtime_current\\latest.json "
                "--output-dir reports\\dean_os\\analyst_core_reasoning_snapshot_current"
            ),
        },
        {
            "command_id": "domain_analyst_thesis_review_packet",
            "command": (
                "python run_agent_domain_analyst_thesis_review_packet.py "
                "--domain-intake-json reports\\dean_os\\semiconductor_analyst_runtime_current\\latest.json "
                "--reasoning-snapshot-json reports\\dean_os\\analyst_core_reasoning_snapshot_current\\latest.json "
                "--domain-instance-contract-json reports\\dean_os\\domain_analyst_instance_contract_current\\latest.json "
                "--architecture-map-json reports\\dean_os\\current_architecture_map_current\\latest.json "
                "--output-dir reports\\dean_os\\domain_analyst_thesis_review_packet_current"
            ),
        },
        {
            "command_id": "domain_analyst_forecast_review_packet",
            "command": (
                "python run_agent_domain_analyst_forecast_review_packet.py "
                "--domain-thesis-review-json reports\\dean_os\\domain_analyst_thesis_review_packet_current\\latest.json "
                "--regime-scenario-json reports\\dean_os\\domain_analyst_regime_scenario_packet_current\\latest.json "
                "--output-dir reports\\dean_os\\domain_analyst_forecast_review_packet_current"
            ),
        },
        {
            "command_id": "domain_analyst_template_standardization_packet",
            "command": (
                "python run_agent_domain_analyst_template_standardization_packet.py "
                "--domain-instance-contract-json reports\\dean_os\\domain_analyst_instance_contract_current\\latest.json "
                "--domain-thesis-review-json reports\\dean_os\\domain_analyst_thesis_review_packet_current\\latest.json "
                "--architecture-map-json reports\\dean_os\\current_architecture_map_current\\latest.json "
                "--output-dir reports\\dean_os\\domain_analyst_template_standardization_packet_current"
            ),
        },
        {
            "command_id": "domain_analyst_case_registry_packet",
            "command": (
                "python run_agent_domain_analyst_case_registry_packet.py "
                "--domain-thesis-review-json reports\\dean_os\\domain_analyst_thesis_review_packet_current\\latest.json "
                "--domain-template-standardization-json reports\\dean_os\\domain_analyst_template_standardization_packet_current\\latest.json "
                "--output-dir reports\\dean_os\\domain_analyst_case_registry_packet_current"
            ),
        },
        {
            "command_id": "saved_ticker_specific_evidence",
            "command": (
                "python run_agent_saved_ticker_specific_evidence.py "
                "reports\\dean_os\\saved_semiconductor_news_evidence_producer_current\\latest.json "
                "--as-of <ISO-8601-timezone-aware-timestamp> "
                "--tickers NVDA AMD INTC TSM "
                "--registry-path dean_os\\config\\semiconductor_issuer_identity_registry.yaml "
                "--output-dir reports\\dean_os\\saved_ticker_specific_evidence_producer_current"
            ),
        },
        {
            "command_id": "sector_to_exact_ticker_pipeline_bridge",
            "command": (
                "python run_agent_sector_to_ticker_bridge.py "
                "--domain-thesis-review-json reports\\dean_os\\domain_analyst_thesis_review_packet_current\\latest.json "
                "--ticker-evidence-json reports\\dean_os\\saved_ticker_specific_evidence_producer_current\\latest.json "
                "--prediction-review-json reports\\dean_os\\pipeline_prediction_source_review_current\\latest.json "
                "--feature-timeframe-audit-json reports\\dean_os\\pipeline_feature_timeframe_audit_current\\latest.json "
                "--pipeline-case-json reports\\dean_os\\pipeline_model_case_packet_current\\latest.json "
                "--output-dir reports\\dean_os\\sector_thesis_to_ticker_basket_current"
            ),
        },
        {
            "command_id": "sector_to_ticker_readiness_review",
            "command": (
                "python run_agent_sector_to_ticker_review_packet.py "
                "--bridge-json reports\\dean_os\\sector_thesis_to_ticker_basket_current\\latest.json "
                "--output-dir reports\\dean_os\\sector_to_ticker_review_packet_current"
            ),
        },
        {
            "command_id": "domain_analyst_feedback_loop_packet",
            "command": (
                "python run_agent_domain_analyst_feedback_loop_packet.py "
                "--case-registry-json reports\\dean_os\\domain_analyst_case_registry_packet_current\\latest.json "
                "--forecast-review-json reports\\dean_os\\domain_analyst_forecast_review_packet_current\\latest.json "
                "--profile-policy-json reports\\dean_os\\domain_analyst_profile_policy_packet_current\\latest.json "
                "--template-decision-json reports\\dean_os\\domain_analyst_template_decision_packet_current\\latest.json "
                "--output-dir reports\\dean_os\\domain_analyst_feedback_loop_packet_current"
            ),
        },
        {
            "command_id": "domain_analyst_template_decision_packet",
            "command": (
                "python run_agent_domain_analyst_template_decision_packet.py "
                "--vertical-slice-json reports\\dean_os\\domain_analyst_vertical_slice_current\\latest.json "
                "--template-standardization-json reports\\dean_os\\domain_analyst_template_standardization_packet_current\\latest.json "
                "--forecast-review-json reports\\dean_os\\domain_analyst_forecast_review_packet_current\\latest.json "
                "--case-registry-json reports\\dean_os\\domain_analyst_case_registry_packet_current\\latest.json "
                "--portability-review-json reports\\dean_os\\domain_analyst_portability_review_current\\latest.json "
                "--architecture-map-json reports\\dean_os\\current_architecture_map_current\\latest.json "
                "--decision pending_review "
                "--output-dir reports\\dean_os\\domain_analyst_template_decision_packet_current"
            ),
        },
        {
            "command_id": "pipeline_metric_input_readiness_gate",
            "command": (
                "python run_agent_pipeline_metric_input_readiness_gate.py "
                "--model-performance performance_data.json "
                "--replay-batch reports\\dean_os\\historical_replay_batch_repaired_expanded\\latest.json "
                "--data-quality diagnostic_reports\\feature_lineage_report_current_cache.json "
                "--output-dir reports\\dean_os\\pipeline_metric_input_readiness_gate_current"
            ),
        },
        {
            "command_id": "pipeline_control_caution_review_packet",
            "command": (
                "python run_agent_pipeline_control_caution_review_packet.py "
                "--pipeline-metric-input-readiness-json reports\\dean_os\\pipeline_metric_input_readiness_gate_current\\latest.json "
                "--pipeline-control-instance-json reports\\dean_os\\pipeline_control_instance_contract_current\\latest.json "
                "--model-performance-report-json reports\\dean_os\\model_performance\\smoke.json "
                "--data-quality-json diagnostic_reports\\feature_lineage_report_current_cache.json "
                "--output-dir reports\\dean_os\\pipeline_control_caution_review_packet_current"
            ),
        },
        {
            "command_id": "pipeline_control_evidence_inventory",
            "command": (
                "python run_agent_pipeline_control_evidence_inventory.py "
                "--output-dir reports\\dean_os\\pipeline_control_evidence_inventory_current"
            ),
        },
        {
            "command_id": "pipeline_control_metric_artifact_materializer",
            "command": (
                "python run_agent_pipeline_control_metric_artifact_materializer.py "
                "--output-dir reports\\dean_os\\pipeline_control_metric_artifact_materializer_current"
            ),
        },
        {
            "command_id": "pipeline_control_locked_evaluation_assembler",
            "command": (
                "python run_agent_pipeline_control_locked_evaluation_assembler.py "
                "--training-candidate-json <stage4_model_evaluation_candidate.json> "
                "--evaluation-candidate-json <stage7_evaluation_metric_candidate.json> "
                "--output-dir reports\\dean_os\\pipeline_control_locked_evaluation_assembler_current"
            ),
        },
        {
            "command_id": "pipeline_control_locked_feature_stability_assembler",
            "command": (
                "python run_agent_pipeline_control_locked_feature_stability_assembler.py "
                "--feature-stability-candidate-json <stage4_feature_stability_candidate.json> "
                "--output-dir reports\\dean_os\\pipeline_control_locked_feature_stability_assembler_current"
            ),
        },
        {
            "command_id": "pipeline_control_metric_fixture_validation",
            "command": (
                "python run_agent_pipeline_control_metric_fixture_validation.py "
                "--output-dir reports\\dean_os\\pipeline_control_metric_fixture_validation_current"
            ),
        },
        {
            "command_id": "pipeline_control_real_metric_evidence_run",
            "command": (
                "python run_agent_pipeline_control_real_metric_evidence_run.py "
                "--model-evaluation-json reports\\dean_os\\pipeline_control_metric_artifact_materializer_current\\model_evaluation\\latest.json "
                "--feature-stability-report reports\\dean_os\\pipeline_control_metric_artifact_materializer_current\\feature_stability\\latest.json "
                "--replay-batch-json reports\\dean_os\\historical_replay_batch_repaired_expanded\\latest.json "
                "--data-quality-json diagnostic_reports\\feature_lineage_report_current_cache.json "
                "--output-dir reports\\dean_os\\pipeline_control_real_metric_evidence_run_current"
            ),
        },
        {
            "command_id": "pipeline_model_case_packet",
            "command": (
                "python run_agent_pipeline_model_case_packet.py "
                "--real-metric-evidence-json reports\\dean_os\\pipeline_control_real_metric_evidence_run_current\\latest.json "
                "--model-evaluation-json reports\\dean_os\\pipeline_control_metric_artifact_materializer_current\\model_evaluation\\latest.json "
                "--feature-stability-json reports\\dean_os\\pipeline_control_metric_artifact_materializer_current\\feature_stability\\latest.json "
                "--output-dir reports\\dean_os\\pipeline_model_case_packet_current"
            ),
        },
        {
            "command_id": "pipeline_model_feedback_packet",
            "command": (
                "python run_agent_pipeline_model_feedback_packet.py "
                "--pipeline-model-case-json reports\\dean_os\\pipeline_model_case_packet_current\\latest.json "
                "--output-dir reports\\dean_os\\pipeline_model_feedback_packet_current"
            ),
        },
        {
            "command_id": "review_index",
            "command": (
                "python run_agent_review_index.py "
                "--output-dir reports\\dean_os\\review_index"
            ),
        },
        {
            "command_id": "chief_review_index",
            "command": (
                "python run_agent_chief_review_index.py "
                "--review-index-path reports\\dean_os\\review_index\\latest.json "
                "--output-dir reports\\dean_os\\chief_review_index"
            ),
        },
        {
            "command_id": "pipeline_control_data_preflight",
            "command": (
                "python run_agent_pipeline_control_data_preflight.py "
                "--output-dir reports\\dean_os\\pipeline_control_data_preflight_current"
            ),
        },
        {
            "command_id": "pipeline_control_saved_data_coverage",
            "command": (
                "python run_agent_pipeline_control_saved_data_coverage.py "
                "--output-dir reports\\dean_os\\pipeline_control_saved_data_coverage_current"
            ),
        },
        {
            "command_id": "pipeline_control_saved_price_repair",
            "command": (
                "python run_agent_pipeline_control_saved_price_repair.py "
                "--coverage-json reports\\dean_os\\pipeline_control_saved_data_coverage_current\\latest.json "
                "--output-dir reports\\dean_os\\pipeline_control_saved_price_repair_current"
            ),
        },
        {
            "command_id": "pipeline_control_bounded_evidence_run",
            "command": (
                "python run_agent_pipeline_control_bounded_evidence.py "
                "--source-path data\\processed\\prices_15m_20260625_125005.parquet "
                "--macro-source-path data\\processed\\features\\macro_data.parquet "
                "--ticker AMD --timeframe 15m --target-name target_intraday_up_15m "
                "--start 2026-05-28T16:45:00+03:00 --max-rows 480 --max-features 40 "
                "--output-dir reports\\dean_os\\pipeline_control_bounded_evidence_run_current"
            ),
        },
        {
            "command_id": "pipeline_control_bounded_evidence_batch",
            "command": (
                "python run_agent_pipeline_control_bounded_evidence_batch.py "
                "--coverage-json reports\\dean_os\\pipeline_control_saved_data_coverage_current\\latest.json "
                "--ticker NVDA --ticker INTC --ticker TSM --ticker SPY "
                "--frozen-context AMD/15m --rows-per-context 480 "
                "--output-dir reports\\dean_os\\pipeline_control_bounded_evidence_batch_current"
            ),
        },
        {
            "command_id": "pipeline_control_feature_causality_audit",
            "command": (
                "python run_agent_pipeline_control_feature_causality_audit.py "
                "--batch-json reports\\dean_os\\pipeline_control_bounded_evidence_batch_current\\latest.json "
                "--ticker NVDA --ticker SPY --max-contexts 2 "
                "--output-dir reports\\dean_os\\pipeline_control_feature_causality_audit_current"
            ),
        },
        {
            "command_id": "pipeline_control_walk_forward_validation",
            "command": (
                "python run_agent_pipeline_control_walk_forward_validation.py "
                "--historical-recovery-json reports\\dean_os\\pipeline_control_historical_price_recovery_current\\latest.json "
                "--ticker NVDA --timeframe 15m --target-name target_intraday_up_15m "
                "--macro-source-path data\\processed\\features\\macro_data.parquet "
                "--acknowledge-development-only --min-train-rows 360 --validation-rows 120 "
                "--step-rows 120 --purge-rows 5 --max-folds 4 --max-features 40 "
                "--output-dir reports\\dean_os\\pipeline_control_walk_forward_validation_current"
            ),
        },
        {
            "command_id": "pipeline_control_forward_data_accrual_plan",
            "command": (
                "python run_agent_pipeline_control_forward_data_accrual_plan.py "
                "--walk-forward-json reports\\dean_os\\pipeline_control_walk_forward_validation_current\\latest.json "
                "--acknowledge-development-refresh-only "
                "--output-dir reports\\dean_os\\pipeline_control_forward_data_accrual_plan_current"
            ),
        },
        {
            "command_id": "pipeline_control_forward_data_accrual_gate",
            "command": (
                "python run_agent_pipeline_control_forward_data_accrual_gate.py "
                "--accrual-plan-json reports\\dean_os\\pipeline_control_forward_data_accrual_plan_current\\latest.json "
                "--source-path <new_immutable_price_artifact.parquet> "
                "--output-dir reports\\dean_os\\pipeline_control_forward_data_accrual_gate_current"
            ),
        },
        {
            "command_id": "pipeline_control_walk_forward_forward_refresh",
            "command": (
                "python run_agent_pipeline_control_walk_forward_validation.py "
                "--historical-recovery-json reports\\dean_os\\pipeline_control_historical_price_recovery_current\\latest.json "
                "--forward-accrual-gate-json reports\\dean_os\\pipeline_control_forward_data_accrual_gate_current\\latest.json "
                "--ticker NVDA --timeframe 15m --target-name target_intraday_up_15m "
                "--macro-source-path data\\processed\\features\\macro_data.parquet "
                "--acknowledge-development-only "
                "--output-dir reports\\dean_os\\pipeline_control_walk_forward_validation_current"
            ),
        },
        {
            "command_id": "pipeline_control_train_validation_diagnostic",
            "command": (
                "python run_agent_pipeline_control_train_validation_diagnostic.py "
                "--batch-json reports\\dean_os\\pipeline_control_bounded_evidence_batch_current\\latest.json "
                "--output-dir reports\\dean_os\\pipeline_control_train_validation_diagnostic_current"
            ),
        },
        {
            "command_id": "dean_os_review_automation",
            "command": (
                "python run_agent_dean_os_review_automation.py "
                "--output-dir reports\\dean_os\\review_only_automation_run_current"
            ),
        },
        {
            "command_id": "shadow_calibration_readiness",
            "command": (
                "python run_agent_shadow_calibration_readiness.py"
            ),
        },
        {
            "command_id": "shadow_calibration_case_index",
            "command": (
                "python run_agent_shadow_calibration_case_index.py "
                "<saved_prediction_review.json> "
                "<immutable_outcome_source.csv_or_parquet_or_json> "
                "--output-dir reports\\dean_os\\"
                "shadow_calibration_case_index_current"
            ),
        },
        {
            "command_id": "shadow_component_case_producer",
            "command": (
                "python run_agent_shadow_component_case_producer.py "
                "<regime|specialist|context_synthesis> "
                "<base_case_index.json> <component_artifact.json> "
                "--output-dir reports\\dean_os\\"
                "shadow_calibration_case_index_current"
            ),
        },
        {
            "command_id": "shadow_calibration_diagnostics",
            "command": (
                "python run_agent_shadow_calibration_diagnostics.py "
                "<shadow_calibration_case_index.json> "
                "--output-dir reports\\dean_os\\"
                "shadow_calibration_diagnostics_current"
            ),
        },
        {
            "command_id": "specialist_context_review",
            "command": (
                "python run_agent_specialist_context_review.py "
                "--ticker AMD --timeframe 15m "
                "--as-of 2026-06-24T19:30:00+00:00 "
                "--output-dir reports\\dean_os\\"
                "specialist_context_review_amd_15m_current"
            ),
        },
        {
            "command_id": "pipeline_feature_timeframe_audit",
            "command": (
                "python run_agent_pipeline_feature_timeframe_audit.py "
                "data\\colab\\accumulated\\main_database\\features.parquet "
                "--stage5-json data\\colab\\accumulated\\main_database\\stage_5_results.json "
                "--ticker AMD --ticker INTC --ticker NVDA --ticker TSM "
                "--output-dir reports\\dean_os\\pipeline_feature_timeframe_audit_current"
            ),
        },
        {
            "command_id": "bounded_pipeline_stage23_regeneration",
            "command": (
                "python run_agent_pipeline_stage23_regeneration.py "
                "data\\colab\\accumulated\\main_database\\"
                "main_database_stage1_raw_data_20260629_195400.parquet "
                "--ticker AMD --ticker INTC --ticker NVDA --ticker TSM "
                "--timeframe 15m --max-rows-per-ticker 300 "
                "--batch-dir data\\colab\\regenerated\\"
                "semiconductor_15m_stage23_current "
                "--output-dir reports\\dean_os\\"
                "pipeline_stage23_regeneration_current"
            ),
        },
        {
            "command_id": "pipeline_target_readiness_audit",
            "command": (
                "python run_agent_pipeline_target_readiness_audit.py "
                "data\\colab\\regenerated\\"
                "semiconductor_15m_stage23_current\\targets.parquet "
                "--features-parquet data\\colab\\regenerated\\"
                "semiconductor_15m_stage23_current\\features.parquet "
                "--batch-metadata-json data\\colab\\regenerated\\"
                "semiconductor_15m_stage23_current\\batch_metadata.json "
                "--ticker AMD --ticker INTC --ticker NVDA --ticker TSM "
                "--timeframe 15m --output-dir reports\\dean_os\\"
                "pipeline_target_readiness_audit_current"
            ),
        },
        {
            "command_id": "pipeline_stage4_exact_context_review",
            "command": (
                "python run_agent_pipeline_stage4_exact_context_review.py "
                "--features-parquet data\\colab\\regenerated\\"
                "nvda_15m_stage23_review600\\features.parquet "
                "--targets-parquet data\\colab\\regenerated\\"
                "nvda_15m_stage23_review600\\targets.parquet "
                "--batch-metadata-json data\\colab\\regenerated\\"
                "nvda_15m_stage23_review600\\batch_metadata.json "
                "--feature-audit-json reports\\dean_os\\"
                "pipeline_stage23_regeneration_nvda_review600\\"
                "feature_timeframe_audit\\latest.json "
                "--target-audit-json reports\\dean_os\\"
                "pipeline_target_readiness_audit_nvda_review600\\latest.json "
                "--ticker NVDA --timeframe 15m "
                "--target-name target_intraday_up_15m "
                "--max-folds 3 --output-dir reports\\dean_os\\"
                "pipeline_stage4_exact_context_review_nvda_15m_review600"
            ),
        },
        {
            "command_id": "pipeline_prediction_source_review",
            "command": (
                "python "
                "run_agent_pipeline_prediction_review_packet.py "
                "data\\colab\\accumulated\\main_database\\"
                "stage_5_results.json "
                "--ticker AMD --ticker INTC --ticker NVDA "
                "--ticker TSM --filter-to-requested-scope "
                "--output-dir reports\\dean_os\\"
                "pipeline_prediction_source_review_current"
            ),
        },
        {
            "command_id": "pipeline_prediction_review_packet",
            "command": (
                "python "
                "run_agent_pipeline_prediction_review_packet.py "
                "data\\colab\\accumulated\\main_database\\"
                "stage_5_results.json "
                "--ticker AMD --ticker INTC --ticker NVDA "
                "--ticker TSM --filter-to-requested-scope "
                "--sector-to-ticker-review-json reports\\dean_os\\sector_to_ticker_review_packet_current\\latest.json "
                "--output-dir reports\\dean_os\\"
                "pipeline_prediction_review_packet_current"
            ),
        },
        {
            "command_id": "composite_semiconductor_domain_pipeline",
            "command": (
                "python run_agent_composite_domain_pipeline.py "
                "--domain-id semiconductor_ai_infrastructure "
                "--as-of 2026-06-30T21:00:00+00:00 "
                "--ticker NVDA --ticker AMD --ticker INTC --ticker TSM "
                "--timeframe 15m --horizon-days 180 "
                "--runtime-json reports\\dean_os\\semiconductor_analyst_runtime_current\\latest.json "
                "--feature-timeframe-audit-json reports\\dean_os\\pipeline_feature_timeframe_audit_current\\latest.json "
                "--prediction-review-json reports\\dean_os\\pipeline_prediction_source_review_current\\latest.json "
                "--sector-to-ticker-review-json reports\\dean_os\\sector_to_ticker_review_packet_current\\latest.json "
                "--output-dir reports\\dean_os\\sector_pipeline_manager_semiconductor_current"
            ),
        },
        {
            "command_id": "agent_capability_matrix",
            "command": (
                "python run_agent_capability_matrix.py "
                "--output-dir "
                "reports\\dean_os\\agent_capability_matrix_current"
            ),
        },
        {
            "command_id": "refresh_current_architecture_map",
            "command": "python run_agent_current_architecture_map.py --output-dir reports/dean_os/current_architecture_map_current",
        },
        {
            "command_id": "build_focus_review_packet",
            "command": (
                "python run_agent_build_focus_review_packet.py "
                "--alignment-review-json reports\\dean_os\\current_system_alignment_review_two_branch_current\\latest.json "
                "--template-standardization-json reports\\dean_os\\domain_analyst_template_standardization_packet_current\\latest.json "
                "--case-registry-json reports\\dean_os\\domain_analyst_case_registry_packet_current\\latest.json "
                "--pipeline-control-instance-json reports\\dean_os\\pipeline_control_instance_contract_current\\latest.json "
                "--output-dir reports\\dean_os\\build_focus_review_packet_current"
            ),
        },
        {
            "command_id": "refresh_current_alignment_review",
            "command": (
                "python run_agent_current_system_alignment_review.py "
                "--architecture-map-json reports/dean_os/current_architecture_map_current/latest.json "
                "--output-dir reports/dean_os/current_system_alignment_review_current"
            ),
        },
    ]


def _run_id(prefix: str) -> str:
    return f"{prefix}_{utc_now_iso().replace(':', '').replace('+', 'Z')}"
