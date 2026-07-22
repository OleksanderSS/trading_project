"""DEAN-OS: multi-agent governance layer for the trading pipeline.

The package root intentionally exposes the existing public API lazily. Importing
one small DEAN-OS module must not initialize the trading pipeline, configuration
stack, plotting libraries, or optional integrations.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORT_GROUPS: dict[str, tuple[str, ...]] = {
    "dean_os.draft.dean_os_agent_system_v7.dean_os.branches": ("AnalyticalBranch", "PipelineBranch"),
    "dean_os.packets.build_focus_review_packet": ("BuildFocusReviewPacket",),
    "dean_os.draft.dean_os_agent_system_v7.dean_os.consensus": ("ConsensusEngine",),
    "dean_os.draft.dean_os_agent_system_v7.dean_os.context_performance": ("AgentPerformanceByContext",),
    "dean_os.draft.dean_os_agent_system_v7.dean_os.current_architecture_map": ("CurrentArchitectureMap",),
    "dean_os.draft.dean_os_agent_system_v7.dean_os.current_system_alignment_review": (
        "CurrentSystemAlignmentReview",
    ),
    "dean_os.draft.dean_os_agent_system_v7.dean_os.decision_logger": ("DecisionLogger",),
    "dean_os.draft.dean_os_agent_system_v7.dean_os.domain_analyst_case_registry_packet": (
        "DomainAnalystCaseRegistryPacket",
    ),
    "dean_os.draft.dean_os_agent_system_v7.dean_os.domain_analyst_event_interpretation_packet": (
        "DomainAnalystEventInterpretationPacket",
    ),
    "dean_os.draft.dean_os_agent_system_v7.dean_os.domain_analyst_feedback_loop_packet": (
        "DomainAnalystFeedbackLoopPacket",
    ),
    "dean_os.draft.dean_os_agent_system_v7.dean_os.domain_analyst_forecast_review_packet": (
        "DomainAnalystForecastReviewPacket",
    ),
    "dean_os.draft.dean_os_agent_system_v7.dean_os.domain_analyst_instance_contract": (
        "DomainAnalystInstanceContract",
    ),
    "dean_os.draft.dean_os_agent_system_v7.dean_os.domain_analyst_pipeline_news_taxonomy": (
        "classify_pipeline_news_context",
    ),
    "dean_os.draft.dean_os_agent_system_v7.dean_os.domain_analyst_portability_review": (
        "DomainAnalystPortabilityReview",
    ),
    "dean_os.draft.dean_os_agent_system_v7.dean_os.domain_analyst_profile_policy_packet": (
        "DomainAnalystProfilePolicyPacket",
    ),
    "dean_os.draft.dean_os_agent_system_v7.dean_os.domain_analyst_regime_scenario_packet": (
        "DomainAnalystRegimeScenarioPacket",
    ),
    "dean_os.draft.dean_os_agent_system_v7.dean_os.domain_analyst_template_decision_packet": (
        "DomainAnalystTemplateDecisionPacket",
    ),
    "dean_os.draft.dean_os_agent_system_v7.dean_os.domain_analyst_template_standardization_packet": (
        "DomainAnalystTemplateStandardizationPacket",
    ),
    "dean_os.draft.dean_os_agent_system_v7.dean_os.domain_analyst_thesis_review_packet": (
        "DomainAnalystThesisReviewPacket",
    ),
    "dean_os.draft.dean_os_agent_system_v7.dean_os.domain_analyst_intake_packet": (
        "DomainAnalystIntakePacket",
    ),
    "dean_os.draft.dean_os_agent_system_v7.dean_os.domain_analyst_vertical_slice_run": (
        "DomainAnalystVerticalSliceRun",
    ),
    "dean_os.world_model.world_model_event_learning": (
        "WORLD_MODEL_EVENT_LEARNING_CONTRACT",
        "WorldModelEventLearningPacket",
        "render_world_model_event_learning_markdown",
    ),
    "dean_os.world_model.world_model_pipeline_context": (
        "DEFAULT_WORLD_MODEL_TIMEFRAMES",
        "WORLD_MODEL_PIPELINE_CONTEXT_CONTRACT",
        "WorldModelPipelineContextDiscovery",
        "metadata_from_pipeline_context_bundle",
        "render_world_model_pipeline_context_markdown",
    ),
    "dean_os.world_model.world_model_replay_review_gate": (
        "WORLD_MODEL_REPLAY_REVIEW_GATE_CONTRACT",
        "WorldModelReplayReviewGate",
        "render_world_model_replay_review_gate_markdown",
    ),
    "dean_os.world_model.world_model_replay_registration": (
        "DEFAULT_OUTCOME_TRACKER_DB",
        "WORLD_MODEL_REPLAY_REGISTRATION_CONTRACT",
        "WorldModelReplayRegistrationBridge",
        "render_world_model_replay_registration_markdown",
    ),
    "dean_os.draft.dean_os_agent_system_v7.dean_os.event_log": ("EventLog",),
    "dean_os.draft.dean_os_agent_system_v7.dean_os.evidence_timestamp_audit": ("EvidenceTimestampAudit",),
    "dean_os.execution.execution_gateway": (
        "ExecutionGateway",
        "ExecutionPolicy",
    ),
    "dean_os.draft.dean_os_agent_system_v7.dean_os.evidence_gap_resolution_plan": (
        "EvidenceGapResolutionPlan",
    ),
    "dean_os.draft.dean_os_agent_system_v7.dean_os.full_system_orchestrator": (
        "AgentSystemRunResult",
        "DEANAgentSystemOrchestrator",
        "create_full_agent_system",
    ),
    "dean_os.draft.dean_os_agent_system_v7.dean_os.pipeline_stage03_bridge": (
        "PipelineArtifactReference",
        "PipelineStage03Bridge",
        "PipelineStage03Packet",
        "PipelineStageState",
    ),
    "dean_os.draft.dean_os_agent_system_v7.dean_os.system_topology": (
        "BranchExecutionRecord",
        "BranchId",
        "BranchPlane",
        "BranchRunStatus",
        "BranchSpec",
        "SystemRunManifest",
        "SystemTopology",
        "load_default_system_topology",
        "load_system_topology",
    ),
    "dean_os.draft.dean_os_agent_system_v7.src.scripts.optimization.factory": (
        "create_dean_orchestrator",
        "create_hybrid_dean_orchestrator",
    ),
    "dean_os.draft.dean_os_agent_system_v7.dean_os.fundamental_input_readiness_gate": (
        "FundamentalInputReadinessGate",
    ),
    "dean_os.draft.dean_os_agent_system_v7.dean_os.agent_lab": ("AgentLabRunner",),
    "dean_os.draft.dean_os_agent_system_v7.dean_os.agent_learning_loop_runbook": (
        "AgentLearningLoopRunbook",
    ),
    "dean_os.draft.dean_os_agent_system_v7.dean_os.analyst_loop_daily_check": ("AnalystLoopDailyCheck",),
    "dean_os.draft.dean_os_agent_system_v7.dean_os.analyst_review_inbox": ("AnalystReviewInbox",),
    "dean_os.draft.dean_os_agent_system_v7.dean_os.analyst_calibration_gate": ("AnalystCalibrationGate",),
    "dean_os.draft.dean_os_agent_system_v7.dean_os.analyst_evidence_pack": (
        "AnalystEvidencePackRunner",
        "documents_from_evidence_pack",
    ),
    "dean_os.draft.dean_os_agent_system_v7.dean_os.analyst_learning_apply_ceremony": (
        "AnalystLearningApplyCeremony",
    ),
    "dean_os.draft.dean_os_agent_system_v7.dean_os.analyst_learning_promotion_bridge": (
        "AnalystLearningPromotionBridge",
    ),
    "dean_os.draft.dean_os_agent_system_v7.dean_os.analyst_outcome_evaluation_loop": (
        "AnalystOutcomeEvaluationLoop",
    ),
    "dean_os.draft.dean_os_agent_system_v7.dean_os.analyst_profile_orchestrator": (
        "AnalystProfileOrchestrator",
    ),
    "dean_os.draft.dean_os_agent_system_v7.dean_os.analyst_profile_scorecard": (
        "AnalystProfileScorecard",
    ),
    "dean_os.replays.historical_replay_batch": (
        "HistoricalReplayBatchRunner",
    ),
    "dean_os.draft.dean_os_agent_system_v7.dean_os.historical_replay": (
        "HistoricalReplayAnalyst",
        "HistoricalReplayRunner",
    ),
    "dean_os.draft.dean_os_agent_system_v7.dean_os.historical_evidence_backfill_plan": (
        "HistoricalEvidenceBackfillPlan",
    ),
    "dean_os.draft.dean_os_agent_system_v7.dean_os.historical_research_replay": (
        "HistoricalResearchReplayRunner",
    ),
    "dean_os.draft.dean_os_agent_system_v7.dean_os.historical_research_replay_batch": (
        "HistoricalResearchReplayBatchRunner",
    ),
    "dean_os.draft.dean_os_agent_system_v7.dean_os.learning": ("LearningStore",),
    "dean_os.draft.dean_os_agent_system_v7.dean_os.market_data_refresh_runbook": (
        "MarketDataRefreshRunbook",
    ),
    "dean_os.draft.dean_os_agent_system_v7.dean_os.manual_implementation_backlog": (
        "ManualImplementationBacklog",
    ),
    "dean_os.draft.dean_os_agent_system_v7.dean_os.material_loaders": (
        "ingest_research_path",
        "load_research_directory",
        "load_research_document",
    ),
    "dean_os.draft.dean_os_agent_system_v7.src.processing.filters.orchestrator": ("DEANOrchestrator",),
    "dean_os.draft.dean_os_agent_system_v7.dean_os.pipeline_metrics": (
        "PipelineMetricNormalizer",
        "PipelineMetricSnapshot",
        "PipelineRunIdentity",
        "ProfitabilityMetrics",
        "RiskMetrics",
        "ValidationMetrics",
        "FeatureStabilityMetrics",
        "DataQualityMetrics",
        "ReplayMetrics",
    ),
    "dean_os.draft.dean_os_agent_system_v7.dean_os.context_grids": (
        "ContextDimensionState",
        "ContextGrid",
        "ContextGridEdge",
        "ContextGridNode",
        "ContextIndicatorGridBuilder",
        "ContextIndicatorPacket",
        "IndicatorObservation",
        "IndicatorStateGrid",
    ),
    "dean_os.draft.dean_os_agent_system_v7.dean_os.minimal_system": (
        "DEANMinimalSystem",
        "MinimalSystemRunResult",
        "create_minimal_system",
    ),
    "dean_os.draft.dean_os_agent_system_v7.dean_os.operation_queue": ("OperationQueue",),
    "dean_os.draft.dean_os_agent_system_v7.dean_os.outcome_evaluation": ("OutcomeEvaluationRunner",),
    "dean_os.draft.dean_os_agent_system_v7.dean_os.outcome_price_coverage_plan": (
        "OutcomePriceCoveragePlan",
    ),
    "dean_os.draft.dean_os_agent_system_v7.dean_os.outcome_readiness_gate": ("OutcomeReadinessGate",),
    "dean_os.draft.dean_os_agent_system_v7.dean_os.paper_autonomy": ("PaperAutonomyRunner",),
    "dean_os.draft.dean_os_agent_system_v7.dean_os.agents.paper_portfolio": ("PaperPortfolioSimulator",),
    "dean_os.draft.dean_os_agent_system_v7.dean_os.paper_trading": (
        "PaperTradeEvaluationRunner",
        "PaperTradeStore",
    ),
    "dean_os.draft.dean_os_agent_system_v7.dean_os.pipeline_adapter": ("HybridPipelineAdapter",),
    "dean_os.draft.dean_os_agent_system_v7.dean_os.agents.pipeline_control.pipeline_control_caution_review_packet": (
        "PipelineControlCautionReviewPacket",
    ),
    "dean_os.draft.dean_os_agent_system_v7.dean_os.agents.pipeline_control.pipeline_control_bounded_evidence_batch": (
        "PipelineControlBoundedEvidenceBatch",
    ),
    "dean_os.draft.dean_os_agent_system_v7.dean_os.agents.pipeline_control.pipeline_control_bounded_evidence_run": (
        "PipelineControlBoundedEvidenceRun",
    ),
    "dean_os.draft.dean_os_agent_system_v7.dean_os.agents.pipeline_control.pipeline_control_feature_causality_audit": (
        "PipelineControlFeatureCausalityAudit",
    ),
    "dean_os.draft.dean_os_agent_system_v7.dean_os.agents.pipeline_control.pipeline_control_train_validation_diagnostic": (
        "PipelineControlTrainValidationDiagnostic",
    ),
    "dean_os.draft.dean_os_agent_system_v7.dean_os.agents.pipeline_control.pipeline_control_train_validation_experiment": (
        "PipelineControlTrainValidationExperiment",
    ),
    "dean_os.draft.dean_os_agent_system_v7.dean_os.agents.pipeline_control.pipeline_control_walk_forward_validation_run": (
        "PipelineControlWalkForwardValidationRun",
    ),
    "dean_os.draft.dean_os_agent_system_v7.dean_os.agents.pipeline_control.pipeline_control_forward_data_accrual_plan": (
        "PipelineControlForwardDataAccrualPlan",
    ),
    "dean_os.draft.dean_os_agent_system_v7.dean_os.agents.pipeline_control.pipeline_control_forward_data_accrual_gate": (
        "PipelineControlForwardDataAccrualGate",
    ),
    "dean_os.draft.dean_os_agent_system_v7.dean_os.agents.pipeline_control.pipeline_control_data_preflight": (
        "PipelineControlDataPreflight",
    ),
    "dean_os.draft.dean_os_agent_system_v7.dean_os.agents.pipeline_control.pipeline_control_saved_data_coverage": (
        "PipelineControlSavedDataCoverage",
    ),
    "dean_os.draft.dean_os_agent_system_v7.dean_os.agents.pipeline_control.pipeline_control_saved_price_repair": (
        "PipelineControlSavedPriceRepair",
    ),
    "dean_os.draft.dean_os_agent_system_v7.dean_os.agents.pipeline_control.pipeline_control_historical_price_recovery": (
        "PipelineControlHistoricalPriceRecovery",
    ),
    "dean_os.draft.dean_os_agent_system_v7.dean_os.agents.pipeline_control.pipeline_control_evidence_inventory": (
        "PipelineControlEvidenceInventory",
    ),
    "dean_os.draft.dean_os_agent_system_v7.dean_os.agents.pipeline_control.pipeline_control_metric_artifact_materializer": (
        "PipelineControlMetricArtifactMaterializer",
    ),
    "dean_os.draft.dean_os_agent_system_v7.dean_os.agents.pipeline_control.pipeline_control_metric_fixture_validation": (
        "PipelineControlMetricFixtureValidation",
    ),
    "dean_os.draft.dean_os_agent_system_v7.dean_os.agents.pipeline_control.pipeline_control_locked_evaluation_assembler": (
        "PipelineControlLockedEvaluationAssembler",
    ),
    "dean_os.draft.dean_os_agent_system_v7.dean_os.agents.pipeline_control.pipeline_control_locked_feature_stability_assembler": (
        "PipelineControlLockedFeatureStabilityAssembler",
    ),
    "dean_os.draft.dean_os_agent_system_v7.dean_os.agents.pipeline_control.pipeline_control_real_metric_evidence_run": (
        "PipelineControlRealMetricEvidenceRun",
    ),
    "dean_os.draft.dean_os_agent_system_v7.dean_os.agents.pipeline_control.pipeline_control_surface": ("PipelineControlSurface",),
    "dean_os.draft.dean_os_agent_system_v7.dean_os.agents.pipeline_control.pipeline_control_instance_contract": (
        "PipelineControlInstanceContract",
    ),
    "dean_os.draft.dean_os_agent_system_v7.dean_os.pipeline_metric_input_readiness_gate": (
        "PipelineMetricInputReadinessGate",
    ),
    "dean_os.draft.dean_os_agent_system_v7.dean_os.pipeline_stage23_runtime_profile": (
        "PIPELINE_STAGE23_RUNTIME_PROFILE_CONTRACT",
        "PipelineStage23RuntimeProfile",
        "render_pipeline_stage23_runtime_profile_markdown",
    ),
    "dean_os.draft.dean_os_agent_system_v7.dean_os.pipeline_timeframe_lane_readiness": (
        "PIPELINE_TIMEFRAME_LANE_READINESS_CONTRACT",
        "PipelineTimeframeLaneReadinessPlan",
        "render_pipeline_timeframe_lane_readiness_markdown",
    ),
    "dean_os.draft.dean_os_agent_system_v7.dean_os.recommendation_memory": (
        "RecommendationMemoryStore",
    ),
    "dean_os.replays.replay_price_normalizer": ("ReplayPriceNormalizer",),
    "dean_os.replays.replay_calibration_readiness_gate": (
        "ReplayCalibrationReadinessGate",
    ),
    "dean_os.replays.replay_evidence_window_selector": (
        "ReplayEvidenceWindowSelector",
    ),
    "dean_os.replays.replay_price_artifact_repair": (
        "ReplayPriceArtifactRepairPlan",
    ),
    "dean_os.replays.replay_price_quality_investigation": (
        "ReplayPriceQualityInvestigationPlan",
    ),
    "dean_os.draft.dean_os_agent_system_v7.dean_os.research_replay_directionality_diagnostic": (
        "ResearchReplayDirectionalityDiagnostic",
    ),
    "dean_os.draft.dean_os_agent_system_v7.dean_os.regime_context": (
        "RegimeContextBuilder",
        "normalize_context_tags",
    ),
    "dean_os.draft.dean_os_agent_system_v7.dean_os.review_only_automation_run": (
        "DeanOSReviewOnlyAutomationRun",
    ),
    "dean_os.draft.dean_os_agent_system_v7.dean_os.review_action_apply_ceremony": (
        "ReviewActionApplyCeremony",
    ),
    "dean_os.draft.dean_os_agent_system_v7.dean_os.review_action_dry_run": ("ReviewActionDryRun",),
    "dean_os.packets.review_decision_packet": ("ReviewDecisionPacket",),
    "dean_os.packets.sector_to_ticker_review_packet": (
        "DomainSpecialistReviewPacket",
        "SectorToTickerReviewPacket",
    ),
    "dean_os.draft.dean_os_agent_system_v7.dean_os.real_source_dropzone_inventory": (
        "RealSourceDropzoneInventory",
    ),
    "dean_os.packets.real_source_normalized_packet": (
        "RealSourceNormalizedPacketBuilder",
    ),
    "dean_os.draft.dean_os_agent_system_v7.dean_os.source_evidence_validation_gate": (
        "SourceEvidenceValidationGate",
    ),
    "dean_os.packets.source_extraction_fixture_packet": (
        "SourceExtractionFixturePacket",
    ),
    "dean_os.draft.dean_os_agent_system_v7.dean_os.source_extraction_fixture_review_gate": (
        "SourceExtractionFixtureReviewGate",
    ),
    "dean_os.packets.source_extraction_review_packet": (
        "SourceExtractionReviewPacket",
    ),
    "dean_os.draft.dean_os_agent_system_v7.dean_os.staged_workbench_integration_review": (
        "StagedWorkbenchIntegrationReview",
    ),
    "dean_os.agents.chief_review": ("ChiefReviewAgent",),
    "dean_os.agents.collector_inventory": (
        "CollectorInventoryAgent",
    ),
    "dean_os.agents.diary_bridge": ("DiaryBridgeAgent",),
    "dean_os.agents.market_data_freshness": (
        "MarketDataFreshnessAgent",
    ),
    "dean_os.agents.model_performance": ("ModelPerformanceAgent",),
    "dean_os.agents.paper_portfolio": ("PaperPortfolioAgent",),
    "dean_os.agents.regime": ("RegimeAgent",),
    "dean_os.agents.source_routing": ("SourceRoutingAgent",),
    "dean_os.agents.tuning": ("TuningAgent",),
    "dean_os.agents.unified_research_agent": ("UnifiedResearchAgent",),
    "dean_os.draft.dean_os_agent_system_v7.dean_os.calibration_proposal_agent": (
        "CalibrationProposalAgent",
    ),
    "dean_os.draft.dean_os_agent_system_v7.dean_os.calibration_review_lifecycle": (
        "CalibrationReviewLifecycle",
    ),
    "dean_os.draft.dean_os_agent_system_v7.dean_os.research_corpus": ("ResearchCorpus",),
    "dean_os.draft.dean_os_agent_system_v7.dean_os.review": ("AgentReviewBuilder",),
    "dean_os.draft.dean_os_agent_system_v7.dean_os.review_actions": ("ReviewActionStore",),
    "dean_os.draft.dean_os_agent_system_v7.dean_os.review_approved_learning_loop": (
        "ReviewApprovedLearningLoop",
    ),
    "dean_os.draft.dean_os_agent_system_v7.dean_os.sample_materials": ("agent_lab_sample_documents",),
    "dean_os.analysts._producers.macro": (
        "SavedMacroEvidenceProducer",
        "load_verified_macro_context_fragment",
    ),
    "dean_os.analysts._producers.sec.filing_index": (
        "SavedSECFilingIndexProducer",
        "verify_sec_filing_index",
        "verify_saved_sec_filing_index",
    ),
    "dean_os.analysts._producers.sec.submissions_index": (
        "SavedSECSubmissionsFilingIndexProducer",
        "verify_saved_sec_submissions_filing_index",
    ),
    "dean_os.analysts._producers.sec.companyfacts": (
        "SavedSECCompanyFactsProducer",
        "load_verified_fundamental_context_fragment",
    ),
    "dean_os.analysts._producers.sec.inline_xbrl": (
        "SavedSECInlineXBRLProducer",
        "load_verified_inline_xbrl_context_fragment",
    ),
    "dean_os.analysts._producers.sec.merger": (
        "SavedSECFundamentalEvidenceMerger",
        "load_verified_merged_fundamental_context_fragment",
    ),
    "dean_os.analysts._producers.sec.ratios": (
        "SavedSECDerivedRatioProducer",
        "load_verified_derived_ratio_context_fragment",
    ),
    "dean_os.analysts._producers.sector_market": (
        "SavedSectorMarketEvidenceProducer",
        "load_verified_sector_market_context_fragment",
    ),
    "dean_os.analysts._producers.news": (
        "SavedSemiconductorNewsEvidenceProducer",
        "load_verified_semiconductor_news_context_fragment",
    ),
    "dean_os.analysts._producers.policy": (
        "SavedOfficialPolicyEvidenceProducer",
        "load_verified_official_policy_context_fragment",
    ),
    "dean_os.analysts._producers.runtime": (
        "SemiconductorAnalystRuntime",
    ),
    "dean_os.draft.dean_os_agent_system_v7.dean_os.sector_thesis_to_ticker_basket_bridge": (
        "SectorThesisToTickerBasketBridge",
    ),
    "dean_os.draft.dean_os_agent_system_v7.dean_os.agents.synthesis": ("EvidenceBoundSynthesizer",),
    "dean_os.draft.dean_os_agent_system_v7.dean_os.ticker_focused_replay_exam_bridge": (
        "TickerFocusedReplayExamBridge",
    ),
    "dean_os.draft.dean_os_agent_system_v7.dean_os.ticker_focused_research_note_builder": (
        "TickerFocusedResearchNoteBuilder",
    ),
    "dean_os.draft.dean_os_agent_system_v7.dean_os.ticker_specific_attribution_audit": (
        "TickerSpecificAttributionAudit",
    ),
    "dean_os.draft.dean_os_agent_system_v7.src.models.prototypes.registry": ("AgentRegistry",),
    "dean_os.schemas": (
        "AgentCapabilities",
        "AgentLabRunReport",
        "AnalyticalReport",
        "BaseAgentReport",
        "ConsensusDecision",
        "EvidenceItem",
        "ExecutionOutcome",
        "AgentLearningRecord",
        "FinancialNLPResult",
        "MarketContext",
        "MarketRegimeSnapshot",
        "PaperTradeRecord",
        "PipelineReport",
        "PipelineActionProposal",
        "RecommendationMemoryRecord",
        "ResearchChunk",
        "ResearchDocument",
        "ResearchNote",
        "ReviewActionRecord",
        "SourceCitation",
    ),
    "dean_os.analyst_core": (
        "OUTCOME_HORIZONS",
        "REGIME_DIMENSIONS",
        "SCENARIO_NODE_TYPES",
        "SCENARIO_EDGE_TYPES",
        "Confidence",
        "EvidenceGap",
        "HistoricalOutcomeCheck",
        "HorizonOutcome",
        "HypothesisLedgerEntry",
        "HypothesisStatus",
        "Priority",
        "RegimeContextVector",
        "RegimeDimensionState",
        "ScenarioEdge",
        "ScenarioNode",
        "ScenarioOutcomeGraph",
        "Trend",
    ),
    "dean_os.analyst_core.lens_contract": (
        "AnalysisPacket",
        "AnalystLens",
        "LensRegistry",
        "ModuleDelta",
    ),
}

_EXPORT_GROUPS.update({
    "dean_os.draft.dean_os_agent_system_v7.dean_os.briefing_contract": ("DailyBriefing", "DailyBriefingBuilder", "CoverageGateItem"),
    "dean_os.draft.dean_os_agent_system_v7.dean_os.daily_agent_run": ("DailyAgentRun", "DailyAgentRunResult"),
    "dean_os.draft.dean_os_agent_system_v7.dean_os.evidence_catalog": (
        "CatalogEvidenceRecord",
        "EvidenceAcquisitionRunManifest",
        "EvidenceCatalogBuilder",
        "SQLiteEvidenceCatalog",
    ),
    "dean_os.draft.dean_os_agent_system_v7.dean_os.replay_scheduler": ("ReplayScheduleItem", "ReplayScheduler"),
    "dean_os.draft.dean_os_agent_system_v7.dean_os.daily_run_store": ("DailyRunRecord", "DailyRunRecordBuilder", "SQLiteDailyRunStore"),
    "dean_os.draft.dean_os_agent_system_v7.dean_os.source_credibility": ("SourceCredibilityAssessment", "SourceCredibilityRegistry"),
    "dean_os.draft.dean_os_agent_system_v7.dean_os.evidence_dedup": ("EvidenceDedupDecision", "EvidenceDedupResult", "SemanticEvidenceDeduplicator"),
    "dean_os.draft.dean_os_agent_system_v7.dean_os.collector_routing": ("CollectorRoute", "DomainCollectorRouter"),
    "dean_os.draft.dean_os_agent_system_v7.dean_os.evidence_gap_planner_v2": ("EvidenceGapTask", "EvidenceGapPlan", "EvidenceGapPlanner"),
    "dean_os.draft.dean_os_agent_system_v7.dean_os.briefing_renderer": ("DailyBriefingRenderer",),
    "dean_os.draft.dean_os_agent_system_v7.dean_os.operator_review_inbox_v2": (
        "ReviewInboxItem",
        "SQLiteOperatorReviewInbox",
        "OperatorReviewInboxBuilder",
    ),
})


_LAZY_EXPORTS = {
    name: (module_name, name)
    for module_name, names in _EXPORT_GROUPS.items()
    for name in names
}
__all__ = list(_LAZY_EXPORTS)


def __getattr__(name: str) -> Any:
    target = _LAZY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute_name = target
    value = getattr(import_module(module_name), attribute_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()).union(__all__))
