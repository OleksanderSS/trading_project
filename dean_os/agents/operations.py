from __future__ import annotations

from dean_os.base import BaseAgent
from dean_os.schemas import MarketContext, PipelineActionProposal, PipelineReport


class OperationsProposalAgent(BaseAgent):
    version = "0.1.0"
    branch = "pipeline"

    async def run(self, context: MarketContext) -> PipelineReport:
        proposals = self._build_proposals(context)
        context.action_proposals.extend(proposals)
        verdict = "caution" if proposals else "clear"
        return PipelineReport(
            agent_name=self.name,
            agent_version=self.version,
            verdict=verdict,
            confidence=0.75 if proposals else 0.6,
            data_quality_score=0.7,
            signal_strength=0.0,
            reasons=[proposal.reason for proposal in proposals] or ["No operational action proposed"],
            risks=[
                "OperationsProposalAgent only proposes actions; execution requires explicit approval",
            ],
            blind_spots=["Freshness metadata must be supplied by the caller or future scheduler"],
            evidence=[
                self.evidence("operation", "context.action_proposals", "proposal_count", len(proposals)),
                self.evidence("config", "agent_config", "proposal_only", True),
            ],
            input_hash=self.context_hash(context),
            metrics_snapshot={"proposal_count": len(proposals), "proposals": [proposal.model_dump(mode="json") for proposal in proposals]},
        )

    def _build_proposals(self, context: MarketContext) -> list[PipelineActionProposal]:
        proposals: list[PipelineActionProposal] = []
        freshness = context.metadata.get("data_freshness", {})
        stale_sources = [name for name, info in freshness.items() if isinstance(info, dict) and info.get("stale")]
        if stale_sources:
            proposals.append(
                PipelineActionProposal(
                    agent_name=self.name,
                    action_type="accumulate",
                    target=",".join(stale_sources),
                    reason=f"Stale data sources detected: {', '.join(stale_sources)}",
                    command_preview="python run_dean_os.py --mode local --tickers <approved> --timeframes <approved>",
                    expected_effect="Refresh parsing/collection inputs before analytical agents form new theses",
                    risks=["May trigger expensive data collection if approved"],
                    evidence=[self.evidence("metric", "context.metadata.data_freshness", "stale_sources", stale_sources)],
                )
            )
        if context.research_documents and not context.research_notes:
            proposals.append(
                PipelineActionProposal(
                    agent_name=self.name,
                    action_type="parse",
                    target="research_documents",
                    reason="Research documents are present but no ResearchNote has been produced yet",
                    command_preview="run ResearchIngestionAgent and SpecialistResearchAgent in isolated Agent Lab",
                    expected_effect="Convert raw materials into structured notes and pattern evidence",
                    risks=["Parsing quality depends on document extraction quality"],
                    evidence=[self.evidence("document", "context.research_documents", "document_count", len(context.research_documents))],
                )
            )
        if context.metadata.get("agent_lab") and context.research_notes:
            proposals.append(
                PipelineActionProposal(
                    agent_name=self.name,
                    action_type="validate",
                    target="agent_lab_report",
                    reason="Research notes were produced and should be reviewed before downstream use",
                    command_preview="review latest reports/dean_os/agent_lab/*.md and approve/reject generated operation proposals",
                    expected_effect="Keep specialist theses evidence-bound before they influence pipeline tuning or watchlists",
                    risks=["Human review is still required; validation does not imply trade readiness"],
                    evidence=[
                        self.evidence("research_note", "context.research_notes", "note_count", len(context.research_notes))
                    ],
                )
            )
        model_performance = context.metadata.get("model_performance", {})
        if (
            isinstance(model_performance, dict)
            and model_performance.get("verdict") == "caution"
            and not _has_tuning_related_proposal(context)
        ):
            failures = model_performance.get("threshold_failures", [])
            action_type = "validate" if "missing_evaluation_metrics" in failures else "tune"
            proposals.append(
                PipelineActionProposal(
                    agent_name=self.name,
                    action_type=action_type,
                    target="model_performance",
                    reason="Model performance evidence is missing, stale, or below configured thresholds",
                    command_preview="review model evaluation metrics and create an approved validation/tuning experiment",
                    expected_effect="Keep model changes behind review before they influence promotion, paper trading, or production config",
                    risks=["Tuning can overfit unless it uses walk-forward validation and explicit risk constraints"],
                    evidence=[
                        self.evidence("metric", "context.metadata.model_performance", "threshold_failures", failures),
                        self.evidence(
                            "metric",
                            "context.metadata.model_performance",
                            "performance_score",
                            model_performance.get("performance_score"),
                        ),
                    ],
                )
            )
        if context.dataframes and "features" not in context.dataframes:
            proposals.append(
                PipelineActionProposal(
                    agent_name=self.name,
                    action_type="enrich",
                    target="features",
                    reason="Raw dataframes exist but enriched features are missing from context",
                    command_preview="approved feature enrichment stage only",
                    expected_effect="Prepare richer inputs for data quality, risk, and specialist agents",
                    risks=["Feature enrichment must preserve temporal availability"],
                    evidence=[self.evidence("dataframe_check", "context.dataframes", "available_frames", list(context.dataframes))],
                )
            )
        return proposals


def _has_tuning_related_proposal(context: MarketContext) -> bool:
    tuning_targets = {"model_performance", "walk_forward_tuning_experiment", "tuning_inputs"}
    return any(
        proposal.action_type in {"tune", "validate"} and proposal.target in tuning_targets
        for proposal in context.action_proposals
    )
