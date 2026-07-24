from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from dean_os.artifact_writer import ReviewArtifactWriter
from dean_os.schemas import utc_now_iso
from dean_os.utils import json_ready

CAPABILITY_CONTRACTS: dict[str, dict[str, Any]] = {
    "semiconductor_analyst": {
        "inputs": [
            "timezone-aware MarketContext",
            "domain profile",
        ],
        "effects": ["standalone review-only sector report"],
        "gap": (
            "mutually exclusive with the composite pipeline manager "
            "for the same domain and phase"
        ),
    },
    "energy_analyst": {
        "inputs": [
            "timezone-aware MarketContext",
            "domain profile",
        ],
        "effects": ["standalone review-only sector report"],
    },
    "macro_analyst": {
        "inputs": [
            "timezone-aware MarketContext",
            "domain profile",
        ],
        "effects": ["standalone review-only domain report"],
    },
    "pipeline_manager": {
        "inputs": [
            "hash-bound producer or runtime artifacts",
            "timezone-aware as_of",
            "optional pipeline readiness artifacts",
        ],
        "effects": [
            "composite review-only sector report",
            "pipeline readiness context",
        ],
        "gap": (
            "current semiconductor ticker pipeline is blocked by "
            "feature timeframe mismatch and quarantined Stage 5 contexts"
        ),
    },
    "pipeline_audit": {
        "inputs": ["saved audit findings", "declared triage file"],
        "effects": ["hard safety veto"],
        "gap": "declared triage file is not independently validated",
    },
    "data_quality": {
        "inputs": ["MarketContext.dataframes"],
        "effects": ["hard safety veto"],
    },
    "risk": {
        "inputs": ["returns", "positions", "offline-target marker"],
        "effects": ["hard safety veto", "risk_context"],
    },
    "model_performance": {
        "inputs": [
            "locked model evaluation",
            "real metric evidence chain",
            "pipeline model case",
            "Stage7 analyzer coverage",
        ],
        "effects": ["context metadata"],
        "gap": (
            "fixed AMD artifact paths are not yet lineage-matched to the "
            "current MarketContext"
        ),
    },
    "regime": {
        "inputs": ["Stage7 per-ticker/timeframe regime review"],
        "effects": ["review report", "context metadata", "context tags"],
        "gap": "shadow-only until per-context decision calibration exists",
    },
    "context_synthesis": {
        "inputs": [
            "Stage5 per-context prediction review",
            "Stage7 per-context regime review",
        ],
        "effects": ["shadow compatibility report", "context metadata"],
        "gap": (
            "directional synthesis and decision influence remain disabled "
            "until freshness and outcome calibration are proven"
        ),
    },
    "tuning": {
        "inputs": ["model", "regime", "freshness", "control surface"],
        "effects": ["proposal candidates"],
    },
    "chief_review": {
        "inputs": ["review reports", "model", "regime", "tuning", "notes"],
        "effects": ["context metadata"],
    },
    "paper_portfolio": {
        "inputs": ["paper records", "prices", "watchlist decision"],
        "effects": ["paper SQLite writes"],
    },
    "diary_bridge": {
        "inputs": ["paper records", "experience diary"],
        "effects": ["proposal candidates"],
    },
    "source_routing": {
        "inputs": ["materials", "collector inventory"],
        "effects": ["context metadata"],
    },
    "operations_proposal": {
        "inputs": ["freshness", "model", "docs", "dataframes"],
        "effects": ["proposal candidates"],
    },
    "research_ingestion": {
        "inputs": ["research documents"],
        "effects": ["research corpus SQLite writes"],
    },
    "specialist_research": {
        "inputs": ["corpus", "news", "fundamentals", "macro"],
        "effects": ["research notes"],
    },
    "unified_research": {
        "inputs": ["duckdb", "corpus", "news", "fundamentals", "macro", "prices", "sec"],
        "effects": ["duckdb intelligence", "cross-source analytics", "data quality metrics"],
    },
    "financial_nlp": {
        "inputs": ["documents", "news"],
        "effects": ["NLP results"],
    },
    "evidence_synthesis": {
        "inputs": ["documents", "NLP results", "notes"],
        "effects": ["context notes", "synthesis result"],
    },
    "macro_policy": {
        "inputs": ["news", "macro"],
        "effects": ["analytical modifier"],
    },
    "geopolitical": {
        "inputs": ["news", "structured context"],
        "effects": ["analytical modifier"],
    },
    "news_catalyst": {
        "inputs": ["news", "structured context"],
        "effects": ["analytical modifier"],
    },
    "sector_cycle": {
        "inputs": ["keyword news", "structured context marker"],
        "effects": ["analytical report"],
    },
    "industry_map": {
        "inputs": ["keyword news", "structured context marker"],
        "effects": ["analytical report"],
        "gap": "keyword detection is not a reliable ticker-to-industry map",
    },
    "historical_analogies": {
        "inputs": ["keyword news", "structured context marker"],
        "effects": ["analytical report"],
    },
    "value_screening": {
        "inputs": ["fundamentals", "readiness marker"],
        "effects": ["analytical report"],
    },
    "contrarian_thesis": {
        "inputs": ["keyword news", "structured context marker"],
        "effects": ["analytical report"],
    },
    "agriculture_analyst": {
        "inputs": [
            "timezone-aware MarketContext",
            "domain profile",
        ],
        "effects": ["standalone review-only sector report"],
    },
    "geopolitics_analyst": {
        "inputs": [
            "timezone-aware MarketContext",
            "domain profile",
        ],
        "effects": ["standalone review-only sector report"],
    },
    "liquidity_credit_analyst": {
        "inputs": [
            "timezone-aware MarketContext",
            "domain profile",
        ],
        "effects": ["standalone review-only sector report"],
    },
    "logistics_analyst": {
        "inputs": [
            "timezone-aware MarketContext",
            "domain profile",
        ],
        "effects": ["standalone review-only sector report"],
    },
    "real_estate_analyst": {
        "inputs": [
            "timezone-aware MarketContext",
            "domain profile",
        ],
        "effects": ["standalone review-only sector report"],
    },
    "agent_evaluation_controller": {
        "inputs": ["recent agent execution traces"],
        "effects": ["hard safety veto on observed unsafe-action attempts"],
        "gap": "quality thresholds only caution until enough reviewed runs exist",
    },
    "pipeline_readiness": {
        "inputs": ["configured Stage 4/5 readiness artifact paths"],
        "effects": ["review report", "context metadata"],
        "gap": "returns needs_more_data when no artifact_paths are configured",
    },
    "coherence_scan": {
        "inputs": [
            "pipeline + analytical agent reports merged after the "
            "analytical branch completes",
        ],
        "effects": ["cross-agent contradiction report"],
        "gap": (
            "runs as an explicit second orchestration pass "
            "(DEANOrchestrator.PEER_SYNTHESIS_AGENTS) rather than inside "
            "the analytical branch's own asyncio.gather() batch, since it "
            "reconciles the verdicts those peers just produced"
        ),
    },
    "freshness_audit": {
        "inputs": ["MarketContext news/macro/prices/fundamentals timestamps"],
        "effects": ["staleness report", "context metadata"],
    },
    "news_event_analyzer": {
        "inputs": ["news items", "vix_data dataframe"],
        "effects": ["event classification report", "causal graph metadata"],
        "gap": "disabled: NewsEvent(**item) construction predates real news collector schemas",
    },
}


class AgentCapabilityMatrixBuilder:
    """Build a review-only view of registry activation and actual contracts."""

    def __init__(
        self,
        registry_path: str | Path = (
            "dean_os/config/agent_registry.yaml"
        ),
        output_dir: str | Path = (
            "reports/dean_os/agent_capability_matrix_current"
        ),
    ):
        self.registry_path = Path(registry_path)
        self.output_dir = Path(output_dir)

    def build(self, save: bool = True) -> dict[str, Any]:
        raw = yaml.safe_load(
            self.registry_path.read_text(encoding="utf-8")
        ) or {}
        registry = raw.get("agents", {})
        entries = [
            self._entry(name, config)
            for name, config in registry.items()
        ]
        undeclared = sorted(
            set(registry) - set(CAPABILITY_CONTRACTS)
        )
        stale_contracts = sorted(
            set(CAPABILITY_CONTRACTS) - set(registry)
        )
        payload = {
            "run_id": _run_id(),
            "created_at": utc_now_iso(),
            "mode": "agent_capability_matrix",
            "schema_version": "dean_agent_capability_matrix_v1",
            "registry_path": str(self.registry_path),
            "summary": {
                "agent_count": len(entries),
                "enabled_count": sum(
                    bool(item["enabled"]) for item in entries
                ),
                "hard_gate_count": sum(
                    item["activation_mode"] == "active_hard_gate"
                    for item in entries
                ),
                "shadow_review_count": sum(
                    item["activation_mode"]
                    == "active_shadow_review"
                    for item in entries
                ),
                "decision_influencing_count": sum(
                    bool(item["decision_influence"])
                    for item in entries
                ),
                "undeclared_contracts": undeclared,
                "stale_contracts": stale_contracts,
                "matrix_complete": not undeclared
                and not stale_contracts,
                "next_integration_target": (
                    "first real chained four-component exact-context "
                    "case set and reviewed diagnostic run"
                ),
            },
            "entries": entries,
            "scope_boundaries": {
                "amd_current_case": "ticker_model_evaluation_only",
                "semiconductor_scope": (
                    "separate domain thesis and sector-to-ticker bridge"
                ),
                "sector_evidence_can_be_ticker_evidence": False,
            },
            "safety": {
                "review_only": True,
                "is_activation_gate": False,
                "changes_registry": False,
                "runs_agents": False,
                "runs_pipeline": False,
                "writes_learning_memory": False,
                "writes_production_config": False,
                "can_trade": False,
            },
        }
        if save:
            saved = ReviewArtifactWriter(self.output_dir).write(
                payload=payload,
                markdown=render_agent_capability_matrix_markdown(
                    payload
                ),
                run_id=payload["run_id"],
            )
            payload["saved_paths"] = saved
        return json_ready(payload)

    def _entry(
        self,
        name: str,
        config: dict[str, Any],
    ) -> dict[str, Any]:
        contract = CAPABILITY_CONTRACTS.get(name, {})
        enabled = bool(config.get("enabled", False))
        branch = str(config.get("branch", "unknown"))
        veto_level = str(config.get("veto_level", "none"))
        shadow_mode = bool(config.get("shadow_mode", False))
        if not enabled:
            activation_mode = "disabled"
        elif shadow_mode:
            activation_mode = "active_shadow_review"
        elif branch == "pipeline" and veto_level == "hard":
            activation_mode = "active_hard_gate"
        elif branch == "analytical":
            activation_mode = "active_analytical_modifier"
        else:
            activation_mode = "active_soft_review"
        decision_influence = enabled and bool(
            config.get("decision_influence", not shadow_mode)
        )
        return {
            "agent_name": name,
            "class_path": config.get("class_path"),
            "branch": branch,
            "enabled": enabled,
            "activation_mode": activation_mode,
            "veto_level": veto_level,
            "proposal_only": bool(
                config.get("proposal_only", False)
            ),
            "run_phases": config.get(
                "run_phases",
                (
                    ["post_pipeline"]
                    if branch == "analytical"
                    else ["pre_pipeline", "pre_trade"]
                ),
            ),
            "actual_inputs": contract.get("inputs", []),
            "effects": contract.get("effects", []),
            "decision_influence": decision_influence,
            "known_gap": contract.get("gap"),
            "contract_declared": name in CAPABILITY_CONTRACTS,
        }


def render_agent_capability_matrix_markdown(
    payload: dict[str, Any],
) -> str:
    summary = payload.get("summary", {})
    lines = [
        "# DEAN-OS Agent Capability Matrix",
        "",
        f"- Agents: {summary.get('agent_count')}",
        f"- Enabled: {summary.get('enabled_count')}",
        f"- Hard gates: {summary.get('hard_gate_count')}",
        f"- Shadow review agents: {summary.get('shadow_review_count')}",
        f"- Matrix complete: {summary.get('matrix_complete')}",
        "",
        "| Agent | Branch | Activation | Inputs | Effects | Decision influence |",
        "|---|---|---|---|---|---|",
    ]
    for item in payload.get("entries", []):
        lines.append(
            "| {agent} | {branch} | {activation} | {inputs} | "
            "{effects} | {influence} |".format(
                agent=item.get("agent_name"),
                branch=item.get("branch"),
                activation=item.get("activation_mode"),
                inputs=", ".join(item.get("actual_inputs", [])),
                effects=", ".join(item.get("effects", [])),
                influence=item.get("decision_influence"),
            )
        )
    lines.extend(
        [
            "",
            "## Scope Boundary",
            "",
            "- AMD current case is one ticker/model evaluation case.",
            "- Semiconductor is a separate domain/sector thesis.",
            "- Sector evidence cannot become ticker evidence without the "
            "sector-to-ticker bridge and direct ticker evidence.",
            "",
            "This matrix is observability, not an activation gate.",
        ]
    )
    return "\n".join(lines) + "\n"


def _run_id() -> str:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S_%fZ")
    return f"agent_capability_matrix_{stamp}"
