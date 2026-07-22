"""Thin domain orchestrator for DEAN-OS.

This is a **composer**, not a new engine. It reuses the existing, already
reviewed building blocks:

* `AgentRegistry` (`dean_os.registry`) — loads the **pipeline-control branch**
  agents (risk, data_quality, regime, context_synthesis, the domain analyst,
  the pipeline manager, …), enforces hard-veto + synthetic blocked reports, and
  respects `run_phases` / `context.phase`.
* `DomainAnalystAgent` (`dean_os.agents.domain_analyst`) — the **analyst
  branch** for a single domain; it is review-only and returns a `PipelineReport`.
* `AnalystProfileOrchestrator` (`dean_os.analyst_core.analyst_profile_orchestrator`) —
  optional extra profile agents (e.g. `value_screening`) for the domain.
* `ReviewArtifactWriter` (`dean_os.artifact_writer`) — safe, atomic, local-only
  report writing (no learning/config/broker writes).

The default path first requires a recursively verified `DomainContextSet`.
Incomplete or unaccepted sets become explicit waiting states and run zero
agents. The older three-step diagnostic path remains available only through an
explicit legacy flag:
1. **Pipeline-control branch** — generic pipeline agents (risk, data_quality,
   regime, context_synthesis, …) loaded via `AgentRegistry`.
2. **Analyst branch** — the domain analyst plus optional profile agents.
3. **Composite pipeline-manager step** — `pipeline_manager`
   (`composite_domain_pipeline_manager`), run last because it consumes the
   domain analyst's runtime artifact.

It collects all reports and writes a single `DomainOrchestratorReport`. It is
**fail-closed**: `can_trade` is always `False` and no live action is ever taken.

Design notes (no duplication):
* We do NOT re-implement agent selection, veto logic, or prerequisite checks —
  `AgentRegistry.load_all` already does that.
* We do NOT re-implement the domain analyst — `DomainAnalystAgent` already
  exists and is configured per-domain in `config/agent_registry.yaml`.
* The orchestrator only wires the two branches together and adds the
  domain-profile context (tickers, lenses) into `MarketContext.metadata`.
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from dean_os.artifact_writer import ReviewArtifactWriter
from dean_os.branches import AnalyticalBranch, PipelineBranch
from dean_os.domain_context_set import load_verified_domain_context_set
from dean_os.domain_profiles import get_profile, get_profile_agents
from dean_os.registry import AgentRegistry
from dean_os.schemas import MarketContext, PipelineReport, utc_now_iso
from dean_os.utils import json_ready


DEFAULT_REGISTRY_PATH = "dean_os/config/agent_registry.yaml"


class DomainOrchestrator:
    """Runs the pipeline-control and analyst branches for one domain."""

    def __init__(
        self,
        registry_path: str | Path = DEFAULT_REGISTRY_PATH,
        output_dir: str | Path = "reports/dean_os/domain_orchestrator",
        project_root: str | Path | None = None,
    ):
        self.registry_path = Path(registry_path)
        self.output_dir = Path(output_dir)
        if project_root is not None:
            self.project_root = Path(project_root).resolve()
        else:
            resolved_registry = self.registry_path.resolve()
            if (
                resolved_registry.parent.name == "config"
                and resolved_registry.parent.parent.name == "dean_os"
            ):
                self.project_root = resolved_registry.parents[2]
            else:
                self.project_root = Path.cwd().resolve()

    def run_sync(
        self,
        domain_id: str,
        as_of: str | None = None,
        tickers: list[str] | None = None,
        phase: str = "pre_trade",
        include_profile_agents: bool = False,
        context_set_path: str | Path | None = None,
        allow_legacy_unbound_context: bool = False,
        save: bool = True,
    ) -> dict[str, Any]:
        """Synchronous wrapper around `run` for CLI / scripts."""
        return asyncio.run(
            self.run(
                domain_id,
                as_of=as_of,
                tickers=tickers,


                phase=phase,
                include_profile_agents=include_profile_agents,
                context_set_path=context_set_path,
                allow_legacy_unbound_context=allow_legacy_unbound_context,
                save=save,
            )
        )

    async def run(
        self,
        domain_id: str,
        as_of: str | None = None,
        tickers: list[str] | None = None,
        phase: str = "pre_trade",
        include_profile_agents: bool = False,
        context_set_path: str | Path | None = None,
        allow_legacy_unbound_context: bool = False,
        save: bool = True,
    ) -> dict[str, Any]:
        profile = get_profile(domain_id)
        effective_as_of = as_of or utc_now_iso()
        resolved_tickers = sorted({
            str(t).strip().upper()
            for t in (tickers or list(profile.ticker_universe_hint))
            if str(t).strip()
        })

        if not allow_legacy_unbound_context:
            gate_payload = self._context_set_gate(
                domain_id=domain_id,
                profile=profile,
                effective_as_of=effective_as_of,
                resolved_tickers=resolved_tickers,
                context_set_path=context_set_path,
                phase=phase,
                include_profile_agents=include_profile_agents,
            )
            if save:
                gate_payload["saved_paths"] = ReviewArtifactWriter(
                    self.output_dir / domain_id
                ).write(
                    payload=gate_payload,
                    markdown=render_domain_orchestrator_markdown(gate_payload),
                    run_id=gate_payload["run_id"],
                )
            return json_ready(gate_payload)

        from dean_os.data_loader import load_duckdb_tables
        try:
            # We load a small subset of records for the preflight context
            # to avoid huge memory overhead in the orchestrator.
            loaded_dfs = load_duckdb_tables(limit=5000)
            data_preflight_error = None
        except Exception as exc:
            loaded_dfs = {}
            data_preflight_error = f"{type(exc).__name__}: {exc}"

        context = MarketContext(
            phase=phase,  # type: ignore[arg-type]
            as_of=effective_as_of,
            tickers=resolved_tickers,
            dataframes=loaded_dfs,
            # RiskAgent expects normalized portfolio weights; a notional cash
            # amount would look like a 100,000x gross-exposure breach.
            positions={"CASH": 1.0},
            returns={"SPY": 0.0},
            metadata={
                "domain_id": domain_id,
                "domain_label": profile.display_name,
                "domain_required_evidence_types": list(
                    profile.required_evidence_types
                ),
                "domain_core_questions": list(profile.core_questions),
                "orchestrator_branch": "domain_orchestrator",
                "data_preflight_error": data_preflight_error,
            },
        )

        # --- Branch 1: pipeline-control (reuses AgentRegistry) ---
        registry = AgentRegistry(self.registry_path, project_root=self.project_root)
        pipeline_agents = registry.load_branch("pipeline", context)
        # Composite domain pipeline managers (e.g. pipeline_manager) are run as a
        # dedicated Step 3 AFTER the analyst branch, because they consume the
        # domain analyst's runtime artifact. Keep them out of the generic loop.
        composite_managers = [
            a
            for a in pipeline_agents
            if getattr(a, "config", {}).get("agent_role") == "composite_domain_pipeline_manager"
        ]
        domain_analysts = [
            a
            for a in pipeline_agents
            if a.__class__.__name__ == "DomainAnalystAgent"
            and getattr(a, "config", {}).get("domain_id") == domain_id
        ]
        # The analyst has its own branch below.  Running it here too would
        # duplicate its runtime artifact and could make the composite manager
        # consume a different run than the one shown in the analyst branch.
        control_agents = [
            a
            for a in pipeline_agents
            if a not in composite_managers
            and a.__class__.__name__ != "DomainAnalystAgent"
        ]
        # This is a diagnostic, review-only composer, so collect all control
        # diagnostics while still preserving each agent's timeout and schema
        # contract.  Hard blocks remain visible and can never grant authority.
        pipeline_reports = [
            _report_to_dict(report)
            for report in await PipelineBranch(
                control_agents,
                soft_mode=True,
            ).run(context)
        ]
        synthetic_reports = [
            _report_to_dict(report) for report in registry.get_synthetic_reports().values()
        ]
        load_errors = dict(registry.get_load_errors())

        # --- Branch 2: analyst (domain analyst + optional profile agents) ---
        analyst_reports = await self._run_analyst_branch(
            domain_id=domain_id,
            context=context,
            profile=profile,
            configured_domain_agents=domain_analysts,
            include_profile_agents=include_profile_agents,
        )

        # --- Step 3: composite domain pipeline manager (consumes analyst runtime) ---
        composite_reports = await self._run_composite_managers(composite_managers, context)

        summary = self._summary(
            domain_id=domain_id,
            profile=profile,
            resolved_tickers=resolved_tickers,
            pipeline_reports=pipeline_reports,
            synthetic_reports=synthetic_reports,
            analyst_reports=analyst_reports,
            composite_reports=composite_reports,
            load_errors=load_errors,
        )

        payload = {
            "run_id": _run_id("domain_orchestrator"),
            "created_at": utc_now_iso(),
            "mode": "domain_orchestrator",
            "domain_id": domain_id,
            "domain_label": profile.display_name,
            "inputs": {
                "as_of": effective_as_of,
                "tickers": resolved_tickers,
                "phase": phase,
                "include_profile_agents": include_profile_agents,
                "registry_path": str(self.registry_path),
                "data_preflight_error": data_preflight_error,
            },
            "summary": summary,
            "pipeline_branch": {
                "agent_count": len(pipeline_reports),
                "synthetic_blocked_count": len(synthetic_reports),
                "load_errors": load_errors,
                "reports": pipeline_reports,
                "synthetic_reports": synthetic_reports,
            },
            "analyst_branch": {
                "report_count": len(analyst_reports),
                "reports": analyst_reports,
            },
            "composite_pipeline_manager": {
                "report_count": len(composite_reports),
                "reports": composite_reports,
            },
            "explicit_non_actions": _explicit_non_actions(),
        }

        if save:
            saved_paths = ReviewArtifactWriter(self.output_dir / domain_id).write(
                payload=payload,
                markdown=render_domain_orchestrator_markdown(payload),
                run_id=payload["run_id"],
            )
            payload["saved_paths"] = saved_paths
        return json_ready(payload)

    def _context_set_gate(
        self,
        *,
        domain_id: str,
        profile: Any,
        effective_as_of: str,
        resolved_tickers: list[str],
        context_set_path: str | Path | None,
        phase: str,
        include_profile_agents: bool,
    ) -> dict[str, Any]:
        context_receipt: dict[str, Any] = {}
        proposals: list[dict[str, Any]] = []
        if context_set_path is None:
            status = "domain_orchestrator_waiting_for_context_set"
            blockers = ["verified_domain_context_set_missing"]
            missing_families = [
                "news",
                "official_policy",
                "macro",
                "fundamentals",
                "sector_market",
                "pipeline_context",
            ]
        else:
            try:
                verified = load_verified_domain_context_set(
                    context_set_path,
                    expected_domain_id=domain_id,
                    expected_analysis_cutoff=effective_as_of,
                )
                context_receipt = {
                    "path": verified["metadata"]["domain_context_set_path"],
                    "sha256": verified["metadata"]["domain_context_set_sha256"],
                    "candidate_set_sha256": verified["candidate_set_sha256"],
                    "status": verified["status"],
                    "complete": verified["complete"],
                    "binding_accepted": verified["binding_accepted"],
                }
                missing_families = list(verified["missing_families"])
                proposals = list(verified.get("collection_proposals") or [])
                if not verified["complete"]:
                    status = "domain_orchestrator_waiting_for_context_families"
                    blockers = [
                        f"domain_context_family_missing:{family}"
                        for family in missing_families
                    ]
                else:
                    status = "domain_orchestrator_waiting_for_binding_acceptance"
                    blockers = ["domain_context_set_binding_not_accepted"]
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                status = "domain_orchestrator_blocked_invalid_context_set"
                blockers = ["domain_context_set_verification_failed"]
                missing_families = []
                context_receipt = {
                    "path": str(Path(context_set_path).resolve()),
                    "verification_error": f"{type(exc).__name__}: {exc}",
                    "complete": False,
                    "binding_accepted": False,
                }
        summary = {
            "automation_status": status,
            "domain_id": domain_id,
            "domain_label": profile.display_name,
            "tickers": resolved_tickers,
            "context_gate_blockers": blockers,
            "missing_context_families": missing_families,
            "pipeline_agent_count": 0,
            "synthetic_blocked_count": 0,
            "analyst_report_count": 0,
            "composite_manager_count": 0,
            "blocked_agents": [],
            "load_error_count": 0,
            "can_propose_context_acquisition": bool(proposals),
            "can_invoke_domain_analysis": False,
            "can_trade": False,
            "can_write_learning_memory": False,
            "can_write_production_config": False,
            "can_create_recommendation": False,
        }
        return {
            "run_id": _run_id("domain_orchestrator"),
            "created_at": utc_now_iso(),
            "mode": "domain_orchestrator",
            "contract": "dean_domain_orchestrator_state_v1",
            "domain_id": domain_id,
            "domain_label": profile.display_name,
            "inputs": {
                "as_of": effective_as_of,
                "tickers": resolved_tickers,
                "phase": phase,
                "include_profile_agents": include_profile_agents,
                "registry_path": str(self.registry_path),
                "context_set_path": (
                    str(Path(context_set_path).resolve())
                    if context_set_path is not None
                    else None
                ),
                "allow_legacy_unbound_context": False,
            },
            "summary": summary,
            "context_set_gate": {
                "status": status,
                "receipt": context_receipt,
                "blockers": blockers,
                "acquisition_proposals": proposals,
                "analyst_invocation_authorized": False,
            },
            "pipeline_branch": {
                "agent_count": 0,
                "synthetic_blocked_count": 0,
                "load_errors": {},
                "reports": [],
                "synthetic_reports": [],
            },
            "analyst_branch": {"report_count": 0, "reports": []},
            "composite_pipeline_manager": {"report_count": 0, "reports": []},
            "explicit_non_actions": _explicit_non_actions(),
        }

    async def _run_analyst_branch(
        self,
        *,
        domain_id: str,
        context: MarketContext,
        profile: Any,
        configured_domain_agents: list[Any],
        include_profile_agents: bool,
    ) -> list[dict[str, Any]]:
        reports: list[dict[str, Any]] = []
        # Reuse the instance from the one registry load above.  A second load
        # previously hid duplicate execution and reset registry diagnostics.
        domain_agent = configured_domain_agents[0] if configured_domain_agents else None
        if domain_agent is None:
            reports.append(
                {
                    "agent_name": f"{domain_id}_analyst",
                    "branch": "pipeline",
                    "verdict": "blocked",
                    "reasons": [
                        "Matching DomainAnalystAgent is not enabled and configured in AgentRegistry."
                    ],
                    "metrics_snapshot": {
                        "synthetic": True,
                        "domain_id": domain_id,
                        "direct_unconfigured_fallback_used": False,
                    },
                }
            )
        else:
            reports.extend(
                _report_to_dict(report)
                for report in await PipelineBranch(
                    [domain_agent],
                    soft_mode=True,
                ).run(context)
            )

        profile_agents = get_profile_agents(domain_id)
        if include_profile_agents and profile_agents:
            reports.extend(
                await self._run_profile_agents(context, list(profile_agents))
            )
        return reports

    async def _run_profile_agents(
        self, context: MarketContext, profile_agents: list[str]
    ) -> list[dict[str, Any]]:
        from dean_os.analyst_core.analyst_profile_orchestrator import (
            DOMAIN_PROFILE_AGENTS,
            AnalystProfileOrchestrator,
        )

        agents: list[Any] = []
        for name in profile_agents:
            agent_cls = DOMAIN_PROFILE_AGENTS.get(name)
            if agent_cls is None:
                continue
            agents.append(agent_cls(name=name, config={"horizon_years": 1.0}))
        # AnalystProfileOrchestrator is available for richer runs; we intentionally
        # use the lightweight per-agent path above to avoid requiring an evidence
        # pack. The orchestrator is kept as a thin composer.
        _ = AnalystProfileOrchestrator
        return [
            _report_to_dict(report)
            for report in await AnalyticalBranch(agents).run_parallel(context)
        ]

    async def _run_composite_managers(
        self, managers: list[Any], context: MarketContext
    ) -> list[dict[str, Any]]:
        """Step 3: run composite domain pipeline managers (e.g. pipeline_manager).

        These are separated from the generic pipeline-control loop because they
        consume the domain analyst's runtime artifact and should run after the
        analyst branch. They are still loaded via `AgentRegistry`, so veto and
        prerequisite logic is unchanged.
        """
        return [
            _report_to_dict(report)
            for report in await PipelineBranch(
                managers,
                soft_mode=True,
            ).run(context)
        ]

    def _summary(
        self,
        *,
        domain_id: str,
        profile: Any,
        resolved_tickers: list[str],
        pipeline_reports: list[dict[str, Any]],
        synthetic_reports: list[dict[str, Any]],
        analyst_reports: list[dict[str, Any]],
        composite_reports: list[dict[str, Any]],
        load_errors: dict[str, str],
    ) -> dict[str, Any]:
        blocked = [
            r.get("agent_name")
            for r in (*pipeline_reports, *synthetic_reports, *analyst_reports, *composite_reports)
            if r.get("verdict") == "blocked"
        ]
        status = "domain_orchestrator_completed"
        if blocked:
            status = "domain_orchestrator_completed_with_blocks"
        return {
            "automation_status": status,
            "domain_id": domain_id,
            "domain_label": profile.display_name,
            "tickers": resolved_tickers,
            "pipeline_agent_count": len(pipeline_reports),
            "synthetic_blocked_count": len(synthetic_reports),
            "analyst_report_count": len(analyst_reports),
            "composite_manager_count": len(composite_reports),
            "blocked_agents": blocked,
            "load_error_count": len(load_errors),
            "can_trade": False,
            "can_write_learning_memory": False,
            "can_write_production_config": False,
            "can_create_recommendation": False,
        }


def _report_to_dict(report: Any) -> dict[str, Any]:
    if hasattr(report, "model_dump"):
        return report.model_dump(mode="json")
    if isinstance(report, dict):
        return report
    return {"verdict": "blocked", "error": str(report)}


def _explicit_non_actions() -> list[str]:
    return [
        "No live collector is started.",
        "No model training, Stage 7 evaluation, replay, backtest, or tuning run is started.",
        "No learning memory, model promotion, or production config write is performed.",
        "No recommendation, allocation, paper order, broker call, or live trade is generated.",
        "The domain analyst and pipeline agents are review-only; can_trade is always False.",
    ]


def render_domain_orchestrator_markdown(payload: dict[str, Any]) -> str:
    summary = payload.get("summary", {})
    lines = [
        "# DEAN-OS Domain Orchestrator",
        "",
        f"- Run ID: `{payload.get('run_id')}`",
        f"- Domain: `{summary.get('domain_label')}` (`{payload.get('domain_id')}`)",
        f"- Automation status: `{summary.get('automation_status')}`",
        f"- Tickers: {', '.join(summary.get('tickers', [])) or 'none'}",
        f"- Pipeline agents: {summary.get('pipeline_agent_count')}",
        f"- Synthetic blocked: {summary.get('synthetic_blocked_count')}",
        f"- Analyst reports: {summary.get('analyst_report_count')}",
        f"- Can trade: {summary.get('can_trade')}",
    ]
    gate = payload.get("context_set_gate") or {}
    if gate:
        lines.extend(["", "## Context Set Gate", ""])
        lines.append(f"- Status: `{gate.get('status')}`")
        for blocker in gate.get("blockers") or []:
            lines.append(f"- Blocker: `{blocker}`")
        for proposal in gate.get("acquisition_proposals") or []:
            lines.append(
                "- Acquisition proposal: `{}`; execution authorized: `{}`".format(
                    proposal.get("context_family"),
                    proposal.get("execution_authorized"),
                )
            )
    lines.extend(["", "## Pipeline Branch", ""])
    for report in payload.get("pipeline_branch", {}).get("reports", []):
        lines.append(
            f"- `{report.get('agent_name')}`: {report.get('verdict')} "
            f"(branch={report.get('branch')})"
        )
    if payload.get("pipeline_branch", {}).get("synthetic_reports"):
        lines.append("")
        lines.append("### Synthetic blocked (hard/block agents with missing prerequisites)")
        for report in payload["pipeline_branch"]["synthetic_reports"]:
            lines.append(f"- `{report.get('agent_name')}`: blocked (synthetic)")
    lines.extend(["", "## Analyst Branch", ""])
    for report in payload.get("analyst_branch", {}).get("reports", []):
        lines.append(
            f"- `{report.get('agent_name')}`: {report.get('verdict')} "
            f"(branch={report.get('branch')})"
        )
    lines.extend(["", "## Step 3: Composite Pipeline Manager", ""])
    for report in payload.get("composite_pipeline_manager", {}).get("reports", []):
        lines.append(
            f"- `{report.get('agent_name')}`: {report.get('verdict')} "
            f"(branch={report.get('branch')})"
        )
    lines.extend(["", "## Explicit Non-Actions", ""])
    lines.extend(f"- {item}" for item in payload.get("explicit_non_actions", []))

    # Append Economy Regime Brief if present
    for report in payload.get("analyst_branch", {}).get("reports", []):
        metrics = report.get("metrics_snapshot", {})
        brief = metrics.get("economy_regime_brief")
        if brief:
            lines.extend(["", "---", "", brief])

    return "\n".join(lines).strip() + "\n"


def _run_id(prefix: str) -> str:
    return f"{prefix}_{utc_now_iso().replace(':', '').replace('+', 'Z')}"


def run_sync(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """Synchronous helper for CLI / scripts."""
    return asyncio.run(DomainOrchestrator(**kwargs.pop("orchestrator_kwargs", {})).run(*args, **kwargs))
