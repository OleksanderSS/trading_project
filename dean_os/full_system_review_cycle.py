from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from dean_os.accumulation_authorization_ledger import AccumulationAuthorizationLedger
from dean_os.agents.pipeline_manager import PipelineManagerAgent
from dean_os.artifact_writer import ReviewArtifactWriter
from dean_os.current_system_manifest import CurrentSystemManifestBuilder
from dean_os.schemas import MarketContext, PipelineReport, utc_now_iso
from dean_os.system_topology import load_system_topology


class FullSystemReviewCycle:
    """Execute the bounded manager+analyst slice and register it in the system DAG."""

    contract = "dean_full_system_review_cycle_v1"

    def __init__(
        self,
        output_dir: str | Path = "reports/dean_os/full_system_review_cycle_current",
        manager_output_dir: str | Path = "reports/dean_os/composite_domain_pipeline_current",
        topology_path: str | Path = "dean_os/config/system_topology.yaml",
        authorization_ledger_path: str | Path = "data/dean_os/accumulation_authorization_ledger.jsonl",
    ) -> None:
        self.output_dir = Path(output_dir)
        self.manager_output_dir = Path(manager_output_dir)
        self.topology_path = Path(topology_path)
        self.authorization_ledger_path = Path(authorization_ledger_path)

    async def run(
        self,
        *,
        domain_id: str,
        as_of: str,
        artifact_paths: dict[str, str | Path],
        timeframe_lane_readiness_path: str | Path,
        tickers: list[str] | None = None,
        horizon_days: int | None = None,
        save: bool = True,
    ) -> dict[str, Any]:
        config = {
            "branch": "pipeline",
            "domain_id": domain_id,
            "horizon_days": horizon_days,
            "news_path": _text(artifact_paths.get("news")),
            "macro_path": _text(artifact_paths.get("macro")),
            "sector_market_path": _text(artifact_paths.get("sector_market")),
            "policy_path": _text(artifact_paths.get("policy")),
            "fundamental_path": _text(artifact_paths.get("fundamental")),
            "timeframe_lane_readiness_path": str(timeframe_lane_readiness_path),
            "output_dir": str(self.manager_output_dir),
            "proposal_only": True,
            "decision_influence": False,
            "run_phases": ["pre_trade"],
        }
        manager = PipelineManagerAgent(
            name=f"{domain_id}_pipeline_manager",
            config=config,
        )
        report = await manager.run(
            MarketContext(
                phase="pre_trade",
                as_of=as_of,
                tickers=list(tickers or []),
            )
        )
        system_manifest = CurrentSystemManifestBuilder(
            topology_path=self.topology_path,
            authorization_ledger_path=self.authorization_ledger_path,
        ).build(as_of=as_of, domain_id=domain_id, save=True)
        payload = compose_full_system_review_cycle(
            report=report,
            system_manifest=system_manifest,
            topology_path=self.topology_path,
            authorization_ledger_path=self.authorization_ledger_path,
            artifact_paths=artifact_paths,
            timeframe_lane_readiness_path=timeframe_lane_readiness_path,
        )
        if save:
            payload["saved_paths"] = ReviewArtifactWriter(self.output_dir).write(
                payload=payload,
                markdown=_markdown(payload),
                run_id=payload["run_id"],
            )
        return payload


def compose_full_system_review_cycle(
    *,
    report: PipelineReport,
    system_manifest: dict[str, Any],
    topology_path: str | Path,
    authorization_ledger_path: str | Path,
    artifact_paths: dict[str, str | Path],
    timeframe_lane_readiness_path: str | Path,
) -> dict[str, Any]:
    topology = load_system_topology(topology_path)
    metrics = report.metrics_snapshot
    readiness = metrics.get("pipeline_readiness") or {}
    analysis_executed = bool(
        metrics.get("evidence_count")
        and not metrics.get("errors")
        and metrics.get("agent_role") == "composite_domain_pipeline_manager"
    )
    ledger_status = AccumulationAuthorizationLedger(authorization_ledger_path).status()
    executed = [
        {
            "branch_id": "artifact_intake",
            "execution_mode": "composite_executed",
            "status": "completed" if metrics.get("artifact_count") else "blocked",
            "summary": {"artifact_count": metrics.get("artifact_count", 0)},
        },
        {
            "branch_id": "evidence_intelligence",
            "execution_mode": "composite_executed",
            "status": "completed" if metrics.get("evidence_count") else "blocked",
            "summary": {"accepted_evidence_count": metrics.get("evidence_count", 0)},
        },
        {
            "branch_id": "pipeline_control",
            "execution_mode": "composite_executed",
            "status": "completed" if readiness.get("is_ready") else "blocked",
            "summary": readiness,
        },
        {
            "branch_id": "domain_analysis",
            "execution_mode": "composite_executed",
            "status": "completed" if analysis_executed else "blocked",
            "summary": {
                "recommendation": metrics.get("recommendation"),
                "stance": metrics.get("stance"),
                "evidence_count": metrics.get("evidence_count", 0),
                "lens_count": metrics.get("lens_count", 0),
                "hypothesis_count": metrics.get("hypothesis_count", 0),
                "evidence_gap_count": metrics.get("evidence_gap_count", 0),
            },
        },
    ]
    deferred = ["world_model", "replay_evaluation", "governance_review"]
    observed = [
        {
            "branch_id": branch_id,
            "execution_mode": "prior_artifact_observed",
            "status": "downstream_refresh_required",
        }
        for branch_id in deferred
    ]
    observed.extend(
        [
            {
                "branch_id": "operations_authorization",
                "execution_mode": "ledger_observed",
                "status": "completed",
                "summary": ledger_status,
            },
            {
                "branch_id": "system_audit",
                "execution_mode": "cycle_manifest_assembled",
                "status": "completed",
            },
        ]
    )
    created_at = utc_now_iso()
    manager_path = (metrics.get("saved_paths") or {}).get("latest_json")
    return {
        "run_id": "full_system_review_cycle_" + created_at.replace(":", "").replace("+00:00", "Z"),
        "created_at": created_at,
        "mode": "full_system_review_cycle",
        "contract": FullSystemReviewCycle.contract,
        "topology": {
            "path": str(topology_path),
            "sha256": topology.topology_sha256,
            "branch_count": len(topology.execution_order()),
        },
        "inputs": {
            "artifacts": {
                name: _binding(path) for name, path in sorted(artifact_paths.items())
            },
            "timeframe_lane_readiness": _binding(timeframe_lane_readiness_path),
        },
        "summary": {
            "cycle_status": (
                "analysis_cycle_completed_downstream_refresh_required"
                if analysis_executed and readiness.get("is_ready")
                else "analysis_cycle_blocked"
            ),
            "analysis_executed": analysis_executed,
            "pipeline_context_ready": bool(readiness.get("is_ready")),
            "recommendation": metrics.get("recommendation"),
            "evidence_count": metrics.get("evidence_count", 0),
            "lens_count": metrics.get("lens_count", 0),
            "hypothesis_count": metrics.get("hypothesis_count", 0),
            "evidence_gap_count": metrics.get("evidence_gap_count", 0),
            "downstream_hash_bound_to_this_cycle": False,
            "downstream_refresh_required": deferred,
            "authorization_ledger_record_count": ledger_status["record_count"],
            "can_register_new_replay_tasks": False,
            "can_write_learning_memory": False,
            "can_trade": False,
        },
        "manager_report": {
            "path": manager_path,
            "sha256": _sha256(Path(manager_path)) if manager_path and Path(manager_path).exists() else None,
            "agent_report": report.model_dump(mode="json"),
        },
        "system_manifest": {
            "path": (system_manifest.get("saved_paths") or {}).get("latest_json"),
            "sha256": system_manifest.get("manifest_sha256"),
            "status": system_manifest.get("status"),
        },
        "branch_records": executed + observed,
        "safety": {
            "review_only": True,
            "composite_execution_disclosed": True,
            "independent_branch_execution_claimed": False,
            "collector_execution_performed": False,
            "pipeline_stage_execution_performed": False,
            "outcome_evaluation_performed": False,
            "replay_registration_performed": False,
            "authorization_write_performed": False,
            "learning_write_performed": False,
            "broker_access_performed": False,
            "can_trade": False,
        },
    }


def _binding(path_value: str | Path) -> dict[str, Any]:
    path = Path(path_value)
    return {
        "path": str(path),
        "exists": path.is_file(),
        "sha256": _sha256(path) if path.is_file() else None,
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _text(value: Any) -> str | None:
    return str(value) if value else None


def _markdown(payload: dict[str, Any]) -> str:
    summary = payload["summary"]
    return (
        "# Full System Review Cycle\n\n"
        f"- Status: `{summary['cycle_status']}`\n"
        f"- Evidence: `{summary['evidence_count']}`\n"
        f"- Lenses: `{summary['lens_count']}`\n"
        f"- Recommendation: `{summary['recommendation']}`\n"
        f"- Downstream refresh required: `{', '.join(summary['downstream_refresh_required'])}`\n"
        "- Authorization ledger registered: `true`\n"
        "- Can trade: `false`\n"
    )


__all__ = ["FullSystemReviewCycle", "compose_full_system_review_cycle"]
