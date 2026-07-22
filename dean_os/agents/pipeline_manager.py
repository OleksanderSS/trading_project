from __future__ import annotations

import hashlib
from datetime import datetime
from pathlib import Path
from typing import Any

from dean_os.agents.pipeline_readiness import (
    load_pipeline_readiness,
)
from dean_os.analyst_core.pipeline_manager import PipelineRunResult, SectorPipelineManager
from dean_os.analyst_core.sector_analyst import SectorReport
from dean_os.artifact_writer import ReviewArtifactWriter
from dean_os.base import BaseAgent
from dean_os.schemas import EvidenceItem, MarketContext, PipelineReport, utc_now_iso
from dean_os.utils import sha256_json


class PipelineManagerAgent(BaseAgent):
    version = "0.2.0"
    branch = "pipeline"

    async def run(self, context: MarketContext) -> PipelineReport:
        domain_id: str = self.config.get("domain_id", "semiconductor_ai_infrastructure")
        context_artifacts = _context_artifacts(context, domain_id)
        base_path: str | None = (
            context_artifacts.get("base_artifact_path")
            or self.config.get("base_artifact_path")
        )
        output_dir: str | None = self.config.get("output_dir")
        as_of: str = context.as_of or self.config.get("as_of", "")
        tickers: list[str] | None = context.tickers or self.config.get("tickers")
        horizon_days: int | None = self.config.get("horizon_days")

        pm = SectorPipelineManager(domain_id=domain_id)

        artifact_dirs: dict[str, Path | None] = {}
        if base_path:
            artifact_dirs = pm.discover_artifacts(base_path)

        explicit_paths = {
            "news": (
                context_artifacts.get("news")
                or self.config.get("news_path")
            ),
            "macro": (
                context_artifacts.get("macro")
                or self.config.get("macro_path")
            ),
            "sector_market": (
                context_artifacts.get("sector_market")
                or self.config.get("sector_market_path")
            ),
            "policy": (
                context_artifacts.get("policy")
                or self.config.get("policy_path")
            ),
            "fundamental": (
                context_artifacts.get("fundamental")
                or self.config.get("fundamental_path")
            ),
            "runtime": (
                context_artifacts.get("runtime")
                or self.config.get("runtime_path")
            ),
        }
        resolved_sources = {
            key: str(explicit_paths.get(key) or artifact_dirs.get(key))
            for key in {
                "news",
                "macro",
                "sector_market",
                "policy",
                "fundamental",
                "runtime",
            }
            if explicit_paths.get(key) or artifact_dirs.get(key)
        }
        readiness_paths = {
            "timeframe_lane_readiness": (
                context_artifacts.get("timeframe_lane_readiness")
                or self.config.get("timeframe_lane_readiness_path")
            ),
            "feature_timeframe_audit": (
                context_artifacts.get("feature_timeframe_audit")
                or self.config.get(
                    "feature_timeframe_audit_path"
                )
            ),
            "target_readiness": (
                context_artifacts.get("target_readiness")
                or self.config.get("target_readiness_path")
            ),
            "stage4_review": (
                context_artifacts.get("stage4_review")
                or self.config.get("stage4_review_path")
            ),
            "prediction_review": (
                context_artifacts.get("prediction_review")
                or self.config.get("prediction_review_path")
            ),
            "sector_to_ticker_review": (
                context_artifacts.get("sector_to_ticker_review")
                or self.config.get(
                    "sector_to_ticker_review_path"
                )
            ),
        }
        readiness = load_pipeline_readiness(readiness_paths)
        input_gaps = []
        if not _timezone_aware_as_of(as_of):
            input_gaps.append("timezone-aware as_of is required")
        if not resolved_sources:
            input_gaps.append(
                "at least one explicit or discovered artifact is required"
            )
        if readiness.get("errors"):
            input_gaps.extend(
                f"pipeline readiness invalid: {error}"
                for error in readiness["errors"]
            )
        if input_gaps:
            report = _input_gap_report(
                self.name,
                self.version,
                domain_id,
                input_gaps,
            )
            report.metrics_snapshot["artifact_sources"] = (
                resolved_sources
            )
            bindings = {
                key: _artifact_binding(value)
                for key, value in resolved_sources.items()
            }
            report.metrics_snapshot["artifact_bindings"] = bindings
            report.metrics_snapshot["pipeline_readiness"] = readiness
            report.input_hash = sha256_json(
                {
                    "context_hash": self.context_hash(context),
                    "artifact_bindings": bindings,
                    "pipeline_readiness": readiness,
                }
            )
            report.config_hash = sha256_json(self.config)
            return report

        result: PipelineRunResult = pm.run_analysis(
            artifact_dirs=artifact_dirs if base_path else None,
            news_path=explicit_paths["news"],
            macro_path=explicit_paths["macro"],
            sector_market_path=explicit_paths["sector_market"],
            policy_path=explicit_paths["policy"],
            fundamental_path=explicit_paths["fundamental"],
            runtime_artifact=explicit_paths["runtime"],
            as_of=as_of or "",
            tickers=tickers,
            horizon_days=horizon_days,
            output_dir=None,
        )

        report = _result_to_pipeline(
            agent_name=self.name,
            agent_version=self.version,
            result=result,
            domain_id=domain_id,
            artifact_count=len(resolved_sources),
        )
        report.metrics_snapshot["artifact_sources"] = resolved_sources
        bindings = {
            key: _artifact_binding(value)
            for key, value in resolved_sources.items()
        }
        report.metrics_snapshot["artifact_bindings"] = bindings
        report.metrics_snapshot["pipeline_readiness"] = readiness
        report.metrics_snapshot.setdefault(
            "artifact_count", len(resolved_sources)
        )
        report.metrics_snapshot.setdefault(
            "decision_influence", False
        )
        report.metrics_snapshot.setdefault(
            "supporting_review_only", True
        )
        report.metrics_snapshot.setdefault("can_trade", False)
        if readiness.get("blocking_reasons"):
            report.risks.extend(
                "Pipeline readiness: " + reason
                for reason in readiness["blocking_reasons"]
                if "Pipeline readiness: " + reason
                not in report.risks
            )
            if report.verdict == "clear":
                report.verdict = "caution"
        report.input_hash = sha256_json(
            {
                "context_hash": self.context_hash(context),
                "artifact_bindings": bindings,
                "pipeline_readiness": readiness,
            }
        )
        report.config_hash = sha256_json(self.config)
        if output_dir:
            saved = _save_composite_report(
                report,
                output_dir=output_dir,
            )
            report.metrics_snapshot["saved_paths"] = saved
        return report


def _context_artifacts(
    context: MarketContext,
    domain_id: str,
) -> dict[str, Any]:
    root = context.metadata.get("domain_artifacts", {})
    if not isinstance(root, dict):
        return {}
    scoped = root.get(domain_id, root)
    return scoped if isinstance(scoped, dict) else {}


def _artifact_binding(value: str | Path) -> dict[str, Any]:
    path = Path(value)
    artifact_path = (
        path / "latest.json" if path.is_dir() else path
    )
    if not artifact_path.is_file():
        return {
            "path": str(path),
            "artifact_path": str(artifact_path),
            "available": False,
            "sha256": None,
        }
    return {
        "path": str(path),
        "artifact_path": str(artifact_path),
        "available": True,
        "sha256": hashlib.sha256(
            artifact_path.read_bytes()
        ).hexdigest(),
    }


def _save_composite_report(
    report: PipelineReport,
    *,
    output_dir: str | Path,
) -> dict[str, str]:
    run_id = (
        "pipeline_manager_agent_report_"
        + utc_now_iso().replace(":", "").replace("+", "")
    )
    readiness = report.metrics_snapshot.get(
        "pipeline_readiness", {}
    )
    payload = {
        "run_id": run_id,
        "created_at": utc_now_iso(),
        "mode": "pipeline_manager_agent_report",
        "schema_version": "dean_pipeline_manager_agent_report_v1",
        "agent_report": report.model_dump(mode="json"),
        "artifact_bindings": report.metrics_snapshot.get(
            "artifact_bindings", {}
        ),
        "pipeline_readiness": readiness,
        "safety": {
            "review_only": True,
            "decision_influence": False,
            "can_create_ticker_forecast": False,
            "can_promote_model": False,
            "can_trade": False,
        },
    }
    blockers = readiness.get("blocking_reasons", [])
    markdown = "\n".join(
        [
            "# DEAN-OS Composite Domain Pipeline Report",
            "",
            f"- Agent: `{report.agent_name}`",
            f"- Verdict: `{report.verdict}`",
            (
                "- Domain: `"
                f"{report.metrics_snapshot.get('domain_id')}`"
            ),
            (
                "- Evidence items: "
                f"{report.metrics_snapshot.get('evidence_count', 0)}"
            ),
            (
                "- Artifact sources: "
                f"{report.metrics_snapshot.get('artifact_count', 0)}"
            ),
            (
                "- Pipeline readiness: "
                f"`{readiness.get('status')}`"
            ),
            "- Decision influence: `False`",
            "- Can trade: `False`",
            "",
            "## Pipeline Readiness Blockers",
            "",
            *(
                [f"- {item}" for item in blockers]
                or ["- None recorded."]
            ),
            "",
            "## Reasons",
            "",
            *[f"- {item}" for item in report.reasons],
            "",
            "## Risks",
            "",
            *[f"- {item}" for item in report.risks],
        ]
    )
    return ReviewArtifactWriter(output_dir).write(
        payload=payload,
        markdown=markdown + "\n",
        run_id=run_id,
    )


def _input_gap_report(
    agent_name: str,
    agent_version: str,
    domain_id: str,
    gaps: list[str],
) -> PipelineReport:
    return PipelineReport(
        agent_name=agent_name,
        agent_version=agent_version,
        verdict="needs_more_data",
        confidence=1.0,
        data_quality_score=0.0,
        signal_strength=0.0,
        reasons=list(gaps),
        risks=["Composite domain analysis did not run"],
        evidence=[
            _ev(
                agent_name,
                "audit_finding",
                "pipeline_manager_input",
                "missing_inputs",
                gaps,
            )
        ],
        metrics_snapshot={
            "domain_id": domain_id,
            "agent_role": "composite_domain_pipeline_manager",
            "decision_influence": False,
            "supporting_review_only": True,
            "analysis_executed": False,
            "can_create_ticker_forecast": False,
            "can_trade": False,
        },
    )


def _timezone_aware_as_of(value: str | None) -> bool:
    if not value:
        return False
    try:
        parsed = datetime.fromisoformat(
            str(value).replace("Z", "+00:00")
        )
    except (TypeError, ValueError):
        return False
    return (
        parsed.tzinfo is not None
        and parsed.utcoffset() is not None
    )


def _result_to_pipeline(
    agent_name: str,
    agent_version: str,
    result: PipelineRunResult,
    domain_id: str,
    artifact_count: int,
) -> PipelineReport:
    analysis = result.analysis_result or {}
    report: SectorReport | None = analysis.get("report")

    if result.errors:
        return _error_report(agent_name, agent_version, result)

    if report is None:
        return PipelineReport(
            agent_name=agent_name,
            agent_version=agent_version,
            verdict="needs_more_data",
            confidence=0.5,
            data_quality_score=0.0,
            signal_strength=-0.25,
            reasons=["No analysis result produced — check artifact paths and domain config"],
            risks=["Insufficient data to produce a domain analysis"],
            evidence=[
                _ev(agent_name, "metric", "pipeline_run", "domain_id", domain_id),
                _ev(agent_name, "metric", "pipeline_run", "artifact_count", artifact_count),
            ],
            metrics_snapshot={
                "domain_id": domain_id,
                "agent_role": "composite_domain_pipeline_manager",
                "decision_influence": False,
                "supporting_review_only": True,
                "can_create_ticker_forecast": False,
                "can_trade": False,
                "errors": result.errors,
                "warnings": result.warnings,
            },
        )

    t = report.thesis
    verdict = _verdict_from_recommendation(report.recommendation)
    evidence_count = analysis.get("evidence_count", report.evidence_count)

    reasons: list[str] = [
        f"Pipeline analysis complete — {evidence_count} evidence items from {artifact_count} artifact sources",
    ]
    if t.thesis:
        reasons.append(f"Thesis: {t.thesis}")
    if report.lens_count:
        reasons.append(f"{report.lens_count} lens deltas applied")

    return PipelineReport(
        agent_name=agent_name,
        agent_version=agent_version,
        verdict=verdict,
        confidence=t.confidence,
        data_quality_score=_quality_score(t.data_quality),
        signal_strength=_signal_from_verdict(verdict),
        reasons=reasons,
        risks=list(t.risks or []),
        blind_spots=[
            "PipelineManagerAgent is review-only — no trade signals, no live execution",
            "Analysis depends on available producer artifacts in the configured paths",
        ],
        evidence=[
            _ev(agent_name, "metric", "pipeline_run", "domain_id", domain_id),
            _ev(agent_name, "metric", "pipeline_run", "evidence_count", evidence_count),
            _ev(agent_name, "metric", "pipeline_run", "artifact_count", artifact_count),
            _ev(agent_name, "metric", "pipeline_run", "lens_count", report.lens_count),
            _ev(agent_name, "metric", "pipeline_run", "recommendation", report.recommendation),
        ],
        metrics_snapshot={
            "domain_id": domain_id,
            "agent_role": "composite_domain_pipeline_manager",
            "decision_influence": False,
            "supporting_review_only": True,
            "can_create_ticker_forecast": False,
            "can_trade": False,
            "as_of": result.as_of,
            "evidence_count": evidence_count,
            "artifact_count": artifact_count,
            "lens_count": report.lens_count,
            "classified_event_count": len(report.classified_events),
            "classified_events": [
                item.model_dump(mode="json") if hasattr(item, "model_dump") else item
                for item in report.classified_events
            ],
            "hypothesis_count": len(report.hypotheses),
            "hypotheses": [
                item.model_dump(mode="json") if hasattr(item, "model_dump") else item
                for item in report.hypotheses
            ],
            "evidence_gap_count": len(report.evidence_gaps),
            "evidence_gaps": [
                item.model_dump(mode="json") if hasattr(item, "model_dump") else item
                for item in report.evidence_gaps
            ],
            "transmission_channels": [
                item.model_dump(mode="json") if hasattr(item, "model_dump") else item
                for item in report.transmission_channels
            ],
            "watch_signals": report.watch_signals,
            "regime_context": (
                report.regime_context.model_dump(mode="json")
                if hasattr(report.regime_context, "model_dump")
                else report.regime_context
            ),
            "expectation_gap": report.expectation_gap,
            "recommendation": report.recommendation,
            "stance": t.stance,
            "confidence": t.confidence,
            "basket_status": report.ticker_basket.basket_status if report.ticker_basket else None,
            "evaluation": result.evaluation_result,
            "knowledge": result.knowledge_result,
            "errors": result.errors,
            "warnings": result.warnings,
        },
    )


def _error_report(
    agent_name: str,
    agent_version: str,
    result: PipelineRunResult,
) -> PipelineReport:
    return PipelineReport(
        agent_name=agent_name,
        agent_version=agent_version,
        verdict="blocked",
        confidence=1.0,
        data_quality_score=0.0,
        signal_strength=-1.0,
        reasons=[f"Pipeline failed: {err}" for err in result.errors[:5]],
        risks=["Pipeline errors must be resolved before analysis can proceed"],
        evidence=[
            _ev(agent_name, "audit_finding", "pipeline_run", "errors", result.errors),
        ],
        metrics_snapshot={
            "domain_id": result.domain_id,
            "agent_role": "composite_domain_pipeline_manager",
            "decision_influence": False,
            "supporting_review_only": True,
            "can_create_ticker_forecast": False,
            "can_trade": False,
            "errors": result.errors,
            "warnings": result.warnings,
        },
    )


def _verdict_from_recommendation(rec: str) -> str:
    return {
        "ready_for_review": "clear",
        "partial_ready_for_review": "caution",
        "needs_more_data": "needs_more_data",
        "blocked": "blocked",
    }.get(rec, "caution")


def _quality_score(quality: str | None) -> float:
    return {"strong": 0.9, "medium": 0.65, "weak": 0.35}.get(str(quality).lower(), 0.5)


def _signal_from_verdict(verdict: str) -> float:
    return {
        "clear": 0.3,
        "caution": 0.0,
        "needs_more_data": -0.25,
        "blocked": -0.5,
    }.get(verdict, 0.0)


def _ev(
    agent_name: str,
    source_type: str,
    source: str,
    key: str,
    value: Any,
) -> EvidenceItem:
    return EvidenceItem(source_type=source_type, source=source, key=key, value=value)
