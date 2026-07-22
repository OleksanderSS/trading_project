from __future__ import annotations

from pathlib import Path
from typing import Any

from dean_os.analyst_core.analyst_evidence_pack import AnalystEvidencePackRunner
from dean_os.analysts import get_domain_profile
from dean_os.artifact_writer import ReviewArtifactWriter
from dean_os.analyst_core.domain_analyst_event_interpretation_packet import DomainAnalystEventInterpretationPacket
from dean_os.analyst_core.domain_analyst_forecast_review_packet import DomainAnalystForecastReviewPacket
from dean_os.analyst_core.domain_analyst_instance_contract import DomainAnalystInstanceContract
from dean_os.analyst_core.domain_analyst_intake_packet import DomainAnalystIntakePacket
from dean_os.analyst_core.domain_analyst_regime_scenario_packet import DomainAnalystRegimeScenarioPacket
from dean_os.analyst_core.domain_analyst_template_standardization_packet import DomainAnalystTemplateStandardizationPacket
from dean_os.analyst_core.domain_analyst_thesis_review_packet import DomainAnalystThesisReviewPacket
from dean_os.schemas import utc_now_iso
from dean_os.source_evidence_validation_gate import SourceEvidenceValidationGate
from dean_os.utils import json_ready

DEFAULT_DOMAIN_ID = "semiconductor_ai_infrastructure"
DEFAULT_NEWS_DATA_PATHS = ["data/processed/features/news_data.parquet"]
DEFAULT_MACRO_DATA_PATHS = ["data/processed/features/macro_data.parquet"]
DEFAULT_ARCHITECTURE_MAP_JSON = "reports/dean_os/current_architecture_map_current/latest.json"


class DomainAnalystVerticalSliceRun:
    """Run one full review-only domain analyst slice from local evidence.

    This is a convenience orchestrator for the analyst branch only. It does not
    accept the template, clone profiles, run live collectors, promote learning,
    recommend, allocate, or trade.
    """

    def __init__(self, output_dir: str | Path = "reports/dean_os/domain_analyst_vertical_slice_current"):
        self.output_dir = Path(output_dir)

    def build(
        self,
        *,
        domain_id: str = DEFAULT_DOMAIN_ID,
        evidence_pack_json: str | Path | None = None,
        source_gate_json: str | Path | None = None,
        pipeline_context_json: str | Path | None = None,
        materials_paths: list[str | Path] | None = None,
        news_data_paths: list[str | Path] | None = None,
        macro_data_paths: list[str | Path] | None = None,
        source_routing_path: str | Path | None = None,
        tickers: list[str] | None = None,
        sectors: list[str] | None = None,
        tags: list[str] | None = None,
        sector_keywords: list[str] | None = None,
        start_at: str | None = None,
        end_at: str | None = None,
        as_of: str | None = None,
        horizon_days: int | None = None,
        max_rows_per_table: int = 200,
        max_documents: int = 500,
        max_items: int = 200,
        max_event_interpretations: int = 80,
        architecture_map_json: str | Path | None = DEFAULT_ARCHITECTURE_MAP_JSON,
        save: bool = True,
    ) -> dict[str, Any]:
        profile = get_domain_profile(domain_id)
        resolved_sectors = _strings(sectors or [profile.sector_label or domain_id])
        resolved_keywords = _strings(sector_keywords or profile.sector_keywords)
        resolved_tags = _strings(tags or ["domain_analyst_vertical_slice"])

        evidence = self._build_or_load_evidence_pack(
            evidence_pack_json=evidence_pack_json,
            materials_paths=materials_paths or [],
            news_data_paths=news_data_paths,
            macro_data_paths=macro_data_paths,
            source_routing_path=source_routing_path,
            tickers=tickers or [],
            sectors=resolved_sectors,
            tags=resolved_tags,
            sector_keywords=resolved_keywords,
            start_at=start_at,
            end_at=end_at,
            as_of=as_of,
            max_rows_per_table=max_rows_per_table,
            max_documents=max_documents,
        )
        evidence_pack_path = evidence["path"]

        source_gate = self._build_or_load_source_gate(source_gate_json=source_gate_json, evidence_pack_path=evidence_pack_path)
        source_gate_path = source_gate["path"]

        event_interpretation = DomainAnalystEventInterpretationPacket(self.output_dir / "event_interpretation").build(
            evidence_pack_json=evidence_pack_path,
            pipeline_context_json=pipeline_context_json,
            domain_id=domain_id,
            max_events=max_event_interpretations,
        )
        event_interpretation_path = event_interpretation["saved_paths"]["latest_json"]

        regime_scenario = DomainAnalystRegimeScenarioPacket(self.output_dir / "regime_scenario").build(
            event_interpretation_json=event_interpretation_path,
            domain_id=domain_id,
            max_events=max_event_interpretations,
        )
        regime_scenario_path = regime_scenario["saved_paths"]["latest_json"]

        intake = DomainAnalystIntakePacket(self.output_dir / "domain_intake").build(
            evidence_pack_json=evidence_pack_path,
            source_gate_json=source_gate_path,
            domain_id=domain_id,
            tickers=tickers or [],
            sectors=resolved_sectors,
            horizon_days=horizon_days or profile.horizon_days_default,
            as_of=as_of,
            max_items=max_items,
        )
        intake_path = intake["saved_paths"]["latest_json"]

        instance = DomainAnalystInstanceContract(self.output_dir / "instance_contract").build(
            evidence_pack_json=evidence_pack_path,
            source_gate_json=source_gate_path,
            domain_intake_json=intake_path,
            architecture_map_json=architecture_map_json,
        )
        instance_path = instance["saved_paths"]["latest_json"]

        thesis = DomainAnalystThesisReviewPacket(self.output_dir / "thesis_review").build(
            domain_intake_json=intake_path,
            domain_instance_contract_json=instance_path,
            regime_scenario_json=regime_scenario_path,
            architecture_map_json=architecture_map_json,
        )
        thesis_path = thesis["saved_paths"]["latest_json"]

        forecast = DomainAnalystForecastReviewPacket(self.output_dir / "forecast_review").build(
            domain_thesis_review_json=thesis_path,
            vertical_slice_json=None,
            regime_scenario_json=regime_scenario_path,
        )
        forecast_path = forecast["saved_paths"]["latest_json"]

        template = DomainAnalystTemplateStandardizationPacket(self.output_dir / "template_standardization").build(
            domain_instance_contract_json=instance_path,
            domain_thesis_review_json=thesis_path,
            regime_scenario_json=regime_scenario_path,
            architecture_map_json=architecture_map_json,
        )
        template_path = template["saved_paths"]["latest_json"]

        payload = {
            "run_id": _run_id("domain_analyst_vertical_slice"),
            "created_at": utc_now_iso(),
            "mode": "domain_analyst_vertical_slice_run",
            "inputs": {
                "domain_id": domain_id,
                "evidence_pack_json": str(evidence_pack_json) if evidence_pack_json else None,
                "source_gate_json": str(source_gate_json) if source_gate_json else None,
                "pipeline_context_json": str(pipeline_context_json) if pipeline_context_json else None,
                "materials_paths": [str(path) for path in materials_paths or []],
                "news_data_paths": [str(path) for path in _paths(news_data_paths, DEFAULT_NEWS_DATA_PATHS)],
                "macro_data_paths": [str(path) for path in _paths(macro_data_paths, DEFAULT_MACRO_DATA_PATHS)],
                "tickers": _strings(tickers or []),
                "sectors": resolved_sectors,
                "tags": resolved_tags,
                "sector_keywords": resolved_keywords,
                "start_at": start_at,
                "end_at": end_at,
                "as_of": as_of,
                "horizon_days": horizon_days or profile.horizon_days_default,
                "max_rows_per_table": max_rows_per_table,
                "max_documents": max_documents,
                "max_items": max_items,
                "max_event_interpretations": max_event_interpretations,
                "architecture_map_json": str(architecture_map_json) if architecture_map_json else None,
            },
            "summary": _summary(
                evidence=evidence["payload"],
                source_gate=source_gate["payload"],
                event_interpretation=event_interpretation,
                regime_scenario=regime_scenario,
                intake=intake,
                instance=instance,
                thesis=thesis,
                forecast=forecast,
                template=template,
                evidence_source=evidence["source"],
            ),
            "branch_readiness": _branch_readiness(intake, event_interpretation, regime_scenario, instance, thesis, forecast, template),
            "artifact_paths": {
                "evidence_pack_json": str(evidence_pack_path),
                "source_gate_json": str(source_gate_path),
                "event_interpretation_json": str(event_interpretation_path),
                "regime_scenario_json": str(regime_scenario_path),
                "domain_intake_json": str(intake_path),
                "domain_instance_contract_json": str(instance_path),
                "domain_thesis_review_json": str(thesis_path),
                "forecast_review_json": str(forecast_path),
                "template_standardization_json": str(template_path),
            },
            "synthetic_fixture_audit": _synthetic_fixture_audit(evidence["payload"], source_gate["payload"], event_interpretation, regime_scenario, intake, instance, thesis, template),
            "manual_gate": _manual_gate(template),
            "recommended_next_steps": _recommended_next_steps(template, intake, forecast, event_interpretation, regime_scenario),
            "explicit_non_actions": _explicit_non_actions(),
        }
        if save:
            saved_paths = ReviewArtifactWriter(self.output_dir).write(
                payload=payload,
                markdown=render_domain_analyst_vertical_slice_markdown(payload),
                run_id=payload["run_id"],
            )
            payload["saved_paths"] = saved_paths
        return json_ready(payload)

    def _build_or_load_evidence_pack(
        self,
        *,
        evidence_pack_json: str | Path | None,
        materials_paths: list[str | Path],
        news_data_paths: list[str | Path] | None,
        macro_data_paths: list[str | Path] | None,
        source_routing_path: str | Path | None,
        tickers: list[str],
        sectors: list[str],
        tags: list[str],
        sector_keywords: list[str],
        start_at: str | None,
        end_at: str | None,
        as_of: str | None,
        max_rows_per_table: int,
        max_documents: int,
    ) -> dict[str, Any]:
        if evidence_pack_json:
            path = Path(evidence_pack_json)
            return {"source": "supplied_local_evidence_pack", "path": path, "payload": _load_json(path)}
        payload = AnalystEvidencePackRunner(self.output_dir / "evidence_pack").run(
            materials_paths=materials_paths,
            news_data_paths=_paths(news_data_paths, DEFAULT_NEWS_DATA_PATHS),
            macro_data_paths=_paths(macro_data_paths, DEFAULT_MACRO_DATA_PATHS),
            source_routing_path=source_routing_path,
            tickers=tickers,
            sectors=sectors,
            tags=tags,
            start_at=start_at,
            end_at=end_at,
            as_of=as_of,
            max_rows_per_table=max_rows_per_table,
            max_documents=max_documents,
            sector_keywords=sector_keywords,
        )
        return {
            "source": "built_from_local_data_paths",
            "path": Path(payload["saved_paths"]["latest_json"]),
            "payload": payload,
        }

    def _build_or_load_source_gate(
        self,
        *,
        source_gate_json: str | Path | None,
        evidence_pack_path: str | Path,
    ) -> dict[str, Any]:
        if source_gate_json:
            path = Path(source_gate_json)
            return {"path": path, "payload": _load_json(path)}
        payload = SourceEvidenceValidationGate(self.output_dir / "source_gate").build(source_json=evidence_pack_path)
        return {"path": Path(payload["saved_paths"]["latest_json"]), "payload": payload}


def render_domain_analyst_vertical_slice_markdown(payload: dict[str, Any]) -> str:
    summary = payload.get("summary", {})
    readiness = payload.get("branch_readiness", {})
    audit = payload.get("synthetic_fixture_audit", {})
    paths = payload.get("artifact_paths", {})
    lines = [
        "# DEAN-OS Domain Analyst Vertical Slice Run",
        "",
        f"- Run ID: `{payload.get('run_id')}`",
        f"- Run status: `{summary.get('run_status')}`",
        f"- Domain: `{summary.get('domain_id')}`",
        f"- Evidence source: `{summary.get('evidence_source')}`",
        f"- Documents: {summary.get('document_count')}",
        f"- Evidence items: {summary.get('evidence_item_count')}",
        f"- Source gate: `{summary.get('source_gate_status')}`",
        f"- Event interpretation: `{summary.get('event_interpretation_status')}` ({summary.get('event_packet_count')} packets)",
        f"- Regime scenario: `{summary.get('regime_scenario_status')}` nodes={summary.get('scenario_node_count')} gaps={summary.get('scenario_evidence_gap_count')}",
        f"- Pipeline context: `{summary.get('pipeline_context_status')}` tags={summary.get('pipeline_context_tag_count')}",
        f"- Pipeline crisis-pattern events: {summary.get('pipeline_crisis_pattern_event_count')}",
        f"- Intake: `{summary.get('intake_status')}`",
        f"- Instance: `{summary.get('instance_status')}`",
        f"- Thesis review: `{summary.get('thesis_review_status')}`",
        f"- Forecast review: `{summary.get('forecast_review_status')}`",
        f"- Template candidate: `{summary.get('template_candidate_status')}`",
        f"- Manual acceptance required: {summary.get('manual_acceptance_required')}",
        f"- Can mark template accepted now: {summary.get('can_mark_template_accepted_now')}",
        f"- Can scale to other domains now: {summary.get('can_scale_to_other_domains_now')}",
        f"- Can trade: {summary.get('can_trade')}",
        "",
        "## Branch Readiness",
        "",
    ]
    for item in readiness.get("steps", []):
        lines.append(f"- `{item.get('step_id')}`: {item.get('status')} - {item.get('note')}")
    lines.extend(
        [
            "",
            "## Synthetic / Fixture Audit",
            "",
            f"- Has synthetic evidence marker: {audit.get('has_synthetic_marker')}",
            f"- Has fixture evidence marker: {audit.get('has_fixture_marker')}",
            f"- Has smoke label: {audit.get('has_smoke_label')}",
            f"- Evidence usable for analyst review: {audit.get('evidence_usable_for_analyst_review')}",
            "",
            "## Artifact Paths",
            "",
        ]
    )
    for key, value in paths.items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(["", "## Manual Gate", ""])
    for item in payload.get("manual_gate", {}).get("checklist", []):
        lines.append(f"- {item}")
    lines.extend(["", "## Recommended Next Steps", ""])
    lines.extend(f"- {item}" for item in payload.get("recommended_next_steps", []))
    lines.extend(["", "## Explicit Non-Actions", ""])
    lines.extend(f"- {item}" for item in payload.get("explicit_non_actions", []))
    return "\n".join(lines).strip() + "\n"


def _summary(
    *,
    evidence: dict[str, Any],
    source_gate: dict[str, Any],
    event_interpretation: dict[str, Any],
    regime_scenario: dict[str, Any],
    intake: dict[str, Any],
    instance: dict[str, Any],
    thesis: dict[str, Any],
    forecast: dict[str, Any],
    template: dict[str, Any],
    evidence_source: str,
) -> dict[str, Any]:
    coverage = evidence.get("coverage", {})
    intake_summary = intake.get("summary", {})
    event_summary = event_interpretation.get("summary", {})
    regime_summary = regime_scenario.get("summary", {})
    instance_summary = instance.get("summary", {})
    thesis_summary = thesis.get("summary", {})
    forecast_summary = forecast.get("summary", {})
    template_summary = template.get("summary", {})
    return {
        "run_status": _run_status(source_gate, event_interpretation, regime_scenario, intake, instance, thesis, forecast, template),
        "domain_id": template_summary.get("domain_id") or intake_summary.get("domain_id"),
        "evidence_source": evidence_source,
        "document_count": coverage.get("document_count", intake_summary.get("document_count")),
        "source_types": coverage.get("by_source_type", {}),
        "data_quality": coverage.get("data_quality"),
        "source_gate_status": source_gate.get("summary", {}).get("gate_status"),
        "event_interpretation_status": event_summary.get("packet_status"),
        "event_packet_count": event_summary.get("event_packet_count"),
        "event_review_required_count": event_summary.get("review_required_count"),
        "event_high_materiality_count": event_summary.get("high_materiality_count"),
        "pipeline_context_supplied": event_summary.get("pipeline_context_supplied"),
        "pipeline_context_status": event_summary.get("pipeline_context_status"),
        "pipeline_context_tag_count": event_summary.get("pipeline_context_tag_count"),
        "pipeline_context_tags": event_summary.get("pipeline_context_tags", []),
        "pipeline_news_context_classified_count": event_summary.get("pipeline_news_context_classified_count"),
        "pipeline_crisis_pattern_event_count": event_summary.get("pipeline_crisis_pattern_event_count"),
        "pipeline_learned_pattern_event_count": event_summary.get("pipeline_learned_pattern_event_count"),
        "regime_scenario_status": regime_summary.get("packet_status"),
        "scenario_node_count": regime_summary.get("scenario_node_count"),
        "scenario_edge_count": regime_summary.get("scenario_edge_count"),
        "scenario_probability_mass_valid": regime_summary.get("probability_mass_valid"),
        "scenario_evidence_gap_count": regime_summary.get("evidence_gap_count"),
        "intake_status": intake_summary.get("intake_status"),
        "instance_status": instance_summary.get("instance_status"),
        "thesis_review_status": thesis_summary.get("packet_status"),
        "forecast_review_status": forecast_summary.get("packet_status"),
        "forecast_candidate_count": forecast_summary.get("forecast_candidate_count"),
        "analyst_control_plane_count": forecast_summary.get("analyst_control_plane_count"),
        "template_candidate_status": template_summary.get("candidate_status"),
        "evidence_item_count": intake_summary.get("evidence_item_count"),
        "required_evidence_missing": intake_summary.get("required_evidence_missing") or [],
        "ticker_direct_count": intake_summary.get("ticker_direct_count"),
        "thesis_stance": thesis_summary.get("thesis_stance"),
        "thesis_expected_direction": thesis_summary.get("expected_direction"),
        "thesis_confidence": thesis_summary.get("confidence"),
        "manual_acceptance_required": template_summary.get("manual_acceptance_required"),
        "can_enter_manual_template_review": template_summary.get("can_enter_manual_template_review"),
        "can_mark_template_accepted_now": False,
        "can_standardize_after_manual_acceptance": template_summary.get("can_standardize_domain_template_after_manual_acceptance"),
        "can_scale_to_other_domains_now": False,
        "can_run_sector_to_ticker_bridge_now": False,
        "can_create_analyst_research_recommendation": True,
        "can_create_detailed_data_news_analysis": True,
        "can_create_event_interpretation": True,
        "can_create_regime_context_scenario_analysis": True,
        "can_create_analyst_self_improvement_proposal": True,
        "can_write_learning_memory": False,
        "can_change_analyst_weights": False,
        "can_write_config": False,
        "can_create_execution_recommendation": False,
        "can_create_buy_sell_hold_recommendation": False,
        "can_create_recommendation": False,
        "can_trade": False,
    }


def _run_status(
    source_gate: dict[str, Any],
    event_interpretation: dict[str, Any],
    regime_scenario: dict[str, Any],
    intake: dict[str, Any],
    instance: dict[str, Any],
    thesis: dict[str, Any],
    forecast: dict[str, Any],
    template: dict[str, Any],
) -> str:
    if source_gate.get("summary", {}).get("can_enter_domain_research") is not True:
        return "blocked_before_domain_research"
    statuses = [
        str(event_interpretation.get("summary", {}).get("packet_status")),
        str(regime_scenario.get("summary", {}).get("packet_status")),
        str(intake.get("summary", {}).get("intake_status")),
        str(instance.get("summary", {}).get("instance_status")),
        str(thesis.get("summary", {}).get("packet_status")),
        str(forecast.get("summary", {}).get("packet_status")),
        str(template.get("summary", {}).get("candidate_status")),
    ]
    if any(status.startswith("blocked") for status in statuses):
        return "blocked_domain_analyst_vertical_slice"
    template_status = template.get("summary", {}).get("candidate_status")
    if template_status in {"ready_for_manual_template_acceptance", "ready_for_manual_template_acceptance_with_cautions"}:
        return "domain_analyst_candidate_complete_pending_manual_acceptance"
    return "domain_analyst_vertical_slice_needs_more_review"


def _branch_readiness(
    intake: dict[str, Any],
    event_interpretation: dict[str, Any],
    regime_scenario: dict[str, Any],
    instance: dict[str, Any],
    thesis: dict[str, Any],
    forecast: dict[str, Any],
    template: dict[str, Any],
) -> dict[str, Any]:
    steps = [
        _step("event_interpretation", event_interpretation.get("summary", {}).get("packet_status"), "News/data interpreted as review-only event hypotheses, mechanisms, counterforces, and evidence gaps."),
        _step("regime_scenario", regime_scenario.get("summary", {}).get("packet_status"), "Events mapped into a regime context vector, news-vs-regime assessments, scenario graph, evidence gaps, and self-check horizons."),
        _step("domain_intake", intake.get("summary", {}).get("intake_status"), "Evidence normalized into domain analyst inputs."),
        _step("instance_contract", instance.get("summary", {}).get("instance_status"), "Reusable analyst instance contract created."),
        _step("thesis_review", thesis.get("summary", {}).get("packet_status"), "Domain thesis reviewed before any ticker bridge."),
        _step("forecast_review", forecast.get("summary", {}).get("packet_status"), "Review-only thesis expectations made evaluable before future learning."),
        _step("template_standardization", template.get("summary", {}).get("candidate_status"), "Template candidate packaged for human acceptance only."),
    ]
    return {
        "readiness_status": "candidate_complete_pending_manual_acceptance"
        if template.get("summary", {}).get("candidate_status") in {"ready_for_manual_template_acceptance", "ready_for_manual_template_acceptance_with_cautions"}
        else "needs_more_review",
        "steps": steps,
    }


def _step(step_id: str, status: Any, note: str) -> dict[str, str]:
    return {"step_id": step_id, "status": str(status), "note": note}


def _synthetic_fixture_audit(*artifacts: dict[str, Any]) -> dict[str, Any]:
    text = " ".join(_flatten_strings(artifacts)).lower()
    has_synthetic = "synthetic" in text
    has_fixture = "fixture://" in text or "fixture_not_evidence" in text
    has_smoke = "smoke" in text
    return {
        "has_synthetic_marker": has_synthetic,
        "has_fixture_marker": has_fixture,
        "has_smoke_label": has_smoke,
        "evidence_usable_for_analyst_review": not has_synthetic and not has_fixture,
        "interpretation": (
            "Smoke labels are treated as caution labels, not as proof of synthetic source content. "
            "Synthetic or fixture markers block treating this as real analyst evidence."
        ),
    }


def _flatten_strings(values: Any) -> list[str]:
    if isinstance(values, dict):
        return [item for value in values.values() for item in _flatten_strings(value)]
    if isinstance(values, (list, tuple, set)):
        return [item for value in values for item in _flatten_strings(value)]
    if isinstance(values, str):
        return [values]
    return []


def _manual_gate(template: dict[str, Any]) -> dict[str, Any]:
    summary = template.get("summary", {})
    return {
        "status": "manual_acceptance_required",
        "can_mark_template_accepted_now": False,
        "template_candidate_status": summary.get("candidate_status"),
        "checklist": template.get("manual_acceptance_checklist", []),
    }


def _recommended_next_steps(
    template: dict[str, Any],
    intake: dict[str, Any],
    forecast: dict[str, Any],
    event_interpretation: dict[str, Any],
    regime_scenario: dict[str, Any],
) -> list[str]:
    summary = template.get("summary", {})
    steps = ["Manually review the vertical-slice markdown plus the instance/thesis/template markdown artifacts."]
    if summary.get("candidate_status") in {"ready_for_manual_template_acceptance", "ready_for_manual_template_acceptance_with_cautions"}:
        steps.append("Record a separate human accept/reject decision for this analyst template before cloning domains.")
    if forecast.get("summary", {}).get("forecast_candidate_count"):
        steps.append("Use the forecast review packet as the pending expectation ledger; evaluate direction and causal reasoning after the horizon matures.")
    if int(event_interpretation.get("summary", {}).get("event_packet_count") or 0) > 0:
        steps.append("Review high-materiality event interpretations before using news/data mechanisms in thesis updates or feedback labels.")
    if regime_scenario.get("summary", {}).get("can_create_regime_context_scenario_analysis"):
        steps.append("Use the regime/scenario packet to review context, evidence gaps, and self-check horizons before judging future analyst outcomes.")
    if int(intake.get("summary", {}).get("ticker_direct_count") or 0) == 0:
        steps.append("Keep this as a sector/domain analyst; ticker thesis remains blocked until a separate bridge has direct ticker evidence.")
    steps.append("After manual acceptance, reuse only the portable slots for other domains: domain_id, sectors, keywords, evidence types, and ticker universe hints.")
    return steps


def _explicit_non_actions() -> list[str]:
    return [
        "No live collector is started.",
        "No claim/event/entity extraction is executed.",
        "No template acceptance decision is recorded.",
        "No domain profile is cloned or enabled.",
        "No sector-to-ticker bridge is executed.",
        "No learning memory, analyst weight, model training, tuning, or production config write is performed.",
        "No execution, buy/sell/hold, allocation, price target, paper order, broker route, or live trade recommendation is generated.",
    ]


def _load_json(path: str | Path) -> dict[str, Any]:
    import json

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def _paths(paths: list[str | Path] | None, default: list[str]) -> list[Path]:
    return [Path(path) for path in (paths if paths is not None else default)]


def _strings(values: list[str] | tuple[str, ...]) -> list[str]:
    return sorted({str(value).strip() for value in values if str(value).strip()})


def _run_id(prefix: str) -> str:
    return f"{prefix}_{utc_now_iso().replace(':', '').replace('+', 'Z')}"
