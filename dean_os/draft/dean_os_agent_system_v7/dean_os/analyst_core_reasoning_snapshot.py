from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from dean_os.analyst_core.artifact_evidence_loader import ArtifactEvidenceLoader
from dean_os.analyst_core.lens_contract import AnalysisPacket
from dean_os.analyst_core.lens_orchestrator import LensOrchestrator
from dean_os.analyst_core.sector_analyst import (
    _build_default_registry,
    _evidence_to_entity_links,
    _evidence_to_event_records,
)
from dean_os.draft.dean_os_agent_system_v7.dean_os.artifact_writer import ReviewArtifactWriter
from dean_os.schemas import utc_now_iso
from dean_os.utils import json_ready

DEFAULT_RUNTIME_JSON = (
    "reports/dean_os/semiconductor_analyst_runtime_current/latest.json"
)
DEFAULT_OUTPUT_DIR = (
    "reports/dean_os/analyst_core_reasoning_snapshot_current"
)
SNAPSHOT_CONTRACT = "dean_analyst_core_reasoning_snapshot_v1"


class AnalystCoreReasoningSnapshot:
    """Run the verified deterministic reasoning core over one saved runtime.

    The snapshot is an analysis aid, not a forecast. It binds every output to
    the exact runtime JSON and preserves the no-training/no-trading boundary.
    """

    def __init__(self, output_dir: str | Path = DEFAULT_OUTPUT_DIR):
        self.output_dir = Path(output_dir)

    def build(
        self,
        *,
        runtime_json: str | Path = DEFAULT_RUNTIME_JSON,
        save: bool = True,
    ) -> dict[str, Any]:
        runtime_path = _resolve_runtime_path(runtime_json)
        runtime = _load_json(runtime_path)
        domain_id = str(runtime.get("domain_id") or "").strip()
        if not domain_id:
            raise ValueError("Runtime artifact is missing domain_id")

        evidence = ArtifactEvidenceLoader().from_runtime_artifact(
            runtime_path,
            domain_id=domain_id,
        )
        as_of = str(
            runtime.get("inputs", {}).get("as_of")
            or runtime.get("adapter", {}).get("as_of")
            or runtime.get("analyst_report", {}).get("as_of")
            or ""
        )
        if not as_of:
            raise ValueError("Runtime artifact is missing as_of")

        packet = AnalysisPacket(
            packet_id=f"reasoning_{runtime.get('run_id') or _run_id()}",
            as_of_date=as_of,
            source_packet_ids=[item.evidence_id for item in evidence],
            event_records=_evidence_to_event_records(evidence),
            entity_links=_evidence_to_entity_links(evidence),
        )
        registry = _build_default_registry()
        enriched, deltas = LensOrchestrator(
            registry,
            config={
                "domain_id": domain_id,
                "default_horizon_days": 180,
                "checkpoint_horizons": [30, 90, 180],
            },
        ).run(packet)

        class_counts = Counter(
            str(item.get("event_class") or "other")
            for item in enriched.classified_events
        )
        explicit_ticker_events = [
            item
            for item in enriched.classified_events
            if item.get("affected_tickers")
        ]
        directional_ticker_events = [
            item
            for item in explicit_ticker_events
            if item.get("event_class") != "other"
        ]
        touched_dimensions = [
            name
            for name, state in (
                enriched.regime_context.dimensions.items()
                if enriched.regime_context is not None
                else []
            )
            if state.evidence_ids
        ]
        checks = _review_checks(
            evidence_count=len(evidence),
            classified_count=len(enriched.classified_events),
            delta_names=[delta.module_name for delta in deltas],
            directional_ticker_event_count=len(directional_ticker_events),
            scenario_graph=enriched.scenario_graph,
            expectation_gap=enriched.expectation_gap,
        )
        status = _status(checks)
        payload = {
            "run_id": _run_id(),
            "created_at": utc_now_iso(),
            "contract": SNAPSHOT_CONTRACT,
            "mode": "analyst_core_reasoning_snapshot",
            "status": status,
            "inputs": {
                "runtime_json": str(runtime_path),
                "runtime_run_id": runtime.get("run_id"),
                "runtime_sha256": _sha256(runtime_path),
                "runtime_contract": runtime.get("runtime_contract"),
                "domain_id": domain_id,
                "as_of": as_of,
                "source_artifacts": runtime.get("source_artifacts", {}),
            },
            "summary": {
                "evidence_count": len(evidence),
                "classified_event_count": len(enriched.classified_events),
                "unclassified_event_count": class_counts.get("other", 0),
                "event_class_count": len(class_counts),
                "transmission_channel_count": len(enriched.transmission_channels),
                "hypothesis_count": len(enriched.hypotheses),
                "evidence_gap_count": len(enriched.evidence_gaps),
                "regime_dimension_count": (
                    len(enriched.regime_context.dimensions)
                    if enriched.regime_context is not None
                    else 0
                ),
                "evidence_touched_regime_dimension_count": len(touched_dimensions),
                "explicit_ticker_attribution_event_count": len(explicit_ticker_events),
                "directional_ticker_reasoning_event_count": len(
                    directional_ticker_events
                ),
                "scenario_graph_status": (
                    "not_generated" if enriched.scenario_graph is None else "present"
                ),
                "expectation_gap_status": (
                    "disabled_unverified"
                    if enriched.expectation_gap is None
                    else "present"
                ),
                "manual_review_required": True,
                "can_create_ticker_forecast": False,
                "can_train_or_tune": False,
                "can_trade": False,
            },
            "module_policy": {
                "verified_modules": [
                    "event_classifier",
                    "regime_context",
                    "transmission_mapper",
                    "hypothesis_ledger",
                    "evidence_gap",
                ],
                "experimental_modules_excluded": {
                    "expectation_gap": (
                        "No empirical expectation or positioning input; static "
                        "priors are not calibrated probabilities."
                    ),
                    "historical_analog": (
                        "Static analog templates are not a verified historical "
                        "outcome database."
                    ),
                    "scenario_graph": (
                        "No scenario-generation lens is implemented in the "
                        "verified runtime."
                    ),
                },
            },
            "event_classification": {
                "class_counts": dict(sorted(class_counts.items())),
                "events": enriched.classified_events,
            },
            "regime_context": (
                enriched.regime_context.model_dump(mode="json")
                if enriched.regime_context is not None
                else None
            ),
            "transmission_channels": enriched.transmission_channels,
            "hypothesis_ledger": [
                item.model_dump(mode="json") for item in enriched.hypotheses
            ],
            "evidence_gaps": [
                item.model_dump(mode="json") for item in enriched.evidence_gaps
            ],
            "reasoning_delta_trail": [
                item.model_dump(mode="json") for item in deltas
            ],
            "ticker_boundary": {
                "explicit_attribution_is_not_directional_thesis": True,
                "plain_text_ticker_promotion_allowed": False,
                "directional_ticker_events": directional_ticker_events,
                "can_create_direct_ticker_thesis": False,
                "can_create_pipeline_prediction": False,
            },
            "scenario_boundary": {
                "scenario_graph": None,
                "probabilities_generated": False,
                "reason": (
                    "The verified core has no calibrated scenario generator. "
                    "Missing output is preserved rather than synthesized."
                ),
            },
            "review_checks": checks,
            "safety": {
                "review_only": True,
                "network_access_performed": False,
                "collector_run_performed": False,
                "pipeline_run_performed": False,
                "training_run_performed": False,
                "tuning_run_performed": False,
                "learning_write_performed": False,
                "production_config_write_performed": False,
                "broker_access_performed": False,
                "live_execution_performed": False,
            },
        }
        if save:
            payload["saved_paths"] = ReviewArtifactWriter(self.output_dir).write(
                payload=payload,
                markdown=render_reasoning_snapshot_markdown(payload),
                run_id=payload["run_id"],
            )
        return json_ready(payload)


def render_reasoning_snapshot_markdown(payload: dict[str, Any]) -> str:
    summary = payload.get("summary", {})
    classes = payload.get("event_classification", {}).get("class_counts", {})
    checks = payload.get("review_checks", [])
    check_counts = Counter(item.get("status") for item in checks)
    lines = [
        "# DEAN-OS Analyst Core Reasoning Snapshot",
        "",
        f"- Status: `{payload.get('status')}`",
        f"- Domain: `{payload.get('inputs', {}).get('domain_id')}`",
        f"- As of: `{payload.get('inputs', {}).get('as_of')}`",
        f"- Evidence/classified: {summary.get('evidence_count')}/{summary.get('classified_event_count')}",
        f"- Regime dimensions touched: {summary.get('evidence_touched_regime_dimension_count')}/{summary.get('regime_dimension_count')}",
        f"- Transmission channels: {summary.get('transmission_channel_count')}",
        f"- Candidate hypotheses: {summary.get('hypothesis_count')}",
        f"- Evidence gaps: {summary.get('evidence_gap_count')}",
        f"- Directional ticker events: {summary.get('directional_ticker_reasoning_event_count')}",
        f"- Checks: pass={check_counts.get('pass', 0)} warn={check_counts.get('warn', 0)} fail={check_counts.get('fail', 0)}",
        "",
        "## Event classes",
        "",
    ]
    lines.extend(f"- `{name}`: {count}" for name, count in classes.items())
    lines.extend(
        [
            "",
            "## Honest boundaries",
            "",
            "- Scenario graph: not generated.",
            "- Expectation-gap probabilities: disabled as unverified.",
            "- Historical analog templates: excluded from the verified path.",
            "- Explicit ticker attribution does not create a ticker thesis.",
            "- No training, tuning, config write, order, or trade authority.",
            "",
        ]
    )
    return "\n".join(lines)


def _review_checks(
    *,
    evidence_count: int,
    classified_count: int,
    delta_names: list[str],
    directional_ticker_event_count: int,
    scenario_graph: Any,
    expectation_gap: Any,
) -> list[dict[str, str]]:
    required_modules = {
        "event_classifier",
        "regime_context",
        "transmission_mapper",
        "hypothesis_ledger",
        "evidence_gap",
    }
    present_modules = set(delta_names)
    return [
        _check(
            "pass" if evidence_count > 0 else "fail",
            "runtime_evidence_present",
            f"Loaded {evidence_count} validated evidence items.",
        ),
        _check(
            "pass" if classified_count == evidence_count else "fail",
            "one_classification_per_evidence_item",
            f"Classified {classified_count}/{evidence_count} evidence items.",
        ),
        _check(
            "pass" if required_modules == present_modules else "fail",
            "verified_module_set_exact",
            f"Modules: {sorted(present_modules)}.",
        ),
        _check(
            "pass" if directional_ticker_event_count == 0 else "fail",
            "no_directional_ticker_leakage",
            f"Directional ticker reasoning events: {directional_ticker_event_count}.",
        ),
        _check(
            "warn" if scenario_graph is None else "fail",
            "scenario_graph_not_yet_implemented",
            "No verified scenario graph is generated.",
        ),
        _check(
            "pass" if expectation_gap is None else "fail",
            "unverified_expectation_gap_disabled",
            "Expectation-gap heuristic is absent from the verified path.",
        ),
    ]


def _check(status: str, check_id: str, detail: str) -> dict[str, str]:
    return {"status": status, "check_id": check_id, "detail": detail}


def _status(checks: list[dict[str, str]]) -> str:
    if any(item["status"] == "fail" for item in checks):
        return "reasoning_snapshot_blocked"
    if any(item["status"] == "warn" for item in checks):
        return "reasoning_snapshot_ready_with_cautions"
    return "reasoning_snapshot_ready_for_review"


def _resolve_runtime_path(value: str | Path) -> Path:
    path = Path(value)
    if path.is_dir():
        path = path / "latest.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _run_id() -> str:
    return "analyst_core_reasoning_snapshot_" + utc_now_iso().replace(
        ":", ""
    ).replace("+", "Z")


__all__ = [
    "AnalystCoreReasoningSnapshot",
    "DEFAULT_OUTPUT_DIR",
    "DEFAULT_RUNTIME_JSON",
    "SNAPSHOT_CONTRACT",
    "render_reasoning_snapshot_markdown",
]
