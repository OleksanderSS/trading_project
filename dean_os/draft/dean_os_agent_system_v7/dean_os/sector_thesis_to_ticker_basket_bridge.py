from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from dean_os.analysts._producers.ticker import (
    load_verified_ticker_specific_evidence_fragment,
)
from dean_os.schemas import utc_now_iso
from dean_os.utils import json_ready

DEFAULT_FOCUSED_REPLAY_BATCH = "reports/dean_os/historical_research_replay_batch_focused_overlay_integration_current/latest.json"


class SectorThesisToTickerBasketBridge:
    """Maps a sector/domain thesis into reviewed ticker candidates.

    This is the contract between a future domain specialist and ticker-level
    evaluation: sector theses are allowed, but they must not become ticker
    theses until direct evidence and focused overlays support the specific
    company.
    """

    def __init__(self, output_dir: str | Path = "reports/dean_os/sector_thesis_to_ticker_basket"):
        self.output_dir = Path(output_dir)

    def build(
        self,
        research_batch_path: str | Path = DEFAULT_FOCUSED_REPLAY_BATCH,
        domain_profile: str = "semiconductor_ai_infrastructure",
        sector: str = "semiconductor",
        save: bool = True,
    ) -> dict[str, Any]:
        research_batch = _load_json(research_batch_path)
        runs = research_batch.get("runs", [])
        sector_thesis = _sector_thesis(
            research_batch=research_batch,
            runs=runs,
            domain_profile=domain_profile,
            sector=sector,
        )
        ticker_candidates = _ticker_candidates(runs)
        summary = _summary(sector_thesis, ticker_candidates, runs)
        payload = {
            "run_id": _run_id("sector_thesis_to_ticker_basket"),
            "created_at": utc_now_iso(),
            "mode": "sector_thesis_to_ticker_basket_bridge",
            "inputs": {
                "research_batch_path": str(research_batch_path),
                "domain_profile": domain_profile,
                "sector": sector,
            },
            "summary": summary,
            "domain_analyst_contract": _domain_analyst_contract(domain_profile, sector),
            "sector_thesis": sector_thesis,
            "ticker_candidates": ticker_candidates,
            "candidate_status_counts": _counts(candidate.get("candidate_status") for candidate in ticker_candidates),
            "mapping_runs": [_mapping_run(run) for run in runs],
            "tasks": _tasks(summary, ticker_candidates),
            "commands": _commands(research_batch_path, domain_profile, sector),
            "safety": {
                "read_only": True,
                "data_mutation_performed": False,
                "collector_run_performed": False,
                "network_access_performed": False,
                "pipeline_run_performed": False,
                "learning_write_performed": False,
                "operation_proposal_created": False,
                "config_write_performed": False,
                "broker_access_performed": False,
            },
            "recommendations": _recommendations(summary),
        }
        if save:
            self.save(payload)
        return payload

    def build_from_current_review(
        self,
        *,
        domain_thesis_review_path: str | Path,
        pipeline_case_paths: list[str | Path] | None = None,
        ticker_evidence_path: str | Path | None = None,
        prediction_review_path: str | Path | None = None,
        feature_timeframe_audit_path: str | Path | None = None,
        save: bool = True,
    ) -> dict[str, Any]:
        """Bind the current sector review to exact ticker pipeline cases.

        Sector evidence remains supporting context. A pipeline case is an
        exact ticker/model/target/timeframe evaluation artifact, not direct
        ticker evidence and not a forecast outcome.
        """
        review_path = Path(domain_thesis_review_path)
        thesis_review = _load_json(review_path)
        _validate_current_thesis_review(thesis_review)
        pipeline_case_bindings = [
            _load_pipeline_case_binding(Path(path))
            for path in pipeline_case_paths or []
        ]
        ticker_evidence_binding = (
            _load_ticker_evidence_binding(
                Path(ticker_evidence_path),
                expected_as_of=(
                    thesis_review.get("thesis_snapshot") or {}
                ).get("as_of"),
            )
            if ticker_evidence_path
            else None
        )
        prediction_review_binding = (
            _load_prediction_review_binding(Path(prediction_review_path))
            if prediction_review_path
            else None
        )
        feature_timeframe_audit_binding = (
            _load_feature_timeframe_audit_binding(
                Path(feature_timeframe_audit_path),
                expected_stage5_sha256=(
                    prediction_review_binding.get(
                        "source_artifact_sha256"
                    )
                    if prediction_review_binding
                    else None
                ),
            )
            if feature_timeframe_audit_path
            else None
        )
        ticker_candidates = _current_ticker_candidates(
            thesis_review,
            pipeline_case_bindings,
            ticker_evidence_binding=ticker_evidence_binding,
            prediction_review_binding=prediction_review_binding,
            feature_timeframe_audit_binding=(
                feature_timeframe_audit_binding
            ),
        )
        sector_thesis = _current_sector_thesis(thesis_review)
        summary = _current_bridge_summary(
            ticker_candidates,
            pipeline_case_bindings,
            prediction_review_binding=prediction_review_binding,
            feature_timeframe_audit_binding=(
                feature_timeframe_audit_binding
            ),
        )
        payload = {
            "run_id": _run_id(
                "sector_thesis_to_ticker_pipeline_bridge"
            ),
            "created_at": utc_now_iso(),
            "mode": "sector_thesis_to_ticker_basket_bridge",
            "bridge_contract": (
                "dean_sector_context_to_exact_ticker_pipeline_v2"
            ),
            "inputs": {
                "domain_thesis_review_path": str(review_path),
                "domain_thesis_review_run_id": thesis_review.get(
                    "run_id"
                ),
                "domain_thesis_review_sha256": _file_sha256(
                    review_path
                ),
                "pipeline_case_paths": [
                    item["path"] for item in pipeline_case_bindings
                ],
                "ticker_evidence_path": (
                    ticker_evidence_binding.get("path")
                    if ticker_evidence_binding
                    else None
                ),
                "ticker_evidence_sha256": (
                    ticker_evidence_binding.get("sha256")
                    if ticker_evidence_binding
                    else None
                ),
                "prediction_review_path": (
                    prediction_review_binding.get("path")
                    if prediction_review_binding
                    else None
                ),
                "prediction_review_sha256": (
                    prediction_review_binding.get("sha256")
                    if prediction_review_binding
                    else None
                ),
                "prediction_source_artifact_sha256": (
                    prediction_review_binding.get(
                        "source_artifact_sha256"
                    )
                    if prediction_review_binding
                    else None
                ),
                "feature_timeframe_audit_path": (
                    feature_timeframe_audit_binding.get("path")
                    if feature_timeframe_audit_binding
                    else None
                ),
                "feature_timeframe_audit_sha256": (
                    feature_timeframe_audit_binding.get("sha256")
                    if feature_timeframe_audit_binding
                    else None
                ),
            },
            "summary": summary,
            "domain_analyst_contract": _domain_analyst_contract(
                str(sector_thesis.get("domain_profile")),
                str(sector_thesis.get("sector")),
            ),
            "sector_thesis": sector_thesis,
            "ticker_candidates": ticker_candidates,
            "candidate_status_counts": _counts(
                candidate.get("candidate_status")
                for candidate in ticker_candidates
            ),
            "mapping_runs": [
                item["mapping_run"]
                for item in pipeline_case_bindings
            ],
            "pipeline_case_bindings": pipeline_case_bindings,
            "ticker_evidence_binding": ticker_evidence_binding,
            "prediction_review_binding": (
                _prediction_review_binding_summary(
                    prediction_review_binding
                )
            ),
            "feature_timeframe_audit_binding": (
                feature_timeframe_audit_binding
            ),
            "exact_ticker_pipeline_contract": (
                _exact_ticker_pipeline_contract()
            ),
            "tasks": _current_bridge_tasks(
                ticker_candidates,
                pipeline_case_bindings,
                prediction_review_binding=prediction_review_binding,
                feature_timeframe_audit_binding=(
                    feature_timeframe_audit_binding
                ),
            ),
            "commands": {
                "rerun_current_bridge": (
                    "python run_agent_sector_to_ticker_bridge.py "
                    f"--domain-thesis-review-json {review_path} "
                    + (
                        "--ticker-evidence-json "
                        f"{ticker_evidence_binding['path']} "
                        if ticker_evidence_binding
                        else ""
                    )
                    + (
                        "--prediction-review-json "
                        f"{prediction_review_binding['path']} "
                        if prediction_review_binding
                        else ""
                    )
                    + (
                        "--feature-timeframe-audit-json "
                        f"{feature_timeframe_audit_binding['path']} "
                        if feature_timeframe_audit_binding
                        else ""
                    )
                    +
                    "--pipeline-case-json "
                    + " ".join(
                        item["path"]
                        for item in pipeline_case_bindings
                    )
                    + " --output-dir reports\\dean_os\\"
                    "sector_thesis_to_ticker_basket_current"
                ).strip(),
            },
            "safety": {
                "read_only": True,
                "data_mutation_performed": False,
                "collector_run_performed": False,
                "network_access_performed": False,
                "pipeline_run_performed": False,
                "learning_write_performed": False,
                "operation_proposal_created": False,
                "config_write_performed": False,
                "broker_access_performed": False,
                "ticker_forecast_created": False,
                "sector_context_promoted_to_ticker_evidence": False,
                "quarantined_prediction_promoted": False,
                "timeframe_mismatch_overridden": False,
            },
            "recommendations": [
                (
                    "Attach sector context only to an exact ticker pipeline "
                    "review identity; never use it to fill missing direct "
                    "ticker evidence."
                ),
                (
                    "Collect ticker-specific catalysts and trustworthy "
                    "Stage 5/pipeline evaluation artifacts independently "
                    "for NVDA, AMD, INTC, and TSM."
                ),
                (
                    "Retain the AMD model case as a negative evaluation "
                    "case and wait for new forward development data."
                ),
            ],
        }
        if save:
            self.save(payload)
        return payload

    def save(self, payload: dict[str, Any]) -> tuple[Path, Path]:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        json_path = self.output_dir / f"{payload['run_id']}.json"
        md_path = self.output_dir / f"{payload['run_id']}.md"
        latest_json = self.output_dir / "latest.json"
        latest_md = self.output_dir / "latest.md"
        payload["saved_paths"] = {
            "json": str(json_path),
            "markdown": str(md_path),
            "latest_json": str(latest_json),
            "latest_markdown": str(latest_md),
        }
        rendered_json = json.dumps(json_ready(payload), indent=2, ensure_ascii=False) + "\n"
        rendered_md = render_sector_thesis_to_ticker_basket_markdown(payload)
        json_path.write_text(rendered_json, encoding="utf-8")
        latest_json.write_text(rendered_json, encoding="utf-8")
        md_path.write_text(rendered_md, encoding="utf-8")
        latest_md.write_text(rendered_md, encoding="utf-8")
        return json_path, md_path


def render_sector_thesis_to_ticker_basket_markdown(payload: dict[str, Any]) -> str:
    summary = payload.get("summary", {})
    sector = payload.get("sector_thesis", {})
    lines = [
        "# DEAN-OS Sector Thesis To Ticker Basket Bridge",
        "",
        f"- Run ID: `{payload.get('run_id')}`",
        f"- Status: `{summary.get('bridge_status')}`",
        f"- Domain profile: `{sector.get('domain_profile')}`",
        f"- Sector stance: `{sector.get('sector_stance')}`",
        f"- Ticker candidates: {summary.get('ticker_candidate_count')}",
        f"- Direct ticker thesis ready: {summary.get('direct_ticker_thesis_ready_count')}",
        f"- Evidence-limited direct candidates: {summary.get('evidence_limited_direct_candidate_count')}",
        f"- Blocked candidates: {summary.get('blocked_candidate_count')}",
        f"- Stage 5 contexts: {summary.get('prediction_context_count', 0)}",
        f"- Stage 5 complete: {summary.get('complete_prediction_context_count', 0)}",
        f"- Stage 5 quarantined: {summary.get('quarantined_prediction_context_count', 0)}",
        f"- Feature timeframe mismatches: {summary.get('timeframe_mismatch_ticker_count', 0)}",
        "",
        "## Sector Thesis",
        "",
        f"- Thesis: {sector.get('thesis')}",
        f"- Thesis level: `{sector.get('thesis_level')}`",
        "",
        "## Ticker Candidates",
        "",
    ]
    for candidate in payload.get("ticker_candidates", []):
        prediction_review = (
            candidate.get("stage5_prediction_review") or {}
        )
        timeframe_audit = (
            candidate.get("feature_timeframe_audit") or {}
        )
        lines.append(
            f"- `{candidate.get('ticker')}` status=`{candidate.get('candidate_status')}` "
            f"overlay_ready={candidate.get('overlay_ready_runs')} blocked={candidate.get('blocked_runs')} "
            f"stance={candidate.get('dominant_focused_stance')} "
            f"stage5_status=`{prediction_review.get('status')}` "
            f"stage5_complete={prediction_review.get('complete_context_count', 0)}/"
            f"{prediction_review.get('context_count', 0)} "
            f"timeframe_audit=`{timeframe_audit.get('status')}` "
            f"declared=`{timeframe_audit.get('declared_timeframe')}` "
            f"observed=`{timeframe_audit.get('observed_timeframe')}`"
        )
    lines.extend(["", "## Tasks", ""])
    tasks = payload.get("tasks", [])
    lines.extend(f"- `{task.get('priority')}` {task.get('task_id')}: {task.get('description')}" for task in tasks) if tasks else lines.append("- None.")
    lines.extend(["", "## Recommendations", ""])
    lines.extend(f"- {item}" for item in payload.get("recommendations", []))
    return "\n".join(lines).strip() + "\n"


def _validate_current_thesis_review(
    thesis_review: dict[str, Any],
) -> None:
    if thesis_review.get("mode") != (
        "domain_analyst_thesis_review_packet"
    ):
        raise ValueError(
            "Current bridge requires a domain thesis review packet"
        )
    summary = thesis_review.get("summary") or {}
    if summary.get("packet_status") not in {
        "domain_thesis_review_ready",
        "domain_thesis_review_ready_with_cautions",
    }:
        raise ValueError(
            "Domain thesis review is not ready for bridge review"
        )
    analytical = thesis_review.get("analytical_review") or {}
    reasoning = _sector_reasoning_context(thesis_review)
    if analytical.get("scope_decision") != "sector_thesis_only":
        raise ValueError(
            "Current bridge requires an explicit sector-only scope"
        )
    if analytical.get("ticker_decision") != (
        "no_direct_ticker_thesis"
    ):
        raise ValueError(
            "Current bridge expects ticker theses to remain blocked"
        )
    verification = (
        thesis_review.get("linked_artifact_verification") or {}
    )
    if verification.get("all_hashes_match") is not True:
        raise ValueError(
            "Domain thesis review linked artifacts are not hash-verified"
        )
    if reasoning.get("available"):
        if reasoning.get("runtime_hash_bound") is not True:
            raise ValueError(
                "Verified reasoning snapshot is not bound to the thesis runtime"
            )
        if int(reasoning.get("directional_ticker_event_count") or 0) != 0:
            raise ValueError(
                "Sector reasoning contains directional ticker leakage"
            )


def _load_pipeline_case_binding(path: Path) -> dict[str, Any]:
    payload = _load_json(path)
    if payload.get("mode") != "pipeline_model_case_packet":
        raise ValueError(
            f"Expected pipeline_model_case_packet: {path}"
        )
    summary = payload.get("summary") or {}
    case = payload.get("case") or {}
    lineage = case.get("lineage") or {}
    required_identity = (
        "ticker",
        "model",
        "target_name",
        "timeframe",
        "context_fingerprint",
    )
    missing = [
        field for field in required_identity if not lineage.get(field)
    ]
    if missing:
        raise ValueError(
            "Pipeline case missing exact identity: "
            + ", ".join(missing)
        )
    if (
        summary.get("case_scope")
        != "ticker_model_evaluation_only"
        or summary.get("eligible_as_domain_evidence") is not False
        or summary.get("can_trade") is not False
    ):
        raise ValueError(
            "Pipeline case violates ticker-only review boundary"
        )
    classification = summary.get("case_classification")
    ticker = str(lineage.get("ticker")).upper()
    mapping_run = {
        "as_of": case.get("evaluated_at"),
        "horizon_days": None,
        "price_ticker": ticker,
        "sector_signal_level": "supporting_sector_context_only",
        "ticker_signal_level": (
            "blocked_pipeline_model_case"
        ),
        "research_stance": "not_applicable",
        "research_expected_direction": None,
        "ticker_specificity": "exact_pipeline_identity",
        "exam_verdict": summary.get("result_label"),
        "focused_overlay_status": (
            "blocked_pipeline_case_not_direct_ticker_evidence"
        ),
        "focused_overlay_applied": False,
        "outcome_label": None,
        "realized_return": None,
        "model": lineage.get("model"),
        "target_name": lineage.get("target_name"),
        "timeframe": lineage.get("timeframe"),
        "context_fingerprint": lineage.get(
            "context_fingerprint"
        ),
        "case_classification": classification,
        "blocked_metric_planes": summary.get(
            "blocked_metric_planes", []
        ),
    }
    return {
        "path": str(path),
        "sha256": _file_sha256(path),
        "run_id": payload.get("run_id"),
        "case_id": summary.get("case_id"),
        "ticker": ticker,
        "model": lineage.get("model"),
        "target_name": lineage.get("target_name"),
        "timeframe": lineage.get("timeframe"),
        "context_fingerprint": lineage.get(
            "context_fingerprint"
        ),
        "case_status": summary.get("case_status"),
        "case_classification": classification,
        "result_label": summary.get("result_label"),
        "review_disposition": summary.get(
            "review_disposition"
        ),
        "blocked_metric_planes": summary.get(
            "blocked_metric_planes", []
        ),
        "eligible_as_domain_evidence": False,
        "can_use_as_forecast_outcome": summary.get(
            "can_use_as_forecast_outcome"
        ),
        "can_create_ticker_forecast": False,
        "mapping_run": mapping_run,
    }


def _load_ticker_evidence_binding(
    path: Path,
    *,
    expected_as_of: str | None,
) -> dict[str, Any]:
    fragment = load_verified_ticker_specific_evidence_fragment(
        path,
        expected_as_of=expected_as_of,
    )
    return {
        "path": str(path),
        "sha256": _file_sha256(path),
        "as_of": fragment.get("as_of"),
        "domain_id": fragment.get("domain_id"),
        "records": fragment.get("records", []),
        "ticker_summary": fragment.get("ticker_summary", []),
        "lane_review": fragment.get("lane_review", []),
        "source_metadata": fragment.get("metadata", {}),
        "can_create_ticker_forecast": False,
    }


def _load_prediction_review_binding(path: Path) -> dict[str, Any]:
    payload = _load_json(path)
    if payload.get("mode") != "pipeline_prediction_review_packet":
        raise ValueError(
            f"Expected pipeline_prediction_review_packet: {path}"
        )
    if payload.get("schema_version") != (
        "dean_stage5_prediction_review_v1"
    ):
        raise ValueError(
            "Unsupported Stage 5 prediction review schema"
        )
    if payload.get("source_contract") != (
        "pipeline_stage5_prediction_results"
    ):
        raise ValueError(
            "Prediction review is not bound to Stage 5 results"
        )
    if payload.get("status") not in {
        "stage5_prediction_review_ready",
        "stage5_prediction_review_partial",
    }:
        raise ValueError(
            "Prediction review has no reviewable Stage 5 contexts"
        )
    source_artifact = payload.get("source_artifact") or {}
    if (
        source_artifact.get("available") is not True
        or source_artifact.get("immutable_binding_ready") is not True
        or len(str(source_artifact.get("sha256") or "")) != 64
    ):
        raise ValueError(
            "Prediction review source artifact is not immutable-bound"
        )
    safety = payload.get("safety") or {}
    if (
        safety.get("supporting_review_only") is not True
        or safety.get("decision_influence") is not False
        or safety.get("can_promote_model") is not False
        or safety.get("can_trade") is not False
    ):
        raise ValueError(
            "Prediction review violates supporting-only safety boundary"
        )
    sector_overlay = payload.get("sector_context_review") or {}
    if sector_overlay.get("available") is True:
        raise ValueError(
            "Bridge requires the base prediction source review without "
            "a sector overlay to avoid circular artifact lineage"
        )
    contexts = payload.get("contexts")
    if not isinstance(contexts, list):
        raise ValueError("Prediction review contexts are missing")
    if int(payload.get("context_count") or 0) != len(contexts):
        raise ValueError(
            "Prediction review context count does not match payload"
        )
    normalized_contexts = []
    for item in contexts:
        if not isinstance(item, dict):
            continue
        ticker = str(item.get("ticker") or "").upper()
        if not ticker:
            continue
        review_issues = sorted(
            {
                str(issue)
                for issue in item.get("review_issues", [])
                if issue
            }
        )
        missing_lineage = sorted(
            {
                str(field)
                for field in item.get(
                    "missing_lineage_fields", []
                )
                if field
            }
        )
        is_complete = (
            item.get("lineage_status") == "complete"
            and not review_issues
        )
        normalized_contexts.append(
            {
                "context_key": item.get("context_key"),
                "ticker": ticker,
                "model_context_id": item.get("model_context_id"),
                "selected_primary_model": item.get(
                    "selected_primary_model"
                ),
                "model_type": item.get("model_type"),
                "target_name": item.get("target_name"),
                "timeframe": item.get("timeframe"),
                "context_fingerprint": item.get(
                    "context_fingerprint"
                ),
                "lineage_status": item.get("lineage_status"),
                "missing_lineage_fields": missing_lineage,
                "review_issues": review_issues,
                "prediction_as_of": (
                    item.get("prediction") or {}
                ).get("as_of"),
                "review_complete": is_complete,
            }
        )
    complete_count = sum(
        item["review_complete"] for item in normalized_contexts
    )
    return {
        "path": str(path),
        "sha256": _file_sha256(path),
        "run_id": payload.get("run_id"),
        "status": payload.get("status"),
        "packet_fingerprint": payload.get("packet_fingerprint"),
        "source_artifact_path": source_artifact.get("path"),
        "source_artifact_sha256": source_artifact.get("sha256"),
        "source_context_count": int(
            payload.get("source_context_count") or 0
        ),
        "excluded_by_scope_count": int(
            payload.get("excluded_by_scope_count") or 0
        ),
        "context_count": len(normalized_contexts),
        "complete_context_count": complete_count,
        "quarantined_context_count": (
            len(normalized_contexts) - complete_count
        ),
        "review_issue_counts": payload.get(
            "review_issue_counts", {}
        ),
        "missing_lineage_field_counts": payload.get(
            "missing_lineage_field_counts", {}
        ),
        "contexts": normalized_contexts,
        "sector_overlay_attached": False,
        "supporting_review_only": True,
        "can_create_ticker_forecast": False,
        "can_clear_model_evaluation": False,
        "can_trade": False,
    }


def _prediction_review_binding_summary(
    binding: dict[str, Any] | None,
) -> dict[str, Any]:
    if not binding:
        return {
            "status": "not_supplied",
            "path": None,
            "context_count": 0,
            "complete_context_count": 0,
            "quarantined_context_count": 0,
            "supporting_review_only": True,
            "can_create_ticker_forecast": False,
            "can_clear_model_evaluation": False,
            "can_trade": False,
        }
    return {
        key: value
        for key, value in binding.items()
        if key != "contexts"
    }


def _load_feature_timeframe_audit_binding(
    path: Path,
    *,
    expected_stage5_sha256: str | None,
) -> dict[str, Any]:
    payload = _load_json(path)
    if payload.get("mode") != "pipeline_feature_timeframe_audit":
        raise ValueError(
            f"Expected pipeline_feature_timeframe_audit: {path}"
        )
    if payload.get("schema_version") != (
        "dean_pipeline_feature_timeframe_audit_v1"
    ):
        raise ValueError(
            "Unsupported pipeline feature timeframe audit schema"
        )
    safety = payload.get("safety") or {}
    if (
        safety.get("read_only") is not True
        or safety.get("training_performed") is not False
        or safety.get("can_promote_model") is not False
        or safety.get("can_trade") is not False
    ):
        raise ValueError(
            "Feature timeframe audit violates read-only boundary"
        )
    inputs = payload.get("inputs") or {}
    feature_path = Path(str(inputs.get("features_path") or ""))
    expected_feature_sha = str(
        inputs.get("features_sha256") or ""
    )
    if (
        not feature_path.is_file()
        or len(expected_feature_sha) != 64
        or _file_sha256(feature_path) != expected_feature_sha
    ):
        raise ValueError(
            "Feature timeframe audit source hash is not verifiable"
        )
    stage5 = payload.get("stage5_candidate_binding") or {}
    audit_stage5_sha = stage5.get("sha256")
    if (
        expected_stage5_sha256
        and audit_stage5_sha != expected_stage5_sha256
    ):
        raise ValueError(
            "Feature timeframe audit and prediction review bind "
            "different Stage 5 artifacts"
        )
    reports = payload.get("ticker_timeframe_reports")
    if not isinstance(reports, list):
        raise ValueError(
            "Feature timeframe audit ticker reports are missing"
        )
    return {
        "path": str(path),
        "sha256": _file_sha256(path),
        "run_id": payload.get("run_id"),
        "status": payload.get("status"),
        "features_path": str(feature_path),
        "features_sha256": expected_feature_sha,
        "stage5_sha256": audit_stage5_sha,
        "stage5_relationship_status": stage5.get(
            "relationship_status"
        ),
        "can_assert_feature_parentage": stage5.get(
            "can_assert_feature_parentage"
        ),
        "summary": payload.get("summary") or {},
        "ticker_timeframe_reports": reports,
        "supporting_audit_only": True,
        "can_override_timeframe": False,
        "can_create_ticker_forecast": False,
        "can_trade": False,
    }


def _sector_reasoning_context(
    thesis_review: dict[str, Any],
) -> dict[str, Any]:
    source = thesis_review.get("reasoning_snapshot_context") or {}
    if not source.get("available"):
        return {
            "available": False,
            "status": "not_supplied",
            "allowed_use": "supporting_sector_context_only",
            "can_influence_ticker_direction": False,
        }
    regime = source.get("regime_context") or {}
    dimensions = {}
    for name, item in (regime.get("dimensions") or {}).items():
        if not isinstance(item, dict):
            continue
        dimensions[name] = {
            "state": item.get("state"),
            "intensity": item.get("intensity"),
            "confidence": item.get("confidence"),
            "evidence_id_count": len(item.get("evidence_ids") or []),
        }
    hypotheses = []
    for item in source.get("hypothesis_ledger", []):
        if not isinstance(item, dict):
            continue
        hypotheses.append(
            {
                "hypothesis_id": item.get("hypothesis_id"),
                "hypothesis": item.get("hypothesis"),
                "confidence": item.get("confidence"),
                "horizons_to_check": item.get("horizons_to_check", []),
                "invalidation_signals": item.get(
                    "invalidation_signals", []
                ),
                "calibration_note": item.get("calibration_note"),
            }
        )
    evidence_gaps = []
    for item in source.get("evidence_gaps", [])[:12]:
        if not isinstance(item, dict):
            continue
        evidence_gaps.append(
            {
                "gap_id": item.get("gap_id"),
                "description": item.get("description"),
                "priority": item.get("priority"),
                "expected_source_type": item.get(
                    "expected_source_type"
                ),
            }
        )
    return {
        "available": True,
        "status": source.get("status"),
        "reasoning_snapshot_run_id": source.get("run_id"),
        "reasoning_snapshot_sha256": source.get("snapshot_sha256"),
        "runtime_hash_bound": source.get("hash_bound"),
        "classified_event_count": source.get("classified_event_count"),
        "transmission_channel_count": source.get(
            "transmission_channel_count"
        ),
        "transmission_channel_counts": source.get(
            "transmission_channel_counts", {}
        ),
        "regime_dimensions": dimensions,
        "candidate_hypotheses": hypotheses,
        "evidence_gaps": evidence_gaps,
        "scenario_graph_status": source.get("scenario_graph_status"),
        "expectation_gap_status": source.get("expectation_gap_status"),
        "directional_ticker_event_count": source.get(
            "directional_ticker_reasoning_event_count"
        ),
        "allowed_use": "supporting_sector_context_only",
        "can_influence_ticker_direction": False,
        "can_change_prediction": False,
        "can_clear_pipeline_evaluation": False,
    }


def _current_sector_thesis(
    thesis_review: dict[str, Any],
) -> dict[str, Any]:
    summary = thesis_review.get("summary") or {}
    thesis = thesis_review.get("thesis_snapshot") or {}
    analytical = thesis_review.get("analytical_review") or {}
    reasoning = _sector_reasoning_context(thesis_review)
    evidence_balance = analytical.get("evidence_balance") or {}
    return {
        "domain_profile": summary.get("domain_id")
        or thesis.get("domain_id"),
        "sector": "semiconductor",
        "thesis_level": "sector_thesis",
        "sector_stance": thesis.get("stance"),
        "expected_direction": thesis.get(
            "expected_direction"
        ),
        "confidence": thesis.get("confidence"),
        "confidence_interpretation": analytical.get(
            "confidence_interpretation"
        ),
        "thesis": thesis.get("thesis"),
        "source_run_count": evidence_balance.get(
            "raw_context_item_count"
        ),
        "required_lane_count": evidence_balance.get(
            "required_lane_count"
        ),
        "satisfied_required_lane_count": evidence_balance.get(
            "satisfied_required_lane_count"
        ),
        "warnings": analytical.get("quality_cautions", []),
        "verified_reasoning_context": reasoning,
        "source_review_run_id": thesis_review.get("run_id"),
        "allowed_use": (
            "supporting_context_for_exact_ticker_pipeline_review_only"
        ),
        "can_close_ticker_evidence_gap": False,
    }


def _ticker_prediction_review(
    *,
    ticker: str,
    contexts: list[dict[str, Any]],
    cases: list[dict[str, Any]],
    binding: dict[str, Any] | None,
) -> dict[str, Any]:
    complete = [
        item for item in contexts if item.get("review_complete") is True
    ]
    if not binding:
        status = "not_supplied"
    elif not contexts:
        status = "not_present_for_ticker"
    elif not complete:
        status = "prediction_review_quarantined"
    elif len(complete) < len(contexts):
        status = "prediction_review_partial"
    else:
        status = "prediction_review_ready"

    complete_identities = {
        (
            str(item.get("ticker") or "").upper(),
            item.get("selected_primary_model"),
            item.get("target_name"),
            item.get("timeframe"),
            item.get("context_fingerprint"),
        )
        for item in complete
    }
    case_identities = {
        (
            str(case.get("ticker") or "").upper(),
            case.get("model"),
            case.get("target_name"),
            case.get("timeframe"),
            case.get("context_fingerprint"),
        )
        for case in cases
    }
    aligned_identities = complete_identities & case_identities
    issue_counts = Counter(
        issue
        for item in contexts
        for issue in item.get("review_issues", [])
    )
    missing_counts = Counter(
        field
        for item in contexts
        for field in item.get("missing_lineage_fields", [])
    )
    target_counts = _counts(
        item.get("target_name") for item in contexts
    )
    model_type_counts = _counts(
        item.get("model_type") for item in contexts
    )
    identity_sample = [
        {
            key: item.get(key)
            for key in (
                "context_key",
                "model_context_id",
                "selected_primary_model",
                "model_type",
                "target_name",
                "timeframe",
                "context_fingerprint",
                "lineage_status",
                "missing_lineage_fields",
                "review_issues",
                "prediction_as_of",
                "review_complete",
            )
        }
        for item in sorted(
            contexts,
            key=lambda value: str(value.get("context_key") or ""),
        )[:12]
    ]
    return {
        "status": status,
        "ticker": ticker,
        "source_review_path": binding.get("path") if binding else None,
        "source_review_sha256": (
            binding.get("sha256") if binding else None
        ),
        "source_review_run_id": (
            binding.get("run_id") if binding else None
        ),
        "source_stage5_artifact_path": (
            binding.get("source_artifact_path") if binding else None
        ),
        "source_stage5_artifact_sha256": (
            binding.get("source_artifact_sha256")
            if binding
            else None
        ),
        "context_count": len(contexts),
        "complete_context_count": len(complete),
        "quarantined_context_count": len(contexts) - len(complete),
        "exact_identity_count": len(complete_identities),
        "aligned_pipeline_case_count": len(aligned_identities),
        "target_counts": target_counts,
        "model_type_counts": model_type_counts,
        "review_issue_counts": dict(sorted(issue_counts.items())),
        "missing_lineage_field_counts": dict(
            sorted(missing_counts.items())
        ),
        "context_identity_sample": identity_sample,
        "sample_limit": 12,
        "sample_truncated": len(contexts) > len(identity_sample),
        "allowed_use": (
            "supporting_stage5_readiness_review_only"
        ),
        "prediction_values_exposed": False,
        "can_fill_missing_lineage": False,
        "can_create_ticker_forecast": False,
        "can_clear_model_evaluation": False,
        "can_trade": False,
    }


def _ticker_feature_timeframe_audit(
    *,
    ticker: str,
    report: dict[str, Any] | None,
    binding: dict[str, Any] | None,
) -> dict[str, Any]:
    if not binding:
        return {
            "ticker": ticker,
            "status": "not_supplied",
            "allowed_use": "supporting_pipeline_readiness_audit_only",
            "can_override_timeframe": False,
            "can_create_ticker_forecast": False,
            "can_trade": False,
        }
    if not report:
        return {
            "ticker": ticker,
            "status": "ticker_not_audited",
            "source_audit_path": binding.get("path"),
            "source_audit_sha256": binding.get("sha256"),
            "allowed_use": "supporting_pipeline_readiness_audit_only",
            "can_override_timeframe": False,
            "can_create_ticker_forecast": False,
            "can_trade": False,
        }
    lineage = report.get("lineage") or {}
    return {
        "ticker": ticker,
        "status": report.get("status"),
        "row_count": report.get("row_count"),
        "declared_timeframe": lineage.get(
            "declared_timeframe"
        ),
        "observed_timeframe": lineage.get(
            "observed_timeframe"
        ),
        "resolved_timeframe": lineage.get(
            "resolved_timeframe"
        ),
        "datetime_timezone_aware": report.get(
            "datetime_timezone_aware"
        ),
        "source_audit_path": binding.get("path"),
        "source_audit_sha256": binding.get("sha256"),
        "features_sha256": binding.get("features_sha256"),
        "stage5_sha256": binding.get("stage5_sha256"),
        "stage5_relationship_status": binding.get(
            "stage5_relationship_status"
        ),
        "can_assert_feature_parentage": binding.get(
            "can_assert_feature_parentage"
        ),
        "allowed_use": "supporting_pipeline_readiness_audit_only",
        "can_override_timeframe": False,
        "can_create_ticker_forecast": False,
        "can_trade": False,
    }


def _current_ticker_candidates(
    thesis_review: dict[str, Any],
    pipeline_case_bindings: list[dict[str, Any]],
    *,
    ticker_evidence_binding: dict[str, Any] | None = None,
    prediction_review_binding: dict[str, Any] | None = None,
    feature_timeframe_audit_binding: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    boundary = thesis_review.get("ticker_bridge_boundary") or {}
    sector_reasoning = _sector_reasoning_context(thesis_review)
    source_candidates = boundary.get("ticker_candidates") or []
    cases_by_ticker: dict[str, list[dict[str, Any]]] = (
        defaultdict(list)
    )
    for binding in pipeline_case_bindings:
        cases_by_ticker[str(binding.get("ticker")).upper()].append(
            binding
        )
    predictions_by_ticker: dict[
        str, list[dict[str, Any]]
    ] = defaultdict(list)
    if prediction_review_binding:
        for context in prediction_review_binding.get("contexts", []):
            predictions_by_ticker[
                str(context.get("ticker")).upper()
            ].append(context)
    timeframe_audit_by_ticker = {
        str(item.get("ticker") or "").upper(): item
        for item in (
            feature_timeframe_audit_binding or {}
        ).get("ticker_timeframe_reports", [])
        if item.get("ticker")
    }
    ticker_evidence_records: dict[
        str, list[dict[str, Any]]
    ] = defaultdict(list)
    ticker_evidence_summary: dict[str, dict[str, Any]] = {}
    if ticker_evidence_binding:
        for record in ticker_evidence_binding.get("records", []):
            ticker_evidence_records[
                str(record.get("ticker")).upper()
            ].append(record)
        ticker_evidence_summary = {
            str(item.get("ticker")).upper(): item
            for item in ticker_evidence_binding.get(
                "ticker_summary", []
            )
            if item.get("ticker")
        }

    candidates = []
    for source in source_candidates:
        ticker = str(source.get("ticker") or "").upper()
        if not ticker:
            continue
        cases = cases_by_ticker.get(ticker, [])
        direct_records = [
            item
            for item in ticker_evidence_records.get(ticker, [])
            if item.get("ticker_thesis_eligible") is True
        ]
        evidence_summary = ticker_evidence_summary.get(
            ticker, {}
        )
        evidence_ready = bool(direct_records)
        negative_cases = [
            case
            for case in cases
            if case.get("case_classification")
            == "negative_evaluation_block_case"
        ]
        prediction_review = _ticker_prediction_review(
            ticker=ticker,
            contexts=predictions_by_ticker.get(ticker, []),
            cases=cases,
            binding=prediction_review_binding,
        )
        prediction_status = prediction_review.get("status")
        timeframe_audit = _ticker_feature_timeframe_audit(
            ticker=ticker,
            report=timeframe_audit_by_ticker.get(ticker),
            binding=feature_timeframe_audit_binding,
        )
        timeframe_blocked = timeframe_audit.get("status") in {
            "timeframe_cadence_mismatch",
            "timeframe_cadence_ambiguous",
        }
        required_inputs = [
            "realized_outcome_calibration_for_target_and_horizon",
        ]
        if timeframe_blocked:
            required_inputs.insert(
                0,
                "regenerate_stage2_stage3_with_cadence_validated_"
                "timeframe_before_stage4_stage5",
            )
        if prediction_status in {
            "not_supplied",
            "not_present_for_ticker",
        }:
            required_inputs.insert(
                0,
                "trustworthy_stage5_prediction_review_exact_identity",
            )
        elif prediction_status == "prediction_review_quarantined":
            required_inputs.insert(
                0,
                "repair_stage4_stage5_lineage_and_regenerate_"
                "immutable_prediction_review",
            )
        elif prediction_status == "prediction_review_partial":
            required_inputs.insert(
                0,
                "repair_incomplete_stage5_contexts_and_regenerate_"
                "immutable_prediction_review",
            )
        if not evidence_ready:
            required_inputs.insert(
                0,
                "ticker_specific_directional_evidence",
            )
        if not cases:
            required_inputs.append(
                "exact_ticker_model_target_timeframe_evaluation_case"
            )
        if negative_cases:
            required_inputs.append(
                "new_forward_development_data_after_blocked_case_window"
            )
        limitations = [
            "sector_context_is_not_direct_ticker_evidence",
        ]
        if timeframe_blocked:
            limitations.extend(
                [
                    "feature_timeframe_cadence_mismatch_blocks_"
                    "stage4_stage5_reuse",
                    "legacy_stage5_feature_parentage_not_hash_bound",
                ]
            )
        if prediction_status in {
            "not_supplied",
            "not_present_for_ticker",
        }:
            limitations.append(
                "no_trustworthy_saved_stage5_prediction_result"
            )
        elif prediction_status == "prediction_review_quarantined":
            limitations.append(
                "saved_stage5_output_quarantined_incomplete_"
                "lineage_or_semantics"
            )
        elif prediction_status == "prediction_review_partial":
            limitations.append(
                "saved_stage5_output_partially_quarantined"
            )
        else:
            limitations.append(
                "stage5_prediction_review_supporting_only_not_"
                "evaluation_or_outcome"
            )
        if not evidence_ready:
            limitations.append(
                "no_eligible_directional_ticker_evidence"
            )
        else:
            limitations.append(
                "ticker_evidence_does_not_override_pipeline_blocks"
            )
        if negative_cases:
            limitations.extend(
                [
                    "existing_pipeline_case_is_negative_evaluation_only",
                    "same_fold_retry_and_threshold_weakening_blocked",
                ]
            )
        if not cases:
            limitations.append(
                "no_exact_pipeline_model_case_attached"
            )
        candidates.append(
            {
                "ticker": ticker,
                "candidate_status": (
                    "ticker_evidence_ready_pipeline_blocked"
                    if evidence_ready
                    else "blocked_missing_ticker_evidence"
                ),
                "source_sector_candidate_status": source.get(
                    "candidate_status"
                ),
                "runs": len(cases),
                "overlay_ready_runs": 0,
                "blocked_runs": len(cases),
                "directional_ready_runs": 0,
                "neutral_ready_runs": 0,
                "dominant_focused_stance": "none",
                "focused_direction_counts": {},
                "exam_verdict_counts": _counts(
                    case.get("result_label") for case in cases
                ),
                "outcome_counts": {},
                "hit_rate_context": None,
                "average_realized_return_context": None,
                "supporting_as_of": [],
                "blocked_as_of": [
                    (case.get("mapping_run") or {}).get("as_of")
                    for case in cases
                ],
                "allocation_guidance": (
                    "blocked_until_direct_evidence_and_exact_pipeline_"
                    "readiness"
                ),
                "limitations": sorted(set(limitations)),
                "ticker_specific_evidence": {
                    "status": evidence_summary.get("status"),
                    "eligible_record_count": len(direct_records),
                    "corroborated_lane_count": (
                        evidence_summary.get(
                            "corroborated_lane_count", 0
                        )
                    ),
                    "corroborated_lanes": evidence_summary.get(
                        "corroborated_lanes", []
                    ),
                    "records": [
                        {
                            key: value
                            for key, value in item.items()
                            if key
                            in {
                                "ticker",
                                "evidence_type",
                                "stance_hint",
                                "source_identity",
                                "source_tier",
                                "published_at",
                                "source_locator",
                                "summary",
                                "record_sha256",
                            }
                        }
                        for item in direct_records
                    ],
                    "can_create_ticker_forecast": False,
                },
                "sector_context": {
                    "allowed_use": (
                        "supporting_context_only"
                    ),
                    "can_influence_ticker_direction": False,
                    "can_close_missing_pipeline_metric_plane": False,
                    "verified_reasoning": sector_reasoning,
                },
                "stage5_prediction_review": prediction_review,
                "feature_timeframe_audit": timeframe_audit,
                "exact_pipeline_contexts": [
                    {
                        key: value
                        for key, value in case.items()
                        if key
                        in {
                            "case_id",
                            "ticker",
                            "model",
                            "target_name",
                            "timeframe",
                            "context_fingerprint",
                            "case_status",
                            "case_classification",
                            "result_label",
                            "review_disposition",
                            "blocked_metric_planes",
                            "sha256",
                        }
                    }
                    for case in cases
                ],
                "required_next_inputs": required_inputs,
                "can_create_ticker_forecast": False,
                "can_train": False,
                "can_tune": False,
                "can_trade": False,
            }
        )
    return candidates


def _current_bridge_summary(
    ticker_candidates: list[dict[str, Any]],
    pipeline_case_bindings: list[dict[str, Any]],
    *,
    prediction_review_binding: dict[str, Any] | None = None,
    feature_timeframe_audit_binding: dict[str, Any] | None = None,
) -> dict[str, Any]:
    negative_case_count = sum(
        1
        for case in pipeline_case_bindings
        if case.get("case_classification")
        == "negative_evaluation_block_case"
    )
    ticker_evidence_ready_count = sum(
        1
        for item in ticker_candidates
        if item.get("candidate_status")
        == "ticker_evidence_ready_pipeline_blocked"
    )
    prediction_status_counts = _counts(
        (item.get("stage5_prediction_review") or {}).get("status")
        for item in ticker_candidates
    )
    prediction_context_count = sum(
        int(
            (item.get("stage5_prediction_review") or {}).get(
                "context_count", 0
            )
            or 0
        )
        for item in ticker_candidates
    )
    complete_prediction_context_count = sum(
        int(
            (item.get("stage5_prediction_review") or {}).get(
                "complete_context_count", 0
            )
            or 0
        )
        for item in ticker_candidates
    )
    timeframe_mismatch_tickers = [
        item.get("ticker")
        for item in ticker_candidates
        if (item.get("feature_timeframe_audit") or {}).get(
            "status"
        )
        in {
            "timeframe_cadence_mismatch",
            "timeframe_cadence_ambiguous",
        }
    ]
    return {
        "bridge_status": "ticker_pipeline_inputs_incomplete",
        "next_action": (
            "collect_ticker_specific_evidence_and_exact_pipeline_outputs"
        ),
        "sector_stance": "mixed",
        "run_count": len(pipeline_case_bindings),
        "ticker_candidate_count": len(ticker_candidates),
        "direct_ticker_thesis_ready_count": 0,
        "evidence_limited_direct_candidate_count": 0,
        "blocked_candidate_count": len(ticker_candidates),
        "ticker_evidence_ready_pipeline_blocked_count": (
            ticker_evidence_ready_count
        ),
        "missing_ticker_evidence_count": (
            len(ticker_candidates) - ticker_evidence_ready_count
        ),
        "exact_pipeline_case_count": len(
            pipeline_case_bindings
        ),
        "negative_pipeline_case_count": negative_case_count,
        "prediction_review_attached": bool(
            prediction_review_binding
        ),
        "prediction_review_status": (
            prediction_review_binding.get("status")
            if prediction_review_binding
            else "not_supplied"
        ),
        "prediction_review_ticker_status_counts": (
            prediction_status_counts
        ),
        "prediction_context_count": prediction_context_count,
        "complete_prediction_context_count": (
            complete_prediction_context_count
        ),
        "quarantined_prediction_context_count": (
            prediction_context_count
            - complete_prediction_context_count
        ),
        "feature_timeframe_audit_attached": bool(
            feature_timeframe_audit_binding
        ),
        "feature_timeframe_audit_status": (
            feature_timeframe_audit_binding.get("status")
            if feature_timeframe_audit_binding
            else "not_supplied"
        ),
        "timeframe_mismatch_ticker_count": len(
            timeframe_mismatch_tickers
        ),
        "timeframe_mismatch_tickers": timeframe_mismatch_tickers,
        "can_create_ticker_basket_review": False,
        "can_attach_sector_context_to_exact_pipeline_review": True,
        "can_create_ticker_forecast": False,
        "can_train": False,
        "can_tune": False,
        "can_change_analyst_weights": False,
        "can_write_learning_memory": False,
        "can_write_config": False,
        "can_trade": False,
    }


def _exact_ticker_pipeline_contract() -> dict[str, Any]:
    return {
        "context_identity_fields": [
            "ticker",
            "model",
            "target_name",
            "timeframe",
            "context_fingerprint",
        ],
        "required_input_families": [
            "sector_context_supporting_only",
            "ticker_specific_directional_evidence",
            "stage5_prediction_review_exact_identity",
            "locked_model_evaluation_and_feature_stability",
            "realized_outcome_calibration",
        ],
        "rules": [
            (
                "Sector evidence may be attached only as supporting "
                "context and cannot determine ticker direction."
            ),
            (
                "Raw statement facts and ratios do not become "
                "directional ticker evidence without an explicit "
                "company mechanism and source-bound catalyst."
            ),
            (
                "A negative model-evaluation case is not a realized "
                "forecast outcome and cannot be used as sector evidence."
            ),
            (
                "A failure or tuning proposal remains scoped to the "
                "same ticker/model/target/timeframe/context fingerprint."
            ),
        ],
        "automatic_scope_inheritance_allowed": False,
        "automatic_ticker_forecast_allowed": False,
        "automatic_training_or_tuning_allowed": False,
        "automatic_trading_allowed": False,
    }


def _current_bridge_tasks(
    ticker_candidates: list[dict[str, Any]],
    pipeline_case_bindings: list[dict[str, Any]],
    *,
    prediction_review_binding: dict[str, Any] | None = None,
    feature_timeframe_audit_binding: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    missing_evidence_tickers = [
        item.get("ticker")
        for item in ticker_candidates
        if item.get("candidate_status")
        == "blocked_missing_ticker_evidence"
    ]
    tasks = [
        {
            "priority": "P0",
            "task_id": (
                "build_ticker_specific_evidence_packets"
            ),
            "description": (
                "Create source-bound company catalyst/mechanism evidence "
                "for each ticker; sector context cannot fill this slot."
            ),
            "tickers": missing_evidence_tickers,
        },
    ]
    quarantined_tickers = [
        item.get("ticker")
        for item in ticker_candidates
        if (item.get("stage5_prediction_review") or {}).get(
            "status"
        )
        in {
            "prediction_review_quarantined",
            "prediction_review_partial",
        }
    ]
    timeframe_mismatch_tickers = [
        item.get("ticker")
        for item in ticker_candidates
        if (item.get("feature_timeframe_audit") or {}).get(
            "status"
        )
        in {
            "timeframe_cadence_mismatch",
            "timeframe_cadence_ambiguous",
        }
    ]
    missing_prediction_tickers = [
        item.get("ticker")
        for item in ticker_candidates
        if (item.get("stage5_prediction_review") or {}).get(
            "status"
        )
        in {"not_supplied", "not_present_for_ticker"}
    ]
    if quarantined_tickers:
        tasks.append(
            {
                "priority": "P0",
                "task_id": (
                    "repair_stage4_stage5_lineage_and_regenerate"
                ),
                "description": (
                    "Real Stage 5 outputs exist but remain quarantined. "
                    "Regenerate them through the repaired lineage path; "
                    "do not backfill mutable values into the old artifact."
                ),
                "tickers": quarantined_tickers,
                "source_review_path": (
                    prediction_review_binding.get("path")
                    if prediction_review_binding
                    else None
                ),
            }
        )
    if timeframe_mismatch_tickers:
        tasks.insert(
            1,
            {
                "priority": "P0",
                "task_id": (
                    "regenerate_cadence_validated_stage2_stage3"
                ),
                "description": (
                    "Rebuild features from saved source partitions with "
                    "declared timeframe verified against observed cadence "
                    "before any Stage 4 or Stage 5 reuse."
                ),
                "tickers": timeframe_mismatch_tickers,
                "source_audit_path": (
                    feature_timeframe_audit_binding.get("path")
                    if feature_timeframe_audit_binding
                    else None
                ),
            },
        )
    if missing_prediction_tickers:
        tasks.append(
            {
                "priority": "P0",
                "task_id": (
                    "materialize_trustworthy_stage5_prediction_reviews"
                ),
                "description": (
                    "Produce exact ticker/model/target/timeframe/context "
                    "prediction-review artifacts only when real Stage 5 "
                    "outputs exist."
                ),
                "tickers": missing_prediction_tickers,
            }
        )
    negative_tickers = sorted(
        {
            str(item.get("ticker"))
            for item in pipeline_case_bindings
            if item.get("case_classification")
            == "negative_evaluation_block_case"
        }
    )
    if negative_tickers:
        tasks.append(
            {
                "priority": "P0",
                "task_id": "respect_negative_pipeline_cases",
                "description": (
                    "Wait for registered new forward development data; "
                    "do not erase the negative case by reusing folds or "
                    "weakening thresholds."
                ),
                "tickers": negative_tickers,
            }
        )
    tasks.append(
        {
            "priority": "P1",
            "task_id": "build_exact_ticker_readiness_matrix",
            "description": (
                "Track evidence, prediction, evaluation, outcome, and "
                "sector-context readiness separately per exact identity."
            ),
            "tickers": [
                item.get("ticker") for item in ticker_candidates
            ],
        }
    )
    return tasks


def _domain_analyst_contract(domain_profile: str, sector: str) -> dict[str, Any]:
    return {
        "profile_id": domain_profile,
        "sector": sector,
        "required_outputs": [
            "sector_thesis",
            "sector_evidence",
            "ticker_candidate_map",
            "ticker_specific_evidence",
            "risks_and_counter_thesis",
        ],
        "thesis_levels": [
            "sector_context_only",
            "basket_candidate",
            "direct_ticker_thesis",
        ],
        "rule": "A sector thesis may propose a basket or candidate list, but it is not a ticker thesis until direct ticker evidence supports the company.",
    }


def _sector_thesis(
    research_batch: dict[str, Any],
    runs: list[dict[str, Any]],
    domain_profile: str,
    sector: str,
) -> dict[str, Any]:
    summary = research_batch.get("summary", {})
    stance_counts = summary.get("research_stance_counts", _counts(run.get("research_stance") for run in runs))
    constructive = int(stance_counts.get("constructive", 0))
    risk = int(stance_counts.get("risk_heavy", 0))
    insufficient = int(stance_counts.get("insufficient_data", 0))
    mixed = int(stance_counts.get("mixed", 0))
    if constructive > max(risk, mixed, insufficient):
        sector_stance = "constructive"
    elif risk > max(constructive, mixed, insufficient):
        sector_stance = "risk_heavy"
    elif insufficient:
        sector_stance = "evidence_limited"
    else:
        sector_stance = "mixed"
    thesis = _sector_thesis_text(sector, sector_stance, summary)
    return {
        "domain_profile": domain_profile,
        "sector": sector,
        "thesis_level": "sector_thesis",
        "sector_stance": sector_stance,
        "thesis": thesis,
        "source_run_count": len(runs),
        "research_stance_counts": stance_counts,
        "exam_verdict_counts": summary.get("exam_verdict_counts", {}),
        "evidence_quality_counts": summary.get("evidence_quality_counts", {}),
        "hit_rate_context": summary.get("hit_rate"),
        "average_realized_return_context": summary.get("average_realized_return"),
        "warnings": _sector_warnings(summary),
    }


def _sector_thesis_text(sector: str, sector_stance: str, summary: dict[str, Any]) -> str:
    if sector_stance == "constructive":
        return f"The {sector} domain has a constructive replay context, but ticker-level mapping still requires direct company evidence."
    if sector_stance == "risk_heavy":
        return f"The {sector} domain has a risk-heavy replay context; candidate tickers should remain blocked until risk evidence is resolved."
    if sector_stance == "evidence_limited":
        weak = summary.get("weak_evidence_runs", 0)
        return f"The {sector} domain has useful context, but {weak} replay windows are evidence-limited and should not become ticker theses."
    return f"The {sector} domain has a mixed replay context; keep it as sector context unless ticker-specific evidence is strong."


def _sector_warnings(summary: dict[str, Any]) -> list[str]:
    warnings: list[str] = []
    if summary.get("weak_evidence_runs", 0):
        warnings.append("Some replay windows have weak or partial evidence coverage.")
    if summary.get("research_inconclusive_runs", 0):
        warnings.append("Some research views remain neutral or insufficient_data, which may be correct behavior.")
    if summary.get("quality_blocked_runs", 0):
        warnings.append("At least one replay run has price-quality warnings.")
    return warnings


def _ticker_candidates(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for run in runs:
        ticker = str(run.get("price_ticker") or "").upper()
        if ticker:
            grouped[ticker].append(run)
    return [_ticker_candidate(ticker, items) for ticker, items in sorted(grouped.items())]


def _ticker_candidate(ticker: str, runs: list[dict[str, Any]]) -> dict[str, Any]:
    ready_runs = [run for run in runs if run.get("focused_overlay_status") == "focused_overlay_ready" and run.get("focused_overlay_applied")]
    blocked_runs = [run for run in runs if run.get("focused_overlay_status") and run.get("focused_overlay_status") != "focused_overlay_ready"]
    directional_ready = [run for run in ready_runs if run.get("research_expected_direction") in {"bullish", "bearish"}]
    neutral_ready = [run for run in ready_runs if run.get("research_expected_direction") == "neutral"]
    if directional_ready:
        status = "direct_ticker_thesis_ready"
    elif ready_runs:
        status = "ticker_context_ready"
    elif blocked_runs:
        status = "blocked_missing_ticker_evidence"
    else:
        status = "sector_context_only"
    return {
        "ticker": ticker,
        "candidate_status": status,
        "runs": len(runs),
        "overlay_ready_runs": len(ready_runs),
        "blocked_runs": len(blocked_runs),
        "directional_ready_runs": len(directional_ready),
        "neutral_ready_runs": len(neutral_ready),
        "dominant_focused_stance": _dominant(run.get("research_stance") for run in ready_runs) or "none",
        "focused_direction_counts": _counts(run.get("research_expected_direction") for run in ready_runs),
        "exam_verdict_counts": _counts(run.get("exam_verdict") for run in runs),
        "outcome_counts": _counts(run.get("outcome_label") for run in runs),
        "hit_rate_context": _hit_rate(runs),
        "average_realized_return_context": _average_return(runs),
        "supporting_as_of": [run.get("as_of") for run in ready_runs],
        "blocked_as_of": [run.get("as_of") for run in blocked_runs],
        "allocation_guidance": _allocation_guidance(status),
        "limitations": _candidate_limitations(status, blocked_runs),
    }


def _allocation_guidance(status: str) -> str:
    if status == "direct_ticker_thesis_ready":
        return "eligible_for_reviewed_ticker_candidate"
    if status == "ticker_context_ready":
        return "watchlist_context_only"
    if status == "blocked_missing_ticker_evidence":
        return "blocked_until_direct_evidence_backfill"
    return "sector_context_only"


def _candidate_limitations(status: str, blocked_runs: list[dict[str, Any]]) -> list[str]:
    limitations: list[str] = []
    if status != "direct_ticker_thesis_ready":
        limitations.append("not_a_directional_ticker_thesis")
    if blocked_runs:
        limitations.append("some_windows_blocked_by_weak_direct_evidence")
    return limitations


def _mapping_run(run: dict[str, Any]) -> dict[str, Any]:
    return {
        "as_of": run.get("as_of"),
        "horizon_days": run.get("horizon_days"),
        "price_ticker": run.get("price_ticker"),
        "sector_signal_level": "sector_context",
        "ticker_signal_level": _run_signal_level(run),
        "research_stance": run.get("research_stance"),
        "research_expected_direction": run.get("research_expected_direction"),
        "ticker_specificity": run.get("ticker_specificity"),
        "exam_verdict": run.get("exam_verdict"),
        "focused_overlay_status": run.get("focused_overlay_status"),
        "focused_overlay_applied": run.get("focused_overlay_applied"),
        "outcome_label": run.get("outcome_label"),
        "realized_return": run.get("realized_return"),
    }


def _run_signal_level(run: dict[str, Any]) -> str:
    if run.get("focused_overlay_status") != "focused_overlay_ready":
        return "blocked_missing_ticker_evidence"
    if run.get("research_expected_direction") in {"bullish", "bearish"}:
        return "direct_ticker_thesis"
    return "ticker_context"


def _summary(sector_thesis: dict[str, Any], ticker_candidates: list[dict[str, Any]], runs: list[dict[str, Any]]) -> dict[str, Any]:
    direct_ready = [candidate for candidate in ticker_candidates if candidate.get("candidate_status") == "direct_ticker_thesis_ready"]
    blocked = [candidate for candidate in ticker_candidates if candidate.get("candidate_status") == "blocked_missing_ticker_evidence"]
    evidence_limited_direct = [
        candidate
        for candidate in direct_ready
        if int(candidate.get("blocked_runs") or 0) > 0
    ]
    if not runs:
        status = "no_runs"
        next_action = "provide_focused_overlay_replay_batch"
    elif direct_ready and (blocked or evidence_limited_direct):
        status = "partial_basket_ready"
        next_action = "review_ready_candidates_and_backfill_blocked_tickers"
    elif direct_ready:
        status = "ticker_basket_ready_for_review"
        next_action = "create_human_review_packet"
    else:
        status = "sector_context_only"
        next_action = "collect_direct_ticker_evidence"
    return {
        "bridge_status": status,
        "next_action": next_action,
        "sector_stance": sector_thesis.get("sector_stance"),
        "run_count": len(runs),
        "ticker_candidate_count": len(ticker_candidates),
        "direct_ticker_thesis_ready_count": len(direct_ready),
        "evidence_limited_direct_candidate_count": len(evidence_limited_direct),
        "blocked_candidate_count": len(blocked),
        "can_create_ticker_basket_review": bool(direct_ready),
        "can_change_analyst_weights": False,
        "can_write_learning_memory": False,
    }


def _tasks(summary: dict[str, Any], ticker_candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    ready = [candidate for candidate in ticker_candidates if candidate.get("candidate_status") == "direct_ticker_thesis_ready"]
    blocked = [candidate for candidate in ticker_candidates if candidate.get("candidate_status") == "blocked_missing_ticker_evidence"]
    if ready:
        tasks.append(
            {
                "priority": "P0",
                "task_id": "create_sector_to_ticker_review_packet",
                "description": "Create a human review packet that keeps the sector thesis separate from ticker-specific candidates.",
                "tickers": [candidate.get("ticker") for candidate in ready],
            }
        )
    if blocked:
        tasks.append(
            {
                "priority": "P1",
                "task_id": "backfill_blocked_ticker_evidence",
                "description": "Backfill direct ticker evidence before blocked candidates can join the sector basket.",
                "tickers": [candidate.get("ticker") for candidate in blocked],
            }
        )
    tasks.append(
        {
            "priority": "P2",
            "task_id": "formalize_domain_specialist_profile",
            "description": "Use this bridge contract as the standard output shape for the first domain specialist before cloning profiles.",
        }
    )
    return tasks


def _commands(research_batch_path: str | Path, domain_profile: str, sector: str) -> dict[str, str]:
    return {
        "rerun_bridge": (
            "python run_agent_sector_to_ticker_bridge.py "
            f"--research-batch-json {research_batch_path} "
            f"--domain-profile {domain_profile} --sector {sector} "
            "--output-dir reports\\dean_os\\sector_thesis_to_ticker_basket_current"
        )
    }


def _recommendations(summary: dict[str, Any]) -> list[str]:
    if summary.get("bridge_status") == "partial_basket_ready":
        return [
            "Use ready ticker candidates for review only; keep blocked tickers out of the candidate basket until direct evidence improves.",
            "Do not clone new domain profiles until this sector-to-ticker contract is stable.",
        ]
    if summary.get("bridge_status") == "ticker_basket_ready_for_review":
        return ["Create a human review packet before any learning, weighting, or paper-autonomy use."]
    if summary.get("bridge_status") == "sector_context_only":
        return ["Keep this as a sector thesis and collect direct ticker evidence before candidate mapping."]
    return ["Provide a focused-overlay integrated research replay batch before mapping sector thesis to tickers."]


def _hit_rate(runs: list[dict[str, Any]]) -> float | None:
    evaluated = [run for run in runs if run.get("evaluation_status") == "evaluated"]
    if not evaluated:
        return None
    hits = sum(1 for run in evaluated if run.get("outcome_label") == "hit")
    return round(hits / len(evaluated), 6)


def _average_return(runs: list[dict[str, Any]]) -> float | None:
    values = [float(run["realized_return"]) for run in runs if run.get("realized_return") is not None]
    if not values:
        return None
    return round(sum(values) / len(values), 6)


def _dominant(values: Any) -> str | None:
    counts = Counter(str(value) for value in values if value)
    if not counts:
        return None
    return counts.most_common(1)[0][0]


def _counts(values: Any) -> dict[str, int]:
    counts = Counter(str(value) for value in values if value)
    return dict(sorted(counts.items()))


def _load_json(path: str | Path) -> dict[str, Any]:
    from dean_os.draft.dean_os_agent_system_v7.dean_os.dean_paths import DeanPaths

    return DeanPaths.load_json(path)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run_id(prefix: str) -> str:
    return f"{prefix}_{utc_now_iso().replace(':', '').replace('-', '').replace('.', '_')}"
