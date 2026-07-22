from __future__ import annotations

import hashlib
import html
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from dean_os.artifact_writer import ReviewArtifactWriter
from dean_os.schemas import utc_now_iso


class HypothesisEvidenceGapReview:
    """Map analyst gaps to verified local evidence without claiming closure."""

    contract = "dean_hypothesis_evidence_gap_review_v1"

    def __init__(
        self,
        output_dir: str | Path = (
            "reports/dean_os/hypothesis_evidence_gap_review_current"
        ),
    ) -> None:
        self.output_dir = Path(output_dir)

    def build(
        self,
        *,
        analyst_review_path: str | Path,
        fundamental_artifact_path: str | Path,
        ratio_artifact_path: str | Path | None = None,
        primary_snapshot_path: str | Path | None = None,
        operational_metrics_path: str | Path | None = None,
        filing_order_evidence_path: str | Path | None = None,
        as_of: str,
        save: bool = True,
    ) -> dict[str, Any]:
        cutoff = _timestamp(as_of, "as_of")
        analyst_path, analyst = _verified_json(analyst_review_path, cutoff)
        fundamental_path, fundamental = _verified_json(
            fundamental_artifact_path, cutoff
        )
        ratio_path: Path | None = None
        ratios: dict[str, Any] = {}
        if ratio_artifact_path:
            ratio_path, ratios = _verified_json(ratio_artifact_path, cutoff)
        primary_path: Path | None = None
        primary: dict[str, Any] = {}
        if primary_snapshot_path:
            primary_path, primary = _verified_json(primary_snapshot_path, cutoff)
        operational_path: Path | None = None
        operational: dict[str, Any] = {}
        if operational_metrics_path:
            operational_path, operational = _verified_json(
                operational_metrics_path, cutoff
            )
        filing_order_path: Path | None = None
        filing_order: dict[str, Any] = {}
        if filing_order_evidence_path:
            filing_order_path, filing_order = _verified_json(
                filing_order_evidence_path, cutoff
            )

        _validate_analyst(analyst)
        _validate_review_only(fundamental, "fundamental")
        if ratios:
            _validate_review_only(ratios, "ratios")
        if primary:
            _validate_primary_snapshot(primary)
        if operational:
            _validate_operational_metrics(operational)
        if filing_order:
            _validate_filing_order_evidence(filing_order)

        facts = list(fundamental.get("facts") or [])
        ratio_rows = list(ratios.get("ratios") or [])
        primary_context = _primary_context(primary, primary_path)
        operational_records = list(operational.get("accepted_records") or [])
        filing_order_observations = list(filing_order.get("observations") or [])
        gaps = list(
            ((analyst.get("agent_report") or {}).get("metrics_snapshot") or {}).get(
                "evidence_gaps"
            )
            or []
        )
        hypotheses = list(
            ((analyst.get("agent_report") or {}).get("metrics_snapshot") or {}).get(
                "hypotheses"
            )
            or []
        )
        reviews = [
            _review_gap(
                gap,
                facts=facts,
                ratios=ratio_rows,
                primary_context=primary_context,
                operational_records=operational_records,
                filing_order_observations=filing_order_observations,
                hypotheses=hypotheses,
            )
            for gap in gaps
        ]
        status_counts: dict[str, int] = {}
        for item in reviews:
            status_counts[item["resolution_status"]] = (
                status_counts.get(item["resolution_status"], 0) + 1
            )
        replay_tasks = [
            _replay_task(hypothesis, reviews) for hypothesis in hypotheses
        ]
        created_at = utc_now_iso()
        run_id = "hypothesis_evidence_gap_review_" + created_at.replace(
            ":", ""
        ).replace("+00:00", "Z")
        payload: dict[str, Any] = {
            "run_id": run_id,
            "created_at": created_at,
            "mode": "hypothesis_evidence_gap_review",
            "contract": self.contract,
            "status": "hypothesis_gaps_reviewed_manual_action_required",
            "inputs": {
                "as_of": as_of,
                "analyst_review": _ref(analyst_path),
                "fundamental_artifact": _ref(fundamental_path),
                "ratio_artifact": _ref(ratio_path) if ratio_path else None,
                "primary_snapshot": _ref(primary_path) if primary_path else None,
                "operational_metrics": _ref(operational_path) if operational_path else None,
                "filing_order_evidence": _ref(filing_order_path) if filing_order_path else None,
            },
            "summary": {
                "hypothesis_count": len(hypotheses),
                "gap_count": len(gaps),
                "status_counts": dict(sorted(status_counts.items())),
                "fully_resolved_gap_count": 0,
                "replay_task_candidate_count": len(replay_tasks),
                "replay_task_registration_allowed": False,
                "can_trade": False,
            },
            "gap_reviews": reviews,
            "replay_task_candidates": replay_tasks,
            "safety": {
                "review_only": True,
                "automatic_gap_closure_performed": False,
                "replay_task_registration_performed": False,
                "learning_write_performed": False,
                "production_config_write_performed": False,
                "training_run_performed": False,
                "broker_access_performed": False,
                "live_execution_performed": False,
                "can_trade": False,
            },
        }
        if save:
            payload["saved_paths"] = ReviewArtifactWriter(self.output_dir).write(
                payload=payload,
                markdown=_render_markdown(payload),
                run_id=run_id,
            )
        return payload


def _review_gap(
    gap: dict[str, Any],
    *,
    facts: list[dict[str, Any]],
    ratios: list[dict[str, Any]],
    primary_context: dict[str, Any],
    operational_records: list[dict[str, Any]],
    filing_order_observations: list[dict[str, Any]],
    hypotheses: list[dict[str, Any]],
) -> dict[str, Any]:
    description = str(gap.get("description") or "")
    lowered = description.lower()
    evidence: list[dict[str, Any]] = []
    status = "missing"
    limitations: list[str] = []

    if "inventory levels" in lowered:
        inventory = _facts(facts, "inventory")
        evidence.extend(_fact_refs(inventory))
        if inventory:
            status = "partial_supported"
            limitations.extend(
                [
                    "Issuer inventory is not full supply-chain inventory.",
                    "Periods and currencies are not fully comparable.",
                ]
            )
    elif "capex breakdown" in lowered:
        capex = _facts(facts, "capital_expenditure")
        capex_ratios = [
            row for row in ratios if row.get("ratio_name") == "capex_to_revenue"
        ]
        evidence.extend(_fact_refs(capex))
        evidence.extend(_ratio_refs(capex_ratios))
        if capex:
            status = "partial_supported"
            limitations.append(
                "SEC totals do not separate maintenance capex from growth capex."
            )
    elif "production capacity" in lowered or "utilization" in lowered:
        metrics = _operational_refs(
            operational_records,
            {"capacity", "production_capacity", "utilization", "capacity_utilization", "yield"},
        )
        evidence.extend(metrics)
        evidence.extend(primary_context.get("capacity", []))
        if _has_observed_metric(metrics):
            status = "partial_supported"
        elif metrics or evidence:
            status = "context_only_not_resolved"
        if not metrics:
            limitations.append("No normalized utilization-rate series is available.")
        else:
            limitations.append("Operational observations require manual comparability and methodology review.")
    elif "backlog" in lowered:
        metrics = _operational_refs(operational_records, {"backlog", "equipment_orders", "orders"})
        evidence.extend(metrics)
        evidence.extend(_filing_order_refs(filing_order_observations))
        evidence.extend(primary_context.get("backlog", []))
        if _has_observed_metric(metrics):
            status = "partial_supported"
        elif metrics or evidence:
            status = "context_only_not_resolved"
        if filing_order_observations and status != "partial_supported":
            status = "partial_supported"
            limitations.append("Remaining performance obligation is only a contracted-revenue proxy, not full order backlog.")
        if not metrics:
            limitations.append("No source-bound quantitative backlog metric is available.")
    elif "capex cycle will sustain" in lowered:
        evidence.extend(_fact_refs(_facts(facts, "capital_expenditure")))
        if evidence:
            status = "partial_supported"
        limitations.append("One historical capex observation cannot validate a 180-day path.")
    elif "ai demand growth" in lowered:
        evidence.extend(_fact_refs(_facts(facts, "revenue")))
        if evidence:
            status = "context_only_not_resolved"
        limitations.append("Revenue totals do not isolate AI demand acceleration.")
    elif "supply constraints will persist" in lowered:
        evidence.extend(_fact_refs(_facts(facts, "inventory")))
        if evidence:
            status = "partial_supported"
        limitations.append("Inventory levels alone do not prove persistent supply constraints.")
    elif "equipment order" in lowered:
        metrics = _operational_refs(operational_records, {"equipment_orders", "orders", "backlog"})
        evidence.extend(metrics)
        if _has_observed_metric(metrics):
            status = "partial_supported"
        elif metrics:
            status = "context_only_not_resolved"
        else:
            limitations.append("No verified supplier equipment-order series is present.")
    elif "hyperscaler capex guidance" in lowered:
        limitations.append("No verified earnings-call guidance/estimate comparison is present.")
    elif "enterprise ai roi" in lowered:
        limitations.append("No verified early-adopter ROI dataset is present.")
    elif "lead time" in lowered:
        metrics = _operational_refs(operational_records, {"lead_time", "lead_time_days", "delivery_lead_time"})
        evidence.extend(metrics)
        if _has_observed_metric(metrics):
            status = "partial_supported"
        elif metrics:
            status = "context_only_not_resolved"
        else:
            limitations.append("No multi-supplier lead-time series is present.")

    linked = [
        item["hypothesis_id"]
        for item in hypotheses
        if _gap_matches_hypothesis(lowered, str(item.get("hypothesis") or "").lower())
    ]
    return {
        "gap_id": gap.get("gap_id"),
        "description": description,
        "priority": gap.get("priority"),
        "expected_source_type": gap.get("expected_source_type"),
        "resolution_status": status,
        "linked_hypothesis_ids": linked,
        "supporting_evidence": evidence,
        "limitations": limitations or ["No compatible verified local evidence was found."],
        "manual_review_required": True,
        "automatic_closure_allowed": False,
    }


def _replay_task(
    hypothesis: dict[str, Any], gap_reviews: list[dict[str, Any]]
) -> dict[str, Any]:
    hypothesis_id = hypothesis.get("hypothesis_id")
    linked = [
        item["gap_id"]
        for item in gap_reviews
        if hypothesis_id in item.get("linked_hypothesis_ids", [])
    ]
    return {
        "task_id": f"replay_candidate_{hypothesis_id}",
        "hypothesis_id": hypothesis_id,
        "hypothesis": hypothesis.get("hypothesis"),
        "horizons_to_check": hypothesis.get("horizons_to_check", []),
        "expected_observations": hypothesis.get("expected_observations", []),
        "invalidation_signals": hypothesis.get("invalidation_signals", []),
        "linked_gap_ids": linked,
        "status": "proposed_not_registered",
        "manual_review_required": True,
        "registration_allowed": False,
    }


def _validate_operational_metrics(payload: dict[str, Any]) -> None:
    if payload.get("contract") != "dean_industry_operational_metrics_v1":
        raise ValueError("unsupported operational metrics contract")
    if not (payload.get("safety") or {}).get("review_only"):
        raise ValueError("operational metrics artifact must be review-only")


def _validate_filing_order_evidence(payload: dict[str, Any]) -> None:
    if payload.get("contract") != "dean_filing_order_evidence_v1":
        raise ValueError("unsupported filing order evidence contract")
    if not (payload.get("safety") or {}).get("review_only"):
        raise ValueError("filing order evidence artifact must be review-only")


def _filing_order_refs(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "source": "filing_order_evidence",
            "ticker": row.get("ticker"),
            "metric_name": row.get("metric_name"),
            "value": row.get("value"),
            "unit": row.get("unit"),
            "period": row.get("period"),
            "available_at": row.get("available_at"),
            "accession_number": row.get("accession_number"),
            "source_locator": row.get("source_locator"),
            "source_sha256": row.get("source_sha256"),
            "observation_sha256": row.get("observation_sha256"),
            "evidence_role": row.get("semantic_role"),
        }
        for row in records
        if row.get("gap_support_role") == "partial_support_only"
        and row.get("current_gap_support_eligible", True) is True
    ]


def _operational_refs(
    records: list[dict[str, Any]], metric_names: set[str]
) -> list[dict[str, Any]]:
    refs = []
    for row in records:
        if row.get("metric_name") not in metric_names:
            continue
        if row.get("lifecycle_status") != "active":
            continue
        refs.append(
            {
                "source": "industry_operational_metrics",
                "record_id": row.get("record_id"),
                "entity": row.get("entity"),
                "metric_name": row.get("metric_name"),
                "value": row.get("value"),
                "unit": row.get("unit"),
                "period": row.get("period"),
                "available_at": row.get("available_at"),
                "value_kind": row.get("value_kind"),
                "source_locator": row.get("source_locator"),
                "source_sha256": row.get("source_sha256"),
                "observation_sha256": row.get("observation_sha256"),
                "evidence_role": (
                    "observed_metric" if row.get("value_kind") == "actual" else "forward_or_estimated_context"
                ),
            }
        )
    return refs


def _has_observed_metric(refs: list[dict[str, Any]]) -> bool:
    return any(item.get("evidence_role") == "observed_metric" for item in refs)


def _gap_matches_hypothesis(gap: str, hypothesis: str) -> bool:
    if "observation needed to test hypothesis:" in gap:
        return bool(hypothesis and hypothesis in gap)
    if "capex cycle" in hypothesis:
        return any(
            term in gap
            for term in ("capex breakdown", "equipment order", "production capacity")
        )
    if "ai demand" in hypothesis:
        return any(
            term in gap
            for term in ("backlog", "hyperscaler capex", "enterprise ai roi")
        )
    if "supply constraints" in hypothesis:
        return any(
            term in gap
            for term in (
                "equipment order",
                "backlog",
                "production capacity",
                "lead time",
                "inventory levels",
            )
        )
    return False


def _facts(facts: list[dict[str, Any]], metric: str) -> list[dict[str, Any]]:
    return [item for item in facts if item.get("metric_name") == metric]


def _fact_refs(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "kind": "sec_fact",
            "ticker": item.get("ticker"),
            "metric": item.get("metric_name"),
            "value": item.get("value"),
            "unit": item.get("unit"),
            "period": item.get("period"),
            "available_at": item.get("available_at"),
            "accession_number": item.get("accession_number"),
            "fact_sha256": item.get("fact_sha256"),
        }
        for item in items
    ]


def _ratio_refs(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "kind": "derived_ratio",
            "ticker": item.get("ticker"),
            "ratio": item.get("ratio_name"),
            "value": item.get("value"),
            "period": item.get("period"),
            "available_at": item.get("available_at"),
            "ratio_sha256": item.get("ratio_sha256"),
        }
        for item in items
    ]


def _primary_context(payload: dict[str, Any], manifest_path: Path | None) -> dict[str, Any]:
    result: dict[str, list[dict[str, Any]]] = {"capacity": [], "backlog": []}
    if not payload or manifest_path is None:
        return result
    for snapshot in payload.get("snapshots") or []:
        raw_path = Path(str(snapshot.get("immutable_path") or ""))
        if not raw_path.is_file() or _sha256(raw_path) != snapshot.get("sha256"):
            continue
        text = html.unescape(raw_path.read_text(encoding="utf-8", errors="ignore"))
        text = re.sub(r"<[^>]+>", " ", text)
        text = " ".join(text.split())
        for family, terms in {
            "capacity": ("capacity", "utilization"),
            "backlog": ("backlog",),
        }.items():
            for term in terms:
                match = re.search(re.escape(term), text, flags=re.IGNORECASE)
                if not match:
                    continue
                start = max(0, match.start() - 140)
                end = min(len(text), match.end() + 220)
                result[family].append(
                    {
                        "kind": "filing_text_context",
                        "ticker": snapshot.get("ticker"),
                        "accession_number": snapshot.get("accession_number"),
                        "term": term,
                        "snippet": text[start:end],
                        "document_sha256": snapshot.get("sha256"),
                        "available_at": snapshot.get("accepted_at"),
                        "context_only": True,
                    }
                )
    return result


def _validate_analyst(payload: dict[str, Any]) -> None:
    if payload.get("contract") != "dean_domain_analyst_review_run_v1":
        raise ValueError("unsupported analyst review contract")
    if (payload.get("safety") or {}).get("review_only") is not True:
        raise ValueError("analyst review is not review-only")


def _validate_review_only(payload: dict[str, Any], label: str) -> None:
    safety = payload.get("safety") or {}
    if safety.get("review_only") is not True:
        raise ValueError(f"{label} artifact is not review-only")
    if safety.get("live_execution_performed") is True:
        raise ValueError(f"{label} artifact performed live execution")


def _validate_primary_snapshot(payload: dict[str, Any]) -> None:
    if payload.get("snapshot_contract") != "dean_sec_primary_document_snapshot_v1":
        raise ValueError("unsupported primary snapshot contract")
    safety = payload.get("safety") or {}
    forbidden = (
        "pipeline_run_performed",
        "training_run_performed",
        "learning_write_performed",
        "production_config_write_performed",
        "paper_execution_performed",
        "live_execution_performed",
    )
    if safety.get("can_trade") is not False or any(
        safety.get(key) is True for key in forbidden
    ):
        raise ValueError("primary snapshot violated review-only consumption boundary")
    if not payload.get("snapshots"):
        raise ValueError("primary snapshot contains no saved documents")


def _verified_json(value: str | Path, cutoff: datetime) -> tuple[Path, dict[str, Any]]:
    path = Path(value)
    if path.is_dir():
        path = path / "latest.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    created = _timestamp(payload.get("created_at"), "artifact created_at")
    if created > cutoff:
        raise ValueError(f"future artifact: {path}")
    return path, payload


def _timestamp(value: Any, label: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be ISO-8601") from exc
    if parsed.tzinfo is None:
        raise ValueError(f"{label} must be timezone-aware")
    return parsed


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _ref(path: Path) -> dict[str, str]:
    return {"path": str(path), "sha256": _sha256(path)}


def _render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Hypothesis Evidence Gap Review",
        "",
        f"- Gaps: `{payload['summary']['gap_count']}`",
        f"- Statuses: `{payload['summary']['status_counts']}`",
        "- Automatic closure: `false`",
        "- Replay registration: `false`",
        "- Can trade: `false`",
        "",
        "## Gap Reviews",
        "",
    ]
    for item in payload["gap_reviews"]:
        lines.append(
            f"- `{item['resolution_status']}` {item['description']} "
            f"({len(item['supporting_evidence'])} linked records)"
        )
    return "\n".join(lines) + "\n"


__all__ = ["HypothesisEvidenceGapReview"]
