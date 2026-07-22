from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from dean_os.artifact_writer import ReviewArtifactWriter
from dean_os.context_evidence_provenance import parse_timezone_aware
from dean_os.schemas import utc_now_iso


CONTRACT = "dean_filing_order_evidence_v1"
RPO_CONCEPT = "RevenueRemainingPerformanceObligation"


class FilingOrderEvidenceBuilder:
    """Extract issuer RPO as a partial backlog proxy, never as full order backlog."""

    def __init__(
        self,
        output_dir: str | Path = "reports/dean_os/filing_order_evidence_current",
    ) -> None:
        self.output_dir = Path(output_dir)

    def build(
        self,
        companyfacts_paths: dict[str, str | Path],
        *,
        as_of: str,
        max_age_days: int = 730,
        save: bool = True,
    ) -> dict[str, Any]:
        cutoff = parse_timezone_aware(as_of)
        if cutoff is None:
            raise ValueError("as_of must be timezone-aware")
        if max_age_days < 0:
            raise ValueError("max_age_days must be non-negative")
        observations = []
        exclusions = []
        source_refs = []
        for ticker, raw_path in sorted(companyfacts_paths.items()):
            path = Path(raw_path)
            payload = _load(path)
            source_refs.append({"ticker": ticker.upper(), "path": str(path), "sha256": _sha256(path)})
            fact = (((payload.get("facts") or {}).get("us-gaap") or {}).get(RPO_CONCEPT))
            if not isinstance(fact, dict):
                exclusions.append({"ticker": ticker.upper(), "reason": "rpo_concept_absent"})
                continue
            candidates = []
            for unit, rows in (fact.get("units") or {}).items():
                for row in rows if isinstance(rows, list) else []:
                    filed_at = _filed_at(row.get("filed"))
                    if filed_at is None or filed_at > cutoff:
                        continue
                    if not isinstance(row.get("val"), (int, float)) or isinstance(row.get("val"), bool):
                        continue
                    candidates.append((filed_at, str(row.get("end") or ""), unit, row))
            if not candidates:
                exclusions.append({"ticker": ticker.upper(), "reason": "no_point_in_time_rpo_observation"})
                continue
            filed_at, period, unit, row = max(candidates, key=lambda item: (item[0], item[1]))
            source_locator = f"{path}#us-gaap:{RPO_CONCEPT}:{row.get('accn')}"
            observation = {
                "ticker": ticker.upper(),
                "entity": payload.get("entityName"),
                "metric_name": "revenue_remaining_performance_obligation",
                "concept": RPO_CONCEPT,
                "value": float(row["val"]),
                "unit": unit,
                "period": period,
                "available_at": filed_at.isoformat(),
                "age_days_at_as_of": (cutoff - filed_at).days,
                "max_age_days": max_age_days,
                "current_gap_support_eligible": (cutoff - filed_at).days <= max_age_days,
                "accession_number": row.get("accn"),
                "form": row.get("form"),
                "fiscal_year": row.get("fy"),
                "fiscal_period": row.get("fp"),
                "source_locator": source_locator,
                "source_sha256": _sha256(path),
                "semantic_role": "contracted_revenue_proxy_not_full_order_backlog",
                "gap_support_role": "partial_support_only",
                "automatic_gap_closure_allowed": False,
                "limitations": [
                    "Remaining performance obligation is contracted revenue not yet recognized, not full customer order backlog.",
                    "It does not measure cancellable demand, supplier equipment orders, production capacity, or utilization.",
                    "Issuer definitions and disclosure exemptions can reduce cross-company comparability.",
                    *(
                        ["Observation is too old for current-gap support and is historical context only."]
                        if (cutoff - filed_at).days > max_age_days else []
                    ),
                ],
            }
            observation["observation_sha256"] = _canonical_sha256(observation)
            observations.append(observation)

        created_at = utc_now_iso()
        run_id = "filing_order_evidence_" + created_at.replace(":", "").replace("+00:00", "Z")
        payload: dict[str, Any] = {
            "run_id": run_id,
            "created_at": created_at,
            "mode": "filing_order_evidence_review_only",
            "contract": CONTRACT,
            "inputs": {"as_of": cutoff.isoformat(), "sources": source_refs},
            "summary": {
                "source_count": len(source_refs),
                "observation_count": len(observations),
                "excluded_source_count": len(exclusions),
                "full_backlog_observation_count": 0,
                "partial_proxy_count": len(observations),
                "current_gap_support_eligible_count": sum(
                    bool(row["current_gap_support_eligible"]) for row in observations
                ),
                "historical_context_only_count": sum(
                    not bool(row["current_gap_support_eligible"]) for row in observations
                ),
                "automatic_gap_closure_allowed": False,
                "can_trade": False,
            },
            "observations": observations,
            "exclusions": exclusions,
            "semantic_boundary": {
                "rpo_equals_full_order_backlog": False,
                "purchase_obligations_are_customer_backlog": False,
                "narrative_mentions_are_numeric_evidence": False,
                "manual_review_required": True,
            },
            "safety": {
                "review_only": True,
                "automatic_gap_closure_performed": False,
                "collector_execution_performed": False,
                "learning_write_performed": False,
                "can_trade": False,
            },
        }
        if save:
            payload["saved_paths"] = ReviewArtifactWriter(self.output_dir).write(
                payload=payload, markdown=_markdown(payload), run_id=run_id
            )
        return payload


def _filed_at(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value)).replace(tzinfo=UTC)
    except ValueError:
        return None


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"artifact must be an object: {path}")
    return payload


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _markdown(payload: dict[str, Any]) -> str:
    return (
        "# Filing Order Evidence\n\n"
        f"- RPO observations: `{payload['summary']['observation_count']}`\n"
        "- Full backlog observations: `0`\n"
        "- Automatic gap closure: `false`\n"
        "- RPO is a partial contracted-revenue proxy, not full order backlog.\n"
    )


__all__ = ["CONTRACT", "FilingOrderEvidenceBuilder"]
