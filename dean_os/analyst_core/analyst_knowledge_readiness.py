from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from dean_os.analyst_knowledge.store import LocalKnowledgeStore
from dean_os.artifact_writer import ReviewArtifactWriter
from dean_os.schemas import utc_now_iso
from dean_os.utils import json_ready, sha256_json


class AnalystKnowledgeReadiness:
    """Audit whether stored analyst knowledge is safe for as-of review retrieval."""

    def __init__(
        self,
        *,
        store_dir: str | Path = "data/dean_os/analyst_knowledge",
        output_dir: str | Path = (
            "reports/dean_os/analyst_knowledge_readiness_current"
        ),
    ) -> None:
        self.store_dir = Path(store_dir)
        self.output_dir = Path(output_dir)

    def build(
        self,
        *,
        as_of: str,
        intended_use: str = "evidence",
        save: bool = True,
    ) -> dict[str, Any]:
        store = LocalKnowledgeStore(self.store_dir)
        records = store.audit_point_in_time(
            as_of=as_of,
            intended_use=intended_use,
        )
        eligible_count = sum(record["eligible"] is True for record in records)
        blocked_count = len(records) - eligible_count
        reason_counts = Counter(
            reason
            for record in records
            for reason in record.get("reasons", [])
        )
        manifests = store.list_packs()
        if not records:
            status = "knowledge_store_empty_blocked"
        elif eligible_count and blocked_count:
            status = "knowledge_review_ready_with_exclusions"
        elif eligible_count:
            status = "knowledge_review_ready"
        else:
            status = "knowledge_review_blocked_no_eligible_items"

        payload: dict[str, Any] = {
            "run_id": _run_id(),
            "created_at": utc_now_iso(),
            "mode": "analyst_knowledge_readiness",
            "schema_version": "dean_analyst_knowledge_readiness_v1",
            "status": status,
            "inputs": {
                "store_dir": str(self.store_dir),
                "as_of": as_of,
                "intended_use": intended_use,
            },
            "summary": {
                "pack_count": len(manifests),
                "item_count": len(records),
                "eligible_item_count": eligible_count,
                "blocked_item_count": blocked_count,
                "source_count": sum(
                    int(manifest.get("source_count") or 0)
                    for manifest in manifests.values()
                ),
                "can_feed_review_only_analyst": eligible_count > 0,
                "can_satisfy_raw_source_gate": False,
                "can_influence_pipeline_prediction": False,
                "can_trade": False,
            },
            "pack_manifests": manifests,
            "reason_counts": dict(sorted(reason_counts.items())),
            "records": records,
            "integration_contract": {
                "allowed_path": (
                    "knowledge store -> strict as-of retrieval -> analyst "
                    "review evidence -> manual domain/ticker review"
                ),
                "blocked_shortcuts": [
                    "knowledge_item_to_raw_source_gate",
                    "knowledge_item_to_stage5_feature_or_prediction",
                    "sector_knowledge_to_direct_ticker_evidence",
                    "knowledge_item_to_consensus_weight",
                    "knowledge_item_to_trade",
                ],
                "pipeline_join_requirement": (
                    "Any later specialist join must independently match "
                    "ticker, timeframe, prediction as_of, manual review, "
                    "and exact-context eligibility."
                ),
            },
            "safety": {
                "review_only": True,
                "live_execution_allowed": False,
                "broker_access_performed": False,
                "training_performed": False,
                "learning_write_performed": False,
                "production_config_write_performed": False,
                "network_access_performed": False,
            },
            "fingerprint": None,
        }
        payload["fingerprint"] = sha256_json(
            {
                "schema_version": payload["schema_version"],
                "status": status,
                "inputs": payload["inputs"],
                "summary": payload["summary"],
                "pack_manifests": manifests,
                "reason_counts": payload["reason_counts"],
                "records": records,
                "integration_contract": payload["integration_contract"],
            }
        )
        if save:
            payload["saved_paths"] = ReviewArtifactWriter(
                self.output_dir
            ).write(
                payload=payload,
                markdown=render_analyst_knowledge_readiness_markdown(
                    payload
                ),
                run_id=payload["run_id"],
            )
        return json_ready(payload)


def render_analyst_knowledge_readiness_markdown(
    payload: dict[str, Any],
) -> str:
    summary = payload.get("summary", {})
    lines = [
        "# DEAN-OS Analyst Knowledge Readiness",
        "",
        f"- Status: `{payload.get('status')}`",
        f"- As of: `{payload.get('inputs', {}).get('as_of')}`",
        f"- Packs: `{summary.get('pack_count', 0)}`",
        f"- Items: `{summary.get('item_count', 0)}`",
        f"- Eligible: `{summary.get('eligible_item_count', 0)}`",
        f"- Blocked: `{summary.get('blocked_item_count', 0)}`",
        f"- Can feed review-only analyst: "
        f"`{summary.get('can_feed_review_only_analyst', False)}`",
        f"- Can influence pipeline prediction: "
        f"`{summary.get('can_influence_pipeline_prediction', False)}`",
        f"- Can trade: `{summary.get('can_trade', False)}`",
        "",
        "## Block Reasons",
        "",
    ]
    for reason, count in payload.get("reason_counts", {}).items():
        lines.append(f"- `{reason}`: {count}")
    if not payload.get("reason_counts"):
        lines.append("- none")
    lines.extend(
        [
            "",
            "## Integration Boundary",
            "",
            f"- Allowed: "
            f"{payload.get('integration_contract', {}).get('allowed_path')}",
        ]
    )
    for shortcut in payload.get("integration_contract", {}).get(
        "blocked_shortcuts", []
    ):
        lines.append(f"- Blocked shortcut: `{shortcut}`")
    return "\n".join(lines).strip() + "\n"


def _run_id() -> str:
    stamp = utc_now_iso().replace(":", "").replace("+", "Z")
    return f"analyst_knowledge_readiness_{stamp}"
