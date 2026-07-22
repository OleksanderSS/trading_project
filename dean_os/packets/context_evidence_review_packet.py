from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from dean_os.analysts.context_adapter import (
    MarketContextEvidenceAdapter,
)
from dean_os.artifact_writer import ReviewArtifactWriter
from dean_os.context_evidence_provenance import (
    CONTEXT_EVIDENCE_CONTRACT,
)
from dean_os.schemas import MarketContext, utc_now_iso
from dean_os.utils import json_ready, sha256_json


class ContextEvidenceReviewPacket:
    """Materialize point-in-time context evidence for human review only."""

    def __init__(
        self,
        *,
        context_json: str | Path,
        domain_id: str,
        output_dir: str | Path = (
            "reports/dean_os/context_evidence_review_current"
        ),
    ) -> None:
        self.context_json = Path(context_json)
        self.domain_id = domain_id
        self.output_dir = Path(output_dir)

    def build(
        self,
        *,
        as_of: str | None = None,
        save: bool = True,
    ) -> dict[str, Any]:
        source_state = _load_context(self.context_json)
        context = source_state.get("context")
        resolved_as_of = as_of or (
            context.as_of if isinstance(context, MarketContext) else None
        )
        if not isinstance(context, MarketContext):
            adaptation = {
                "contract": CONTEXT_EVIDENCE_CONTRACT,
                "status": "blocked_context_unavailable",
                "as_of": resolved_as_of,
                "evidence": [],
                "exclusions": [],
                "summary": {
                    "evidence_count": 0,
                    "excluded_count": 0,
                    "can_influence_pipeline_prediction": False,
                    "can_trade": False,
                },
            }
        elif not resolved_as_of:
            adaptation = {
                "contract": CONTEXT_EVIDENCE_CONTRACT,
                "status": "blocked_context_as_of_missing",
                "as_of": None,
                "evidence": [],
                "exclusions": [
                    {
                        "family": "context",
                        "status": "excluded",
                        "reasons": ["context_as_of_missing"],
                    }
                ],
                "summary": {
                    "evidence_count": 0,
                    "excluded_count": 1,
                    "can_influence_pipeline_prediction": False,
                    "can_trade": False,
                },
            }
        else:
            adaptation = MarketContextEvidenceAdapter(
                self.domain_id
            ).adapt(
                context,
                as_of=resolved_as_of,
            )

        evidence = [
            item.model_dump(mode="json")
            if hasattr(item, "model_dump")
            else item
            for item in adaptation.get("evidence", [])
        ]
        payload: dict[str, Any] = {
            "run_id": _run_id(),
            "created_at": utc_now_iso(),
            "mode": "context_evidence_review_packet",
            "schema_version": CONTEXT_EVIDENCE_CONTRACT,
            "status": adaptation["status"],
            "inputs": {
                "context_json": str(self.context_json),
                "context_sha256": _sha256(self.context_json),
                "domain_id": self.domain_id,
                "as_of": resolved_as_of,
            },
            "summary": adaptation.get("summary", {}),
            "evidence": evidence,
            "exclusions": adaptation.get("exclusions", []),
            "source_errors": source_state.get("errors", []),
            "integration_boundary": {
                "allowed_use": "human_review_context_only",
                "can_satisfy_raw_source_gate": False,
                "can_become_stage5_feature": False,
                "can_influence_consensus": False,
                "can_trade": False,
            },
            "safety": {
                "review_only": True,
                "collectors_run": False,
                "pipeline_run": False,
                "training_run": False,
                "learning_write": False,
                "recommendation_created": False,
                "live_execution_allowed": False,
                "can_trade": False,
            },
            "fingerprint": None,
        }
        payload["fingerprint"] = sha256_json(
            {
                key: value
                for key, value in payload.items()
                if key not in {"created_at", "run_id", "fingerprint"}
            }
        )
        if save:
            payload["saved_paths"] = ReviewArtifactWriter(
                self.output_dir
            ).write(
                payload=payload,
                markdown=render_context_evidence_review_markdown(
                    payload
                ),
                run_id=payload["run_id"],
            )
        return json_ready(payload)


def render_context_evidence_review_markdown(
    payload: dict[str, Any],
) -> str:
    summary = payload.get("summary", {})
    lines = [
        "# DEAN-OS Context Evidence Review",
        "",
        f"- Status: `{payload.get('status')}`",
        f"- Domain: `{payload.get('inputs', {}).get('domain_id')}`",
        f"- As of: `{payload.get('inputs', {}).get('as_of')}`",
        f"- Evidence: `{summary.get('evidence_count', 0)}`",
        f"- Excluded: `{summary.get('excluded_count', 0)}`",
        f"- Can influence pipeline prediction: "
        f"`{summary.get('can_influence_pipeline_prediction', False)}`",
        f"- Can trade: `{summary.get('can_trade', False)}`",
        "",
        "## Exclusions",
        "",
    ]
    for item in payload.get("exclusions", []):
        lines.append(
            f"- `{item.get('family')}`: "
            f"{', '.join(item.get('reasons') or [])}"
        )
    if not payload.get("exclusions"):
        lines.append("- none")
    lines.extend(
        [
            "",
            "This packet is supporting review context only. It cannot "
            "satisfy the raw-source gate, modify Stage5, influence "
            "consensus, or authorize trading.",
        ]
    )
    return "\n".join(lines).strip() + "\n"


def _load_context(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {
            "context": None,
            "errors": [f"context_json_not_found:{path}"],
        }
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        return {
            "context": None,
            "errors": [f"context_json_invalid:{exc!r}"],
        }
    if not isinstance(payload, dict):
        return {
            "context": None,
            "errors": ["context_json_not_object"],
        }
    raw_context = payload.get("context", payload)
    if not isinstance(raw_context, dict):
        return {
            "context": None,
            "errors": ["context_object_missing"],
        }
    try:
        return {"context": MarketContext(**raw_context), "errors": []}
    except Exception as exc:
        return {
            "context": None,
            "errors": [f"context_schema_invalid:{exc!r}"],
        }


def _sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run_id() -> str:
    stamp = utc_now_iso().replace(":", "").replace("+", "Z")
    return f"context_evidence_review_{stamp}"
