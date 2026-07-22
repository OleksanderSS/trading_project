from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from dean_os.analysts.schemas import AnalystEvidenceItem
from dean_os.utils import sha256_json

PIPELINE_CONTEXT_CONTRACT = "dean_world_model_pipeline_context_v1"
PIPELINE_CONTEXT_MODE = "world_model_pipeline_context_discovery"


class PipelineContextEvidenceLoader:
    """Load hash-verified exact pipeline lanes as review-only analyst context."""

    def load(
        self,
        artifact_path: str | Path,
        *,
        domain_id: str,
        as_of: str,
        tickers: list[str] | None = None,
    ) -> list[AnalystEvidenceItem]:
        path = _latest_path(Path(artifact_path))
        payload = _load_json(path)
        _validate_bundle(payload, path=path, as_of=as_of)
        bundle_sha256 = _sha256(path)
        requested_tickers = sorted(
            {
                str(ticker).strip().upper()
                for ticker in (tickers or [])
                if str(ticker).strip()
            }
        )

        evidence: list[AnalystEvidenceItem] = []
        for lane in payload.get("timeframe_lanes", []):
            if lane.get("status") != "pipeline_lane_exact_context_available":
                continue
            timeframe = str(lane.get("timeframe") or "").strip()
            if not timeframe:
                continue
            lane_tickers = sorted(
                set(requested_tickers)
                & {
                    str(ticker).strip().upper()
                    for ticker in lane.get("tickers", [])
                    if str(ticker).strip()
                }
            ) or requested_tickers
            stage23_ref = (lane.get("artifacts") or {}).get(
                "stage23_regeneration"
            ) or {}
            _validate_artifact_ref(stage23_ref, parent_path=path)
            stage4_refs = list(
                (lane.get("artifacts") or {}).get(
                    "stage4_exact_context"
                )
                or []
            )
            stage4_payloads = [
                _load_verified_ref(ref, parent_path=path)
                for ref in stage4_refs
            ]
            contract_passed = any(
                (item.get("summary") or {}).get("contract_passed") is True
                for item in stage4_payloads
            )
            shard_count = int(lane.get("stage3_shard_count") or 0)
            evidence_id = "pipeline_context_" + sha256_json(
                {
                    "bundle_sha256": bundle_sha256,
                    "timeframe": timeframe,
                }
            )[:24]
            lane_sha256 = sha256_json(
                {
                    "bundle_sha256": bundle_sha256,
                    "timeframe": timeframe,
                    "stage23_sha256": stage23_ref.get("sha256"),
                    "stage4_sha256": [ref.get("sha256") for ref in stage4_refs],
                }
            )
            evidence.append(
                AnalystEvidenceItem(
                    evidence_id=evidence_id,
                    source_type="market",
                    source=str(path),
                    published_at=str(payload["created_at"]),
                    as_of=as_of,
                    domain_id=domain_id,
                    tickers=lane_tickers,
                    sectors=[domain_id],
                    evidence_type="market_confirmation",
                    summary=(
                        f"Verified {timeframe} pipeline context is available "
                        f"with {shard_count} source-bound Stage3 shards; "
                        f"Stage4 validation contract passed={str(contract_passed).lower()}."
                    ),
                    stance_hint="neutral",
                    strength=0.60 if contract_passed else 0.35,
                    freshness_score=0.85,
                    directness="market",
                    reliability_score=0.75,
                    limitations=[
                        "Review-only pipeline context; not a trading signal.",
                        *(
                            []
                            if contract_passed
                            else [
                                "No linked Stage4 candidate passed its validation contract."
                            ]
                        ),
                    ],
                    provenance={
                        "producer_contract": PIPELINE_CONTEXT_CONTRACT,
                        "source_sha256": bundle_sha256,
                        "canonical_record_sha256": lane_sha256,
                        "timeframe": timeframe,
                        "stage3_shard_count": shard_count,
                        "stage3_cache_status": lane.get(
                            "stage3_cache_status"
                        ),
                        "stage4_exact_context_count": len(stage4_payloads),
                        "stage4_validation_contract_passed": contract_passed,
                        "required_lane_eligible": contract_passed,
                        "ticker_thesis_eligible": False,
                    },
                    point_in_time={
                        "status": "point_in_time_compatible",
                        "available_at": payload["created_at"],
                        "as_of": as_of,
                        "artifact_sha256": bundle_sha256,
                    },
                )
            )
        if not evidence:
            raise ValueError(
                "Pipeline context artifact has no exact, verified timeframe lanes"
            )
        return evidence


def _validate_bundle(
    payload: dict[str, Any],
    *,
    path: Path,
    as_of: str,
) -> None:
    if payload.get("contract") != PIPELINE_CONTEXT_CONTRACT:
        raise ValueError("Unexpected pipeline context contract")
    if payload.get("mode") != PIPELINE_CONTEXT_MODE:
        raise ValueError("Unexpected pipeline context mode")
    if (payload.get("summary") or {}).get("status") != "pipeline_context_bundle_ready":
        raise ValueError("Pipeline context bundle is not ready")
    safety = payload.get("safety") or {}
    if safety.get("review_only") is not True or safety.get("can_trade") is not False:
        raise ValueError("Pipeline context is not review-only")
    if safety.get("learning_memory_write_performed") is True:
        raise ValueError("Pipeline context performed a learning-memory write")
    created_at = _timestamp(payload.get("created_at"), "created_at")
    cutoff = _timestamp(as_of, "as_of")
    if created_at > cutoff:
        raise ValueError(
            f"Pipeline context is future evidence: {created_at.isoformat()} > "
            f"{cutoff.isoformat()}"
        )
    if not path.is_file():
        raise FileNotFoundError(path)


def _validate_artifact_ref(ref: dict[str, Any], *, parent_path: Path) -> None:
    _load_verified_ref(ref, parent_path=parent_path)


def _load_verified_ref(
    ref: dict[str, Any],
    *,
    parent_path: Path,
) -> dict[str, Any]:
    raw_path = str(ref.get("path") or "").strip()
    expected = str(ref.get("sha256") or "").strip().lower()
    if not raw_path or len(expected) != 64:
        raise ValueError("Linked pipeline artifact requires path and sha256")
    path = _resolve_linked_path(raw_path, parent_path)
    if not path.is_file():
        raise FileNotFoundError(path)
    actual = _sha256(path)
    if actual != expected:
        raise ValueError(
            f"Linked pipeline artifact hash mismatch: {path}"
        )
    return _load_json(path)


def _resolve_linked_path(raw_path: str, parent_path: Path) -> Path:
    candidate = Path(raw_path)
    if candidate.is_absolute():
        return candidate
    for root in (Path.cwd(), *parent_path.parents):
        resolved = root / candidate
        if resolved.exists():
            return resolved
    return Path.cwd() / candidate


def _latest_path(path: Path) -> Path:
    return path if path.is_file() else path / "latest.json"


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Artifact must contain a JSON object: {path}")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _timestamp(value: Any, label: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be ISO-8601") from exc
    if parsed.tzinfo is None:
        raise ValueError(f"{label} must be timezone-aware")
    return parsed


__all__ = [
    "PIPELINE_CONTEXT_CONTRACT",
    "PIPELINE_CONTEXT_MODE",
    "PipelineContextEvidenceLoader",
]
