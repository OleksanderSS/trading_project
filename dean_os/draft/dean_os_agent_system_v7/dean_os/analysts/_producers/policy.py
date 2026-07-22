from __future__ import annotations

__all__ = [
    'CONTRACT',
    'DEFAULT_REGISTRY',
    'SNAPSHOT_CONTRACT',
    'SavedOfficialPolicyEvidenceProducer',
    'load_verified_official_policy_context_fragment',
    'render_policy_markdown',
]

import hashlib
import json
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import yaml

from dean_os.analysts._producers.news import (
    load_verified_semiconductor_news_context_fragment,
)
from dean_os.analysts.profiles import get_domain_profile
from dean_os.draft.dean_os_agent_system_v7.dean_os.artifact_writer import ReviewArtifactWriter
from dean_os.draft.dean_os_agent_system_v7.dean_os.context_evidence_provenance import (
    audit_news_records,
    parse_timezone_aware,
)
from dean_os.schemas import utc_now_iso

CONTRACT = "dean_saved_official_policy_evidence_producer_v1"
SNAPSHOT_CONTRACT = "dean_official_policy_source_snapshot_v1"
_DOMAIN_ID = "semiconductor_ai_infrastructure"
_DEFAULT_TICKERS = list(get_domain_profile(_DOMAIN_ID).ticker_universe_hint)
DEFAULT_REGISTRY = (
    Path(__file__).parent
    / "config"
    / "official_policy_evidence_registry.yaml"
)


class SavedOfficialPolicyEvidenceProducer:
    """Bind one official policy source to independent news corroboration."""

    def __init__(
        self,
        output_dir: str | Path = (
            "reports/dean_os/saved_official_policy_evidence_producer"
        ),
    ):
        self.output_dir = Path(output_dir)

    def build(
        self,
        *,
        snapshot_artifact_path: str | Path,
        corroborating_news_artifact_path: str | Path,
        as_of: str,
        registry_path: str | Path = DEFAULT_REGISTRY,
        save: bool = True,
    ) -> dict[str, Any]:
        as_of_dt = parse_timezone_aware(as_of)
        if as_of_dt is None:
            raise ValueError("official policy as_of must be timezone-aware")
        snapshot_path = Path(snapshot_artifact_path)
        news_path = Path(corroborating_news_artifact_path)
        registry_source = Path(registry_path)
        snapshot = json.loads(
            snapshot_path.read_text(encoding="utf-8")
        )
        if (
            snapshot.get("snapshot_contract") != SNAPSHOT_CONTRACT
            or snapshot.get("status")
            != "official_policy_snapshot_ready"
        ):
            raise ValueError("official policy snapshot is not ready")
        source = snapshot.get("source", {})
        raw_path = Path(str(source.get("immutable_path") or ""))
        source_sha = str(source.get("sha256") or "")
        if (
            not raw_path.exists()
            or _sha256_file(raw_path) != source_sha
            or not raw_path.read_bytes().startswith(b"%PDF")
        ):
            raise ValueError("official policy raw source hash mismatch")
        host = (
            urlsplit(str(source.get("final_url") or "")).hostname or ""
        ).lower()
        if host not in {"bis.gov", "www.bis.gov", "media.bis.gov"}:
            raise ValueError("official policy source host is not BIS")
        registry = yaml.safe_load(
            registry_source.read_text(encoding="utf-8")
        )
        document = registry.get("documents", {}).get(source_sha)
        if not isinstance(document, dict):
            raise ValueError("official policy source is not registered")
        published_at = parse_timezone_aware(
            document.get("published_at")
        )
        if published_at is None or published_at > as_of_dt:
            raise ValueError("official policy publication is after as_of")
        age_days = (as_of_dt - published_at).total_seconds() / 86400
        if age_days > 120:
            raise ValueError("official policy source is stale")
        if (
            document.get("source_url") != source.get("final_url")
            or document.get("source_identity")
            != source.get("source_identity")
            or document.get("source_tier") != source.get("source_tier")
            or document.get("evidence_type")
            != "policy_or_geopolitical"
        ):
            raise ValueError("official policy registry/source mismatch")

        news = load_verified_semiconductor_news_context_fragment(
            news_path,
            expected_as_of=as_of_dt.isoformat(),
        )
        news_payload = json.loads(news_path.read_text(encoding="utf-8"))
        policy_lane = next(
            (
                lane
                for lane in news_payload.get("lane_review", [])
                if lane.get("evidence_type")
                == "policy_or_geopolitical"
            ),
            None,
        )
        if not policy_lane:
            raise ValueError("policy corroboration lane is missing")
        corroborating = set(
            policy_lane.get("independent_strong_sources", [])
        )
        combined = sorted(
            corroborating | {str(document["source_identity"])}
        )
        eligible = len(combined) >= int(
            policy_lane.get("minimum_independent_sources", 2)
        )
        if not eligible:
            raise ValueError(
                "official policy evidence lacks independent corroboration"
            )

        record = {
            "title": document["title"],
            "summary": document["semantic_claim"],
            "source": "U.S. Bureau of Industry and Security",
            "published_at": published_at.isoformat(),
            "url": document["source_url"],
            "_dean_semantic_evidence": {
                "producer_contract": CONTRACT,
                "evidence_type": "policy_or_geopolitical",
                "required_lane_eligible": True,
                "source_tier": document["source_tier"],
                "source_identity": document["source_identity"],
                "matched_terms": document["matched_terms"],
                "candidate_sha256": _canonical_sha256(
                    {
                        "source_sha256": source_sha,
                        "news_fingerprint": news["metadata"].get(
                            "saved_semiconductor_news_fingerprint"
                        ),
                        "combined_sources": combined,
                    }
                ),
                "stance_hint": "unknown",
            },
        }
        audit = audit_news_records(
            [record],
            as_of=as_of_dt.isoformat(),
            requested_tickers=_DEFAULT_TICKERS,
        )
        accepted = [
            {
                key: value
                for key, value in item.items()
                if key != "_dean_context_provenance"
            }
            for item in audit["accepted"]
        ]
        run_id = _run_id()
        payload = {
            "run_id": run_id,
            "created_at": utc_now_iso(),
            "producer_contract": CONTRACT,
            "status": "official_policy_evidence_ready",
            "inputs": {
                "snapshot_artifact_path": str(snapshot_path),
                "snapshot_artifact_sha256": _sha256_file(snapshot_path),
                "corroborating_news_artifact_path": str(news_path),
                "corroborating_news_artifact_sha256": _sha256_file(
                    news_path
                ),
                "registry_path": str(registry_source),
                "registry_sha256": _sha256_file(registry_source),
                "as_of": as_of_dt.isoformat(),
            },
            "source_provenance": {
                **source,
                "published_at": published_at.isoformat(),
                "age_days": age_days,
                "registry_review_status": registry.get(
                    "review_status"
                ),
            },
            "corroboration": {
                "existing_independent_strong_sources": sorted(
                    corroborating
                ),
                "official_source_identity": document[
                    "source_identity"
                ],
                "combined_independent_sources": combined,
                "combined_independent_source_count": len(combined),
                "minimum_independent_sources": int(
                    policy_lane.get("minimum_independent_sources", 2)
                ),
                "policy_lane_eligible": True,
            },
            "summary": {
                "accepted_policy_record_count": len(accepted),
                "accepted_fingerprint": _canonical_sha256(accepted),
                "policy_lane_ready": True,
                "can_enter_market_context_review": True,
                "can_trade": False,
            },
            "market_context_fragment": {
                "as_of": as_of_dt.isoformat(),
                "news": accepted,
                "metadata": {
                    "saved_official_policy_run_id": run_id,
                    "source_sha256": source_sha,
                    "combined_independent_sources": combined,
                    "ready_required_lanes": [
                        "policy_or_geopolitical"
                    ],
                },
            },
            "integration_boundary": {
                "review_only": True,
                "official_source_hash_bound": True,
                "independent_corroboration_required": True,
                "plain_text_ticker_promotion_allowed": False,
                "automatic_prediction_influence": False,
                "automatic_trading_allowed": False,
            },
            "safety": _safety(),
        }
        if save:
            payload["saved_paths"] = ReviewArtifactWriter(
                self.output_dir
            ).write(
                payload=payload,
                markdown=render_policy_markdown(payload),
                run_id=run_id,
            )
        return payload


def load_verified_official_policy_context_fragment(
    artifact_path: str | Path,
    *,
    expected_as_of: str | None = None,
) -> dict[str, Any]:
    path = Path(artifact_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        payload.get("producer_contract") != CONTRACT
        or payload.get("status") != "official_policy_evidence_ready"
    ):
        raise ValueError("official policy evidence artifact is not ready")
    inputs = payload.get("inputs", {})
    as_of = parse_timezone_aware(inputs.get("as_of"))
    if as_of is None:
        raise ValueError("official policy artifact as_of invalid")
    if expected_as_of is not None:
        expected = parse_timezone_aware(expected_as_of)
        if expected is None or expected != as_of:
            raise ValueError("official policy expected as_of mismatch")
    for path_key, sha_key in (
        ("snapshot_artifact_path", "snapshot_artifact_sha256"),
        (
            "corroborating_news_artifact_path",
            "corroborating_news_artifact_sha256",
        ),
        ("registry_path", "registry_sha256"),
    ):
        source = Path(str(inputs.get(path_key) or ""))
        if (
            not source.exists()
            or _sha256_file(source) != inputs.get(sha_key)
        ):
            raise ValueError("official policy source artifact hash mismatch")
    news_path = Path(inputs["corroborating_news_artifact_path"])
    load_verified_semiconductor_news_context_fragment(
        news_path,
        expected_as_of=as_of.isoformat(),
    )
    snapshot = json.loads(
        Path(inputs["snapshot_artifact_path"]).read_text(
            encoding="utf-8"
        )
    )
    raw = Path(snapshot["source"]["immutable_path"])
    if (
        not raw.exists()
        or _sha256_file(raw) != snapshot["source"]["sha256"]
    ):
        raise ValueError("official policy raw source hash mismatch")
    records = payload.get("market_context_fragment", {}).get("news")
    audit = audit_news_records(
        records,
        as_of=as_of.isoformat(),
        requested_tickers=_DEFAULT_TICKERS,
    )
    stripped = [
        {
            key: value
            for key, value in item.items()
            if key != "_dean_context_provenance"
        }
        for item in audit["accepted"]
    ]
    if (
        audit["excluded_count"] != 0
        or _canonical_sha256(stripped)
        != payload.get("summary", {}).get("accepted_fingerprint")
    ):
        raise ValueError("official policy evidence fingerprint mismatch")
    return {
        "as_of": as_of.isoformat(),
        "news": stripped,
        "metadata": {
            **payload.get("market_context_fragment", {}).get(
                "metadata", {}
            ),
            "saved_official_policy_verified": True,
            "artifact_path": str(path),
            "artifact_sha256": _sha256_file(path),
        },
    }


def _safety() -> dict[str, Any]:
    return {
        "review_only": True,
        "network_access_performed": False,
        "pipeline_run_performed": False,
        "training_run_performed": False,
        "learning_write_performed": False,
        "live_execution_performed": False,
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()


def _run_id() -> str:
    return (
        "saved_official_policy_evidence_"
        + utc_now_iso().replace(":", "").replace("+", "Z")
    )


def render_policy_markdown(payload: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Saved Official Policy Evidence",
            "",
            f"- Status: `{payload.get('status')}`",
            (
                "- Combined sources: `"
                + ", ".join(
                    payload.get("corroboration", {}).get(
                        "combined_independent_sources", []
                    )
                )
                + "`"
            ),
            "- Policy lane ready: `true`",
            "",
            "The source is official BIS guidance, hash-bound and "
            "independently corroborated. Output remains review-only.",
            "",
        ]
    )
