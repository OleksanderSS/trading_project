from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd

from dean_os.artifact_writer import ReviewArtifactWriter
from dean_os.context_evidence_provenance import parse_timezone_aware
from dean_os.replays.replay_outcome_lifecycle_orchestrator import (
    ReplayOutcomeLifecycleOrchestrator,
)
from dean_os.schemas import utc_now_iso
from dean_os.verified_market_source_router import validate_local_market_snapshot


class VerifiedLocalSnapshotIngestion:
    """Validate and atomically ingest one operator-supplied market snapshot."""

    contract = "dean_verified_local_market_snapshot_ingestion_v1"

    def __init__(
        self,
        output_dir: str | Path = (
            "reports/dean_os/verified_local_snapshot_ingestion_current"
        ),
        artifact_dir: str | Path = (
            "data/dean_os/historical_outcome_market_snapshots"
        ),
    ) -> None:
        self.output_dir = Path(output_dir)
        self.artifact_dir = Path(artifact_dir)

    def build(
        self,
        *,
        source_router_json: str | Path,
        candidate_path: str | Path | None,
        registration_json: str | Path,
        review_gate_json: str | Path,
        as_of: str,
        pipeline_paths: list[str | Path],
        prior_outcome_json_paths: list[str | Path],
        packet_json: str | Path | None = None,
        journal_path: str | Path = "data/dean_os/system_journal.jsonl",
        apply_ingestion: bool = False,
        save: bool = True,
    ) -> dict[str, Any]:
        router_path = Path(source_router_json)
        router = _load(router_path)
        if router.get("contract") != "dean_verified_market_source_router_v1":
            raise ValueError("unsupported verified source router contract")
        cutoff = parse_timezone_aware(as_of)
        if cutoff is None:
            raise ValueError("ingestion as_of must be timezone-aware")
        candidate = Path(candidate_path) if candidate_path is not None else None
        route = _local_route(router)
        validation: dict[str, Any] | None = None
        snapshot_path: Path | None = None
        lifecycle: dict[str, Any] | None = None

        if route is None:
            status = "no_local_snapshot_route"
        elif candidate is None:
            status = "awaiting_candidate"
        else:
            validation = validate_local_market_snapshot(
                candidate,
                required_tickers=list(route.get("required_tickers") or []),
                due_at=str(route.get("due_at")),
                as_of=cutoff,
            )
            if not validation.get("valid"):
                status = "candidate_rejected"
            elif not apply_ingestion:
                status = "candidate_valid_ready_for_ingestion"
            else:
                snapshot_path = self._ingest(candidate, validation)
                lifecycle = ReplayOutcomeLifecycleOrchestrator().build(
                    registration_json=registration_json,
                    review_gate_json=review_gate_json,
                    packet_json=packet_json,
                    as_of=as_of,
                    verified_price_paths=[snapshot_path],
                    pipeline_paths=pipeline_paths,
                    prior_outcome_json_paths=prior_outcome_json_paths,
                    journal_path=journal_path,
                    save=save,
                )
                status = "snapshot_ingested_lifecycle_completed"

        created_at = utc_now_iso()
        run_id = "verified_local_snapshot_ingestion_" + created_at.replace(
            ":", ""
        ).replace("+00:00", "Z")
        payload: dict[str, Any] = {
            "run_id": run_id,
            "created_at": created_at,
            "mode": "verified_local_market_snapshot_ingestion",
            "contract": self.contract,
            "inputs": {
                "source_router_json": str(router_path),
                "candidate_path": str(candidate) if candidate is not None else None,
                "registration_json": str(registration_json),
                "review_gate_json": str(review_gate_json),
                "as_of": cutoff.isoformat(),
                "apply_ingestion": apply_ingestion,
            },
            "summary": {
                "status": status,
                "candidate_present": candidate is not None,
                "candidate_valid": bool(validation and validation.get("valid")),
                "snapshot_ingested": snapshot_path is not None,
                "lifecycle_rerun": lifecycle is not None,
                "post_ingestion_lifecycle_status": (
                    ((lifecycle or {}).get("summary") or {}).get("status")
                ),
                "automatic_source_polling_allowed": False,
                "can_trade": False,
            },
            "selected_route": route,
            "validation": validation,
            "snapshot": {
                "path": str(snapshot_path),
                "sha256": _sha256(snapshot_path),
                "source_candidate_sha256": validation.get("sha256")
                if validation and snapshot_path is not None
                else None,
                "format": "parquet",
            }
            if snapshot_path is not None
            else None,
            "lifecycle": _artifact_summary(lifecycle),
            "safety": {
                "atomic_write": snapshot_path is not None,
                "source_candidate_mutated": False,
                "network_access_performed": False,
                "legacy_database_write_performed": False,
                "outcome_scoring_performed": False,
                "learning_memory_write_performed": False,
                "production_rule_update_performed": False,
                "broker_access_performed": False,
                "can_trade": False,
            },
        }
        if save:
            payload["saved_paths"] = ReviewArtifactWriter(self.output_dir).write(
                payload=payload,
                markdown=_markdown(payload),
                run_id=run_id,
            )
        return payload

    def _ingest(self, candidate: Path, validation: dict[str, Any]) -> Path:
        frame = _read_table(candidate).copy()
        frame["datetime"] = pd.to_datetime(frame["datetime"], utc=True, errors="raise")
        frame["ticker"] = frame["ticker"].astype(str).str.upper()
        frame["close"] = pd.to_numeric(frame["close"], errors="raise")
        if "interval" not in frame:
            frame["interval"] = "1d"
        frame["interval"] = "1d"
        if "source_interval" not in frame:
            frame["source_interval"] = "1d"
        frame["source_provider"] = "local_validated_snapshot"
        frame["source_candidate_sha256"] = str(validation["sha256"])
        frame["hash"] = frame.apply(
            lambda row: hashlib.sha256(
                (
                    row["datetime"].isoformat()
                    + str(row["ticker"])
                    + str(row["interval"])
                    + str(row["close"])
                    + str(validation["sha256"])
                ).encode("utf-8")
            ).hexdigest(),
            axis=1,
        )
        frame = frame.sort_values(["ticker", "datetime"]).reset_index(drop=True)
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        timestamp = utc_now_iso().replace(":", "").replace("+00:00", "Z")
        output = self.artifact_dir / f"verified_local_market_{timestamp}.parquet"
        fd, tmp_name = tempfile.mkstemp(
            prefix=".verified_local.", suffix=".parquet.tmp", dir=str(self.artifact_dir)
        )
        os.close(fd)
        tmp_path = Path(tmp_name)
        try:
            frame.to_parquet(tmp_path, index=False)
            os.replace(tmp_path, output)
        finally:
            if tmp_path.exists():
                tmp_path.unlink()
        return output


def _local_route(router: dict[str, Any]) -> dict[str, Any] | None:
    for item in router.get("routes") or []:
        selected = item.get("selected_provider") or {}
        if selected.get("provider_id") == "local_validated_snapshot":
            return item
    return None


def _read_table(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.suffix.lower() == ".csv" else pd.read_parquet(path)


def _artifact_summary(payload: dict[str, Any] | None) -> dict[str, Any] | None:
    if payload is None:
        return None
    return {
        "run_id": payload.get("run_id"),
        "contract": payload.get("contract"),
        "summary": payload.get("summary"),
        "saved_paths": payload.get("saved_paths"),
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"artifact must be an object: {path}")
    return payload


def _markdown(payload: dict[str, Any]) -> str:
    summary = payload["summary"]
    lines = [
        "# Verified Local Market Snapshot Ingestion",
        "",
        f"- Status: `{summary['status']}`",
        f"- Candidate present: `{summary['candidate_present']}`",
        f"- Candidate valid: `{summary['candidate_valid']}`",
        f"- Snapshot ingested: `{summary['snapshot_ingested']}`",
        f"- Lifecycle rerun: `{summary['lifecycle_rerun']}`",
        f"- Post-ingestion lifecycle: `{summary['post_ingestion_lifecycle_status']}`",
        "",
        "The source file is never mutated. No network, legacy database, learning-rule, or trading action is allowed.",
    ]
    return "\n".join(lines).strip() + "\n"


__all__ = ["VerifiedLocalSnapshotIngestion"]
