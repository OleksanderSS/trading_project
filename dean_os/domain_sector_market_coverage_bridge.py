from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from dean_os.analyst_core.domain_analyst_lifecycle_profile import (
    DomainAnalystLifecycleProfileCompiler,
)
from dean_os.artifact_writer import ReviewArtifactWriter
from dean_os.clean_yahoo_market_snapshot import (
    load_verified_clean_yahoo_market_snapshot,
)
from dean_os.pipeline_control.pipeline_control_saved_data_coverage import (
    PipelineControlSavedDataCoverage,
)
from dean_os.schemas import utc_now_iso
from dean_os.utils import json_ready


CONTRACT = "dean_domain_sector_market_coverage_bridge_v1"
DEFAULT_OUTPUT_DIR = (
    "reports/dean_os/domain_sector_market_coverage_bridge_current"
)


class DomainSectorMarketCoverageBridge:
    """Bind one verified clean snapshot to one domain's exact market scope."""

    def __init__(self, output_dir: str | Path = DEFAULT_OUTPUT_DIR):
        self.output_dir = Path(output_dir)

    def build(
        self,
        *,
        domain_id: str,
        analysis_cutoff: str,
        snapshot_manifest_path: str | Path,
        min_rows: int = 180,
        max_rows: int = 600,
        max_abs_return: float = 0.25,
        min_cadence_ratio: float = 0.75,
        save: bool = True,
    ) -> dict[str, Any]:
        cutoff = _aware(analysis_cutoff)
        lifecycle = DomainAnalystLifecycleProfileCompiler().compile(domain_id)
        measurement = (lifecycle.get("domain_overlay") or {}).get(
            "market_measurement"
        ) or {}
        universe = _tickers(measurement.get("primary_universe") or [])
        benchmark = str(measurement.get("benchmark_ticker") or "").strip().upper()
        required_tickers = sorted(set(universe + ([benchmark] if benchmark else [])))
        blockers: list[str] = []
        if lifecycle.get("readiness", {}).get("schema_valid") is not True:
            blockers.append("domain_lifecycle_profile_invalid")
        if not universe:
            blockers.append("domain_sector_market_universe_missing")
        if not benchmark:
            blockers.append("domain_sector_market_benchmark_missing")

        verified: dict[str, Any] = {}
        verification_error: str | None = None
        try:
            verified = load_verified_clean_yahoo_market_snapshot(
                snapshot_manifest_path
            )
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            verification_error = f"{type(exc).__name__}: {exc}"
            blockers.append("clean_market_snapshot_verification_failed")

        actual_tickers = _tickers(verified.get("tickers") or [])
        actual_timeframes = sorted(
            {str(item).strip().lower() for item in verified.get("timeframes") or []}
        )
        if verified:
            if actual_tickers != required_tickers:
                blockers.append("domain_market_ticker_scope_mismatch")
            if "15m" not in actual_timeframes:
                blockers.append("clean_snapshot_15m_lane_missing")
            try:
                if _aware(str(verified.get("end_date") or "")) > cutoff:
                    blockers.append("clean_snapshot_after_analysis_cutoff")
            except ValueError:
                blockers.append("clean_snapshot_end_date_invalid")

        coverage: dict[str, Any] = {}
        if verified:
            coverage = PipelineControlSavedDataCoverage().build(
                configured_tickers=required_tickers,
                price_paths=[str(verified["snapshot_path"])],
                macro_paths=[],
                min_rows=min_rows,
                max_rows=max_rows,
                max_abs_return=max_abs_return,
                min_cadence_ratio=min_cadence_ratio,
                save=False,
            )
        eligible_15m = [
            item
            for item in coverage.get("eligible_contexts") or []
            if item.get("timeframe") == "15m"
        ]
        eligible_tickers = _tickers(
            item.get("ticker") for item in eligible_15m
        )
        if verified and eligible_tickers != required_tickers:
            blockers.append("domain_15m_eligible_scope_incomplete")
        for item in eligible_15m:
            if not item.get("effective_start"):
                blockers.append("eligible_context_effective_start_missing")
            if item.get("source_sha256") != verified.get("snapshot_file_sha256"):
                blockers.append("eligible_context_source_hash_mismatch")

        blockers = sorted(set(blockers))
        ready = not blockers
        status = (
            "domain_sector_market_coverage_ready"
            if ready
            else "domain_sector_market_coverage_blocked"
        )
        manifest_path = Path(snapshot_manifest_path).resolve()
        payload: dict[str, Any] = {
            "run_id": _run_id("domain_sector_market_coverage_bridge"),
            "created_at": utc_now_iso(),
            "mode": "domain_sector_market_coverage_bridge",
            "contract": CONTRACT,
            "domain_id": domain_id,
            "status": status,
            "inputs": {
                "domain_id": domain_id,
                "analysis_cutoff": cutoff.isoformat(),
                "snapshot_manifest_path": str(manifest_path),
                "snapshot_manifest_sha256": (
                    _sha256_file(manifest_path) if manifest_path.is_file() else None
                ),
                "min_rows": min_rows,
                "max_rows": max_rows,
                "max_abs_return": max_abs_return,
                "min_cadence_ratio": min_cadence_ratio,
                "profile_domain_overlay_sha256": lifecycle.get(
                    "domain_overlay_sha256"
                ),
            },
            "required_market_scope": {
                "primary_universe": universe,
                "benchmark_ticker": benchmark,
                "required_tickers": required_tickers,
            },
            "source_snapshot": {
                "verified": bool(verified),
                "verification_error": verification_error,
                "manifest_path": verified.get("manifest_path"),
                "manifest_sha256": verified.get("manifest_sha256"),
                "snapshot_path": verified.get("snapshot_path"),
                "snapshot_file_sha256": verified.get("snapshot_file_sha256"),
                "snapshot_frame_sha256": verified.get("snapshot_frame_sha256"),
                "tickers": actual_tickers,
                "timeframes": actual_timeframes,
                "end_date": verified.get("end_date"),
            },
            "coverage_preview": coverage,
            "eligible_contexts": eligible_15m if ready else [],
            "summary": {
                "status": status,
                "structural_blockers": blockers,
                "required_ticker_count": len(required_tickers),
                "snapshot_ticker_count": len(actual_tickers),
                "eligible_15m_ticker_count": len(eligible_tickers),
                "source_snapshot_verified": bool(verified),
                "domain_scope_verified": actual_tickers == required_tickers,
                "candidate_ready_for_saved_price_repair": ready,
                "collector_run_performed": False,
                "network_access_performed": False,
                "repair_run_performed": False,
                "can_train": False,
                "can_trade": False,
            },
            "safety": {
                "review_only": True,
                "explicit_source_only": True,
                "automatic_filesystem_discovery_performed": False,
                "collector_run_performed": False,
                "network_access_performed": False,
                "database_write_performed": False,
                "repair_run_performed": False,
                "learning_write_performed": False,
                "broker_access_performed": False,
                "live_execution_performed": False,
            },
        }
        payload["stable_content_sha256"] = _stable_sha(payload)
        if save:
            payload["saved_paths"] = ReviewArtifactWriter(self.output_dir).write(
                payload=payload,
                markdown=render_markdown(payload),
                run_id=payload["run_id"],
            )
        return json_ready(payload)


def load_verified_domain_sector_market_coverage_bridge(
    artifact_path: str | Path,
    *,
    expected_domain_id: str | None = None,
) -> dict[str, Any]:
    path = Path(artifact_path).resolve()
    payload = _load_json(path)
    if payload.get("contract") != CONTRACT:
        raise ValueError("unsupported domain sector-market coverage contract")
    if payload.get("mode") != "domain_sector_market_coverage_bridge":
        raise ValueError("unsupported domain sector-market coverage mode")
    if payload.get("status") != "domain_sector_market_coverage_ready":
        raise ValueError("domain sector-market coverage is not ready")
    domain_id = str(payload.get("domain_id") or "")
    if expected_domain_id is not None and domain_id != expected_domain_id:
        raise ValueError("domain sector-market coverage domain mismatch")
    inputs = payload.get("inputs") or {}
    rebuilt = DomainSectorMarketCoverageBridge().build(
        domain_id=domain_id,
        analysis_cutoff=str(inputs.get("analysis_cutoff") or ""),
        snapshot_manifest_path=str(inputs.get("snapshot_manifest_path") or ""),
        min_rows=int(inputs.get("min_rows", 180)),
        max_rows=int(inputs.get("max_rows", 600)),
        max_abs_return=float(inputs.get("max_abs_return", 0.25)),
        min_cadence_ratio=float(inputs.get("min_cadence_ratio", 0.75)),
        save=False,
    )
    if (
        rebuilt.get("status") != payload.get("status")
        or rebuilt.get("stable_content_sha256")
        != payload.get("stable_content_sha256")
    ):
        raise ValueError("domain sector-market coverage content mismatch")
    return {
        "domain_id": domain_id,
        "analysis_cutoff": inputs.get("analysis_cutoff"),
        "required_market_scope": payload.get("required_market_scope") or {},
        "source_snapshot": payload.get("source_snapshot") or {},
        "eligible_contexts": list(payload.get("eligible_contexts") or []),
        "artifact_path": str(path),
        "artifact_sha256": _sha256_file(path),
    }


def render_markdown(payload: dict[str, Any]) -> str:
    summary = payload["summary"]
    lines = [
        "# Domain Sector-Market Coverage Bridge",
        "",
        f"- Domain: `{payload['domain_id']}`",
        f"- Status: `{payload['status']}`",
        f"- Required/snapshot/eligible 15m tickers: {summary['required_ticker_count']}/{summary['snapshot_ticker_count']}/{summary['eligible_15m_ticker_count']}",
        f"- Snapshot verified: {summary['source_snapshot_verified']}",
        f"- Ready for repair: {summary['candidate_ready_for_saved_price_repair']}",
        "- Collector/network run: false",
        "- Training/trading: false",
        "",
        "## Blockers",
        "",
    ]
    lines.extend(
        f"- {item}" for item in summary["structural_blockers"] or ["none"]
    )
    return "\n".join(lines).strip() + "\n"


def _stable_sha(payload: dict[str, Any]) -> str:
    coverage = payload.get("coverage_preview") or {}
    stable_coverage = {
        key: coverage.get(key)
        for key in (
            "mode",
            "contract",
            "inputs",
            "summary",
            "configured_assets",
            "price_sources",
            "contexts",
            "eligible_contexts",
            "known_contract_blocks",
            "explicit_non_actions",
        )
    }
    stable = {
        key: payload.get(key)
        for key in (
            "mode",
            "contract",
            "domain_id",
            "status",
            "inputs",
            "required_market_scope",
            "source_snapshot",
            "eligible_contexts",
            "summary",
            "safety",
        )
    }
    stable["coverage_preview"] = stable_coverage
    return _sha256_json(stable)


def _tickers(values: Any) -> list[str]:
    return sorted(
        {
            str(item).strip().upper()
            for item in (values or [])
            if str(item).strip()
        }
    )


def _aware(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (TypeError, ValueError) as exc:
        raise ValueError("timestamp must be ISO-8601") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("timestamp must be timezone-aware")
    return parsed


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sha256_json(value: Any) -> str:
    encoded = json.dumps(
        json_ready(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _run_id(prefix: str) -> str:
    return f"{prefix}_{utc_now_iso().replace(':', '').replace('+', 'Z')}"


__all__ = [
    "CONTRACT",
    "DomainSectorMarketCoverageBridge",
    "load_verified_domain_sector_market_coverage_bridge",
]
