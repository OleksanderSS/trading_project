from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from dean_os.artifact_writer import ReviewArtifactWriter
from dean_os.schemas import utc_now_iso
from dean_os.utils import json_ready, sha256_json


class SpecialistContextReviewPacket:
    """Join domain/sector review to one ticker context without conflation."""

    def __init__(
        self,
        output_dir: str | Path = (
            "reports/dean_os/specialist_context_review_current"
        ),
    ):
        self.output_dir = Path(output_dir)

    def build(
        self,
        *,
        sector_to_ticker_review_path: str | Path,
        ticker: str,
        timeframe: str | None,
        context_as_of: str | None,
        domain_thesis_path: str | Path | None = None,
        max_evidence_age_days: float = 30.0,
        save: bool = True,
    ) -> dict[str, Any]:
        review_path = Path(sector_to_ticker_review_path)
        review = _load_json(review_path)
        thesis_path = (
            Path(domain_thesis_path)
            if domain_thesis_path is not None
            else None
        )
        thesis = _load_json(thesis_path) if thesis_path else {}
        ticker_value = str(ticker).upper()
        timeframe_value = (
            str(timeframe).lower() if timeframe else None
        )
        matches = [
            item
            for item in review.get("ticker_review_map", [])
            if isinstance(item, dict)
            and str(item.get("ticker") or "").upper()
            == ticker_value
        ]
        issues: list[dict[str, Any]] = []
        if review.get("mode") != "sector_to_ticker_review_packet":
            issues.append(
                _issue(
                    "invalid_sector_to_ticker_review_mode",
                    "Input is not a sector-to-ticker review packet.",
                )
            )
        if len(matches) > 1:
            issues.append(
                _issue(
                    "duplicate_ticker_review_entries",
                    "More than one ticker review entry matches.",
                    observed=len(matches),
                )
            )
        candidate = matches[0] if len(matches) == 1 else {}
        scope = _evidence_scope(candidate)
        if not candidate:
            issues.append(
                _issue(
                    "direct_ticker_review_missing",
                    "No exact ticker entry exists; sector context cannot "
                    "be promoted to ticker evidence.",
                )
            )
        direct = _mapping(candidate.get("direct_evidence"))
        supporting_as_of = sorted(
            {
                str(item)
                for item in direct.get("supporting_as_of", [])
                if item
            }
        )
        latest_evidence_at = _latest_timestamp(supporting_as_of)
        context_time = _parse_timestamp(context_as_of)
        age_days = None
        point_in_time_status = "unverifiable_missing_as_of"
        if latest_evidence_at and context_time:
            age_days = (
                context_time - latest_evidence_at
            ).total_seconds() / 86400.0
            if age_days < 0:
                point_in_time_status = "future_evidence_conflict"
                issues.append(
                    _issue(
                        "future_specialist_evidence",
                        "Specialist evidence is later than the requested "
                        "pipeline context.",
                        age_days=round(age_days, 6),
                    )
                )
            elif age_days > max_evidence_age_days:
                point_in_time_status = "older_than_review_window"
                issues.append(
                    _issue(
                        "specialist_evidence_age_exceeded",
                        "Latest direct ticker evidence is older than the "
                        "configured review window.",
                        age_days=round(age_days, 6),
                        threshold_days=max_evidence_age_days,
                    )
                )
            else:
                point_in_time_status = "point_in_time_compatible"
        elif not latest_evidence_at:
            issues.append(
                _issue(
                    "direct_ticker_as_of_missing",
                    "No direct ticker supporting-as-of timestamp exists.",
                )
            )
        source_timeframe = (
            str(
                candidate.get("timeframe")
                or direct.get("timeframe")
            ).lower()
            if (
                candidate.get("timeframe")
                or direct.get("timeframe")
            )
            else None
        )
        if source_timeframe is None:
            timeframe_status = (
                "unverified_source_timeframe_not_declared"
            )
            issues.append(
                _issue(
                    "specialist_timeframe_unaligned",
                    "The sector-to-ticker review does not declare a "
                    "timeframe, so it cannot be treated as exact "
                    f"{timeframe_value} evidence.",
                )
            )
        elif timeframe_value == source_timeframe:
            timeframe_status = "aligned"
        else:
            timeframe_status = "mismatch"
            issues.append(
                _issue(
                    "specialist_timeframe_mismatch",
                    "Specialist evidence timeframe does not match the "
                    "requested pipeline context.",
                    requested=timeframe_value,
                    source=source_timeframe,
                )
            )
        manual_review_required = bool(
            _mapping(review.get("summary")).get(
                "manual_review_required",
                True,
            )
        )
        manual_review_decision = candidate.get(
            "manual_review_decision"
        )
        approved_ticker_thesis = (
            manual_review_decision == "accepted"
            and not manual_review_required
        )
        if candidate and manual_review_required:
            issues.append(
                _issue(
                    "manual_ticker_review_pending",
                    "Ticker evidence remains a manual-review candidate.",
                )
            )
        direct_review_candidate = scope == (
            "direct_ticker_review_candidate"
        )
        exact_pipeline_context_eligible = (
            direct_review_candidate
            and point_in_time_status == "point_in_time_compatible"
            and timeframe_status == "aligned"
            and approved_ticker_thesis
            and not issues
        )
        summary = _mapping(review.get("summary"))
        thesis_summary = _mapping(thesis.get("summary"))
        packet = {
            "run_id": _run_id(),
            "created_at": utc_now_iso(),
            "mode": "specialist_context_review_packet",
            "schema_version": (
                "dean_specialist_context_review_v1"
            ),
            "status": (
                "specialist_context_exact_match_ready"
                if exact_pipeline_context_eligible
                else "specialist_context_review_only_with_limits"
            ),
            "requested_context": {
                "ticker": ticker_value,
                "timeframe": timeframe_value,
                "as_of": context_as_of,
            },
            "domain_scope": {
                "domain_id": summary.get("domain_profile")
                or thesis_summary.get("domain_id"),
                "sector": summary.get("sector"),
                "sector_stance": summary.get("sector_stance"),
                "domain_ticker_direct_count": thesis_summary.get(
                    "ticker_direct_count"
                ),
                "domain_thesis_is_ticker_evidence": False,
            },
            "ticker_scope": {
                "ticker": ticker_value,
                "evidence_scope": scope,
                "candidate_found": bool(candidate),
                "candidate_status": candidate.get(
                    "candidate_status"
                ),
                "review_status": candidate.get("review_status"),
                "thesis_level": candidate.get("thesis_level"),
                "allowed_use": candidate.get("allowed_use"),
                "direct_evidence": direct,
                "blocked_evidence": _mapping(
                    candidate.get("blocked_evidence")
                ),
                "risk_and_counter_thesis_flags": list(
                    candidate.get(
                        "risk_and_counter_thesis_flags"
                    )
                    or []
                ),
                "eligible_as_direct_ticker_review_context": (
                    direct_review_candidate
                ),
                "eligible_as_approved_ticker_thesis": (
                    approved_ticker_thesis
                ),
                "manual_review_decision": manual_review_decision,
            },
            "point_in_time": {
                "latest_direct_evidence_at": (
                    latest_evidence_at.isoformat()
                    if latest_evidence_at
                    else None
                ),
                "context_as_of": context_as_of,
                "age_days": (
                    round(age_days, 6)
                    if age_days is not None
                    else None
                ),
                "max_evidence_age_days": max_evidence_age_days,
                "status": point_in_time_status,
            },
            "timeframe_alignment": {
                "requested_timeframe": timeframe_value,
                "source_timeframe": source_timeframe,
                "status": timeframe_status,
            },
            "review_issues": issues,
            "source_provenance": {
                "sector_to_ticker_review_path": str(review_path),
                "sector_to_ticker_review_sha256": _sha256(
                    review_path
                ),
                "sector_to_ticker_review_run_id": review.get(
                    "run_id"
                ),
                "domain_thesis_path": (
                    str(thesis_path) if thesis_path else None
                ),
                "domain_thesis_sha256": (
                    _sha256(thesis_path) if thesis_path else None
                ),
                "domain_thesis_run_id": thesis.get("run_id"),
            },
            "packet_fingerprint": None,
            "safety": {
                "supporting_review_only": True,
                "sector_context_promoted_to_ticker": False,
                "manual_review_required": manual_review_required,
                "eligible_for_exact_pipeline_context": (
                    exact_pipeline_context_eligible
                ),
                "directional_synthesis_allowed": False,
                "decision_influence": False,
                "can_write_learning_memory": False,
                "can_create_recommendation": False,
                "can_trade": False,
            },
        }
        packet["packet_fingerprint"] = sha256_json(
            {
                "requested_context": packet["requested_context"],
                "domain_scope": packet["domain_scope"],
                "ticker_scope": packet["ticker_scope"],
                "point_in_time": packet["point_in_time"],
                "timeframe_alignment": packet[
                    "timeframe_alignment"
                ],
                "source_provenance": packet["source_provenance"],
            }
        )
        if save:
            saved = ReviewArtifactWriter(self.output_dir).write(
                payload=packet,
                markdown=render_specialist_context_review_markdown(
                    packet
                ),
                run_id=packet["run_id"],
            )
            packet["saved_paths"] = saved
        return json_ready(packet)


def _evidence_scope(candidate: dict[str, Any]) -> str:
    if not candidate:
        return "sector_context_only"
    direct = _mapping(candidate.get("direct_evidence"))
    if (
        candidate.get("review_status")
        in {"review_ready", "review_ready_with_evidence_limits"}
        and int(direct.get("directional_ready_runs") or 0) > 0
    ):
        return "direct_ticker_review_candidate"
    if candidate.get("candidate_status") in {
        "ticker_context_ready",
        "direct_ticker_thesis_ready",
    }:
        return "basket_or_ticker_context_candidate"
    return "sector_context_only"


def render_specialist_context_review_markdown(
    payload: dict[str, Any],
) -> str:
    requested = payload.get("requested_context", {})
    domain = payload.get("domain_scope", {})
    ticker = payload.get("ticker_scope", {})
    point = payload.get("point_in_time", {})
    timeframe = payload.get("timeframe_alignment", {})
    lines = [
        "# DEAN-OS Specialist Context Review",
        "",
        f"- Status: `{payload.get('status')}`",
        f"- Requested: `{requested.get('ticker')}/"
        f"{requested.get('timeframe')}`",
        f"- Domain/sector: `{domain.get('domain_id')}/"
        f"{domain.get('sector')}`",
        f"- Ticker evidence scope: "
        f"`{ticker.get('evidence_scope')}`",
        f"- Point-in-time status: `{point.get('status')}`",
        f"- Timeframe alignment: `{timeframe.get('status')}`",
        f"- Exact pipeline-context eligible: "
        f"{payload.get('safety', {}).get('eligible_for_exact_pipeline_context')}",
        "",
        "## Review Issues",
        "",
    ]
    lines.extend(
        f"- `{item.get('code')}`: {item.get('message')}"
        for item in payload.get("review_issues", [])
    )
    lines.extend(
        [
            "",
            "Sector context is never promoted to ticker evidence. A "
            "direct-ticker review candidate is still not an approved "
            "thesis, recommendation, consensus signal, or trade.",
        ]
    )
    return "\n".join(lines) + "\n"


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _latest_timestamp(values: list[str]) -> datetime | None:
    parsed = [
        value
        for value in (_parse_timestamp(item) for item in values)
        if value is not None
    ]
    return max(parsed) if parsed else None


def _parse_timestamp(value: Any) -> datetime | None:
    if value is None:
        return None
    try:
        parsed = datetime.fromisoformat(
            str(value).replace("Z", "+00:00")
        )
    except (TypeError, ValueError):
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _issue(
    code: str,
    message: str,
    **details: Any,
) -> dict[str, Any]:
    return {
        "code": code,
        "message": message,
        "details": details,
    }


def _run_id() -> str:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S_%fZ")
    return f"specialist_context_review_{stamp}"
