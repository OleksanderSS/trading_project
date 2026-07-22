from __future__ import annotations

from typing import Any


def render_analyst_payload_markdown(payload: dict[str, Any]) -> str:
    analyst_report = payload.get("analyst_report") or {}
    analytical_report = payload.get("analytical_report") or {}
    thesis = analyst_report.get("thesis") or {}
    basket = analyst_report.get("ticker_basket") or {}
    quality = analyst_report.get("quality_gates") or {}
    review_packet = analyst_report.get("review_packet") or {}
    outcome = analyst_report.get("outcome_tracking_plan") or {}

    lines: list[str] = [
        "# DEAN-OS Domain Analyst Review",
        "",
        f"- Mode: `{payload.get('mode', 'domain_analyst')}`",
        f"- Agent: `{analytical_report.get('agent_name') or analyst_report.get('agent_name')}`",
        f"- Domain: `{analyst_report.get('domain_id') or analytical_report.get('asset_or_sector')}`",
        f"- As of: `{analyst_report.get('as_of')}`",
        f"- Recommendation: `{analyst_report.get('recommendation')}`",
        f"- Review required: `{analyst_report.get('review_required', True)}`",
        f"- Live execution allowed: `{analyst_report.get('live_execution_allowed', False)}`",
        "",
        "## Thesis",
        "",
        f"- Stance: `{thesis.get('stance')}`",
        f"- Expected direction: `{thesis.get('expected_direction')}`",
        f"- Confidence: `{thesis.get('confidence')}`",
        f"- Data quality: `{thesis.get('data_quality')}`",
        "",
        str(thesis.get("thesis") or ""),
        "",
        "## Ticker Basket",
        "",
        f"- Basket status: `{basket.get('basket_status')}`",
        f"- Direct ready count: `{basket.get('direct_ready_count')}`",
        f"- Basket candidate count: `{basket.get('basket_candidate_count')}`",
        f"- Blocked count: `{basket.get('blocked_count')}`",
        "",
    ]

    candidates = basket.get("candidates") or []
    if candidates:
        lines.extend(["| Ticker | Status | Confidence | Missing | Blocked windows |", "|---|---:|---:|---|---|"])
        for candidate in candidates:
            lines.append(
                "| {ticker} | {status} | {confidence} | {missing} | {blocked} |".format(
                    ticker=candidate.get("ticker"),
                    status=candidate.get("candidate_status"),
                    confidence=candidate.get("confidence"),
                    missing=", ".join(candidate.get("required_missing_evidence") or []),
                    blocked=", ".join(candidate.get("blocked_windows") or []),
                )
            )
        lines.append("")

    lines.extend(
        [
            "## Quality Gates",
            "",
            f"- Required evidence complete: `{quality.get('required_evidence_complete')}`",
            f"- Evidence count: `{quality.get('evidence_count')}`",
            f"- Evidence quality score: `{quality.get('evidence_quality_score')}`",
            f"- Direct ticker guardrail enabled: `{quality.get('direct_ticker_guardrail_enabled')}`",
            "",
            "## Risks",
            "",
        ]
    )

    for risk in thesis.get("risks") or analytical_report.get("risks") or []:
        lines.append(f"- {risk}")

    lines.extend(["", "## Blind Spots", ""])
    for item in thesis.get("blind_spots") or analytical_report.get("blind_spots") or []:
        lines.append(f"- {item}")

    lines.extend(["", "## Outcome Tracking", ""])
    for key in ("track_domain_thesis", "track_ticker_candidates", "horizon_days", "requires_future_outcome_evaluation"):
        if key in outcome:
            lines.append(f"- {key}: `{outcome.get(key)}`")

    lines.extend(
        [
            "",
            "## Safety",
            "",
        ]
    )
    safety = payload.get("safety") or {}
    artifact_safety = payload.get("artifact_safety") or {}
    for key in sorted({*safety.keys(), *artifact_safety.keys()}):
        value = safety.get(key, artifact_safety.get(key))
        lines.append(f"- {key}: `{value}`")

    lines.extend(["", "## Operator Note", "", str(review_packet.get("operator_note") or "Review-only artifact.")])
    return "\n".join(lines).strip() + "\n"
