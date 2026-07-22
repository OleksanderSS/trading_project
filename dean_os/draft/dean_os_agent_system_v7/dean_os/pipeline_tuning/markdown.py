from __future__ import annotations

from typing import Any


def render_pipeline_tuning_payload_markdown(payload: dict[str, Any]) -> str:
    report = payload.get("pipeline_report") or {}
    plan = payload.get("pipeline_tuning_plan") or {}

    lines: list[str] = [
        "# DEAN-OS Pipeline Tuning Controller Review",
        "",
        f"- Mode: `{payload.get('mode', 'pipeline_tuning_controller')}`",
        f"- Agent: `{report.get('agent_name')}`",
        f"- Plan status: `{plan.get('status')}`",
        f"- Target: `{plan.get('target')}`",
        f"- Review required: `{plan.get('review_required', True)}`",
        f"- Live execution allowed: `{plan.get('live_execution_allowed', False)}`",
        f"- Production config write allowed: `{plan.get('production_config_write_allowed', False)}`",
        "",
        "## Reasons",
        "",
    ]

    for reason in plan.get("reasons") or report.get("reasons") or []:
        lines.append(f"- {reason}")

    lines.extend(["", "## Tuning Planes", ""])
    planes = plan.get("planes") or []
    if planes:
        lines.extend(["| Plane | Status | Bounds |", "|---|---:|---|"])
        for plane in planes:
            lines.append(
                "| {plane} | {status} | {bounds} |".format(
                    plane=plane.get("plane_id"),
                    status=plane.get("status"),
                    bounds=plane.get("proposed_bounds"),
                )
            )
    else:
        lines.append("- No tuning planes proposed.")

    lines.extend(["", "## Guardrails", ""])
    for item in plan.get("guardrails") or []:
        lines.append(f"- {item}")

    lines.extend(["", "## Risks", ""])
    for item in plan.get("risks") or report.get("risks") or []:
        lines.append(f"- {item}")

    lines.extend(["", "## Action Proposals", ""])
    proposals = payload.get("action_proposals") or []
    if proposals:
        for proposal in proposals:
            lines.extend(
                [
                    f"### {proposal.get('action_type')} -> {proposal.get('target')}",
                    "",
                    f"- Reason: {proposal.get('reason')}",
                    f"- Command preview: `{proposal.get('command_preview')}`",
                    "",
                ]
            )
    else:
        lines.append("- No action proposal created.")

    lines.extend(["", "## Safety", ""])
    safety = payload.get("safety") or {}
    artifact_safety = payload.get("artifact_safety") or {}
    for key in sorted({*safety.keys(), *artifact_safety.keys()}):
        value = safety.get(key, artifact_safety.get(key))
        lines.append(f"- {key}: `{value}`")

    return "\n".join(lines).strip() + "\n"
