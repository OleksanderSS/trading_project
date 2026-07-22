from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any


class DailyBriefingRenderer:
    def render_markdown(self, briefing: Any, *, evidence_gap_plan: Any | None = None) -> str:
        data = briefing.model_dump(mode="json") if hasattr(briefing, "model_dump") else dict(briefing)
        plan = evidence_gap_plan.model_dump(mode="json") if hasattr(evidence_gap_plan, "model_dump") else (dict(evidence_gap_plan) if evidence_gap_plan else {})
        lines = [
            f"# DEAN-OS Daily Briefing — {data.get('domain_id')}",
            "",
            f"- As of: `{data.get('as_of')}`",
            f"- Briefing ID: `{data.get('briefing_id')}`",
            "- Mode: `review_only`",
            "",
            "## 0. Regime Snapshot",
            "",
            "```json",
            json.dumps(data.get("regime_snapshot", {}), ensure_ascii=False, indent=2),
            "```",
            "",
            "## 1. Mandatory Coverage Gate",
            "",
        ]
        for item in data.get("mandatory_coverage_gate", []) or []:
            evidence = ", ".join(item.get("evidence_ids", []) or []) or "none"
            lines.append(f"- **{item.get('label')}** — `{item.get('status')}`; evidence: {evidence}. {item.get('conclusion')}")
        lines.extend(["", "## 2. Top Developments", ""])
        developments = data.get("top_developments", []) or []
        lines.extend(
            f"- `{item.get('event_type')}` — {item.get('summary')}" for item in developments
        )
        if not developments:
            lines.append("- No classified material development in the bounded evidence set.")
        lines.extend(["", "## 3. Context Grid + Indicator State Grid", "", "```json"])
        lines.append(json.dumps({"context_grid": data.get("context_grid", {}), "indicator_state_grid": data.get("indicator_state_grid", {})}, ensure_ascii=False, indent=2))
        lines.extend(["```", "", "## 4. Scenario Probabilities", ""])
        for node in data.get("scenario_probabilities", {}).get("nodes", []) or []:
            lines.append(f"- {node.get('label') or node.get('scenario_node_id')}: {float(node.get('probability', 0.0)):.1%}")
        lines.extend(["", "## 5. Practical Implications", ""])
        lines.extend(f"- {item}" for item in data.get("practical_implications", []) or [])
        lines.extend(["", "## 6. Risks / Evidence Gaps", ""])
        lines.extend(f"- {item}" for item in data.get("risks_and_evidence_gaps", []) or [])
        for task in plan.get("tasks", []) or []:
            lines.append(f"- `{task.get('priority')}` gap `{task.get('coverage_id')}`: {task.get('reason')}")
        lines.extend(["", "## 7. Replay / Analyst Journal", "", "```json"])
        lines.append(json.dumps(data.get("replay_journal", {}), ensure_ascii=False, indent=2))
        lines.extend(["```", ""])
        return "\n".join(lines)

    def render_html(self, briefing: Any, *, evidence_gap_plan: Any | None = None) -> str:
        markdown = self.render_markdown(briefing, evidence_gap_plan=evidence_gap_plan)
        return (
            "<!doctype html><html><head><meta charset='utf-8'><title>DEAN-OS Daily Briefing</title>"
            "<style>body{font-family:system-ui,sans-serif;max-width:1100px;margin:40px auto;padding:0 24px;line-height:1.5}"
            "pre{white-space:pre-wrap;background:#f4f4f4;padding:16px;border-radius:8px}</style></head><body>"
            f"<pre>{html.escape(markdown)}</pre></body></html>"
        )

    def save(self, briefing: Any, output_dir: str | Path, *, evidence_gap_plan: Any | None = None) -> tuple[Path, Path]:
        directory = Path(output_dir)
        directory.mkdir(parents=True, exist_ok=True)
        data = briefing.model_dump(mode="json") if hasattr(briefing, "model_dump") else dict(briefing)
        stem = str(data.get("briefing_id") or "daily_briefing")
        md = directory / f"{stem}.md"
        html_path = directory / f"{stem}.html"
        md.write_text(self.render_markdown(briefing, evidence_gap_plan=evidence_gap_plan), encoding="utf-8")
        html_path.write_text(self.render_html(briefing, evidence_gap_plan=evidence_gap_plan), encoding="utf-8")
        return md, html_path


__all__ = ["DailyBriefingRenderer"]
