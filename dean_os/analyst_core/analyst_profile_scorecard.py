from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

from dean_os.schemas import utc_now_iso
from dean_os.utils import json_ready


class AnalystProfileScorecard:
    """Aggregates analyst profile runs into activation guidance."""

    def __init__(self, output_dir: str | Path = "reports/dean_os/analyst_profile_scorecard"):
        self.output_dir = Path(output_dir)

    def build(
        self,
        profile_runs_dir: str | Path = "reports/dean_os/analyst_profiles",
        min_completed_runs: int = 3,
        min_avg_confidence: float = 0.55,
        min_avg_citations: float = 1.0,
        save: bool = True,
    ) -> dict[str, Any]:
        runs = _load_orchestrator_runs(profile_runs_dir)
        profiles = _profile_buckets(runs)
        scorecards = {
            profile: _score_profile(
                profile=profile,
                items=items,
                min_completed_runs=min_completed_runs,
                min_avg_confidence=min_avg_confidence,
                min_avg_citations=min_avg_citations,
            )
            for profile, items in sorted(profiles.items())
        }
        payload = {
            "run_id": _run_id("analyst_profile_scorecard"),
            "created_at": utc_now_iso(),
            "mode": "analyst_profile_scorecard",
            "inputs": {
                "profile_runs_dir": str(profile_runs_dir),
                "min_completed_runs": min_completed_runs,
                "min_avg_confidence": min_avg_confidence,
                "min_avg_citations": min_avg_citations,
            },
            "summary": {
                "orchestrator_run_count": len(runs),
                "profile_count": len(scorecards),
                "activation_ready_profiles": [
                    profile for profile, card in scorecards.items() if card["activation_status"] == "ready_to_activate"
                ],
                "keep_candidate_profiles": [
                    profile for profile, card in scorecards.items() if card["activation_status"] == "keep_candidate"
                ],
                "blocked_profiles": [
                    profile for profile, card in scorecards.items() if card["activation_status"] == "blocked"
                ],
            },
            "profiles": scorecards,
            "recommendations": _recommendations(scorecards, len(runs)),
        }
        if save:
            self.save(payload)
        return payload

    def save(self, payload: dict[str, Any]) -> tuple[Path, Path]:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        json_path = self.output_dir / f"{payload['run_id']}.json"
        md_path = self.output_dir / f"{payload['run_id']}.md"
        latest_json = self.output_dir / "latest.json"
        latest_md = self.output_dir / "latest.md"
        payload["saved_paths"] = {
            "json": str(json_path),
            "markdown": str(md_path),
            "latest_json": str(latest_json),
            "latest_markdown": str(latest_md),
        }
        rendered_json = json.dumps(json_ready(payload), indent=2, ensure_ascii=False) + "\n"
        rendered_md = render_analyst_profile_scorecard_markdown(payload)
        json_path.write_text(rendered_json, encoding="utf-8")
        latest_json.write_text(rendered_json, encoding="utf-8")
        md_path.write_text(rendered_md, encoding="utf-8")
        latest_md.write_text(rendered_md, encoding="utf-8")
        return json_path, md_path


def render_analyst_profile_scorecard_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# DEAN-OS Analyst Profile Scorecard",
        "",
        f"- Run ID: `{payload.get('run_id')}`",
        f"- Orchestrator runs: {payload.get('summary', {}).get('orchestrator_run_count', 0)}",
        f"- Profiles: {payload.get('summary', {}).get('profile_count', 0)}",
        f"- Ready: {', '.join(payload.get('summary', {}).get('activation_ready_profiles', [])) or 'none'}",
        "",
        "## Profiles",
        "",
    ]
    for profile, card in payload.get("profiles", {}).items():
        lines.extend(
            [
                f"### {profile}",
                "",
                f"- Status: `{card.get('activation_status')}`",
                f"- Completed: {card.get('completed_count')}",
                f"- Skipped: {card.get('skipped_count')}",
                f"- Avg confidence: {card.get('avg_confidence')}",
                f"- Avg citations: {card.get('avg_citations')}",
                f"- Recommendation: {card.get('recommendation')}",
                "",
            ]
        )
    lines.extend(["## Recommendations", ""])
    lines.extend(f"- {item}" for item in payload.get("recommendations", []))
    return "\n".join(lines).strip() + "\n"


def _load_orchestrator_runs(profile_runs_dir: str | Path) -> list[dict[str, Any]]:
    root = Path(profile_runs_dir)
    if not root.exists():
        return []
    runs: list[dict[str, Any]] = []
    for path in sorted(root.glob("*.json")):
        if path.name == "latest.json":
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if payload.get("mode") == "analyst_profile_orchestrator":
            payload["_path"] = str(path)
            runs.append(payload)
    latest = root / "latest.json"
    if latest.exists():
        try:
            payload = json.loads(latest.read_text(encoding="utf-8"))
            if payload.get("mode") == "analyst_profile_orchestrator":
                payload["_path"] = str(latest)
                if not any(item.get("run_id") == payload.get("run_id") for item in runs):
                    runs.append(payload)
        except Exception:
            pass
    return runs


def _profile_buckets(runs: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for run in runs:
        for item in run.get("profile_runs", []):
            profile = item.get("profile")
            if profile:
                buckets[profile].append({"kind": "run", "orchestrator_run_id": run.get("run_id"), **item})
        for skipped in run.get("profile_plan", {}).get("skipped_profiles", []):
            profile = skipped.get("profile")
            if profile:
                buckets[profile].append({"kind": "skipped", "orchestrator_run_id": run.get("run_id"), **skipped})
    return buckets


def _score_profile(
    profile: str,
    items: list[dict[str, Any]],
    min_completed_runs: int,
    min_avg_confidence: float,
    min_avg_citations: float,
) -> dict[str, Any]:
    completed = [item for item in items if item.get("status") == "completed"]
    skipped = [item for item in items if item.get("kind") == "skipped" or item.get("status") == "skipped"]
    confidences = [_confidence(item) for item in completed]
    citations = [_citation_count(item) for item in completed]
    note_counts = [int(item.get("note_count") or 0) for item in completed]
    verdict_counts = Counter(_verdict(item) for item in completed if _verdict(item))
    skipped_reasons = Counter(str(item.get("reason") or "unknown") for item in skipped)
    avg_confidence = round(mean(confidences), 4) if confidences else None
    avg_citations = round(mean(citations), 4) if citations else None

    blockers: list[str] = []
    if len(completed) < min_completed_runs:
        blockers.append(f"Needs at least {min_completed_runs} completed runs.")
    if avg_confidence is None or avg_confidence < min_avg_confidence:
        blockers.append(f"Needs average confidence >= {min_avg_confidence}.")
    if avg_citations is None or avg_citations < min_avg_citations:
        blockers.append(f"Needs average citations >= {min_avg_citations}.")
    if skipped and not completed:
        blockers.append("Profile has only skipped attempts.")

    if blockers:
        activation_status = "blocked" if not completed else "keep_candidate"
        recommendation = "Keep as candidate; " + " ".join(blockers)
    else:
        activation_status = "ready_to_activate"
        recommendation = "Profile has enough reviewed run evidence to consider default activation."

    return {
        "profile": profile,
        "completed_count": len(completed),
        "skipped_count": len(skipped),
        "avg_confidence": avg_confidence,
        "avg_citations": avg_citations,
        "avg_note_count": round(mean(note_counts), 4) if note_counts else None,
        "verdict_counts": dict(sorted(verdict_counts.items())),
        "skipped_reasons": dict(sorted(skipped_reasons.items())),
        "activation_status": activation_status,
        "blockers": blockers,
        "recommendation": recommendation,
    }


def _confidence(item: dict[str, Any]) -> float:
    report = item.get("report") if isinstance(item.get("report"), dict) else {}
    if report.get("confidence") is not None:
        return float(report["confidence"])
    summary = item.get("summary") if isinstance(item.get("summary"), dict) else {}
    if summary.get("avg_nlp_sentiment") is not None:
        return min(0.9, 0.4 + abs(float(summary["avg_nlp_sentiment"])) * 0.3)
    if item.get("note_count"):
        return min(0.8, 0.3 + int(item["note_count"]) * 0.08)
    return 0.0


def _citation_count(item: dict[str, Any]) -> int:
    report = item.get("report") if isinstance(item.get("report"), dict) else {}
    if report.get("evidence"):
        return len(report.get("evidence", []))
    if item.get("note_count"):
        return int(item["note_count"])
    return 0


def _verdict(item: dict[str, Any]) -> str | None:
    report = item.get("report") if isinstance(item.get("report"), dict) else {}
    return report.get("verdict")


def _recommendations(scorecards: dict[str, dict[str, Any]], run_count: int) -> list[str]:
    if not run_count:
        return ["Run AnalystProfileOrchestrator before building scorecards."]
    ready = [profile for profile, card in scorecards.items() if card["activation_status"] == "ready_to_activate"]
    if ready:
        return [f"Review ready profiles before changing defaults: {', '.join(ready)}."]
    return ["Keep generalist_base_analyst as default until candidate profiles have enough completed, cited runs."]


def _run_id(prefix: str) -> str:
    return f"{prefix}_{utc_now_iso().replace(':', '').replace('-', '').replace('.', '_')}"

