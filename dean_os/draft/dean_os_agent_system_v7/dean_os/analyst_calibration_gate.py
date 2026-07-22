from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

from dean_os.draft.dean_os_agent_system_v7.dean_os.analyst_outcome_evaluation_loop import ANALYST_LEARNING_FLAG
from dean_os.draft.dean_os_agent_system_v7.dean_os.analyst_profile_scorecard import AnalystProfileScorecard
from dean_os.draft.dean_os_agent_system_v7.dean_os.context_performance import AgentPerformanceByContext
from dean_os.draft.dean_os_agent_system_v7.dean_os.learning import LearningStore
from dean_os.schemas import AgentLearningRecord, utc_now_iso
from dean_os.utils import json_ready


class AnalystCalibrationGate:
    """Conservative promotion gate for analyst profiles and weights.

    This gate is proposal-only. It never writes config, never changes consensus
    weights, and never promotes profiles automatically.
    """

    def __init__(self, output_dir: str | Path = "reports/dean_os/analyst_calibration_gate"):
        self.output_dir = Path(output_dir)

    def run(
        self,
        profile_scorecard_path: str | Path | None = None,
        profile_runs_dir: str | Path = "reports/dean_os/analyst_profiles",
        learning_path: str | Path = "data/dean_os/agent_learning.sqlite",
        memory_path: str | Path = "data/dean_os/recommendation_memory.sqlite",
        min_profile_runs: int = 3,
        min_completed_outcomes: int = 3,
        min_hit_rate: float = 0.55,
        max_miss_rate: float = 0.4,
        require_scorecard_ready: bool = True,
        save: bool = True,
    ) -> dict[str, Any]:
        scorecard = _load_or_build_scorecard(
            profile_scorecard_path=profile_scorecard_path,
            profile_runs_dir=profile_runs_dir,
        )
        outcome_profiles = _profile_outcomes(learning_path)
        context_summary = AgentPerformanceByContext(learning_path, memory_path).build_summary()
        profiles = sorted(set(scorecard.get("profiles", {})) | set(outcome_profiles))
        calibration = {
            profile: _calibrate_profile(
                profile=profile,
                scorecard=scorecard.get("profiles", {}).get(profile, {}),
                outcomes=outcome_profiles.get(profile, _empty_outcomes()),
                context_summary=context_summary,
                min_profile_runs=min_profile_runs,
                min_completed_outcomes=min_completed_outcomes,
                min_hit_rate=min_hit_rate,
                max_miss_rate=max_miss_rate,
                require_scorecard_ready=require_scorecard_ready,
            )
            for profile in profiles
        }
        payload = {
            "run_id": _run_id("analyst_calibration_gate"),
            "created_at": utc_now_iso(),
            "mode": "analyst_calibration_gate",
            "inputs": {
                "profile_scorecard_path": str(profile_scorecard_path) if profile_scorecard_path else None,
                "profile_runs_dir": str(profile_runs_dir),
                "learning_path": str(learning_path),
                "memory_path": str(memory_path),
                "min_profile_runs": min_profile_runs,
                "min_completed_outcomes": min_completed_outcomes,
                "min_hit_rate": min_hit_rate,
                "max_miss_rate": max_miss_rate,
                "require_scorecard_ready": require_scorecard_ready,
            },
            "summary": _summary(calibration),
            "profile_scorecard_summary": scorecard.get("summary", {}),
            "context_performance": context_summary,
            "profiles": calibration,
            "recommendations": _recommendations(calibration),
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
        rendered_md = render_analyst_calibration_gate_markdown(payload)
        json_path.write_text(rendered_json, encoding="utf-8")
        latest_json.write_text(rendered_json, encoding="utf-8")
        md_path.write_text(rendered_md, encoding="utf-8")
        latest_md.write_text(rendered_md, encoding="utf-8")
        return json_path, md_path


def render_analyst_calibration_gate_markdown(payload: dict[str, Any]) -> str:
    summary = payload.get("summary", {})
    lines = [
        "# DEAN-OS Analyst Calibration Gate",
        "",
        f"- Run ID: `{payload.get('run_id')}`",
        f"- Profiles: {summary.get('profile_count', 0)}",
        f"- Ready for review: {', '.join(summary.get('ready_for_review_profiles', [])) or 'none'}",
        f"- Keep candidate: {', '.join(summary.get('keep_candidate_profiles', [])) or 'none'}",
        f"- Blocked: {', '.join(summary.get('blocked_profiles', [])) or 'none'}",
        "",
        "## Profiles",
        "",
    ]
    for profile, card in payload.get("profiles", {}).items():
        lines.extend(
            [
                f"### {profile}",
                "",
                f"- Status: `{card.get('calibration_status')}`",
                f"- Suggested weight delta: {card.get('suggested_weight_delta')}",
                f"- Completed outcomes: {card.get('outcomes', {}).get('completed_count')}",
                f"- Hit rate: {card.get('outcomes', {}).get('hit_rate')}",
                f"- Miss rate: {card.get('outcomes', {}).get('miss_rate')}",
                f"- Recommendation: {card.get('recommendation')}",
                "",
            ]
        )
    lines.extend(["## Recommendations", ""])
    lines.extend(f"- {item}" for item in payload.get("recommendations", []))
    return "\n".join(lines).strip() + "\n"


def _load_or_build_scorecard(
    profile_scorecard_path: str | Path | None,
    profile_runs_dir: str | Path,
) -> dict[str, Any]:
    if profile_scorecard_path:
        path = Path(profile_scorecard_path)
        if path.exists():
            payload = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                return payload
    return AnalystProfileScorecard().build(profile_runs_dir=profile_runs_dir, save=False)


def _profile_outcomes(learning_path: str | Path) -> dict[str, dict[str, Any]]:
    records = [
        record
        for record in LearningStore(learning_path).list_records()
        if record.metadata.get(ANALYST_LEARNING_FLAG)
    ]
    buckets: dict[str, list[AgentLearningRecord]] = defaultdict(list)
    for record in records:
        profile = str(record.metadata.get("profile") or record.agent_name or "unknown")
        buckets[profile].append(record)
    return {profile: _summarize_outcomes(items) for profile, items in sorted(buckets.items())}


def _summarize_outcomes(records: list[AgentLearningRecord]) -> dict[str, Any]:
    completed = [record for record in records if record.outcome_label is not None]
    counts = Counter(record.outcome_label or "pending" for record in records)
    returns = [record.realized_return for record in completed if record.realized_return is not None]
    completed_count = len(completed)
    return {
        "record_count": len(records),
        "completed_count": completed_count,
        "pending_count": len(records) - completed_count,
        "outcome_counts": dict(sorted(counts.items())),
        "hit_rate": counts.get("hit", 0) / completed_count if completed_count else None,
        "miss_rate": counts.get("miss", 0) / completed_count if completed_count else None,
        "avg_realized_return": mean(returns) if returns else None,
    }


def _empty_outcomes() -> dict[str, Any]:
    return {
        "record_count": 0,
        "completed_count": 0,
        "pending_count": 0,
        "outcome_counts": {},
        "hit_rate": None,
        "miss_rate": None,
        "avg_realized_return": None,
    }


def _calibrate_profile(
    profile: str,
    scorecard: dict[str, Any],
    outcomes: dict[str, Any],
    context_summary: dict[str, Any],
    min_profile_runs: int,
    min_completed_outcomes: int,
    min_hit_rate: float,
    max_miss_rate: float,
    require_scorecard_ready: bool,
) -> dict[str, Any]:
    blockers: list[str] = []
    cautions: list[str] = []
    completed_runs = int(scorecard.get("completed_count") or 0)
    completed_outcomes = int(outcomes.get("completed_count") or 0)
    hit_rate = outcomes.get("hit_rate")
    miss_rate = outcomes.get("miss_rate")

    if completed_runs < min_profile_runs:
        blockers.append(f"Needs at least {min_profile_runs} completed profile runs.")
    if require_scorecard_ready and scorecard.get("activation_status") != "ready_to_activate":
        blockers.append("Profile scorecard is not ready_to_activate.")
    if completed_outcomes < min_completed_outcomes:
        blockers.append(f"Needs at least {min_completed_outcomes} completed outcomes.")
    if hit_rate is None or hit_rate < min_hit_rate:
        blockers.append(f"Needs hit_rate >= {min_hit_rate}.")
    if miss_rate is not None and miss_rate > max_miss_rate:
        blockers.append(f"Needs miss_rate <= {max_miss_rate}.")
    if _profile_has_weak_context(profile, context_summary):
        cautions.append("Context performance contains weak contexts for this profile/agent.")

    if blockers:
        status = "blocked" if completed_outcomes == 0 or not scorecard else "keep_candidate"
        suggested_delta = 0.0 if status == "blocked" else -0.02 if miss_rate and hit_rate and miss_rate > hit_rate else 0.0
        recommendation = "Do not change profile defaults or consensus weights. " + " ".join(blockers)
    elif cautions:
        status = "ready_with_caution"
        suggested_delta = 0.02
        recommendation = "Ready for human review only; address context cautions before increasing weights."
    else:
        status = "ready_for_review"
        suggested_delta = 0.05
        recommendation = "Eligible for human-reviewed calibration proposal; do not auto-apply."

    return {
        "profile": profile,
        "calibration_status": status,
        "suggested_weight_delta": suggested_delta,
        "scorecard": scorecard,
        "outcomes": outcomes,
        "blockers": blockers,
        "cautions": cautions,
        "recommendation": recommendation,
    }


def _profile_has_weak_context(profile: str, context_summary: dict[str, Any]) -> bool:
    for item in context_summary.get("weak_contexts", []):
        if item.get("agent_name") == profile:
            return True
    return False


def _summary(calibration: dict[str, dict[str, Any]]) -> dict[str, Any]:
    ready = [
        profile
        for profile, card in calibration.items()
        if card["calibration_status"] in {"ready_for_review", "ready_with_caution"}
    ]
    keep = [profile for profile, card in calibration.items() if card["calibration_status"] == "keep_candidate"]
    blocked = [profile for profile, card in calibration.items() if card["calibration_status"] == "blocked"]
    return {
        "profile_count": len(calibration),
        "ready_for_review_profiles": ready,
        "keep_candidate_profiles": keep,
        "blocked_profiles": blocked,
        "status_counts": dict(sorted(Counter(card["calibration_status"] for card in calibration.values()).items())),
    }


def _recommendations(calibration: dict[str, dict[str, Any]]) -> list[str]:
    if not calibration:
        return ["No analyst profiles or outcomes found; run evidence/profile/learning loops first."]
    ready = [profile for profile, card in calibration.items() if card["calibration_status"] == "ready_for_review"]
    if ready:
        return [f"Prepare a human-reviewed calibration proposal for: {', '.join(ready)}. Do not auto-apply weights."]
    cautious = [profile for profile, card in calibration.items() if card["calibration_status"] == "ready_with_caution"]
    if cautious:
        return [f"Profiles need review before calibration: {', '.join(cautious)}."]
    return ["Keep current analyst defaults; collect more completed profile runs and evaluated outcomes."]


def _run_id(prefix: str) -> str:
    return f"{prefix}_{utc_now_iso().replace(':', '').replace('-', '').replace('.', '_')}"
