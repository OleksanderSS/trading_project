from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from dean_os.schemas import utc_now_iso
from dean_os.utils import json_ready


DEFAULT_REPAIR_REPORT = "reports/dean_os/replay_price_artifact_repair_current/latest.json"
DEFAULT_PRICE_QUALITY_REPORT = "reports/dean_os/replay_price_quality_investigation_repaired_artifact_only_v2/latest.json"
DEFAULT_REPLAY_BATCH_REPORT = "reports/dean_os/historical_replay_batch_repaired_202603_202604/latest.json"
DEFAULT_RESEARCH_BATCH_REPORT = "reports/dean_os/historical_research_replay_batch_repaired_202603_202604/latest.json"


class ReplayCalibrationReadinessGate:
    """Read-only gate before using replay exams for analyst calibration."""

    def __init__(self, output_dir: str | Path = "reports/dean_os/replay_calibration_readiness_gate"):
        self.output_dir = Path(output_dir)

    def build(
        self,
        repair_report_path: str | Path | None = DEFAULT_REPAIR_REPORT,
        price_quality_report_path: str | Path | None = DEFAULT_PRICE_QUALITY_REPORT,
        replay_batch_path: str | Path | None = DEFAULT_REPLAY_BATCH_REPORT,
        research_batch_path: str | Path | None = DEFAULT_RESEARCH_BATCH_REPORT,
        min_clean_replay_runs: int = 10,
        min_clean_research_runs: int = 10,
        max_quality_blocked_runs: int = 0,
        max_price_warning_records: int = 0,
        max_weak_evidence_runs: int = 0,
        min_directional_research_ratio: float = 0.25,
        save: bool = True,
    ) -> dict[str, Any]:
        inputs = {
            "repair_report": _load_optional_json(repair_report_path),
            "price_quality_report": _load_optional_json(price_quality_report_path),
            "replay_batch": _load_optional_json(replay_batch_path),
            "research_batch": _load_optional_json(research_batch_path),
        }
        thresholds = {
            "min_clean_replay_runs": min_clean_replay_runs,
            "min_clean_research_runs": min_clean_research_runs,
            "max_quality_blocked_runs": max_quality_blocked_runs,
            "max_price_warning_records": max_price_warning_records,
            "max_weak_evidence_runs": max_weak_evidence_runs,
            "min_directional_research_ratio": min_directional_research_ratio,
        }
        checks = {
            "price_quality": _price_quality_check(inputs, thresholds),
            "replay_sample": _replay_sample_check(inputs, thresholds),
            "research_sample": _research_sample_check(inputs, thresholds),
            "evidence_coverage": _evidence_coverage_check(inputs, thresholds),
            "research_directionality": _research_directionality_check(inputs, thresholds),
        }
        gate = _gate(checks)
        payload = {
            "run_id": _run_id("replay_calibration_readiness_gate"),
            "created_at": utc_now_iso(),
            "mode": "replay_calibration_readiness_gate",
            "inputs": {
                "repair_report_path": str(repair_report_path) if repair_report_path else None,
                "price_quality_report_path": str(price_quality_report_path) if price_quality_report_path else None,
                "replay_batch_path": str(replay_batch_path) if replay_batch_path else None,
                "research_batch_path": str(research_batch_path) if research_batch_path else None,
                **thresholds,
            },
            "summary": {
                "readiness_status": gate["status"],
                "can_create_calibration_review_packet": gate["can_create_calibration_review_packet"],
                "can_write_learning_memory": False,
                "can_change_analyst_weights": False,
                "blocker_count": len(gate["blockers"]),
                "caution_count": len(gate["cautions"]),
                "next_action": gate["next_action"],
            },
            "gate": gate,
            "checks": checks,
            "source_reports": _source_report_summary(inputs),
            "commands": _commands(inputs, gate),
            "safety": {
                "read_only": True,
                "data_mutation_performed": False,
                "collector_run_performed": False,
                "pipeline_run_performed": False,
                "learning_write_performed": False,
                "operation_proposal_created": False,
                "config_write_performed": False,
                "broker_access_performed": False,
            },
            "recommendations": _recommendations(gate, checks),
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
        rendered_md = render_replay_calibration_readiness_gate_markdown(payload)
        json_path.write_text(rendered_json, encoding="utf-8")
        latest_json.write_text(rendered_json, encoding="utf-8")
        md_path.write_text(rendered_md, encoding="utf-8")
        latest_md.write_text(rendered_md, encoding="utf-8")
        return json_path, md_path


def render_replay_calibration_readiness_gate_markdown(payload: dict[str, Any]) -> str:
    summary = payload.get("summary", {})
    gate = payload.get("gate", {})
    lines = [
        "# DEAN-OS Replay Calibration Readiness Gate",
        "",
        f"- Run ID: `{payload.get('run_id')}`",
        f"- Status: `{summary.get('readiness_status')}`",
        f"- Next action: `{summary.get('next_action')}`",
        f"- Can create calibration review packet: {summary.get('can_create_calibration_review_packet')}",
        f"- Can change analyst weights: {summary.get('can_change_analyst_weights')}",
        "",
        "## Blockers",
        "",
    ]
    blockers = gate.get("blockers", [])
    lines.extend(f"- `{item.get('check')}`: {item.get('reason')}" for item in blockers) if blockers else lines.append("- None.")
    lines.extend(["", "## Cautions", ""])
    cautions = gate.get("cautions", [])
    lines.extend(f"- `{item.get('check')}`: {item.get('reason')}" for item in cautions) if cautions else lines.append("- None.")
    lines.extend(["", "## Checks", ""])
    for name, check in payload.get("checks", {}).items():
        metrics = check.get("metrics", {})
        lines.append(f"- `{name}`: {check.get('status')} metrics={metrics}")
    lines.extend(["", "## Recommendations", ""])
    lines.extend(f"- {item}" for item in payload.get("recommendations", []))
    return "\n".join(lines).strip() + "\n"


def _price_quality_check(inputs: dict[str, dict[str, Any]], thresholds: dict[str, Any]) -> dict[str, Any]:
    repair = inputs["repair_report"].get("payload", {})
    price = inputs["price_quality_report"].get("payload", {})
    repair_summary = repair.get("summary", {}) if isinstance(repair, dict) else {}
    price_summary = price.get("summary", {}) if isinstance(price, dict) else {}
    artifact_diagnostics = price.get("artifact_diagnostics", []) if isinstance(price.get("artifact_diagnostics"), list) else []
    artifact_warning_count = sum(len(item.get("warnings", [])) for item in artifact_diagnostics if isinstance(item, dict))
    warning_records = int(price_summary.get("warning_record_count") or 0)
    extreme_warnings = int(price_summary.get("extreme_benchmark_warning_count") or 0)
    candidate_warnings = int(repair_summary.get("candidate_quality_warning_count") or 0)
    metrics = {
        "repair_status": repair_summary.get("repair_status"),
        "candidate_quality_warning_count": candidate_warnings,
        "investigation_status": price_summary.get("investigation_status"),
        "warning_record_count": warning_records,
        "extreme_benchmark_warning_count": extreme_warnings,
        "artifact_warning_count": artifact_warning_count,
        "artifact_path": repair.get("artifact", {}).get("path") if isinstance(repair.get("artifact"), dict) else None,
    }
    if not inputs["price_quality_report"]["loaded"]:
        return _check("blocked", metrics, "Price-quality report is missing or unreadable.")
    if candidate_warnings > 0 or warning_records > int(thresholds["max_price_warning_records"]) or extreme_warnings > 0 or artifact_warning_count > 0:
        return _check("blocked", metrics, "Repaired artifact still has price-quality warnings.")
    if price_summary.get("investigation_status") != "clear":
        return _check("blocked", metrics, "Artifact-only price investigation is not clear.")
    if not inputs["repair_report"]["loaded"]:
        return _check("caution", metrics, "Repair report is missing; artifact quality is clear but provenance is incomplete.")
    return _check("pass", metrics, "Candidate repaired artifact passed price-quality checks.")


def _replay_sample_check(inputs: dict[str, dict[str, Any]], thresholds: dict[str, Any]) -> dict[str, Any]:
    replay = inputs["replay_batch"].get("payload", {})
    summary = replay.get("summary", {}) if isinstance(replay.get("summary"), dict) else {}
    clear_runs = int(summary.get("clear_evaluated_runs") or 0)
    quality_blocked = int(summary.get("quality_blocked_runs") or 0)
    metrics = {
        "total_runs": int(summary.get("total_runs") or 0),
        "evaluated_runs": int(summary.get("evaluated_runs") or 0),
        "clear_evaluated_runs": clear_runs,
        "quality_blocked_runs": quality_blocked,
        "clear_hit_rate": summary.get("clear_hit_rate"),
        "average_realized_return": summary.get("clear_average_realized_return"),
    }
    if not inputs["replay_batch"]["loaded"]:
        return _check("blocked", metrics, "Historical replay batch report is missing or unreadable.")
    if quality_blocked > int(thresholds["max_quality_blocked_runs"]):
        return _check("blocked", metrics, "Historical replay batch still has quality-blocked runs.")
    if clear_runs < int(thresholds["min_clean_replay_runs"]):
        return _check("blocked", metrics, "Clean historical replay sample is too small for calibration.")
    return _check("pass", metrics, "Historical replay sample is large enough and quality-clean.")


def _research_sample_check(inputs: dict[str, dict[str, Any]], thresholds: dict[str, Any]) -> dict[str, Any]:
    research = inputs["research_batch"].get("payload", {})
    summary = research.get("summary", {}) if isinstance(research.get("summary"), dict) else {}
    clear_runs = int(summary.get("clear_evaluated_runs") or 0)
    quality_blocked = int(summary.get("quality_blocked_runs") or 0)
    metrics = {
        "total_runs": int(summary.get("total_runs") or 0),
        "evaluated_runs": int(summary.get("evaluated_runs") or 0),
        "clear_evaluated_runs": clear_runs,
        "quality_blocked_runs": quality_blocked,
        "clear_hit_rate": summary.get("clear_hit_rate"),
        "research_stance_counts": summary.get("research_stance_counts", {}),
    }
    if not inputs["research_batch"]["loaded"]:
        return _check("blocked", metrics, "Historical research replay batch report is missing or unreadable.")
    if quality_blocked > int(thresholds["max_quality_blocked_runs"]):
        return _check("blocked", metrics, "Historical research replay batch still has quality-blocked runs.")
    if clear_runs < int(thresholds["min_clean_research_runs"]):
        return _check("blocked", metrics, "Clean historical research replay sample is too small for analyst calibration.")
    return _check("pass", metrics, "Historical research replay sample is large enough and quality-clean.")


def _evidence_coverage_check(inputs: dict[str, dict[str, Any]], thresholds: dict[str, Any]) -> dict[str, Any]:
    research = inputs["research_batch"].get("payload", {})
    summary = research.get("summary", {}) if isinstance(research.get("summary"), dict) else {}
    weak = int(summary.get("weak_evidence_runs") or 0)
    total = int(summary.get("total_runs") or 0)
    evidence_quality_counts = summary.get("evidence_quality_counts", {})
    metrics = {
        "weak_evidence_runs": weak,
        "total_runs": total,
        "evidence_quality_counts": evidence_quality_counts,
        "strong_evidence_ratio": _safe_ratio(int(evidence_quality_counts.get("strong", 0)) if isinstance(evidence_quality_counts, dict) else 0, total),
    }
    if not inputs["research_batch"]["loaded"]:
        return _check("blocked", metrics, "Research replay report is missing, so evidence coverage cannot be verified.")
    if weak > int(thresholds["max_weak_evidence_runs"]):
        return _check("blocked", metrics, "At least one research replay has weak or partial evidence coverage.")
    return _check("pass", metrics, "Research replay evidence coverage is acceptable.")


def _research_directionality_check(inputs: dict[str, dict[str, Any]], thresholds: dict[str, Any]) -> dict[str, Any]:
    research = inputs["research_batch"].get("payload", {})
    summary = research.get("summary", {}) if isinstance(research.get("summary"), dict) else {}
    runs = research.get("runs", []) if isinstance(research.get("runs"), list) else []
    total = len(runs)
    directional = [
        run
        for run in runs
        if run.get("research_expected_direction") in {"bullish", "bearish"} or run.get("research_price_agreement") == "confirmed"
    ]
    ratio = _safe_ratio(len(directional), total)
    metrics = {
        "total_research_runs": total,
        "directional_or_confirmed_runs": len(directional),
        "directional_ratio": ratio,
        "research_inconclusive_runs": int(summary.get("research_inconclusive_runs") or 0),
        "exam_verdict_counts": summary.get("exam_verdict_counts", {}),
    }
    if not inputs["research_batch"]["loaded"]:
        return _check("blocked", metrics, "Research replay report is missing, so directionality cannot be checked.")
    if total == 0:
        return _check("blocked", metrics, "No research replay runs are available.")
    if ratio < float(thresholds["min_directional_research_ratio"]):
        return _check("caution", metrics, "Research replay is mostly neutral/inconclusive; calibrate conservatism before directional skill.")
    return _check("pass", metrics, "Research replay includes enough directional or confirmed runs for review.")


def _gate(checks: dict[str, dict[str, Any]]) -> dict[str, Any]:
    blockers = [
        {"check": name, "reason": check["reason"], "metrics": check["metrics"]}
        for name, check in checks.items()
        if check["status"] == "blocked"
    ]
    cautions = [
        {"check": name, "reason": check["reason"], "metrics": check["metrics"]}
        for name, check in checks.items()
        if check["status"] == "caution"
    ]
    if any(item["check"] == "price_quality" for item in blockers):
        status = "price_quality_blocked"
        next_action = "repair_or_refresh_price_artifact"
    elif any(item["check"] == "replay_sample" for item in blockers):
        status = "need_more_replay_samples"
        next_action = "expand_historical_replay_batch"
    elif any(item["check"] == "research_sample" for item in blockers):
        status = "need_more_research_replay_samples"
        next_action = "expand_historical_research_replay_batch"
    elif any(item["check"] == "evidence_coverage" for item in blockers):
        status = "need_evidence_backfill"
        next_action = "backfill_research_evidence"
    elif blockers:
        status = "blocked"
        next_action = "resolve_blockers"
    elif cautions:
        status = "ready_for_manual_review_with_caution"
        next_action = "manual_review_replay_calibration_packet"
    else:
        status = "ready_for_manual_review"
        next_action = "manual_review_replay_calibration_packet"
    return {
        "status": status,
        "can_create_calibration_review_packet": not blockers,
        "can_write_learning_memory": False,
        "can_change_analyst_weights": False,
        "blockers": blockers,
        "cautions": cautions,
        "passed_checks": [name for name, check in checks.items() if check["status"] == "pass"],
        "next_action": next_action,
        "status_counts": dict(sorted(Counter(check["status"] for check in checks.values()).items())),
    }


def _commands(inputs: dict[str, dict[str, Any]], gate: dict[str, Any]) -> dict[str, str | None]:
    artifact_path = _artifact_path(inputs)
    replay_command = None
    research_command = None
    if artifact_path:
        replay_command = (
            "python run_agent_historical_replay_batch.py "
            f"{artifact_path} --tickers AMD NVDA MSFT AAPL TSM QQQ SPY "
            "--start-as-of 2025-09-01T00:00:00+00:00 --end-as-of 2026-03-01T00:00:00+00:00 "
            "--step-days 14 --lookback-days 180 --horizon-days 30 60 "
            "--news-data data\\colab\\backup_20260510_153551\\stage2_news_20260505_151233.parquet "
            "--macro-data data\\colab\\backup_20260510_153551\\stage2_macro_20260507_191104.parquet "
            "--output-dir reports\\dean_os\\historical_replay_batch_repaired_expanded"
        )
        research_command = (
            "python run_agent_historical_research_replay_batch.py "
            f"{artifact_path} --tickers AMD NVDA MSFT AAPL TSM QQQ SPY "
            "--start-as-of 2025-09-01T00:00:00+00:00 --end-as-of 2026-03-01T00:00:00+00:00 "
            "--step-days 30 --lookback-days 180 --horizon-days 30 "
            "--news-data data\\colab\\backup_20260510_153551\\stage2_news_20260505_151233.parquet "
            "--macro-data data\\colab\\backup_20260510_153551\\stage2_macro_20260507_191104.parquet "
            "--tags historical_replay ai_cycle repaired_price_artifact expanded_batch "
            "--output-dir reports\\dean_os\\historical_research_replay_batch_repaired_expanded"
        )
    return {
        "rerun_artifact_quality_only": (
            f"python run_agent_replay_price_quality_investigation.py --artifact-only --price-data {artifact_path} "
            "--benchmark-ticker SPY --output-dir reports\\dean_os\\replay_price_quality_investigation_repaired_artifact_only"
            if artifact_path
            else None
        ),
        "expand_historical_replay_batch": replay_command,
        "expand_historical_research_replay_batch": research_command,
        "next_recommended_command": {
            "repair_or_refresh_price_artifact": (
                "python run_agent_replay_price_artifact_repair.py data\\colab\\backup_20260510_153551\\stage2_prices_1d_20260505_151233.parquet "
                "--tickers AMD NVDA MSFT AAPL TSM QQQ SPY --benchmark-ticker SPY --write-artifact"
            ),
            "expand_historical_replay_batch": replay_command,
            "expand_historical_research_replay_batch": research_command,
            "backfill_research_evidence": "python run_agent_evidence_gap_plan.py --inbox-json reports\\dean_os\\analyst_review_inbox\\latest.json",
            "manual_review_replay_calibration_packet": None,
            "resolve_blockers": None,
        }.get(gate["next_action"]),
    }


def _recommendations(gate: dict[str, Any], checks: dict[str, dict[str, Any]]) -> list[str]:
    status = gate["status"]
    if status == "price_quality_blocked":
        return ["Do not use replay hit/miss for calibration; repair or refresh the price artifact first."]
    if status == "need_more_replay_samples":
        return ["Expand historical replay batch on the repaired artifact before judging repeatability."]
    if status == "need_more_research_replay_samples":
        return ["Expand historical research replay batch before analyst calibration."]
    if status == "need_evidence_backfill":
        return ["Backfill or narrow evidence coverage before using research replay for analyst calibration."]
    if status == "ready_for_manual_review_with_caution":
        return [
            "Create a manual calibration review packet, but treat neutral/inconclusive research as a calibration target.",
            "Do not auto-promote weights; review whether the analyst is correctly conservative.",
        ]
    return [
        "Replay calibration evidence is ready for manual review.",
        "The next step should be a review packet, not automatic learning, config writes, or live/paper trade creation.",
    ]


def _source_report_summary(inputs: dict[str, dict[str, Any]]) -> dict[str, Any]:
    return {
        name: {
            "path": item.get("path"),
            "loaded": item.get("loaded"),
            "mode": item.get("payload", {}).get("mode") if isinstance(item.get("payload"), dict) else None,
            "error": item.get("error"),
        }
        for name, item in inputs.items()
    }


def _artifact_path(inputs: dict[str, dict[str, Any]]) -> str | None:
    repair = inputs["repair_report"].get("payload", {})
    if isinstance(repair.get("artifact"), dict) and repair["artifact"].get("path"):
        return str(repair["artifact"]["path"])
    price = inputs["price_quality_report"].get("payload", {})
    diagnostics = price.get("artifact_diagnostics", []) if isinstance(price.get("artifact_diagnostics"), list) else []
    for item in diagnostics:
        if isinstance(item, dict) and item.get("path"):
            return str(item["path"])
    return None


def _load_optional_json(path: str | Path | None) -> dict[str, Any]:
    if path is None:
        return {"path": None, "loaded": False, "payload": {}, "error": "not_provided"}
    resolved = Path(path)
    if not resolved.exists():
        return {"path": str(resolved), "loaded": False, "payload": {}, "error": "missing"}
    try:
        payload = json.loads(resolved.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return {"path": str(resolved), "loaded": False, "payload": {}, "error": f"invalid_json: {exc}"}
    return {"path": str(resolved), "loaded": True, "payload": payload if isinstance(payload, dict) else {"items": payload}}


def _check(status: str, metrics: dict[str, Any], reason: str) -> dict[str, Any]:
    return {"status": status, "metrics": json_ready(metrics), "reason": reason}


def _safe_ratio(numerator: int, denominator: int) -> float | None:
    if denominator <= 0:
        return None
    return round(numerator / denominator, 6)


def _run_id(prefix: str) -> str:
    return f"{prefix}_{utc_now_iso().replace(':', '').replace('-', '').replace('.', '_')}"
