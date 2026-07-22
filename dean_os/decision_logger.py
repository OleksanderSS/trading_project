from __future__ import annotations

import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from dean_os.schemas import AnalyticalReport, ConsensusDecision, PipelineReport
from dean_os.utils import json_ready, sha256_json

LOG_CONTRACT = "dean_decision_audit_v2"


def _git_commit() -> str:
    """Return short git HEAD commit hash, or 'unknown' if unavailable."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=2,
        )
        return result.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


class DecisionLogger:
    """Cryptographically-verifiable decision audit trail.

    Every decision produces one JSONL line containing:
    - SHA-256 of input context snapshot
    - SHA-256 of each agent's report (decision_chain)
    - SHA-256 of engine config
    - Git commit hash of current codebase
    - Full decision dump

    These hashes form an auditable chain: you can verify at any later
    point that a specific model/config/data combination produced a given
    decision, and trace which agents drove it.
    """

    def __init__(self, log_path: str | Path = "logs/dean_os/decisions.jsonl"):
        self.log_path = Path(log_path)

    def log(
        self,
        decision: ConsensusDecision,
        pipeline_reports: list[PipelineReport],
        analytical_reports: list[AnalyticalReport],
        input_snapshot: dict[str, Any],
        config: dict[str, Any],
        pipeline_git_commit: str | None = None,
    ) -> str:
        """Write a fully-hashed decision entry to the audit log.

        Returns the decision_id for downstream reference.
        """
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        git_commit = pipeline_git_commit or _git_commit()
        all_reports: list[PipelineReport | AnalyticalReport] = [*pipeline_reports, *analytical_reports]

        # Per-agent SHA-256 chain: agent_name → sha256(full report JSON)
        decision_chain = [
            {
                "agent": r.agent_name,
                "branch": r.branch,
                "verdict": r.verdict,
                "report_sha256": sha256_json(r),
            }
            for r in all_reports
        ]
        agent_report_hashes = {item["agent"]: item["report_sha256"] for item in decision_chain}

        input_hash = sha256_json(input_snapshot)
        config_hash = sha256_json(config)

        # Top-level entry hash (covers all inputs to this decision)
        entry_payload_hash = sha256_json({
            "input_hash": input_hash,
            "config_hash": config_hash,
            "git_commit": git_commit,
            "agent_report_hashes": agent_report_hashes,
            "final_decision": decision.decision,
            "final_score": decision.final_score,
        })

        world_state_summary = None
        if isinstance(input_snapshot, dict):
            metadata = input_snapshot.get("metadata", {})
            world_state_summary = metadata.get("world_state_summary")

        entry = {
            "_contract": LOG_CONTRACT,
            "event_type": "decision",
            "decision_id": decision.decision_id,
            "timestamp": decision.timestamp or datetime.now(UTC).isoformat(),
            "entry_sha256": entry_payload_hash,
            "git_commit": git_commit,
            "input_hash": input_hash,
            "config_hash": config_hash,
            "world_state_summary": world_state_summary,
            # Human-readable summary
            "summary": (
                f"{decision.decision} | score={decision.final_score:.3f} "
                f"| conf={decision.confidence:.2f} "
                f"| agents={len(all_reports)}"
                + (f" | KILL_SWITCH" if getattr(decision, 'anxiety_kill_switch_triggered', False) else "")
            ),
            # Full decision_chain for verification
            "decision_chain": decision_chain,
            "agent_report_hashes": agent_report_hashes,
            # Context snapshot (excludes large DataFrames)
            "input_snapshot": json_ready(input_snapshot),
            "config": json_ready(config),
            # Full decision
            "final_decision": decision.decision,
            "final_score": decision.final_score,
            "confidence": decision.confidence,
            "decision": decision.model_dump(mode="json"),
        }
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, sort_keys=True, ensure_ascii=True) + "\n")
        return decision.decision_id

    @property
    def log_file(self) -> Path:
        return self.log_path
