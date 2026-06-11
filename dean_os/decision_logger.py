from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from dean_os.schemas import AnalyticalReport, ConsensusDecision, PipelineReport
from dean_os.utils import json_ready, sha256_json


class DecisionLogger:
    def __init__(self, log_path: str | Path = "logs/dean_os/decisions.jsonl"):
        self.log_path = Path(log_path)

    def log(
        self,
        decision: ConsensusDecision,
        pipeline_reports: list[PipelineReport],
        analytical_reports: list[AnalyticalReport],
        input_snapshot: dict[str, Any],
        config: dict[str, Any],
        pipeline_git_commit: str = "unknown",
    ) -> str:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        entry = {
            "event_type": "decision",
            "decision_id": decision.decision_id,
            "timestamp": decision.timestamp,
            "input_hash": sha256_json(input_snapshot),
            "config_hash": sha256_json(config),
            "pipeline_commit": pipeline_git_commit,
            "input_snapshot": json_ready(input_snapshot),
            "agent_report_hashes": {
                report.agent_name: sha256_json(report) for report in [*pipeline_reports, *analytical_reports]
            },
            "final_decision": decision.decision,
            "final_score": decision.final_score,
            "decision": decision.model_dump(mode="json"),
        }
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, sort_keys=True, ensure_ascii=True) + "\n")
        return decision.decision_id
