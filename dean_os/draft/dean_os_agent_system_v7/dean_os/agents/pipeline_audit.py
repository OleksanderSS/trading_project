from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from dean_os.base import BaseAgent
from dean_os.schemas import MarketContext, PipelineReport


class PipelineAuditAgent(BaseAgent):
    version = "0.1.0"
    branch = "pipeline"

    async def run(self, context: MarketContext) -> PipelineReport:
        findings_path = self._project_path(self.config.get("findings_path", "audit_reports/findings.json"))
        max_age_hours = float(self.config.get("max_findings_age_hours", 24))
        blocker_severities = set(self.config.get("blocker_severities", ["P0"]))
        caution_severities = set(self.config.get("caution_severities", ["P1"]))

        findings = self._load_findings(findings_path)
        severity_counts = self._severity_counts(findings)
        age_hours = self._age_hours(findings_path)
        evidence = [
            self.evidence("file", str(findings_path), "age_hours", round(age_hours, 3)),
            self.evidence("audit_finding", str(findings_path), "severity_counts", severity_counts),
        ]

        stale = age_hours > max_age_hours
        blockers = sum(severity_counts.get(severity, 0) for severity in blocker_severities)
        cautions = sum(severity_counts.get(severity, 0) for severity in caution_severities)

        if stale:
            verdict = "blocked"
            reasons = [f"Audit findings are stale: {age_hours:.1f}h old"]
            risks = ["Pipeline could run against outdated safety findings"]
            signal_strength = -1.0
            confidence = 1.0
        elif blockers:
            verdict = "blocked"
            reasons = [f"Unresolved blocker findings: {blockers}"]
            risks = ["Hard audit blockers must be resolved before trusting the pipeline"]
            signal_strength = -1.0
            confidence = 1.0
        elif cautions:
            verdict = "caution"
            reasons = [f"Unresolved caution findings: {cautions}"]
            risks = ["Pipeline can run, but results should stay in review/paper mode"]
            signal_strength = -0.2
            confidence = 0.85
        else:
            verdict = "clear"
            reasons = ["No blocker audit findings detected"]
            risks = []
            signal_strength = 0.5
            confidence = 0.9

        return PipelineReport(
            agent_name=self.name,
            agent_version=self.version,
            verdict=verdict,
            confidence=confidence,
            data_quality_score=1.0 if findings else 0.5,
            signal_strength=signal_strength,
            reasons=reasons,
            risks=risks,
            blind_spots=["Audit files only reflect the last audit run"],
            evidence=evidence,
            input_hash=self.context_hash(context),
            metrics_snapshot={
                "finding_count": len(findings),
                "severity_counts": severity_counts,
                "age_hours": age_hours,
            },
        )

    def check_prerequisites(self, context: MarketContext) -> bool:
        findings_path = self._project_path(self.config.get("findings_path", "audit_reports/findings.json"))
        return findings_path.exists()

    def _project_path(self, relative_or_absolute: str) -> Path:
        path = Path(relative_or_absolute)
        if path.is_absolute():
            return path
        return Path(self.config.get("project_root", ".")).resolve() / path

    def _load_findings(self, path: Path) -> list[dict[str, Any]]:
        with path.open("r", encoding="utf-8") as handle:
            raw = json.load(handle)
        if isinstance(raw, list):
            return [item for item in raw if isinstance(item, dict)]
        if isinstance(raw, dict):
            findings = raw.get("findings", [])
            return [item for item in findings if isinstance(item, dict)]
        return []

    def _severity_counts(self, findings: list[dict[str, Any]]) -> dict[str, int]:
        counts: dict[str, int] = {}
        for finding in findings:
            severity = str(finding.get("severity", "unknown"))
            counts[severity] = counts.get(severity, 0) + 1
        return counts

    def _age_hours(self, path: Path) -> float:
        modified_at = datetime.fromtimestamp(path.stat().st_mtime, tz=UTC)
        return (datetime.now(UTC) - modified_at).total_seconds() / 3600
