from __future__ import annotations

import asyncio
import hashlib
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from dean_os.agents.domain_analyst import DomainAnalystAgent
from dean_os.draft.dean_os_agent_system_v7.dean_os.artifact_writer import ReviewArtifactWriter
from dean_os.schemas import MarketContext


class DomainAnalystReviewRun:
    """Run a generic domain analyst from verified saved evidence, review-only."""

    contract = "dean_domain_analyst_review_run_v1"

    def __init__(
        self,
        output_dir: str | Path = "reports/dean_os/domain_analyst_review_current",
    ) -> None:
        self.output_dir = Path(output_dir)

    def run(
        self,
        *,
        domain_id: str,
        as_of: str,
        tickers: list[str],
        producer_artifact_paths: dict[str, str] | None = None,
        pipeline_context_artifact_path: str | None = None,
        research_corpus_path: str | None = None,
        research_query: str | None = None,
        research_top_k: int = 20,
        horizon_days: int | None = None,
    ) -> dict[str, Any]:
        config: dict[str, Any] = {
            "domain_id": domain_id,
            "tickers": tickers,
            "producer_artifact_paths": producer_artifact_paths or {},
        }
        if pipeline_context_artifact_path:
            config["pipeline_context_artifact_path"] = (
                pipeline_context_artifact_path
            )
        if research_corpus_path:
            config["research_corpus_path"] = research_corpus_path
            config["research_top_k"] = research_top_k
            if research_query:
                config["research_query"] = research_query
        if horizon_days is not None:
            config["horizon_days"] = horizon_days

        report = asyncio.run(
            DomainAnalystAgent(f"{domain_id}_analyst", config).run(
                MarketContext(as_of=as_of, tickers=tickers)
            )
        )
        created_at = datetime.now(UTC).isoformat()
        run_id = "domain_analyst_review_" + created_at.replace(
            ":", ""
        ).replace("+00:00", "Z")
        inputs = {
            "as_of": as_of,
            "tickers": sorted(set(tickers)),
            "horizon_days": horizon_days,
            "producer_artifacts": {
                key: _artifact_ref(value)
                for key, value in sorted((producer_artifact_paths or {}).items())
            },
            "pipeline_context_artifact": (
                _artifact_ref(pipeline_context_artifact_path)
                if pipeline_context_artifact_path
                else None
            ),
            "research_corpus": (
                _artifact_ref(research_corpus_path)
                if research_corpus_path
                else None
            ),
            "research_query": research_query,
            "research_top_k": research_top_k if research_corpus_path else None,
        }
        payload: dict[str, Any] = {
            "run_id": run_id,
            "created_at": created_at,
            "mode": "domain_analyst_review_run",
            "contract": self.contract,
            "domain_id": domain_id,
            "status": f"domain_analyst_review_{report.verdict}",
            "inputs": inputs,
            "summary": {
                "verdict": report.verdict,
                "recommendation": report.metrics_snapshot.get("recommendation"),
                "stance": report.metrics_snapshot.get("stance"),
                "confidence": report.confidence,
                "evidence_count": report.metrics_snapshot.get("evidence_count", 0),
                "lens_count": report.metrics_snapshot.get("lens_count", 0),
                "hypothesis_count": report.metrics_snapshot.get(
                    "hypothesis_count", 0
                ),
                "evidence_gap_count": report.metrics_snapshot.get(
                    "evidence_gap_count", 0
                ),
                "can_trade": False,
            },
            "agent_report": report.model_dump(mode="json"),
            "safety": {
                "review_only": True,
                "network_access_performed": False,
                "pipeline_run_performed": False,
                "training_run_performed": False,
                "tuning_run_performed": False,
                "learning_write_performed": False,
                "production_config_write_performed": False,
                "broker_access_performed": False,
                "live_execution_performed": False,
                "can_trade": False,
            },
        }
        payload["saved_paths"] = ReviewArtifactWriter(self.output_dir).write(
            payload=payload,
            markdown=_render_markdown(payload),
            run_id=run_id,
        )
        return payload


def _artifact_ref(value: str | Path) -> dict[str, str]:
    path = Path(value)
    if path.is_dir():
        path = path / "latest.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return {"path": str(path), "sha256": digest}


def _render_markdown(payload: dict[str, Any]) -> str:
    summary = payload["summary"]
    missing = payload["agent_report"].get("reasons") or []
    lines = [
        "# Domain Analyst Review",
        "",
        f"- Domain: `{payload['domain_id']}`",
        f"- As of: `{payload['inputs']['as_of']}`",
        f"- Verdict: `{summary['verdict']}`",
        f"- Recommendation: `{summary['recommendation']}`",
        f"- Evidence: `{summary['evidence_count']}`",
        f"- Lenses: `{summary['lens_count']}`",
        "- Can trade: `false`",
        "",
        "## Reasons",
        "",
    ]
    lines.extend(f"- {item}" for item in missing)
    return "\n".join(lines) + "\n"


__all__ = ["DomainAnalystReviewRun"]
