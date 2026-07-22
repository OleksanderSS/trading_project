"""Contract tests for the standalone DomainAnalystAgent."""
from __future__ import annotations

import asyncio
import json

from dean_os.agents.domain_analyst import DomainAnalystAgent
from dean_os.schemas import MarketContext, PipelineReport


CONFIG = {
    "domain_id": "semiconductor_ai_infrastructure",
    "horizon_days": 180,
}


def _run(agent: DomainAnalystAgent, context: MarketContext) -> PipelineReport:
    return asyncio.run(agent.run(context))


def _populated_context() -> MarketContext:
    return MarketContext(
        as_of="2026-07-05T10:00:00+00:00",
        news=[
            {
                "title": "Semiconductor demand update",
                "summary": "Data-center semiconductor demand remains firm.",
                "source": "reuters",
                "published_at": "2026-07-05T09:00:00+00:00",
            }
        ],
    )


def test_empty_context_returns_needs_more_data():
    report = _run(
        DomainAnalystAgent("semiconductor_analyst", CONFIG),
        MarketContext(),
    )

    assert report.verdict == "needs_more_data"
    assert report.metrics_snapshot["analysis_executed"] is False
    assert any("as_of" in reason.lower() for reason in report.reasons)


def test_as_of_without_timezone_returns_needs_more_data():
    report = _run(
        DomainAnalystAgent("semiconductor_analyst", CONFIG),
        MarketContext(as_of="2026-07-05T10:00:00"),
    )

    assert report.verdict == "needs_more_data"
    assert any("timezone" in reason.lower() for reason in report.reasons)


def test_valid_as_of_without_evidence_fails_cheaply():
    report = _run(
        DomainAnalystAgent("semiconductor_analyst", CONFIG),
        MarketContext(as_of="2026-07-05T10:00:00+00:00"),
    )

    assert report.verdict == "needs_more_data"
    assert report.metrics_snapshot["analysis_executed"] is False
    assert any("evidence source" in reason.lower() for reason in report.reasons)


def test_populated_context_runs_review_only_analysis():
    report = _run(
        DomainAnalystAgent("semiconductor_analyst", CONFIG),
        _populated_context(),
    )

    assert report.metrics_snapshot["domain_id"] == (
        "semiconductor_ai_infrastructure"
    )
    assert report.metrics_snapshot["decision_influence"] is False
    assert report.metrics_snapshot["can_trade"] is False
    assert "hypotheses" in report.metrics_snapshot
    assert "evidence_gaps" in report.metrics_snapshot
    assert "regime_context" in report.metrics_snapshot
    assert report.agent_name == "semiconductor_analyst"


def test_configured_missing_runtime_is_rejected_not_silently_ignored(
    tmp_path,
):
    report = _run(
        DomainAnalystAgent(
            "semiconductor_analyst",
            {
                **CONFIG,
                "runtime_artifact_path": str(tmp_path / "missing"),
            },
        ),
        _populated_context(),
    )

    assert report.verdict == "needs_more_data"
    assert report.metrics_snapshot["analysis_executed"] is False
    assert any("runtime artifact was rejected" in reason for reason in report.reasons)


def test_configured_producer_artifact_runs_additively(tmp_path):
    artifact = tmp_path / "news"
    artifact.mkdir()
    (artifact / "latest.json").write_text(
        json.dumps(
            {
                "created_at": "2026-07-05T09:30:00+00:00",
                "status": "semiconductor_news_evidence_ready_with_gaps",
                "producer_contract": "test_news_v1",
                "inputs": {"as_of": "2026-07-05T09:00:00+00:00"},
                "market_context_fragment": {
                    "news": [
                        {
                            "summary": "Verified sector demand context.",
                            "published_at": "2026-07-05T08:00:00+00:00",
                            "_dean_semantic_evidence": {
                                "producer_contract": "test_news_v1",
                                "evidence_type": "sector_demand",
                                "source_identity": "test_source",
                                "required_lane_eligible": False,
                            },
                        }
                    ]
                },
                "safety": {"review_only": True},
            }
        ),
        encoding="utf-8",
    )
    report = _run(
        DomainAnalystAgent(
            "semiconductor_analyst",
            {
                **CONFIG,
                "producer_artifact_paths": {"news": str(artifact)},
            },
        ),
        MarketContext(as_of="2026-07-05T10:00:00+00:00"),
    )

    assert report.metrics_snapshot["evidence_count"] == 1


def test_report_contract_has_hashes():
    report = _run(
        DomainAnalystAgent("semiconductor_analyst", CONFIG),
        _populated_context(),
    )

    assert isinstance(report, PipelineReport)
    assert report.input_hash
    assert report.config_hash
