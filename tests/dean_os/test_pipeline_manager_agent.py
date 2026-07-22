"""Contract tests for the composite PipelineManagerAgent."""
from __future__ import annotations

import asyncio

from dean_os.agents.pipeline_manager import PipelineManagerAgent
from dean_os.schemas import MarketContext, PipelineReport


CONFIG = {
    "domain_id": "semiconductor_ai_infrastructure",
    "horizon_days": 180,
}


def _run(context: MarketContext) -> PipelineReport:
    return asyncio.run(
        PipelineManagerAgent("pipeline_manager", CONFIG).run(context)
    )


def test_empty_context_returns_needs_more_data():
    report = _run(MarketContext())

    assert report.verdict == "needs_more_data"
    assert any("as_of" in reason.lower() for reason in report.reasons)


def test_as_of_without_timezone_returns_needs_more_data():
    report = _run(MarketContext(as_of="2026-07-05T10:00:00"))

    assert report.verdict == "needs_more_data"
    assert any("timezone" in reason.lower() for reason in report.reasons)


def test_valid_as_of_no_artifacts_returns_needs_more_data():
    report = _run(
        MarketContext(as_of="2026-07-05T10:00:00+00:00")
    )

    assert report.verdict == "needs_more_data"
    assert any("artifact" in reason.lower() for reason in report.reasons)


def test_report_is_pipeline_report_with_hashes():
    report = _run(
        MarketContext(as_of="2026-07-05T10:00:00+00:00")
    )

    assert isinstance(report, PipelineReport)
    assert report.input_hash
    assert report.config_hash


def test_report_includes_artifact_bindings():
    report = _run(
        MarketContext(as_of="2026-07-05T10:00:00+00:00")
    )

    assert isinstance(
        report.metrics_snapshot.get("artifact_bindings", {}),
        dict,
    )
