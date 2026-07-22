"""Tests for ArtifactEvidenceLoader and CLI entry point.

Tests use synthetic artifact fixtures, not real saved data.
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from dean_os.analyst_core.artifact_evidence_loader import (
    ArtifactEvidenceLoader,
    load_evidence_from_artifacts,
)
from dean_os.analyst_core.sector_analyst import SectorAnalyst, SectorReport
from dean_os.analysts.schemas import AnalystEvidenceItem


# ──────────────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────────────


def _make_news_artifact(tmp: Path) -> Path:
    """Create a minimal news producer artifact for testing."""
    artifact_dir = tmp / "news_artifact"
    artifact_dir.mkdir()
    data = {
        "run_id": "test_run_001",
        "created_at": "2026-07-01T00:00:00Z",
        "producer_contract": "dean_saved_semiconductor_news_evidence_producer_v1",
        "status": "semiconductor_news_evidence_ready_with_gaps",
        "summary": {"accepted_news_record_count": 2},
        "context_adapter": {
            "market_context_fragment": {
                "news": [
                    {
                        "title": "AI demand accelerates",
                        "summary": "NVIDIA reports record GPU orders on AI infrastructure spending",
                        "source": "reuters",
                        "published_at": "2026-06-28T10:00:00Z",
                        "_dean_semantic_evidence": {
                            "producer_contract": "dean_saved_semiconductor_news_evidence_producer_v1",
                            "evidence_type": "sector_demand",
                            "required_lane_eligible": True,
                            "source_tier": "tier_2_strong_context",
                            "source_identity": "reuters",
                            "matched_terms": ["gpu orders", "ai infrastructure"],
                            "candidate_sha256": "abc123",
                            "stance_hint": "positive",
                        },
                    },
                    {
                        "title": "Export controls tighten",
                        "summary": "New semiconductor export restrictions announced",
                        "source": "bloomberg",
                        "published_at": "2026-06-29T14:00:00Z",
                        "_dean_semantic_evidence": {
                            "producer_contract": "dean_saved_semiconductor_news_evidence_producer_v1",
                            "evidence_type": "policy_or_geopolitical",
                            "required_lane_eligible": True,
                            "source_tier": "tier_2_strong_context",
                            "source_identity": "bloomberg",
                            "matched_terms": ["export restrictions"],
                            "candidate_sha256": "def456",
                            "stance_hint": "negative",
                        },
                    },
                ],
            },
        },
        "safety": {"review_only": True},
    }
    with open(artifact_dir / "latest.json", "w") as f:
        json.dump(data, f)
    return artifact_dir


def _make_macro_artifact(tmp: Path) -> Path:
    """Create a minimal macro producer artifact for testing."""
    artifact_dir = tmp / "macro_artifact"
    artifact_dir.mkdir()
    data = {
        "run_id": "test_macro_001",
        "created_at": "2026-07-01T00:00:00Z",
        "producer_contract": "dean_saved_macro_evidence_producer_v1",
        "status": "macro_evidence_ready_with_exclusions",
        "selected_observations": [
            {
                "context_key": "fed_funds_rate",
                "value": 5.25,
                "unit": "percent",
                "period": "2026-06-26",
                "available_at": "2026-06-29T23:59:59Z",
                "source_locator": "https://fred.stlouisfed.org/series/FEDFUNDS",
                "required_lane_eligible": False,
                "stance_hint": "unknown",
            },
            {
                "context_key": "cpi_yoy",
                "value": 3.2,
                "unit": "percent",
                "period": "2026-06-15",
                "available_at": "2026-06-25T23:59:59Z",
                "source_locator": "https://fred.stlouisfed.org/series/CPIAUCSL",
                "required_lane_eligible": False,
                "stance_hint": "unknown",
            },
        ],
        "safety": {"review_only": True},
    }
    with open(artifact_dir / "latest.json", "w") as f:
        json.dump(data, f)
    return artifact_dir


def _make_sector_market_artifact(tmp: Path) -> Path:
    """Create a minimal sector market producer artifact for testing."""
    artifact_dir = tmp / "sector_market_artifact"
    artifact_dir.mkdir()
    data = {
        "run_id": "test_sector_001",
        "created_at": "2026-07-01T00:00:00Z",
        "producer_contract": "dean_saved_sector_market_evidence_producer_v1",
        "status": "sector_market_evidence_ready",
        "metrics": [
            {
                "name": "sector_median_return_20_session",
                "value": 3.556,
                "unit": "percent",
                "period": "2026-05-27/2026-06-26",
                "available_at": "2026-06-27T12:06:33Z",
                "source_locator": "test_source",
                "evidence_type": "market_confirmation",
                "required_lane_eligible": True,
                "stance_hint": "positive",
            },
        ],
        "safety": {"review_only": True},
    }
    with open(artifact_dir / "latest.json", "w") as f:
        json.dump(data, f)
    return artifact_dir


def _make_runtime_artifact(tmp: Path) -> Path:
    """Create a minimal runtime artifact with adapted evidence."""
    artifact_dir = tmp / "runtime_artifact"
    artifact_dir.mkdir()
    data = {
        "run_id": "test_runtime_001",
        "created_at": "2026-07-01T00:00:00Z",
        "runtime_contract": "dean_semiconductor_analyst_runtime_v1",
        "status": "semiconductor_analysis_ready_for_review",
        "adapter": {
            "evidence": [
                {
                    "evidence_id": "ev_001",
                    "source_type": "news",
                    "source": "reuters",
                    "as_of": "2026-07-01T00:00:00Z",
                    "domain_id": "semiconductor_ai_infrastructure",
                    "tickers": [],
                    "sectors": ["semiconductor_ai_infrastructure"],
                    "evidence_type": "sector_demand",
                    "summary": "AI demand accelerates",
                    "stance_hint": "positive",
                    "strength": 0.8,
                    "freshness_score": 0.9,
                    "directness": "sector",
                    "reliability_score": 0.75,
                },
                {
                    "evidence_id": "ev_002",
                    "source_type": "macro",
                    "source": "https://fred.stlouisfed.org/series/FEDFUNDS",
                    "as_of": "2026-07-01T00:00:00Z",
                    "domain_id": "semiconductor_ai_infrastructure",
                    "tickers": [],
                    "sectors": ["semiconductor_ai_infrastructure"],
                    "evidence_type": "macro_context",
                    "summary": "macro observation fed_funds_rate=5.25 percent",
                    "stance_hint": "unknown",
                    "strength": 0.55,
                    "freshness_score": 0.7,
                    "directness": "macro",
                    "reliability_score": 0.55,
                },
            ],
        },
        "safety": {"review_only": True},
    }
    with open(artifact_dir / "latest.json", "w") as f:
        json.dump(data, f)
    return artifact_dir


# ──────────────────────────────────────────────────────────────────────────────
# ArtifactEvidenceLoader tests
# ──────────────────────────────────────────────────────────────────────────────


class TestArtifactEvidenceLoaderFromRuntime:
    def test_loads_from_runtime_artifact(self, tmp_path):
        artifact = _make_runtime_artifact(tmp_path)
        loader = ArtifactEvidenceLoader()
        evidence = loader.from_runtime_artifact(artifact)

        assert len(evidence) == 2
        assert evidence[0].evidence_id == "ev_001"
        assert evidence[0].evidence_type == "sector_demand"
        assert evidence[1].evidence_type == "macro_context"

    def test_runtime_all_items_are_valid(self, tmp_path):
        artifact = _make_runtime_artifact(tmp_path)
        loader = ArtifactEvidenceLoader()
        evidence = loader.from_runtime_artifact(artifact)

        for item in evidence:
            assert isinstance(item, AnalystEvidenceItem)
            assert item.as_of
            assert item.domain_id

    def test_runtime_missing_dir_raises(self, tmp_path):
        loader = ArtifactEvidenceLoader()
        with pytest.raises(FileNotFoundError):
            loader.from_runtime_artifact(tmp_path / "nonexistent")

    def test_runtime_empty_evidence_raises(self, tmp_path):
        artifact_dir = tmp_path / "empty_runtime"
        artifact_dir.mkdir()
        with open(artifact_dir / "latest.json", "w") as f:
            json.dump({"adapter": {"evidence": []}}, f)

        loader = ArtifactEvidenceLoader()
        with pytest.raises(ValueError, match="no adapted evidence"):
            loader.from_runtime_artifact(artifact_dir)


class TestArtifactEvidenceLoaderFromProducers:
    def test_loads_from_news_artifact(self, tmp_path):
        artifact = _make_news_artifact(tmp_path)
        loader = ArtifactEvidenceLoader()
        evidence = loader.from_producer_artifacts(
            news_path=artifact,
            domain_id="semiconductor_ai_infrastructure",
            as_of="2026-07-01T00:00:00Z",
        )

        assert len(evidence) == 2
        assert evidence[0].evidence_type == "sector_demand"
        assert evidence[0].stance_hint == "positive"
        assert evidence[1].evidence_type == "policy_or_geopolitical"
        assert evidence[1].stance_hint == "negative"

    def test_loads_from_macro_artifact(self, tmp_path):
        artifact = _make_macro_artifact(tmp_path)
        loader = ArtifactEvidenceLoader()
        evidence = loader.from_producer_artifacts(
            macro_path=artifact,
            domain_id="semiconductor_ai_infrastructure",
            as_of="2026-07-01T00:00:00Z",
        )

        assert len(evidence) == 2
        assert evidence[0].evidence_type == "macro_context"
        assert "fed_funds_rate" in evidence[0].summary

    def test_loads_from_sector_market_artifact(self, tmp_path):
        artifact = _make_sector_market_artifact(tmp_path)
        loader = ArtifactEvidenceLoader()
        evidence = loader.from_producer_artifacts(
            sector_market_path=artifact,
            domain_id="semiconductor_ai_infrastructure",
            as_of="2026-07-01T00:00:00Z",
        )

        assert len(evidence) == 1
        assert evidence[0].evidence_type == "market_confirmation"
        assert evidence[0].stance_hint == "positive"

    def test_loads_from_multiple_producers(self, tmp_path):
        news = _make_news_artifact(tmp_path)
        macro = _make_macro_artifact(tmp_path)
        sector = _make_sector_market_artifact(tmp_path)

        loader = ArtifactEvidenceLoader()
        evidence = loader.from_producer_artifacts(
            news_path=news,
            macro_path=macro,
            sector_market_path=sector,
            domain_id="semiconductor_ai_infrastructure",
            as_of="2026-07-01T00:00:00Z",
        )

        # 2 news + 2 macro + 1 sector = 5
        assert len(evidence) == 5

    def test_producer_evidence_ids_are_deterministic(self, tmp_path):
        artifact = _make_news_artifact(tmp_path)
        loader = ArtifactEvidenceLoader()
        first = loader.from_producer_artifacts(
            news_path=artifact,
            domain_id="semiconductor_ai_infrastructure",
            as_of="2026-07-01T00:00:00Z",
        )
        second = loader.from_producer_artifacts(
            news_path=artifact,
            domain_id="semiconductor_ai_infrastructure",
            as_of="2026-07-01T00:00:00Z",
        )

        assert [item.evidence_id for item in first] == [
            item.evidence_id for item in second
        ]

    def test_future_producer_artifact_is_rejected(self, tmp_path):
        artifact = _make_news_artifact(tmp_path)
        loader = ArtifactEvidenceLoader()

        with pytest.raises(ValueError, match="future evidence"):
            loader.from_producer_artifacts(
                news_path=artifact,
                domain_id="semiconductor_ai_infrastructure",
                as_of="2026-06-30T23:59:59Z",
            )

    def test_news_without_semantic_evidence_skipped(self, tmp_path):
        artifact_dir = tmp_path / "bad_news"
        artifact_dir.mkdir()
        data = {
            "created_at": "2026-07-01T00:00:00Z",
            "status": "semiconductor_news_evidence_ready_with_gaps",
            "context_adapter": {
                "market_context_fragment": {
                    "news": [
                        {"title": "No semantic evidence", "summary": "test"},
                    ],
                },
            },
            "safety": {"review_only": True},
        }
        with open(artifact_dir / "latest.json", "w") as f:
            json.dump(data, f)

        loader = ArtifactEvidenceLoader()
        evidence = loader.from_producer_artifacts(
            news_path=artifact_dir,
            domain_id="test",
            as_of="2026-07-01T00:00:00Z",
        )
        assert len(evidence) == 0


class TestLoadEvidenceFromArtifacts:
    def test_convenience_function_with_runtime(self, tmp_path):
        artifact = _make_runtime_artifact(tmp_path)
        evidence = load_evidence_from_artifacts(
            artifact_paths={"runtime": str(artifact)},
            domain_id="semiconductor_ai_infrastructure",
        )
        assert len(evidence) == 2

    def test_convenience_function_with_producers(self, tmp_path):
        news = _make_news_artifact(tmp_path)
        macro = _make_macro_artifact(tmp_path)
        evidence = load_evidence_from_artifacts(
            artifact_paths={"news": str(news), "macro": str(macro)},
            domain_id="semiconductor_ai_infrastructure",
            as_of="2026-07-01T00:00:00Z",
        )
        assert len(evidence) == 4  # 2 news + 2 macro


# ──────────────────────────────────────────────────────────────────────────────
# Integration: loader → SectorAnalyst
# ──────────────────────────────────────────────────────────────────────────────


class TestLoaderToSectorAnalyst:
    def test_full_pipeline_from_runtime(self, tmp_path):
        artifact = _make_runtime_artifact(tmp_path)
        evidence = load_evidence_from_artifacts(
            artifact_paths={"runtime": str(artifact)},
            domain_id="semiconductor_ai_infrastructure",
        )

        analyst = SectorAnalyst(domain_id="semiconductor_ai_infrastructure")
        report = analyst.run_from_evidence(
            evidence=evidence,
            as_of="2026-07-01T00:00:00Z",
        )

        assert isinstance(report, SectorReport)
        assert report.domain_id == "semiconductor_ai_infrastructure"
        assert report.evidence_count == 2
        assert report.lens_count > 0
        assert report.review_required is True
        assert report.live_execution_allowed is False

    def test_full_pipeline_from_producers(self, tmp_path):
        news = _make_news_artifact(tmp_path)
        macro = _make_macro_artifact(tmp_path)
        sector = _make_sector_market_artifact(tmp_path)

        evidence = load_evidence_from_artifacts(
            artifact_paths={
                "news": str(news),
                "macro": str(macro),
                "sector_market": str(sector),
            },
            domain_id="semiconductor_ai_infrastructure",
            as_of="2026-07-01T00:00:00Z",
        )

        analyst = SectorAnalyst(domain_id="semiconductor_ai_infrastructure")
        report = analyst.run_from_evidence(
            evidence=evidence,
            as_of="2026-07-01T00:00:00Z",
        )

        assert isinstance(report, SectorReport)
        assert report.evidence_count == 5
        assert report.thesis is not None

    def test_report_to_dict_from_loaded_evidence(self, tmp_path):
        artifact = _make_runtime_artifact(tmp_path)
        evidence = load_evidence_from_artifacts(
            artifact_paths={"runtime": str(artifact)},
            domain_id="semiconductor_ai_infrastructure",
        )

        analyst = SectorAnalyst(domain_id="semiconductor_ai_infrastructure")
        report = analyst.run_from_evidence(
            evidence=evidence,
            as_of="2026-07-01T00:00:00Z",
        )

        d = report.to_dict()
        assert d["report_type"] == "sector_analysis"
        assert d["stats"]["evidence_count"] == 2
        assert d["stats"]["lens_count"] > 0
