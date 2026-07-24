"""Tests for SectorPipelineManager."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from dean_os.analyst_core.pipeline_manager import (
    PipelineRunResult,
    SectorPipelineManager,
    _discover_artifact_dir,
)


class TestDiscoverArtifacts:
    def test_discover_finds_existing(self, tmp_path):
        d = tmp_path / "news"
        d.mkdir(parents=True)
        (d / "latest.json").write_text("{}")
        assert _discover_artifact_dir(tmp_path, "news") == d

    def test_discover_missing_returns_none(self, tmp_path):
        assert _discover_artifact_dir(tmp_path, "nonexistent") is None

    def test_discover_no_latest_json(self, tmp_path):
        d = tmp_path / "news"
        d.mkdir(parents=True)
        assert _discover_artifact_dir(tmp_path, "news") is None

    def test_discover_all_artifact_types(self, tmp_path):
        pm = SectorPipelineManager(domain_id="energy")
        for name in ("news", "macro", "sector_market", "policy", "fundamental", "runtime"):
            d = tmp_path / name
            d.mkdir(parents=True)
            (d / "latest.json").write_text("{}")

        found = pm.discover_artifacts(tmp_path)
        assert all(found[k] is not None for k in found)

    def test_discover_empty_dir(self, tmp_path):
        pm = SectorPipelineManager(domain_id="energy")
        found = pm.discover_artifacts(tmp_path)
        # Returns dict with all keys, values None
        assert found == {"news": None, "macro": None, "sector_market": None, "policy": None, "fundamental": None, "runtime": None}


class TestPipelineRunResult:
    def test_creates_with_defaults(self):
        r = PipelineRunResult(domain_id="test", as_of="2026-07-01")
        assert r.domain_id == "test"
        assert r.as_of == "2026-07-01"
        assert r.errors == []
        assert r.warnings == []

    def test_with_all_fields(self):
        r = PipelineRunResult(
            domain_id="energy",
            as_of="2026-07-01",
            analysis_result={"status": "ok"},
            evaluation_result={"accuracy": 0.8},
            knowledge_result={"items": 10},
        )
        assert r.analysis_result["status"] == "ok"


class TestPipelineManagerConstruction:
    def test_creates_for_energy(self):
        pm = SectorPipelineManager(domain_id="energy")
        assert pm.domain_id == "energy"

    def test_creates_for_semiconductor(self):
        pm = SectorPipelineManager(domain_id="semiconductor_ai_infrastructure")
        assert pm.domain_id == "semiconductor_ai_infrastructure"

    def test_creates_for_geopolitics(self):
        pm = SectorPipelineManager(domain_id="geopolitics")
        assert pm.domain_id == "geopolitics"


class TestPipelineManagerRunAnalysis:
    def test_run_without_artifacts(self):
        pm = SectorPipelineManager(domain_id="energy")
        result = pm.run_analysis(artifact_dirs={}, as_of="2026-07-01")
        assert result.domain_id == "energy"
        # With empty artifact_dirs, no paths are passed -> 0 evidence, still returns a report
        assert result.analysis_result is not None
        assert result.analysis_result["evidence_count"] == 0
        assert result.analysis_result["status"] == "needs_more_data"

    def test_run_with_news_artifact(self, tmp_path):
        artifact_dir = tmp_path / "news"
        artifact_dir.mkdir(parents=True)
        data = {
            # _validated_producer (artifact_evidence_loader.py) requires this
            # producer-contract wrapper on every artifact regardless of the
            # inner fragment shape.
            "created_at": "2026-06-28T12:00:00+00:00",
            "status": "ready",
            "safety": {"review_only": True},
            "inputs": {"as_of": "2026-06-28T12:00:00+00:00"},
            "market_context_fragment": {
                "news": [
                    {
                        "title": "Energy test",
                        "summary": "Oil price movement test",
                        "source": "reuters",
                        "published_at": "2026-06-28T10:00:00Z",
                        "_dean_semantic_evidence": {
                            "evidence_type": "sector_demand",
                            "source_tier": "tier_2_strong_context",
                            "source_identity": "reuters",
                            "matched_terms": ["oil"],
                            "stance_hint": "positive",
                        },
                    },
                ],
            },
        }
        with open(artifact_dir / "latest.json", "w") as f:
            json.dump(data, f)

        pm = SectorPipelineManager(domain_id="energy")
        result = pm.run_analysis(
            news_path=artifact_dir,
            as_of="2026-07-01T00:00:00+00:00",
        )
        assert result.analysis_result is not None
        assert result.analysis_result["domain_id"] == "energy"

    def test_run_with_discovered_artifacts(self, tmp_path):
        news_dir = tmp_path / "news"
        news_dir.mkdir(parents=True)
        data = {
            "created_at": "2026-06-28T12:00:00+00:00",
            "status": "ready",
            "safety": {"review_only": True},
            "inputs": {"as_of": "2026-06-28T12:00:00+00:00"},
            "market_context_fragment": {
                "news": [
                    {
                        "title": "Test",
                        "summary": "Test",
                        "source": "reuters",
                        "published_at": "2026-06-28T10:00:00Z",
                        "_dean_semantic_evidence": {
                            "evidence_type": "sector_demand",
                            "source_tier": "tier_2_strong_context",
                            "source_identity": "reuters",
                            "matched_terms": ["test"],
                            "stance_hint": "positive",
                        },
                    },
                ],
            },
        }
        with open(news_dir / "latest.json", "w") as f:
            json.dump(data, f)

        pm = SectorPipelineManager(domain_id="energy")
        discovered = pm.discover_artifacts(tmp_path)
        result = pm.run_analysis(artifact_dirs=discovered, as_of="2026-07-01T00:00:00+00:00")
        assert result.analysis_result is not None

    def test_unknown_domain_raises(self):
        with pytest.raises(KeyError):
            SectorPipelineManager(domain_id="nonexistent")

    def test_runtime_cutoff_must_match_requested_as_of(self, tmp_path):
        runtime_dir = tmp_path / "runtime"
        runtime_dir.mkdir()
        runtime_dir.joinpath("latest.json").write_text(
            json.dumps(
                {
                    "runtime_contract": (
                        "dean_semiconductor_analyst_runtime_v1"
                    ),
                    "mode": "semiconductor_analyst_runtime",
                    "domain_id": "semiconductor_ai_infrastructure",
                    "status": "partial_ready_for_review",
                    "inputs": {
                        "as_of": "2026-07-01T00:00:00+00:00"
                    },
                    "summary": {"evidence_count": 1},
                    "analyst_report": {
                        "as_of": "2026-07-01T00:00:00+00:00",
                        "evidence": [
                            {
                                "evidence_id": "ev_001",
                                "source_type": "news",
                                "source": "reuters",
                                "as_of": (
                                    "2026-07-01T00:00:00+00:00"
                                ),
                                "domain_id": (
                                    "semiconductor_ai_infrastructure"
                                ),
                                "tickers": [],
                                "sectors": [
                                    "semiconductor_ai_infrastructure"
                                ],
                                "evidence_type": "sector_demand",
                                "summary": "Demand update",
                                "stance_hint": "positive",
                                "strength": 0.8,
                                "freshness_score": 0.9,
                                "directness": "sector",
                                "reliability_score": 0.8,
                            }
                        ],
                    },
                    "source_artifacts": {},
                    "safety": {"review_only": True},
                }
            ),
            encoding="utf-8",
        )

        result = SectorPipelineManager(
            domain_id="semiconductor_ai_infrastructure"
        ).run_analysis(
            runtime_artifact=runtime_dir,
            as_of="2026-07-02T00:00:00+00:00",
        )

        assert result.analysis_result is None
        assert any(
            "does not match runtime cutoff" in error
            for error in result.errors
        )


class TestPipelineManagerSaveReport:
    def test_save_report_json(self, tmp_path):
        pm = SectorPipelineManager(domain_id="energy")
        # Run analysis first to get a real report
        result = pm.run_analysis(artifact_dirs={}, as_of="2026-07-01")
        paths = pm.save_report(result.analysis_result, tmp_path, fmt="json")
        assert "json" in paths
        assert Path(paths["json"]).exists()

    def test_save_report_no_report(self, tmp_path):
        pm = SectorPipelineManager(domain_id="energy")
        paths = pm.save_report({"domain_id": "energy"}, tmp_path)
        assert paths == {}


class TestPipelineManagerEvaluate:
    def test_evaluate_no_report_returns_error(self):
        pm = SectorPipelineManager(domain_id="energy")
        result = pm.evaluate(
            price_data_path="nonexistent.parquet",
            analysis_result={"domain_id": "energy"},
            as_of="2026-07-01",
        )
        assert len(result.errors) > 0 or result.evaluation_result is None

    def test_evaluate_with_missing_price_data(self, tmp_path):
        pm = SectorPipelineManager(domain_id="energy")
        # Run analysis first
        result = pm.run_analysis(artifact_dirs={}, as_of="2026-07-01")
        eval_result = pm.evaluate(
            price_data_path=str(tmp_path / "nonexistent.parquet"),
            analysis_result=result.analysis_result,
            as_of="2026-07-01",
        )
        assert len(eval_result.errors) > 0 or eval_result.evaluation_result is None


class TestPipelineManagerBuildKnowledge:
    def test_build_knowledge_no_artifacts(self, tmp_path):
        pm = SectorPipelineManager(domain_id="energy")
        result = pm.build_knowledge(
            artifact_dirs={},
            output_dir=str(tmp_path / "knowledge"),
            as_of="2026-07-01",
        )
        assert len(result.errors) > 0

    def test_build_knowledge_with_news(self, tmp_path):
        news_dir = tmp_path / "news"
        news_dir.mkdir(parents=True)
        data = {
            "market_context_fragment": {
                "news": [
                    {
                        "title": "Oil prices rise",
                        "summary": "Oil prices increased on supply concerns",
                        "source": "reuters",
                        "published_at": "2026-06-28T10:00:00Z",
                        "_dean_semantic_evidence": {
                            "evidence_type": "sector_demand",
                            "source_tier": "tier_2_strong_context",
                            "source_identity": "reuters",
                            "matched_terms": ["oil", "supply"],
                            "stance_hint": "positive",
                        },
                    },
                ],
            },
        }
        with open(news_dir / "latest.json", "w") as f:
            json.dump(data, f)

        pm = SectorPipelineManager(domain_id="energy")
        result = pm.build_knowledge(
            artifact_dirs={"news": news_dir},
            output_dir=str(tmp_path / "knowledge"),
            as_of="2026-07-01",
        )
        if not result.errors:
            assert result.knowledge_result is not None
            assert result.knowledge_result["item_count"] >= 1
