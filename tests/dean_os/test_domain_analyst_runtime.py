"""Tests for DomainAnalystRuntime."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from dean_os.analyst_core.domain_analyst_runtime import (
    DomainAnalystRuntime,
    list_available_domains,
    create_analyst,
)


def _make_news_artifact(tmp: Path, domain: str = "energy") -> Path:
    artifact_dir = tmp / "news"
    artifact_dir.mkdir()
    data = {
        # _validated_producer (artifact_evidence_loader.py) requires this
        # producer-contract wrapper on every artifact -- created_at/status/
        # safety.review_only -- regardless of the inner fragment shape.
        "created_at": "2026-06-28T12:00:00+00:00",
        "status": "ready",
        "safety": {"review_only": True},
        "inputs": {"as_of": "2026-06-28T12:00:00+00:00"},
        "market_context_fragment": {
            "news": [
                {
                    "title": "Test news",
                    "summary": f"Test evidence for {domain}",
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
    with open(artifact_dir / "latest.json", "w") as f:
        json.dump(data, f)
    return artifact_dir.parent


class TestDomainAnalystRuntime:
    def test_creates_for_energy(self):
        runtime = DomainAnalystRuntime(domain_id="energy")
        assert runtime.domain_id == "energy"
        assert runtime.profile.domain_id == "energy"
    
    def test_creates_for_semiconductor(self):
        runtime = DomainAnalystRuntime(domain_id="semiconductor_ai_infrastructure")
        assert runtime.domain_id == "semiconductor_ai_infrastructure"
    
    def test_creates_for_geopolitics(self):
        runtime = DomainAnalystRuntime(domain_id="geopolitics")
        assert runtime.domain_id == "geopolitics"
    
    def test_rejects_unknown_domain(self):
        with pytest.raises(KeyError):
            DomainAnalystRuntime(domain_id="unknown_domain")


class TestRun:
    def test_run_with_news(self, tmp_path):
        artifact_dir = _make_news_artifact(tmp_path)
        runtime = DomainAnalystRuntime(domain_id="energy")
        
        result = runtime.run(
            news_path=artifact_dir / "news",
            as_of="2026-07-01T00:00:00Z",
        )
        
        assert result["domain_id"] == "energy"
        assert result["evidence_count"] == 1
        assert result["review_required"] is True
        assert result["live_execution_allowed"] is False
    
    def test_run_without_evidence(self):
        runtime = DomainAnalystRuntime(domain_id="energy")
        
        result = runtime.run(
            as_of="2026-07-01T00:00:00Z",
        )
        
        assert result["domain_id"] == "energy"
        assert result["evidence_count"] == 0


class TestClone:
    def test_clone_to_energy(self):
        semi = DomainAnalystRuntime(domain_id="semiconductor_ai_infrastructure")
        energy = semi.clone(
            domain_id="energy",
            ticker_universe=["XLE", "USO", "XOM"],
        )
        
        assert energy.domain_id == "energy"
        assert energy.profile.ticker_universe_hint == ["XLE", "USO", "XOM"]
    
    def test_clone_presues_analyst(self):
        semi = DomainAnalystRuntime(domain_id="semiconductor_ai_infrastructure")
        energy = semi.clone(domain_id="energy")
        
        # Should share the same lens registry
        assert energy.analyst.registry is semi.analyst.registry


class TestFactory:
    def test_create_analyst(self):
        runtime = create_analyst(domain_id="energy")
        assert runtime.domain_id == "energy"
    
    def test_list_available_domains(self):
        domains = list_available_domains()
        assert "energy" in domains
        assert "semiconductor_ai_infrastructure" in domains


class TestRunWithArtifactDir:
    def test_run_from_artifact_dir(self, tmp_path):
        # Create artifact structure
        artifact_dir = tmp_path / "artifacts"
        news_dir = artifact_dir / "news"
        news_dir.mkdir(parents=True)
        
        data = {
            # See _make_news_artifact's comment above -- _validated_producer
            # requires this wrapper regardless of the inner fragment shape.
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
        
        runtime = DomainAnalystRuntime(domain_id="energy")
        result = runtime.run_from_artifacts(
            artifact_dir=artifact_dir,
            as_of="2026-07-01T00:00:00Z",
        )
        
        assert result["evidence_count"] == 1
