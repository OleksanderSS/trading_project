"""Tests for SectorAnalyst clone functionality."""
from __future__ import annotations

import pytest

from dean_os.analyst_core.sector_analyst import SectorAnalyst
from dean_os.analyst_core.artifact_evidence_loader import load_evidence_from_artifacts
from dean_os.analysts.profiles import list_domain_profiles


class TestCloneBasics:
    def test_clone_creates_new_instance(self):
        semi = SectorAnalyst(domain_id="semiconductor_ai_infrastructure")
        energy = semi.clone(domain_id="energy")

        assert energy.domain_id == "energy"
        assert energy.agent_name == "energy_sector_analyst"
        assert energy is not semi

    def test_clone_shares_registry(self):
        semi = SectorAnalyst(domain_id="semiconductor_ai_infrastructure")
        energy = semi.clone(domain_id="energy")

        # Same registry instance (shared lenses)
        assert energy.registry is semi.registry

    def test_clone_custom_name(self):
        semi = SectorAnalyst(domain_id="semiconductor_ai_infrastructure")
        energy = semi.clone(domain_id="energy", agent_name="oil_gas_expert")

        assert energy.agent_name == "oil_gas_expert"


class TestCloneProfileOverride:
    def test_clone_overrides_tickers(self):
        semi = SectorAnalyst(domain_id="semiconductor_ai_infrastructure")
        energy = semi.clone(
            domain_id="energy",
            ticker_universe=["XLE", "USO", "XOM", "CVX", "OXY", "UNG"],
        )

        assert energy.profile.ticker_universe_hint == ["XLE", "USO", "XOM", "CVX", "OXY", "UNG"]
        # Original unchanged
        assert semi.profile.ticker_universe_hint != ["XLE", "USO", "XOM", "CVX", "OXY", "UNG"]
        assert energy.base_agent.profile is energy.profile
        assert energy.evidence_adapter.profile is energy.profile
        assert energy.lens_orchestrator.config["ticker_universe"] == [
            "XLE",
            "USO",
            "XOM",
            "CVX",
            "OXY",
            "UNG",
        ]

    def test_clone_overrides_keywords(self):
        semi = SectorAnalyst(domain_id="semiconductor_ai_infrastructure")
        energy = semi.clone(
            domain_id="energy",
            sector_keywords=["oil", "gas", "OPEC", "inventories", "refinery"],
        )

        assert "oil" in energy.profile.sector_keywords
        assert "OPEC" in energy.profile.sector_keywords
        assert energy.lens_orchestrator.config["sector_keywords"] == [
            "oil",
            "gas",
            "OPEC",
            "inventories",
            "refinery",
        ]

    def test_clone_overrides_questions(self):
        semi = SectorAnalyst(domain_id="semiconductor_ai_infrastructure")
        questions = [
            "Is energy supply tightening or loosening?",
            "Is demand improving or weakening?",
            "Are OPEC decisions changing the balance?",
        ]
        energy = semi.clone(
            domain_id="energy",
            core_questions=questions,
        )

        assert energy.profile.core_questions == questions
        assert energy.base_agent.profile.core_questions == questions


class TestCloneFromRealProfile:
    def test_clone_energy_from_profiles(self):
        semi = SectorAnalyst(domain_id="semiconductor_ai_infrastructure")
        energy = semi.clone(
            domain_id="energy",
            ticker_universe=["XLE", "USO", "XOM", "CVX", "OXY", "UNG"],
            sector_keywords=["oil", "gas", "OPEC", "inventories", "refinery", "energy"],
            required_evidence_types=["supply", "demand", "inventories", "policy_or_geopolitical", "market_confirmation"],
        )

        report = energy.run_from_evidence(
            evidence=[],
            as_of="2026-07-01T00:00:00Z",
        )

        assert report.domain_id == "energy"
        assert report.recommendation in ("no_evidence", "needs_more_data", "partial_ready_for_review", "ready_for_review")


class TestCloneRunWithEvidence:
    def test_cloned_analyst_with_evidence(self, tmp_path):
        news = _make_news_artifact(tmp_path)

        semi = SectorAnalyst(domain_id="semiconductor_ai_infrastructure")
        energy = semi.clone(
            domain_id="energy",
            ticker_universe=["XLE", "USO", "XOM"],
        )

        evidence = load_evidence_from_artifacts(
            artifact_paths={"news": str(news)},
            domain_id="energy",
            as_of="2026-07-01T00:00:00Z",
        )

        report = energy.run_from_evidence(
            evidence=evidence,
            as_of="2026-07-01T00:00:00Z",
        )

        assert report.domain_id == "energy"
        assert report.evidence_count == 1


def _make_news_artifact(tmp):
    import json
    from pathlib import Path

    artifact_dir = tmp / "news_artifact"
    artifact_dir.mkdir()
    data = {
        "created_at": "2026-06-29T00:00:00Z",
        "inputs": {"as_of": "2026-06-29T00:00:00Z"},
        "status": "ready_for_review",
        "safety": {
            "review_only": True,
            "training_run_performed": False,
            "tuning_run_performed": False,
            "learning_write_performed": False,
            "production_config_write_performed": False,
            "broker_access_performed": False,
            "live_execution_performed": False,
        },
        "market_context_fragment": {
            "news": [
                {
                    "title": "Oil supply tightens",
                    "summary": "OPEC cuts production by 1 million barrels",
                    "source": "reuters",
                    "published_at": "2026-06-28T10:00:00Z",
                    "_dean_semantic_evidence": {
                        "evidence_type": "supply",
                        "source_tier": "tier_2_strong_context",
                        "source_identity": "reuters",
                        "matched_terms": ["OPEC", "production"],
                        "stance_hint": "negative",
                    },
                },
            ],
        },
    }
    with open(artifact_dir / "latest.json", "w") as f:
        json.dump(data, f)
    return artifact_dir
