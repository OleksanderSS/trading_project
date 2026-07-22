"""Tests for knowledge pack builder."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from build_knowledge_pack import (
    build_knowledge_pack,
    _evidence_to_knowledge_items,
    _evidence_type_to_item_type,
    _reliability_to_quality,
)
from dean_os.analyst_core.artifact_evidence_loader import ArtifactEvidenceLoader
from dean_os.analyst_knowledge.pack_loader import load_knowledge_pack, save_knowledge_pack
from dean_os.analyst_knowledge.store import LocalKnowledgeStore
from dean_os.analyst_knowledge.retriever import KnowledgeRetriever
from dean_os.analysts.schemas import AnalystEvidenceItem


# ──────────────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────────────


def _make_evidence_item(
    evidence_id: str = "ev_test_001",
    evidence_type: str = "sector_demand",
    stance: str = "positive",
    reliability: float = 0.75,
    strength: float = 0.8,
) -> AnalystEvidenceItem:
    return AnalystEvidenceItem(
        evidence_id=evidence_id,
        source_type="news",
        source="reuters",
        as_of="2026-07-01T00:00:00Z",
        domain_id="semiconductor_ai_infrastructure",
        tickers=["NVDA"],
        sectors=["semiconductor_ai_infrastructure"],
        evidence_type=evidence_type,
        summary=f"Test evidence {evidence_id} about {evidence_type}",
        stance_hint=stance,
        strength=strength,
        freshness_score=0.7,
        directness="sector",
        reliability_score=reliability,
    )


def _make_news_artifact(tmp: Path) -> Path:
    artifact_dir = tmp / "news_artifact"
    artifact_dir.mkdir()
    data = {
        "created_at": "2026-07-01T00:00:00Z",
        "status": "semiconductor_news_evidence_ready_with_gaps",
        "inputs": {"as_of": "2026-07-01T00:00:00Z"},
        "market_context_fragment": {
            "news": [
                {
                    "title": "AI demand accelerates",
                    "summary": "NVIDIA reports record GPU orders",
                    "source": "reuters",
                    "published_at": "2026-06-28T10:00:00Z",
                    "_dean_semantic_evidence": {
                        "evidence_type": "sector_demand",
                        "source_tier": "tier_2_strong_context",
                        "source_identity": "reuters",
                        "matched_terms": ["gpu orders"],
                        "stance_hint": "positive",
                    },
                },
            ],
        },
        "safety": {"review_only": True},
    }
    with open(artifact_dir / "latest.json", "w") as f:
        json.dump(data, f)
    return artifact_dir


# ──────────────────────────────────────────────────────────────────────────────
# Unit tests
# ──────────────────────────────────────────────────────────────────────────────


class TestEvidenceTypeMapping:
    def test_sector_demand_maps_to_driver(self):
        assert _evidence_type_to_item_type("sector_demand") == "driver"

    def test_supply_chain_maps_to_risk(self):
        assert _evidence_type_to_item_type("supply_chain") == "risk"

    def test_market_confirmation_maps_to_metric(self):
        assert _evidence_type_to_item_type("market_confirmation") == "metric"

    def test_unknown_type_maps_to_concept(self):
        assert _evidence_type_to_item_type("unknown_type") == "concept"


class TestReliabilityMapping:
    def test_high_reliability(self):
        assert _reliability_to_quality(0.9) == "high"

    def test_medium_reliability(self):
        assert _reliability_to_quality(0.65) == "medium"

    def test_low_reliability(self):
        assert _reliability_to_quality(0.45) == "low"

    def test_unverified_reliability(self):
        assert _reliability_to_quality(0.2) == "unverified"


class TestEvidenceToKnowledgeItems:
    def test_converts_single_item(self):
        evidence = [_make_evidence_item()]
        items, sources = _evidence_to_knowledge_items(evidence, "test_domain")

        assert len(items) == 1
        assert len(sources) == 1
        assert items[0].domain_id == "test_domain"
        assert items[0].item_type == "driver"
        assert items[0].tickers == ["NVDA"]
        assert items[0].source_ids == [sources[0].source_id]
        assert len(sources[0].content_sha256 or "") == 64
        assert "normalized evidence record" in sources[0].known_limitations[0]
        assert items[0].metadata["required_lane_eligible"] is False

    def test_converts_multiple_items(self):
        evidence = [
            _make_evidence_item("ev_001", "sector_demand"),
            _make_evidence_item("ev_002", "market_confirmation", stance="negative"),
        ]
        items, sources = _evidence_to_knowledge_items(evidence, "test_domain")

        assert len(items) == 2
        assert len(sources) == 2
        assert items[0].stance_hint == "positive"
        assert items[1].stance_hint == "negative"

    def test_empty_evidence(self):
        items, sources = _evidence_to_knowledge_items([], "test_domain")
        assert len(items) == 0
        assert len(sources) == 0


class TestBuildKnowledgePack:
    def test_builds_from_news_artifact(self, tmp_path):
        news = _make_news_artifact(tmp_path)
        pack = build_knowledge_pack(
            domain_id="semiconductor_ai_infrastructure",
            artifact_paths={"news": str(news)},
        )

        assert pack.domain_id == "semiconductor_ai_infrastructure"
        assert len(pack.items) == 1
        assert len(pack.sources) == 1
        assert pack.items[0].item_type == "driver"

    def test_pack_is_valid(self, tmp_path):
        news = _make_news_artifact(tmp_path)
        pack = build_knowledge_pack(
            domain_id="semiconductor_ai_infrastructure",
            artifact_paths={"news": str(news)},
        )
        # Should not raise
        assert pack.pack_id.startswith("pack_")

    def test_empty_artifacts_raises(self):
        with pytest.raises(ValueError, match="No evidence loaded"):
            build_knowledge_pack(
                domain_id="test_domain",
                artifact_paths={},
            )


class TestPackRoundTrip:
    def test_save_and_load(self, tmp_path):
        news = _make_news_artifact(tmp_path)
        pack = build_knowledge_pack(
            domain_id="semiconductor_ai_infrastructure",
            artifact_paths={"news": str(news)},
        )

        pack_path = tmp_dir = tmp_path / "pack.json"
        save_knowledge_pack(pack, pack_path)

        loaded = load_knowledge_pack(pack_path)
        assert loaded.pack_id == pack.pack_id
        assert len(loaded.items) == len(pack.items)
        assert len(loaded.sources) == len(pack.sources)


class TestPackWithKnowledgeStore:
    def test_add_pack_and_search(self, tmp_path):
        evidence = [_make_evidence_item()]
        items, sources = _evidence_to_knowledge_items(evidence, "test_domain")

        from dean_os.analyst_knowledge.schemas import KnowledgePack
        pack = KnowledgePack(
            pack_id="test_pack_001",
            domain_id="test_domain",
            name="Test Pack",
            sources=sources,
            items=items,
        )

        store = LocalKnowledgeStore(tmp_path / "store")
        store.add_pack(pack)

        retriever = KnowledgeRetriever(store)
        result = retriever.retrieve(
            "AI demand GPU",
            domain_id="test_domain",
        )

        assert len(result.hits) == 1
        assert result.hits[0].item.tickers == ["NVDA"]
