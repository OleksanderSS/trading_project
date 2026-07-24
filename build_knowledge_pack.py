"""Build a knowledge pack from saved producer artifacts.

Converts news, macro, sector market, and fundamental evidence into
KnowledgeItem[] and saves as a pack.json that can be loaded by the
knowledge store.

Usage:
    python build_knowledge_pack.py --domain semiconductor_ai_infrastructure
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# Add project root to path
project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from dean_os.analyst_core.artifact_evidence_loader import ArtifactEvidenceLoader
from dean_os.analyst_knowledge.schemas import KnowledgeItem, KnowledgePack, KnowledgeSource
from dean_os.analyst_knowledge.pack_loader import save_knowledge_pack
from dean_os.analysts.schemas import AnalystEvidenceItem
from dean_os.utils import sha256_json


def _evidence_to_knowledge_items(
    evidence: list[AnalystEvidenceItem],
    domain_id: str,
) -> tuple[list[KnowledgeItem], list[KnowledgeSource]]:
    """Convert AnalystEvidenceItem list to KnowledgeItem[] and KnowledgeSource[]."""
    items: list[KnowledgeItem] = []
    sources: list[KnowledgeSource] = []
    source_ids_seen: set[str] = set()

    for ev in evidence:
        # Create a source for each evidence item
        source_id = f"src_{ev.evidence_id}"
        if source_id not in source_ids_seen:
            source_ids_seen.add(source_id)
            sources.append(KnowledgeSource(
                source_id=source_id,
                title=ev.summary[:100],
                source_type=ev.source_type,
                reference=ev.source,
                published_at=ev.published_at,
                retrieved_at=ev.as_of,
                content_sha256=sha256_json(ev.summary),
                known_limitations=[
                    "This is a normalized evidence record summary, not the verbatim raw source content."
                ],
                reliability=_reliability_to_quality(ev.reliability_score),
            ))

        # Create knowledge item
        item_type = _evidence_type_to_item_type(ev.evidence_type)
        stance = ev.stance_hint if ev.stance_hint in ("positive", "negative", "neutral", "mixed") else "unknown"

        items.append(KnowledgeItem(
            item_id=f"ki_{ev.evidence_id}",
            domain_id=domain_id,
            item_type=item_type,
            title=ev.summary[:120],
            body=ev.summary,
            stance_hint=stance,
            tags=_extract_tags(ev),
            tickers=ev.tickers or [],
            sectors=ev.sectors or [domain_id],
            metrics=[ev.evidence_type],
            source_ids=[source_id],
            confidence=ev.reliability_score,
            importance=_importance_from_evidence(ev),
            updated_at=ev.as_of or datetime.now(UTC).isoformat(),
            metadata={
                "evidence_type": ev.evidence_type,
                "source_type": ev.source_type,
                "directness": ev.directness,
                "freshness_score": ev.freshness_score,
                "strength": ev.strength,
                "required_lane_eligible": bool(ev.provenance.get("required_lane_eligible", False)),
            },
        ))

    return items, sources


def _evidence_type_to_item_type(evidence_type: str) -> str:
    """Map evidence_type to KnowledgeItemType."""
    mapping = {
        "sector_demand": "driver",
        "capex_cycle": "driver",
        "supply_chain": "risk",
        "policy_or_geopolitical": "risk",
        "market_confirmation": "metric",
        "earnings_guidance": "metric",
        "order_backlog": "metric",
        "macro_context": "metric",
        "fundamental_context": "metric",
        "valuation_context": "metric",
        "relative_strength": "metric",
        "analyst_revision": "metric",
    }
    return mapping.get(evidence_type, "concept")


def _reliability_to_quality(score: float) -> str:
    """Map reliability_score to KnowledgeQuality."""
    if score >= 0.8:
        return "high"
    elif score >= 0.6:
        return "medium"
    elif score >= 0.4:
        return "low"
    return "unverified"


def _extract_tags(ev: AnalystEvidenceItem) -> list[str]:
    """Extract tags from evidence item."""
    tags = set()
    tags.add(ev.evidence_type)
    tags.add(ev.source_type)
    if ev.stance_hint:
        tags.add(ev.stance_hint)
    return sorted(tags)


def _importance_from_evidence(ev: AnalystEvidenceItem) -> int:
    """Map evidence importance to 1-5 scale."""
    # Higher reliability and strength = higher importance
    score = ev.reliability_score * 0.6 + ev.strength * 0.4
    if score >= 0.8:
        return 5
    elif score >= 0.6:
        return 4
    elif score >= 0.4:
        return 3
    elif score >= 0.2:
        return 2
    return 1


def build_knowledge_pack(
    domain_id: str,
    artifact_paths: dict[str, str],
    *,
    pack_name: str | None = None,
    as_of: str = "",
) -> KnowledgePack:
    """Build a KnowledgePack from saved producer artifacts.

    Args:
        domain_id: Domain identifier.
        artifact_paths: Dict mapping artifact type to directory path.
        pack_name: Optional name for the pack.
        as_of: Point-in-time cutoff.

    Returns:
        KnowledgePack with items and sources.
    """
    as_of = as_of or datetime.now(UTC).isoformat()
    loader = ArtifactEvidenceLoader()

    # Load evidence from artifacts
    if "runtime" in artifact_paths:
        evidence = loader.from_runtime_artifact(
            Path(artifact_paths["runtime"]),
            domain_id=domain_id,
        )
    else:
        evidence = loader.from_producer_artifacts(
            news_path=Path(artifact_paths["news"]) if "news" in artifact_paths else None,
            macro_path=Path(artifact_paths["macro"]) if "macro" in artifact_paths else None,
            sector_market_path=Path(artifact_paths["sector_market"]) if "sector_market" in artifact_paths else None,
            policy_path=Path(artifact_paths["policy"]) if "policy" in artifact_paths else None,
            fundamental_path=Path(artifact_paths["fundamental"]) if "fundamental" in artifact_paths else None,
            domain_id=domain_id,
            as_of=as_of,
        )

    if not evidence:
        raise ValueError(f"No evidence loaded from artifacts for domain={domain_id}")

    # Convert to knowledge items
    items, sources = _evidence_to_knowledge_items(evidence, domain_id)

    # Build pack
    now = datetime.now(UTC).isoformat()
    pack = KnowledgePack(
        pack_id=f"pack_{domain_id}_{now[:10]}",
        domain_id=domain_id,
        name=pack_name or f"{domain_id} knowledge pack",
        version="0.1.0",
        description=f"Knowledge pack built from saved producer artifacts for {domain_id}",
        tags=[domain_id, "auto-generated", "from-artifacts"],
        tickers=[],
        sources=sources,
        items=items,
    )

    return pack


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build a knowledge pack from saved producer artifacts.",
    )
    parser.add_argument("--domain", type=str, required=True, help="Domain ID.")
    parser.add_argument("--news-artifact", type=str, default=None, help="News artifact path.")
    parser.add_argument("--macro-artifact", type=str, default=None, help="Macro artifact path.")
    parser.add_argument("--sector-market-artifact", type=str, default=None, help="Sector market artifact path.")
    parser.add_argument("--policy-artifact", type=str, default=None, help="Policy artifact path.")
    parser.add_argument("--fundamental-artifact", type=str, default=None, help="Fundamental artifact path.")
    parser.add_argument("--runtime-artifact", type=str, default=None, help="Runtime artifact path.")
    parser.add_argument("--as-of", type=str, default="", help="Point-in-time cutoff.")
    parser.add_argument("--output", type=str, default=None, help="Output path for pack.json.")

    args = parser.parse_args(argv)

    # Build artifact paths
    artifact_paths: dict[str, str] = {}
    if args.runtime_artifact:
        artifact_paths["runtime"] = args.runtime_artifact
    else:
        if args.news_artifact:
            artifact_paths["news"] = args.news_artifact
        if args.macro_artifact:
            artifact_paths["macro"] = args.macro_artifact
        if args.sector_market_artifact:
            artifact_paths["sector_market"] = args.sector_market_artifact
        if args.policy_artifact:
            artifact_paths["policy"] = args.policy_artifact
        if args.fundamental_artifact:
            artifact_paths["fundamental"] = args.fundamental_artifact

    if not artifact_paths:
        parser.error("Provide at least one artifact path.")

    # Build pack
    try:
        pack = build_knowledge_pack(
            domain_id=args.domain,
            artifact_paths=artifact_paths,
            as_of=args.as_of,
        )
    except (FileNotFoundError, ValueError) as e:
        print(f"Error building pack: {e}", file=sys.stderr)
        return 1

    # Save
    output_path = args.output or f"dean_os/analyst_knowledge/packs/{args.domain}/pack.json"
    save_knowledge_pack(pack, output_path)

    print(f"Knowledge pack saved to {output_path}")
    print(f"  Items: {len(pack.items)}")
    print(f"  Sources: {len(pack.sources)}")
    print(f"  Domain: {pack.domain_id}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
