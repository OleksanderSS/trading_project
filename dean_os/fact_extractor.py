from __future__ import annotations

import re
from typing import Protocol

from dean_os.schemas import ExtractedEvent, ExtractedFact, ResearchChunk

# Simple pattern sets for a rule-based approach (can be replaced by LLMs later)
CLAIM_PATTERNS = [
    r"(?i)\b(revenue|sales|income|eps|margins?)\s+(?:grew|declined|increased|decreased|jumped|fell|rose|dropped)\b",
    r"(?i)\b(?:launched|introduced|released)\b\s+(?:new product|service|platform|model)",
    r"(?i)\b(?:expects?|guides?|projects?)\b\s+(?:growth|decline|revenue|earnings)",
]

EVENT_PATTERNS = {
    "acquisition": r"(?i)\b(?:acquired|merger with|buying)\b",
    "lawsuit": r"(?i)\b(?:lawsuit|sued|litigation|settlement)\b",
    "regulatory": r"(?i)\b(?:fda approved|sec probe|antitrust|investigation)\b",
    "leadership_change": r"(?i)\b(?:stepped down|resigned|appointed as ceo|new cfo)\b",
    "layoffs": r"(?i)\b(?:layoffs|cut jobs|reducing headcount|restructuring)\b",
}


class FactExtractionContract(Protocol):
    def extract(self, chunks: list[ResearchChunk]) -> tuple[list[ExtractedFact], list[ExtractedEvent]]:
        ...


class RuleBasedFactExtractor:
    """Deterministic extraction of facts and events.
    
    Provides structured claims and events linked to stable Anchor IDs
    before plugging in a full LLM extractor.
    """

    def extract(self, chunks: list[ResearchChunk]) -> tuple[list[ExtractedFact], list[ExtractedEvent]]:
        facts: list[ExtractedFact] = []
        events: list[ExtractedEvent] = []

        for chunk in chunks:
            # Skip chunks that were quarantined (e.g., disclaimers, third-party ratings)
            if any(flag in ("legal_disclaimer", "third_party_rating", "advertising_navigation_author_bio") for flag in chunk.quarantine_flags):
                continue

            text = chunk.text

            # Find claims
            for pattern in CLAIM_PATTERNS:
                for match in re.finditer(pattern, text):
                    # We just use the sentence containing the match as the description
                    description = _extract_sentence(text, match.start())

                    # Avoid adding duplicates from the same chunk
                    if not any(f.description == description for f in facts):
                        facts.append(
                            ExtractedFact(
                                chunk_id=chunk.chunk_id,
                                fact_type="claim",
                                description=description,
                                is_trading_signal=False,
                                confidence=0.8,
                                source_citation=chunk.citations[0] if chunk.citations else None
                            )
                        )

            # Find events
            for event_type, pattern in EVENT_PATTERNS.items():
                for match in re.finditer(pattern, text):
                    description = _extract_sentence(text, match.start())

                    if not any(e.description == description for e in events):
                        events.append(
                            ExtractedEvent(
                                chunk_id=chunk.chunk_id,
                                event_type=event_type,
                                description=description,
                                is_trading_signal=False,
                                confidence=0.8,
                                source_citation=chunk.citations[0] if chunk.citations else None
                            )
                        )

        return facts, events


def _extract_sentence(text: str, match_index: int) -> str:
    """Helper to extract the sentence boundary around a regex match."""
    # Simple heuristic to find start of sentence
    start = max(0, text.rfind(".", 0, match_index) + 1)
    # Simple heuristic to find end of sentence
    end = text.find(".", match_index)
    if end == -1:
        end = len(text)
    else:
        end += 1
    return text[start:end].strip()
