import hashlib

from dean_os.material_loaders import partition_text_by_quarantine
from dean_os.schemas import ResearchChunk, ResearchDocument, SourceCitation


def generate_anchor_id(document_id: str, chunk_index: int, text: str) -> str:
    """Generate a stable anchor ID for provenance."""
    hasher = hashlib.sha256()
    hasher.update(f"{document_id}_{chunk_index}_{text}".encode())
    return hasher.hexdigest()[:16]

def generate_content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

def normalize_and_chunk(document: ResearchDocument, chunk_size: int = 1200) -> list[ResearchChunk]:
    """
    Robust intake normalizer that:
    1. Identifies quarantine zones using RegEx.
    2. Segments text into safe vs quarantined blocks.
    3. Chunks the blocks and assigns stable anchor IDs and hashes.
    """
    if not document.text:
        return []

    blocks = partition_text_by_quarantine(document.text)
    chunks: list[ResearchChunk] = []
    chunk_index = 0

    for block in blocks:
        block_text = " ".join(block["text"].split())
        if not block_text:
            continue

        start = 0
        while start < len(block_text):
            end = min(start + chunk_size, len(block_text))
            if end < len(block_text):
                boundary = block_text.rfind(" ", start, end)
                if boundary > start:
                    end = boundary

            chunk_text = block_text[start:end].strip()
            if chunk_text:
                anchor_id = generate_anchor_id(document.document_id, chunk_index, chunk_text)
                content_hash = generate_content_hash(chunk_text)

                citation = SourceCitation(
                    source_id=document.document_id,
                    source_type=document.source_type,
                    title=document.title,
                    uri=document.uri,
                    locator=f"anchor:{anchor_id}",
                    excerpt=chunk_text[:280],
                    timestamp=document.published_at,
                )

                chunks.append(
                    ResearchChunk(
                        chunk_id=anchor_id, # Use anchor_id as the primary chunk_id for stability
                        document_id=document.document_id,
                        chunk_index=chunk_index,
                        text=chunk_text,
                        token_estimate=max(1, len(chunk_text.split())),
                        citations=[citation],
                        metadata={
                            "title": document.title,
                            "source_type": document.source_type,
                            "source_content_hash": content_hash,
                            "normalized_text_hash": content_hash,
                            "anchor_id": anchor_id,
                            "source_span_start": block["start"],
                            "source_span_end": block["end"],
                            "quarantine_flags": block["quarantine_flags"],
                            "quality_precheck": block["quality_precheck"],
                        },
                        quarantine_flags=block["quarantine_flags"],
                        quality_precheck=block["quality_precheck"],
                    )
                )
                chunk_index += 1
            start = end + 1

    return chunks
