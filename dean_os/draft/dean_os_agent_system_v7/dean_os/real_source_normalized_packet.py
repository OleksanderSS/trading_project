from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from dean_os.draft.dean_os_agent_system_v7.dean_os.artifact_writer import ReviewArtifactWriter
from dean_os.draft.dean_os_agent_system_v7.dean_os.intake_normalizer import normalize_and_chunk
from dean_os.draft.dean_os_agent_system_v7.dean_os.material_loaders import load_research_document
from dean_os.schemas import ResearchChunk, ResearchDocument, utc_now_iso
from dean_os.utils import json_ready

BLOCK_ID = "real_source_offline_normalized_packet_v1"
SCHEMA_VERSION = "real_source_normalized_packet_v1_review_only"

SOURCE_TYPE_ID_BY_DOCUMENT_TYPE = {
    "filing": "company_filings_and_sec_reports",
    "transcript": "earnings_transcripts_and_presentations",
    "news": "news_articles_general_business",
    "article": "specialized_industry_news",
    "report": "industry_reports_and_whitepapers",
    "book": "historical_context_and_case_studies",
}

PARSER_PROFILE_BY_EXTENSION = {
    ".txt": "plain_text_or_markdown_article",
    ".md": "plain_text_or_markdown_article",
    ".markdown": "plain_text_or_markdown_article",
    ".html": "html_article",
    ".htm": "html_article",
    ".json": "json_api_snapshot_payload",
    ".pdf": "pdf_report_or_filing",
    ".docx": "docx_report_or_note",
}

FORBIDDEN_OUTPUT_FLAGS = {
    "live_fetch_allowed",
    "external_api_call_allowed",
    "source_retrieval_allowed_now",
    "claim_extraction_execution_allowed_now",
    "event_extraction_execution_allowed_now",
    "entity_resolution_execution_allowed_now",
    "event_propagation_execution_allowed_now",
    "company_specific_thesis_allowed_now",
    "actual_ratio_interpretation_allowed_now",
    "valuation_allowed",
    "recommendation_allowed",
    "rating_allowed",
    "buy_sell_hold_allowed",
    "price_target_allowed",
    "trade_signal_allowed",
    "position_sizing_allowed",
    "order_generation_allowed",
    "broker_routing_allowed",
    "trading_allowed",
    "production_write_allowed",
    "database_write_allowed",
    "learning_write_allowed",
}


class RealSourceNormalizedPacketBuilder:
    """Build review-only normalized packets from operator-supplied local material."""

    def __init__(self, output_dir: str | Path = "reports/dean_os/real_source_normalized_packet_current"):
        self.output_dir = Path(output_dir)

    def build_from_path(
        self,
        path: str | Path,
        *,
        source_type: str | None = None,
        source_type_id: str | None = None,
        intake_mode_id: str = "operator_supplied_file",
        tickers: list[str] | None = None,
        sectors: list[str] | None = None,
        tags: list[str] | None = None,
        chunk_size: int = 1200,
        save: bool = True,
    ) -> dict[str, Any]:
        source_path = Path(path)
        document = load_research_document(
            source_path,
            source_type=source_type,
            tickers=tickers,
            sectors=sectors,
            tags=tags,
        )
        packet = self._packet_from_document(
            document=document,
            source_path=source_path,
            source_type_id=source_type_id or _source_type_id(document),
            intake_mode_id=intake_mode_id,
            chunk_size=chunk_size,
        )
        payload = self._payload([packet], source_path=source_path)
        if save:
            saved_paths = ReviewArtifactWriter(self.output_dir).write(
                payload=payload,
                markdown=render_real_source_normalized_packet_markdown(payload),
                run_id=payload["run_id"],
            )
            payload["saved_paths"] = saved_paths
        return payload

    def _packet_from_document(
        self,
        *,
        document: ResearchDocument,
        source_path: Path,
        source_type_id: str,
        intake_mode_id: str,
        chunk_size: int,
    ) -> dict[str, Any]:
        chunks = normalize_and_chunk(document, chunk_size=chunk_size)
        content_units, anchors = _content_units_and_anchors(document, chunks)
        quarantine_partitions = _quarantine_partitions(document, chunks)
        normalized_text = [unit["normalized_text"] for unit in content_units]
        suffix = source_path.suffix.lower()
        packet = {
            "packet_id": f"real_norm_{document.document_id[:12]}",
            "parent_source_packet_id": document.document_id,
            "source_type_id": source_type_id,
            "source_name": document.title,
            "intake_mode_id": intake_mode_id,
            "source_material_status": "operator_supplied_review_only_not_promoted_evidence",
            "real_source_content_supplied": True,
            "synthetic_or_fixture_content_used_for_ci": False,
            "provenance": {
                "acquisition_mode": intake_mode_id,
                "original_reference_or_file_id": str(source_path),
                "source_originator_or_publisher": document.metadata.get("publisher") or "operator_supplied_local_material",
                "rights_notes_or_usage_boundary": "Operator-supplied local material; review-only until human evidence promotion.",
                "acquisition_timestamp_utc": document.ingested_at,
                "document_uri": document.uri,
                "published_at": document.published_at,
            },
            "hashes": {
                "source_content_hash": _stable_hash(document.text),
                "normalized_text_hash": _stable_hash(normalized_text),
                "dedupe_key": f"{source_type_id}:{_stable_hash(document.text)[:16]}",
                "duplicate_status": "not_checked_against_external_corpus",
            },
            "parser_profile": {
                "selected_parser_profile": PARSER_PROFILE_BY_EXTENSION.get(suffix, "plain_text_material"),
                "parser_confidence": "operator_supplied_local_parse",
                "parser_warnings": _parser_warnings(document, chunks),
                "source_extension": suffix,
            },
            "content_units": content_units,
            "anchors": anchors,
            "quarantine_partitions": quarantine_partitions,
            "quality_precheck": _quality_precheck(document, quarantine_partitions),
            "routing_prefilter": {
                "candidate_routing_lanes": [
                    "future_claim_extraction_lane",
                    "future_event_extraction_lane",
                    "future_topic_sector_asset_linking_lane",
                ],
                "candidate_topics": sorted(set(document.tags)),
                "candidate_sectors": sorted(set(document.sectors)),
                "candidate_assets_or_entities": sorted(set(document.tickers)),
                "candidate_links_are_final": False,
            },
            "normalization_gate_status": {
                "normalization_readiness_status": "real_source_normalized_packet_ready_for_review_only_validation",
                "blocking_issues": [],
                "human_review_required_before_evidence_promotion": True,
                "promotion_allowed_after_validation_only": False,
            },
            "output_boundary": _output_boundary(),
            "downstream_extraction_outputs": [],
            "evidence_eligible_unit_ids": [
                unit["content_unit_id"]
                for unit in content_units
                if unit["extraction_eligible"]
            ],
            "contract_trace": {
                "derived_from_template_block": "245_review_only_real_source_normalized_packet_fixture_v1",
                "local_builder_block": BLOCK_ID,
                "real_source_content_supplied_by_operator": True,
            },
        }
        packet["packet_hash"] = _stable_hash(packet)
        return packet

    def _payload(self, packets: list[dict[str, Any]], *, source_path: Path) -> dict[str, Any]:
        candidate_assets = sorted({item for packet in packets for item in packet["routing_prefilter"]["candidate_assets_or_entities"]})
        candidate_topics = sorted({item for packet in packets for item in packet["routing_prefilter"]["candidate_topics"]})
        candidate_sectors = sorted({item for packet in packets for item in packet["routing_prefilter"]["candidate_sectors"]})
        quarantine_count = sum(len(packet["quarantine_partitions"]) for packet in packets)
        payload = {
            "run_id": _run_id("real_source_normalized_packet"),
            "block_id": BLOCK_ID,
            "schema_version": SCHEMA_VERSION,
            "created_at_utc": utc_now_iso(),
            "mode": "real_source_offline_normalized_packet_review_only",
            "input": {"source_path": str(source_path)},
            "source_first_pipeline_position": [
                "operator_supplied_real_source_material",
                "real_source_intake",
                "quarantine_aware_intake_normalizer",
                "normalized_packet_review_only_materialization",
                "normalized_packet_validation_gate",
                "future_claim_event_entity_extraction_contract",
                "future_human_review_gate",
            ],
            "summary": {
                "normalized_packet_count": len(packets),
                "source_type_count": len({packet["source_type_id"] for packet in packets}),
                "content_unit_count": sum(len(packet["content_units"]) for packet in packets),
                "anchor_count": sum(len(packet["anchors"]) for packet in packets),
                "quarantine_partition_instance_count": quarantine_count,
                "candidate_asset_or_entity_count": len(candidate_assets),
                "candidate_topic_count": len(candidate_topics),
                "candidate_sector_count": len(candidate_sectors),
                "real_source_content_supplied": True,
                "fixtures_are_production_evidence": False,
                "claim_extraction_performed": False,
                "event_extraction_performed": False,
                "event_propagation_performed": False,
                "company_thesis_generated": False,
            },
            "normalized_packet_rows": packets,
            "candidate_asset_or_entity_index": candidate_assets,
            "candidate_topic_index": candidate_topics,
            "candidate_sector_index": candidate_sectors,
            "safety_flags": _safety_flags(),
            "explicit_non_actions": _explicit_non_actions(),
            "next_recommended_block": "246_review_only_real_source_normalized_packet_validation_gate_v1",
        }
        payload["content_hash"] = _stable_hash({k: v for k, v in payload.items() if k != "content_hash"})
        return payload


def render_real_source_normalized_packet_markdown(payload: dict[str, Any]) -> str:
    summary = payload.get("summary", {})
    lines = [
        "# DEAN-OS Real Source Normalized Packet",
        "",
        f"- Run ID: `{payload.get('run_id')}`",
        f"- Mode: `{payload.get('mode')}`",
        f"- Packets: {summary.get('normalized_packet_count')}",
        f"- Content units: {summary.get('content_unit_count')}",
        f"- Anchors: {summary.get('anchor_count')}",
        f"- Quarantine partitions: {summary.get('quarantine_partition_instance_count')}",
        f"- Candidate assets/entities: `{', '.join(payload.get('candidate_asset_or_entity_index', [])) or 'none'}`",
        f"- Candidate sectors: `{', '.join(payload.get('candidate_sector_index', [])) or 'none'}`",
        "",
        "## Boundary",
        "",
    ]
    lines.extend(f"- {item}" for item in payload.get("explicit_non_actions", []))
    lines.extend(["", "## Packet Samples", ""])
    for packet in payload.get("normalized_packet_rows", [])[:5]:
        lines.extend(
            [
                f"- `{packet.get('packet_id')}` `{packet.get('source_type_id')}`",
                f"  content_units={len(packet.get('content_units', []))}, quarantine={len(packet.get('quarantine_partitions', []))}, promotion_allowed={packet.get('normalization_gate_status', {}).get('promotion_allowed_after_validation_only')}",
            ]
        )
    lines.extend(["", "## Next", "", f"`{payload.get('next_recommended_block')}`"])
    return "\n".join(lines).strip() + "\n"


def _content_units_and_anchors(document: ResearchDocument, chunks: list[ResearchChunk]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    content_units: list[dict[str, Any]] = []
    anchors: list[dict[str, Any]] = []
    for chunk in chunks:
        anchor_id = chunk.metadata.get("anchor_id") or chunk.chunk_id
        unit_id = f"{document.document_id}_unit_{chunk.chunk_index:04d}"
        extraction_eligible = not chunk.quarantine_flags
        content_units.append(
            {
                "content_unit_id": unit_id,
                "content_unit_type_id": "text_chunk",
                "normalized_text": chunk.text,
                "anchor_id": anchor_id,
                "extraction_eligible": extraction_eligible,
                "quarantine_flags": chunk.quarantine_flags,
                "quality_precheck": chunk.quality_precheck,
                "claim_extraction_performed": False,
                "event_extraction_performed": False,
            }
        )
        anchors.append(
            {
                "anchor_id": anchor_id,
                "content_unit_id": unit_id,
                "anchor_type": "normalized_text_chunk",
                "anchor_value": f"{document.uri or document.document_id}#chunk={chunk.chunk_index}",
                "source_span_start": chunk.metadata.get("source_span_start"),
                "source_span_end": chunk.metadata.get("source_span_end"),
                "stable_for_review_replay": True,
            }
        )
    return content_units, anchors


def _quarantine_partitions(document: ResearchDocument, chunks: list[ResearchChunk]) -> list[dict[str, Any]]:
    partitions = []
    for chunk in chunks:
        if not chunk.quarantine_flags:
            continue
        partitions.append(
            {
                "partition_id": f"quarantine_{chunk.chunk_index:04d}",
                "content_unit_anchor_id": chunk.metadata.get("anchor_id") or chunk.chunk_id,
                "quarantine_flags": chunk.quarantine_flags,
                "status": "quarantined_for_sentiment_and_extraction",
                "text_preview": chunk.text[:240],
            }
        )
    if partitions:
        return partitions
    return [
        {
            "partition_id": f"document_flag_{idx:02d}_{flag}",
            "quarantine_flags": [flag],
            "status": "document_level_quarantine_flag",
            "text_preview": "",
        }
        for idx, flag in enumerate(document.quarantine_flags, start=1)
    ]


def _quality_precheck(document: ResearchDocument, quarantine_partitions: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "primary_secondary_classification": "operator_supplied_material_not_yet_promoted_evidence",
        "freshness_status": "requires_published_at_review" if not document.published_at else "timestamp_supplied",
        "methodology_transparency_status": "requires_human_review",
        "conflict_or_bias_risk_status": "requires_human_review",
        "corroboration_requirement": "required_before_real_evidence_promotion",
        "quarantine_status": "quarantine_detected" if quarantine_partitions else "no_quarantine_detected",
        "document_quality_precheck": document.quality_precheck,
    }


def _parser_warnings(document: ResearchDocument, chunks: list[ResearchChunk]) -> list[str]:
    warnings = []
    if not document.published_at:
        warnings.append("missing_published_at")
    if not chunks:
        warnings.append("no_content_units_created")
    if document.quarantine_flags:
        warnings.append("quarantine_flags_detected")
    return warnings


def _output_boundary() -> dict[str, bool]:
    return {
        "claims_emitted_now": False,
        "events_emitted_now": False,
        "entities_resolved_now": False,
        "event_propagation_executed_now": False,
        "company_thesis_generated_now": False,
        "ratio_interpretation_generated_now": False,
        "valuation_output_now": False,
        "recommendation_output_now": False,
        "trade_signal_output_now": False,
    }


def _safety_flags() -> dict[str, bool]:
    flags = dict.fromkeys(FORBIDDEN_OUTPUT_FLAGS, False)
    flags.update(
        {
            "real_source_content_supplied_by_operator": True,
            "real_source_content_promoted_to_evidence": False,
            "normalized_packet_is_production_evidence": False,
            "human_review_required_before_evidence_promotion": True,
        }
    )
    return flags


def _explicit_non_actions() -> list[str]:
    return [
        "No live fetch or external API call is performed.",
        "No claim, event, or entity extraction is executed.",
        "No company thesis, valuation, recommendation, price target, or trade signal is generated.",
        "No learning memory, production config, database, broker, or trading state is written.",
        "Candidate ticker/sector/topic links are routing hints only and are not conclusions.",
    ]


def _source_type_id(document: ResearchDocument) -> str:
    return SOURCE_TYPE_ID_BY_DOCUMENT_TYPE.get(document.source_type, "news_articles_general_business")


def _stable_hash(payload: Any) -> str:
    return hashlib.sha256(json.dumps(json_ready(payload), sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def _run_id(prefix: str) -> str:
    return f"{prefix}_{utc_now_iso().replace(':', '').replace('+', 'Z')}"
