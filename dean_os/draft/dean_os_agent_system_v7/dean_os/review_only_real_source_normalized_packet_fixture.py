
from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

BLOCK_ID = '245_review_only_real_source_normalized_packet_fixture_v1'
UPSTREAM_BLOCK = '244_review_only_real_source_packet_intake_normalizer_contract_v1'
SCHEMA_VERSION = 'real_source_normalized_packet_fixture_v1_review_only'

FORBIDDEN_FALSE_FLAGS = {
    'live_fetch_allowed', 'external_api_call_allowed', 'source_retrieval_allowed_now',
    'normalization_execution_against_external_sources_allowed_now',
    'claim_extraction_execution_allowed_now', 'event_extraction_execution_allowed_now',
    'entity_resolution_execution_allowed_now', 'event_propagation_execution_allowed_now',
    'company_specific_thesis_allowed_now', 'actual_ratio_interpretation_allowed_now',
    'valuation_allowed', 'recommendation_allowed', 'rating_allowed', 'buy_sell_hold_allowed',
    'price_target_allowed', 'trade_signal_allowed', 'position_sizing_allowed',
    'order_generation_allowed', 'broker_routing_allowed', 'trading_allowed',
    'production_write_allowed', 'database_write_allowed', 'tool_write_allowed',
    'publication_allowed', 'autonomous_loop_allowed', 'scheduler_allowed',
    'codex_execution_allowed', 'patch_application_allowed',
}

SOURCE_TYPE_TO_FIXTURE = {
    'company_filings_and_sec_reports': {
        'parser_profile_id': 'pdf_report_or_filing',
        'intake_mode_id': 'offline_archive_path',
        'candidate_topics': ['company_filing', 'financial_disclosure'],
        'candidate_sectors': ['semiconductors'],
        'candidate_assets_or_entities': ['AMD'],
        'content_units': [
            ('title_or_headline', 'Offline fixture filing packet for parser/schema validation only.'),
            ('paragraph', 'Fixture management discussion unit; not sourced from a real filing.'),
            ('table', 'Fixture financial table unit with preserved headers and units but no claim extraction.'),
        ],
        'quarantine': [],
    },
    'earnings_transcripts_and_presentations': {
        'parser_profile_id': 'transcript',
        'intake_mode_id': 'user_uploaded_file',
        'candidate_topics': ['earnings_transcript', 'ai_capex_cycle'],
        'candidate_sectors': ['ai_big_tech', 'semiconductors'],
        'candidate_assets_or_entities': ['NVDA', 'AMD', 'MSFT'],
        'content_units': [
            ('title_or_headline', 'Offline fixture earnings transcript packet.'),
            ('speaker_turn', 'Fixture prepared remarks unit; no speaker claim promoted.'),
            ('speaker_turn', 'Fixture Q&A unit; no event extracted in this block.'),
        ],
        'quarantine': ['legal_disclaimer'],
    },
    'news_articles_general_business': {
        'parser_profile_id': 'plain_text_or_markdown_article',
        'intake_mode_id': 'user_pasted_text',
        'candidate_topics': ['interest_rates', 'consumer_demand'],
        'candidate_sectors': ['market_etfs', 'finance', 'consumer_staples'],
        'candidate_assets_or_entities': ['SPY', 'IWM', 'JPM', 'BAC', 'WMT'],
        'content_units': [
            ('title_or_headline', 'Offline fixture general business news packet.'),
            ('paragraph', 'Fixture paragraph about broad market context; not real news.'),
        ],
        'quarantine': ['advertising_navigation_author_bio'],
    },
    'specialized_industry_news': {
        'parser_profile_id': 'plain_text_or_markdown_article',
        'intake_mode_id': 'connector_fetched_reference',
        'candidate_topics': ['rare_earths_critical_minerals', 'semiconductor_export_controls'],
        'candidate_sectors': ['semiconductors', 'ai_big_tech'],
        'candidate_assets_or_entities': ['NVDA', 'AMD', 'AAPL', 'TSM'],
        'content_units': [
            ('title_or_headline', 'Offline fixture specialized industry news packet.'),
            ('paragraph', 'Fixture rare-earth supply pressure context; not external evidence.'),
            ('quote', 'Fixture quoted-source unit with no claim extraction.'),
        ],
        'quarantine': [],
    },
    'analyst_research_and_broker_notes': {
        'parser_profile_id': 'pdf_report_or_filing',
        'intake_mode_id': 'user_uploaded_file',
        'candidate_topics': ['analyst_note', 'ai_capex_cycle'],
        'candidate_sectors': ['ai_big_tech', 'semiconductors'],
        'candidate_assets_or_entities': ['NVDA', 'MSFT', 'AMD'],
        'content_units': [
            ('title_or_headline', 'Offline fixture analyst note packet.'),
            ('paragraph', 'Fixture analyst narrative unit; not accepted as DEAN-OS conclusion.'),
            ('disclaimer_rating_price_target_language', 'Fixture third-party rating/price target language quarantined.'),
        ],
        'quarantine': ['third_party_rating_or_price_target', 'legal_disclaimer'],
    },
    'industry_reports_and_whitepapers': {
        'parser_profile_id': 'pdf_report_or_filing',
        'intake_mode_id': 'offline_archive_path',
        'candidate_topics': ['industry_report', 'semiconductor_supply_chain'],
        'candidate_sectors': ['semiconductors', 'ai_big_tech'],
        'candidate_assets_or_entities': ['TSM', 'NVDA', 'AMD', 'INTC'],
        'content_units': [
            ('title_or_headline', 'Offline fixture industry report packet.'),
            ('paragraph', 'Fixture methodology overview unit.'),
            ('figure_caption', 'Fixture chart caption unit with no underlying figure parsing.'),
        ],
        'quarantine': [],
    },
    'macro_data_and_central_bank_context': {
        'parser_profile_id': 'json_api_snapshot_payload',
        'intake_mode_id': 'api_snapshot_payload',
        'candidate_topics': ['interest_rates_discount_rates'],
        'candidate_sectors': ['market_etfs', 'finance', 'ai_big_tech'],
        'candidate_assets_or_entities': ['SPY', 'QQQ', 'IWM', 'JPM', 'GS'],
        'content_units': [
            ('title_or_headline', 'Offline fixture macro snapshot packet.'),
            ('time_series_observation', 'Fixture rate observation row; no trading signal.'),
        ],
        'quarantine': [],
    },
    'policy_regulation_and_law': {
        'parser_profile_id': 'policy_or_legal_document',
        'intake_mode_id': 'connector_fetched_reference',
        'candidate_topics': ['platform_regulation_antitrust_privacy', 'semiconductor_export_controls'],
        'candidate_sectors': ['ai_big_tech', 'semiconductors'],
        'candidate_assets_or_entities': ['AAPL', 'GOOGL', 'AMZN', 'NVDA', 'TSM'],
        'content_units': [
            ('title_or_headline', 'Offline fixture policy/regulation packet.'),
            ('legal_or_policy_clause', 'Fixture policy clause unit with jurisdiction capture.'),
        ],
        'quarantine': [],
    },
    'geopolitics_and_security_context': {
        'parser_profile_id': 'plain_text_or_markdown_article',
        'intake_mode_id': 'user_pasted_text',
        'candidate_topics': ['geopolitics_security', 'semiconductor_export_controls'],
        'candidate_sectors': ['semiconductors', 'energy', 'market_etfs'],
        'candidate_assets_or_entities': ['TSM', 'NVDA', 'XOM', 'SPY'],
        'content_units': [
            ('title_or_headline', 'Offline fixture geopolitics/security context packet.'),
            ('paragraph', 'Fixture regional risk context; no event propagation.'),
        ],
        'quarantine': [],
    },
    'historical_context_and_case_studies': {
        'parser_profile_id': 'historical_case_note',
        'intake_mode_id': 'offline_archive_path',
        'candidate_topics': ['historical_supply_shock_analogy'],
        'candidate_sectors': ['semiconductors', 'energy', 'consumer_staples'],
        'candidate_assets_or_entities': ['AMD', 'NVDA', 'XOM', 'WMT'],
        'content_units': [
            ('title_or_headline', 'Offline fixture historical context packet.'),
            ('paragraph', 'Fixture case-study paragraph with explicit timeframe limitation.'),
        ],
        'quarantine': ['stale_without_covered_period'],
    },
    'commodity_and_input_price_series': {
        'parser_profile_id': 'json_api_snapshot_payload',
        'intake_mode_id': 'api_snapshot_payload',
        'candidate_topics': ['rare_earths_critical_minerals', 'energy_price_shock'],
        'candidate_sectors': ['semiconductors', 'energy', 'ai_big_tech'],
        'candidate_assets_or_entities': ['AMD', 'NVDA', 'AAPL', 'XOM'],
        'content_units': [
            ('title_or_headline', 'Offline fixture commodity/input price snapshot packet.'),
            ('time_series_observation', 'Fixture commodity observation row with provider/time window placeholder.'),
        ],
        'quarantine': [],
    },
    'ticker_market_price_volume_series': {
        'parser_profile_id': 'market_price_volume_snapshot',
        'intake_mode_id': 'api_snapshot_payload',
        'candidate_topics': ['ticker_market_snapshot'],
        'candidate_sectors': ['market_etfs', 'semiconductors', 'ai_big_tech'],
        'candidate_assets_or_entities': ['SPY', 'QQQ', 'AMD', 'NVDA', 'AAPL'],
        'content_units': [
            ('title_or_headline', 'Offline fixture ticker market snapshot packet.'),
            ('time_series_observation', 'Fixture price/volume observation row; no trade signal.'),
        ],
        'quarantine': ['market_snapshot_without_observation_window'],
    },
}


def _load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding='utf-8'))


def _stable_hash(payload: Any) -> str:
    data = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode('utf-8')
    return hashlib.sha256(data).hexdigest()


def _make_content_units(packet_id: str, specs: list[tuple[str, str]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    units = []
    anchors = []
    for idx, (unit_type, text) in enumerate(specs, start=1):
        unit_id = f'{packet_id}_unit_{idx:02d}'
        anchor_id = f'{packet_id}_anchor_{idx:02d}'
        units.append({
            'content_unit_id': unit_id,
            'content_unit_type_id': unit_type,
            'normalized_text': text,
            'anchor_id': anchor_id,
            'extraction_eligible': unit_type not in {'disclaimer_rating_price_target_language', 'boilerplate_or_ad'},
            'fixture_content_status': 'offline_fixture_not_external_evidence',
            'claim_extraction_performed': False,
            'event_extraction_performed': False,
        })
        anchors.append({
            'anchor_id': anchor_id,
            'content_unit_id': unit_id,
            'anchor_type': 'fixture_sequence_anchor',
            'anchor_value': f'fixture://{packet_id}/unit/{idx:02d}',
            'stable_for_fixture_replay': True,
        })
    return units, anchors


def _packet_for_source_type(source_type: str, source_name: str, contract_row: dict[str, Any], idx: int) -> dict[str, Any]:
    cfg = SOURCE_TYPE_TO_FIXTURE[source_type]
    packet_id = f'norm245_{idx:02d}_{source_type}'
    units, anchors = _make_content_units(packet_id, cfg['content_units'])
    evidence_eligible_unit_ids = [u['content_unit_id'] for u in units if u['extraction_eligible']]
    quarantined = cfg['quarantine']
    packet_core = {
        'packet_id': packet_id,
        'parent_source_packet_id': f'intake_fixture_{idx:02d}_{source_type}',
        'source_type_id': source_type,
        'source_name': source_name,
        'intake_mode_id': cfg['intake_mode_id'],
        'source_fixture_status': 'offline_normalized_packet_fixture_not_real_external_source',
        'real_source_path_supported_by_contract': True,
        'real_source_content_supplied_in_245': False,
        'synthetic_or_fixture_content_used_for_ci': True,
        'provenance': {
            'acquisition_mode': cfg['intake_mode_id'],
            'original_reference_or_file_id': f'fixture_reference://block245/{source_type}',
            'source_originator_or_publisher': 'fixture_originator_for_schema_validation_only',
            'rights_notes_or_usage_boundary': 'No external copyrighted source text included; fixture is not evidence.',
            'acquisition_timestamp_utc': 'fixture_static_timestamp_not_market_time',
        },
        'hashes': {
            'source_content_hash': _stable_hash(cfg['content_units']),
            'normalized_text_hash': _stable_hash([u['normalized_text'] for u in units]),
            'dedupe_key': f'block245:{source_type}',
            'duplicate_status': 'not_checked_against_external_corpus',
        },
        'parser_profile': {
            'selected_parser_profile': cfg['parser_profile_id'],
            'parser_confidence': 'fixture_high_schema_only',
            'parser_warnings': ['fixture_content_not_real_source', 'no_external_corpus_deduplication'],
        },
        'content_units': units,
        'anchors': anchors,
        'quarantine_partitions': [{'partition_id': q, 'source_type_id': source_type, 'status': 'quarantined_in_fixture'} for q in quarantined],
        'quality_precheck': {
            'primary_secondary_classification': 'fixture_not_evidence',
            'freshness_status': 'not_applicable_fixture',
            'methodology_transparency_status': 'schema_fixture_only',
            'conflict_or_bias_risk_status': 'not_assessed_for_fixture',
            'corroboration_requirement': 'required_before_real_evidence_promotion',
        },
        'routing_prefilter': {
            'candidate_routing_lanes': ['future_claim_extraction_lane', 'future_event_extraction_lane', 'future_topic_sector_asset_linking_lane'],
            'candidate_topics': cfg['candidate_topics'],
            'candidate_sectors': cfg['candidate_sectors'],
            'candidate_assets_or_entities': cfg['candidate_assets_or_entities'],
            'candidate_links_are_final': False,
        },
        'normalization_gate_status': {
            'normalization_readiness_status': 'fixture_normalized_packet_ready_for_validation_only',
            'blocking_issues': [],
            'human_review_required_before_evidence_promotion': True,
            'promotion_allowed_after_validation_only': False,
        },
        'output_boundary': {
            'claims_emitted_now': False,
            'events_emitted_now': False,
            'entities_resolved_now': False,
            'event_propagation_executed_now': False,
            'company_thesis_generated_now': False,
            'recommendation_output_now': False,
            'trade_signal_output_now': False,
        },
        'downstream_extraction_outputs': [],
        'evidence_eligible_unit_ids': evidence_eligible_unit_ids,
        'contract_trace': {
            'upstream_block': UPSTREAM_BLOCK,
            'source_type_contract_status': contract_row.get('normalizer_contract_status'),
            'field_group_ids_inherited': contract_row.get('field_group_ids_required', []),
        },
    }
    packet_core['packet_hash'] = _stable_hash(packet_core)
    return packet_core


def build_fixture(upstream_contract_path: str | Path) -> dict[str, Any]:
    upstream = _load_json(upstream_contract_path)
    now = datetime.now(UTC).isoformat()

    source_rows = upstream['source_type_normalizer_contract_rows']
    packets = []
    for idx, row in enumerate(source_rows, start=1):
        source_type = row['source_type_id']
        if source_type not in SOURCE_TYPE_TO_FIXTURE:
            raise ValueError(f'Missing fixture mapping for source type: {source_type}')
        packets.append(_packet_for_source_type(source_type, row['source_name'], row, idx))

    content_unit_count = sum(len(p['content_units']) for p in packets)
    anchor_count = sum(len(p['anchors']) for p in packets)
    quarantined_unit_count = sum(len(p['quarantine_partitions']) for p in packets)
    asset_link_candidates = sorted({a for p in packets for a in p['routing_prefilter']['candidate_assets_or_entities']})
    topic_candidates = sorted({t for p in packets for t in p['routing_prefilter']['candidate_topics']})
    sector_candidates = sorted({s for p in packets for s in p['routing_prefilter']['candidate_sectors']})

    validation_checks = []
    def check(check_id: str, passed: bool, details: dict[str, Any] | None = None):
        validation_checks.append({'check_id': check_id, 'status': 'passed' if passed else 'failed', 'details': details or {}})

    check('all_12_source_types_have_normalized_packet_fixture', len(packets) == 12)
    check('all_packets_have_content_units', all(len(p['content_units']) > 0 for p in packets))
    check('all_packets_have_anchors', all(len(p['anchors']) == len(p['content_units']) for p in packets))
    check('all_packets_preserve_hashes', all(p['hashes']['source_content_hash'] and p['hashes']['normalized_text_hash'] and p['packet_hash'] for p in packets))
    check('all_packets_have_quality_precheck', all('quality_precheck' in p for p in packets))
    check('routing_links_are_candidate_only', all(p['routing_prefilter']['candidate_links_are_final'] is False for p in packets))
    check('downstream_outputs_empty', all(p['downstream_extraction_outputs'] == [] for p in packets))
    check('real_source_not_claimed_as_supplied', all(p['real_source_content_supplied_in_245'] is False for p in packets))
    check('fixtures_not_evidence', all(p['source_fixture_status'] == 'offline_normalized_packet_fixture_not_real_external_source' for p in packets))
    check('human_review_required_before_evidence_promotion', all(p['normalization_gate_status']['human_review_required_before_evidence_promotion'] is True for p in packets))

    safety_flags = dict.fromkeys(FORBIDDEN_FALSE_FLAGS, False)
    safety_flags.update({
        'normalized_packet_fixtures_materialized_in_245': True,
        'normalized_packet_fixtures_are_real_external_sources': False,
        'normalized_packet_fixtures_are_production_evidence': False,
        'real_source_intake_path_supported': True,
        'real_source_content_supplied_in_245': False,
        'synthetic_fixture_is_production_evidence': False,
    })

    safety_assertions = []
    for flag in sorted(FORBIDDEN_FALSE_FLAGS):
        safety_assertions.append({'assertion_id': f'forbidden_flag_false__{flag}', 'status': 'passed' if safety_flags[flag] is False else 'failed', 'observed': safety_flags[flag]})
    for p in packets:
        safety_assertions.append({'assertion_id': f'no_downstream_outputs__{p["packet_id"]}', 'status': 'passed' if not p['downstream_extraction_outputs'] else 'failed'})
        safety_assertions.append({'assertion_id': f'not_production_evidence__{p["packet_id"]}', 'status': 'passed' if p['quality_precheck']['primary_secondary_classification'] == 'fixture_not_evidence' else 'failed'})

    failed_validation_check_count = sum(1 for c in validation_checks if c['status'] != 'passed')
    failed_safety_assertion_count = sum(1 for a in safety_assertions if a['status'] != 'passed')
    validation_errors = []
    if failed_validation_check_count:
        validation_errors.append('One or more normalized packet fixture validation checks failed.')
    if failed_safety_assertion_count:
        validation_errors.append('One or more safety assertions failed.')

    result = {
        'block_id': BLOCK_ID,
        'schema_version': SCHEMA_VERSION,
        'created_at_utc': now,
        'upstream_blocks': [UPSTREAM_BLOCK, '240_review_only_assets_universe_and_multisource_intelligence_contract_v1'],
        'fixture_status': 'review_only_normalized_packet_fixture_materialized_for_validation_not_evidence',
        'source_first_pipeline_position': [
            'real_source_intake',
            'intake_normalizer_contract',
            'normalized_packet_fixture_materialization',
            'future_normalized_packet_validation_gate',
            'future_claim_event_entity_extraction_contract',
            'future_topic_sector_asset_candidate_linking',
            'future_financial_implication_candidate_mapping',
            'future_human_review_gate',
        ],
        'summary': {
            'normalized_packet_fixture_count': len(packets),
            'source_type_count': len({p['source_type_id'] for p in packets}),
            'content_unit_count': content_unit_count,
            'anchor_count': anchor_count,
            'quarantine_partition_instance_count': quarantined_unit_count,
            'candidate_asset_or_entity_count': len(asset_link_candidates),
            'candidate_topic_count': len(topic_candidates),
            'candidate_sector_count': len(sector_candidates),
            'real_source_content_supplied_in_245': False,
            'fixtures_are_production_evidence': False,
            'claim_extraction_performed_in_245': False,
            'event_extraction_performed_in_245': False,
            'event_propagation_performed_in_245': False,
            'company_thesis_generated_in_245': False,
        },
        'normalized_packet_fixture_rows': packets,
        'candidate_asset_or_entity_index': asset_link_candidates,
        'candidate_topic_index': topic_candidates,
        'candidate_sector_index': sector_candidates,
        'real_data_usage_note': {
            'why_fixtures_exist': 'They validate schema, parsing boundaries, anchors, quarantine rules, and downstream safety without depending on live or copyrighted external sources.',
            'how_real_data_enters_next': 'A real article/report/filing/API snapshot must be supplied by upload, paste, connector reference, API snapshot payload, or offline archive path, then normalized using the same packet schema.',
            'not_allowed_to_infer_from_fixture': 'No investment thesis, valuation, recommendation, price target, or trade signal may be derived from these fixtures.',
        },
        'safety_flags': safety_flags,
        'validation_checks': validation_checks,
        'safety_assertions': safety_assertions,
        'validation_check_count': len(validation_checks),
        'safety_assertion_count': len(safety_assertions),
        'validation_error_count': len(validation_errors),
        'failed_validation_check_count': failed_validation_check_count,
        'failed_safety_assertion_count': failed_safety_assertion_count,
        'validation_errors': validation_errors,
        'content_hash': '',
        'next_recommended_block': '246_review_only_real_source_normalized_packet_validation_gate_v1',
        'next_chat_recommended': True,
    }
    result['content_hash'] = _stable_hash({k: v for k, v in result.items() if k != 'content_hash'})
    return result
