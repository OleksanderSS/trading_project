from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from typing import Any

BLOCK_ID = '246_review_only_real_source_normalized_packet_validation_gate_v1'
UPSTREAM_BLOCK = '245_review_only_real_source_normalized_packet_fixture_v1'
SCHEMA_VERSION = 'real_source_normalized_packet_validation_gate_v1_review_only'

def _stable_hash(payload: Any) -> str:
    data = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode('utf-8')
    return hashlib.sha256(data).hexdigest()

def validate_packet(packet: dict[str, Any]) -> dict[str, Any]:
    validation_checks = []

    def check(check_id: str, passed: bool, details: dict[str, Any] | None = None):
        validation_checks.append({
            'check_id': check_id,
            'status': 'passed' if passed else 'failed',
            'details': details or {}
        })

    # 1. Structure and Hashes
    has_hashes = (
        'hashes' in packet and
        bool(packet['hashes'].get('source_content_hash')) and
        bool(packet['hashes'].get('normalized_text_hash'))
    )
    check('has_consistent_provenance_hashes', has_hashes)

    # 2. Anchors and Content Units
    has_units = 'content_units' in packet and isinstance(packet['content_units'], list) and len(packet['content_units']) > 0
    check('has_content_units', has_units)

    if has_units:
        all_have_anchors = all('anchor_id' in u and u['anchor_id'] for u in packet['content_units'])
        check('all_content_units_have_anchors', all_have_anchors)
    else:
        check('all_content_units_have_anchors', False)

    has_anchors_list = 'anchors' in packet and isinstance(packet['anchors'], list) and len(packet['anchors']) > 0
    check('has_anchors_list', has_anchors_list)

    # 3. Provenance
    has_provenance = (
        'provenance' in packet and
        bool(packet['provenance'].get('acquisition_mode')) and
        bool(packet['provenance'].get('original_reference_or_file_id'))
    )
    check('has_valid_provenance', has_provenance)

    # 4. Quarantine partitions
    has_quarantine = 'quarantine_partitions' in packet and isinstance(packet['quarantine_partitions'], list)
    check('has_quarantine_partitions_field', has_quarantine)

    # 5. Extraction safeguards
    has_no_downstream_outputs = (
        'downstream_extraction_outputs' in packet and
        packet['downstream_extraction_outputs'] == []
    )
    check('extraction_not_performed_yet', has_no_downstream_outputs)

    no_claims_emitted = (
        'output_boundary' in packet and
        packet['output_boundary'].get('claims_emitted_now') is False and
        packet['output_boundary'].get('events_emitted_now') is False
    )
    check('output_boundary_safeguards_active', no_claims_emitted)

    failed_checks = [c for c in validation_checks if c['status'] == 'failed']
    is_valid = len(failed_checks) == 0

    return {
        'packet_id': packet.get('packet_id', 'unknown'),
        'is_valid': is_valid,
        'validation_checks': validation_checks,
        'failed_checks': failed_checks
    }

def build_validation_gate(upstream_payload: dict[str, Any]) -> dict[str, Any]:
    now = datetime.now(UTC).isoformat()
    packets, packet_collection_key = _packet_rows(upstream_payload)

    packet_validations = [validate_packet(p) for p in packets]

    all_valid = bool(packets) and all(pv['is_valid'] for pv in packet_validations)
    failed_count = sum(1 for pv in packet_validations if not pv['is_valid'])

    result = {
        'run_id': _run_id('normalized_packet_validation_gate'),
        'block_id': BLOCK_ID,
        'schema_version': SCHEMA_VERSION,
        'created_at_utc': now,
        'upstream_blocks': [UPSTREAM_BLOCK],
        'gate_status': 'passed' if all_valid else 'failed',
        'packet_collection_key': packet_collection_key,
        'summary': {
            'total_packets_evaluated': len(packets),
            'valid_packets': len(packets) - failed_count,
            'invalid_packets': failed_count,
            'all_packets_valid': all_valid,
        },
        'packet_validations': packet_validations,
        'next_recommended_block': '247_review_only_real_source_claim_event_entity_extraction_contract_v1' if all_valid else None,
        'next_chat_recommended': all_valid,
        'content_hash': ''
    }

    result['content_hash'] = _stable_hash({k: v for k, v in result.items() if k != 'content_hash'})
    return result


def _run_id(prefix: str) -> str:
    return f"{prefix}_{datetime.now(UTC).isoformat().replace(':', '').replace('+', 'Z')}"


def render_validation_gate_markdown(payload: dict[str, Any]) -> str:
    summary = payload.get('summary', {})
    lines = [
        '# DEAN-OS Real Source Normalized Packet Validation Gate',
        '',
        f"- Block ID: `{payload.get('block_id')}`",
        f"- Gate status: `{payload.get('gate_status')}`",
        f"- Packet collection: `{payload.get('packet_collection_key')}`",
        '',
        '## Summary',
        '',
    ]
    for key, value in summary.items():
        lines.append(f'- `{key}`: `{value}`')
    lines.extend(['', '## Failed Packets', ''])
    failed = [row for row in payload.get('packet_validations', []) if not row.get('is_valid')]
    if not failed:
        lines.append('- none')
    else:
        for row in failed[:20]:
            failed_checks = ', '.join(check.get('check_id', 'unknown') for check in row.get('failed_checks', []))
            lines.append(f"- `{row.get('packet_id')}`: {failed_checks or 'unknown failure'}")
    lines.extend(
        [
            '',
            '## Boundary',
            '',
            '- This gate validates normalized packet shape only.',
            '- It does not execute claim/event/entity extraction.',
            '- It does not promote evidence, write learning memory, generate recommendations, or trade.',
            '',
            '## Next',
            '',
            f"`{payload.get('next_recommended_block')}`",
        ]
    )
    return '\n'.join(lines).strip() + '\n'


def _packet_rows(upstream_payload: dict[str, Any]) -> tuple[list[dict[str, Any]], str]:
    if isinstance(upstream_payload.get('normalized_packet_rows'), list):
        return upstream_payload['normalized_packet_rows'], 'normalized_packet_rows'
    if isinstance(upstream_payload.get('normalized_packet_fixture_rows'), list):
        return upstream_payload['normalized_packet_fixture_rows'], 'normalized_packet_fixture_rows'
    return [], 'missing_normalized_packet_rows'
