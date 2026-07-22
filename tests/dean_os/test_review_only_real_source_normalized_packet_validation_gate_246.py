import json
from pathlib import Path
from typing import Any

from dean_os.review_only_real_source_normalized_packet_validation_gate import build_validation_gate
from dean_os.review_only_real_source_normalized_packet_fixture import build_fixture

HERE = Path(__file__).resolve().parent
FIXTURES_DIR = HERE.parent / 'fixtures'
UPSTREAM_244 = FIXTURES_DIR / 'real_source_packet_intake_normalizer_contract_output_244.json'

def _upstream_245() -> dict[str, Any]:
    # Dynamic generation to ensure we always test against valid 245 output
    return build_fixture(UPSTREAM_244)

def _result() -> dict[str, Any]:
    return build_validation_gate(_upstream_245())

def test_validation_gate_passes_for_valid_fixtures():
    r = _result()
    assert r['gate_status'] == 'passed'
    assert r['summary']['invalid_packets'] == 0
    assert r['summary']['all_packets_valid'] is True

def test_validation_gate_checks_provenance():
    r = _result()
    # Check that all packets passed the provenance hash check
    for p_val in r['packet_validations']:
        checks = {c['check_id']: c['status'] for c in p_val['validation_checks']}
        assert checks['has_consistent_provenance_hashes'] == 'passed'
        assert checks['has_valid_provenance'] == 'passed'

def test_validation_gate_checks_anchors_and_units():
    r = _result()
    for p_val in r['packet_validations']:
        checks = {c['check_id']: c['status'] for c in p_val['validation_checks']}
        assert checks['has_content_units'] == 'passed'
        assert checks['all_content_units_have_anchors'] == 'passed'
        assert checks['has_anchors_list'] == 'passed'

def test_validation_gate_checks_quarantine_and_extraction_safeguards():
    r = _result()
    for p_val in r['packet_validations']:
        checks = {c['check_id']: c['status'] for c in p_val['validation_checks']}
        assert checks['has_quarantine_partitions_field'] == 'passed'
        assert checks['extraction_not_performed_yet'] == 'passed'
        assert checks['output_boundary_safeguards_active'] == 'passed'

def test_validation_gate_detects_invalid_packet():
    upstream = _upstream_245()
    # Mutate one packet to make it invalid
    upstream['normalized_packet_fixture_rows'][0]['hashes'] = {}
    r = build_validation_gate(upstream)
    assert r['gate_status'] == 'failed'
    assert r['summary']['invalid_packets'] == 1
    assert r['summary']['all_packets_valid'] is False
    
    # Verify the specific check that failed
    failed_packet_val = r['packet_validations'][0]
    assert not failed_packet_val['is_valid']
    checks = {c['check_id']: c['status'] for c in failed_packet_val['validation_checks']}
    assert checks['has_consistent_provenance_hashes'] == 'failed'

def test_validation_gate_blocks_missing_packet_rows():
    r = build_validation_gate({})
    assert r['gate_status'] == 'failed'
    assert r['packet_collection_key'] == 'missing_normalized_packet_rows'
    assert r['summary']['total_packets_evaluated'] == 0
    assert r['summary']['all_packets_valid'] is False
    assert r['next_chat_recommended'] is False
    assert r['next_recommended_block'] is None

def test_next_block_is_recommended():
    r = _result()
    assert r['next_chat_recommended'] is True
    assert '247_' in r['next_recommended_block']
