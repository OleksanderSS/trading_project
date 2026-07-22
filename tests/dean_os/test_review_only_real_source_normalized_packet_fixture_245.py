
from pathlib import Path
from dean_os.review_only_real_source_normalized_packet_fixture import build_fixture

HERE = Path(__file__).resolve().parents[1]
UPSTREAM = HERE / 'fixtures' / 'real_source_packet_intake_normalizer_contract_output_244.json'


def _result():
    return build_fixture(UPSTREAM)


def test_creates_one_normalized_packet_fixture_per_source_type():
    r = _result()
    assert r['summary']['normalized_packet_fixture_count'] == 12
    assert r['summary']['source_type_count'] == 12
    assert len(r['normalized_packet_fixture_rows']) == 12


def test_packets_have_required_normalized_structures():
    r = _result()
    for packet in r['normalized_packet_fixture_rows']:
        for key in ['provenance', 'hashes', 'parser_profile', 'content_units', 'anchors', 'quality_precheck', 'routing_prefilter', 'normalization_gate_status']:
            assert key in packet
        assert len(packet['content_units']) > 0
        assert len(packet['anchors']) == len(packet['content_units'])


def test_fixtures_are_not_presented_as_real_evidence():
    r = _result()
    assert r['summary']['real_source_content_supplied_in_245'] is False
    assert r['summary']['fixtures_are_production_evidence'] is False
    for packet in r['normalized_packet_fixture_rows']:
        assert packet['source_fixture_status'] == 'offline_normalized_packet_fixture_not_real_external_source'
        assert packet['quality_precheck']['primary_secondary_classification'] == 'fixture_not_evidence'
        assert packet['real_source_content_supplied_in_245'] is False


def test_claim_event_entity_outputs_are_empty():
    r = _result()
    for packet in r['normalized_packet_fixture_rows']:
        assert packet['downstream_extraction_outputs'] == []
        assert packet['output_boundary']['claims_emitted_now'] is False
        assert packet['output_boundary']['events_emitted_now'] is False
        assert packet['output_boundary']['entities_resolved_now'] is False
        assert packet['output_boundary']['event_propagation_executed_now'] is False


def test_candidate_links_are_candidate_only_not_final_asset_theses():
    r = _result()
    assert {'AMD', 'NVDA', 'AAPL', 'SPY'} <= set(r['candidate_asset_or_entity_index'])
    for packet in r['normalized_packet_fixture_rows']:
        assert packet['routing_prefilter']['candidate_links_are_final'] is False
        assert packet['output_boundary']['company_thesis_generated_now'] is False


def test_analyst_note_ratings_and_market_snapshot_are_quarantined():
    r = _result()
    by_type = {p['source_type_id']: p for p in r['normalized_packet_fixture_rows']}
    analyst_partitions = {q['partition_id'] for q in by_type['analyst_research_and_broker_notes']['quarantine_partitions']}
    assert 'third_party_rating_or_price_target' in analyst_partitions
    market_partitions = {q['partition_id'] for q in by_type['ticker_market_price_volume_series']['quarantine_partitions']}
    assert 'market_snapshot_without_observation_window' in market_partitions


def test_safety_flags_block_recommendations_and_trading():
    r = _result()
    flags = r['safety_flags']
    for key in ['live_fetch_allowed','external_api_call_allowed','claim_extraction_execution_allowed_now','event_propagation_execution_allowed_now','company_specific_thesis_allowed_now','recommendation_allowed','rating_allowed','buy_sell_hold_allowed','price_target_allowed','trade_signal_allowed','position_sizing_allowed','broker_routing_allowed','trading_allowed']:
        assert flags[key] is False


def test_validation_and_safety_clean():
    r = _result()
    assert r['validation_error_count'] == 0
    assert r['failed_validation_check_count'] == 0
    assert r['failed_safety_assertion_count'] == 0
    assert all(row['status'] == 'passed' for row in r['validation_checks'])
    assert all(row['status'] == 'passed' for row in r['safety_assertions'])


def test_source_first_pipeline_keeps_validation_before_extraction():
    r = _result()
    pipeline = r['source_first_pipeline_position']
    assert pipeline.index('normalized_packet_fixture_materialization') < pipeline.index('future_normalized_packet_validation_gate')
    assert pipeline.index('future_normalized_packet_validation_gate') < pipeline.index('future_claim_event_entity_extraction_contract')


def test_next_block_is_validation_gate_and_next_chat_recommended():
    r = _result()
    assert r['next_recommended_block'] == '246_review_only_real_source_normalized_packet_validation_gate_v1'
    assert r['next_chat_recommended'] is True
