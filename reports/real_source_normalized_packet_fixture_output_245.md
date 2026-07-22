# Block 245 — Real Source Normalized Packet Fixture

Block ID: `245_review_only_real_source_normalized_packet_fixture_v1`

Fixture status: `review_only_normalized_packet_fixture_materialized_for_validation_not_evidence`

## Summary
- `normalized_packet_fixture_count`: `12`
- `source_type_count`: `12`
- `content_unit_count`: `29`
- `anchor_count`: `29`
- `quarantine_partition_instance_count`: `6`
- `candidate_asset_or_entity_count`: `16`
- `candidate_topic_count`: `17`
- `candidate_sector_count`: `6`
- `real_source_content_supplied_in_245`: `False`
- `fixtures_are_production_evidence`: `False`
- `claim_extraction_performed_in_245`: `False`
- `event_extraction_performed_in_245`: `False`
- `event_propagation_performed_in_245`: `False`
- `company_thesis_generated_in_245`: `False`

## Real-data boundary

- These are offline normalized packet fixtures, not real external evidence.
- The real-source path is defined and supported, but no real source content is supplied in block 245.
- A future block can normalize a real uploaded/pasted/connector/API snapshot packet using the same schema.

## Candidate routing indexes

- Assets/entities: `AAPL, AMD, AMZN, BAC, GOOGL, GS, INTC, IWM, JPM, MSFT, NVDA, QQQ, SPY, TSM, WMT, XOM`
- Topics: `ai_capex_cycle, analyst_note, company_filing, consumer_demand, earnings_transcript, energy_price_shock, financial_disclosure, geopolitics_security, historical_supply_shock_analogy, industry_report, interest_rates, interest_rates_discount_rates, platform_regulation_antitrust_privacy, rare_earths_critical_minerals, semiconductor_export_controls, semiconductor_supply_chain, ticker_market_snapshot`
- Sectors: `ai_big_tech, consumer_staples, energy, finance, market_etfs, semiconductors`

## Output boundary

- No live fetch.
- No external API calls.
- No claim/event/entity extraction.
- No event propagation.
- No company thesis.
- No ratio interpretation, valuation, recommendation, price target, trade signal, or trading output.

## Next

`246_review_only_real_source_normalized_packet_validation_gate_v1`
