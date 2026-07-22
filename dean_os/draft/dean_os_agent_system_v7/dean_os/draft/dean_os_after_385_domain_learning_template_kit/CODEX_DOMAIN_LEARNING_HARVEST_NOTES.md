# Codex Domain Learning Harvest Notes After 385

This kit extends the after-385 staged design inventory with domain-learning templates. It should not restart the numbered assistant_workbench ladder.

## What to harvest first

High-value artifacts:

1. `SECTOR_PROFILE_TEMPLATE_HEAVY_INDUSTRY.yaml`
   - Useful for domain/profile configuration.
   - Adapt into the repo's profile/config layer if such a layer exists.

2. `SOURCE_REGISTRY_TEMPLATE_HEAVY_INDUSTRY.yaml`
   - Useful for source tiering and metadata contracts.
   - Do not hard-code sources without checking current project conventions.

3. `INGESTION_FILTERS_TEMPLATE.yaml`
   - Useful for fail-closed ingestion policy.
   - Important for avoiding misdated, unitless, weak-source, and low-table-confidence evidence.

4. `RAG_RETRIEVAL_SETTINGS_TEMPLATE.yaml`
   - Useful for retrieval config defaults.
   - Prefer hybrid retrieval and metadata filters over vector-only retrieval.

5. `EVIDENCE_SCORING_TEMPLATE.yaml`
   - Useful for source quality scoring and evidence gating.
   - Preserve final-claim thresholds and contradiction handling.

6. `EVAL_PACK_TEMPLATE_HEAVY_INDUSTRY.json`
   - Useful for test fixtures/evaluation cases.
   - Start with unit traps, source-tier precedence, and time leakage tests.

7. `FINE_TUNING_DATASET_SCHEMA_TEMPLATE.json`
   - Useful only after enough reviewer-approved examples exist.
   - Do not fine-tune to memorize current facts.

8. `HUMAN_FEEDBACK_SCHEMA_TEMPLATE.json`
   - Useful for feedback-driven improvement loop.
   - High priority if the project needs analyst review workflows.

## What not to do

- Do not copy these templates directly into production without adapting paths, config loaders, and schema validation.
- Do not create autonomous trading behaviors.
- Do not allow uncited numeric claims.
- Do not bypass human review for material outputs.
- Do not treat company presentations as audited evidence.
- Do not use live fetch or external APIs in tests unless explicitly configured.

## Integration approach

Recommended order:

1. Define profile/config schema.
2. Add source registry structure.
3. Add ingestion metadata validation.
4. Add evidence scoring and claim thresholds.
5. Add RAG metadata filters and retrieval settings.
6. Add eval fixtures.
7. Add feedback schema.
8. Consider fine-tuning only after reviewed examples accumulate.

## Relationship to after-385

The after-385 full bundle is the full design inventory. This domain-learning kit is a practical add-on for making the Analyst Branch domain-configurable.

Use both together:

- after-385: full multi-agent lifecycle map.
- integration kit: how to harvest/adapt.
- domain-learning kit: how to configure and evaluate a sector analyst.


## v2 harvest targets: news/context interpretation

Codex should consider harvesting/adapting the added news/context templates to support daily
domain-learning runs:

- route news collectors into analyst profiles, not only sentiment scoring;
- extract events, mechanisms, and affected value chains;
- match causal patterns and identify counterforces;
- produce structured hypotheses and evidence gaps;
- create watchlist/review queue items;
- keep all outputs review-only and non-trading.

High-priority files:

- `NEWS_COLLECTOR_TO_DOMAIN_ANALYST_ROUTING_TEMPLATE.md`
- `NEWS_EVENT_INTERPRETATION_SCHEMA_TEMPLATE.json`
- `CAUSAL_PATTERN_SCHEMA_TEMPLATE.yaml`
- `CAUSAL_PATTERNS_HEAVY_INDUSTRY.yaml`
- `DAILY_DOMAIN_LEARNING_RUN_TEMPLATE.yaml`
- `SAFE_AUTOMATION_BOUNDARY_TEMPLATE.yaml`
