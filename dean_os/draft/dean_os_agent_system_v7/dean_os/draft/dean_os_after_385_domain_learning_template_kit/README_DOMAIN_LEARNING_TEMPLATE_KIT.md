# DEAN-OS After-385 Domain Learning Template Kit

Purpose: provide practical, non-numbered domain-learning templates for Codex to harvest/adapt into the DEAN-OS project after the numbered assistant_workbench ladder has stopped at block 385.

This kit is not production code and not a patch. It is a structured template library for configuring domain analysts, starting with `heavy_industry` as the reference domain.

Use together with:

- `dean_os_after_385_full_context_bundle.zip` — full staged design inventory.
- `dean_os_after_385_codex_integration_kit.zip` — Codex harvest/adapt guidance.

Core principle:

> Do not try to “teach the LLM everything.” Build a governed domain-learning loop: source registry, ingestion filters, normalization, evidence store, hybrid retrieval, evidence scoring, evaluation, and human feedback. Fine-tuning is optional and should come after evaluation proves it is useful.

Recommended Codex usage:

1. Inspect these templates as design inputs.
2. Compare against the current repo.
3. Harvest/adapt useful schemas, configs, tests, and documentation.
4. Do not copy this kit directly into production code as-is.
5. Preserve no-autonomous-trading and human-review boundaries.


## v2 addition: news collectors, causal patterns, and daily domain-learning automation

This kit now includes templates for routing news collectors into domain analysts.
The intent is to avoid sentiment-only processing and support:

- event extraction;
- source quality filtering;
- entity/sector/geography mapping;
- causal pattern matching;
- indirect mechanism reasoning;
- counterforce detection;
- evidence-gap generation;
- materiality/watchlist scoring;
- safe daily data accumulation and review queues.

These templates remain non-production, review-only, and intended for Codex harvest/adapt.
They are separate from the core numbered assistant_workbench agent-system templates.


## Consolidated discussion notes

The file `DOMAIN_LEARNING_DISCUSSION_NOTES_AFTER_385.md` preserves the long-form guidance from
the post-385 discussion: how to provide information to agents, why news should not be reduced to
sentiment, how indirect causal mechanisms should be handled, how daily data accumulation should work,
and what must remain review-only.
