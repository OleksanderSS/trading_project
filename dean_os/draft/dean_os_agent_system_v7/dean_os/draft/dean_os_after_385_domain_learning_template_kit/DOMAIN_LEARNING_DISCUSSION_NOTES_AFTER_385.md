# Domain Learning Discussion Notes After 385

This file preserves the conceptual guidance discussed after `dean_os_after_385_full_context_bundle.zip`.
It is intentionally non-numbered and separate from the core assistant_workbench lifecycle templates.

The purpose is to help Codex understand how a domain analyst should be configured, supplied with
information, and evaluated. These are design notes and reusable ideas, not production code and not a
blind merge target.

## 1. Main principle

Do not try to “teach” a domain analyst by forcing all knowledge into model weights.

A correct domain-learning setup should look like:

```text
sources
→ source registry
→ ingestion filters
→ metadata normalization
→ document store
→ structured fact store
→ evidence graph
→ hybrid retrieval
→ causal pattern library
→ analyst interpretation templates
→ evaluation pack
→ human feedback loop
```

Fine-tuning is optional and should come later. It should not be used as the primary way to store
current facts, market data, filings, news, or company-specific numbers.

## 2. What “learning” means for a domain analyst

For DEAN-OS, domain learning means that the analyst has controlled access to:

- source registries;
- sector profiles;
- company/entity mappings;
- value-chain maps;
- causal pattern templates;
- RAG retrieval settings;
- evidence scoring;
- event interpretation schemas;
- materiality scoring;
- evaluation tests;
- human feedback.

The analyst should learn how to reason with sources, not merely memorize text.

## 3. Heavy industry analyst example

For heavy industry, relevant domains include:

- steel;
- green steel;
- mining;
- cement;
- chemicals;
- industrial energy;
- machinery;
- capital goods;
- infrastructure materials.

Important reasoning lenses:

- value chain;
- cost curve;
- commodity cycle;
- energy sensitivity;
- carbon/emissions regulation;
- trade policy;
- subsidies and industrial policy;
- capex cycle;
- balance-sheet risk;
- capacity/utilization;
- production vs shipments vs demand;
- regional differences;
- company fundamentals.

## 4. Source strategy

The analyst should not treat every source equally.

Recommended source tiers:

### Tier 1: core evidence

- regulatory filings;
- audited annual reports;
- official statistical agencies;
- government/central bank/economic datasets;
- regulator documents;
- official trade/energy/mineral statistics.

### Tier 2: strong context

- investor presentations;
- earnings call transcripts;
- sustainability reports;
- industry association reports;
- company press releases;
- official policy announcements.

### Tier 3: event context

- reputable financial news;
- trade press;
- local industrial news;
- specialist sector publications.

### Tier 4: weak/context only

- blogs;
- social media;
- unsourced reposts;
- unattributed commentary.

Tier 3 and Tier 4 sources may trigger investigation or watchlist items, but should not produce strong
conclusions without confirmation.

## 5. News collectors should not be sentiment-only

News collector output should not stop at:

```json
{"sentiment": "positive"}
```

That is too weak. News should be transformed into event/context packets and routed to analysts.

Correct flow:

```text
news collector
→ deduplication
→ source quality classification
→ event extraction
→ entity/sector/geography mapping
→ causal pattern matching
→ domain analyst routing
→ structured hypothesis
→ evidence gaps
→ materiality/watchlist label
→ review queue item
→ daily digest
```

Sentiment may be stored, but only as a weak auxiliary feature.

## 6. Causal/context reasoning

The analyst must handle indirect mechanisms, not only direct action-result chains.

Example:

```text
renewable energy manufacturing grows
→ possible increase in steel-intensive infrastructure demand
→ possible demand shift toward green steel if procurement rules, subsidies, or carbon rules favor low-carbon materials
→ check actual green-steel capacity, price premium, order books, subsidy details, procurement rules, and policy enforcement
→ produce a hypothesis/watchlist item, not a recommendation
```

The analyst must distinguish:

```text
event
→ mechanism
→ intermediate variables
→ counterforces
→ evidence gaps
→ confidence/materiality
```

The system should not jump from “green energy grows” to “green steel demand will definitely grow”.

## 7. Causal pattern memory

RAG finds documents. Pattern memory stores reusable mechanisms.

Useful pattern examples:

- renewable energy growth → possible green steel demand;
- rate hikes → industrial capex pressure;
- carbon policy → cost-curve shift;
- tariffs → domestic utilization shift;
- sanctions → supply-chain rerouting;
- China property weakness → steel/cement demand pressure;
- gas prices → chemicals margin pressure;
- defense spending → specialty metals demand;
- grid expansion → copper/steel/electrical equipment demand.

Each causal pattern should define:

- trigger events;
- mechanism chain;
- affected value chain;
- intermediate variables;
- confirming evidence;
- contradicting evidence;
- counterforces;
- common false positives;
- time horizon;
- materiality rules;
- required review.

## 8. Direct vs indirect events

Direct events:

- plant shutdown;
- tariff enacted;
- production up/down;
- company cuts guidance;
- interest rate hike;
- strike begins;
- sanctions announced.

Indirect/context events:

- green energy manufacturing expansion;
- infrastructure bill;
- defense spending growth;
- industrial policy subsidy;
- carbon regulation;
- China property weakness;
- data center power demand growth;
- shipping route disruption.

Indirect events require more careful hypothesis handling.

## 9. Materiality scoring

For indirect news/context, materiality should not be based only on sentiment or mention frequency.

Suggested dimensions:

- sector relevance;
- mechanism strength;
- policy/subsidy support;
- company/asset exposure;
- evidence quality;
- time-horizon clarity;
- counterforces/contradictions.

Materiality score should create review priority, not a trading signal.

Allowed labels:

- archive;
- watchlist_low;
- watchlist_medium;
- watchlist_high;
- review_required.

Forbidden outputs:

- buy/sell/hold;
- price target;
- trade signal;
- broker order;
- autonomous portfolio action.

## 10. Evidence gaps

For every hypothesis, the analyst should state what is missing.

Example evidence gaps for green steel:

- no confirmed procurement rule;
- no subsidy level;
- no company-specific order backlog;
- no green-steel capacity disclosure;
- no verified price premium;
- no regional policy enforcement evidence;
- no steel intensity assumption for renewable projects.

Evidence gaps should become next collection tasks.

## 11. Daily data accumulation loop

The user’s current goal is not paper trading or live trading. The goal is daily accumulation and analysis
of information.

Safe daily loop:

```text
daily scheduled run
→ collect allowlisted news/reports/data
→ deduplicate
→ normalize metadata
→ extract events/claims/numbers
→ route to domain analysts
→ match causal patterns
→ create hypotheses
→ identify evidence gaps
→ update document store / structured store / evidence graph / vector index
→ create review queue items
→ produce internal digest
```

This is allowed only as data accumulation and review-only analysis.

## 12. Separation of responsibilities

### Pipeline

The pipeline should handle:

- fetching from allowlisted sources;
- parsing;
- deduplication;
- source hashing;
- metadata normalization;
- entity/date/unit normalization;
- storing documents;
- updating indexes;
- creating event packets.

### Analyst agent

The analyst should handle:

- domain interpretation;
- causal mechanism reasoning;
- pattern matching;
- counterforce detection;
- evidence-gap generation;
- materiality/watchlist scoring;
- review packet creation.

### Orchestrator

The orchestrator should handle:

- routing;
- dependency ordering;
- blocked/incident states;
- review queue management;
- daily digest coordination;
- safe handoff boundaries.

## 13. RAG settings

Recommended default for domain reports/news:

```yaml
chunk_size_tokens: 800-1200
chunk_overlap_tokens: 120-180
retrieval_mode: hybrid_bm25_dense
initial_top_k: 50
rerank_top_k: 12
final_context_chunks: 6-10
require_citations: true
require_period_metadata: true
require_unit_metadata_for_numeric_claims: true
```

Use metadata filters for:

- date;
- entity;
- sector;
- geography;
- source tier;
- source type;
- period covered;
- document type.

## 14. Structured stores

Vector search alone is not enough.

Recommended stores:

1. Document store  
   Raw and chunked filings, reports, news, transcripts, policy documents.

2. Structured fact store  
   Metrics, capacity, production, shipments, energy costs, commodity prices, macro series.

3. Evidence graph  
   Entity → event → claim → source → confidence → contradiction/corroboration links.

## 15. Unit and concept traps

Domain analysts must not confuse:

- capacity vs production;
- production vs shipments;
- demand vs apparent consumption;
- metric tons vs short tons;
- quarterly vs annualized figures;
- nominal vs real values;
- company-level vs segment-level data;
- policy proposal vs enacted regulation;
- subsidy announcement vs actual disbursement;
- capex plan vs completed capacity.

These should become evaluation tests.

## 16. Evaluation pack

The domain analyst should be evaluated on:

- source-grounded QA;
- numeric extraction with units/periods;
- contradiction handling;
- time leakage prevention;
- source-tier preference;
- indirect mechanism interpretation;
- evidence-gap generation;
- false positive avoidance.

Example fail conditions:

- using future data for a past-as-of-date question;
- treating capacity as production;
- treating sentiment as investment signal;
- missing source citation for a numeric claim;
- forming a strong conclusion from weak news alone;
- ignoring counterforces.

## 17. Fine-tuning guidance

Do not fine-tune to memorize current facts.

Fine-tuning can be useful for:

- document classification;
- stable JSON extraction;
- event type classification;
- materiality routing;
- domain-specific memo style;
- causal pattern detection;
- evidence-gap formatting.

Fine-tuning should not be used for:

- current market facts;
- company financial numbers;
- fresh news;
- forecasts as truth;
- buy/sell decisions;
- trading signals.

## 18. Automation boundary

Allowed automation:

- scheduled daily collection;
- parsing;
- deduplication;
- metadata normalization;
- event extraction;
- causal pattern matching;
- watchlist scoring;
- evidence-gap generation;
- evidence store update;
- vector index update;
- review queue creation;
- internal digest.

Forbidden automation:

- broker API calls;
- order creation;
- autonomous trading;
- portfolio rebalancing;
- buy/sell/hold generation;
- price target generation;
- final investment recommendation;
- production config mutation without review;
- model promotion without review.

## 19. Correct Codex interpretation

Codex should use this kit as a design inventory for domain-learning implementation.

Codex should harvest/adapt:

- source registry templates;
- RAG settings;
- causal pattern schemas;
- news collector routing logic;
- event interpretation schemas;
- evidence-gap schemas;
- materiality scoring;
- daily run templates;
- safe automation boundaries;
- evaluation ideas.

Codex should not blind-merge the kit as production code.

## 20. Summary

The correct mental model:

```text
news/data are not signals by themselves
→ they become event packets
→ event packets are routed to analysts
→ analysts create hypotheses and evidence gaps
→ the system accumulates evidence over time
→ humans review high-materiality interpretations
→ no autonomous trading or recommendations
```

This is the intended domain-learning layer after 385.
