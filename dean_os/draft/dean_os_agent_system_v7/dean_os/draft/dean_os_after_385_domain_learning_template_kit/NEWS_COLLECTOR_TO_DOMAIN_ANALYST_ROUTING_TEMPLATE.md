# News Collector to Domain Analyst Routing Template

Purpose: define how raw news collectors feed domain analysts. This is not a sentiment-only pipeline.
News collectors should produce normalized event/context packets that can be routed into analyst profiles
for causal pattern reasoning, evidence-gap detection, materiality scoring, and human-review queue creation.

## Core principle

Do not treat news sentiment as a sufficient analytical signal.

Allowed news-derived outputs:
- normalized event packet;
- candidate causal pattern;
- structured hypothesis;
- materiality/watchlist score;
- evidence-gap list;
- next collection tasks;
- human-review item;
- internal digest item.

Forbidden news-derived outputs:
- buy/sell/hold;
- price target;
- trade signal;
- autonomous portfolio action;
- broker/order action;
- production trading configuration mutation;
- model promotion without human review.

## Routing flow

```text
raw news collector
→ source quality classification
→ deduplication
→ event extraction
→ entity/sector/geography mapping
→ causal pattern matching
→ domain analyst routing
→ hypothesis/evidence-gap generation
→ materiality scoring
→ evidence store update
→ review queue / daily digest
```

## Collector packet minimum fields

```yaml
collector_packet:
  source_id: string
  source_type: news | company_release | regulator | trade_press | government | industry_body
  trust_tier: tier_1 | tier_2 | tier_3 | tier_4
  title: string
  body_or_summary: string
  url_or_reference: string
  published_at: datetime
  event_date: datetime | null
  language: string
  geography: list[string]
  raw_entities: list[string]
  raw_topics: list[string]
  ingestion_timestamp: datetime
  source_hash: string
```

## Analyst routing fields

```yaml
analyst_routing:
  candidate_domains:
    - heavy_industry
    - energy
    - macro_policy
    - geopolitics
    - semiconductors
  candidate_sector_profiles:
    - steel
    - green_steel
    - mining
    - chemicals
    - cement
    - industrial_energy
  routing_reason: string
  routing_confidence: low | medium | high
  requires_human_review: true
```

## Sentiment is only one weak feature

Sentiment may be stored as a secondary feature, but it must not override:
- source quality;
- causal mechanism strength;
- evidence directness;
- company/sector exposure;
- contradictory evidence;
- time horizon;
- materiality score.
