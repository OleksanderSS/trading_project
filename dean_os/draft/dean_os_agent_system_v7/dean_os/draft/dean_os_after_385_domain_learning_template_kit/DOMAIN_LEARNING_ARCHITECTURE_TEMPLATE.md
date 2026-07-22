# Domain Learning Architecture Template

## Scope

This template defines how a DEAN-OS domain analyst should learn and operate in a specialized sector such as heavy industry.

The target is not model memorization. The target is a controlled domain-intelligence system:

```text
source registry
→ ingestion filters
→ normalization
→ evidence store
→ hybrid retrieval
→ evidence scoring
→ analyst reasoning templates
→ evaluation pack
→ human feedback loop
→ optional fine-tuning
```

## Main components

### 1. Source registry

A structured catalog of allowed source types, trust tiers, metadata requirements, freshness expectations, and allowed uses.

Mandatory metadata:

- `source_id`
- `source_type`
- `publisher`
- `domain`
- `geography`
- `period_covered`
- `publication_date`
- `ingestion_date`
- `trust_tier`
- `license_status`
- `allowed_for`
- `not_allowed_for`

### 2. Ingestion filters

Filters must prevent low-quality or misdated documents from entering the evidence layer without quarantine.

Core filters:

- source type allowlist / blocklist
- date and period validation
- entity resolution
- duplicate detection
- language handling
- table extraction confidence
- unit/currency normalization
- weak-source quarantine

### 3. Evidence store

Use a structured evidence layer, not only a vector database.

Minimum evidence record:

- entity
- claim / metric / event
- period
- unit
- source anchor
- source trust tier
- confidence
- contradiction status
- reviewer status

### 4. RAG and retrieval

Use hybrid retrieval: dense vectors + BM25 + metadata filters + reranking.

Do not allow uncited numeric claims. Do not allow weak sources to drive final conclusions without corroboration.

### 5. Evaluation pack

Before deployment, create tests for:

- source-grounded QA
- metric extraction
- unit traps
- period traps
- capacity vs production vs shipment distinctions
- contradiction handling
- time leakage
- source tier precedence

### 6. Optional fine-tuning

Fine-tuning is allowed only after the RAG/evidence loop has enough labeled examples.

Good fine-tuning targets:

- document classification
- extraction into stable schemas
- materiality classification
- routing between reasoning lenses
- style consistency for internal memos

Bad fine-tuning targets:

- memorizing current facts
- replacing source retrieval
- producing investment recommendations
- producing trading signals
- bypassing review gates

## Safety and review constraints

Forbidden outputs:

- buy/sell/hold recommendation
- price target
- autonomous trading action
- broker/order routing
- uncited numeric claim
- production config mutation without review
- live fetch without explicit system permission


## News collector → domain analyst interpretation layer

Domain learning should include a news/context interpretation layer. News collectors should not
only produce sentiment scores. They should produce normalized event packets that can be routed
to the appropriate domain analyst or sector profile.

The analyst should convert the event into:

- event type;
- affected value chain;
- direct/indirect mechanism;
- candidate causal pattern;
- intermediate variables;
- counterforces;
- confirming evidence;
- contradicting evidence;
- evidence gaps;
- materiality/watchlist label;
- review queue item.

Example:

```text
renewable energy manufacturing grows
→ possible steel-intensive infrastructure demand
→ possible green-steel preference if subsidies/procurement rules apply
→ check green-steel capacity, price premium, procurement rules, order books, and policy enforcement
→ produce hypothesis/watchlist item, not recommendation
```

This layer is implemented as templates in:

- `NEWS_COLLECTOR_TO_DOMAIN_ANALYST_ROUTING_TEMPLATE.md`
- `NEWS_EVENT_INTERPRETATION_SCHEMA_TEMPLATE.json`
- `CAUSAL_PATTERN_SCHEMA_TEMPLATE.yaml`
- `CAUSAL_PATTERNS_HEAVY_INDUSTRY.yaml`
- `MATERIALITY_SCORING_TEMPLATE.yaml`
- `EVIDENCE_GAP_SCHEMA_TEMPLATE.json`
- `DAILY_DOMAIN_LEARNING_RUN_TEMPLATE.yaml`
- `ANALYST_NEWS_INTERPRETATION_PROMPT_TEMPLATE.md`
- `SAFE_AUTOMATION_BOUNDARY_TEMPLATE.yaml`
