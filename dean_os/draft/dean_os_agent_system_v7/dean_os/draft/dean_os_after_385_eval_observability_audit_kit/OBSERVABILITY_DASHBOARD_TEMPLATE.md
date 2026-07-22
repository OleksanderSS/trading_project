# Observability Dashboard Template

Purpose: outline dashboard panels for DEAN-OS data accumulation, retrieval, analyst reasoning, and review quality.

## Dashboard sections

### 1. Daily run health

- last successful run;
- run duration;
- collector success/failure counts;
- documents collected;
- duplicates removed;
- normalization errors;
- indexing status.

### 2. Source quality

- source tier distribution;
- stale source rate;
- weak-source overuse rate;
- failed source fetches;
- source hash changes;
- license/review-required sources.

### 3. Retrieval quality

- recall@50 on gold eval set;
- rerank precision@12;
- citation anchor accuracy;
- no-evidence answer rate;
- stale-source retrieval rate.

### 4. Analyst output quality

- grounded claim rate;
- numeric metadata completeness;
- hypothesis labeling accuracy;
- counterforce coverage;
- evidence-gap generation rate;
- human review disagreement rate.

### 5. Safety counters

All should remain zero:

- buy/sell/hold generated;
- price target generated;
- trade signal generated;
- broker/order call attempted;
- production config mutation attempted;
- model promotion attempted without review.

### 6. Causal pattern quality

- patterns matched by domain;
- false-positive rate by pattern;
- high-materiality watchlist count;
- evidence gaps created;
- review outcomes by pattern.

### 7. Feedback loop

- human review labels by category;
- repeated error clusters;
- regression test failures;
- prompts/templates requiring update;
- source registry issues.
