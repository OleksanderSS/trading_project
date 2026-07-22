# DEAN-OS Addendum — Data Pipeline vs Agentic Analyst Layer

Date: 2026-06-24  
Purpose: clarify the architectural boundary between the existing data/feature pipeline and the agentic analyst system.

---

## 1. Core distinction

DEAN-OS should separate two layers:

```text
Data / Feature Pipeline
  parses, normalizes, calculates, enriches, stores datasets

Agentic Analyst Layer
  reasons over events, regimes, scenarios, patterns, analogs, and outcomes
```

The pipeline creates structured inputs.  
The agentic layer interprets those inputs.

---

## 2. Data / Feature Pipeline responsibilities

The existing pipeline should remain primarily deterministic and reproducible.

It can build datasets with:

```text
news text
source metadata
sentiment scores
entity links
topic tags
candle features
technical indicators
macro indicators
fundamental snapshots
commodity prices
sector returns
volatility metrics
liquidity metrics
timestamps
as_of_date
```

The pipeline can enrich data, but it should not become the final analyst.

Allowed outputs:

```text
FeaturePacket
MarketSnapshot
MacroSnapshot
NewsEvidencePacket
SourceQualityScore
SentimentScore
EntityLinkSet
TechnicalFeatureSet
```

The pipeline answers:

```text
What data exists?
How was it parsed?
What features were calculated?
What was known as of this time?
Is the data clean and auditable?
```

---

## 3. Agentic Analyst Layer responsibilities

The deeper analysis should be handled by agents.

Agents should analyze:

```text
what the event means
which regime it interacts with
whether the news confirms or contradicts context
what was already priced
which transmission channels matter
which historical analogs are relevant
what happened after analogous events
which scenario paths are plausible
what evidence is missing
how prior reasoning should be self-checked
```

Agent outputs should be review-only:

```text
Regime Snapshot
Event Interpretation
Expectation Gap
Transmission Map
Scenario Outcome Graph
Historical Analog Review
Outcome Horizon Plan
Hypothesis Ledger Update
Evidence Gap List
Analyst Report
Human Review Packet
```

Agents answer:

```text
Why might this matter?
Through which channels?
Against which regime context?
Which scenarios changed?
What should be checked later?
Where might the analysis be wrong?
```

---

## 4. Correct data flow

The recommended architecture:

```text
raw sources
-> ingestion / parsing
-> normalized evidence packets
-> feature calculation
-> feature store / event store / macro store
-> analysis request
-> agentic analyst modules
-> scenario graph + analyst report
-> human review
-> outcome tracking
-> calibration
```

The agent layer should consume the pipeline outputs, not replace the pipeline.

---

## 5. Shared interface objects

### 5.1 NewsEvidencePacket

```text
packet_id:
as_of_date:
publication_time:
retrieval_time:
source_id:
source_quality:
title:
body:
entities:
claims:
topics:
sentiment_score:
language:
provenance:
```

### 5.2 MarketFeatureSnapshot

```text
snapshot_id:
as_of_date:
asset_universe:
candles:
technical_indicators:
volatility:
liquidity:
sector_returns:
commodity_prices:
macro_indicators:
```

### 5.3 AnalysisRequest

```text
request_id:
as_of_date:
news_packet_ids:
market_snapshot_ids:
macro_snapshot_ids:
requested_lenses:
  regime
  event_classification
  expectation_gap
  transmission
  analogs
  scenario_graph
  evidence_gaps
review_only: true
```

### 5.4 AnalystOutputPacket

```text
output_id:
as_of_date:
regime_snapshot:
event_interpretation:
transmission_channels:
expectation_gap:
scenario_graph:
historical_analogs:
evidence_gaps:
hypotheses:
watch_signals:
self_check_plan:
human_review_fields:
forbidden_outputs:
```

---

## 6. ML models as tools, not final authority

Traditional ML models can be useful, but they should be treated as modules/tools inside the analyst system.

Examples:

```text
sentiment model
event classifier
entity linker
volatility forecaster
macro nowcaster
sector reaction estimator
similarity search model
embedding retriever
```

But the final analyst output should combine:

```text
features
+ regime context
+ historical analogs
+ expectation gap
+ scenario graph
+ evidence quality
+ human review
```

Do not let a single return predictor become the entire analyst.

---

## 7. Forecasting boundary

Forecasting should be scenario-based and probabilistic.

Allowed:

```text
scenario probabilities
confidence levels
horizons to track
base-rate comparison
calibration notes
paper/replay evaluation
```

Forbidden:

```text
live order
buy/sell/hold recommendation
position size
broker routing
autonomous execution
production price target
```

---

## 8. Why this boundary matters

If the feature pipeline tries to do the whole analysis, it becomes brittle and hard to audit.

If agents do parsing and feature calculation ad hoc, the analysis becomes unreproducible.

Correct separation:

```text
pipeline = reproducible evidence and features
agents = structured reasoning and interpretation
evaluation layer = outcome tracking and calibration
human review = correction and governance
```

---

## 9. Codex implementation target

Suggested module split:

```text
data_plane/
  source_ingestion
  source_normalization
  feature_calculation
  feature_store
  macro_store
  market_snapshot_store

analysis_plane/
  agent_orchestrator
  regime_context_builder
  event_interpreter
  expectation_gap_engine
  transmission_mapper
  historical_analog_retriever
  scenario_outcome_graph_builder
  hypothesis_ledger
  analyst_report_builder

evaluation_plane/
  outcome_horizon_tracker
  calibration_metrics
  replay_engine
  analyst_self_check

review_plane/
  human_review_console
  correction_logger
  audit_trail
```

---

## 10. Design rule

The pipeline should produce the best possible structured evidence.

The agent system should produce the best possible structured reasoning.

They should communicate through explicit packets, not hidden prompt state.
