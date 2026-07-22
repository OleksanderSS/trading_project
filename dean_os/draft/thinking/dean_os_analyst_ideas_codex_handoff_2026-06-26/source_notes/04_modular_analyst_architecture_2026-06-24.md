# DEAN-OS Addendum — Modular Analyst Architecture

Date: 2026-06-24  
Purpose: define a modular architecture for extensible news/text analysis with regime states, patterns, historical analogs, scenario graphs, and outcome self-check.

This extends the prior DEAN-OS notes:
- macro / industrial / AI analyst journal;
- outcome memory and timeframe self-check;
- regime context and scenario outcome graph;
- additional analyst observations.

---

## 1. Core principle

The analyst system should not be one monolithic "news -> answer" agent.

It should be a modular pipeline:

```text
source/news/text
-> normalized evidence packet
-> modular analyst lenses
-> shared analysis state
-> scenario graph
-> journal/review packet
-> outcome tracking
-> calibration
```

Each module should add, revise, or challenge part of the analysis without needing to rewrite the whole system.

---

## 2. Why modularity is necessary

The system will keep expanding:

```text
regime context
historical analogs
expectation gap
industrial M&A
AI capex
safe havens
commodity shocks
contradiction matrix
hypothesis ledger
scenario outcome graphs
outcome self-check
```

If all logic lives inside one agent prompt or one predictor, the system becomes fragile.

A modular design allows new analysis lenses to be added as plugins:

```text
new_module(input_packet, current_analysis_state) -> updated_analysis_state
```

---

## 3. Base objects

### 3.1 Source packet

Raw or normalized input from news, reports, filings, transcripts, macro releases, or company disclosures.

```text
source_packet:
  source_id:
  source_type:
  publication_time:
  retrieval_time:
  title:
  body:
  url_or_reference:
  source_quality:
  language:
  raw_entities:
  raw_claims:
  as_of_date:
```

### 3.2 Analysis packet

The shared state that modules update.

```text
analysis_packet:
  packet_id:
  as_of_date:
  source_packet_ids:
  event_records:
  entity_links:
  regime_context:
  event_clusters:
  transmission_channels:
  expectation_gap:
  scenario_graph:
  historical_analogs:
  evidence_gaps:
  contradictions:
  hypotheses:
  watch_signals:
  review_notes:
  forbidden_outputs:
```

This packet is the main object passed through the pipeline.

---

## 4. Modular lens pattern

Each module should follow the same interface:

```text
module_name:
module_version:
input_contract:
output_contract:
required_fields:
optional_fields:
confidence_output:
evidence_ids_used:
assumptions_added:
evidence_gaps_added:
safety_boundary:
```

Generic function shape:

```text
analyze(input_packet, analysis_state, config) -> analysis_state_delta
```

The module should not overwrite the entire state. It should return a delta.

---

## 5. Suggested pipeline

```text
1. Source Normalizer
2. Source Quality Scorer
3. Event Classifier
4. Entity / Asset / Sector Linker
5. Regime Context Builder
6. Event Cluster Detector
7. Transmission Mapper
8. Expectation Gap Engine
9. Historical Analog Retriever
10. Historical Outcome Retriever
11. Contradiction Matrix Builder
12. Evidence Gap Prioritizer
13. Hypothesis Ledger Updater
14. Scenario Outcome Graph Builder
15. Watch Signal Builder
16. Analyst Report Builder
17. Human Review Console
18. Outcome Horizon Tracker
19. Self-Check / Calibration Layer
```

Not every daily news item needs every module. The orchestrator should decide which modules are relevant.

---

## 6. Module details

### 6.1 Source Normalizer

Purpose:

```text
Convert raw text/news/filing/report into a normalized source packet.
```

Outputs:

```text
clean text
timestamps
source type
source metadata
language
basic provenance
```

---

### 6.2 Source Quality Scorer

Purpose:

```text
Evaluate reliability, source class, recency, directness, and risk of unsupported claims.
```

Useful fields:

```text
source_quality:
  primary_source | reputable_news | analyst_report | social_media | unknown

directness:
  direct_statement | reported_by_media | anonymous_sources | market_rumor

confidence:
  low | medium | high
```

---

### 6.3 Event Classifier

Purpose:

```text
Classify what happened.
```

Examples:

```text
war_escalation
de_escalation
sanctions_change
central_bank_decision
inflation_release
strategic_industrial_asset_mna
plant_closure
plant_expansion
ai_capex_announcement
memory_supply_constraint
power_grid_constraint
defense_contract_change
commodity_supply_shock
```

---

### 6.4 Entity / Asset / Sector Linker

Purpose:

```text
Map text entities to companies, sectors, commodities, geographies, and assets.
```

Examples:

```text
company -> sector -> supply chain role
country -> geopolitical bloc -> sanctions exposure
commodity -> CPI/PPI component -> affected sectors
factory -> strategic capacity -> defense relevance
```

---

### 6.5 Regime Context Builder

Purpose:

```text
Build or update the date-specific regime vector.
```

Fields:

```text
geopolitical_state
economic_phase
inflation_rates_context
liquidity_credit_context
market_state
commodity_real_economy_stress
ai_tech_cycle
safe_haven_behavior
```

Each field should have:

```text
state
intensity
trend
confidence
evidence_ids
notes
```

---

### 6.6 Event Cluster Detector

Purpose:

```text
Detect whether multiple news items form a bigger theme.
```

Examples:

```text
oil shock + CPI + Fed repricing
AI capex + HBM shortage + power bottleneck
steel M&A + defense budget + tariffs
food shock + fertilizer + shipping stress
```

A cluster may matter more than any single news item.

---

### 6.7 Transmission Mapper

Purpose:

```text
Map the event into economic and market channels.
```

Examples:

```text
oil -> gasoline -> headline CPI -> Fed pricing -> growth multiples
steel M&A -> industrial policy -> defense supply chain -> capex -> margins
AI capex -> HBM -> foundry capacity -> power demand -> utilities/grid
food shock -> grocery inflation -> consumer margins -> rates expectations
```

---

### 6.8 Expectation Gap Engine

Purpose:

```text
Estimate actual outcome vs what was priced or expected.
```

Fields:

```text
expected_scenario_before:
actual_development:
priced_probability_estimate:
surprise_direction:
surprise_magnitude:
already_priced_likelihood:
positioning_crowdedness:
```

This module prevents simplistic logic like:

```text
bad news -> asset down
good news -> asset up
```

---

### 6.9 Historical Analog Retriever

Purpose:

```text
Find structurally similar historical cases.
```

Similarity should include:

```text
event type
regime context
transmission channel
expectation gap
asset/sector exposure
policy reaction
```

It should return both confirming and disconfirming analogs.

---

### 6.10 Historical Outcome Retriever

Purpose:

```text
Retrieve what happened after similar events over fixed horizons.
```

Default horizons:

```text
1d / 5d / 20d / 60d / 120d
```

This is needed for base rates and self-check.

---

### 6.11 Contradiction Matrix Builder

Purpose:

```text
Track where evidence supports or contradicts other evidence.
```

Example:

```text
AI capex supports semis
higher rates pressure growth valuations
power bottleneck constrains data-center expansion
oil de-escalation lowers inflation fear
```

Relationships:

```text
supports
contradicts
weakens
depends_on
unknown
```

---

### 6.12 Evidence Gap Prioritizer

Purpose:

```text
Rank missing information by how much it would change scenario probabilities.
```

Examples:

```text
Hormuz:
  tanker flow, insurance rates, sanctions terms

AI:
  HBM supply, data-center power, customer concentration, lease financing

Strategic M&A:
  national-security terms, capex commitments, defense supply links
```

---

### 6.13 Hypothesis Ledger Updater

Purpose:

```text
Store analyst hypotheses explicitly.
```

Schema:

```text
hypothesis_id
as_of_date
hypothesis
confidence
supporting_evidence
contradicting_evidence
expected_observations
invalidation_signals
horizons_to_check
status
calibration_note
```

This prevents analyst reasoning from becoming vague narrative.

---

### 6.14 Scenario Outcome Graph Builder

Purpose:

```text
Build potential future paths from current regime + event + transmission channels.
```

Graph shape:

```text
regime_state
-> event
-> transmission_channel
-> expectation_gap
-> scenario_node
-> outcome_horizon
-> self_check
```

Nodes:

```text
regime_state
event
transmission_channel
expectation_gap
scenario
observable_signal
outcome_horizon
invalidation_signal
self_check
```

Edges:

```text
causal_channel
conditional_update
supports
contradicts
confirms
invalidates
leads_to
observed_after
calibrates
```

---

### 6.15 Watch Signal Builder

Purpose:

```text
Generate non-execution watch signals.
```

Allowed:

```text
watch this data
track this horizon
check this source
revisit this hypothesis
```

Forbidden:

```text
buy
sell
hold
position size
trade execution
price target as production recommendation
```

---

### 6.16 Analyst Report Builder

Purpose:

```text
Create a readable analyst report from structured state.
```

Daily report format:

```text
Date
Regime Snapshot
Top developments
News vs regime
Scenario graph updates
Practical implications
Risks / uncertainties
What to watch next
DEAN-OS journal notes
```

---

### 6.17 Human Review Console

Purpose:

```text
Allow human correction of labels, assumptions, analogs, evidence gaps, and scenario probabilities.
```

Review outputs:

```text
corrected_event_class
corrected_regime_state
accepted_analogs
rejected_analogs
missing_evidence
scenario_probability_adjustment
reviewer_notes
```

---

### 6.18 Outcome Horizon Tracker

Purpose:

```text
Track realized outcomes after 1d / 5d / 20d / 60d / 120d.
```

This turns analyst notes into evaluation data.

---

### 6.19 Self-Check / Calibration Layer

Purpose:

```text
Compare predicted scenario probabilities with realized outcomes.
```

Metrics:

```text
scenario_hit
brier_score
rank_of_realized_path
overconfidence_flag
false_analogy_flag
missed_channel_flag
expectation_gap_error
```

---

## 7. Orchestrator logic

The orchestrator should not make analysis itself. It should route packets.

Example:

```text
if event_class in [war_escalation, de_escalation, sanctions_change]:
  run regime_context_builder
  run transmission_mapper
  run expectation_gap_engine
  run historical_analog_retriever
  run scenario_graph_builder

if event_class in [strategic_industrial_asset_mna, plant_closure, plant_expansion]:
  run industrial_asset_classifier
  run supply_chain_mapper
  run strategic_capacity_assessor
  run second_order_exposure_graph

if event_class in [ai_capex_announcement, memory_supply_constraint, power_grid_constraint]:
  run ai_cycle_decomposer
  run infrastructure_bottleneck_mapper
  run valuation_expectation_gap
```

The orchestrator should produce an audit trail:

```text
which modules ran
why they ran
which modules were skipped
what fields changed
what evidence was used
```

---

## 8. Plugin registry

Codex should implement a registry so modules can be added without changing the whole pipeline.

```text
module_registry:
  module_name:
  version:
  input_fields_required:
  output_fields_produced:
  event_classes_supported:
  regime_fields_supported:
  safety_level:
  tests:
```

This enables later extension:

```text
add new module:
  weather_agriculture_shock_mapper
  shipping_insurance_rate_tracker
  defense_procurement_delay_analyzer
  ai_agent_observability_evaluator
```

---

## 9. State update discipline

Modules should not freely overwrite previous analysis.

Use deltas:

```text
analysis_state_before
module_delta
analysis_state_after
```

Each delta should record:

```text
module_name
module_version
fields_added
fields_modified
evidence_ids
confidence
reason_for_change
```

This makes the system auditable and debuggable.

---

## 10. Safety / authority boundary

The modular architecture must preserve the review-only boundary.

Allowed outputs:

```text
classification
regime snapshot
scenario probabilities
evidence gaps
historical analogs
outcome tracking
watch signals
review packets
calibration notes
```

Forbidden outputs:

```text
live order
buy/sell/hold instruction
position sizing
broker routing
autonomous execution
production price targets
model promotion without gates
```

---

## 11. Minimal implementation plan

Phase 1:

```text
schemas:
  SourcePacket
  AnalysisPacket
  RegimeContextVector
  EventRecord
  ModuleDelta
  ScenarioOutcomeGraph
```

Phase 2:

```text
module registry
source normalizer
event classifier
regime context builder
transmission mapper
expectation gap engine
analyst report builder
```

Phase 3:

```text
historical analog retriever
historical outcome retriever
outcome horizon tracker
self-check / calibration
```

Phase 4:

```text
event cluster detector
contradiction matrix
evidence gap prioritizer
hypothesis ledger
second-order exposure graph
```

Phase 5:

```text
human review console
paper/replay evaluation
module quality metrics
```

---

## 12. Key design rule

Do not build a single "smart analyst prompt" as the core system.

Build a modular analyst workbench:

```text
shared state
+ plugin lenses
+ schema contracts
+ audit trail
+ review console
+ outcome memory
```

This lets DEAN-OS expand from simple news analysis to regime-aware, pattern-aware, historically calibrated macro/industrial/AI analysis.
