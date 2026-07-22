# DEAN-OS Addendum — Pipeline Control Branch + Domain Analyst Branch

Date: 2026-06-24  
Purpose: clarify the architecture where the existing pipeline parses/enriches/trains/compares models, while separate agentic branches tune the pipeline within guarded limits and perform deep qualitative domain analysis.

---

## 1. Core architecture

DEAN-OS should be separated into two major branches that communicate through explicit packets and gates.

```text
Data / ML Pipeline Branch
  parses
  normalizes
  enriches
  builds features
  trains models
  compares model runs
  evaluates metrics
  produces datasets, model reports, and experiment results

Agentic System Branch
  does not replace the pipeline
  tunes the pipeline only within allowed guardrails
  performs deep news and domain analysis
  builds scenario graphs and hypotheses
  performs historical analog and outcome reasoning
  sends review packets and parameter proposals
```

The pipeline is the reproducible machine.  
The agentic system is the reasoning, supervision, and interpretation layer.

---

## 2. Data / ML Pipeline Branch

The pipeline may perform deterministic and ML operations:

```text
source ingestion
news parsing
entity linking
sentiment scoring
macro feature enrichment
candle feature calculation
technical indicator calculation
dataset construction
train/test split creation
model training
model comparison
walk-forward validation
backtest / replay
metric calculation
experiment logging
```

Pipeline outputs:

```text
FeatureDataset
NewsEvidencePacket
MarketFeatureSnapshot
MacroSnapshot
TrainingRunReport
BacktestReport
ModelComparisonReport
DataQualityReport
```

Important: the pipeline can train and compare models, but it should not make qualitative macro/geopolitical/industrial judgments by itself.

---

## 3. Pipeline Controller Agent Branch

A separate agentic branch can supervise and tune the pipeline, but only within strict limits.

Suggested name:

```text
Pipeline Controller Agent
```

Purpose:

```text
read pipeline reports
detect degradation
propose parameter changes
adjust allowed config values through gates
request retraining or replay
flag data-quality problems
compare train/test behavior
prevent overfitting and leakage
```

Allowed tuning domains:

```text
train/test split ratio within approved range
walk-forward window length within approved range
feature inclusion/exclusion proposals
model hyperparameter ranges
data-quality thresholds
minimum source quality thresholds
retraining cadence
experiment selection for paper/replay
risk metric thresholds
```

Forbidden:

```text
direct live trading
unbounded hyperparameter mutation
changing production configs without approval gates
bypassing no-lookahead checks
selecting models solely by PnL
removing validation constraints to improve apparent results
promoting models without review
```

---

## 4. Pipeline tuning must be gated

The controller agent should not directly rewrite the pipeline freely.

Use a gated control contract:

```text
PipelineMetricPacket
-> ControllerAgentReview
-> ParameterChangeProposal
-> GuardrailValidation
-> HumanReview / Auto-allowed low-risk change
-> ExperimentRun
-> OutcomeComparison
-> PromotionGate
```

Every change must record:

```text
proposal_id
as_of_date
metric_trigger
current_value
proposed_value
allowed_range
reason
expected_effect
risk
guardrail_check
review_status
result_after_change
```

---

## 5. Important metrics for controller agent

The controller agent should monitor more than PnL.

Core groups:

```text
Performance:
  PnL
  return
  Sharpe / Sortino
  max drawdown
  win rate
  profit factor

Generalization:
  train/test ratio
  train vs test performance gap
  walk-forward stability
  out-of-sample degradation
  cross-period robustness

Risk:
  drawdown
  tail loss
  volatility
  exposure concentration
  turnover
  transaction costs
  slippage

Data quality:
  missing values
  timestamp integrity
  source quality
  duplicate rate
  stale data
  feature drift

Leakage / validity:
  no-lookahead violations
  target leakage
  label leakage
  survivorship bias
  publication-time mismatch

Model behavior:
  prediction calibration
  confidence vs realized outcome
  feature importance drift
  regime-specific performance
  overfitting flags
```

PnL is important, but it must never be the only optimization target.

---

## 6. Domain Analyst Agent Branch

A separate branch should contain domain analysts.

Each analyst is not just a sentiment model.  
Each analyst should know:

```text
economics
history
politics / geopolitics
sector structure
industrial capacity
company fundamentals
supply chains
financial markets
evidence quality
historical analogs
scenario reasoning
```

Analysts should be specialized by domain:

```text
Macro Analyst
Energy Analyst
Metals / Industrial Analyst
Food / Agriculture Analyst
Defense / Geopolitics Analyst
Semiconductor / AI Infrastructure Analyst
Financials / Credit Analyst
Consumer / Retail Analyst
Healthcare / Biotech Analyst
Logistics / Shipping Analyst
```

Each analyst consumes structured pipeline packets plus raw evidence references.

---

## 7. What domain analysts produce

Domain analysts should perform qualitative, evidence-based reasoning:

```text
news interpretation
event classification
regime interaction
historical analog search
transmission channel mapping
expectation-gap analysis
scenario outcome graph
evidence-gap prioritization
contradiction matrix update
hypothesis ledger update
watch signals
human-review packet
```

They should not output direct live trading instructions.

---

## 8. Domain Analyst interface

Input:

```text
AnalysisRequest:
  as_of_date
  source_packets
  feature_snapshots
  macro_snapshots
  market_snapshots
  current_regime_context
  requested_domain
  review_only: true
```

Output:

```text
DomainAnalystReport:
  analyst_id
  domain
  as_of_date
  event_interpretation
  affected_entities
  affected_sectors
  affected_commodities
  transmission_channels
  expectation_gap
  historical_analogs
  scenario_graph
  evidence_gaps
  hypotheses
  watch_signals
  confidence
  human_review_fields
  forbidden_outputs
```

---

## 9. Analyst Orchestrator

An orchestrator should route news to relevant analysts.

Example:

```text
Hormuz / oil shock:
  Macro Analyst
  Energy Analyst
  Geopolitics Analyst
  Shipping Analyst
  Inflation/Rates lens

Nippon Steel / U.S. Steel:
  Industrial Analyst
  Geopolitics Analyst
  Defense Supply Chain Analyst
  Materials Analyst

AI data-center power bottleneck:
  AI Infrastructure Analyst
  Utilities / Power Analyst
  Semiconductor Analyst
  Credit / Financing Analyst
```

The orchestrator should combine their outputs into a synthesis, while preserving disagreements.

---

## 10. Multi-analyst disagreement is useful

Analysts should not be forced into one answer too early.

The system should record:

```text
analyst_agreement:
  high | medium | low

disagreement_points:
  channel importance
  scenario probabilities
  expected time horizon
  evidence quality
  analog relevance
  market pricing assumptions
```

Example:

```text
Energy Analyst:
  oil risk premium may fade quickly if physical disruption is absent.

Geopolitics Analyst:
  tail risk remains high because political incentives are unstable.

Macro Analyst:
  inflation impact depends on persistence, not the first oil move.
```

This disagreement is signal, not a bug.

---

## 11. Integration between branches

The two agentic branches should communicate but not collapse into each other.

```text
Pipeline Controller Agent:
  focuses on data, models, metrics, training behavior, configuration, evaluation

Domain Analyst Agents:
  focus on meaning, regimes, history, politics, economics, sectors, scenarios
```

Where they meet:

```text
domain analyst identifies important new variable
-> proposes feature idea
-> pipeline controller checks feasibility and data quality
-> pipeline runs experiment
-> evaluation layer checks whether the feature improves out-of-sample behavior
-> human review approves or rejects integration
```

Example:

```text
Domain analyst:
  tanker insurance rates may matter during Hormuz risk.

Pipeline controller:
  check if data source exists, timestamp quality, missingness, lag, and whether feature improves replay.

Evaluation:
  compare performance on similar historical chokepoint episodes.
```

---

## 12. Proposed high-level architecture

```text
data_plane/
  ingestion
  parsing
  feature_engineering
  dataset_builder
  training_runner
  model_comparator
  backtest_replay
  metric_store

control_plane/
  pipeline_controller_agent
  parameter_change_proposal
  guardrail_validator
  experiment_scheduler
  promotion_gate

analysis_plane/
  analyst_orchestrator
  macro_analyst
  energy_analyst
  industrial_analyst
  geopolitics_analyst
  semiconductor_ai_analyst
  credit_analyst
  sector_analysts

evaluation_plane/
  outcome_memory
  horizon_tracker
  calibration_metrics
  replay_evaluator
  feature_value_evaluator

review_plane/
  human_review_console
  correction_logger
  audit_trail
  approval_workflow
```

---

## 13. Key design rule

Do not make the pipeline "smart" in a qualitative sense.

Do not make agents do uncontrolled data engineering.

Correct split:

```text
pipeline:
  reproducible calculation, training, comparison

pipeline controller agent:
  guarded tuning and experiment proposals

domain analyst agents:
  deep qualitative analysis and scenario reasoning

evaluation layer:
  checks what worked by horizon and regime

human review:
  approves, corrects, rejects, and teaches
```

---

## 14. Failure modes

```text
optimizing only PnL
overfitting train/test splits
agent bypassing validation to improve results
qualitative analyst hallucinating facts not in evidence
pipeline treating sentiment as full understanding
domain analyst ignoring market pricing
controller agent making too many small config changes
models promoted without out-of-sample stability
sector analyst outputs collapsing into direct trade recommendations
```

---

## 15. Codex implementation priority

Phase 1:

```text
define plane boundaries:
  data_plane
  control_plane
  analysis_plane
  evaluation_plane
  review_plane
```

Phase 2:

```text
define packets:
  PipelineMetricPacket
  ParameterChangeProposal
  GuardrailValidationReport
  DomainAnalystReport
  AnalystSynthesisPacket
```

Phase 3:

```text
implement Pipeline Controller Agent in review-only / proposal-only mode
```

Phase 4:

```text
implement Analyst Orchestrator and 2–3 first domain analysts:
  Macro
  Energy
  Semiconductor / AI Infrastructure
```

Phase 5:

```text
connect analyst feature proposals to pipeline experiments through gates
```

---

## 16. Authority boundary

Allowed:

```text
pipeline metric review
parameter change proposals
guardrail checks
paper/replay experiments
domain analyst reports
scenario probabilities
feature proposals
human review packets
calibration notes
```

Forbidden:

```text
live order
direct buy/sell/hold recommendation
position sizing
broker routing
autonomous execution
ungated production config write
model promotion without gates
unbounded self-modification
```
