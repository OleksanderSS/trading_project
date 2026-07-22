# DEAN-OS Template Kit — How to Add a New Sector

## Architecture (one picture)

```
┌──────────────────────────────────────────────────────────┐
│                    ORCHESTRATOR (1)                        │
│  run_agent_orchestrator.py                                │
├──────────────────────┬───────────────────────────────────┤
│  PIPELINE BRANCH      │  ANALYTICAL BRANCH               │
│  (sequential)         │  (parallel)                      │
│                       │                                  │
│  DomainAnalystAgent×6 │  NewsEventAnalyzerAgent         │
│  PipelineManagerAgent  │  KeywordDomainAgent×8           │
│  RegimeAgent           │  ResearchIngestionAgent         │
│  ContextSynthesisAgent  │  SpecialistResearchAgent        │
│  PipelineAuditAgent    │  FinancialNLPAgent             │
│  DataQualityAgent      │  EvidenceSynthesisAgent         │
│  RiskAgent             │  UnifiedResearchAgent           │
└──────────────────────┴───────────────────────────────────┘
         │                          │
         └──────────┬───────────────┘
                    ▼
          ConsensusEngine → Decision
```

## Core Concept

Each **sector** gets one `DomainAnalystAgent`. The agent is a wrapper around `SectorAnalyst` — the same engine, different **domain profile** (YAML). Cloning a sector = creating a new YAML.

```
SectorAnalyst (1 engine, any domain)
 ├── DomainProfile (YAML — what changes per sector)
 ├── BaseAnalystAgent (evidence → thesis → ticker basket)
 └── 5 lenses (EventClassifier, RegimeContext, TransmissionMapper, HypothesisLedger, EvidenceGap)
```

## How to Add a New Sector

### Step 1: Create domain profile

```bash
python dean_domain_scaffold.py create my_sector --name "My Sector Display Name"
```

This generates `config/domain_profiles/my_sector.yaml` with:
- Core questions (what does this sector analyze)
- Required/useful evidence types
- Sector keywords for NLP matching
- Ticker universe hint
- Contradiction rules
- Direct ticker evidence rules

**Important:** Edit the generated YAML to fill in real sector-specific content. The scaffold provides sensible defaults, but you need to tailor:
- `core_questions` — what matters in this sector
- `sector_keywords` — terms for keyword matching
- `ticker_universe_hint` — public companies in this sector
- `contradiction_rules` — domain-specific logic conflicts

### Step 2: Add to agent registry

The scaffold prints a ready-to-paste YAML entry. Add it under `agents:` in `config/agent_registry.yaml`:

```yaml
my_sector_analyst:
  class_path: dean_os.agents.domain_analyst:DomainAnalystAgent
  branch: pipeline
  veto_level: none
  enabled: true
  error_behavior: skip
  timeout_seconds: 30
  domain_id: my_sector
  horizon_days: 180
  agent_role: standalone_domain_analysis
  decision_influence: false
  execution_group: my_sector_domain_analysis
  run_phases:
    - pre_trade
```

### Step 3: Provide data

The agent reads evidence from `MarketContext`. Data flows via preload flags:

```
--preload-news         → context.news
--preload-macro        → context.macro
--preload-fundamentals → context.fundamentals
--preload-prediction   → context.metadata["stage5_prediction_review"]
--preload-regime       → context.metadata["stage7_regime_review"]
--preload-prices       → context.dataframes["prices"]
--preload-risk         → context.returns
```

If a runtime artifact exists at `reports/dean_os/<domain_id>_analyst_runtime_current/latest.json`, it takes priority. Otherwise the agent falls back to live context evidence.

### Step 4: Run

```bash
python run_agent_orchestrator.py \
  --ticker NVDA \
  --timeframe 15m \
  --as-of 2026-06-30T21:00:00+00:00 \
  --preload-regime \
  --preload-fundamentals latest \
  --preload-prediction latest \
  --soft-mode
```

## Existing Sectors (reference)

| Agent | Domain ID | Profile File | Tickers |
|-------|-----------|-------------|---------|
| semiconductor_analyst | semiconductor_ai_infrastructure | `config/domain_profiles/semiconductor_ai_infrastructure.yaml` | NVDA, AMD, TSM, ASML, ... |
| energy_analyst | energy | `config/domain_profiles/energy.yaml` | XLE, USO, XOM, CVX, ... |
| macro_analyst | macro_policy | `config/domain_profiles/macro_policy.yaml` | (macro indicators) |
| agriculture_analyst | agriculture | `config/domain_profiles/agriculture.yaml` | ADM, BG, DE, MOS, ... |
| logistics_analyst | logistics | `config/domain_profiles/logistics.yaml` | FDX, UPS, JBHT, XPO, ... |
| real_estate_analyst | real_estate | `config/domain_profiles/real_estate.yaml` | PLD, AMT, EQIX, WELL, ... |
| geopolitics_analyst | geopolitics | `config/domain_profiles/geopolitics.yaml` | (macro indicators) |
| liquidity_credit_analyst | liquidity_credit | `config/domain_profiles/liquidity_credit.yaml` | (credit spreads) |

## Adding Analytical Agents

For keyword-based analytical agents (like `macro_policy`, `geopolitical`, `news_catalyst`), extend `KeywordDomainAgent` in `dean_os/agents/domain_research.py`:

```python
class MySectorKeywordAgent(KeywordDomainAgent):
    keywords = ("keyword1", "keyword2")
    bullish_terms = ("positive_term",)
    bearish_terms = ("negative_term",)
    thesis_template = "My sector thesis template"
    asset_or_sector = "my_sector"
```

Then register in `agent_registry.yaml` with `branch: analytical`.

## Pipeline Manager (singleton)

The `PipelineManagerAgent` is a **single instance** for the composite semiconductor pipeline. It is NOT cloned per sector. It manages:
- Feature/timeframe audit
- Target readiness
- Stage4 review
- Prediction review
- Sector-to-ticker mapping

## Orchestrator (singleton)

The orchestrator (`run_agent_orchestrator.py`) runs all enabled agents. New domain analysts are auto-discovered via the agent registry — no code changes needed.

## World State Snapshot

After the orchestrator runs, build a unified world state:

```bash
python run_build_world_state.py --summary
```

Output:
```
World State @ 2026-07-01T12:00:00Z
  Decision: watchlist (confidence=0.25)
  Regime: CRISIS (0.8)
  Sectors: 6
    Energy: bearish (conf=0.40)
    Semiconductors & AI Infrastructure: mixed (conf=0.55)
    ...
  Unknowns: 4
    [high] Real-time CoWoS capacity utilization
    [high] GPU availability lead times for enterprise vs hyperscaler
```

Module: `dean_os/world_state.py` — `WorldStateBuilder.build(reports, decision)`

## Dependency Graph

Structural economic relationships defined in YAML:

```yaml
# config/dependency_graphs/semiconductor_ai_infrastructure.yaml
edges:
  - from: hbm_memory
    to: gpu
    type: structural
    lag: immediate
    strength: 0.95
  - from: gpu
    to: ai_servers
    type: structural
    lag: months
    strength: 0.85
```

Query with `dean_os/dependency_graph.py`:
```python
g = load_dependency_graph("semiconductor_ai_infrastructure")
g.traverse(["hbm_memory"], max_depth=3)  # → [gpu, ai_servers, data_centers]
g.find_path("hbm_memory", "cloud_revenue")  # → paths with strength/lag
```

## News Event Analyzer

Replaces simple keyword counting with structured event classification. Agent `news_event_analyzer` classifies each news item by:
- **Event type**: macro, monetary_policy, geopolitical, corporate, supply_chain, technology, energy, ...
- **Shock**: positive/negative/neutral with confidence
- **Impact magnitude**: -1 to 1
- **Predictability**: 0 (surprise) to 1 (expected)
- **Affected sectors**: auto-detected
- **Time to impact**: immediate, days, weeks, months, quarters

Module: `dean_os/agents/news_event_analyzer.py` — `NewsEventAnalyzerAgent`

## Unknown Graph

First-class tracking of system unknowns:

```python
from dean_os.unknown_graph import get_domain_unknowns
graph = get_domain_unknowns("semiconductor_ai_infrastructure")
graph.get_high_priority()     # → urgent unknowns
graph.get_collector_fixable() # → can be automated
graph.summary()               # → "[semiconductor] 6 unknowns (3 high, 2 collector-fixable)"
```

Module: `dean_os/unknown_graph.py` — `UnknownGraph`, `UnknownEntry`

## Outcome Tracker

The OutcomeTracker accumulates events + predictions and checks outcomes at fixed replay intervals.

Core loop:
1. **Register** — when NewsEventAnalyzer detects a significant event (|impact|>0.3 or geopolitical/natural_disaster/credit_financial), it auto-registers with predictions at 1/5/30/60/120 day intervals
2. **Check** — `tracker.check_due(stances)` checks all elapsed intervals against current sector stances
3. **Calibrate** — `tracker.calibrate()` calculates Brier score + accuracy rate per interval
4. **Adjust** — HistoricalAnalogiesAgent uses calibration to penalize confidence (low accuracy → lower confidence)

```bash
# View tracker stats
python dean_domain_scaffold.py calibration
```

Module: `dean_os/outcome_tracker.py` — `OutcomeTracker`, `REPLAY_INTERVALS=[1,5,30,60,120]`

Over time, the tracker builds a calibration curve: _"When the news analyzer says bullish with 0.8 confidence, we're right X% of the time at 30 days."_

## Paper Trade ↔ Outcome Tracker Bridge

When a paper trade is created, `OutcomeTracker.register_paper_trade()` auto-registers it at matching intervals (if horizon=30d, intervals 1/5/30 get registered). When `check_due()` runs, it returns hit/miss labels that can update `PaperTradeRecord.outcome_label`.

```python
from dean_os.outcome_tracker import OutcomeTracker
tracker = OutcomeTracker()
trade = {"trade_id": "abc", "thesis": "NVDA bullish on earnings", "expected_direction": "bullish",
         "horizon_days": 30, "confidence": 0.7}
tracker.register_paper_trade(trade)
```

## Coherence Scan

`CoherenceScanAgent` cross-references all domain agent verdicts against an overlap map and flags contradictions. Loaded as agent `coherence_scan` (analytical branch, enabled).

```bash
# View overlap pairs and tracker calibration context
python dean_domain_scaffold.py coherence
```

Module: `dean_os/agents/coherence_scan.py` — `CoherenceScanAgent`, `OVERLAP_PAIRS` (13 pairs)

## Domain Readiness Check

Validates all domain profiles at a glance — YAML integrity, agent registration, keyword/ticker counts, enabled status.

```bash
python dean_domain_scaffold.py check
```

## Freshness Audit

`FreshnessAuditAgent` checks each data point in `MarketContext` against `as_of`:
- News >7 days stale, Macro >30d, Prices >5d, Fundamentals >90d
- Flags stale items, missing timestamps, produces freshness score

Agent: `freshness_audit` (analytical, enabled by default).

## System Health Check

Validates all critical infrastructure in one command:

```bash
python dean_domain_scaffold.py health
```

Checks: DuckDB (tables/rows), OutcomeTracker, Registry (conflicts), Domain Profiles, Keyword Index, Artifacts (features/macro/predictions/runtime).

## Agent Run Statistics

Tracks every agent run in SQLite. View:

```bash
python dean_domain_scaffold.py stats
```

Shows: total runs by agent, by verdict, average confidence, latest runs.

Module: `dean_os/agent_stats.py` — `AgentStatsStore.log_run()` for agent integration.

## CLI Reference

| Command | What |
|---------|------|
| `create <id>` | New domain profile (YAML + optional --name) |
| `list` / `list --details` | List domain profiles / with evidence/keyword/ticker counts |
| `calibration` | Outcome tracker calibration (Brier, accuracy by interval) |
| `check` | Domain readiness check |
| `coherence` | Coherence scan overlap map (14 agents, 13 pairs) |
| `health` / `health --json` | System health check |
| `stats` / `stats --json` | Agent run statistics |
| `inventory` / `inventory --json` | DuckDB table inventory (cols, rows, dates) |
| `search <col>` | Search columns across DuckDB tables |
| `list-agents` | Agent registry table (38 agents, enabled/disabled) |
| `registry show <name>` | Show agent config from registry |
| `profiles show <id>` | Show domain profile metadata |
| `validate-config` | Validate YAML configs + class_path resolution |
| `outcomes` | Outcome tracker events + paper trades + calibration |
| `diag` | One-page system diagnostic (registry, DuckDB, profiles) |
| `dq` | DuckDB data quality report (null ratios, dates, duplicates) |
| `--json` | JSON output flag (works with: health, stats, inventory) |

## Architecture Rules

1. **BaseAgent** — abstract, requires `async def run(context) -> Report`
2. **Pipeline agents** — sequential, can veto (hard/soft)
3. **Analytical agents** — parallel, research-only, no veto
4. **Domain analyst** — a single `DomainAnalystAgent` + YAML profile = one sector
5. **Evidence** — structured `AnalystEvidenceItem` with source, stance, reliability, freshness
6. **Lenses** — plugins that enrich `AnalysisPacket`; registered in `_build_default_registry()`
7. **Consensus** — deterministic scoring: model (40%) + risk (35%) + regime (25%) + analytical modifier
8. **World State** — `WorldStateBuilder` aggregates all reports into a unified snapshot
9. **Dependency Graph** — structural economic relationships with lag/strength/confidence
10. **News Events** — structured classification replaces keyword counting
11. **Unknowns** — `UnknownGraph` tracks what the system knows it doesn't know
12. **Outcomes** — `OutcomeTracker` accumulates event→prediction→outcome→calibration at fixed intervals
13. **Coherence** — `CoherenceScanAgent` cross-references overlapping domains for contradictions
14. **Readiness** — `dean_domain_scaffold.py check` validates all domain profiles at once
