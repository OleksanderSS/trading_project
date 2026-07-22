# DEAN-OS World Model Architecture Principles v2

Date: 2026-07-06  
Status: Architecture principles / working specification  
Purpose: Core DEAN-OS ideas for Codex/agent handoff.

---

## 0. Core Thesis

DEAN-OS should not be treated as a trading bot or a simple news summarizer.

It should be a **probabilistic world-modeling system**.

The primary object is not the market, not a ticker, not a headline, and not a single forecast.

The primary object is:

```text
World State
```

Markets, companies, rates, inflation, logistics, wars, AI, industrial capacity, commodities, and consumer behavior are all different projections of one large, evolving world model.

---

## 1. World State Is the Primary Object

The system models the state of the world.

News is only an update to that state.

Wrong framing:

```text
News
→ Summary
→ Prediction
```

Correct framing:

```text
World State
→ News / Data Update
→ Context Graph Update
→ Scenario Probability Update
→ Replay / Calibration
```

A daily briefing is not a list of headlines. It is a daily update of the DEAN-OS World Model.

---

## 2. Multi-Resolution Context

Context exists at multiple time scales.

```text
Civilizational context: decades
Structural context: 5–15 years
Business-cycle context: years
Current-regime context: months
Current-event context: days/weeks
Intraday context: hours
```

A news item should update only the level that it actually changes.

Example:

```text
AI infrastructure buildout = structural context
Samsung raises near-term memory guidance = current-event context
```

The system must not confuse short-term headlines with long-term structural changes.

---

## 3. Two Independent Context Systems

DEAN-OS should maintain two separate context systems.

### 3.1 Context Grid

Qualitative/news/event context.

Examples:

```text
war
peace
sanctions
export controls
industrial policy
new factories
plant closures
M&A
supply disruptions
political shocks
natural disasters
geopolitical events
company guidance
```

### 3.2 Indicator State Grid

Quantitative/metric context.

Examples:

```text
CPI
PPI
rates
yield curve
GDP
PMI
labor
credit spreads
liquidity
retail sales
housing starts
freight rates
electricity demand
oil
gas
copper
steel
food prices
inventories
margins
capex
```

Only after these two systems are built separately should they be merged into a World State.

---

## 4. World State Vector

Each point in time should be represented by a multi-layer World State Vector.

Example:

```text
geopolitics = regional conflict
inflation = sticky
rates = restrictive
labor = cooling
GDP = slowing
credit = neutral/tight
liquidity = neutral
oil = easing
freight = rising
consumer = softening
industry = mixed
AI = capex boom
power = constrained
semiconductors = memory bottleneck
food = mixed
defense = capacity expansion
```

This vector should exist globally, regionally, sectorally, and where possible at company/asset level.

---

## 5. Sector Universe and Coverage Gate

The system must maintain a mandatory universe of major economic nodes.

Example sector universe:

```text
Macro / central banks
Government / fiscal policy
Credit / banking / liquidity
Consumer / FMCG
Retail
Non-FMCG / discretionary
Housing
Construction
Building materials
Industrials
Capital goods
Manufacturing
Logistics
Shipping
Ports
Rail
Trucking
Air cargo
Autos
Steel
Copper
Aluminium
Mining
Chemicals
Fertilizers
Agriculture
Food
Oil
Gas
LNG
Refining
Electricity
Utilities
Telecom
Healthcare
Defense
Semiconductors
AI
Data centers
Real estate
Strategic industrial assets
```

Every briefing must explicitly check each major node.

If there is no credible sourced material update, write:

```text
No credible material update found.
```

Do not silently omit sectors.

This prevents the briefing from collapsing back into only:

```text
AI
Oil
Fed/labor
Markets
```

---

## 6. Context Grid Layers

Context should be layered.

```text
Global
  ↓
Regional
  ↓
Country
  ↓
Sector
  ↓
Industry
  ↓
Company
  ↓
Asset / factory / plant / mine / port / data center
```

For every important news item, the system should map:

```text
global context
regional/local context
sector context
adjacent sectors
structural dependencies
transmission channels
probability/impact update
evidence gaps
```

---

## 7. Similar World States, Not Similar News

The historical search engine should not search primarily for similar headlines.

Wrong:

```text
Find similar news about tariffs.
```

Correct:

```text
Find historical World States with similar inflation, rates, labor, credit,
geopolitical, commodity, logistics, industrial, and consumer structure.
```

DEAN-OS should search for clusters of historically similar World States.

Different analogs may match different layers:

```text
Energy analog
Credit analog
Labor analog
Consumer analog
Logistics analog
Industrial analog
AI/capex analog
Geopolitical analog
```

The output should be an analog cluster, not one best match.

---

## 8. State Transition Graph

DEAN-OS should not only know the current state.

It should model historical transitions between states.

Example macro transition:

```text
Expansion
  ↓
Inflation pressure
  ↓
Rate hiking
  ↓
Credit tightening
  ↓
Slowdown
  ↓
Recession risk
  ↓
Recovery
```

Example geopolitical-industrial transition:

```text
Peace
  ↓
Regional conflict
  ↓
Sanctions
  ↓
Supply shock
  ↓
Energy shock
  ↓
Industrial relocation
  ↓
New equilibrium
```

The system should ask:

```text
Given the current World State, what states have historically tended to follow?
```

This is not a deterministic forecast. It is a conditional probability map.

---

## 9. Dependency Graph

The Dependency Graph contains mostly structural relationships.

Examples:

```text
Natural gas
  → ammonia
  → fertilizers
  → crop yields
  → food prices
  → inflation
```

```text
Copper
  → grid infrastructure
  → electricity capacity
  → data centers
  → AI deployment
```

```text
HBM
  → GPUs
  → AI servers
  → data centers
  → cloud capex
```

Dependency Graph edges should be classified as:

```text
Structural
Cyclical
Event-driven
```

### Structural

Stable physical, technological, or economic relationships.

### Cyclical

Relationships that depend on the business cycle or regime.

### Event-driven

Temporary links activated by specific events.

---

## 10. Transmission Graph

The Transmission Graph describes how shocks propagate.

Each edge should track:

```text
strength
lag
persistence
decay
confidence
historical stability
```

Example:

```text
Oil price increase
  → diesel cost
  → freight cost
  → retail margins
  → consumer prices
  → CPI
  → rates
```

Transmission should not be assumed. It should be evidence-backed and replay-audited.

---

## 11. Propagation Speed

Different channels propagate at different speeds.

Fast:

```text
Fed decision
  → Treasuries
  → FX
  → bank stocks
  → credit conditions
```

Slow:

```text
New copper mine
  → future copper supply
  → grid cost
  → utility capex
  → power availability
```

The system should model propagation speed as part of the graph.

---

## 12. Context Persistence and Decay

Every context item has a life cycle.

```text
birth
active
decay
archived
```

Examples:

```text
Payroll report: active for weeks
New semiconductor fab: active for years
War/sanctions regime: active for months/years
Temporary company rumor: may decay within days
```

The model should not allow old context to stay active forever without refresh.

---

## 13. Node Importance

Not all nodes are equally important.

Each node should have:

```text
global importance
regional importance
sector importance
structural importance
market sensitivity
real-economy sensitivity
```

Example:

```text
US labor market
```

has much higher global macro importance than a single retailer’s minor guidance update.

But a minor update may still matter if it activates a larger graph, such as:

```text
retail inventory stress
  → freight
  → consumer demand
  → margins
  → inflation
```

---

## 14. Edge Reliability

Not all relationships are equally reliable.

Example:

```text
Oil → headline inflation
```

is historically stronger than:

```text
AI announcements → broad consumer spending
```

Each edge should have:

```text
weight
confidence
stability
regime dependency
historical evidence
```

---

## 15. Event Taxonomy

Events must be classified by type.

Example taxonomy:

```text
macro
policy
political
military
industrial
corporate
natural disaster
technology
financial
regulatory
trade
sanctions
logistics
supply-chain
credit
labor
commodity
```

This prevents all events from being treated as equivalent.

---

## 16. Shock Classification

Each event should be classified as:

```text
positive shock
negative shock
neutral update
structural shift
noise
uncertain
```

A news item can be predictable but low impact.

It can also be unpredictable but high impact.

Therefore the model must separate:

```text
predictability
impact
confidence
```

---

## 17. Probability Is Not Impact

This is a core rule.

Never collapse these into one score:

```text
probability
impact magnitude
confidence
market reaction
fundamental change
```

Examples:

```text
High probability + low impact:
  routine expected policy decision

Low probability + high impact:
  chokepoint closure

High market reaction + low fundamental change:
  crowded positioning unwind

Low market reaction + high fundamental change:
  early-stage industrial capacity shift
```

---

## 18. Scenario Competition

DEAN-OS should not forecast one future.

It should maintain several competing scenarios.

```text
Scenario A
Scenario B
Scenario C
Scenario D
```

Every news item updates all relevant scenarios.

A new event may increase one probability while decreasing another.

The model should track:

```text
scenario probability
impact
confidence
required evidence
contradicting evidence
time horizon
```

---

## 19. Evidence Graph

Each claim should be supported by evidence.

Evidence should include:

```text
source
source type
source quality
timestamp
primary/secondary status
supporting claim
contradictions
age
confidence
```

Evidence types:

```text
official data
company filing
company guidance
news report
market data
satellite/AIS
academic research
analyst interpretation
curated source
```

Do not treat all sources equally.

---

## 20. Evidence Competition

Evidence can conflict.

The system should not force premature resolution.

Maintain:

```text
supporting evidence
contradicting evidence
missing evidence
stale evidence
low-quality evidence
```

Probability should update gradually as evidence quality improves.

---

## 21. Unknown Graph

Unknowns are first-class objects.

The system should explicitly track:

```text
missing data
unverified reports
conflicting reports
unknown transmission strength
unknown time lag
unknown impact magnitude
need collector
need human review
```

Example:

```text
Hormuz risk
  → unknown real tanker traffic
  → need AIS data
```

Unknowns should feed the collector backlog.

---

## 22. Expectation Graph

Market reaction depends on expectations.

Track:

```text
actual outcome
consensus expectation
market-implied probability
positioning
crowdedness
surprise magnitude
```

A major event may have low market impact if already priced.

A small event may have large market impact if it contradicts consensus.

---

## 23. News Impact Probability Web

Each news item should be processed as a node in a web.

Pipeline:

```text
News
  ↓
World Context Graph
  ↓
Regional Context Graph
  ↓
Sector Graph
  ↓
Adjacent Sector Graph
  ↓
Dependency Graph
  ↓
Transmission Graph
  ↓
Probability Graph
  ↓
Impact Magnitude
  ↓
Replay / Audit
```

For every news item evaluate:

```text
event classification
activated context nodes
affected regions
affected sectors
adjacent sectors
transmission channels
predictability
impact strength
confidence
time-to-impact
duration
evidence
evidence gaps
scenario probability changes
realized outcomes over 1d/5d/20d/60d/120d
```

The system must learn from:

```text
strong impact
weak impact
delayed impact
second-order impact
no impact
false signals
```

---

## 24. Causal Memory

The system should store not only what happened, but the causal chain that led to it.

Example:

```text
Russia cuts gas
  ↓
gas prices rise
  ↓
ammonia production falls
  ↓
fertilizer prices rise
  ↓
food inflation rises
  ↓
central banks tighten
  ↓
housing slows
```

Example:

```text
AI demand rises
  ↓
HBM shortage
  ↓
memory prices rise
  ↓
data-center capex rises
  ↓
power demand rises
  ↓
grid investment rises
```

Historical cases should store causal graphs, not only event lists.

This allows DEAN-OS to search for similar mechanisms, not only similar events.

---

## 25. Replay First

Replay is mandatory.

After any scenario or event update, audit over fixed horizons:

```text
1 day
5 days
20 days
60 days
120 days
```

Track:

```text
what happened
what did not happen
which channels activated
which assumptions failed
which evidence gaps closed
probability calibration
Brier-style score
base-rate comparison
hit/miss of scenarios
```

No-lookahead discipline is required.

---

## 26. Collector Learning

Collectors should also be evaluated.

Track collector value:

```text
collector utility
collector precision
collector recall
collector freshness
collector reliability
collector contribution to scenario updates
```

If a collector never changes any scenario probability or evidence gap, it may be low value.

If a collector repeatedly provides early signals, increase its priority.

---

## 27. Learning Loop

DEAN-OS should learn through a closed loop.

```text
World
  ↓
Collectors
  ↓
Pipeline
  ↓
Agents
  ↓
World Model
  ↓
Scenarios
  ↓
Reality
  ↓
Replay
  ↓
Calibration
  ↓
Collector improvement
  ↓
Agent improvement
  ↓
World Model
```

The system’s value compounds through structured experience.

---

## 28. Knowledge Lake

Do not store all information as one undifferentiated database.

Use layers:

```text
Facts
  verified primary data

Observations
  news, company commentary, filings, events

Interpretations
  analyst views, research, hypotheses

World Model
  context graphs, state vectors, scenario probabilities

Replay
  realized outcomes and calibration history
```

Interpretations are hypothesis generators, not truth.

---

## 29. Collector Layer vs Agent Layer

Collectors should be deterministic.

Collectors do:

```text
fetch
parse
normalize
validate schema
timestamp
store
log errors
```

Collectors should not:

```text
invent meaning
make forecasts
silently fill unknown data
rank importance without explicit rules
```

Agents reason.

Agents do:

```text
discover indicators
classify events
map to graphs
estimate probability and impact
find historical analogs
identify evidence gaps
write analyst notes
trigger collector backlog
```

---

## 30. Daily Briefing Algorithm

A DEAN-OS daily briefing should follow this logic:

```text
1. Check Sector Universe
2. Update Context Grid
3. Update Indicator State Grid
4. Build World State Snapshot
5. Search Historical Analog Clusters
6. Activate Dependency/Transmission Graphs
7. Update Probability/Impact Graphs
8. Check Expectation Gap
9. Identify Evidence Gaps
10. Register Replay Horizons
11. Produce briefing as World Model delta
```

The briefing should answer:

```text
What changed in the main economic nodes?
In what context?
Which sectors and adjacent sectors are affected?
Which dependency graphs activated?
Which scenarios changed probability?
Where is impact strong, weak, delayed, or absent?
What evidence is missing?
What should be checked later?
```

---

## 31. World Model Journal

Every daily briefing should create a World Model Journal entry.

Journal fields:

```text
date
world state summary
major context changes
indicator changes
activated graphs
scenario probability changes
evidence gaps
unknowns
historical analogs
replay tasks
analyst self-check notes
```

The journal is the memory of DEAN-OS.

---

## 32. Design Boundary

DEAN-OS should not produce unsupported recommendations.

No:

```text
trading orders
position sizing
price targets
unsupported forecasts
hallucinated data
single-point predictions without uncertainty
```

Allowed:

```text
context updates
scenario probability ranges
impact analysis
evidence gaps
historical analogs
replay tasks
risk/uncertainty notes
```

---

## 33. Final Architecture Summary

DEAN-OS is a probabilistic world-modeling system.

Core pipeline:

```text
Sector Universe
  ↓
Context Grid
  ↓
Indicator State Grid
  ↓
World State Snapshot
  ↓
Historical Analog Cluster
  ↓
Dependency Graph
  ↓
Transmission Graph
  ↓
Probability / Impact Graph
  ↓
Expectation Graph
  ↓
Evidence / Unknown Graph
  ↓
Replay / Calibration
```

This is the central architecture.

Everything else — collectors, agents, lenses, orchestrators, journals, daily briefings — should serve this world-modeling loop.

---

## 34. Regime / Cycle / Overheating Layer

Every daily analysis must first identify the background regime in which events occur.

No event happens in isolation.  
A single news item must be interpreted against the broader cycle, valuation regime, liquidity/credit environment, geopolitical background, and sector-specific mini-cycle.

### 34.1 Daily regime classification

For each day, DEAN-OS should classify the current background across the following dimensions:

```text
Economic phase:
  recession risk
  stagnation
  fragile recovery
  expansion
  overheating

Market phase:
  risk-on
  risk-off
  crowded theme
  bubble-risk
  valuation reset

Credit phase:
  loose
  neutral
  tight
  stressed

Inflation phase:
  disinflation
  sticky inflation
  energy-led inflation
  wage-led inflation
  food-led inflation

AI cycle:
  early adoption
  enterprise adoption
  capex boom
  bubble-risk
  correction
  productivity proof

Geopolitical phase:
  peace
  hybrid conflict
  localized war
  regional escalation risk
  de-escalation
```

### 34.2 Why this matters

A news item such as:

```text
Samsung and SK Hynix commit large capex because they believe in AI memory growth
```

should not be analyzed alone.

It must be analyzed as:

```text
AI capex boom
+ memory / HBM bottleneck
+ high valuations
+ crowded positioning
+ possible AI bubble-risk
+ Samsung / SK Hynix capex commitments
→ good fundamental signal
→ but possible expectation gap
→ and possible future overcapacity risk
```

The same fact can be both bullish and risky depending on the background regime.

Large capex commitments can mean:

```text
demand durability
+ management confidence
+ structural bottleneck confirmation
```

but also:

```text
future supply glut
+ return-on-invested-capital risk
+ long payback period
+ valuation disappointment if profits lag
```

### 34.3 AI bubble / post-bubble logic

The AI cycle may follow a classic innovation pattern:

```text
large expectations
→ capital flood
→ overinvestment
→ disappointment / bubble burst
→ weak players fail
→ infrastructure and productive use cases mature
→ surviving leaders compound value
```

This does not mean AI is a bad technology.

It means market expectations, valuations, and profit timing can move far ahead of real adoption and cash flow.

A bubble burst can destroy many companies while the underlying technology still becomes fundamental.

Historical analogy:

```text
dot-com bubble
→ many internet companies failed
→ internet still became foundational infrastructure
```

Therefore DEAN-OS should distinguish:

```text
technology validity
from
valuation excess
from
near-term profitability
from
long-term productivity impact
```

### 34.4 Post-Bubble Survivor Analysis Lens

After a bubble/crowded-theme correction, DEAN-OS should not simply assume that every fallen asset is attractive.

It should evaluate survivors through a specific lens:

```text
Did the technology thesis survive?
Did the business model survive?
Is demand real, or was it mostly hype?
Does the company have cash flow?
Does it control a bottleneck?
Does it have infrastructure, data, distribution, or ecosystem power?
Can it survive 2–3 years of funding pressure?
Does it benefit from consolidation after weaker players fail?
Did valuation normalize?
Is there evidence of real productivity/adoption?
Has sentiment moved from greed to despair?
```

This lens is especially relevant for AI companies, cloud providers, semiconductor suppliers, data-center infrastructure, memory producers, power/grid assets, and enterprise automation platforms.

### 34.5 Scandal Drawdown vs Bubble Burst

DEAN-OS should separate two different types of drawdown.

#### Idiosyncratic scandal drawdown

Example pattern:

```text
large platform company scandal
→ reputational/regulatory pressure
→ temporary valuation compression
→ core business remains intact
→ recovery possible if cash flow and ecosystem remain strong
```

This can happen with structurally embedded companies.

However, the correct framing is not:

```text
too big to fail
```

A better framing is:

```text
too structurally embedded to disappear quickly
```

#### Full bubble burst

Example pattern:

```text
AI valuation bubble
→ multiple compression
→ capex expectations fall
→ funding dries up
→ weak companies fail
→ strong platforms/infrastructure survive
→ new productivity cycle begins later
```

The full-bubble case is more dangerous because even strong companies can take years to recover to prior valuation peaks.

### 34.6 Hype vs quiet productive automation

The public AI hype is often concentrated in visible demos:

```text
chatbots
image generation
coding assistants
agents
humanoid robots
voice assistants
```

But the more durable productivity impact may come from quieter automation:

```text
industrial process control
chemical process optimization
food production automation
quality control
logistics routing
warehouse operations
medical imaging
clinical workflow support
enterprise back-office automation
data extraction
compliance monitoring
predictive maintenance
energy optimization
```

These applications may be less visible and less hyped, but they can be closer to real productivity improvement.

DEAN-OS should therefore distinguish:

```text
attention/hype layer
from
revenue layer
from
productivity layer
from
infrastructure layer
```

### 34.7 Robotics and physical automation realism

Humanoid or highly dynamic robots may attract attention, but much of near-term economic value may come from narrower, less spectacular automation.

Examples:

```text
fixed robotic arms
machine vision inspection
warehouse sorting
picking and packing
process monitoring
food line automation
chemical plant optimization
medical imaging assistance
industrial safety monitoring
```

A robot that walks, jumps, or performs a public demo is not automatically economically superior to a boring system that reliably improves throughput or reduces downtime.

DEAN-OS should track:

```text
demo value
vs
production value
vs
unit economics
vs
deployment friction
vs
safety/regulatory burden
```

### 34.8 Political / Leadership Behavior Lens

Events such as Hormuz escalation, sanctions, war decisions, tariffs, and export controls depend not only on economics but also on political decision systems.

DEAN-OS should analyze regimes and leaders retrospectively through observed behavior, not speculation.

For each political actor or regime, track:

```text
past action
context
constraint
stated goal
actual outcome
repeated pattern
reaction function
confidence
contradicting evidence
```

Examples of decision-structure analysis:

```text
Iran:
  supreme leader
  IRGC / security apparatus
  clerical establishment
  economic interests
  regime survival logic
  external deterrence
  street pressure

US executive behavior:
  media-cycle sensitivity
  negotiation-through-escalation pattern
  market sensitivity
  symbolic dominance
  institutional constraints

Russia:
  escalation to test red lines
  ambiguity
  willingness to absorb economic pain
  regime-survival priority

Ukraine:
  coalition management
  Western support dependency
  military necessity
  domestic legitimacy
  resilience signaling
```

This lens must remain evidence-based.

Do not create psychological fiction.  
Track repeated behavior patterns and validate them through replay.

### 34.9 Event analysis with regime background

Example:

```text
Current background:
  regional war risk
  fragile Hormuz flows
  sticky inflation
  high AI valuations
  crowded AI positioning
  AI capex boom
  memory bottleneck
  power/grid constraint

Event:
  commercial vessel damaged near Hormuz

Analysis questions:
  Does this confirm escalation?
  Is it controlled signaling?
  Is it deniable pressure?
  Is it bargaining behavior?
  Does it raise insurance/freight/oil risk?
  Does it change inflation/rates probabilities?
  Does it hit AI/data-center power indirectly through energy markets?
```

Correct analysis is:

```text
event
+ regime/cycle background
+ political behavior lens
+ transmission graph
+ expectation gap
+ scenario competition
```

not:

```text
headline
→ simple market prediction
```

### 34.10 Module implications

Potential modules/lenses:

```text
RegimeCycleClassifier
BubbleCrowdingDetector
PostBubbleSurvivorLens
ScandalDrawdownLens
StrategicCapexCommitmentLens
PoliticalLeadershipBehaviorLens
RegimeReactionFunctionLibrary
QuietAutomationProductivityLens
RoboticsProductionValueLens
```

These modules should update:

```text
World State Vector
Scenario Probability Graph
Expectation Gap
Evidence Graph
Replay Queue
```

