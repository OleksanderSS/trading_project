# Codex Integration Prompt — Regime Context + Scenario Outcome Graph

Use the files in this package as a design supplement for DEAN-OS.

Goal:
Implement a review-only analyst layer that combines:
- date-specific regime context;
- event classification;
- expectation-gap analysis;
- transmission-channel mapping;
- scenario outcome graphs;
- historical event-regime-outcome graph retrieval;
- fixed-horizon outcome tracking;
- self-check and calibration.

Strict boundaries:
- no live trading;
- no buy/sell/hold output;
- no position sizing;
- no broker routing;
- no production price targets;
- no autonomous execution.

Implementation priority:
1. Define schemas for regime_context_vector and scenario_outcome_graph.
2. Add validators:
   - allowed taxonomy values;
   - probability mass sums to 1;
   - graph is acyclic per as_of packet;
   - as_of_date exists;
   - no-lookahead guard;
   - explicit evidence gaps.
3. Build review-only report output:
   - daily regime snapshot;
   - news vs regime;
   - scenario graph update;
   - horizons to track;
   - self-check questions.
4. Add tests before integration with any runtime.
