# Automation Maturity Ladder

1. **Level 1 — Data accumulation**  
   Automated collection, source hashes, dedupe, normalization, storage, data-quality log.

2. **Level 2 — Automated analysis / review-only**  
   Event packets, analyst routing, risk archetypes, expectation gap, hypothesis tokens, evidence gaps, digest.

3. **Level 3 — Automated eval/audit**  
   Grounding, retrieval eval, leakage checks, unit traps, decision lineage, failed-run alerts.

4. **Level 4 — Replay**  
   Historical as-of replay with no future leakage.

5. **Level 5 — Paper trading**  
   Live data, simulated broker/virtual portfolio only.

6. **Level 6 — Shadow live**  
   Hypothetical live orders, broker send blocked.

7. **Level 7 — Supervised live**  
   Small limits, allowed assets, human approval/emergency stop.

8. **Level 8 — Constrained autonomous**  
   Execution only through gateway, risk engine, kill switch, hard limits, full decision lineage.
