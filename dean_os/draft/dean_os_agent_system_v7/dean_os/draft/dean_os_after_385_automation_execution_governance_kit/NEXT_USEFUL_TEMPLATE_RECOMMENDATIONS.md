# Next Useful Template Recommendations

Recommended after this kit, in priority order.

## 1. Existing Pipeline Integration Map Kit

Purpose: more detailed module-by-module integration map from `src(6).zip` to the five/six Codex kits.

Files:
- MODULE_TO_KIT_MAPPING.md
- EXISTING_MODULE_CAPABILITY_INVENTORY.md
- GAP_ANALYSIS_BY_AUTOMATION_PHASE.md
- CODEX_INTEGRATION_SEQUENCE.md

Why: Codex will know exactly where to attach each concept.

## 2. Data Provenance / Reproducibility Kit

Purpose: know what the system knew as of a date.

Files:
- AS_OF_DATA_MANIFEST_SCHEMA.json
- SOURCE_VERSIONING_TEMPLATE.yaml
- FEATURE_SNAPSHOT_MANIFEST_SCHEMA.json
- MODEL_STATE_MANIFEST_SCHEMA.json
- REPLAY_REPRODUCIBILITY_CHECKLIST.md

Why: mandatory for replay, backtest, audit, and no-future-leakage.

## 3. Analyst Debate / Devil's Advocate Kit

Purpose: avoid first-hypothesis lock-in.

Files:
- BULL_CASE_ANALYST_TEMPLATE.md
- BEAR_CASE_ANALYST_TEMPLATE.md
- NEUTRAL_REVIEWER_TEMPLATE.md
- COUNTERARGUMENT_SCHEMA.json
- CONSENSUS_REVIEW_PACKET_TEMPLATE.md

Why: useful for major macro/policy hypotheses.

## 4. Capital Allocation / Portfolio Governance Kit

Purpose: later layer between signal quality and portfolio action.

Files:
- STRATEGY_CAPITAL_BUCKET_TEMPLATE.yaml
- EXPOSURE_BUDGET_TEMPLATE.yaml
- CORRELATION_CLUSTER_LIMITS.yaml
- DRAWDOWN_RESPONSE_POLICY.md

Why: needed before live capital, but after paper/shadow gates.

## 5. Incident Response / Postmortem Kit

Purpose: when system fails, learn safely.

Files:
- INCIDENT_REPORT_SCHEMA.json
- TRADE_OR_DECISION_POSTMORTEM_TEMPLATE.md
- ROOT_CAUSE_LABELS.json
- REGRESSION_TEST_CREATION_POLICY.md

Why: prevents repeating failures and improves audit quality.
