# Component Audit System — How To Use

## Run full audit
```bash
# Step 1: Engagement scan (fast, ~30s)
python diagnostics/component_engagement_audit.py --root src --out diagnostic_reports

# Step 2: Harness test (with timeout, ~10min for 80 components)
python diagnostics/component_harness_runner.py --root src --out diagnostic_reports --timeout 10 --max-components 80

# Step 3: Run pipeline to generate lineage report
python run_hybrid_pipeline.py --mode prepare --epochs 1 --max-iterations 1

# Step 4: Regenerate value report with all data
python diagnostics/component_value_report.py --reports diagnostic_reports

# Step 5: Read summary
python generate_audit_report.py
```

## Output files
| File | Contents |
|------|---------|
| `component_engagement.csv` | All 1024 components with engagement status |
| `component_engagement_summary.md` | Status counts by category |
| `component_harness_results.csv` | 80 components tested in isolation |
| `component_harness_summary.json` | EXECUTED/FAILED/TIMEOUT counts |
| `feature_lineage_report.json` | Which enricher output reaches model input |
| `component_value_report.csv` | Final merged report with value status |
| `component_value_report.md` | Summary counts |
| `domain_rule_findings.csv` | Code quality rule violations |

## Status meanings

### Value status
- `EXECUTED_OUTPUT_UNKNOWN_LINEAGE` — ran OK, adds columns, but lineage not confirmed
- `OUTPUT_REACHES_MODEL` — confirmed: enricher output reaches X_train  ← best status
- `OUTPUT_DROPPED_OR_NOT_MARKED` — adds columns but they disappear before model
- `NEEDS_FIX_BEFORE_VALUE_TEST` — has risk findings (TARGET_LEAK, FUTURE_SHIFT, etc.)
- `UNUSED_POTENTIALLY_VALUABLE` — not engaged, but category suggests useful logic
- `ACTIVE_VALUE_UNKNOWN` — engaged but no harness/lineage data yet

### Harness status
- `EXECUTED` — ran on sample df, output captured ✅
- `EXECUTION_FAILED` — ran but failed (usually needs real data like portfolio_returns)
- `TIMEOUT` — initialization takes >10s (usually imports DuckDB/TF on load)
- `INSTANTIATE_SKIPPED` — __init__ requires mandatory config
- `NO_KNOWN_METHOD` — class has no enrich/analyze/calculate/etc method

## Key findings (current run)

### 17 EXECUTED enrichers/calculators
Clean, instantiate with no args, add expected columns:
- **VolatilityEnricher** → 7 cols (atr_14, gk_volatility, returns, volatility_*)
- **VolumeEnricher** → 6 cols (obv, price_volume_trend, volume_roc, volume_rs, *)
- **MarketContextEnricher** → 18 cols
- **ContextMapEnricher** → 11 cols (context_fingerprint, context_anxiety_index, *)
- **SignificanceFeaturesEnricher** → 5 cols
- **DerivedFeaturesEnricher** → 16 cols (LAG_*, ACCELERATION_*, VELOCITY_*)
- **DecayFeaturesEnricher** → 1 col (is_significant_decayed)
- **CriticalSignalDetector** → 3 cols (price_shock_detected, volume_spike_detected, *)
- **TimeFeaturesEnricher**, **HypeEnricher**, **NewsQualityEnricher** → run OK, context-dependent output

### 4 EXECUTION_FAILED (need domain data)
- **PerformanceAttributionAnalyzer** — needs portfolio_returns
- **RiskDecompositionAnalyzer** — needs portfolio_returns  
- **NewsImpactAnalyzer** — needs news text column
- **ModelComparisonAnalyzer** — needs results DataFrame

### 42 TIMEOUT
Most timeout due to DuckDB/config initialization at import time.
DriftAnalyzer, AdvancedAnalyticsEnricher, etc. are technically fine but
slow to initialize. Use in pipeline directly rather than harness.

## Next steps
1. **Add golden tests** for 11 clean enrichers (no warnings, add expected columns)
2. **Wire PerformanceAttributionAnalyzer** into Stage 7 analytics with proper data
3. **Add portfolio_returns** to harness sample data to test portfolio analyzers
4. **Investigate TIMEOUT** components — if they're slow at import, consider lazy imports
