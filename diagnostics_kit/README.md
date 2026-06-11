# Component Engagement + Value Audit v4

This kit is specifically for your question:

> Is a useful enricher/calculator/analyzer/algorithm/context map actually engaged, correctly executed, and does its output reach the model/evaluation?

It is not a Sonar/Bandit replacement. It focuses on component usefulness.

## Main scripts

```bash
python diagnostics/component_engagement_audit.py --root src --configs configs . --out diagnostic_reports
python diagnostics/component_harness_runner.py --root src --out diagnostic_reports
python diagnostics/component_value_report.py --reports diagnostic_reports
```

Or:

```bash
python diagnostics/run_component_value_audit.py
```

## Outputs

```text
diagnostic_reports/
  component_engagement.csv
  component_engagement_summary.md
  component_import_edges.csv
  component_harness_results.csv
  component_harness_summary.json
  component_value_report.csv
  component_value_report.md
```

## What it checks

### Engagement

For each component:
- exists in code
- category: enricher/calculator/analyzer/algorithm/context_map/validator/etc.
- imported by other source files
- referenced by strings/config
- likely reachable from entrypoints
- has test reference
- has expected methods such as enrich/calculate/analyze/detect/select/validate

### Isolated execution

`component_harness_runner.py` tries to:
- import component class
- instantiate with no args or config={}
- run enrich/calculate/analyze/detect/select/validate/transform/run on tiny two-ticker dataframe
- capture added/removed/modified columns
- warn about target_* columns, row count changes, high NaN, non-dataframe output

Many components will be skipped because they need real dependencies/configs. That is useful information.

### Feature lineage

`feature_lineage_tracker.py` is a library to add into your pipeline checkpoints:

```python
tracker.capture_step("raw", df)
out = enricher.enrich(df)
tracker.capture_component_output("TechnicalAnalysisEnricher", before=df, after=out)
tracker.capture_step("after_feature_selection", selected_df)
tracker.mark_model_input(X_train)
tracker.save()
```

This answers:
- which component added which columns
- which columns reached model input
- which columns were dropped
- why output did not reach the model

### Value report

`component_value_report.py` merges:
- engagement report
- harness report
- lineage report if available
- ablation results if available

Statuses include:
- NEEDS_FIX_BEFORE_VALUE_TEST
- EXECUTED_BUT_LEAKAGE_RISK
- OUTPUT_REACHES_MODEL
- OUTPUT_DROPPED_OR_NOT_MARKED
- EXECUTED_OUTPUT_UNKNOWN_LINEAGE
- ACTIVE_VALUE_UNKNOWN
- UNUSED_POTENTIALLY_VALUABLE

## Ablation

`ablation_experiment_runner.py` is an adapter skeleton. You must connect it to your offline minimal pipeline.

Goal:
- baseline
- with component
- without component
- metric delta
- regime delta later

## Recommended workflow

1. Run `python diagnostics/run_component_value_audit.py`.
2. Review `ACTIVE_RISKY` and `ACTIVE_OUTPUT_UNTESTED`.
3. Add `FeatureLineageTracker` checkpoints to feature pipeline.
4. Run prepare/continue on tiny offline dataset.
5. Generate `feature_lineage_report.json`.
6. Run `component_value_report.py`.
7. Only then decide:
   - keep
   - fix
   - integrate
   - experiment
   - quarantine
   - delete later
