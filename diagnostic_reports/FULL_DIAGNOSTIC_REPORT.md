# Full Diagnostic Report


---


# Module Diagnostic Summary

- **python_files**: 723
- **classes**: 794
- **functions**: 5298
- **risk_findings**: 426
- **orphans_static**: 316

---


# Domain Rule Findings

- feature_module_must_not_emit_target_columns: 27
- future_shift_requires_groupby_ticker: 2
- risk_metrics_no_fillna_zero_returns: 25

---


# Registry Consistency Report

- Factory files: 10
- Registry mentions: 11
- Lazy mentions: 1
- Duplicate model refs: 14

---


# Config Reachability Summary

- **config_files_scanned**: 813
- **references_found**: 1608
- **possibly_missing**: 513

Review `config_reachability.csv`. Missing references are not automatically bugs; dynamic loading may require manual confirmation.

---


# Dead Code Classification

- **ACTIVE_BUGGY_OR_RISKY**: 45
- **ACTIVE_UNTESTED_OR_OK**: 288
- **UNUSED_DELETE_CANDIDATE**: 291
- **UNUSED_LEGACY**: 1
- **UNUSED_RISKY_QUARANTINE**: 52
- **UNUSED_USEFUL_CANDIDATE**: 46

Do not delete modules based only on static classification. Check config and runtime reports first.

---

## Files to review

- `risk_findings.csv`
- `domain_rule_findings.csv`
- `module_inventory.csv`
- `orphan_modules.txt`
- `registry_consistency_report.json`
- `config_reachability.csv`
- `dead_code_classification.csv`