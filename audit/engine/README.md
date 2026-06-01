# audit/engine

Use `full_audit_workflow.py` for normal operation:

```bash
python audit/engine/full_audit_workflow.py --root src --mode scan
python audit/engine/full_audit_workflow.py --root src --mode baseline
python audit/engine/full_audit_workflow.py --root src --mode check --fail-on P0,P1
```

Reports are written to `audit_reports/`.

The scanner is offline and dependency-light. It is intentionally conservative: P0/P1 findings should be manually triaged, fixed, or suppressed with a reason and expiry.

## Baseline & Suppressions

- Baseline file (commit this): `audit/engine/audit_baseline.json`
- Suppressions file (commit this): `audit/engine/audit_suppressions.yaml`
- Reports directory (generated): `audit_reports/`

Typical flow:

1) `--mode scan` to review `audit_reports/triage_P0_P1.md`
2) Fix issues, or add targeted suppressions with `expires`
3) `--mode baseline` to update the committed baseline
4) `--mode check` in CI to fail on new/unsuppressed P0/P1
