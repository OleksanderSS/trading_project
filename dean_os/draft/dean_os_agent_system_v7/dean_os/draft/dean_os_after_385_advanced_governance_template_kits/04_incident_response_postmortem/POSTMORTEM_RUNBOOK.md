# Postmortem Runbook

1. Freeze relevant run manifests.
2. Collect source, feature, model, hypothesis, and decision lineage.
3. Classify failure root cause.
4. Identify whether failure was data, analyst, pipeline, orchestrator, risk, or execution.
5. Add regression test.
6. Update source registry / prompt / pattern / eval / gate as needed.
7. Close only after regression passes.

No incident should be resolved only by narrative explanation.
A test or gate must be added when possible.
