# Codex Synchronous Review Harvest Notes

Use this kit to implement operator review as a learning surface.

High-priority harvest targets:

1. Review session state schema
2. Human-agent parallel analysis schema
3. Review label taxonomy
4. Feedback-to-learning pipeline
5. Review queue prioritization
6. Report-to-training example schema
7. Operator daily review report template
8. Operator console workflow

Key integration idea:

```text
Agent report is not the end. It is input to synchronous review.
Synchronous review produces corrected records and learning/eval candidates.
```

Suggested repo areas:

- dean_os/review/
- dean_os/operator_console/
- dean_os/learning_feedback/
- dean_os/eval/
- dean_os/audit/
- docs/dean_os/
- tests/fixtures/review/
