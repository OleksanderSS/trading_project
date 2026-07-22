# Event Graph Learning Closure

Purpose: formalize the user's core idea of teaching agents from events, causal graphs, scenario branches,
evidence gaps, and outcome review.

The learning loop:

```text
event
→ normalized event packet
→ risk archetype tags
→ macro regime snapshot
→ historical analog candidates
→ expectation gap
→ causal graph
→ scenario tree
→ watch metrics
→ outcome review
→ error attribution
→ pattern update
→ eval/regression update
→ future analyst behavior improves
```

## What the agent learns

The agent should not merely memorize events. It should learn:

- which mechanisms were active;
- which analogs were misleading;
- which counterforces mattered;
- which evidence gaps predicted uncertainty;
- which scenario branch materialized;
- which outcome metrics were useful;
- whether the market had already priced the event;
- how confidence should change in similar future cases.

## What the agent must not learn

- fake precision;
- direct trading rules from one event;
- overfitted sector conclusions;
- LLM-generated probabilities as truth;
- deterministic historical analogy.
