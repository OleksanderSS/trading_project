# Data Provenance / Reproducibility Architecture

Purpose: ensure DEAN-OS can reconstruct what the system knew at a given time.

This is mandatory for:

- historical replay;
- paper/shadow comparison;
- event outcome review;
- time leakage prevention;
- model promotion;
- postmortem analysis;
- regulatory-style audit.

## Core question

```text
At timestamp T, what sources, features, models, hypotheses, and configs were actually known to the system?
```

## Required manifests

- source manifest;
- normalized packet manifest;
- feature snapshot manifest;
- model state manifest;
- hypothesis ledger snapshot;
- run manifest;
- decision lineage;
- config manifest;
- output manifest.

## Reproducibility rule

No replay, model promotion, or later execution can be trusted without versioned input state.
