# Minimal First Implementation Plan

This is the shortest practical path that gives value without creating execution risk.

## Step 1 — Run manifest and source manifest

Implement or adapt:

- `run_id`;
- source manifest;
- collector status;
- source hashes;
- publication/event/ingestion date fields;
- basic dedupe status.

Success condition:

```text
Every daily run can be reconstructed.
```

## Step 2 — Pipeline Controller daily governor

Implement daily governor around existing collectors.

The controller should:

- start enabled collectors;
- enforce timeouts;
- collect status;
- block invalid downstream stages;
- write daily audit;
- create review queue items.

Success condition:

```text
Collectors are not just scripts; they are governed daily processes.
```

## Step 3 — Normalized event packet

Convert collector outputs into event packets.

Required fields:

- event ID;
- source IDs;
- event type;
- publication/event/ingestion dates;
- affected entities/sectors;
- source quality;
- dedupe cluster;
- direct/indirect/contextual flag.

Success condition:

```text
News becomes structured input for analysts.
```

## Step 4 — Tag-only analyst output

Analyst should output:

- risk archetype;
- macro regime context;
- expectation gap candidate;
- evidence gaps;
- affected/baseline sectors;
- confidence label.

Analyst must not output:

- final probability without source;
- buy/sell/hold;
- price target;
- order.

Success condition:

```text
LLM interprets context, not trading math.
```

## Step 5 — Review report and feedback labels

Create daily operator report and review labels.

Success condition:

```text
User corrections become structured learning candidates.
```

## Step 6 — Outcome review skeleton

For hypothesis tokens, track:

- expected path;
- actual metrics;
- what was right/wrong;
- learning candidate.

Success condition:

```text
The system starts learning from event outcomes.
```

## Not yet

- live execution;
- broker integration;
- autonomous trading;
- unconstrained strategy promotion.
