# DEAN-OS After 385 — Evaluation / Observability / Audit Kit

This is a non-numbered practical kit created after the closed after-385 staged bundle.

Purpose: give Codex reusable templates for checking whether DEAN-OS agents, retrieval, domain-learning
flows, news interpretation, causal patterns, and daily data accumulation runs are behaving correctly.

This kit is not production code and not a blind-merge patch. Codex should harvest/adapt useful pieces.

## What this kit covers

- source-grounding evaluation;
- retrieval quality evaluation;
- numeric/unit/period trap tests;
- time-leakage tests;
- causal-pattern false positive tests;
- analyst output quality metrics;
- daily run audit logs;
- observability dashboard sketch;
- alerting rules;
- human-review feedback labels;
- regression test runbook.

## Core rule

A good analyst system must be measured. It is not enough for an agent to produce plausible text.

The system should continuously ask:

- Did the agent cite the right source?
- Did it use the correct period and unit?
- Did it confuse capacity, production, shipments, and demand?
- Did it use future data in a past-as-of-date question?
- Did it overclaim from weak news?
- Did it treat sentiment as a signal?
- Did it miss counterforces?
- Did daily ingestion run correctly?
- Did retrieval quality degrade?
- Did the evidence graph preserve lineage?
- Did review feedback reduce repeated errors?

## Safe boundary

This kit supports review-only evaluation and observability.

Forbidden:
- trading signals;
- buy/sell/hold;
- price targets;
- broker/order calls;
- autonomous portfolio changes;
- production config mutation without review;
- model promotion without review.
