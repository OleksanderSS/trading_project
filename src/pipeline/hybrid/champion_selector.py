"""Empirical champion selection -- picks the best-performing model_type for
every (ticker, target) from a Colab training run's real metrics.

This is deliberately separate from src/models/model_selector/
model_competence_map.json, which is a static, hand-assigned prior ("lstm is
generally ~0.70 competent for classification") used for runtime regime-based
selection (ModelSelectionService.select_best_model_for_context) when no
empirical result exists yet. This module answers a different question --
"which model actually performed best on this specific (ticker, target) in
this training run" -- from real metrics, not priors.

`filter_to_champions` is wired into ResultsProcessor.build_models_metadata()
as a hard filter: only the champion model_type per (ticker, target) survives
into Stage 5's models_metadata, so prediction never runs an inferior
architecture just because it happened to also get trained. A group with no
comparable metric (every candidate was skipped/errored) is dropped entirely
rather than keeping an arbitrary, unverified model.

scripts/colab/select_champions.py is the standalone CLI wrapper around
`select_champions` for offline inspection of a colab_results.json without
running the pipeline.
"""
from __future__ import annotations

from typing import Any

from src.config.target_type_registry import CLASSIFICATION_BINARY_TYPE, CLASSIFICATION_TARGET_TYPES

NO_REAL_METRIC_KEYS = ({"info"}, {"error"})


def _has_real_metric(metrics: dict[str, Any]) -> bool:
    return bool(metrics) and set(metrics.keys()) not in NO_REAL_METRIC_KEYS


def _score(metrics: dict[str, Any], target_type: str) -> tuple[float, str] | None:
    """Return (comparable_score, score_name) where higher score = better,
    or None if this entry has no metric usable for comparison.

    Keras trainers (cnn/lstm/gru/transformer) report val_accuracy/val_auc/
    val_loss; sklearn/TabNet trainers (mlp/tabnet) report the unprefixed
    accuracy/auc/mse -- both naming conventions must be recognized or
    every Keras-architecture entry would silently drop out of comparison.
    """
    if not _has_real_metric(metrics):
        return None

    # `score` carries the metric the GATE judged this model by -- balanced
    # accuracy for classification, R2 for regression -- already oriented so
    # that higher is better. Prefer it, because two selections deciding the
    # same thing by two metrics is how the original defect worked.
    #
    # `base_trainer` line 448 records the first half of that defect and its
    # fix: "The arena must select on the SAME metric the gate judges by. It
    # did not: selection took F1, so the winner of every classification
    # context was whichever model said 'yes' most confidently." That was
    # repaired inside the arena. This function is a SECOND selection, run
    # afterwards over the arena's winners, and it went on ranking by
    # `accuracy` -- the metric measured to hand 0.7381 to a predictor that
    # never fires while the model itself scored 0.5257 balanced (REGISTER
    # #187). Fixing one selector and leaving the other is the same defect
    # with a smaller blast radius.
    if "score" in metrics and metrics["score"] is not None:
        return float(metrics["score"]), "score"

    if target_type in CLASSIFICATION_TARGET_TYPES:
        if target_type == CLASSIFICATION_BINARY_TYPE:
            for key in ("auc", "val_auc"):
                if key in metrics:
                    return float(metrics[key]), "auc"
        for key in ("accuracy", "val_accuracy"):
            if key in metrics:
                return float(metrics[key]), "accuracy"
        return None
    if "mse" in metrics:
        return -float(metrics["mse"]), "mse"  # negate: higher score == lower mse
    return None


def select_champions(models_metadata: dict[str, Any], target_types: dict[str, str]) -> dict[str, Any]:
    """Group models_metadata entries by (ticker, timeframe, target) and pick
    the highest-scoring model_type in each group.

    The timeframe belongs in that key and was missing until 2026-09-01. The
    intent was to choose the best ARCHITECTURE for a (ticker, target); without
    the timeframe it also chose the best CADENCE, by comparing scores that are
    not comparable. Measured on run 7: `Built metadata for 9 models` became
    `Filtered to 7 champion model(s)`, and the two that vanished were exactly
    the pair of names that repeat across frames --
    `target_hourly_breakout_1h` and `target_volatility_spike_1h`, each present
    on 15m and on 60m. Both had passed the gate as separate contexts, with
    separate verdicts and separate files, and both carry the timeframe in
    their own context id. The log said 9 and then 7; nothing said that two had
    been discarded.

    Why it is not a detail: our own measurement (CLAIMS.md R6) says a coarser
    cadence scores higher because the clock opponent weakens, not because the
    model improves. So the missing key systematically preferred the coarser
    frame -- and R8 then showed that the coarser intraday frame is the one with
    barely two years of history behind it.

    Returns {"{ticker}::{target}": {champion payload}}, where the payload's
    `source_key` is the winning entry's *actual* key in `models_metadata`
    (not a reconstructed "{ticker}_{target}_{model_type}" string) -- callers
    that need to look the winner back up in the original dict, like
    `filter_to_champions`, must never assume any particular key-naming
    convention on the caller's data.

    A group where nothing is comparable comes back with status
    "no_champion" rather than a fabricated winner or a silent skip.
    """
    groups: dict[tuple[str, str, str], list[tuple[str, dict[str, Any]]]] = {}
    for source_key, entry in models_metadata.items():
        ticker = entry.get("ticker")
        target = entry.get("target")
        if not ticker or not target:
            continue
        # An entry with no timeframe gets its own group rather than joining
        # one: an unknown cadence is not the same cadence as another unknown.
        timeframe = str(entry.get("timeframe") or "unknown")
        groups.setdefault((ticker, timeframe, target), []).append((source_key, entry))

    champions: dict[str, Any] = {}
    for (ticker, timeframe, target), entries in groups.items():
        target_type = target_types.get(target, "regression")
        scored = []
        for source_key, entry in entries:
            result = _score(entry.get("metrics", {}), target_type)
            if result is not None:
                score, score_name = result
                scored.append((score, score_name, source_key, entry))

        key = f"{ticker}::{timeframe}::{target}"
        if not scored:
            champions[key] = {
                "ticker": ticker,
                "timeframe": timeframe,
                "target": target,
                "target_type": target_type,
                "status": "no_champion",
                "reason": f"none of {len(entries)} model(s) had a comparable metric",
                "candidates_considered": len(entries),
            }
            continue

        scored.sort(key=lambda item: item[0], reverse=True)
        best_score, best_score_name, best_source_key, best_entry = scored[0]
        champions[key] = {
            "ticker": ticker,
            "timeframe": timeframe,
            "target": target,
            "target_type": target_type,
            "status": "champion_selected",
            "champion_model_type": best_entry.get("model_type"),
            "champion_metrics": best_entry.get("metrics"),
            "selection_score": best_score if best_score_name != "mse" else -best_score,
            "selection_metric": best_score_name,
            "model_path": best_entry.get("model_path"),
            "source_key": best_source_key,
            "candidates_considered": len(entries),
            "candidates_comparable": len(scored),
            "ranking": [
                {
                    "model_type": entry.get("model_type"),
                    "score": score if score_name != "mse" else -score,
                    "metric": score_name,
                    "metrics": entry.get("metrics"),
                }
                for score, score_name, _source_key, entry in scored
            ],
        }
    return champions


def filter_to_champions(models_metadata: dict[str, Any], target_types: dict[str, str]) -> dict[str, Any]:
    """Filter a flat models_metadata dict (one entry per ticker/target/
    model_type) down to just the champion model_type per (ticker, target).

    Uses each champion's actual `source_key` (its real key in
    `models_metadata`) rather than reconstructing one from
    ticker/target/model_type -- correct regardless of what key-naming
    convention the caller's models_metadata happens to use.

    A (ticker, target) group with no comparable metric is dropped entirely
    -- Stage 5 simply has no model for that combination rather than
    receiving an arbitrarily-chosen, unverified one.
    """
    champions = select_champions(models_metadata, target_types)
    keep_keys = {
        result["source_key"]
        for result in champions.values()
        if result["status"] == "champion_selected"
    }
    return {key: entry for key, entry in models_metadata.items() if key in keep_keys}
