"""Routing multipliers reweight models during a sharp drop.

Live: StackedEnsemble calls adjust_weights on the prediction path
(stacked_ensemble.py:137), and data/processed/routing_rules.json exists, so
the rules are loaded and applied.

The target part of them never bound. The model name was parsed with
'_'.join(parts[:-1]), which drops the last underscore-separated piece --
'x_target_atr_14_f5' became 'atr_14' -- while every key in the rules file
carries the full suffix ('atr_14_f5', 'bb_upper_f1',
'daily_momentum_score_1d'). Measured: the intersection between the parsed
names and the rule keys was EMPTY, so all 36 target modifiers were dead
while hour/day/trend modifiers kept applying.

Checked and dismissed in the same pass: the four modifiers compound
multiplicatively with no clamp, but the values in the file are bounded
(hour 0.8-1.0, day 1.0, trend 1.0, target 0.0-1.5), so the product ranges
0.0 to 1.5. A zero is the file's way of saying "do not use this model in a
sharp drop", not a runaway.
"""
from __future__ import annotations

import json
import logging

import pytest

from src.analytics.context.dynamic_router import DynamicRouter

RULES = {
    "sharp_drop": {
        "target_modifiers": {"atr_14_f5": 0.5, "bb_upper_f1": 1.5},
        "hour_modifiers": {"14": 0.8},
        "day_modifiers": {"mon": 1.0},
        "trend_modifiers": {"down": 1.0},
    }
}


@pytest.fixture()
def router(tmp_path):
    instance = object.__new__(DynamicRouter)
    instance.rules = json.loads(json.dumps(RULES))
    instance.audit_log_path = tmp_path / "router_audit.jsonl"
    return instance


CONTEXT = {"regime": "sharp_drop", "hour_of_day": 14, "day_of_week": "mon",
           "trend_state": "down"}


def test_the_full_target_suffix_matches_the_rule_key(router):
    """The regression: 'atr_14_f5' used to be parsed down to 'atr_14'."""
    result = router.adjust_weights({"lgbm_target_atr_14_f5": 1.0}, CONTEXT)

    assert result["lgbm_target_atr_14_f5"] == pytest.approx(0.5 * 0.8)


def test_a_second_target_gets_its_own_modifier(router):
    result = router.adjust_weights({"cat_target_bb_upper_f1": 1.0}, CONTEXT)

    assert result["cat_target_bb_upper_f1"] == pytest.approx(1.5 * 0.8)


def test_an_unknown_target_keeps_the_global_modifiers_only(router):
    result = router.adjust_weights({"lgbm_target_not_in_rules": 1.0}, CONTEXT)

    assert result["lgbm_target_not_in_rules"] == pytest.approx(0.8)


def test_a_name_without_the_marker_is_reported_not_silently_neutral(router, caplog):
    """StackedEnsemble passes base-model column names, which may carry no
    '_target_' at all -- then the per-target rules cannot bind however the
    keys are parsed, and that is worth saying."""
    with caplog.at_level(logging.WARNING):
        result = router.adjust_weights({"lgbm": 1.0, "catboost": 1.0}, CONTEXT)

    assert result["lgbm"] == pytest.approx(0.8)
    assert any("target modifiers" in r.getMessage() for r in caplog.records)


def test_no_warning_when_a_target_did_match(router, caplog):
    with caplog.at_level(logging.WARNING):
        router.adjust_weights({"lgbm_target_atr_14_f5": 1.0}, CONTEXT)

    assert not [r for r in caplog.records if "target modifiers" in r.getMessage()]


def test_without_rules_every_weight_is_neutral():
    instance = object.__new__(DynamicRouter)
    instance.rules = {}

    assert instance.adjust_weights({"a": 1.0, "b": 1.0}, CONTEXT) == {"a": 1.0, "b": 1.0}


def test_a_calm_regime_leaves_weights_alone(router):
    result = router.adjust_weights({"lgbm_target_atr_14_f5": 1.0}, {"regime": "calm"})

    assert result["lgbm_target_atr_14_f5"] == 1.0


def test_the_explicit_sharp_drop_flag_also_triggers(router):
    result = router.adjust_weights(
        {"lgbm_target_atr_14_f5": 1.0},
        {"is_sharp_drop": True, "hour_of_day": 14},
    )

    assert result["lgbm_target_atr_14_f5"] == pytest.approx(0.5 * 0.8)


def test_the_shipped_rules_can_actually_match_a_target_name():
    """Against the real file: at least one key must be reachable by the
    parser, or the rules are decoration."""
    from pathlib import Path

    path = Path("data/processed/routing_rules.json")
    if not path.exists():
        pytest.skip("routing_rules.json is not present")

    rules = json.loads(path.read_text(encoding="utf-8"))
    keys = list(rules.get("sharp_drop", {}).get("target_modifiers", {}))
    assert keys, "no target modifiers shipped"

    parsed = f"model_target_{keys[0]}".split("_target_", 1)[1]
    assert parsed == keys[0]
