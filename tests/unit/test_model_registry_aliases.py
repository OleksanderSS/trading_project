"""One model, one entry -- and the alias spellings are the ones on disk.

MODELS carries two alias entries, 'lightgbm' -> 'lgbm' and
'random_forest' -> 'rf'. Two problems came out of that:

1. get_all_model_names() returned MODELS.keys() outright, so 'lgbm' and
   'lightgbm' counted as two models. battle_groups feeds that list straight
   into BATTLE_GROUPS ('light_vs_heavy', 'all_models'), so one model entered
   a tournament twice and drew twice the battle slots. It also disagreed with
   get_models_by_type, which skips aliases because they have no 'type': 18
   names one way, 16 the other.

   Latent, not live: ArenaOrchestrator is constructed nowhere outside its own
   module, and the arena lines in logs/system.log come from tests and end in
   "arena offline".

2. The models that ACTUALLY trained are recorded under the alias spellings.
   From experience_diary: catboost, knn, lightgbm, linear, random_forest,
   svm, xgboost -- five canonical, two aliases. So code that matches stored
   artifacts against get_models_by_type('light'), which yields 'lgbm' and
   'rf', would miss exactly those two.
"""
from __future__ import annotations

import pytest

from src.models.registry.model_registry import ModelRegistry

# Verified against experience_diary on 2026-08-01.
NAMES_THAT_ACTUALLY_TRAINED = [
    "catboost", "knn", "lightgbm", "linear", "random_forest", "svm", "xgboost",
]
ALIASES = ["lightgbm", "random_forest"]


def test_each_model_is_listed_once():
    names = ModelRegistry.get_all_model_names()
    assert len(names) == len(set(names))
    for alias in ALIASES:
        assert alias not in names


def test_aliases_are_still_reachable_when_asked_for():
    names = ModelRegistry.get_all_model_names(include_aliases=True)
    for alias in ALIASES:
        assert alias in names


def test_the_name_list_agrees_with_the_type_lists():
    """These two disagreed: 18 names against 16."""
    by_type: list[str] = []
    for model_type in ("light", "heavy", "enhanced"):
        by_type += ModelRegistry.get_models_by_type(model_type)

    assert sorted(ModelRegistry.get_all_model_names()) == sorted(by_type)


@pytest.mark.parametrize("alias,canonical", [
    ("lightgbm", "lgbm"),
    ("random_forest", "rf"),
])
def test_an_alias_resolves_to_its_canonical_name(alias, canonical):
    assert ModelRegistry.resolve_model_name(alias) == canonical


def test_a_canonical_name_resolves_to_itself():
    assert ModelRegistry.resolve_model_name("catboost") == "catboost"


def test_an_unknown_name_is_returned_unchanged():
    """Callers pass names from disk; resolving must not invent or drop one."""
    assert ModelRegistry.resolve_model_name("something_else") == "something_else"


@pytest.mark.parametrize("name", NAMES_THAT_ACTUALLY_TRAINED)
def test_every_trained_model_resolves_to_a_real_config(name):
    """Both spellings must lead to a config, since both appear on disk."""
    resolved = ModelRegistry.resolve_model_name(name)

    assert resolved in ModelRegistry.get_all_model_names()
    config = ModelRegistry.get_model_config(name)
    assert config and "alias_for" not in config
    assert config["type"] in ("light", "heavy", "enhanced")


def test_battle_groups_no_longer_field_a_model_twice():
    from src.analytics.arena.battle_groups import BATTLE_GROUPS

    for name, group in BATTLE_GROUPS.items():
        assert len(group.models) == len(set(group.models)), (
            f"battle group {name!r} lists a model more than once"
        )
