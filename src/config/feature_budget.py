"""How many features one model may be trained on. One answer, one place.

Three copies of this number existed and the configured one lost:

- `src/config/models.yaml` -> `models.per_model.<type>.max_features`, labelled
  "Number of features for each model". Read by base_trainer and
  light_model_trainer, but passed to the MODEL CONSTRUCTOR as a
  hyperparameter, so it never limited anything. (lightgbm silently drops it;
  sklearn's own `max_features` means "features considered per split", which is
  a different concept entirely.)
- `src/features/feature_selector.py` -> a hardcoded map. Zero importers.
- `scripts/colab/colab_clean_cell.py` -> its own copy of that map. **This is
  the one that ran.**

Measured on 4,613 heavy-model artifacts: cnn trained on exactly 64 features,
lstm/gru/transformer/autoencoder on 128, mlp/tabnet up to 256 — matching the
hardcoded map on all seven types and the config on none. Light models were not
capped at all: median 388 selected features, max 1,145.

Why the budget matters more than which model you pick: with ~308 training rows
per daily context, 100 features leaves three observations per feature. That
does not produce a weak model, it produces an unfalsifiable one — you cannot
tell a real edge from a fitted coincidence. At 35 features the same context
gives ~9 observations each, and 15m contexts (~656 rows) give ~19.
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

#: Used when config carries no value for a model. Deliberately the same for
#: every type: the binding constraint is how many rows the context has, not
#: how complex the architecture is. The old numbers had it backwards, giving
#: neural nets (90-115) more features than trees (35-50) when neural nets need
#: MORE observations per parameter, not fewer.
DEFAULT_MAX_FEATURES = 35


def get_model_max_features(model_type: str, config_manager=None) -> int:
    """Feature budget for `model_type`, from `models.per_model.*.max_features`.

    Falls back to DEFAULT_MAX_FEATURES when the key is missing or unusable,
    so a new model type gets a sane budget rather than an unlimited one.
    """
    if not model_type:
        return DEFAULT_MAX_FEATURES

    try:
        if config_manager is None:
            from src.config.unified_config_manager import get_current_config
            config_manager = get_current_config()
        value = config_manager.get(
            f"models.per_model.{model_type.lower()}.max_features",
            DEFAULT_MAX_FEATURES,
        )
        budget = int(value)
        return budget if budget > 0 else DEFAULT_MAX_FEATURES
    except (ImportError, AttributeError, TypeError, ValueError) as e:
        # Say it. A budget silently reverting to the default is exactly the
        # shape of defect this project keeps finding: the operator edits
        # models.yaml, nothing changes, and nothing explains why.
        logger.warning(
            "Could not read the feature budget for '%s' (%s); falling back to %d.",
            model_type, e, DEFAULT_MAX_FEATURES,
        )
        return DEFAULT_MAX_FEATURES


#: Safety margin between the pre-screen ceiling and the largest budget any
#: model may actually spend. The pre-screen ranks columns by the SAME
#: statistic the per-model budget uses, so a ceiling equal to the largest
#: budget would already be enough -- the top 35 of the top 70 are the top 35.
#: The doubling exists for the one case where that reasoning is not exact:
#: the pre-screen ranks the raw column, the budget ranks it after scaling,
#: and two columns whose correlations agree to twelve decimal places can
#: swap places between the two. A swap inside the margin cannot change what
#: any model is trained on; a swap at the ceiling itself could.
PRESELECTION_MARGIN = 2

#: Never pre-screen below this, whatever the configured budgets say. A run
#: with every budget set to 5 should still leave the arena something to
#: choose between.
MIN_PRESELECTION_CEILING = 64


def get_preselection_ceiling(config_manager=None) -> int:
    """How many columns are worth imputing and scaling at all.

    The pipeline used to impute, scale and sequence EVERY numeric column and
    then hand each model its budget of 5 to 35. Measured on the pooled daily
    context of 2026-08-31: 474 columns across 490,799 rows, of which the
    largest budget spends 35. The other 439 were carried through
    `SimpleImputer(strategy='median')`, which sorts through a masked array --
    a `(490799, 200)` int64 index, 749 MiB for one block -- and the stage died
    there with a MemoryError after eight hours.

    So the ceiling is derived from what the budgets can actually spend rather
    than being a new tuning knob: the largest configured budget, doubled, and
    never below `MIN_PRESELECTION_CEILING`. Nothing here decides WHICH columns
    survive; that is the caller's ranking, and it is the same ranking the
    budget itself uses.
    """
    budgets: list[int] = []
    try:
        if config_manager is None:
            from src.config.unified_config_manager import get_current_config
            config_manager = get_current_config()
        per_model = config_manager.get("models.per_model", {}) or {}
        for settings in per_model.values():
            if not isinstance(settings, dict):
                continue
            value = settings.get("max_features")
            if value is None:
                continue
            budget = int(value)
            if budget > 0:
                budgets.append(budget)
    except (ImportError, AttributeError, TypeError, ValueError, KeyError) as e:
        # KeyError belongs here: a config manager that raises on a missing
        # key is a normal shape, and letting it through would take the whole
        # modelling stage down over a ceiling that has a documented default.
        logger.warning(
            "Could not read the per-model feature budgets (%s); the "
            "pre-screen ceiling falls back to the default budget.", e,
        )

    largest = max(budgets) if budgets else DEFAULT_MAX_FEATURES
    return max(MIN_PRESELECTION_CEILING, largest * PRESELECTION_MARGIN)
