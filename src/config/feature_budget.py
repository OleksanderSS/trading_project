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
    except (ImportError, AttributeError, TypeError, ValueError):
        return DEFAULT_MAX_FEATURES
