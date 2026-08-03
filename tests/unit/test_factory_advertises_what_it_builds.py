"""ModelFactory advertised seven models it could not build.

get_available_models() returned the whole ModelRegistry, which describes
every model the SYSTEM knows about -- including the seven trained in Colab.
Its callers read it as "what can be trained here":

  - training/constants.py: DEFAULT_ENABLED_MODEL_TYPES
  - UnifiedTrainingManager._get_available_model_names, the fallback when no
    category is configured
  - ContextualModelSelector(available_models=...), i.e. the universe it ranks
    over -- so it could recommend a model no local code path can construct

The seven classes were archived on 2026-08-02 (unreachable from both
training paths, and prediction loads artifacts by path rather than by
reconstructing project classes). The registry keeps describing them, because
they are real models that really do get trained -- just not here.
"""
from __future__ import annotations

import pytest

from src.factories.model_factory import ModelFactory
from src.models.registry.model_registry import ModelRegistry

COLAB_ONLY = {"lstm", "gru", "cnn", "transformer", "tabnet", "mlp", "autoencoder"}


def test_every_advertised_model_can_actually_be_built():
    """The property the whole change exists to restore."""
    for name in ModelFactory.get_available_models():
        assert name.lower() not in COLAB_ONLY, (
            f"{name} is advertised as available but is built in Colab, not here"
        )


def test_the_registry_still_describes_the_colab_models():
    """They are real models that really get trained -- just not locally.
    Removing them from the registry would lose that."""
    known = {name.lower() for name in ModelRegistry.get_all_model_names()}

    assert COLAB_ONLY <= known


@pytest.mark.parametrize("name", sorted(COLAB_ONLY))
def test_asking_for_a_colab_model_explains_itself(name):
    """Distinguishable from a typo: models.yaml categories.heavy still lists
    these, so a caller can arrive holding a perfectly valid name."""
    with pytest.raises(ValueError, match="Colab"):
        ModelFactory.create_model(name)


def test_an_unknown_name_still_reads_as_unknown():
    with pytest.raises(ValueError, match="Unsupported model name"):
        ModelFactory.create_model("not_a_model")


def test_a_local_model_still_builds():
    assert ModelFactory.create_model("catboost") is not None


def test_the_sequence_builder_was_not_archived_with_the_rest():
    """loader.py uses it to shape inputs for Colab-trained sequence models at
    prediction time, so it is live even though every class beside it went."""
    from src.models.neural.sequence_builder import SequenceBuilder

    assert SequenceBuilder is not None


def test_the_archived_classes_are_gone_from_the_live_tree():
    import importlib

    for module in (
        "src.models.neural.lstm_model",
        "src.models.neural.mlp_model",
        "src.models.neural.base_neural",
    ):
        with pytest.raises(ModuleNotFoundError):
            importlib.import_module(module)
