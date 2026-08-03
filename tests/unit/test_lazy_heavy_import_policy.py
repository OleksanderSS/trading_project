import json
import subprocess
import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _import_state_after(module_name: str) -> dict[str, bool]:
    code = (
        "import importlib, json, sys; "
        f"importlib.import_module({module_name!r}); "
        "print(json.dumps({"
        "'torch': 'torch' in sys.modules, "
        "'transformers': 'transformers' in sys.modules, "
        "'spacy': 'spacy' in sys.modules, "
        "'yfinance': 'yfinance' in sys.modules"
        "}))"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(result.stdout.strip().splitlines()[-1])


def test_the_colab_side_package_is_no_longer_in_the_live_tree():
    """This used to assert src.colab.models.model_factory imports lazily.

    The whole src/colab/ package was archived on 2026-08-02: nothing outside
    it imported it, and the real Colab entry point --
    scripts/colab/colab_clean_cell.py, which the notebook now %runs directly
    -- builds its models inline with sklearn/tf.keras/pytorch_tabnet and
    imports none of it. The lazy-import property it guarded no longer has a
    subject.
    """
    import importlib

    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("src.colab.models.model_factory")


def test_finbert_pipeline_import_does_not_import_heavy_nlp_dependencies():
    state = _import_state_after("src.features.nlp.models.finbert_pipeline")

    assert state["torch"] is False
    assert state["transformers"] is False


def test_entity_extractor_import_does_not_import_spacy():
    state = _import_state_after("src.features.nlp.extractors.entity_extractor")

    assert state["spacy"] is False


def test_trading_calendar_import_does_not_import_yfinance():
    state = _import_state_after("src.utils.trading_calendar")

    assert state["yfinance"] is False


def test_enhanced_ensemble_import_does_not_import_torch():
    state = _import_state_after("src.archive.models_dead.enhanced_ensemble")

    assert state["torch"] is False
