"""The light/heavy split must have one source, not two that happen to agree.

`models.yaml` `models.categories.heavy` and the Colab trainer's list of
models to train were two hand-maintained copies of the same decision. They
agreed, but nothing kept them agreeing -- and this codebase has paid for
that shape repeatedly: the issuer registry at 4 vs 12 tickers,
CAPABILITY_CONTRACTS at 28 vs 39, TELEGRAM_TOKEN vs TELEGRAM_BOT_TOKEN,
`colab_results.json` vs `colab_results_summary.json` (which cost Stage 5
every heavy model it should have resolved).

The Colab script now READS models.yaml, mirroring how it already reads
targets.yaml through the shared registry loader. A hardcoded fallback
remains for the case where the repo on Drive is incomplete -- GPU time
should not be lost to a missing YAML -- and these tests keep that fallback
honest. A fallback that silently disagrees with the config is worse than no
fallback, because it looks like it worked.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
COLAB_CELL = ROOT / "scripts" / "colab" / "colab_clean_cell.py"
MODELS_YAML = ROOT / "src" / "config" / "models.yaml"


def _configured_categories() -> dict[str, list[str]]:
    config = yaml.safe_load(MODELS_YAML.read_text(encoding="utf-8")) or {}
    return ((config.get("models") or {}).get("categories") or {})


def _colab_fallback() -> list[str]:
    """Read the literal out of the source without importing it.

    Importing the cell drags in TensorFlow and the whole project; parsing the
    assignment is exact and costs nothing.
    """
    tree = ast.parse(COLAB_CELL.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "_HEAVY_MODELS_FALLBACK":
                    return [str(v) for v in ast.literal_eval(node.value)]
    raise AssertionError("_HEAVY_MODELS_FALLBACK not found in the Colab cell")


def test_the_colab_fallback_matches_the_configured_heavy_list():
    assert _colab_fallback() == _configured_categories().get("heavy"), (
        "The Colab trainer's fallback list has drifted from "
        "models.yaml models.categories.heavy. Update the fallback, or better, "
        "confirm the config read is still the primary path."
    )


def test_the_colab_trainer_reads_the_config_rather_than_a_literal():
    """The fallback existing is fine; the fallback being the only path is not."""
    source = COLAB_CELL.read_text(encoding="utf-8")

    assert "config_loader.heavy_models" in source
    assert "models.yaml" in source
    # The old literal, inline in run_training_pipeline.
    assert "heavy_models = ['mlp'" not in source


def test_the_two_categories_do_not_overlap():
    """A model in both lists would be trained twice, on different code paths,
    and the champion comparison would rank one against the other."""
    categories = _configured_categories()
    light = set(categories.get("light") or [])
    heavy = set(categories.get("heavy") or [])

    assert not (light & heavy), f"in both categories: {sorted(light & heavy)}"


def test_no_heavy_model_is_advertised_as_locally_buildable():
    """The other half of the split, from the local side.

    ModelFactory.get_available_models is read as "what can be trained here"
    by DEFAULT_ENABLED_MODEL_TYPES and by ContextualModelSelector's candidate
    universe.
    """
    from src.factories.model_factory import ModelFactory

    buildable = {str(name).lower() for name in ModelFactory.get_available_models()}
    heavy = {str(name).lower() for name in (_configured_categories().get("heavy") or [])}

    assert not (buildable & heavy), (
        f"advertised as locally buildable but assigned to Colab: "
        f"{sorted(buildable & heavy)}"
    )


@pytest.mark.parametrize("name", ["light", "heavy"])
def test_both_categories_are_present_and_non_empty(name):
    """_select_models_for_ticker branches on which of these exists; an empty
    or missing list silently changes which branch decides."""
    value = _configured_categories().get(name)

    assert isinstance(value, list) and value, f"models.categories.{name} is {value!r}"
