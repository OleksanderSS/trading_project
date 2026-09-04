"""Balanced accuracy and raw accuracy are not two scores. They are two
quantities, and ranking one against the other is not a comparison.

REGISTER #153. The local side and the Colab side both feed one
`models_metadata`, and `filter_to_champions` picks the highest number per
(ticker, timeframe, target). What each side puts in that number is not the
same thing:

    LOCAL     `score` -- what the gate judged the model by: balanced accuracy
              against a naive opponent, over folds, with a refusal if it does
              not beat chance on three folds of four.

    COLAB     val_accuracy, val_auc, accuracy, auc -- from ONE validation
              split. Measured 2026-09-04 against the 2,077-line cell: zero
              mentions of `folds`, `balanced_accuracy`, `naive`, `baseline`,
              `sealed`, `SEAL_START` or `2023-09-01`.

Raw accuracy is the metric this project already measured handing 0.7381 to a
predictor that never fires, while the model itself scored 0.5257 balanced
(#187). So a heavy model did not have to be better to win a group -- it only
had to be scored more generously.

WHAT WAS CHOSEN AND WHY. The register offered two options: teach the Colab
cell to compute the gate's evidence, or stop heavy models being champion
candidates. Neither is necessary. The comparison itself is the defect, and it
lives in code we own:

    when any candidate in a group faced the gate, only gated candidates
    compete, and the others are recorded by name with their metric rather
    than silently losing or silently winning;

    every champion now carries `gated`, so a winner chosen on raw accuracy
    cannot read downstream like one that met a naive opponent.

This neither rewrites the Colab cell nor disables heavy models. It stops one
number being mistaken for another.
"""
from __future__ import annotations

import pytest

from src.pipeline.hybrid.champion_selector import select_champions

TARGETS = {"target_up_1d": "classification_binary"}


def _entry(model_type: str, metrics: dict, timeframe: str = "1d") -> dict:
    return {
        "ticker": "AAPL",
        "timeframe": timeframe,
        "target": "target_up_1d",
        "model_type": model_type,
        "metrics": metrics,
        "model_path": f"/models/{model_type}.pkl",
    }


def _champion(metadata: dict) -> dict:
    return select_champions(metadata, TARGETS)["AAPL::1d::target_up_1d"]


def test_a_raw_accuracy_cannot_beat_a_gated_score():
    """The exact shape: the heavy model's number is bigger and means less."""
    champion = _champion({
        "local_linear": _entry("linear", {"score": 0.5257}),
        "colab_lstm": _entry("lstm", {"accuracy": 0.7381}),
    })

    assert champion["champion_model_type"] == "linear", (
        "a model scored on raw accuracy outranked one judged by the gate; "
        "0.7381 is the number #187 measured for a predictor that never fires"
    )
    assert champion["selection_metric"] == "score"
    assert champion["gated"] is True


def test_the_excluded_candidate_is_named_with_its_metric():
    """Dropping it silently would trade one invisible defect for another."""
    champion = _champion({
        "local_linear": _entry("linear", {"score": 0.5257}),
        "colab_lstm": _entry("lstm", {"accuracy": 0.7381}),
    })

    excluded = champion["excluded_incomparable"]
    assert len(excluded) == 1
    assert excluded[0]["model_type"] == "lstm"
    assert excluded[0]["metric"] == "accuracy"
    assert "gate" in excluded[0]["reason"]


def test_gated_models_still_compete_normally_among_themselves():
    champion = _champion({
        "a": _entry("linear", {"score": 0.52}),
        "b": _entry("catboost", {"score": 0.58}),
    })
    assert champion["champion_model_type"] == "catboost"
    assert champion["gated"] is True
    assert champion["excluded_incomparable"] == []


def test_an_all_ungated_group_still_picks_one_but_says_it_was_not_gated():
    """Refusing outright would silently drop every Colab-only context. The
    honest outcome is a champion that admits what it is."""
    champion = _champion({
        "colab_lstm": _entry("lstm", {"accuracy": 0.71}),
        "colab_gru": _entry("gru", {"accuracy": 0.66}),
    })

    assert champion["status"] == "champion_selected"
    assert champion["champion_model_type"] == "lstm"
    assert champion["gated"] is False, (
        "a champion chosen on raw accuracy claims to have faced the gate"
    )


def test_a_group_with_no_comparable_metric_is_still_no_champion():
    """The pre-existing refusal must survive the new one."""
    champion = _champion({"x": _entry("linear", {})})
    assert champion["status"] == "no_champion"


@pytest.mark.parametrize("metric", ["accuracy", "val_accuracy", "auc", "val_auc"])
def test_every_metric_the_colab_cell_emits_loses_to_a_gated_score(metric):
    """Named individually because the cell emits four and a fix covering three
    would be found by accident later."""
    champion = _champion({
        "local": _entry("linear", {"score": 0.51}),
        "heavy": _entry("lstm", {metric: 0.99}),
    })
    assert champion["champion_model_type"] == "linear"
    assert champion["gated"] is True
