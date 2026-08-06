"""Three timeframes' light models were three writes to one filename.

Stage 4 runs the training suite once per (ticker, timeframe). BaseTrainer
wrote every candidate to

    model_{ticker}_{target}_{model_type}.joblib

and every promoted winner to

    CHAMP_{ticker}_{target}.joblib

in a single output directory. So the 15m fit, the 60m fit and the 1d fit
were three writes to the same path, and only the last one survived.

The metadata never noticed. Champions have always been keyed by
{ticker}_{timeframe}_{target}_{pattern}, so the results listed three
distinct champions while the files behind them were one file -- and
whichever timeframe Stage 4 processed last answered for all three. A 1d
model scoring 15m bars is not a smaller edge; it is a different question
answered by accident.

The heavy branch had the same defect from the other direction (one model
trained on all three timeframes at once) -- see
test_colab_timeframe_split.py and test_heavy_model_key_carries_timeframe.py.
"""
from __future__ import annotations

import inspect

import pytest

from src.pipeline.constants import champion_filename, model_candidate_filename
from src.training.base_trainer import BaseTrainer


# ------------------------------------------------------------- the filenames


def test_candidates_of_different_timeframes_get_different_files():
    names = {
        model_candidate_filename("AAPL", tf, "target_return", "lightgbm")
        for tf in ("15m", "60m", "1d")
    }

    assert len(names) == 3


def test_champions_of_different_timeframes_get_different_files():
    names = {
        champion_filename("AAPL", tf, "target_return")
        for tf in ("15m", "60m", "1d")
    }

    assert len(names) == 3


def test_an_unknown_timeframe_leaves_no_hole_in_the_name():
    """An empty field would produce model_AAPL__target_x_lgbm -- a name with
    a gap in it that nothing parses. Omit the field instead."""
    name = model_candidate_filename("AAPL", "", "target_return", "lightgbm")

    assert "__" not in name
    assert name == "model_AAPL_target_return_lightgbm.joblib"


def test_the_candidate_name_is_what_the_resolver_can_parse():
    """The name has to survive the round trip, or Stage 5 cannot find it."""
    from src.pipeline.stages.prediction.model_resolver import ModelResolver

    resolver = ModelResolver.__new__(ModelResolver)
    name = model_candidate_filename("AAPL", "15m", "target_return", "lightgbm")
    stem = name[: -len(".joblib")]

    assert resolver._parse_model_stem(stem) == (
        "aapl", "15m", "target_return", "lightgbm",
    )


# ------------------------------------------------------------- the plumbing


def test_the_trainer_names_files_through_the_shared_helpers():
    """Not through a literal it keeps its own copy of."""
    for method in (BaseTrainer._save_model_candidate,
                   BaseTrainer._promote_champion_file,
                   BaseTrainer._save_champion):
        source = inspect.getsource(method)
        assert ".joblib\"" not in source and ".joblib'" not in source, (
            f"{method.__name__} builds a filename of its own again"
        )


@pytest.mark.parametrize("method", [
    BaseTrainer._save_model_candidate,
    BaseTrainer._promote_champion_file,
])
def test_the_trainer_accepts_a_timeframe(method):
    assert "timeframe" in inspect.signature(method).parameters


def test_the_modeling_stage_passes_the_timeframe_it_is_iterating():
    """The value exists at the call site; it simply was not handed over."""
    from src.pipeline.stages.modeling.orchestrator import ModelingStage

    context = ModelingStage._build_unified_training_context
    assert "timeframe" in inspect.signature(context).parameters

    run_source = inspect.getsource(ModelingStage.run)
    assert "timeframe=str(timeframe)," in run_source


def test_the_training_context_carries_it_to_the_trainer():
    source = inspect.getsource(
        __import__(
            "src.pipeline.stages.modeling.orchestrator",
            fromlist=["ModelingStage"],
        ).ModelingStage._build_unified_training_context
    )

    assert '"timeframe": timeframe' in source


def test_the_results_dict_carries_it_to_champion_promotion():
    """_finalize_ticker_results names the champion from `results`, not from
    the data dict -- so the timeframe has to reach both."""
    source = inspect.getsource(BaseTrainer._train_ticker_suite)

    assert '"timeframe": data.get("timeframe"' in source
