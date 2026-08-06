"""Three heavy models per (ticker, target, type) share one key -- two vanish.

Colab now trains one heavy model per timeframe. The local side keyed them
as {ticker}_{target}_{model_type} and ASSIGNED into models_metadata[key],
so the 15m and 60m entries were overwritten by the 1d one on the way in.
No exception, no log line: the dict simply ended up a third the size, and
Stage 5 scored 15m rows with whichever model happened to be written last.

The light branch has keyed champions by
{ticker}_{timeframe}_{target}_{pattern} since it was written. This makes
the heavy branch agree, through one shared definition rather than three
string literals that happen to match today.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from src.pipeline.constants import heavy_model_key
from src.pipeline.stages.prediction.model_resolver import ModelResolver


@pytest.fixture
def resolver():
    return ModelResolver(config_manager=None, model_pool=None, model_loader=None)


# ------------------------------------------------------------------- the key


def test_the_key_separates_timeframes():
    keys = {
        heavy_model_key("AAPL", tf, "target_return", "mlp")
        for tf in ("15m", "60m", "1d")
    }

    assert len(keys) == 3


def test_the_key_matches_the_light_branch_field_order():
    """ticker, timeframe, target, ... -- as in stages/modeling/orchestrator."""
    assert heavy_model_key("AAPL", "1d", "target_return", "mlp").startswith(
        "AAPL_1d_target_return"
    )


def test_the_filename_is_the_key_with_a_prefix_and_extension():
    """What makes model_resolver's '*{context_id}*' glob find the file."""
    key = heavy_model_key("AAPL", "1d", "target_return", "mlp")

    assert key in f"model_{key}.pkl"


# ------------------------------------------------------------ parsing a name


def test_a_timeframed_name_parses_into_four_fields(resolver):
    parsed = resolver._parse_model_stem("model_aapl_15m_target_return_mlp")

    assert parsed == ("aapl", "15m", "target_return", "mlp")


def test_a_multiword_model_type_still_parses(resolver):
    parsed = resolver._parse_model_stem("model_aapl_1d_target_return_random_forest")

    assert parsed == ("aapl", "1d", "target_return", "random_forest")


def test_an_older_name_without_a_timeframe_still_parses(resolver):
    """A file already on disk should not become unreadable because the
    naming convention grew a field."""
    parsed = resolver._parse_model_stem("model_aapl_target_return_mlp")

    assert parsed == ("aapl", "", "target_return", "mlp")


def test_a_target_beginning_with_a_number_is_not_read_as_a_timeframe(resolver):
    """The second field is a timeframe only if it spells one.

    Counting fields instead would make 'target_5' or any two-part target
    name shift every other field by one.
    """
    parsed = resolver._parse_model_stem("model_aapl_target_return_5_mlp")

    assert parsed == ("aapl", "", "target_return_5", "mlp")


# ----------------------------------------------------------- matching a file


def _files(*names):
    return {name: Path(f"{name}.pkl") for name in names}


def test_the_right_timeframe_wins(resolver):
    available = _files(
        "model_aapl_1d_target_return_mlp",
        "model_aapl_15m_target_return_mlp",
        "model_aapl_60m_target_return_mlp",
    )

    matched = resolver._match_model_file(
        "AAPL", "15m", "target_return", "mlp", available
    )

    assert matched is not None
    assert "15m" in matched.name


def test_an_alias_matches_its_canonical_timeframe(resolver):
    """1h and 60m are the same timeframe under two names -- a distinction
    that has cost this project a defect before."""
    available = _files("model_aapl_60m_target_return_mlp")

    matched = resolver._match_model_file(
        "AAPL", "1h", "target_return", "mlp", available
    )

    assert matched is not None


def test_an_unlabelled_file_still_matches(resolver):
    """It is the only candidate there is."""
    available = _files("model_aapl_target_return_mlp")

    matched = resolver._match_model_file(
        "AAPL", "1d", "target_return", "mlp", available
    )

    assert matched is not None


def test_a_different_timeframe_is_not_served_as_a_substitute(resolver):
    available = _files("model_aapl_1d_target_return_mlp")

    matched = resolver._match_model_file(
        "AAPL", "15m", "target_return", "mlp", available
    )

    assert matched is None, (
        "a 1d model was handed to a 15m context -- the exact substitution "
        "the timeframe was added to the key to prevent"
    )


# -------------------------------------------------- the two readers agree


def test_colab_results_yield_one_entry_per_timeframe(resolver):
    colab_results = {
        "ticker_results": {
            "AAPL": {
                "timeframes": {
                    tf: {
                        "results": {
                            "target_return": {
                                "models": {"mlp": {"metrics": {"r2": 0.1}}}
                            }
                        }
                    }
                    for tf in ("15m", "60m", "1d")
                }
            }
        }
    }
    models_metadata: dict = {}

    resolver._process_ticker_results_from_colab(colab_results, models_metadata)

    assert len(models_metadata) == 3, (
        f"expected one entry per timeframe, got {sorted(models_metadata)}"
    )
    assert {m["timeframe"] for m in models_metadata.values()} == {"15m", "60m", "1d"}


def test_selected_features_land_on_the_model_they_belong_to():
    """ResultsProcessor updates through `if key in models_metadata`, so a
    key built differently updates nothing and reports nothing."""
    from src.pipeline.hybrid.results_processor import ResultsProcessor

    processor = ResultsProcessor.__new__(ResultsProcessor)
    import logging

    processor.logger = logging.getLogger("test")

    models_metadata = {
        heavy_model_key("AAPL", tf, "target_return", "mlp"): {"selected_features": []}
        for tf in ("15m", "1d")
    }
    colab_results = {
        "ticker_results": {
            "AAPL": {
                "timeframes": {
                    "15m": {
                        "results": {
                            "target_return": {
                                "models": {
                                    "mlp": {"selected_features": ["rsi_15m"]}
                                }
                            }
                        }
                    },
                    "1d": {
                        "results": {
                            "target_return": {
                                "models": {
                                    "mlp": {"selected_features": ["rsi_1d"]}
                                }
                            }
                        }
                    },
                }
            }
        }
    }

    processor._update_selected_features_from_ticker_results(
        models_metadata, colab_results
    )

    assert models_metadata[
        heavy_model_key("AAPL", "15m", "target_return", "mlp")
    ]["selected_features"] == ["rsi_15m"]
    assert models_metadata[
        heavy_model_key("AAPL", "1d", "target_return", "mlp")
    ]["selected_features"] == ["rsi_1d"]
