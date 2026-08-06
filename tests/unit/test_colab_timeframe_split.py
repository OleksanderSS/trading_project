"""The heavy branch trained one model over three timeframes at once.

Two defects, one visible and one not.

The visible one stopped the run: features and targets were merged on
(ticker, datetime) while the export stacks 15m, 60m and 1d into a single
file, so a 15m bar at 15:30 and a 60m bar at 15:30 are two rows sharing a
key. validate='one_to_one' refused, with

    Merge keys are not unique in either left or right dataset;
    not a one-to-one merge

Measured on the 2026-08-06 batch: 2,550 of 26,989 rows duplicate on
(ticker, datetime); zero duplicate once interval joins the key.

The silent one had been there all along. Targets are almost perfectly
partitioned by timeframe in the export -- target_return_1d appears only on
1d rows, target_intraday_return_15m only on 15m -- so the pooled fit was
mostly self-selecting through mask = notna(). Two exceptions were not:

- target_hourly_return_1h, _up_1h and _volume_spike_1h are populated on
  BOTH 15m and 60m rows (10,234 and 3,054), so those three mixed two bar
  sizes into one fit, with `interval` dropped as a string column so the
  model could not tell them apart.
- Every model was recorded under the timeframe 'all'. Stage 5 selects
  prediction rows by the model's timeframe, and a label naming no
  timeframe matches nothing.

Both halves of the hybrid now agree: one model per (ticker, timeframe,
target, model_type), named for exactly that.
"""
from __future__ import annotations

import inspect
import io
import tokenize

import pandas as pd
import pytest

from src.pipeline.constants import heavy_model_key


def _load_controller_class():
    """Import the Colab trainer without its heavy optional dependencies."""
    import importlib.util
    from pathlib import Path

    path = Path("scripts/colab/colab_clean_cell.py")
    if not path.exists():
        pytest.skip("colab trainer script not present")
    spec = importlib.util.spec_from_file_location("colab_clean_cell", path)
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as exc:  # pragma: no cover - depends on the environment
        pytest.skip(f"colab trainer imports unavailable here: {exc}")
    return module.ColabTrainingController


def _code_without_comments(func) -> str:
    """Source with comments stripped, string literals KEPT.

    The assertions below are about string literals ('interval', 'all'), so
    the lookahead scanner's _code_only -- which blanks strings too -- would
    erase the very evidence. Comments still have to go: this module's own
    prose quotes the expressions it forbids.
    """
    import textwrap

    source = textwrap.dedent(inspect.getsource(func))
    return " ".join(
        token.string
        for token in tokenize.generate_tokens(io.StringIO(source).readline)
        if token.type != tokenize.COMMENT
    )


# ----------------------------------------------------------------- the key


def test_the_merge_key_includes_the_interval():
    controller = _load_controller_class()
    code = _code_without_comments(controller._process_ticker)

    assert "'interval'" in code or '"interval"' in code, (
        "the merge key is back to (ticker, datetime), which is not unique "
        "across timeframes"
    )


def test_the_merge_validation_is_not_the_thing_that_was_removed():
    """The failure was the diagnosis. Deleting it would hide the defect."""
    controller = _load_controller_class()
    code = _code_without_comments(controller._process_ticker)

    assert "one_to_one" in code


def test_the_real_export_is_unique_only_with_the_interval():
    """The measurement this fix rests on, against the real artifact."""
    from pathlib import Path

    path = Path("data/colab/accumulated/main_database/features.parquet")
    if not path.exists():
        pytest.skip("no prepared batch on disk")

    frame = pd.read_parquet(path, columns=["ticker", "datetime", "interval"])

    assert frame.duplicated(["ticker", "datetime"]).sum() > 0, (
        "if this is zero the export changed shape and this test is stale"
    )
    assert frame.duplicated(["ticker", "datetime", "interval"]).sum() == 0


# --------------------------------------------------------------- the split


def test_rows_are_grouped_by_timeframe():
    controller = _load_controller_class()
    merged = pd.DataFrame({
        "interval": ["1d", "15m", "1d", "60m"],
        "close": [1.0, 2.0, 3.0, 4.0],
    })

    groups = dict(controller._by_timeframe(merged))

    assert set(groups) == {"1d", "15m", "60m"}
    assert len(groups["1d"]) == 2
    assert len(groups["15m"]) == 1


def test_a_frame_without_the_column_still_trains():
    """Under an honest label, rather than being skipped."""
    controller = _load_controller_class()
    merged = pd.DataFrame({"close": [1.0, 2.0]})

    groups = list(controller._by_timeframe(merged))

    assert len(groups) == 1
    timeframe, rows = groups[0]
    assert timeframe == "all"
    assert len(rows) == 2


def test_the_interval_column_is_not_offered_to_the_model():
    """It is constant within a fit, so it carries nothing but risk."""
    controller = _load_controller_class()
    code = _code_without_comments(controller._process_target)

    assert "'interval'" in code or '"interval"' in code


def test_results_are_no_longer_filed_under_the_literal_all():
    controller = _load_controller_class()

    for method in (controller._process_target, controller._train_model):
        code = _code_without_comments(method)
        assert "'all'" not in code and '"all"' not in code, (
            f"{method.__name__} still hardcodes the 'all' timeframe bucket"
        )


# ----------------------------------------------------------------- the name


def test_the_model_name_carries_the_timeframe():
    controller = _load_controller_class()

    assert controller._context_id("AAPL", "15m", "target_x", "mlp") == (
        "AAPL_15m_target_x_mlp"
    )


def test_both_halves_of_the_hybrid_build_the_same_key():
    """One definition, two readers -- this is the whole point of the helper."""
    controller = _load_controller_class()

    assert controller._context_id("AAPL", "1d", "target_x", "lstm") == (
        heavy_model_key("AAPL", "1d", "target_x", "lstm")
    )


def test_the_trainers_do_not_rebuild_the_filename():
    """Seven trainers each composed their own copy of the name.

    The skip check, _save_sidecar and the results file all use the caller's
    version. A trainer composing it independently is a name waiting to
    diverge, and it would diverge silently: a model written where nothing
    looks for it, and therefore retrained forever.
    """
    controller = _load_controller_class()

    trainers = [
        controller._train_mlp_model,
        controller._train_cnn_model,
        controller._train_lstm_model,
        controller._train_gru_model,
        controller._train_transformer_model,
        controller._train_tabnet_model,
        controller._train_autoencoder_model,
    ]
    for trainer in trainers:
        code = _code_without_comments(trainer)
        assert "batch_dir /" not in code, (
            f"{trainer.__name__} builds its own model path again"
        )
        assert "model_path" in inspect.signature(trainer).parameters, (
            f"{trainer.__name__} does not accept the caller's model_path"
        )


def test_tabnet_is_handed_a_stem_not_a_zip():
    """pytorch_tabnet appends the extension itself.

    save_model calls shutil.make_archive(path, "zip"), which writes
    path + ".zip". Given a path already ending in .zip it produced
    model_..._tabnet.zip.zip, while the skip check, the sidecar and Stage 5
    all looked for the single-suffix name -- so every TabNet model was
    written where nothing would find it.
    """
    controller = _load_controller_class()
    seen = {}

    class _FakeTabNet:
        def save_model(self, path):
            seen["path"] = path

    from pathlib import Path

    controller._save_tabnet(_FakeTabNet(), Path("dir/model_AAPL_1d_t_tabnet.zip"))

    assert not seen["path"].endswith(".zip"), seen["path"]
    assert seen["path"].endswith("model_AAPL_1d_t_tabnet")
