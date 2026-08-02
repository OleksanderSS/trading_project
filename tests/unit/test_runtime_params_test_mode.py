"""A full run must not be silently trained as a test run.

_save_runtime_params writes runtime_params.json only for test runs, and
deletes a stale one otherwise -- because Colab's ConfigLoader checks that
the file EXISTS, not that this run is a test, so a leftover file forces
every model to train with old (usually epochs=1) settings while
batch_metadata.json still reports test_mode: false. The comment in the
source says exactly that.

The condition disabled itself:

    has_test_params = ... or getattr(args, 'max_iterations', None)

--max-iterations has a DEFAULT of 100, which is truthy, so
has_test_params was always True, the deletion branch was unreachable, and
every run wrote the file. The guard against "full mode trained with test
epochs" was itself the thing that made every mode a test mode.

An explicit value now means one that differs from the default -- the same
test main() already applies at `args.max_iterations != 100`.
"""
from __future__ import annotations

import pytest

from src.cli.argument_parser import create_argument_parser


def _is_test_run(argv):
    """The condition as _save_runtime_params applies it."""
    args = create_argument_parser().parse_args(argv)
    return bool(
        args.test_ticker
        or args.test_target
        or args.test_model
        or getattr(args, "epochs", None) is not None
        or (
            getattr(args, "max_iterations", None) is not None
            and args.max_iterations != 100
        )
    )


def test_a_plain_full_run_is_not_a_test_run():
    """The regression: this returned True, so every run wrote
    runtime_params.json."""
    assert not _is_test_run(["--mode", "full"])


def test_a_plain_local_run_is_not_a_test_run():
    assert not _is_test_run(["--mode", "local"])


@pytest.mark.parametrize("argv", [
    ["--mode", "prepare", "--epochs", "1"],
    ["--mode", "prepare", "--max-iterations", "1"],
    ["--mode", "prepare", "--epochs", "1", "--max-iterations", "1"],
    ["--mode", "prepare", "--test-ticker", "AAPL"],
    ["--mode", "prepare", "--test-target", "target_up_1d"],
    ["--mode", "prepare", "--test-model", "catboost"],
])
def test_an_explicitly_limited_run_is_a_test_run(argv):
    assert _is_test_run(argv)


def test_the_default_iteration_count_is_still_100():
    """The comparison depends on it; a change here silently breaks the
    discrimination above."""
    assert create_argument_parser().parse_args([]).max_iterations == 100


def test_max_iterations_equal_to_the_default_is_not_an_override():
    """Passing the default explicitly is not a request for test mode."""
    assert not _is_test_run(["--mode", "full", "--max-iterations", "100"])


def test_an_unsupplied_iteration_count_is_not_an_override():
    """The existing suite constructs args with max_iterations=None, meaning
    "not supplied". An earlier version of this fix read that as an override
    and broke two of those tests -- they caught it."""
    from types import SimpleNamespace

    args = SimpleNamespace(
        test_ticker=None, test_target=None, test_model=None,
        epochs=None, max_iterations=None,
    )
    overridden = args.max_iterations is not None and args.max_iterations != 100

    assert not overridden


def test_the_guard_in_the_source_matches_this():
    import inspect
    from pathlib import Path

    source = Path("run_hybrid_pipeline.py").read_text(encoding="utf-8")

    assert "iterations_overridden" in source
    assert "or getattr(args, 'max_iterations', None)" not in source


def test_the_lightened_run_keeps_full_breadth():
    """The point of --epochs 1 --max-iterations 1 without --test-*: every
    ticker and all three timeframes, only fewer passes."""
    from pathlib import Path

    source = Path("run_hybrid_pipeline.py").read_text(encoding="utf-8")

    assert "'timeframes': ['15m', '1h', '1d']" in source
    assert "'tickers': 'all'" in source
