"""The net test's book must return an edge that was put into it.

R9 planted an edge in stage 7 and R10 in the learning path. Neither touched the
book in `net_test_every_survivor.py` -- the cross-sectional rank, the
dollar-neutralisation, the friction on both legs, the phase averaging -- and
that book is what produced R22, R23, R28, R30 and R32, which is every "nothing
survives" verdict this project has.

A verdict from an uncalibrated instrument cannot be read. These tests are the
standing version of that calibration: fast, synthetic, and they fail if an edit
makes the book blind or makes it invent an edge that is not there.

They are NOT a claim about the market. The size of the edge the REAL panel can
carry is measured by
`scripts/diagnostics/what_edge_would_the_net_test_have_seen.py`.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]

_spec = importlib.util.spec_from_file_location(
    "planted_edge_control",
    PROJECT_ROOT / "scripts" / "diagnostics"
    / "what_edge_would_the_net_test_have_seen.py")
CONTROL = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(CONTROL)

NAMES = 30
DATES = 400
HOLD = 5


def _panel(rho: float, seed: int):
    """A synthetic cross-section with a planted edge of correlation `rho`.

    Frictionless on purpose: this asks whether the book can SEE, and charging
    a cost here would mix "blind" with "eaten by fees", which is exactly the
    confusion R30 ran into.
    """
    generator = np.random.default_rng(seed)
    stamps = pd.date_range("2010-01-01", periods=DATES, freq="D")
    dates = np.repeat(stamps.to_numpy(), NAMES)
    forward = generator.standard_normal(DATES * NAMES) * 0.01
    z_forward = CONTROL._standardise(forward, dates)
    noise = CONTROL._standardise(generator.standard_normal(DATES * NAMES), dates)
    column = rho * z_forward + np.sqrt(1.0 - rho ** 2) * noise
    return column, dates, forward, np.zeros(DATES * NAMES)


def test_a_planted_edge_comes_back_out():
    """rho=0.30 must read as clearly positive, or the book cannot see."""
    column, dates, forward, friction = _panel(0.30, seed=7)
    sharpe = CONTROL._book_sharpe(column, dates, forward, friction, HOLD)
    assert sharpe > 1.0, (
        f"planted IC 0.30 came back as Sharpe {sharpe:.3f}. The book is not "
        "returning an edge it was handed, so every 'nothing survives' verdict "
        "it has produced is a statement about the book.")


def test_a_bigger_edge_reads_as_bigger():
    """Order must be preserved, or the number is not a measurement of size."""
    small = CONTROL._book_sharpe(*_panel(0.10, seed=11), HOLD)
    large = CONTROL._book_sharpe(*_panel(0.30, seed=11), HOLD)
    assert large > small, (
        f"IC 0.30 read {large:.3f} and IC 0.10 read {small:.3f}. A book whose "
        "output does not rise with the edge cannot rank candidates.")


@pytest.mark.parametrize("seed", [1, 2, 3, 4, 5])
def test_no_edge_does_not_manufacture_one(seed):
    """rho=0 must land near zero on every seed.

    The threshold is deliberately loose: one draw of a Sharpe over 400 bars
    has a real standard error, and a tight bound here would be a flaky test
    pretending to be a contract. What it catches is a BIAS -- the shape that
    made seven degenerate columns score ~1.00 net on 2026-09-04 before the
    per-date demeaning was added.
    """
    sharpe = CONTROL._book_sharpe(*_panel(0.0, seed=seed), HOLD)
    assert abs(sharpe) < 1.0, (
        f"an unplanted column scored {sharpe:.3f}. The book is manufacturing "
        "an edge from noise.")
