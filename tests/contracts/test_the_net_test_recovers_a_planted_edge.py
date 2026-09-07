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


# ---------------------------------------------------------------------------
# The rotated null, which lives in the INSTRUMENT and decides whether a net
# Sharpe is an edge or a tilt. Added 2026-09-06 with #287(a); before it, this
# script's own headline of seven days' standing was a low-volatility tilt read
# as the project's best result (CLAIMS R51).
# ---------------------------------------------------------------------------

_net_spec = importlib.util.spec_from_file_location(
    "net_test", PROJECT_ROOT / "scripts/diagnostics/net_test_every_survivor.py")
NET = importlib.util.module_from_spec(_net_spec)
_net_spec.loader.exec_module(NET)


def _rotation_panel():
    """A frame shaped the way `_rotation_index` requires: ticker, then date."""
    stamps = pd.date_range("2010-01-01", periods=DATES, freq="D")
    frame = pd.DataFrame({
        "ticker": np.repeat([f"T{i}" for i in range(NAMES)], DATES),
        "datetime": np.tile(stamps.to_numpy(), NAMES),
    })
    return frame.sort_values(["ticker", "datetime"]).reset_index(drop=True)


def test_rotation_moves_each_name_within_its_own_block():
    """A shift must never carry one name's position onto another name."""
    frame = _rotation_panel()
    moved = NET._rotation_index(frame, 37)
    assert (frame["ticker"].to_numpy()[moved]
            == frame["ticker"].to_numpy()).all(), (
        "rotation crossed a name boundary, so the 'same book, different dates' "
        "claim is false and every z built on it is meaningless.")


def test_a_static_tilt_reads_as_a_tilt():
    """A column that never changes rank holds the same book at any date.

    This is the case the constant bar could not see. The rotated book IS the
    real book, so the difference must be nothing.
    """
    frame = _rotation_panel()
    dates = frame["datetime"].to_numpy()
    codes, uniques = pd.factorize(dates, sort=True)
    generator = np.random.default_rng(3)
    per_name = dict(zip(sorted(frame["ticker"].unique()),
                        generator.standard_normal(NAMES)))
    tilt = frame["ticker"].map(per_name).to_numpy()
    position = CONTROL._position(tilt, dates)
    moved = position[NET._rotation_index(frame, 91)]
    moved = moved - NET._mean_by_date(moved, codes, len(uniques))[codes]
    assert np.abs(moved - position).max() < 1e-9, (
        "a column constant within each name produced a DIFFERENT book after "
        "rotation, so the null would credit a static tilt with timing.")


def test_the_mean_by_date_matches_the_pandas_it_replaced():
    """The fast path exists for speed and must not change an answer."""
    frame = _rotation_panel()
    dates = frame["datetime"].to_numpy()
    codes, uniques = pd.factorize(dates, sort=True)
    values = np.random.default_rng(5).standard_normal(len(frame))
    values[::17] = np.nan
    fast = NET._mean_by_date(values, codes, len(uniques))
    slow = (pd.DataFrame({"d": dates, "v": values})
            .groupby("d")["v"].mean().sort_index().to_numpy())
    assert np.allclose(fast, slow, equal_nan=True), (
        "the bincount date-mean disagrees with the groupby it replaced; every "
        "Sharpe in the net test runs through it.")


# ---------------------------------------------------------------------------
# Beta against the constant opponent, wired into the instrument on 2026-09-07.
# Dollar-neutral is not market-neutral: subtracting the per-date mean removes
# the level of the cross-section and leaves the beta. A third of the project's
# 1,404 books move with the opponent at |corr| > 0.2, and the best result it
# ever produced was 0.746 of it (CLAIMS R61).
# ---------------------------------------------------------------------------


def test_a_book_that_is_the_market_reads_as_beta_one():
    """A book identical to the opponent must not be reported as neutral."""
    generator = np.random.default_rng(19)
    market = generator.standard_normal(2000) * 0.01
    beta, correlation = NET._beta_on(market.copy(), market)
    assert abs(beta - 1.0) < 1e-9 and abs(correlation - 1.0) < 1e-9, (
        f"a book that IS the opponent reported beta {beta:.3f} and correlation "
        f"{correlation:.3f}. If the measure cannot see a perfect market book it "
        "cannot see a partial one either.")


def test_an_independent_book_reads_as_beta_zero():
    generator = np.random.default_rng(23)
    market = generator.standard_normal(4000) * 0.01
    book = generator.standard_normal(4000) * 0.01
    beta, correlation = NET._beta_on(book, market)
    assert abs(beta) < 0.1 and abs(correlation) < 0.1, (
        f"two independent series reported beta {beta:.3f}, correlation "
        f"{correlation:.3f}")


def test_beta_survives_gaps_in_either_series():
    """Real books carry NaN where a horizon runs off the end of the panel."""
    generator = np.random.default_rng(29)
    market = generator.standard_normal(3000) * 0.01
    book = 0.5 * market + generator.standard_normal(3000) * 0.005
    book[::13] = np.nan
    market[7::29] = np.nan
    beta, _ = NET._beta_on(book, market)
    assert abs(beta - 0.5) < 0.1, (
        f"with gaps in both series the beta came out {beta:.3f} instead of "
        "about 0.5, so the pairing of the two series is wrong somewhere")
