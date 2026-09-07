"""The Sharpe threshold must come from the null we ran, not from a formula.

`net_test_every_survivor.SHARPE_SE` scales every threshold the instrument
applies. It was 0.193 until 2026-09-07 -- sqrt(1/27), the textbook standard
error of an annualised Sharpe over the 27 explorable years, which assumes the
daily returns are independent draws. That is an assumption, and this project
has the measurement: the rotated nulls are books with the timing broken, so
their Sharpes are draws from the null distribution.

Measured, the pooled spread is 0.234 -- 21% WIDER than the formula, meaning
every threshold had been 21% too permissive. That is the direction that invents
findings, and it had not yet invented one: Bonferroni moves 0.798 -> 0.968 and
the expected maximum of noise 0.621 -> 0.753, with nothing clearing either bar
before or after (the best net Sharpe anywhere is 0.586).

CLAIMS R50 measured the same rotations and reported SMALLER numbers -- 0.113 at
hold 120, "the constant is too HIGH". Both are right and they are different
objects: R50 measured the spread WITHIN one column, which is the null for "does
THIS book's timing matter"; the threshold here is applied to the RAW Sharpe
against zero, and under the null a column sits at its own rotated null, not at
zero, so the spread of those centres across columns (0.222) belongs in it too.

These tests recompute the number from the run that is in the repository, so a
revert to a formula, or a panel that has genuinely changed, fails with the
measurement in hand rather than being argued about.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUN = PROJECT_ROOT / "diagnostic_reports" / "net_test_varying.csv"

_spec = importlib.util.spec_from_file_location(
    "net_test_every_survivor",
    PROJECT_ROOT / "scripts" / "diagnostics" / "net_test_every_survivor.py")
NET = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(NET)


@pytest.fixture(scope="module")
def run():
    if not RUN.exists():
        pytest.skip(f"{RUN.name} is not in the repository")
    frame = pd.read_csv(RUN)
    if not {"rotated_sd", "rotated_null"} <= set(frame.columns):
        pytest.skip("this run predates the rotated null columns")
    return frame


def _pooled(frame) -> float:
    """sqrt(within^2 + across^2) over the rotated nulls."""
    within = float(np.sqrt(np.mean(frame["rotated_sd"] ** 2)))
    across = float(frame["rotated_null"].std(ddof=1))
    return float(np.sqrt(within ** 2 + across ** 2))


def test_the_constant_matches_the_null_it_claims_to_describe(run):
    measured = _pooled(run)
    assert abs(NET.SHARPE_SE - measured) < 0.02, (
        f"SHARPE_SE is {NET.SHARPE_SE:.3f} and the rotated nulls in "
        f"{RUN.name} spread by {measured:.3f}. The threshold has stopped "
        "describing the null it is applied to.")


def test_it_is_not_the_textbook_formula_again(run):
    """The specific regression: someone re-derives 1/sqrt(years)."""
    theoretical = (1.0 / 27.0) ** 0.5
    assert abs(NET.SHARPE_SE - theoretical) > 0.02, (
        f"SHARPE_SE is back at {NET.SHARPE_SE:.3f}, which is sqrt(1/27). That "
        "assumes independent days; the measured null spreads 21% wider, so "
        "every threshold built on it is too permissive.")


def test_the_within_column_spread_is_not_used_by_mistake(run):
    """The tempting wrong number, and it is wrong by a factor of three.

    Within-column spread answers "does THIS book's timing matter" and is what
    `z_vs_own_null` uses. A THRESHOLD asks how far the best of N attempts can
    get by luck, and the N attempts are N different features producing N null
    books with different risk -- so the across-column spread belongs in it and
    dominates it.
    """
    within = float(np.sqrt(np.mean(run["rotated_sd"] ** 2)))
    across = float(run["rotated_null"].std(ddof=1))
    assert across > within, (
        "the across-column spread no longer dominates, so the reasoning in "
        "SHARPE_SE's note needs re-deriving rather than the number nudging")
    assert NET.SHARPE_SE > within * 2, (
        f"SHARPE_SE is {NET.SHARPE_SE:.3f}, close to the WITHIN-column spread "
        f"{within:.3f}. That cuts every threshold to a third and manufactures "
        "significance.")


def test_a_wider_bar_is_never_quietly_narrower_than_the_old_one(run):
    """A correction that loosened the bar would need saying out loud."""
    assert NET.SHARPE_SE >= (1.0 / 27.0) ** 0.5, (
        "the measured bar came out BELOW the theoretical one. That is "
        "possible, but it makes every past negative result weaker rather than "
        "stronger, so it may not land as a quiet constant change.")


def test_the_thresholds_still_reject_everything_this_run_produced(run):
    """The conclusion the correction must not silently create.

    The project's standing result is that nothing clears the multiplicity
    correction. Raising SHARPE_SE raises the bar, so it cannot create a
    survivor -- but it is asserted rather than reasoned, because "my change
    could only help" is how a finding gets manufactured.
    """
    attempts = len(run) * 6
    bonferroni, _ = NET._thresholds(attempts)
    survivors = run.loc[run["best_net"] > bonferroni, "feature"].tolist()
    assert not survivors, (
        f"{survivors} now clear the Bonferroni bar of {bonferroni:.3f}. "
        "Nothing did before the standard error was re-measured, and a wider "
        "bar cannot admit anything -- so this is a bug, not a discovery.")
