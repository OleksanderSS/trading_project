"""A verdict about a column with no data is not a weak finding. It is not one.

On 2026-09-07 the feature catalogue described 67 columns as "market-wide: use
as interaction" -- among them every `sentiment_*`, `news_*` and `hype_*` column
on the daily frame. Each of those holds ONE single value across all 623,398
rows before the seal. The verdict was produced by a single test, `varies <
0.05`: does the column differ between names on the same date? A filled default
does not, so it fell into the same bucket as a genuine macro series, and the
catalogue then told the reader to build an interaction term out of a constant.

A further 935 of 1,390 columns had no finite row at all in the explorable
window and left NO line in the report, because the loop skipped them. A reader
could not tell "measured, found nothing" from "never reached a measurement" --
which is the whole point of the report (REGISTER #293, CLAIMS R62).

The discriminator is TIME, and these tests pin it: a real market-wide series
moves date to date and just moves for everyone at once; an absent one does not
move at all.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

_spec = importlib.util.spec_from_file_location(
    "leading_feature_report",
    PROJECT_ROOT / "scripts/diagnostics/leading_feature_report.py")
REPORT = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(REPORT)

NAMES = 20
DATES = 400


def _book():
    """A panel shaped the way `_examine` reads it: one row per name per date."""
    stamps = pd.date_range("2010-01-01", periods=DATES, freq="B")
    frame = pd.DataFrame({
        "ticker": np.tile([f"T{i}" for i in range(NAMES)], DATES),
        "datetime": np.repeat(stamps.to_numpy(), NAMES),
    })
    generator = np.random.default_rng(11)
    outcome = generator.standard_normal(len(frame)) * 0.02
    cut = frame["datetime"].quantile(0.70)
    return frame, {
        "outcome": outcome,
        "outcome_demeaned": REPORT._demean(outcome, frame["ticker"].to_numpy()),
        "is_train": (frame["datetime"] <= cut).to_numpy(),
        "dates": frame["datetime"],
        "tickers": frame["ticker"].to_numpy(),
        "horizon": 5,
    }


def _verdict_for(values: np.ndarray) -> str:
    frame, book = _book()
    row = pd.Series(REPORT._examine("probe", values, book))
    row["passes_fdr"] = True
    return REPORT._verdict(row)


def test_a_filled_default_is_not_called_market_wide():
    """The 67-column defect: one value everywhere, described as a series."""
    frame, _ = _book()
    values = np.zeros(len(frame))
    verdict = _verdict_for(values)
    assert verdict == "one value everywhere: a filled default", verdict
    assert "interaction" not in verdict, (
        "a column holding a single value was described as usable as an "
        "interaction term. An interaction with a constant is the constant.")


def test_a_filled_default_of_any_value_is_caught_not_just_zero():
    """`sentiment_pos_threshold_1d` is filled with a threshold, not with 0."""
    frame, _ = _book()
    assert _verdict_for(np.full(len(frame), 0.35)) == \
        "one value everywhere: a filled default"


def test_a_genuine_macro_series_is_still_market_wide():
    """The check must not destroy the case it was built beside.

    FRED_ICSA is one number per date for every name. That IS market-wide and
    the old verdict was right about it. If this test fails, the fix has taken
    the true positives with the false ones.
    """
    frame, _ = _book()
    per_date = np.random.default_rng(13).standard_normal(DATES)
    values = np.repeat(per_date, NAMES)
    assert _verdict_for(values) == "market-wide: use as interaction"


def test_a_column_with_no_finite_row_says_so():
    frame, _ = _book()
    assert _verdict_for(np.full(len(frame), np.nan)) == \
        "absent before the seal: nothing to judge"


def test_a_cross_sectional_feature_is_untouched_by_any_of_this():
    """A column that varies between names must reach the real checks."""
    frame, _ = _book()
    values = np.random.default_rng(17).standard_normal(len(frame))
    verdict = _verdict_for(values)
    assert verdict not in {
        "market-wide: use as interaction",
        "one value everywhere: a filled default",
        "absent before the seal: nothing to judge",
        "flat in both directions: nothing to judge",
    }, f"a varying column was disqualified as absent: {verdict}"


def test_the_examine_pass_reports_movement_in_time_separately():
    """`varies` and `time_sd` must be two different measurements.

    They were one before: `varies` alone decided the verdict, and it cannot
    tell a series that moves for everyone at once from one that never moves.
    """
    frame, book = _book()
    per_date = np.random.default_rng(19).standard_normal(DATES)
    macro = REPORT._examine("macro", np.repeat(per_date, NAMES), book)
    default = REPORT._examine("default", np.zeros(len(frame)), book)
    assert macro["varies"] < 0.05 and default["varies"] < 0.05, (
        "the premise of the test is gone: these two are supposed to be "
        "indistinguishable on the OLD measurement")
    assert macro["time_sd"] > 0 and default["time_sd"] == 0, (
        f"time_sd macro {macro['time_sd']}, default {default['time_sd']} -- "
        "the new measurement does not separate them either")


@pytest.mark.parametrize("verdict", [
    "absent before the seal: nothing to judge",
    "one value everywhere: a filled default",
    "flat in both directions: nothing to judge",
    "market-wide: use as interaction",
])
def test_every_verdict_the_report_can_emit_has_a_meaning_in_the_catalogue(verdict):
    """The catalogue prints `MEANING.get(verdict, '')` -- a missing key is a
    silent blank cell in docs/FEATURE_ROLES.md, which is how a new verdict
    would arrive unexplained."""
    catalogue_spec = importlib.util.spec_from_file_location(
        "feature_catalogue",
        PROJECT_ROOT / "scripts/diagnostics/feature_catalogue.py")
    catalogue = importlib.util.module_from_spec(catalogue_spec)
    catalogue_spec.loader.exec_module(catalogue)
    assert catalogue.MEANING.get(verdict), (
        f"the report can emit {verdict!r} and the catalogue has no meaning "
        "for it, so it would appear in FEATURE_ROLES.md with an empty cell.")
