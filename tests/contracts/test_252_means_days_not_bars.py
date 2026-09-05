""""252" is a number of DAYS, and it was being used as a number of BARS.

REGISTER #183, filed as a data defect, closed prematurely, and reopened on
2026-09-04 when `stale_state_scan.py` rule G noticed the row ended on
"Виправляти їх треба тим самим `infer_periods_per_year`" -- future tense, and
the code agreed with the tense.

The project runs 15m, 60m and 1d frames side by side. A flat `sqrt(252)`
understates a 15-minute Sharpe by sqrt(26) -- a factor of 5.1 -- and a flat
`/ 252` overstates the per-period risk-free rate by 26. Both appeared, three
lines apart, in the same function.

WORSE THAN THE SCALE, in the feature enricher: `window = 252` was a rolling
window of 252 BARS. On the daily frame that is a year. On 15-minute bars it is
about a day and a half, so a column named SHARPE_RATIO measured something else
entirely on two of the three frames -- not a mis-scaled year, a different
question.

THE FIX HAD TO START ELSEWHERE. `infer_periods_per_year` fell back to 252 in
silence when handed a series without a DatetimeIndex, so replacing constants
with calls to it would have changed nothing while making the code READ as
cadence-aware -- worse than the constant, which at least admitted what it was.
The fallback is now audible: it warns once per call site, naming the site and
the reason, on any series long enough to have shown a cadence. Same invariant
as #182 -- a default must be accompanied by something saying it was a default.
"""
from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.metrics.financial import financial_metrics_library as lib
from src.metrics.utils.calculation_tools import calculate_rolling_volatility

ROOT = Path(__file__).resolve().parents[2]

#: Every file that carried a live hard-coded 252 on 2026-09-05, and the
#: enricher that had already been fixed and must stay fixed.
CLEANED = (
    "src/features/enrichers/technical_analysis_enricher.py",
    "src/features/enrichers/volatility_enricher.py",
    "src/features/analysis/market_conditions_analyzer.py",
    "src/models/analysis/regime/detector.py",
    "src/metrics/utils/calculation_tools.py",
    "src/analytics/analyzers/risk_decomposition_analyzer.py",
    "src/analytics/analyzers/performance_attribution_analyzer.py",
    "src/simulation/simulation_engine.py",
)


def _series(freq: str, n: int = 300, seed: int = 0) -> pd.Series:
    rng = np.random.default_rng(seed)
    return pd.Series(rng.normal(scale=0.01, size=n),
                     index=pd.date_range("2020-01-03", periods=n, freq=freq))


@pytest.mark.parametrize("freq,expected", [
    ("D", 252),
    ("15min", 252 * 26),
    ("h", 252 * 7),
    ("W", 52),
])
def test_the_cadence_is_read_off_the_index(freq, expected):
    value, reason = lib.periods_per_year_with_reason(_series(freq))
    assert value == expected
    assert reason.startswith("inferred"), reason


def test_a_series_without_an_index_says_it_defaulted():
    value, reason = lib.periods_per_year_with_reason(
        pd.Series(np.zeros(300)))
    assert value == 252
    assert reason.startswith("default"), (
        "the fallback does not report itself, so a caller cannot tell an "
        "inferred 252 from a defaulted one -- which is the whole reason "
        "replacing the constants was not enough"
    )


def test_the_fallback_warns_once_and_names_the_caller(monkeypatch):
    said: list[str] = []

    class _Recorder:
        def warning(self, message, *args):
            said.append(message % args if args else message)

        def __getattr__(self, name):
            return lambda *a, **k: None

    monkeypatch.setattr(lib, "logger", _Recorder())
    monkeypatch.setattr(lib, "_WARNED_FALLBACKS", set())

    plain = pd.Series(np.zeros(300))
    # From ONE line, repeatedly -- which is the real shape: a rolling loop
    # calling the same statement thousands of times. The key is (call site,
    # reason), so two DIFFERENT lines warning twice is correct and this test
    # would have been asserting the wrong thing had it kept them apart.
    for _ in range(50):
        lib.infer_periods_per_year(plain)

    assert len(said) == 1, (
        f"the fallback warned {len(said)} times; inside a rolling loop that "
        "floods the log, and a message that floods is one that gets filtered"
    )
    assert "#183" in said[0]
    assert "252" in said[0]


def test_a_short_series_does_not_warn(monkeypatch):
    """A five-row series honestly cannot show its cadence. Warning there
    would train the reader to ignore the warning."""
    said: list[str] = []
    monkeypatch.setattr(lib, "logger",
                        type("R", (), {"warning": lambda self, *a: said.append(a),
                                       "__getattr__": lambda self, n: (lambda *a, **k: None)})())
    monkeypatch.setattr(lib, "_WARNED_FALLBACKS", set())
    lib.infer_periods_per_year(pd.Series(np.zeros(5)))
    assert not said


def test_the_same_returns_annualise_differently_by_cadence():
    """The measurement that makes the defect visible: identical numbers, two
    cadences, and the answer must differ by sqrt(bars per day)."""
    daily = calculate_rolling_volatility(_series("D", seed=1))
    intraday = calculate_rolling_volatility(_series("15min", seed=1))
    ratio = float(intraday.iloc[-1] / daily.iloc[-1])
    assert ratio == pytest.approx(np.sqrt(26), rel=0.02), (
        f"the two cadences annualise to a ratio of {ratio:.2f}; before the fix "
        "they returned the same number"
    )


@pytest.mark.parametrize("path", CLEANED)
def test_no_hard_coded_252_survives(path):
    """Comments and docstrings may say 252 -- several explain why it was
    wrong. Executable lines may not."""
    text = (ROOT / path).read_text(encoding="utf-8")
    offenders = []
    in_docstring = False
    for number, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if stripped.count('"""') % 2 == 1:
            in_docstring = not in_docstring
            continue
        if in_docstring or stripped.startswith("#"):
            continue
        code = line.split("#", 1)[0]
        if "sqrt(252)" in code or "* 252" in code or "/ 252" in code:
            offenders.append(f"{path}:{number}: {stripped[:90]}")
    assert not offenders, "\n".join(offenders)


def test_the_enricher_sizes_its_window_in_bars_of_a_year():
    """The half that is not about scale: 252 BARS is a year of daily bars and
    a day and a half of 15-minute ones."""
    from src.features.enrichers import technical_analysis_enricher

    source = inspect.getsource(technical_analysis_enricher)
    assert "window = periods" in source, (
        "the rolling window is still a flat 252 bars, so SHARPE_RATIO on the "
        "intraday frames measures roughly a day and a half and is named for a "
        "year"
    )
    assert "np.sqrt(periods)" in source
