"""market_phase was the constant 'neutral' in every export ever produced.

The four rules that classify a phase compared the market regime to 0 and 1:

    volatility < 0.02 and regime == 0   -> calm_bull
    volatility < 0.02 and regime == 1   -> calm_bear
    ...
    True                                -> neutral

But MARKET_REGIME is a WORD — TRENDING_UP, TRENDING_DOWN, RANGING,
MEAN_REVERSION, NORMAL. So 'RANGING' == 0 is false, all four rules failed on
every row, each fell through to the catch-all, and the feature arrived at the
models as the single value 4 on all three timeframes.

Underneath the rules sat the reason they could not have been written any other
way: `_eval_factor` did `rhs_val = float(rhs)` and returned False on
ValueError, so the condition evaluator supported numbers only. A rule saying
`regime == 'TRENDING_UP'` would have been rejected just as silently. Equality
between words is now defined; ordering between them is not, because asking
whether 'RANGING' > 'NORMAL' has no meaning and answering alphabetically would
be worse than refusing.

RANGING and MEAN_REVERSION still map to neutral, and that is the right answer
rather than a failure to classify: they are genuinely neither bull nor bear.
"""
import numpy as np
import pandas as pd
import pytest

from src.analytics.context.market_phase_analyzer import MarketPhaseAnalyzer
from src.features.enrichers.advanced_analytics_enricher import (
    AdvancedAnalyticsEnricher,
)

_CODES = {"calm_bull": 0, "calm_bear": 1, "volatile_bull": 2,
          "volatile_bear": 3, "neutral": 4}


@pytest.fixture
def enricher():
    return AdvancedAnalyticsEnricher()


def _frame(n: int = 200) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "ticker": ["AAPL"] * n,
        "datetime": pd.date_range("2026-01-01", periods=n, freq="D", tz="UTC"),
        "close": np.linspace(100, 140, n),
        "VOLATILITY_20": rng.choice([0.005, 0.05], n),
        "SMA_50": np.linspace(99, 138, n),
        "MARKET_REGIME": rng.choice(
            ["TRENDING_UP", "TRENDING_DOWN", "RANGING", "MEAN_REVERSION"], n
        ),
    })


def test_the_phase_is_no_longer_one_value(enricher):
    df = _frame()

    enricher._add_market_phase_detection(df)

    assert pd.to_numeric(df["market_phase"]).nunique() > 1, (
        "every export this project produced held the single value 4"
    )


def test_a_calm_uptrend_is_a_calm_bull(enricher):
    df = _frame()

    enricher._add_market_phase_detection(df)

    phase = pd.to_numeric(df["market_phase"])
    calm_up = (df["MARKET_REGIME"] == "TRENDING_UP") & (df["VOLATILITY_20"] < 0.02)
    assert calm_up.sum() > 0
    assert (phase[calm_up] == _CODES["calm_bull"]).all()


def test_a_volatile_downtrend_is_a_volatile_bear(enricher):
    df = _frame()

    enricher._add_market_phase_detection(df)

    phase = pd.to_numeric(df["market_phase"])
    wild_down = (df["MARKET_REGIME"] == "TRENDING_DOWN") & (df["VOLATILITY_20"] >= 0.02)
    assert wild_down.sum() > 0
    assert (phase[wild_down] == _CODES["volatile_bear"]).all()


def test_ranging_stays_neutral_on_purpose(enricher):
    """Not a failure to classify: a range really is neither direction."""
    df = _frame()

    enricher._add_market_phase_detection(df)

    phase = pd.to_numeric(df["market_phase"])
    sideways = df["MARKET_REGIME"].isin(["RANGING", "MEAN_REVERSION"])
    assert (phase[sideways] == _CODES["neutral"]).all()


# --- the evaluator underneath -------------------------------------------


@pytest.fixture
def analyzer():
    return MarketPhaseAnalyzer({"indicators": {}, "rules": []})


def test_a_word_can_be_compared_for_equality(analyzer):
    values = {"regime": "TRENDING_UP"}

    assert analyzer._eval_factor("regime == 'TRENDING_UP'", values) is True
    assert analyzer._eval_factor("regime == 'RANGING'", values) is False
    assert analyzer._eval_factor("regime != 'RANGING'", values) is True


def test_quoting_style_and_case_do_not_decide_the_answer(analyzer):
    values = {"regime": "trending_up"}

    assert analyzer._eval_factor('regime == "TRENDING_UP"', values) is True


def test_ordering_between_words_is_refused_not_answered(analyzer):
    """'RANGING' > 'NORMAL' has no meaning; alphabetical order is not one."""
    values = {"regime": "RANGING"}

    assert analyzer._eval_factor("regime > 'NORMAL'", values) is False
    assert analyzer._eval_factor("regime < 'NORMAL'", values) is False


def test_a_number_still_compares_as_a_number(analyzer):
    values = {"volatility": 0.05}

    assert analyzer._eval_factor("volatility >= 0.02", values) is True
    assert analyzer._eval_factor("volatility < 0.02", values) is False


def test_a_word_against_a_number_is_false_rather_than_a_crash(analyzer):
    values = {"regime": "TRENDING_UP"}

    assert analyzer._eval_factor("regime >= 0.02", values) is False


def test_the_whitelist_admits_a_quoted_regime(analyzer):
    """The rule was rejected before it was ever evaluated."""
    assert analyzer._is_valid_condition_string(
        "volatility < 0.02 and regime == 'TRENDING_UP'"
    )
