import pandas as pd

from src.analytics.context.market_phase_analyzer import MarketPhaseAnalyzer


def _mk_df(**cols):
    return pd.DataFrame(cols)


def test_condition_and_or_and_operator_precedence():
    analyzer = MarketPhaseAnalyzer(
        config={
            "indicators": {"gdp": "gdp", "vix": "vix", "infl": "infl"},
            "rules": [
                {"condition": "gdp >= 0.5 and vix < 20", "phase": "risk_on"},
                {"condition": "vix >= 30 or infl > 3.0", "phase": "risk_off"},
            ],
        }
    )

    # rule 1 matches
    data = {"market_data": _mk_df(gdp=[0.6], vix=[19.0], infl=[2.0])}
    assert analyzer.analyze(data)["market_phase"] == "risk_on"

    # rule 2 matches via OR (left side false, right side true)
    data = {"market_data": _mk_df(gdp=[0.4], vix=[25.0], infl=[3.5])}
    assert analyzer.analyze(data)["market_phase"] == "risk_off"


def test_unknown_when_condition_invalid_or_missing_values():
    analyzer = MarketPhaseAnalyzer(
        config={
            "indicators": {"gdp": "gdp"},
            "rules": [{"condition": "__import__('os').system('echo nope')", "phase": "x"}],
        }
    )
    data = {"market_data": _mk_df(gdp=[1.0])}
    assert analyzer.analyze(data)["market_phase"] == "unknown"

