import numpy as np
import pandas as pd

from src.archive.risk.metrics import (
    analyze_market_conditions,
    calculate_portfolio_metrics,
    calculate_portfolio_returns,
    calculate_position_metrics,
)


def make_market_data(symbols, days=10):
    idx = pd.date_range(end=pd.Timestamp.today(), periods=days)
    close = pd.DataFrame({s: 100 + np.arange(days) for s in symbols}, index=idx)
    return {'close': close}


def test_empty_portfolio_returns_empty():
    market_data = {'close': pd.DataFrame()}
    assert calculate_portfolio_returns({}, market_data) == []


def test_simple_metrics_flow():
    symbols = ['AAA', 'BBB']
    market_data = make_market_data(symbols, days=8)
    portfolio = {
        'AAA': {'current_value': 100.0},
        'BBB': {'current_value': 200.0}
    }

    returns = calculate_portfolio_returns(portfolio, market_data)
    assert isinstance(returns, list)
    assert len(returns) > 0

    pmetrics = calculate_portfolio_metrics(portfolio, market_data)
    assert 'portfolio_value' in pmetrics
    assert pmetrics['portfolio_value'] == 300.0
    assert pmetrics['max_drawdown'] <= 0.0
    assert pmetrics['max_drawdown_signed'] <= 0.0
    assert pmetrics['max_drawdown_pct'] >= 0.0
    assert pmetrics['current_drawdown_pct'] >= 0.0

    pos_metrics = calculate_position_metrics(portfolio, market_data)
    assert set(pos_metrics.keys()) == set(symbols)
    for metrics in pos_metrics.values():
        assert metrics['max_drawdown_signed'] <= 0.0
        assert metrics['max_drawdown_pct'] >= 0.0

    mc = analyze_market_conditions(market_data)
    assert 'volatility_regime' in mc
