import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd

from src.pipeline.stages.evaluation import analytics as eval_analytics
from src.pipeline.stages.evaluation import io as eval_io
from src.pipeline.stages.evaluation import reporting as eval_reporting


class DummyBacktester:
    async def run(self, price_pivot, signals):
        return {
            'performance': {'total_return_pct': 0.05},
            'portfolio_history': pd.DataFrame({'total_value': [100.0, 105.0], 'returns': [0.0, 0.05]}, index=pd.date_range('2026-01-01', periods=2))
        }


def test_create_evaluation_summary_returns_expected_keys():
    summary = eval_analytics.create_evaluation_summary(
        financial_metrics={'total_return_pct': 0.05},
        backtest_results={'performance': {'sharpe_ratio': 1.2}},
        analysis_results={'risk_report': {'max_drawdown': 0.03}},
    )

    assert 'metrics' in summary
    assert 'backtest_stats' in summary
    assert 'analysis' in summary
    assert 'timestamp' in summary
    assert summary['metrics']['total_return_pct'] == 0.05
    assert summary['backtest_stats']['sharpe_ratio'] == 1.2


def test_plot_equity_curve_creates_file(tmp_path):
    portfolio_history = pd.DataFrame(
        {'total_value': [100.0, 105.0]},
        index=pd.date_range('2026-01-01', periods=2),
    )
    metrics = {'total_return_pct': 0.05}
    output = eval_reporting.plot_equity_curve(portfolio_history, metrics)

    assert output.exists()
    assert output.suffix == '.png'
    assert output.parent.name == 'charts'


def test_save_evaluation_summary_async_writes_json(tmp_path):
    summary = {'metrics': {'total_return_pct': 0.05}}
    path = tmp_path / 'evaluation_summary.json'

    asyncio.run(eval_io.save_evaluation_summary_async(path, summary))

    assert path.exists()
    loaded = json.loads(path.read_text(encoding='utf-8'))
    assert loaded['metrics']['total_return_pct'] == 0.05


def test_run_backtest_converts_simple_signals_and_returns_dict():
    from src.pipeline.stages.evaluation.backtest_adapter import run_backtest

    signals_df = pd.DataFrame(
        {
            'timestamp': pd.date_range('2026-01-01', periods=2),
            'ticker': ['AAPL', 'AAPL'],
            'price': [100.0, 101.0],
            'signal': ['BUY', 'SELL'],
        }
    )
    backtester = DummyBacktester()

    result = asyncio.run(run_backtest(backtester, signals_df))
    assert isinstance(result, dict)
    assert 'performance' in result
    assert 'portfolio_history' in result
