"""Stage 7 threw away the only out-of-sample evidence it had.

`run()` asks `can_run_backtest(signals_df)` first. That check is about Stage
5's live signals, which are one bar per context and carry no price series,
so it refuses -- correctly. The whole comprehensive branch was then skipped,
and with it `_holdout_equity`, which reads a completely different artifact.

The 2026-08-12 run ended with:

    total_signals: 47
    trades_executed: 0
    portfolio_value: 0

while data/results/holdout_predictions_20260812_191514.parquet held 11,047
bars across 65 contexts that the models never saw. The holdout curve needs
no price series at all -- each row carries its own realised value, and for a
return target that value IS the return. Nothing about the failing precondition
applied to it.

So the basic evaluation now carries the holdout result too, including when
it cannot be built: 'no_return_targets' is an answer, and on that run it was
the answer -- none of the 65 surviving champions predicted a return.
"""
import logging

import pandas as pd
import pytest

from src.pipeline.stages.evaluation.orchestrator import EvaluationStage


class _ReportGen:
    def __init__(self):
        self.saved = []

    def save_summary(self, summary, results_dir):
        self.saved.append(summary)


def _stage(holdout_result):
    stage = object.__new__(EvaluationStage)
    stage.logger = logging.getLogger("EvaluationStageTest")
    stage.report_gen = _ReportGen()
    stage.results_dir = "data/results"
    stage._holdout_equity = lambda signals_data: dict(holdout_result)
    return stage


def _signals_data():
    return {'trading_activity': [], 'portfolio_summary': {}}


def test_a_built_curve_reaches_the_summary_even_without_a_backtest():
    history = pd.DataFrame(
        {'total_value': [100_000.0, 100_500.0, 101_000.0]},
        index=pd.date_range('2026-05-01', periods=3, freq='D', tz='UTC'),
    )
    stage = _stage({
        'status': 'built',
        'bar_count': 3,
        'portfolio_history': history,
        'returns': pd.Series([0.0, 0.005, 0.00497]),
        '_frame': pd.DataFrame({'x': [1]}),
    })

    result = stage._create_basic_evaluation(pd.DataFrame({'a': [1, 2]}), _signals_data())
    summary = result['evaluation_summary']

    assert summary['holdout_equity']['status'] == 'built'
    assert summary['holdout_equity']['bar_count'] == 3
    assert summary['holdout_equity']['final_value'] == pytest.approx(101_000.0)
    # The raw frame is working state, not a result; it must not be serialised
    # into the summary alongside it.
    assert '_frame' not in summary['holdout_equity']
    assert 'portfolio_history' not in summary['holdout_equity']


def test_the_reason_survives_when_no_curve_can_be_built():
    """'no_return_targets' is a finding, not an absence.

    On the 2026-08-12 run every one of the 65 champions sat on a
    classification target -- volatility spikes, breakouts, volume ratios --
    so there was no return series to compound. Reporting that is the point;
    a missing key would read as "not attempted".
    """
    stage = _stage({'status': 'no_return_targets'})

    result = stage._create_basic_evaluation(pd.DataFrame({'a': [1]}), _signals_data())

    assert result['evaluation_summary']['holdout_equity'] == {
        'status': 'no_return_targets'
    }


def test_the_original_metrics_are_not_disturbed():
    stage = _stage({'status': 'no_holdout_artifact'})

    summary = stage._create_basic_evaluation(
        pd.DataFrame({'a': [1, 2, 3]}), _signals_data()
    )['evaluation_summary']

    assert summary['metrics']['total_signals'] == 3
    assert summary['notification_status'] == 'basic_evaluation_not_sent'
