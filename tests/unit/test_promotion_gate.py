"""Stage 4 may only write CHAMP_ for a model measured outside its own selection.

Two defects these tests pin down, both reported by the Codex audit (§2.2) and
both confirmed against real artifacts before being fixed:

1. `_record_winner_test_score` read `data['X_test']`, and Stage 4's
   orchestrator supplies the VALIDATION split under that key. So the winner
   was re-scored on the rows that had just chosen it, and the result was
   published as a test metric.
2. Promotion was an unconditional `shutil.copy2`. Whichever candidate scored
   highest became the champion Stage 5 loads, however badly it did.
"""
import numpy as np
import pandas as pd
import pytest

from src.training.base_trainer import BaseTrainer


class _Host(BaseTrainer):
    """Minimal concrete BaseTrainer that skips the heavy __init__."""

    def __init__(self, gate_cfg=None):
        from src.metrics.model.ml_evaluator import MLEvaluator
        self.evaluator = MLEvaluator()
        self.config_manager = _StubConfig(gate_cfg)
        self.logger = _NullLogger()

    def train(self, *args, **kwargs):  # pragma: no cover - abstract stub
        raise NotImplementedError

    def _prepare_ticker_groups(self, *args, **kwargs):  # pragma: no cover
        raise NotImplementedError

    def _train_ticker_group(self, *args, **kwargs):  # pragma: no cover
        raise NotImplementedError


class _StubConfig:
    def __init__(self, gate_cfg):
        self._gate = {} if gate_cfg is None else gate_cfg

    def get(self, key, default=None):
        if key == 'training.promotion_gate':
            return self._gate
        return default


class _NullLogger:
    def warning(self, *a, **k):
        pass

    def info(self, *a, **k):
        pass

    def error(self, *a, **k):
        pass

    def debug(self, *a, **k):
        pass

    def isEnabledFor(self, _level):
        return False


class _ConstantModel:
    """Predicts one value regardless of input — the thing a gate must reject."""

    def __init__(self, value):
        self.value = value

    def predict(self, X):
        return np.full(len(X), self.value)


class _PerfectModel:
    def __init__(self, answers):
        self.answers = np.asarray(answers)

    def predict(self, X):
        return self.answers[: len(X)]


def _classification_data(n=60):
    rng = np.random.default_rng(11)
    y_train = np.array([0] * 40 + [1] * 20)
    y_holdout = np.array([0, 1] * (n // 2))[:n]
    return {
        'X_train': pd.DataFrame({'f': rng.normal(size=60)}),
        'y_train': y_train,
        'X_holdout': pd.DataFrame({'f': rng.normal(size=n)}),
        'y_holdout': y_holdout,
    }


def test_absent_holdout_is_reported_not_scored():
    """No holdout must produce no score — not a zero that reads as measured."""
    host = _Host()
    results = {}
    data = {'X_train': pd.DataFrame({'f': [1, 2, 3]}), 'y_train': np.array([0, 1, 0])}

    host._record_winner_test_score(_ConstantModel(1), data, True, results)

    metrics = results['winner_holdout_metrics']
    assert metrics['status'] == 'no_holdout_available'
    assert 'score' not in metrics
    assert host._evaluate_promotion_gate(results)['passed'] is False


def test_gate_rejects_a_model_that_cannot_beat_the_naive_baseline():
    host = _Host()
    data = _classification_data()
    results = {}

    # Predicts the majority class of train — exactly the baseline.
    host._record_winner_test_score(_ConstantModel(0), data, True, results)

    metrics = results['winner_holdout_metrics']
    assert metrics['status'] == 'measured'
    assert metrics['baseline_score'] is not None
    assert metrics['score'] == pytest.approx(metrics['baseline_score'])

    gate = host._evaluate_promotion_gate(results)
    assert gate['passed'] is False
    assert any('naive baseline' in r for r in gate['reasons'])


def test_gate_admits_a_model_that_beats_the_baseline():
    host = _Host()
    data = _classification_data()
    results = {}

    # Better than the majority-class baseline, but nowhere near perfect —
    # a plausible edge, which is the only thing that should be promoted.
    noisy = data['y_holdout'].copy()
    noisy[::5] = 1 - noisy[::5]

    host._record_winner_test_score(_PerfectModel(noisy), data, True, results)

    metrics = results['winner_holdout_metrics']
    assert metrics['score'] > metrics['baseline_score']
    assert metrics['score'] < host.IMPLAUSIBLE_SCORE
    assert host._evaluate_promotion_gate(results)['passed'] is True


def test_gate_rejects_a_holdout_too_small_to_act_on():
    host = _Host()
    data = _classification_data(n=6)
    results = {}

    host._record_winner_test_score(_PerfectModel(data['y_holdout']), data, True, results)
    gate = host._evaluate_promotion_gate(results)

    assert gate['passed'] is False
    assert any('minimum is 20' in r for r in gate['reasons'])


def test_regression_baseline_is_the_train_mean():
    """R2 of the train mean on the holdout is the bar a regressor must clear."""
    host = _Host()
    rng = np.random.default_rng(5)
    y_holdout = rng.normal(size=50)
    data = {
        'X_train': pd.DataFrame({'f': rng.normal(size=50)}),
        'y_train': np.full(50, 0.0),
        'X_holdout': pd.DataFrame({'f': rng.normal(size=50)}),
        'y_holdout': y_holdout,
    }
    results = {}

    host._record_winner_test_score(_ConstantModel(0.0), data, False, results)

    metrics = results['winner_holdout_metrics']
    assert metrics['metric'] == 'R2'
    # Winner and baseline are the same constant, so neither wins.
    assert metrics['score'] == pytest.approx(metrics['baseline_score'])
    assert host._evaluate_promotion_gate(results)['passed'] is False


def test_holdout_predictions_are_kept_with_their_timestamps():
    """The only out-of-sample time series the pipeline produces must survive.

    Stage 7 builds its equity curve from Stage 5, which predicts the LATEST
    bar of each context — 540 predictions pivoted to a (3, 22) table. Three
    time points, hence Sharpe -329.82 on volatility 8.46e-05. The holdout is
    ~100-220 purged bars per context that the model never saw; those rows,
    with their timestamps, are what an honest curve needs.
    """
    host = _Host()
    index = pd.date_range("2026-03-01", periods=40, freq="D", tz="UTC")
    data = _classification_data(n=40)
    data['X_holdout'] = pd.DataFrame(
        {'f': np.arange(40.0)}, index=index
    )
    data['y_holdout'] = data['y_holdout'][:40]
    results = {}

    host._record_winner_test_score(_PerfectModel(data['y_holdout']), data, True, results)

    series = results['winner_holdout_predictions']
    assert len(series) == 40
    assert series[0]['datetime'].startswith('2026-03-01')
    assert series[-1]['datetime'].startswith('2026-04-09')
    assert {'datetime', 'prediction', 'actual'} == set(series[0])
    # A real series, not three points.
    assert len({row['datetime'] for row in series}) == 40


def test_holdout_predictions_survive_a_missing_index():
    host = _Host()
    data = _classification_data(n=30)
    data['X_holdout'] = np.zeros((30, 3))  # no index at all
    results = {}

    host._record_winner_test_score(_PerfectModel(data['y_holdout'][:30]), data, True, results)

    series = results['winner_holdout_predictions']
    assert len(series) == 30
    assert all(row['datetime'] is None for row in series)


def test_a_persistent_series_must_be_beaten_by_persistence_not_by_the_mean():
    """The bar for a slow-moving target is "tomorrow equals today", not the mean.

    Seven indicator-prediction targets produced 138 of 354 champions because
    the only opponent was the train mean. Measured on the real batch,
    persistence alone scores R2 0.9994 on target_sma_20_f1 and 0.9984 on
    target_bb_upper_f1 — a model at 0.998 there has added nothing, while
    against the mean it looked like a triumph.
    """
    host = _Host()
    # A drifting series: the mean is a terrible predictor, persistence is
    # nearly perfect — exactly the shape of SMA_20 one bar ahead.
    n = 80
    trend = np.linspace(100.0, 140.0, n) + np.sin(np.arange(n) / 5.0) * 0.05
    data = {
        'X_train': pd.DataFrame({'f': trend[:40]}),
        'y_train': trend[:40],
        'X_holdout': pd.DataFrame({'f': trend[40:]}),
        'y_holdout': trend[40:],
    }

    scores = host._score_naive_baselines(data, False, 'regression', 'R2')

    assert scores['baseline_persistence_score'] > scores['baseline_constant_score']
    # The published bar is the stronger opponent.
    assert scores['baseline_score'] == scores['baseline_persistence_score']
    assert scores['baseline_kind'] == 'persistence'
    assert scores['baseline_score'] > 0.9


def test_the_constant_still_wins_when_the_series_has_no_momentum():
    """On noise, persistence is bad and the mean is the honest bar."""
    host = _Host()
    rng = np.random.default_rng(4)
    noise = rng.normal(size=120)
    data = {
        'X_train': pd.DataFrame({'f': noise[:60]}),
        'y_train': noise[:60],
        'X_holdout': pd.DataFrame({'f': noise[60:]}),
        'y_holdout': noise[60:],
    }

    scores = host._score_naive_baselines(data, False, 'regression', 'R2')

    assert scores['baseline_kind'] == 'constant'
    assert scores['baseline_persistence_score'] < scores['baseline_constant_score']


def test_classification_keeps_the_majority_class_bar_only():
    host = _Host()
    data = _classification_data()

    scores = host._score_naive_baselines(data, True, 'classification', 'F1')

    assert scores['baseline_kind'] == 'constant'
    assert 'baseline_persistence_score' not in scores


def test_gate_can_be_disabled_to_restore_the_old_behaviour():
    host = _Host(gate_cfg={'enabled': False})
    assert host._evaluate_promotion_gate({})['passed'] is True


def test_a_near_perfect_score_blocks_promotion_instead_of_earning_it():
    """Codex §8.7: R2 ~0.99 on market data is a leakage alarm, not a champion."""
    host = _Host()
    data = _classification_data()
    results = {'training_sanity': {'blocking': [], 'warnings': []}}

    host._record_winner_test_score(_PerfectModel(data['y_holdout']), data, True, results)
    assert results['winner_holdout_metrics']['score'] == pytest.approx(1.0)

    gate = host._evaluate_promotion_gate(results)
    assert gate['passed'] is False
    assert any('implausibly high' in r for r in gate['reasons'])


def test_more_features_than_rows_warns_by_default_and_blocks_when_configured():
    """The ratio is reported either way; whether it refuses is the operator's.

    Measured on the 2026-08-06 batch, every context is over-parameterised
    (AAPL 1d 327 features / ~196 train rows), so a blocking default would stop
    the light branch producing champions entirely. The holdout-versus-baseline
    comparison is the empirical overfitting check; this is a heuristic.
    """
    payload = {
        'X_train': pd.DataFrame(np.zeros((30, 40))),
        'feature_names': [f'f{i}' for i in range(40)],
    }

    lenient = _Host()._check_training_sanity(payload)
    assert not lenient['blocking']
    assert any('40 features on 30 training rows' in w for w in lenient['warnings'])

    strict_host = _Host(gate_cfg={'block_when_features_exceed_rows': True})
    strict = strict_host._check_training_sanity(payload)
    assert strict['blocking']
    assert '40 features on 30 training rows' in strict['blocking'][0]
    assert strict_host._evaluate_promotion_gate(
        {'training_sanity': strict}
    )['passed'] is False


def test_no_training_rows_always_blocks():
    sanity = _Host()._check_training_sanity({'X_train': pd.DataFrame()})
    assert sanity['blocking'] == ['no training rows']


def test_a_high_but_survivable_feature_ratio_only_warns():
    host = _Host()
    sanity = host._check_training_sanity({
        'X_train': pd.DataFrame(np.zeros((192, 237))),
        'feature_names': [f'f{i}' for i in range(237)],
        'X_val': pd.DataFrame(np.zeros((19, 237))),
    })

    # 237 features on 192 rows is roughly the project's own measured shape.
    assert any('237 features on 192 training rows' in w for w in sanity['warnings'])
    assert not sanity['blocking']

    # 100 features on 200 rows: fittable, so not blocking, but well past the
    # one-third mark, so it must still be said out loud.
    ok = host._check_training_sanity({
        'X_train': pd.DataFrame(np.zeros((200, 100))),
        'feature_names': [f'f{i}' for i in range(100)],
        'X_val': pd.DataFrame(np.zeros((60, 100))),
    })
    assert not ok['blocking']
    assert any('ratio' in w for w in ok['warnings'])

    # Comfortably below it: nothing to report.
    quiet = host._check_training_sanity({
        'X_train': pd.DataFrame(np.zeros((500, 30))),
        'feature_names': [f'f{i}' for i in range(30)],
        'X_val': pd.DataFrame(np.zeros((60, 30))),
    })
    assert not quiet['blocking']
    assert not any('ratio' in w for w in quiet['warnings'])
