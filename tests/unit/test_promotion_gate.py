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

    # Predicts one constant class. It cannot beat the best constant, which is
    # what the bar is now — always-zero used to BE the bar and tied with it,
    # scoring 0.0 against 0.0, until the negative control showed that a bar of
    # zero is no bar at all.
    host._record_winner_test_score(_ConstantModel(0), data, True, results)

    metrics = results['winner_holdout_metrics']
    assert metrics['status'] == 'measured'
    assert metrics['baseline_score'] is not None
    assert metrics['score'] <= metrics['baseline_score']

    gate = host._evaluate_promotion_gate(results)
    assert gate['passed'] is False
    # The refusal must NAME the rung that bound. "Does not beat the naive
    # baseline" was true of every refusal and told the reader nothing; losing
    # to a constant and losing to the clock are different diagnoses.
    refusal = next(r for r in gate['reasons']
                   if 'does not beat the' in r and 'baseline' in r)
    assert metrics['baseline_kind'] in refusal
    assert 'constant' in refusal and 'clock' in refusal


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
    # `probability` joined these three once it emerged that a hard 0/1 made a
    # coin flip and a near-certainty indistinguishable downstream. Still an
    # exact set: a key silently appearing or vanishing here is the failure
    # this line exists to catch.
    assert {'datetime', 'prediction', 'actual', 'probability'} == set(series[0])
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


def test_classification_now_meets_persistence_too():
    """Classification used to be exempt from the persistence opponent.

    It was scored for regression only, on the reasoning that "tomorrow equals
    today" is a statement about a continuous series. It is not: "this already
    happened h bars ago" is just as legitimate against a binary target, and on
    a persistent one it is a strong opponent that classification simply never
    met. Seven 15m champions passed this gate on 2026-08-30 and then lost to
    opponents run by hand on the same holdouts.

    On this fixture the holdout alternates 0,1,0,1..., so lag-1 predicts the
    opposite class every time and the constant still binds — which is the
    point: adding a rung must not change which rung wins when it should not.
    """
    host = _Host()
    data = _classification_data()

    scores = host._score_naive_baselines(data, True, 'classification', 'F1')

    assert scores['baseline_persistence_score'] is not None
    assert scores['baseline_kind'] == 'constant'
    assert scores['baseline_score'] == scores['baseline_constant_score']


def test_persistence_lags_inside_a_series_not_across_the_pooled_frame():
    """A pooled frame interleaves tickers; lagging by ROW crosses companies.

    With 22 names at one timestamp, `y[t-h]` is another company minutes
    earlier rather than this company h bars back, so the opponent measured
    something no one meant. `holdout_groups` carries the ticker per row.
    """
    host = _Host()
    horizon, names, bars = 3, list('ABCD'), 60
    rng = np.random.default_rng(2)
    # Each name's own series is exactly periodic with period h, so a correct
    # lag-h opponent reproduces it and a row-wise one cannot.
    series = {n: np.tile(rng.integers(0, 2, horizon), bars // horizon)
              for n in names}
    rows = [(b, n) for b in range(bars) for n in names]
    groups = np.array([n for _b, n in rows])
    y = np.array([series[n][b] for b, n in rows], dtype=float)

    grouped = host._persistence_prediction(
        y, y, horizon=horizon, groups=groups, is_classif=True)
    rowwise = host._persistence_prediction(
        y, y, horizon=horizon, groups=None, is_classif=True)

    tail = slice(horizon * len(names), None)
    assert (grouped[tail] == y[tail]).all()
    assert not (rowwise[tail] == y[tail]).all()


def test_the_clock_is_an_opponent_and_says_which_cut_of_it_won():
    """A target that is nothing but the hour of day must not pass the gate.

    The rung the gate was missing entirely. On intraday targets the effect is
    large and has nothing to do with any feature here: the first and last bars
    of a session carry most of the volatility and most of the volume, every
    day.
    """
    host = _Host()
    index = pd.date_range('2024-01-01', periods=12000, freq='h', tz='UTC')
    y = index.hour.isin([14, 15, 20]).astype(float)
    split = 9000
    data = {
        'X_train': pd.DataFrame(index=index[:split]),
        'y_train': y[:split],
        'X_holdout': pd.DataFrame(index=index[split:]),
        'y_holdout': y[split:],
    }

    scores = host._score_naive_baselines(data, True, 'classification', 'F1')

    assert scores['baseline_kind'] == 'clock'
    assert scores['baseline_clock_score'] == pytest.approx(1.0)
    assert scores['baseline_clock_scheme'] in ('hour', 'weekday_hour')
    assert scores['baseline_clock_score'] > scores['baseline_constant_score']


def test_the_clock_reports_nothing_when_it_cannot_be_measured():
    """An unmeasured opponent must be absent, never a passing zero."""
    host = _Host()
    plain = pd.DataFrame(index=range(50))
    assert host._clock_prediction(
        {'X_train': plain, 'X_holdout': plain}, np.zeros(50), True) == {}

    # Timestamps, but too few training rows for any bucket to earn an answer.
    index = pd.date_range('2024-01-01', periods=40, freq='h', tz='UTC')
    sparse = pd.DataFrame(index=index)
    assert host._clock_prediction(
        {'X_train': sparse, 'X_holdout': sparse}, np.zeros(40), True) == {}


def test_the_classification_bar_is_the_best_constant_not_the_most_common_one():
    """The hole the negative control found, in one test.

    F1 with average='binary' scores the POSITIVE class, so "always the
    majority class" is always-zero and scores exactly 0.0. Any model that
    predicts a single true positive clears it. Models trained on a SHUFFLED
    target passed the gate at the same rate as models trained on the real one
    — 28% each — by degenerating to almost-all-ones, which scores 2p/(1+p):
    0.61 on a holdout with a 44% positive rate, against a bar of zero.

    Kept, and still passing 'F1' explicitly, although F1 no longer GOVERNS
    anything (see `CLASSIFICATION_METRIC`, changed 2026-08-31):
    `_score_naive_baselines` is metric-agnostic and must stay correct for
    whatever metric it is handed. The test for the metric that governs today
    is `test_the_governing_metric_cannot_be_won_by_predicting_one_class`.
    """
    host = _Host()
    n = 60
    y_train = np.array([0] * 40 + [1] * 20)          # majority is 0
    y_holdout = np.array([0, 1, 1] * (n // 3))       # 2/3 positive
    data = {
        'X_train': pd.DataFrame({'f': np.zeros(len(y_train))}),
        'y_train': y_train,
        'X_holdout': pd.DataFrame({'f': np.zeros(n)}),
        'y_holdout': y_holdout,
    }

    scores = host._score_naive_baselines(data, True, 'classification', 'F1')

    # Always-zero would score 0.0 here; always-one scores 2p/(1+p) = 0.8.
    assert scores['baseline_score'] > 0.5, (
        "the bar must be the best a model can do while learning nothing"
    )

    # And a model that merely predicts everything positive must NOT pass.
    results = {'winner_holdout_metrics': {
        'status': 'measured',
        'score': scores['baseline_score'],
        'baseline_score': scores['baseline_score'],
        'holdout_sample_count': n,
    }}
    assert host._evaluate_promotion_gate(results)['passed'] is False


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


def test_a_promoted_champion_records_what_it_beat():
    """A gate whose passes are unauditable is a gate you have to trust.

    `_collect_gate_refusal` kept the numbers for every REFUSED context, so
    "why did nothing get promoted" was answerable while "what did this
    champion actually beat" was not. Noticed on 2026-08-31, on the first
    champion of the run that added the missing ladder rungs: no artifact on
    disk could say whether the clock opponent had been measured at all.

    An unmeasured rung must stay None. Zero would read as "the model beat
    it", which is the opposite of what an absent measurement means.
    """
    from src.pipeline.stages.modeling.orchestrator import ModelingStage

    measured = ModelingStage._ladder_evidence({
        'winner_holdout_metrics': {
            'score': 0.61, 'metric': 'F1',
            'baseline_kind': 'clock',
            'baseline_score': 0.55,
            'baseline_constant_score': 0.40,
            'baseline_persistence_score': 0.48,
            'baseline_persistence_lag_bars': 4,
            'baseline_clock_score': 0.55,
            'baseline_clock_scheme': 'hour',
            'single_feature_score': 0.50,
        }
    })
    assert measured['binding_opponent'] == 'clock'
    assert measured['baseline_clock_score'] == 0.55
    assert measured['baseline_constant_score'] == 0.40
    assert measured['single_feature_score'] == 0.50

    # A rung that could not be measured must be ABSENT, never zero: zero
    # reads as "the model beat it", the opposite of an absent measurement.
    unmeasured = ModelingStage._ladder_evidence({
        'winner_holdout_metrics': {'score': 0.61, 'baseline_kind': 'constant',
                                   'baseline_constant_score': 0.40}
    })
    assert unmeasured.get('baseline_clock_score') is None
    assert unmeasured.get('baseline_persistence_score') is None

    # And no holdout at all must not fabricate a full ladder.
    assert ModelingStage._ladder_evidence({}).get('score') is None


def test_the_champion_record_cannot_fall_behind_the_gate():
    """Every number the gate weighs must reach the record, without a list.

    The first version of `_ladder_evidence` enumerated fields, and within the
    hour it was out of date: `baseline_margin_sigma`, the value that decides
    promotion since #186, was added to the gate and not to the list, so the
    champion record still could not say what had cleared it. That is the same
    defect the method was written to fix, reappearing inside the fix.
    """
    from src.pipeline.stages.modeling.orchestrator import ModelingStage

    holdout = {
        'status': 'measured', 'score': 0.61, 'metric': 'F1',
        'baseline_score': 0.55, 'baseline_kind': 'clock',
        'baseline_margin': 0.06, 'baseline_margin_sigma': 0.004,
        'excess_over_passive': 0.001, 'excess_risk_adjusted': 0.0004,
        'single_feature_score': 0.50, 'single_feature_status': 'measured',
        'holdout_sample_count': 30_494, 'holdout_event_count': 7_987,
        'a_field_invented_tomorrow': 42,
        '_baseline_prediction': [0, 1, 0],     # internal, must not be copied
    }
    evidence = ModelingStage._ladder_evidence({
        'winner_holdout_metrics': holdout,
        'promotion_gate': {'passed': True, 'reasons': ['ok']},
    })

    for key, value in holdout.items():
        if key.startswith('_'):
            assert key not in evidence, key
        else:
            assert evidence[key] == value, key
    assert evidence['gate']['passed'] is True


def test_a_margin_smaller_than_its_own_noise_is_refused():
    """The decision of 2026-08-31, taken on a real number.

    The first champion of run 7 cleared its strongest opponent by 0.0046 of
    F1 — 0.4197 against a constant's 0.4151 — and `min_baseline_margin: 0.0`
    promoted it. #175 had already settled the principle on a different
    target: an edge of 0.041 against an opponent whose own weekly figure
    varied by 0.07 is not an edge, it is a sample too short to answer the
    question.

    The bar is now the opponent plus one standard error of the difference
    between them, measured on that same holdout.
    """
    host = _Host()
    base = {
        'status': 'measured', 'holdout_sample_count': 25_000,
        'holdout_event_count': 9_000,
        'score': 0.4197, 'baseline_score': 0.4151,
        'baseline_kind': 'constant', 'baseline_constant_score': 0.4151,
    }

    thin = host._evaluate_promotion_gate(
        {'winner_holdout_metrics': dict(base, baseline_margin_sigma=0.0120)})
    assert thin['passed'] is False
    assert any('one standard error' in r for r in thin['reasons'])

    # The same margin against a quieter measurement is evidence.
    solid = host._evaluate_promotion_gate(
        {'winner_holdout_metrics': dict(base, baseline_margin_sigma=0.0004)})
    assert solid['passed'] is True


def test_an_unmeasurable_margin_does_not_pass_as_a_measured_one():
    """Too short to resample means "cannot tell", not "good enough"."""
    host = _Host()
    gate = host._evaluate_promotion_gate({'winner_holdout_metrics': {
        'status': 'measured', 'holdout_sample_count': 40,
        'holdout_event_count': 15,
        'score': 0.90, 'baseline_score': 0.10,
        'baseline_kind': 'constant', 'baseline_margin_sigma': None,
    }})
    assert gate['passed'] is False
    assert any('could not be measured' in r for r in gate['reasons'])


def test_the_sigma_rule_can_be_switched_off():
    """It is a policy, and policies are named in config, not buried."""
    host = _Host({'require_baseline_margin_sigma': False})
    gate = host._evaluate_promotion_gate({'winner_holdout_metrics': {
        'status': 'measured', 'holdout_sample_count': 25_000,
        'holdout_event_count': 9_000,
        'score': 0.4197, 'baseline_score': 0.4151,
        'baseline_kind': 'constant', 'baseline_margin_sigma': 0.0120,
    }})
    assert gate['passed'] is True


def test_the_bootstrap_sigma_grows_when_the_two_series_really_differ():
    """A sanity check on the measurement itself, not on the gate.

    Blocks, not single rows: holdout rows are a time series, and resampling
    them independently would understate the spread and hand back a margin
    that looks significant because the resampling pretended otherwise.
    """
    host = _Host()
    rng = np.random.default_rng(3)
    n = 600
    truth = (rng.normal(size=n) > 0).astype(int)

    # A model that IS the truth against a coin: the difference is huge and
    # stable, so its standard error must be small relative to the gap.
    sharp = host._block_bootstrap_sigma(
        truth, truth, rng.integers(0, 2, n),
        task_type='classification', metric_key='F1')

    # Two coins: no real difference, so the spread is what noise looks like.
    noisy = host._block_bootstrap_sigma(
        truth, rng.integers(0, 2, n), rng.integers(0, 2, n),
        task_type='classification', metric_key='F1')

    assert sharp is not None and noisy is not None
    assert 0.0 < sharp < 0.5 and 0.0 < noisy < 0.5

    # Too short to resample must be None, never zero.
    assert host._block_bootstrap_sigma(
        truth[:30], truth[:30], truth[:30],
        task_type='classification', metric_key='F1') is None


def test_the_governing_metric_cannot_be_won_by_predicting_one_class():
    """Under balanced accuracy every constant scores exactly 0.5.

    That is the whole reason for the change of 2026-08-31. On F1 the bar
    depended on the class balance — 0.0 for always-no, 2p/(1+p) for
    always-yes — so a model could clear it by shifting its threshold rather
    than by knowing anything. On balanced accuracy both constants sit at 0.5
    whatever the balance, so the bar measures the model.
    """
    host = _Host()
    for positives in (0.05, 0.26, 0.50, 0.90):
        n = 400
        rng = np.random.default_rng(int(positives * 100))
        y_train = (rng.random(n) < positives).astype(int)
        y_holdout = (rng.random(n) < positives).astype(int)
        data = {
            'X_train': pd.DataFrame({'f': np.zeros(n)}),
            'y_train': y_train,
            'X_holdout': pd.DataFrame({'f': np.zeros(n)}),
            'y_holdout': y_holdout,
        }

        scores = host._score_naive_baselines(
            data, True, 'classification', 'BalancedAccuracy')

        assert scores['baseline_constant_score'] == pytest.approx(0.5), positives


def test_the_arena_and_the_gate_judge_by_the_same_metric():
    """They did not, and that is how a "yes"-machine became a champion.

    Selection took F1, so the winner of every classification context was
    whichever model said yes most confidently; the gate then measured that
    winner against opponents on a metric it had never competed on. One name
    in one place is the fix — a second copy is how `max_features` ended up
    with three answers and Sharpe with two.

    F1 may still appear in this module as prose or as a metric other code
    asks for; what must not appear is a second place that DECIDES with it.
    """
    import inspect

    from src.training import base_trainer

    deciding = [
        line.strip() for line in inspect.getsource(base_trainer).splitlines()
        if ("metric_key = " in line or "score.get(" in line)
        and ("'F1'" in line or '"F1"' in line)
    ]
    assert not deciding, deciding
    assert base_trainer.CLASSIFICATION_METRIC == 'BalancedAccuracy'
    assert base_trainer._governing_metric(True) == 'BalancedAccuracy'
    assert base_trainer._governing_metric(False) == 'R2'
