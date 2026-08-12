"""Negative control: does the promotion gate reject a target made of noise?

The gate now refuses a winner that cannot beat a naive baseline on an
untouched holdout, and 41 of 95 contexts were refused on the 2026-08-12 run.
That number only means something if the gate ALSO refuses a model trained on
a target with no signal in it at all. If a shuffled target still passes, the
gate is measuring its own arithmetic and every champion is worthless.

Method, per sampled context:

  1. build the real splits with prepare_data_for_models -- the same function
     Stage 4 uses, so the purge, the imputer and the scaler are identical;
  2. train each model type twice: once on the real training target, once on
     the SAME target shuffled;
  3. score both on the UNTOUCHED holdout, whose target is never shuffled;
  4. compare each against the naive baseline the gate uses.

Shuffling only the training target is what makes this a control: the model is
handed a real feature matrix and a meaningless answer key, so anything it
learns is coincidence, and the honest outcome on real holdout rows is failure.

A shuffled run that beats the baseline is not a curiosity. It means the
comparison itself is broken -- leakage from the holdout, a metric that
rewards the wrong thing, or a baseline that is weaker than it looks.

Usage:
    python scripts/diagnostics/negative_control.py [--contexts 12] [--seed 7]
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.config.feature_budget import get_model_max_features  # noqa: E402
from src.factories.model_factory import ModelFactory  # noqa: E402
from src.metrics.model.ml_evaluator import MLEvaluator  # noqa: E402
from src.models.adapters.data_preparation import prepare_data_for_models  # noqa: E402

BATCH = Path('data/colab/accumulated/main_database')
MODEL_TYPES = ('linear', 'random_forest', 'lightgbm')


def _naive_baseline(y_train, y_holdout, is_classif, evaluator, task_type, metric_key):
    """The same two opponents BaseTrainer uses: constant, and persistence."""
    y_train = np.asarray(y_train).ravel()
    y_true = np.asarray(y_holdout).ravel()
    n = y_true.size
    if is_classif:
        candidates = list(np.unique(y_train[~pd.isna(y_train)]))
    else:
        candidates = [float(np.nanmean(y_train))]
    best = max(
        float(evaluator.calculate(
            y_holdout, np.full(n, candidate), task_type=task_type
        ).get(metric_key, 0.0))
        for candidate in candidates
    )

    if not is_classif and n >= 3:
        persistence = np.empty(n, dtype=float)
        persistence[0] = y_true[0]
        persistence[1:] = y_true[:-1]
        score = float(evaluator.calculate(
            y_holdout, persistence, task_type=task_type
        ).get(metric_key, 0.0))
        best = max(best, score)
    return best


def _select_columns(X_train, y_train, budget):
    """Train-only ranking, mirroring BaseTrainer._select_features_for_model."""
    if len(X_train.columns) <= budget:
        return list(X_train.columns)
    numeric = X_train.select_dtypes(include=[np.number])
    y = pd.Series(np.asarray(y_train).ravel(), index=numeric.index).astype(float)
    ranked = numeric.corrwith(y).abs().fillna(-1.0).sort_values(
        ascending=False, kind='mergesort'
    )
    return list(ranked.index[:budget])


def _walk_forward_control(contexts, features, targets, seed):
    """The same control, aimed at the walk-forward stability gate.

    That gate is doing most of the filtering — 94% of contexts refused on the
    2026-08-12 run — so whether it measures anything is the question that
    decides whether the whole result is trustworthy. If a shuffled target
    holds signal across three quarters of the folds as often as a real one,
    the gate is counting coincidences.
    """
    from src.pipeline.stages.stage_4_modeling import ModelingStage

    stage = object.__new__(ModelingStage)
    rows = []
    for ticker, interval, target in contexts:
        frame = features[
            (features.ticker == ticker) & (features.interval == interval)
        ].copy()
        frame[target] = targets.loc[frame.index, target]
        frame = frame[frame[target].notna()]
        if len(frame) < 200:
            continue

        shuffled = frame.copy()
        values = shuffled[target].to_numpy().copy()
        np.random.default_rng(seed).shuffle(values)
        shuffled[target] = values

        for label, candidate in (('real', frame), ('shuffled', shuffled)):
            try:
                verdict = stage._walk_forward_stability(
                    candidate, ticker=ticker, timeframe=interval,
                    target_name=target, context_key=f"{ticker}/{interval}",
                )
            except Exception as e:                  # noqa: BLE001 - diagnostic
                print(f"  wf {ticker}/{interval}/{target}/{label}: "
                      f"{type(e).__name__} {str(e)[:60]}")
                continue
            # Contexts the gate declares unmeasurable are excluded: counting
            # them as passes would flatter it, counting them as failures would
            # slander it.
            if not verdict or verdict.get('measured') is False:
                continue
            if verdict.get('fold_count', 0) < 2:
                continue
            rows.append({
                'kind': label,
                'passed': bool(verdict.get('passed')),
                'folds_above': verdict.get('folds_above_majority'),
                'fold_count': verdict.get('fold_count'),
                'worst': verdict.get('worst_fold_balanced_accuracy'),
            })
    return rows


def run(n_contexts: int, seed: int) -> int:
    rng = random.Random(seed)
    evaluator = MLEvaluator()

    features = pd.read_parquet(BATCH / 'features.parquet')
    targets = pd.read_parquet(BATCH / 'targets.parquet')
    target_names = [c for c in targets.columns if c.startswith('target_')]

    contexts = []
    for (ticker, interval), group in features.groupby(['ticker', 'interval']):
        for target in target_names:
            column = targets.loc[group.index, target]
            if column.notna().sum() >= 120:
                contexts.append((ticker, interval, target))
    rng.shuffle(contexts)
    contexts = contexts[:n_contexts]
    print(f"sampled {len(contexts)} contexts (seed={seed})\n")

    rows = []
    for ticker, interval, target in contexts:
        frame = features[
            (features.ticker == ticker) & (features.interval == interval)
        ].copy()
        frame[target] = targets.loc[frame.index, target]
        frame = frame[frame[target].notna()]
        if len(frame) < 120:
            continue
        try:
            prepared = prepare_data_for_models(
                frame, ticker=ticker, timeframe=interval,
                target_cols=[target], gap_size=2, val_size=0.2, test_size=0.2,
            )
        except Exception as e:                      # noqa: BLE001 - diagnostic script
            print(f"  skip {ticker}/{interval}/{target}: {type(e).__name__} {e}")
            continue

        light = prepared['light_models']
        X_train, y_train = light['X_train'], np.asarray(light['y_train']).ravel()
        X_hold, y_hold = light['X_test'], np.asarray(light['y_test']).ravel()
        if X_hold is None or len(X_hold) < 20:
            continue

        is_classif = bool(pd.Series(y_train).dropna().isin([0, 1]).all())
        task_type = 'classification' if is_classif else 'regression'
        metric_key = 'F1' if is_classif else 'R2'
        baseline = _naive_baseline(
            y_train, y_hold, is_classif, evaluator, task_type, metric_key
        )

        shuffled = y_train.copy()
        np.random.default_rng(seed).shuffle(shuffled)

        for model_type in MODEL_TYPES:
            budget = get_model_max_features(model_type)
            for label, y in (('real', y_train), ('shuffled', shuffled)):
                columns = _select_columns(X_train, y, budget)
                try:
                    model = ModelFactory.create_model(
                        model_name=model_type, config={},
                        task_type=task_type, is_classification=is_classif,
                    )
                    model.train(X_train[columns], y)
                    preds = model.predict(X_hold[columns])
                    score = float(evaluator.calculate(
                        y_hold, preds, task_type=task_type
                    ).get(metric_key, 0.0))
                except Exception as e:              # noqa: BLE001 - diagnostic script
                    print(f"  {ticker}/{interval}/{target}/{model_type}/{label}: "
                          f"{type(e).__name__} {str(e)[:60]}")
                    continue
                rows.append({
                    'context': f"{ticker}/{interval}/{target}",
                    'model': model_type, 'kind': label,
                    'score': score, 'baseline': baseline,
                    'passes_gate': score > baseline,
                })

    stability_rows = _walk_forward_control(contexts, features, targets, seed)

    if not rows:
        print("no contexts produced a result")
        return 1

    df = pd.DataFrame(rows)
    print(df.groupby('kind').agg(
        runs=('score', 'size'),
        passed_gate=('passes_gate', 'sum'),
        pass_rate=('passes_gate', 'mean'),
        median_score=('score', 'median'),
    ).to_string())

    if stability_rows:
        wf = pd.DataFrame(stability_rows)
        print("\n=== walk-forward stability gate ===")
        print(wf.groupby('kind').agg(
            runs=('passed', 'size'),
            passed=('passed', 'sum'),
            pass_rate=('passed', 'mean'),
            median_folds_above=('folds_above', 'median'),
            median_worst_fold=('worst', 'median'),
        ).to_string())
        real_rate = float(wf[wf.kind == 'real']['passed'].mean() or 0)
        noise_rate = float(wf[wf.kind == 'shuffled']['passed'].mean() or 0)
        if noise_rate >= real_rate and noise_rate > 0:
            print("\n  WARNING: shuffled contexts clear the stability gate as often "
                  "as real ones. The gate is counting coincidences.")
        else:
            print(f"\n  stability gate separates signal from noise: "
                  f"real {real_rate:.0%} vs shuffled {noise_rate:.0%}")

    shuffled_passes = df[(df.kind == 'shuffled') & df.passes_gate]
    print("\n--- VERDICT ---")
    if shuffled_passes.empty:
        print("PASS: no run on a shuffled target beat the naive baseline.")
        print("The gate is refusing noise, so its refusals of real targets mean something.")
        return 0
    rate = len(shuffled_passes) / int((df.kind == 'shuffled').sum())
    print(f"SUSPECT: {len(shuffled_passes)} shuffled run(s) beat the baseline "
          f"({rate:.0%}).")
    print(shuffled_passes[['context', 'model', 'score', 'baseline']].to_string(index=False))
    print("\nA target with no signal should not clear the bar. Investigate the "
          "comparison before trusting any champion.")
    return 2


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--contexts', type=int, default=12)
    parser.add_argument('--seed', type=int, default=7)
    args = parser.parse_args()
    raise SystemExit(run(args.contexts, args.seed))
