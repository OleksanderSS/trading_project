import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd

from src.pipeline.stages.modeling.utils import (
    determine_task_type,
    get_context_fingerprint,
    get_light_model_training_data,
    get_light_model_types,
    log_to_diary,
)


def test_get_context_fingerprint_returns_unknown_when_missing():
    assert get_context_fingerprint({}) == 'unknown'
    assert get_context_fingerprint({'context_fingerprint': 'fp_123'}) == 'fp_123'


def test_determine_task_type_based_on_target_name():
    assert determine_task_type('future_return_5d') == 'regression'
    assert determine_task_type('predicted_price') == 'regression'
    assert determine_task_type('direction') == 'classification'


def test_get_light_model_training_data_returns_none_when_missing():
    assert get_light_model_training_data({}) is None
    assert get_light_model_training_data({'light_models': {'X_train': None, 'y_train': None}}) is None


def test_get_light_model_training_data_returns_tuple_when_present():
    data = {
        'light_models': {
            'X_train': pd.DataFrame({'x': [1, 2]}),
            'y_train': pd.Series([0, 1]),
            'X_test': pd.DataFrame({'x': [3]}),
            'y_test': pd.Series([1]),
        }
    }

    result = get_light_model_training_data(data)
    assert result is not None
    X_train, y_train, X_test, y_test = result
    assert list(X_train['x']) == [1, 2]
    assert list(y_train) == [0, 1]
    assert list(X_test['x']) == [3]
    assert list(y_test) == [1]


def test_get_light_model_types_contains_expected_backends():
    expected = {'catboost', 'lightgbm', 'xgboost', 'random_forest', 'linear', 'svm', 'knn'}
    assert set(get_light_model_types()) == expected


def test_log_to_diary_appends_csv_entry():
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        diary_path = tmp_path / 'diary.csv'
        info = {
            'timestamp': '2026-05-23T00:00:00',
            'ticker': 'AAPL',
            'target': 'future_return',
            'winner': 'model_x',
            'context': 'fp_123',
        }
        tf = '1d'

        # Write header first to mimic diary file format
        pd.DataFrame([{
            'timestamp': 'timestamp',
            'ticker': 'ticker',
            'tf': 'tf',
            'target': 'target',
            'model_name': 'model_name',
            'context_fingerprint': 'context_fingerprint',
            'is_champion': 'is_champion',
            'cpu_usage': 'cpu_usage',
            'ram_usage': 'ram_usage',
        }]).to_csv(diary_path, index=False)

        log_to_diary(diary_path, info, tf)
        df = pd.read_csv(diary_path)
        assert df.iloc[-1]['ticker'] == 'AAPL'
        assert df.iloc[-1]['tf'] == '1d'
        assert df.iloc[-1]['model_name'] == 'model_x'
