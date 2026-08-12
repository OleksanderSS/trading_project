"""A model must be served features in the space it was trained in.

Training standardises. `prepare_data_for_models` fits a SimpleImputer and a
StandardScaler on the training split and hands every light model z-scores; it
returned both objects in `light_data` and the prediction path collected
neither. Stage 5 sliced raw columns out of the feature frame instead.

Measured on a real 35-feature champion from the 2026-08-12 run, the identical
model returned:

    z-scored input (as trained) -> [ 0.033, -0.023,  0.156, -0.074,  0.137]
    raw input      (as served)  -> [128288, 127314, 133867, 122286, 129896]

Nothing raised. Trees compared thresholds learned in z-space against a close
of 120; linear models multiplied coefficients fitted against unit variance by
a volume of 5e7.
"""
import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from src.pipeline.constants import preprocessor_filename
from src.pipeline.stages.prediction.data_preparation_service import (
    DataPreparationService,
)


@pytest.fixture(autouse=True)
def _trust_tmp_path(monkeypatch, tmp_path):
    """The preprocessor is a joblib artifact, so it goes through the same
    trusted-root check as a model. Widen the trusted set to the test's own
    directory rather than bypassing the check."""
    monkeypatch.setenv("TRADING_TRUSTED_ARTIFACT_ROOTS", str(tmp_path))


@pytest.fixture
def service():
    instance = object.__new__(DataPreparationService)
    instance.logger = _NullLogger()
    return instance


class _NullLogger:
    def debug(self, *a, **k):
        pass

    def info(self, *a, **k):
        pass

    def warning(self, *a, **k):
        pass

    def error(self, *a, **k):
        pass

    def isEnabledFor(self, _level):
        return False


def _fit_preprocessor(tmp_path, frame, ticker="AAPL", timeframe="1d", target="target_up_1d"):
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    imputed = imputer.fit_transform(frame)
    scaler.fit(imputed)
    path = tmp_path / preprocessor_filename(ticker, timeframe, target)
    joblib.dump(
        {"imputer": imputer, "scaler": scaler, "feature_names": list(frame.columns)},
        path,
    )
    return path, imputer, scaler


def _meta(tmp_path, ticker="AAPL", timeframe="1d", target="target_up_1d"):
    return {
        "ticker": ticker,
        "timeframe": timeframe,
        "target": target,
        "model_path": str(tmp_path / "CHAMP_AAPL_1d_target_up_1d.joblib"),
    }


def test_features_are_standardised_the_way_training_standardised_them(service, tmp_path):
    train = pd.DataFrame({
        "close": np.linspace(100.0, 140.0, 100),
        "volume": np.linspace(4.8e7, 5.2e7, 100),
    })
    _fit_preprocessor(tmp_path, train)

    live = pd.DataFrame({"close": [120.0, 130.0], "volume": [5.0e7, 5.1e7]})
    out = service._apply_training_preprocessor(live, _meta(tmp_path), "ctx")

    # Raw scale is gone: these are z-scores, not dollars.
    assert abs(out["close"].abs().max()) < 5
    assert abs(out["volume"].abs().max()) < 5
    assert out.shape[0] == 2


def test_columns_are_reordered_to_the_fit_time_order(service, tmp_path):
    """A StandardScaler on the same columns in another order is another transform."""
    train = pd.DataFrame({
        "a": np.linspace(0.0, 1.0, 50),
        "b": np.linspace(1000.0, 2000.0, 50),
    })
    _, imputer, scaler = _fit_preprocessor(tmp_path, train)

    # Same data, columns presented back to front.
    live = pd.DataFrame({"b": [1500.0], "a": [0.5]})
    out = service._apply_training_preprocessor(live, _meta(tmp_path), "ctx")

    expected = scaler.transform(imputer.transform(pd.DataFrame({"a": [0.5], "b": [1500.0]})))
    assert np.allclose(out[["a", "b"]].to_numpy(), expected)


def test_columns_the_scaler_never_saw_are_kept_but_not_transformed(service, tmp_path):
    train = pd.DataFrame({"close": np.linspace(100.0, 140.0, 50)})
    _fit_preprocessor(tmp_path, train)

    live = pd.DataFrame({"close": [120.0], "context_pattern_id": ["abc"]})
    out = service._apply_training_preprocessor(live, _meta(tmp_path), "ctx")

    assert "context_pattern_id" in out.columns
    assert out["context_pattern_id"].iloc[0] == "abc"


def test_a_row_with_one_gap_is_imputed_the_way_training_imputed_it(service, tmp_path):
    """Training filled missing values with the train median; prediction dropped
    the row. Two opposite readings of one gap. Prediction now reads it the way
    training does."""
    train = pd.DataFrame({
        "a": np.linspace(0.0, 10.0, 50),
        "b": np.linspace(100.0, 200.0, 50),
        "c": np.linspace(-1.0, 1.0, 50),
        "d": np.linspace(5.0, 15.0, 50),
    })
    _fit_preprocessor(tmp_path, train)

    live = pd.DataFrame({"a": [5.0], "b": [150.0], "c": [0.0], "d": [np.nan]})
    out = service._apply_training_preprocessor(live, _meta(tmp_path), "ctx")

    assert len(out) == 1, "one missing feature of four must not discard the row"
    assert out.notna().all(axis=None)


def test_a_mostly_invented_row_is_refused(service, tmp_path):
    """Filling one feature of four is tolerance; filling three is invention.

    The intraday case makes this concrete: ctx_1d_* columns are absent on the
    newest bars because the day has not closed, so imputing them fabricates
    the very context being asked about.
    """
    train = pd.DataFrame({
        "a": np.linspace(0.0, 10.0, 50),
        "b": np.linspace(100.0, 200.0, 50),
        "c": np.linspace(-1.0, 1.0, 50),
        "d": np.linspace(5.0, 15.0, 50),
    })
    _fit_preprocessor(tmp_path, train)

    live = pd.DataFrame({
        "a": [5.0, 5.0],
        "b": [150.0, np.nan],
        "c": [0.0, np.nan],
        "d": [10.0, np.nan],
    })
    out = service._apply_training_preprocessor(live, _meta(tmp_path), "ctx")

    assert len(out) == 1, "the 75%-imputed row must be refused"
    assert out.index.tolist() == [0]


def test_a_missing_preprocessor_leaves_the_frame_alone(service, tmp_path):
    """Champions promoted before this artifact existed must not be guessed at."""
    live = pd.DataFrame({"close": [120.0, 130.0]})

    out = service._apply_training_preprocessor(live, _meta(tmp_path), "ctx")

    pd.testing.assert_frame_equal(out, live)


def test_a_broken_transform_refuses_rather_than_predicting_on_raw_values(service, tmp_path):
    train = pd.DataFrame({"close": np.linspace(100.0, 140.0, 50)})
    path, _, _ = _fit_preprocessor(tmp_path, train)
    # A payload whose scaler expects a different width than it declares.
    joblib.dump(
        {"imputer": None, "scaler": _ExplodingScaler(), "feature_names": ["close"]},
        path,
    )

    out = service._apply_training_preprocessor(
        pd.DataFrame({"close": [120.0]}), _meta(tmp_path), "ctx"
    )

    assert out.empty, "a failed transform must not fall back to raw features"


class _ExplodingScaler:
    def transform(self, values):
        raise ValueError("X has 1 features, but StandardScaler is expecting 35")
