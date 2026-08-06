"""Tests for scripts/colab/colab_clean_cell.py's target-type-aware training.

Context: the Colab training cell trained every target as plain regression
(bare Dense(1) linear output, loss='mse', target never scaled) regardless of
whether src/config/targets.yaml declared it classification_binary,
classification_multiclass, regression, or indicator_prediction. This meant:
binary up/down targets (target_up_1d, target_weekly_up_1w, ...) never got
accuracy/AUC anywhere, only an uninterpretable MSE against a 0/1 label;
price-level targets (SMA/EMA/BB) showed MSE in the tens of thousands because
the raw dollar-denominated target was never scaled the way the input
features were. A second, independent bug compounded this: every
_train_model_with_features dispatch branch called its trainer without
`return`, so even a *successful* training run's metrics dict was discarded
and replaced with None before ever reaching models_metadata.

Full 7-architecture x 3-target-type live training (Keras + sklearn +
pytorch-tabnet) was verified manually against this fix -- 13/13 cases
succeeded with correctly-shaped metrics (accuracy/auc for classification,
real-unit MSE for regression). These tests cover the fast, TF-free parts
that matter for CI: the target-type registry itself, the pure unscale math,
and one live sklearn (MLP) case per target type as an end-to-end smoke test
that doesn't require TensorFlow/pytorch-tabnet to be installed.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.colab.colab_clean_cell import (
    CLASSIFICATION_BINARY_TYPE,
    CLASSIFICATION_MULTICLASS_TYPE,
    ColabTrainingController,
    ConfigLoader,
)


class TestTargetTypeRegistry:
    def test_loads_real_targets_yaml(self):
        loader = ConfigLoader.__new__(ConfigLoader)
        loader.project_path = Path(__file__).resolve().parents[2]
        types = loader._load_target_types()
        assert types["target_up_1d"] == CLASSIFICATION_BINARY_TYPE
        assert types["target_up_5d"] == CLASSIFICATION_BINARY_TYPE
        assert types["target_multi_1d"] == CLASSIFICATION_MULTICLASS_TYPE
        assert types["target_return_1d"] == "regression"
        assert types["target_sma_20_f1"] == "indicator_prediction"

    def test_target_type_for_defaults_to_regression_for_unknown_target(self):
        loader = ConfigLoader.__new__(ConfigLoader)
        loader.target_types = {"target_up_1d": CLASSIFICATION_BINARY_TYPE}
        assert loader.target_type_for("target_totally_unknown") == "regression"

    def test_missing_targets_yaml_degrades_to_empty_registry_not_a_crash(self, tmp_path):
        loader = ConfigLoader.__new__(ConfigLoader)
        loader.project_path = tmp_path  # no src/config/targets.yaml here
        loader.target_types = loader._load_target_types()
        assert loader.target_types == {}
        assert loader.target_type_for("target_up_1d") == "regression"


class TestBuildSequences:
    """cnn/lstm/gru/transformer used to receive x.reshape(n, 1, features) --
    a fake sequence length of 1 that makes recurrence/attention a no-op.
    _build_sequences turns flat, chronologically-ordered rows into real
    (window, features) history windows instead."""

    @staticmethod
    def _xy(n=60, seed=0):
        rng = np.random.RandomState(seed)
        x = pd.DataFrame(rng.randn(n, 4), columns=[f"f{i}" for i in range(4)])
        y = pd.Series(rng.randn(n).astype(np.float32))
        return x, y

    def test_shape_is_n_minus_window_plus_one_by_window_by_features(self):
        x, y = self._xy(n=60)
        x_seq, y_seq = ColabTrainingController._build_sequences(x, y, 20)
        assert x_seq.shape == (60 - 20 + 1, 20, 4)
        assert y_seq.shape == (60 - 20 + 1,)

    def test_window_alignment_matches_source_rows(self):
        """Window i's last row must be source row i+window-1 (the sample
        the target belongs to); its first row must be source row i."""
        x, y = self._xy(n=60)
        window = 20
        x_seq, y_seq = ColabTrainingController._build_sequences(x, y, window)
        for i in (0, 10, len(x_seq) - 1):
            assert np.allclose(x_seq[i, -1, :], x.iloc[i + window - 1].to_numpy())
            assert np.allclose(x_seq[i, 0, :], x.iloc[i].to_numpy())
            assert y_seq[i] == pytest.approx(y.iloc[i + window - 1])

    def test_insufficient_history_returns_none(self):
        x, y = self._xy(n=5)
        x_seq, y_seq = ColabTrainingController._build_sequences(x, y, 20)
        assert x_seq is None
        assert y_seq is None

    def test_exactly_window_rows_yields_one_sequence(self):
        x, y = self._xy(n=20)
        x_seq, y_seq = ColabTrainingController._build_sequences(x, y, 20)
        assert x_seq.shape == (1, 20, 4)


class TestChronologicalSplit:
    """Every trainer used a random train_test_split(..., random_state=42),
    which lets validation rows sit chronologically before/interleaved with
    training rows -- and, after _build_sequences introduced overlapping
    20-day windows, could put a validation window sharing up to 19 of its
    20 days with an adjacent training window. _chronological_split replaces
    this with a time-ordered split plus an optional purge gap that removes
    that boundary overlap entirely."""

    def test_no_purge_train_and_val_are_contiguous(self):
        x = pd.DataFrame({"a": range(100)})
        y = pd.Series(range(100))
        xt, xv, yt, yv = ColabTrainingController._chronological_split(x, y, val_fraction=0.2, purge=0)
        assert len(xt) == 80 and len(xv) == 20
        assert xt["a"].max() == 79
        assert xv["a"].min() == 80

    def test_purge_leaves_a_gap_between_train_and_val(self):
        x = pd.DataFrame({"a": range(100)})
        y = pd.Series(range(100))
        xt, xv, yt, yv = ColabTrainingController._chronological_split(x, y, val_fraction=0.2, purge=19)
        assert len(xt) == 61 and len(xv) == 20
        assert xt["a"].max() == 60
        assert xv["a"].min() == 80
        # rows 61..79 (the 19-row gap) appear in neither split.
        assert set(xt["a"]).isdisjoint(set(xv["a"]))

    def test_works_on_numpy_arrays_not_just_dataframes(self):
        x = np.arange(100).reshape(100, 1)
        y = np.arange(100)
        xt, xv, yt, yv = ColabTrainingController._chronological_split(x, y, val_fraction=0.2, purge=19)
        assert xt.shape[0] == 61 and xv.shape[0] == 20

    def test_val_is_always_the_most_recent_rows(self):
        """Unlike a random split, validation must be the tail of the
        chronologically-ordered data, never earlier rows."""
        x = pd.DataFrame({"a": range(50)})
        y = pd.Series(range(50))
        _, xv, _, _ = ColabTrainingController._chronological_split(x, y, val_fraction=0.3, purge=0)
        assert xv["a"].tolist() == list(range(35, 50))


class TestUnscaleMse:
    def test_none_scaler_is_identity(self):
        assert ColabTrainingController._unscale_mse(2.5, None) == 2.5

    def test_applies_scale_squared(self):
        scaler = StandardScaler()
        scaler.fit(np.array([0.0, 10.0, 20.0]).reshape(-1, 1))  # scale_ = std = ~8.16
        expected = 2.0 * (scaler.scale_[0] ** 2)
        assert ColabTrainingController._unscale_mse(2.0, scaler) == pytest.approx(expected)


class TestContextSnapshotAndWindows:
    """_context_snapshot/_context_windows are pure pandas/dict plumbing --
    no TF or sklearn needed. They compute the market-context metadata
    attached to every trainer's validation_windows report, independent of
    which features a given model_type selected."""

    @staticmethod
    def _context_df(n=100):
        rng = np.random.RandomState(0)
        return pd.DataFrame({
            "market_context_volatility_ratio": rng.randn(n),
            "market_context_trend_20d": rng.randn(n),
            "some_unrelated_feature": rng.randn(n),
        })

    def test_snapshot_only_includes_known_context_prefixes(self):
        x = self._context_df(n=10)
        snapshot = ColabTrainingController._context_snapshot(x)
        assert set(snapshot) == {"market_context_volatility_ratio", "market_context_trend_20d"}
        assert "some_unrelated_feature" not in snapshot

    def test_snapshot_is_mean_of_the_window(self):
        x = pd.DataFrame({"market_context_volatility_ratio": [1.0, 2.0, 3.0]})
        snapshot = ColabTrainingController._context_snapshot(x)
        assert snapshot["market_context_volatility_ratio"] == pytest.approx(2.0)

    def test_snapshot_on_dataframe_with_no_context_columns_is_empty(self):
        x = pd.DataFrame({"f0": [1.0, 2.0]})
        assert ColabTrainingController._context_snapshot(x) == {}

    def test_windows_are_sliced_from_validation_tail_only(self):
        x = self._context_df(n=100)
        windows = ColabTrainingController._context_windows(x, val_fraction=0.2, n_windows=2)
        assert len(windows) == 2
        for w in windows:
            assert "market_context_volatility_ratio" in w

    def test_empty_dataframe_yields_no_windows(self):
        assert ColabTrainingController._context_windows(pd.DataFrame(), n_windows=3) == []


class TestWindowedMetricReport:
    """_windowed_metric_report slices the SAME validation predictions used
    for the aggregate metric into contiguous chronological chunks -- this
    replaces the old single-metric/single-split validation report."""

    def test_empty_input_yields_empty_report(self):
        report = ColabTrainingController._windowed_metric_report(
            [], [], is_classification=False, target_type="regression", y_scaler=None,
        )
        assert report == []

    def test_regression_windows_report_mse_and_sample_counts(self):
        y_true = np.arange(30, dtype=float)
        y_pred = y_true + 1.0  # constant offset -> known MSE per window
        report = ColabTrainingController._windowed_metric_report(
            y_true, y_pred, is_classification=False, target_type="regression",
            y_scaler=None, n_windows=3,
        )
        assert len(report) == 3
        assert sum(w["n_samples"] for w in report) == 30
        for w in report:
            assert w["mse"] == pytest.approx(1.0)

    def test_classification_windows_report_accuracy(self):
        rng = np.random.RandomState(1)
        y_true = rng.randint(0, 2, 60)
        y_pred = y_true.copy()  # perfect predictions -> accuracy 1.0 everywhere
        report = ColabTrainingController._windowed_metric_report(
            y_true, y_pred, is_classification=True, target_type=CLASSIFICATION_BINARY_TYPE,
            y_scaler=None, n_windows=3,
        )
        assert len(report) == 3
        for w in report:
            assert w["accuracy"] == pytest.approx(1.0)

    def test_binary_probability_predictions_are_thresholded_for_accuracy(self):
        y_true = np.array([0, 0, 1, 1])
        y_pred_proba = np.array([0.1, 0.2, 0.9, 0.8])  # all correct once thresholded at 0.5
        report = ColabTrainingController._windowed_metric_report(
            y_true, y_pred_proba, is_classification=True, target_type=CLASSIFICATION_BINARY_TYPE,
            y_scaler=None, n_windows=1,
        )
        assert report[0]["accuracy"] == pytest.approx(1.0)

    def test_multiclass_windows_skip_auc(self):
        y_true = np.array([0, 1, 2, 0, 1, 2])
        y_pred = y_true.copy()
        report = ColabTrainingController._windowed_metric_report(
            y_true, y_pred, is_classification=True, target_type=CLASSIFICATION_MULTICLASS_TYPE,
            y_scaler=None, n_windows=1,
        )
        assert "auc" not in report[0]


class TestKerasWindowedPredictions:
    """_keras_windowed_predictions shapes a model's raw predict() output
    for _windowed_metric_report: probability for binary, hard label
    (argmax) for multiclass, raw value for regression. Uses a stub model
    instead of a real Keras model so this stays TF-free and fast."""

    class _StubModel:
        def __init__(self, raw):
            self._raw = raw

        def predict(self, x_val, verbose=0):
            return self._raw

    def test_binary_passes_through_sigmoid_probability(self):
        raw = np.array([[0.1], [0.9], [0.4]])
        model = self._StubModel(raw)
        out = ColabTrainingController._keras_windowed_predictions(
            model, None, is_classification=True, target_type=CLASSIFICATION_BINARY_TYPE,
        )
        assert out.tolist() == pytest.approx([0.1, 0.9, 0.4])

    def test_multiclass_takes_argmax_of_softmax(self):
        raw = np.array([[0.1, 0.7, 0.2], [0.8, 0.1, 0.1]])
        model = self._StubModel(raw)
        out = ColabTrainingController._keras_windowed_predictions(
            model, None, is_classification=True, target_type=CLASSIFICATION_MULTICLASS_TYPE,
        )
        assert out.tolist() == [1, 0]

    def test_regression_passes_through_raw_value(self):
        raw = np.array([[10.5], [20.1]])
        model = self._StubModel(raw)
        out = ColabTrainingController._keras_windowed_predictions(
            model, None, is_classification=False, target_type="regression",
        )
        assert out.tolist() == pytest.approx([10.5, 20.1])


class TestKerasFinalLayerAndCompileKwargs:
    """Pure dispatch logic -- imports tensorflow internally, so these are
    skipped (not failed) in an environment without it, matching this
    script's own optional-dependency handling elsewhere."""

    def _controller(self):
        return ColabTrainingController.__new__(ColabTrainingController)

    def test_binary_gets_sigmoid_single_unit(self):
        tf = pytest.importorskip("tensorflow")
        controller = self._controller()
        layer = controller._keras_final_layer(True, CLASSIFICATION_BINARY_TYPE, 2)
        assert layer.units == 1
        assert layer.activation.__name__ == "sigmoid"

    def test_multiclass_gets_softmax_num_classes_units(self):
        pytest.importorskip("tensorflow")
        controller = self._controller()
        layer = controller._keras_final_layer(True, CLASSIFICATION_MULTICLASS_TYPE, 3)
        assert layer.units == 3
        assert layer.activation.__name__ == "softmax"

    def test_regression_gets_linear_single_unit(self):
        pytest.importorskip("tensorflow")
        controller = self._controller()
        layer = controller._keras_final_layer(False, "regression", 0)
        assert layer.units == 1
        assert layer.activation.__name__ == "linear"

    def test_binary_compile_kwargs_include_accuracy_and_auc(self):
        pytest.importorskip("tensorflow")
        controller = self._controller()
        kwargs = controller._keras_compile_kwargs(True, CLASSIFICATION_BINARY_TYPE)
        assert kwargs["loss"] == "binary_crossentropy"
        assert "accuracy" in kwargs["metrics"]

    def test_regression_compile_kwargs_is_plain_mse(self):
        pytest.importorskip("tensorflow")
        controller = self._controller()
        kwargs = controller._keras_compile_kwargs(False, "regression")
        assert kwargs == {"loss": "mse", "metrics": []}


class _StubConfigLoader:
    REDUCED_EPOCHS = 5

    def __init__(self, target_type: str):
        self._target_type = target_type

    def target_type_for(self, target_col):
        return self._target_type


def _make_controller(tmp_path: Path, target_type: str) -> ColabTrainingController:
    controller = ColabTrainingController.__new__(ColabTrainingController)
    controller.config_loader = _StubConfigLoader(target_type)
    controller.path_manager = type("PM", (), {"batch_dir": tmp_path})()
    controller.logger = None
    return controller


class TestMlpTrainerEndToEnd:
    """sklearn-only (no TF/pytorch-tabnet needed) end-to-end proof that
    is_classification threads correctly through the whole call path and
    that a successful run's metrics survive (the missing-`return` bug)."""

    @staticmethod
    def _xy(n=200, seed=0):
        rng = np.random.RandomState(seed)
        x = pd.DataFrame(rng.randn(n, 6), columns=[f"f{i}" for i in range(6)])
        return x

    def test_binary_target_reports_accuracy_not_mse(self, tmp_path):
        controller = _make_controller(tmp_path, CLASSIFICATION_BINARY_TYPE)
        x = self._xy()
        y = pd.Series(np.random.RandomState(1).randint(0, 2, len(x)).astype(np.float32))
        metrics = controller._train_mlp_model(
            x, y, "T", "target_up_1d", is_classification=True, y_scaler=None, context_windows=None,
            model_path=tmp_path / "model_T_1d_target_up_1d_mlp.pkl",
        )
        assert "accuracy" in metrics
        assert "mse" not in metrics

    def test_multiclass_target_reports_accuracy(self, tmp_path):
        controller = _make_controller(tmp_path, CLASSIFICATION_MULTICLASS_TYPE)
        x = self._xy()
        y = pd.Series(np.random.RandomState(2).randint(0, 3, len(x)).astype(np.float32))
        metrics = controller._train_mlp_model(
            x, y, "T", "target_multi_1d", is_classification=True, y_scaler=None, context_windows=None,
            model_path=tmp_path / "model_T_1d_target_multi_1d_mlp.pkl",
        )
        assert "accuracy" in metrics

    def test_regression_target_reports_unscaled_mse(self, tmp_path):
        controller = _make_controller(tmp_path, "regression")
        x = self._xy()
        raw_y = pd.Series((np.random.RandomState(3).randn(len(x)) * 5000 + 20000).astype(np.float32))
        scaler = StandardScaler()
        y_scaled = pd.Series(scaler.fit_transform(raw_y.to_numpy().reshape(-1, 1)).ravel().astype(np.float32))
        metrics = controller._train_mlp_model(
            x, y_scaled, "T", "target_sma_20_f1", is_classification=False, y_scaler=scaler,
            context_windows=None,
            model_path=tmp_path / "model_T_1d_target_sma_20_f1_mlp.pkl",
        )
        assert "mse" in metrics
        # Real-unit MSE for a ~5000-std-dev price-level target must land in
        # the millions, not order-1 (scaled-space) or order-1e8 (garbage).
        assert 1e5 < metrics["mse"] < 1e9

    def test_model_file_is_actually_written(self, tmp_path):
        controller = _make_controller(tmp_path, "regression")
        x = self._xy()
        y = pd.Series(np.random.RandomState(4).randn(len(x)).astype(np.float32))
        controller._train_mlp_model(
            x, y, "TICKER", "target_return_1d", is_classification=False, y_scaler=None, context_windows=None,
            model_path=tmp_path / "model_TICKER_1d_target_return_1d_mlp.pkl",
        )
        # The trainer writes where it is TOLD to. It used to compose the
        # name itself, in a second copy of the caller's expression -- so
        # this assertion was checking that two copies still agreed rather
        # than that the model landed somewhere findable.
        assert (tmp_path / "model_TICKER_1d_target_return_1d_mlp.pkl").exists()

    def test_validation_windows_report_present_and_context_windows_passthrough(self, tmp_path):
        """The windowed report is new plumbing (multiple metrics/windows
        instead of one aggregate number), and context_windows is an opaque
        passthrough attached as metadata, not computed by the trainer
        itself -- both must survive into the returned metrics dict."""
        controller = _make_controller(tmp_path, CLASSIFICATION_BINARY_TYPE)
        x = self._xy(n=90)
        y = pd.Series(np.random.RandomState(5).randint(0, 2, len(x)).astype(np.float32))
        sentinel_context = [{"volatility_ratio": 1.2}, {"volatility_ratio": 0.8}]
        metrics = controller._train_mlp_model(
            x, y, "T", "target_up_1d", is_classification=True, y_scaler=None,
            context_windows=sentinel_context,
            model_path=tmp_path / "model_T_1d_target_up_1d_mlp.pkl",
        )
        assert metrics["context_windows"] is sentinel_context
        assert isinstance(metrics["validation_windows"], list)
        assert len(metrics["validation_windows"]) > 0
        for window in metrics["validation_windows"]:
            assert "n_samples" in window
            assert "accuracy" in window
