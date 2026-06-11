"""Unit tests for CalibrationEngine and AdaptiveConfidenceCalibrator."""

from unittest.mock import MagicMock

import numpy as np


class TestCalibrationEngine:
    """Tests for CalibrationEngine in src/calibration/calibration_engine.py"""

    def test_define_hyperparameter_space_returns_all_keys(self):
        """Test that define_hyperparameter_space returns all expected keys."""
        from src.calibration.calibration_engine import CalibrationEngine

        engine = CalibrationEngine.__new__(CalibrationEngine)
        expected_keys = {
            'actor_lr', 'critic_lr', 'hidden_dim', 'num_layers',
            'batch_size', 'replay_buffer_size', 'gamma', 'tau',
            'exploration_noise', 'dropout', 'weight_decay',
            'actor_n_estimators', 'actor_max_depth',
            'actor_min_samples_split', 'actor_min_samples_leaf'
        }

        mock_trial = MagicMock()
        mock_trial.suggest_float.return_value = 0.001
        mock_trial.suggest_categorical.return_value = 128
        mock_trial.suggest_int.return_value = 32

        result = engine.define_hyperparameter_space(mock_trial)

        assert set(result.keys()) == expected_keys
        assert 'actor_lr' in result
        assert 'actor_n_estimators' in result

    def test_mock_evaluation_returns_float(self):
        """Test that _mock_evaluation returns a float."""
        from src.calibration.calibration_engine import CalibrationEngine

        engine = CalibrationEngine.__new__(CalibrationEngine)
        result = engine._mock_evaluation({'actor_lr': 0.001, 'hidden_dim': 256})

        assert isinstance(result, float)
        assert result > 0

    def test_calculate_sharpe_ratio_with_perfect_prediction(self):
        """Test Sharpe ratio calculation with perfect predictions.

        With perfect predictions the model correctly follows every return direction,
        so realised trade returns equal the absolute y_true values → strictly positive
        mean return → Sharpe > 0.  The implementation clips Sharpe to [-5, 5].
        """
        from src.calibration.calibration_engine import CalibrationEngine

        engine = CalibrationEngine.__new__(CalibrationEngine)
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        result = engine._calculate_sharpe_ratio(y_true, y_pred)
        # Perfect predictions → positive Sharpe (clipped at 5.0 by the implementation)
        assert result > 0.0

    def test_calculate_sharpe_ratio_with_variance(self):
        """Test Sharpe ratio calculation with varying predictions."""
        from src.calibration.calibration_engine import CalibrationEngine

        engine = CalibrationEngine.__new__(CalibrationEngine)
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.1, 2.2, 2.9, 4.1, 4.8])

        result = engine._calculate_sharpe_ratio(y_true, y_pred)
        assert isinstance(result, float)
        assert -5.0 <= result <= 5.0

    def test_evaluate_on_synthetic_empty_scenarios(self):
        """Test synthetic evaluation with empty scenarios."""
        from src.calibration.calibration_engine import CalibrationEngine

        engine = CalibrationEngine.__new__(CalibrationEngine)
        result = engine._evaluate_on_synthetic(MagicMock(), {'typical': [], 'shock': [], 'context': []})

        assert result == 0.0

    def test_evaluate_on_synthetic_with_scenarios(self):
        """Test synthetic evaluation with mock scenarios."""
        from src.calibration.calibration_engine import CalibrationEngine

        engine = CalibrationEngine.__new__(CalibrationEngine)
        scenarios = {
            'typical': [{'metrics': {'sharpe_ratio': 1.5}}, {'metrics': {'sharpe_ratio': -0.5}}],
            'shock': [{'metrics': {'sharpe_ratio': 0.8}}],
            'context': []
        }
        result = engine._evaluate_on_synthetic(MagicMock(), scenarios)

        assert isinstance(result, float)
        assert result >= 0

    def test_chronological_split_preserves_order(self):
        """Validation data must come after training data."""
        import pandas as pd
        from src.calibration.calibration_engine import CalibrationEngine

        engine = CalibrationEngine.__new__(CalibrationEngine)
        X = pd.DataFrame({"feature": range(10)})
        y = pd.Series(range(10))

        X_train, X_val, y_train, y_val = engine._chronological_split(X, y)

        assert X_train["feature"].tolist() == list(range(8))
        assert X_val["feature"].tolist() == [8, 9]
        assert y_train.tolist() == list(range(8))
        assert y_val.tolist() == [8, 9]


class TestAdaptiveConfidenceCalibrator:
    """Tests for AdaptiveConfidenceCalibrator in src/calibration/adaptive_confidence_calibrator.py"""

    def test_calibrate_clips_extreme_values(self):
        """Test that calibration clips values to [0.01, 0.99]."""
        from src.calibration.adaptive_confidence_calibrator import AdaptiveConfidenceCalibrator

        calibrator = AdaptiveConfidenceCalibrator()

        low = calibrator.calibrate(0.0)
        high = calibrator.calibrate(1.0)

        assert low >= 0.01
        assert high <= 0.99

    def test_update_with_outcome_records_history(self):
        """Test that update_with_outcome records to history."""
        from src.calibration.adaptive_confidence_calibrator import AdaptiveConfidenceCalibrator

        calibrator = AdaptiveConfidenceCalibrator()
        calibrator.update_with_outcome(0.7, 1)

        assert len(calibrator.calibration_history) == 1
        entry = calibrator.calibration_history[0]
        assert entry['raw_confidence'] == 0.7
        assert entry['actual_outcome'] == 1

    def test_get_calibration_report_structure(self):
        """Test that calibration report has expected structure."""
        from src.calibration.adaptive_confidence_calibrator import AdaptiveConfidenceCalibrator

        calibrator = AdaptiveConfidenceCalibrator()
        report = calibrator.get_calibration_report()

        assert 'is_calibrated' in report
        assert 'mae' in report
        assert 'expected_calibration_error' in report
        assert 'history_size' in report
        assert 'distribution_shift_detected' in report
        assert 'models_active' in report

    def test_window_size_limit(self):
        """Test that calibration history respects window size limit."""
        from src.calibration.adaptive_confidence_calibrator import AdaptiveConfidenceCalibrator

        calibrator = AdaptiveConfidenceCalibrator(window_size=10)

        for i in range(25):
            calibrator.update_with_outcome(0.5, i % 2)

        assert len(calibrator.calibration_history) <= 10
