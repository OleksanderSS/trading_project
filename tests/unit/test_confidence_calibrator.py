import pytest
import numpy as np
from src.models.ensemble.confidence_calibrator import ConfidenceCalibrator
from src.core.exceptions import DataProcessingError

def test_confidence_calibrator_isotonic():
    calibrator = ConfidenceCalibrator(method="isotonic")
    predictions = np.array([0.1, 0.4, 0.6, 0.9])
    targets = np.array([0, 0, 1, 1])
    
    results = calibrator.fit(predictions, targets)
    assert results['calibrator_type'] == 'isotonic_regression'
    assert 'metrics' in results
    assert results['metrics']['brier_score'] >= 0
    
    calibrated = calibrator.transform(predictions)
    assert len(calibrated) == 4


def test_confidence_calibrator_save_and_load(tmp_path):
    predictions = np.array([0.1, 0.4, 0.6, 0.9])
    targets = np.array([0, 0, 1, 1])
    calibrator = ConfidenceCalibrator(method="isotonic")
    calibrator.fit(predictions, targets)
    expected = calibrator.transform(predictions)

    path = tmp_path / "calibrator.joblib"
    assert calibrator.save_calibrator(str(path)) is True

    loaded = ConfidenceCalibrator(method="platt")
    assert loaded.load_calibrator(str(path)) is True
    np.testing.assert_allclose(loaded.transform(predictions), expected)

def test_confidence_calibrator_error():
    calibrator = ConfidenceCalibrator(method="platt")
    # Невалідні дані для Platt Scaling
    with pytest.raises(DataProcessingError):
        calibrator.fit(np.array([0.1]), np.array([0, 1])) # Різна довжина
