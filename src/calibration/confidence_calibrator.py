"""
Confidence Calibrator - Calibrating raw confidence to real accuracy
"""

import numpy as np
from sklearn.isotonic import IsotonicRegression
import pickle
from pathlib import Path

from src.core.logging.logger import ProjectLogger

logger = ProjectLogger.get_logger(__name__)

class ConfidenceCalibrator:
    """Calibrating confidence scores through isotonic regression"""

    def __init__(self, logger=None):
        self.calibration_model = IsotonicRegression(out_of_bounds='clip')
        self.is_calibrated = False
        self.logger = logger or ProjectLogger.get_logger(__name__)
        self.calibration_accuracy = None

    def calibrate_on_history(self, diary_engine, window=500):
        """
        Train on historical data

        Args:
            diary_engine: DiaryEngine instance
            window: How many recent trades to use
        """
        try:
            # 1. Gather history
            history = diary_engine.get_recent_trades(window=window)

            if len(history) < 50:
                self.logger.warning(
                    f"Not enough history: {len(history)} trades. Skip calibration."
                )
                return False

            # 2. Calculate accuracy for bins
            confidence_bins = np.linspace(0, 1, 11)
            accuracies = []
            counts = []

            for i in range(len(confidence_bins) - 1):
                bin_min = confidence_bins[i]
                bin_max = confidence_bins[i + 1]

                mask = (
                    (history['confidence'] >= bin_min) &
                    (history['confidence'] < bin_max)
                )
                count = mask.sum()

                if count > 0:
                    # Calculate accuracy for this bin
                    accuracy = (
                        history[mask]['prediction_sign'] ==
                        history[mask]['actual_sign']
                    ).mean()
                    accuracies.append(accuracy)
                    counts.append(count)
                else:
                    accuracies.append(0.5)  # Default for empty bins
                    counts.append(0)

            # 3. Train isotonic regression
            self.calibration_model.fit(confidence_bins[:-1], accuracies)
            self.is_calibrated = True
            self.calibration_accuracy = np.mean(accuracies)

            self.logger.info(
                f"Calibration complete: {len(history)} trades, "
                f"avg accuracy: {self.calibration_accuracy:.2%}"
            )

            return True

        except Exception as e:
            self.logger.error(f"Calibration failed: {e}")
            return False

    def calibrate(self, raw_confidence):
        """
        Convert raw confidence to calibrated

        Args:
            raw_confidence: 0.0-1.0

        Returns:
            Calibrated confidence 0.01-0.99
        """
        if not self.is_calibrated:
            # If not calibrated, return raw
            return np.clip(raw_confidence, 0.01, 0.99)

        try:
            calibrated = self.calibration_model.predict([raw_confidence])[0]
            return np.clip(calibrated, 0.01, 0.99)
        except Exception as e:
            self.logger.warning(f"Calibration prediction failed: {e}")
            return np.clip(raw_confidence, 0.01, 0.99)

    def save(self, filepath):
        """Save calibration model"""
        try:
            Path(filepath).parent.mkdir(parents=True, exist_ok=True)
            with open(filepath, 'wb') as f:
                pickle.dump({
                    'model': self.calibration_model,
                    'is_calibrated': self.is_calibrated,
                    'calibration_accuracy': self.calibration_accuracy
                }, f)
            self.logger.info(f"Calibration model saved to {filepath}")
        except Exception as e:
            self.logger.error(f"Failed to save calibration model: {e}")

    def load(self, filepath):
        """Load calibration model"""
        try:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
                self.calibration_model = data['model']
                self.is_calibrated = data['is_calibrated']
                self.calibration_accuracy = data.get('calibration_accuracy')
            self.logger.info(f"Calibration model loaded from {filepath}")
            return True
        except Exception as e:
            self.logger.warning(f"Failed to load calibration model: {e}")
            return False