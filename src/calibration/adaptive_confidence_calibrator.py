"""
Advanced Adaptive Confidence Calibrator
- Platt scaling for fast adaptation
- Isotonic regression for accuracy
- Online learning with exponential decay
- Distribution shift detection
- Graceful fallback when optuna unavailable
"""
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy import stats

try:
    from sklearn.isotonic import IsotonicRegression
    from sklearn.linear_model import LogisticRegression
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
from src.core.logging.logger import ProjectLogger
from src.utils.artifact_security import resolve_trusted_artifact_path

rng = np.random.default_rng(42)
logger = ProjectLogger.get_logger(__name__)


class AdaptiveConfidenceCalibrator:
    """
    Elite-grade confidence calibration with adaptive retraining

    Features:
    - Platt scaling (fast adaptation)
    - Isotonic regression (accuracy)
    - Online learning with exponential decay
    - Distribution shift detection
    - Graceful fallback when sklearn unavailable
    """

    def __init__(self, logger=None, window_size=500, decay_rate=0.95):
        """
        Args:
            window_size: How many recent trades to use for adaptation
            decay_rate: Exponential decay for old data (0.95 = 5% hourly decay)
        """
        self.logger = logger or ProjectLogger.get_logger(__name__)
        if not SKLEARN_AVAILABLE:
            self.logger.warning(
                '⚠️ sklearn not available. Calibrator will use simple fallback mode.'
                )
        if SKLEARN_AVAILABLE:
            self.isotonic_model = IsotonicRegression(out_of_bounds='clip')
            self.platt_model = LogisticRegression(solver='lbfgs',
                random_state=42)
        else:
            self.isotonic_model = None
            self.platt_model = None
        self.window_size = window_size
        self.decay_rate = decay_rate
        self.calibration_history = []
        self.is_isotonic_calibrated = False
        self.is_platt_calibrated = False
        self.last_retrain_time = None
        self.distribution_shift_detected = False
        self.current_accuracy = None
        self.mae = np.inf
        self.calibration_error = np.inf
        self.expected_calibration_error = np.inf
        self.simple_calibration_map = {}

    def calibrate(self, raw_confidence):
        """
        Calibration: first Platt (fast), then Isotonic (accurate)
        Falls back to simple binning if sklearn unavailable
        """
        if not SKLEARN_AVAILABLE:
            return self._simple_calibrate(raw_confidence)
        if not self.is_platt_calibrated and not self.is_isotonic_calibrated:
            return np.clip(raw_confidence, 0.01, 0.99)
        try:
            if self.is_platt_calibrated:
                platt_pred = self.platt_model.predict_proba([[raw_confidence]]
                    )[0, 1]
            else:
                platt_pred = raw_confidence
            if self.is_isotonic_calibrated:
                isotonic_pred = self.isotonic_model.predict([platt_pred])[0]
            else:
                isotonic_pred = platt_pred
            return np.clip(isotonic_pred, 0.01, 0.99)
        except Exception as e:
            self.logger.error(f'Виникла помилка: {e}', exc_info=True)
            self.logger.warning(
                f'Calibration failed: {e}. Returning raw confidence.')
            return np.clip(raw_confidence, 0.01, 0.99)

    def _simple_calibrate(self, raw_confidence):
        """
        Simple binning calibration (fallback when sklearn unavailable)
        """
        if not self.simple_calibration_map:
            return np.clip(raw_confidence, 0.01, 0.99)
        bin_idx = int(raw_confidence * 10)
        bin_idx = np.clip(bin_idx, 0, 9)
        calibrated = self.simple_calibration_map.get(bin_idx, raw_confidence)
        return np.clip(calibrated, 0.01, 0.99)

    def update_with_outcome(self, raw_confidence, actual_outcome):
        """
        Update model when we learn the result (online learning)

        Args:
            raw_confidence: Model gave this confidence
            actual_outcome: 1 if signal was correct, 0 - no
        """
        try:
            calibrated_conf = self.calibrate(raw_confidence)
            error = abs(calibrated_conf - actual_outcome)
            self.calibration_history.append({'timestamp': datetime.now(),
                'raw_confidence': raw_confidence, 'calibrated_confidence':
                calibrated_conf, 'actual_outcome': actual_outcome, 'error':
                error, 'weight': 1.0})
            if len(self.calibration_history) > self.window_size:
                self.calibration_history = self.calibration_history[-self.
                    window_size:]
            if len(self.calibration_history) % 50 == 0:
                self._check_distribution_shift()
                if self.distribution_shift_detected or len(self.
                    calibration_history) % 200 == 0:
                    self._retrain_models()
        except Exception as e:
            self.logger.error(f'Виникла помилка: {e}', exc_info=True)
            self.logger.warning(f'Update failed: {e}')
            raise

    def _apply_exponential_decay(self):
        """
        Apply exponential decay to old data
        New data has weight 1.0, old data - less
        """
        if not self.calibration_history:
            return
        now = datetime.now()
        for entry in self.calibration_history:
            age_hours = (now - entry['timestamp']).total_seconds() / 3600
            entry['weight'] = self.decay_rate ** (age_hours / 24)
            if entry['weight'] < 0.05:
                entry['weight'] = 0.0

    def _check_distribution_shift(self):
        """
        Detect when data distribution changed (concept drift)
        Use Kolmogorov-Smirnov test
        """
        if len(self.calibration_history) < 100:
            return
        mid = len(self.calibration_history) // 2
        old_outcomes = [e['actual_outcome'] for e in self.
            calibration_history[:mid]]
        new_outcomes = [e['actual_outcome'] for e in self.
            calibration_history[mid:]]
        if len(old_outcomes) > 10 and len(new_outcomes) > 10:
            ks_stat, p_value = stats.ks_2samp(old_outcomes, new_outcomes)
            self.distribution_shift_detected = p_value < 0.05
            if self.distribution_shift_detected:
                self.logger.warning(
                    f'Distribution shift detected! KS stat={ks_stat:.3f}, p={p_value:.4f}'
                    )
            else:
                if self.logger.isEnabledFor(logging.DEBUG):
                    self.logger.debug(
                        f'Distribution stable. KS stat={ks_stat:.3f}, p={p_value:.4f}'
                        )

    def _retrain_models(self):
        """Retrain both models on current data with decay."""
        if len(self.calibration_history) < 30:
            self.logger.warning('Not enough history for retraining')
            return
        try:
            self._apply_exponential_decay()
            train_data = self._prepare_calibration_data()
            if train_data is None:
                return
            raw_confs, outcomes, weights = train_data
            if SKLEARN_AVAILABLE:
                self._execute_platt_training(raw_confs, outcomes, weights)
                self._execute_isotonic_training(raw_confs, outcomes, weights)
            else:
                self._execute_simple_training(raw_confs, outcomes, weights)
            self._compute_metrics()
            self.last_retrain_time = datetime.now()
            self.logger.info(
                f'📊 Calibration metrics - MAE: {self.mae:.4f}, ECE: {self.expected_calibration_error:.4f}'
                )
        except Exception as e:
            self.logger.error(f'Retraining process failed: {e}')

    def _prepare_calibration_data(self) ->(tuple | None):
        """Extracts and filters data for retraining."""
        raw_confs = np.array([e['raw_confidence'] for e in self.
            calibration_history])
        outcomes = np.array([e['actual_outcome'] for e in self.
            calibration_history])
        weights = np.array([e['weight'] for e in self.calibration_history])
        mask = weights > 0.01
        if np.sum(mask) < 10:
            self.logger.warning('Not enough weighted samples for calibration')
            return None
        return raw_confs[mask], outcomes[mask], weights[mask]

    def _execute_platt_training(self, raw_confs, outcomes, weights):
        """Handles Platt scaling model training."""
        try:
            x_platt = raw_confs.reshape(-1, 1)
            self.platt_model.fit(x_platt, outcomes, sample_weight=weights)
            self.is_platt_calibrated = True
            self.logger.info('Platt scaling retrained')
        except Exception as e:
            self.logger.error(f'Виникла помилка: {e}', exc_info=True)
            self.logger.warning(f'Platt training failed: {e}')
            raise

    def _execute_isotonic_training(self, raw_confs, outcomes, weights):
        """Handles Isotonic regression model training via bootstrap sampling."""
        try:
            n_samples = min(len(raw_confs), 200)
            sample_indices = rng.choice(len(raw_confs), size=n_samples, p=
                weights / weights.sum())
            self.isotonic_model.fit(raw_confs[sample_indices], outcomes[
                sample_indices])
            self.is_isotonic_calibrated = True
            self.logger.info('✅ Isotonic regression retrained')
        except Exception as e:
            self.logger.error(f'Виникла помилка: {e}', exc_info=True)
            self.logger.warning(f'Isotonic training failed: {e}')
            raise

    def _compute_metrics(self):
        """
        Обчислити calibration metrics для моніторингу
        """
        try:
            if not self.calibration_history:
                return
            calibrated = np.array([e['calibrated_confidence'] for e in self
                .calibration_history])
            outcomes = np.array([e['actual_outcome'] for e in self.
                calibration_history])
            self.mae = np.mean(np.abs(calibrated - outcomes))
            n_bins = 10
            bin_edges = np.linspace(0, 1, n_bins + 1)
            ece = 0.0
            for i in range(n_bins):
                mask = (calibrated >= bin_edges[i]) & (calibrated <
                    bin_edges[i + 1])
                if mask.sum() > 0:
                    bin_accuracy = outcomes[mask].mean()
                    bin_confidence = calibrated[mask].mean()
                    bin_size = mask.sum()
                    ece += np.abs(bin_accuracy - bin_confidence) * (bin_size /
                        len(calibrated)) if len(calibrated) > 0 else 0.0
            self.expected_calibration_error = ece
            self.current_accuracy = outcomes.mean()
        except Exception as e:
            self.logger.error(f'Виникла помилка: {e}', exc_info=True)
            self.logger.warning(f'Metric computation failed: {e}')
            raise

    def save(self, filepath):
        """Зберегти обидві моделі"""
        try:
            Path(filepath).parent.mkdir(parents=True, exist_ok=True)
            data = {'isotonic_model': self.isotonic_model, 'platt_model':
                self.platt_model, 'is_isotonic_calibrated': self.
                is_isotonic_calibrated, 'is_platt_calibrated': self.
                is_platt_calibrated, 'mae': self.mae, 'ece': self.
                expected_calibration_error, 'current_accuracy': self.
                current_accuracy, 'calibration_history': self.
                calibration_history[-100:], 'last_retrain_time': self.
                last_retrain_time}
            import joblib
            joblib.dump(data, filepath)
            self.logger.info(f'✅ Calibrator saved to {filepath}')
        except Exception as e:
            self.logger.error(f'Save failed: {e}')

    def load(self, filepath):
        """Завантажити моделі калібрування"""
        try:
            path_obj = resolve_trusted_artifact_path(
                filepath,
                allowed_suffixes={'.joblib', '.pkl', '.pickle'},
                must_exist=True,
            )
            import joblib
            data = joblib.load(path_obj)  # audit-ignore: UNSAFE_MODEL_OR_PICKLE_LOAD
            self.isotonic_model = data['isotonic_model']
            self.platt_model = data['platt_model']
            self.is_isotonic_calibrated = data['is_isotonic_calibrated']
            self.is_platt_calibrated = data['is_platt_calibrated']
            self.mae = data.get('mae', np.inf)
            self.expected_calibration_error = data.get('ece', np.inf)
            self.current_accuracy = data.get('current_accuracy')
            self.calibration_history = data.get('calibration_history', [])
            self.last_retrain_time = data.get('last_retrain_time')
            self.logger.info(f'✅ Calibrator loaded from {filepath}')
            return True
        except Exception as e:
            self.logger.error(f'Виникла помилка: {e}', exc_info=True)
            self.logger.warning(f'Load failed: {e}')
            return False

    def get_calibration_report(self):
        """Генерувати звіт про calibration"""
        return {'is_calibrated': self.is_isotonic_calibrated or self.
            is_platt_calibrated, 'mae': self.mae,
            'expected_calibration_error': self.expected_calibration_error,
            'current_accuracy': self.current_accuracy, 'history_size': len(
            self.calibration_history), 'distribution_shift_detected': self.
            distribution_shift_detected, 'last_retrain': self.
            last_retrain_time.isoformat() if self.last_retrain_time else
            None, 'models_active': {'platt': self.is_platt_calibrated,
            'isotonic': self.is_isotonic_calibrated}}

    def _execute_simple_training(self, raw_confs, outcomes, weights):
        """Simple binning calibration (fallback when sklearn unavailable)"""
        try:
            for bin_idx in range(10):
                bin_min = bin_idx / 10.0
                bin_max = (bin_idx + 1) / 10.0
                mask = (raw_confs >= bin_min) & (raw_confs < bin_max)
                if np.sum(mask) > 0:
                    bin_accuracy = np.average(outcomes[mask], weights=
                        weights[mask])
                    self.simple_calibration_map[bin_idx] = bin_accuracy
            self.logger.info('✅ Simple binning calibration trained')
        except Exception as e:
            self.logger.error(f'Виникла помилка: {e}', exc_info=True)
            self.logger.warning(f'Simple training failed: {e}')
            raise
