import logging

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

logger = logging.getLogger(__name__)

class AnomalyDetector:
    """
    Anomaly detector based on the Isolation Forest algorithm.
    This class is designed to be trained once and then used for detection.
    """

    def __init__(self, contamination: float = 0.1, random_state: int = 42):
        """
        Initializes the AnomalyDetector.

        Args:
            contamination (float): The proportion of outliers in the data set.
            random_state (int): The random seed for reproducibility.
        """
        self.contamination = contamination
        self.isolation_forest = IsolationForest(
            contamination=self.contamination,
            random_state=random_state,
            n_estimators=100
        )
        self._is_fitted = False
        self.feature_columns: list[str] = []
        self.feature_medians = pd.Series(dtype=float)

    def fit(self, features: pd.DataFrame):
        """
        Trains the Isolation Forest model on the provided features.
        """
        logger.info(f"Training Isolation Forest on {features.shape[1]} features...")

        numeric_features = features.select_dtypes(include=[np.number])
        if numeric_features.empty:
            logger.warning("No numeric features found for anomaly detection training. The model was not fitted.")
            return

        # Drop rows with NaNs as the model cannot handle them
        numeric_features = numeric_features.dropna()
        if numeric_features.empty:
            logger.warning("All rows contained NaNs after selection. The model was not fitted.")
            return

        self.feature_columns = list(numeric_features.columns)
        self.feature_medians = numeric_features.median(numeric_only=True)
        self.isolation_forest.fit(numeric_features)
        self._is_fitted = True
        logger.info("Isolation Forest training complete.")

    def detect(self, features: pd.DataFrame) -> pd.Series:
        """
        Detects anomalies in the given feature set.

        Returns:
            pd.Series: A series of binary flags (1 for anomaly, 0 for normal).
        """
        if not self._is_fitted:
            logger.warning("Isolation Forest has not been trained. Cannot detect anomalies.")
            return pd.Series(0, index=features.index, dtype=int)

        numeric_features = features.select_dtypes(include=[np.number])
        if numeric_features.empty:
            logger.warning("No numeric features found for anomaly detection.")
            return pd.Series(0, index=features.index, dtype=int)

        numeric_features = numeric_features.reindex(columns=self.feature_columns)
        prediction_features = self._impute_with_training_medians(numeric_features)
        anomaly_labels = self.isolation_forest.predict(prediction_features)

        # Convert labels from -1 (anomaly)/1 (normal) to 1 (anomaly)/0 (normal)
        anomaly_flags = (anomaly_labels == -1).astype(int)

        logger.info(f"Detected {anomaly_flags.sum()} anomalies out of {len(anomaly_flags)} records.")
        return pd.Series(anomaly_flags, index=features.index)

    def score_anomaly_strength(self, features: pd.DataFrame) -> pd.Series:
        """
        Returns a continuous anomaly strength score in [0, 1].
        0 = completely normal, 1 = extreme anomaly.

        Uses Isolation Forest's decision_function which returns
        negative values for anomalies and positive for normal data.
        """
        if not self._is_fitted:
            logger.warning("Isolation Forest has not been trained. Returning zero scores.")
            return pd.Series(0.0, index=features.index, dtype=float)

        numeric_features = features.select_dtypes(include=[np.number])
        if numeric_features.empty:
            return pd.Series(0.0, index=features.index, dtype=float)

        numeric_features = numeric_features.reindex(columns=self.feature_columns)
        prediction_features = self._impute_with_training_medians(numeric_features)

        # decision_function returns: positive = normal, negative = anomaly
        raw_scores = self.isolation_forest.decision_function(prediction_features)

        # Convert to [0, 1] where 1 = strong anomaly, 0 = normal
        # Clamp negative scores to [0, max_abs], then normalise
        anomaly_scores = np.clip(-raw_scores, 0, None)
        max_score = anomaly_scores.max()
        if max_score > 0:
            anomaly_scores = anomaly_scores / max_score

        return pd.Series(anomaly_scores, index=features.index, dtype=float)

    def _impute_with_training_medians(self, numeric_features: pd.DataFrame) -> pd.DataFrame:
        imputed = numeric_features.copy()
        for column in imputed.columns:
            median = self.feature_medians.get(column, np.nan)
            if not np.isfinite(median):
                median = 0.0
            imputed[column] = imputed[column].mask(imputed[column].isna(), median)
        return imputed

    @staticmethod
    def calculate_anomaly_impact_weights(anomaly_flags: pd.Series,
                                         base_weights: pd.Series | None = None,
                                         reduction_factor: float = 0.5,
                                         anomaly_scores: pd.Series | None = None) -> pd.Series:
        """
        Calculates signal weights, reducing them during anomalous periods.

        If anomaly_scores (continuous, [0,1]) are provided, the reduction is
        proportional to the anomaly strength instead of a flat binary cut.

        Args:
            anomaly_flags: Binary flags where 1 indicates an anomaly.
            base_weights: A series of base weights to modify. Defaults to 1.0.
            reduction_factor: The factor by which to reduce weights during anomalies
                              (used as the floor when continuous scores are available).
            anomaly_scores: Optional continuous anomaly strength in [0, 1].

        Returns:
            pd.Series: The adjusted weights.
        """
        if base_weights is None:
            base_weights = pd.Series(1.0, index=anomaly_flags.index)
        else:
            # Ensure alignment
            base_weights = base_weights.reindex(anomaly_flags.index, fill_value=1.0)

        anomaly_weights = base_weights.copy()

        if anomaly_scores is not None:
            # Continuous reduction: stronger anomaly -> bigger penalty
            # weight_multiplier goes from 1.0 (score=0) to reduction_factor (score=1)
            weight_multiplier = 1.0 - anomaly_scores * (1.0 - reduction_factor)
            anomaly_weights *= weight_multiplier
            n_affected = int((anomaly_scores > 0.01).sum())
            logger.info(
                f"Applied continuous anomaly reduction to {n_affected} periods "
                f"(max reduction: {(1 - weight_multiplier.min()) * 100:.1f}%)"
            )
        else:
            # Legacy binary mode
            anomaly_weights[anomaly_flags == 1] *= reduction_factor
            logger.info(f"Reduced weights for {int(anomaly_flags.sum())} anomalous periods.")

        return anomaly_weights
