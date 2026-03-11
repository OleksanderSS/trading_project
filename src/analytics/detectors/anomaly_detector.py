import pandas as pd
import numpy as np
import logging
from sklearn.ensemble import IsolationForest
from typing import Optional

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

        # The model expects the same columns it was trained on.
        # We will predict on the numeric columns and fill NaNs with 0.
        # A more robust solution might involve column alignment and imputation.
        anomaly_labels = self.isolation_forest.predict(numeric_features.fillna(0))
        
        # Convert labels from -1 (anomaly)/1 (normal) to 1 (anomaly)/0 (normal)
        anomaly_flags = (anomaly_labels == -1).astype(int)
        
        logger.info(f"Detected {anomaly_flags.sum()} anomalies out of {len(anomaly_flags)} records.")
        return pd.Series(anomaly_flags, index=features.index)
    
    @staticmethod
    def calculate_anomaly_impact_weights(anomaly_flags: pd.Series, 
                                         base_weights: Optional[pd.Series] = None, 
                                         reduction_factor: float = 0.5) -> pd.Series:
        """
        Calculates signal weights, reducing them during anomalous periods.

        Args:
            anomaly_flags (pd.Series): Binary flags where 1 indicates an anomaly.
            base_weights (pd.Series, optional): A series of base weights to modify. Defaults to 1.0.
            reduction_factor (float): The factor by which to reduce weights during anomalies (e.g., 0.5 for 50% reduction).

        Returns:
            pd.Series: The adjusted weights.
        """
        if base_weights is None:
            base_weights = pd.Series(1.0, index=anomaly_flags.index)
        else:
            # Ensure alignment
            base_weights = base_weights.reindex(anomaly_flags.index, fill_value=1.0)

        anomaly_weights = base_weights.copy()
        anomaly_weights[anomaly_flags == 1] *= reduction_factor
        
        logger.info(f"Reduced weights for {int(anomaly_flags.sum())} anomalous periods.")
        return anomaly_weights
