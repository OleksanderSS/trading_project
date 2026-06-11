# src/processing/normalization_manager.py
import os
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from src.core.logging.logger import ProjectLogger

logger = ProjectLogger.get_logger("NormalizationManager")

class NormalizationManager:
    """
    Manages the normalization and denormalization of features.
    This class fits scalers to the training data and saves them to be applied
    consistently to training, validation, and testing datasets.
    """

    def __init__(self, scaler_dir: str = "data/scalers"):
        """
        Initializes the NormalizationManager.
        """
        self.scaler_dir = Path(scaler_dir)
        self.scalers: dict[str, Any] = {}
        self.unified_scaler_path = self.scaler_dir / "unified_scalers.joblib"

        if not self.scaler_dir.exists():
            self.scaler_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created scaler directory at: {self.scaler_dir}")

    def save_all_scalers(self):
        """Saves all fitted scalers into a single unified file."""
        if not self.scalers:
            logger.warning("No scalers to save.")
            return

        try:
            joblib.dump(self.scalers, self.unified_scaler_path)
            logger.info(f"✅ All {len(self.scalers)} scalers saved to {self.unified_scaler_path}")
        except Exception as e:
            logger.error(f"❌ Failed to save unified scalers: {e}")

    def load_scalers(self, features: list[str] | None = None):
        """Loads all scalers from the unified file."""
        if not self.unified_scaler_path.exists():
            logger.warning(f"Unified scaler file not found at {self.unified_scaler_path}")
            # Try legacy loading for backward compatibility during transition
            self._load_legacy_scalers(features)
            return

        try:
            loaded_scalers = joblib.load(self.unified_scaler_path)
            if features:
                self.scalers.update({f: loaded_scalers[f] for f in features if f in loaded_scalers})
            else:
                self.scalers.update(loaded_scalers)
            logger.info(f"✅ Successfully loaded {len(self.scalers)} scalers from unified file.")
        except Exception as e:
            logger.error(f"❌ Failed to load unified scalers: {e}")

    # Legacy fallback loader is defined below.

    def fit_scalers(self, data: pd.DataFrame, features_to_normalize: list[dict[str, Any]]):
        """
        Fits scalers for the specified features and saves them.

        Args:
            data (pd.DataFrame): The training dataframe.
            features_to_normalize (List[Dict[str, Any]]): A list of dictionaries,
                each specifying a 'feature' name and 'scaler_type' ('min_max' or 'standard').
        """
        if not features_to_normalize:
            logger.info("No features to normalize. Skipping scaler fitting.")
            return

        logger.info(f"Fitting scalers for {len(features_to_normalize)} features...")
        for config in features_to_normalize:
            self._fit_single_scaler(data, config)

        logger.info("Scaler fitting complete.")

    def _fit_single_scaler(self, data: pd.DataFrame, config: dict[str, Any]):
        """Fit a single scaler for a feature."""
        feature = config['feature']
        scaler_type = config.get('scaler_type', 'min_max')

        if not self._validate_feature_exists(data, feature):
            return

        if not self._validate_feature_data(data, feature):
            return

        scaler = self._create_scaler(scaler_type, feature)
        self._fit_and_save_scaler(data, scaler, feature, scaler_type)

    def _validate_feature_exists(self, data: pd.DataFrame, feature: str) -> bool:
        """Validate that feature exists in DataFrame."""
        if feature not in data.columns:
            logger.warning(f"Feature '{feature}' not found in DataFrame. Skipping.")
            return False
        return True

    def _validate_feature_data(self, data: pd.DataFrame, feature: str) -> bool:
        """Validate that feature has data after dropping NaNs."""
        feature_data = data[[feature]].dropna()
        if feature_data.empty:
            logger.warning(f"No data for feature '{feature}' after dropping NaNs. Skipping.")
            return False
        return True

    def _create_scaler(self, scaler_type: str, feature: str):
        """Create appropriate scaler instance."""
        if scaler_type == 'min_max':
            return MinMaxScaler()
        if scaler_type == 'standard':
            return StandardScaler()

        logger.warning(f"Unknown scaler type '{scaler_type}' for feature '{feature}'. Defaulting to MinMaxScaler.")
        return MinMaxScaler()

    def _fit_and_save_scaler(self, data: pd.DataFrame, scaler, feature: str, scaler_type: str):
        """Fit scaler and save it with error handling."""
        try:
            feature_data = data[[feature]].dropna()
            scaler.fit(feature_data)
            self.scalers[feature] = scaler
            self._save_scaler(feature)
            logger.debug(f"Fitted and saved '{scaler_type}' scaler for feature '{feature}'.")
        except Exception as e:
            logger.error(f"Error fitting scaler for feature '{feature}': {e}")

    def transform_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Applies pre-fitted normalization to the data.

        Args:
            data (pd.DataFrame): The dataframe to transform.

        Returns:
            pd.DataFrame: The dataframe with normalized features.
        
        Raises:
            KeyError: If an expected feature is missing in the data.
        """
        data_transformed = data.copy()
        for feature, scaler in self.scalers.items():
            if feature not in data_transformed.columns:
                # ✅ ENHANCED: Fail-fast if expected feature is missing
                logger.error(f"❌ Feature '{feature}' missing during transform. Integrity violation.")
                raise KeyError(f"Feature '{feature}' is missing, required by NormalizationManager.")
            
            feature_data = data_transformed[[feature]].dropna()
            if not feature_data.empty:
                data_transformed.loc[feature_data.index, feature] = scaler.transform(feature_data)
                logger.debug(f"Transformed feature '{feature}'.")
        return data_transformed

    def inverse_transform_feature(self, data: pd.DataFrame, feature: str) -> pd.DataFrame:
        """
        Applies inverse transformation to a single feature in the dataframe.

        Args:
            data (pd.DataFrame): The dataframe with the normalized feature.
            feature (str): The name of the feature to inverse-transform.

        Returns:
            pd.DataFrame: DataFrame with the feature in its original scale.
        """
        data_inv = data.copy()
        if feature in self.scalers:
            scaler = self.scalers[feature]
            feature_data = data_inv[[feature]].dropna()
            if not feature_data.empty:
                data_inv.loc[feature_data.index, feature] = scaler.inverse_transform(feature_data)
                logger.debug(f"Inverse transformed feature '{feature}'.")
        else:
            logger.warning(f"No scaler found for feature '{feature}'. Inverse transform skipped.")

        return data_inv

    def _save_scaler(self, feature: str):
        """Saves a single scaler to disk."""
        scaler_path = os.path.join(self.scaler_dir, f"{feature}_scaler.joblib")
        try:
            joblib.dump(self.scalers[feature], scaler_path)
            logger.info(f"Scaler for '{feature}' saved to {scaler_path}")
        except Exception as e:
            logger.error(f"Failed to save scaler for '{feature}': {e}")

    def _load_legacy_scalers(self, features: list[str] | None = None):
        """
        Loads scalers from disk as individual files. If features are specified, loads only those.
        Otherwise, loads all scalers found in the directory.
        """
        logger.info(f"Loading scalers from {self.scaler_dir}...")
        if features:
            files_to_load = [f"{f}_scaler.joblib" for f in features]
        else:
            files_to_load = [f for f in os.listdir(self.scaler_dir) if f.endswith("_scaler.joblib")]

        for filename in files_to_load:
            feature_name = filename.replace("_scaler.joblib", "")
            scaler_path = os.path.join(self.scaler_dir, filename)
            try:
                scaler = joblib.load(scaler_path)
                self.scalers[feature_name] = scaler
                logger.info(f"Successfully loaded scaler for '{feature_name}'.")
            except FileNotFoundError:
                logger.warning(f"Scaler file not found for feature '{feature_name}' at {scaler_path}.")
            except Exception as e:
                logger.error(f"Failed to load scaler for '{feature_name}': {e}")
