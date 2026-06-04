import logging
import os
from typing import Any

import joblib

# src/processing/normalization_manager.py
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from src.core.logging.logger import ProjectLogger
from src.utils.artifact_security import resolve_trusted_artifact_path

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

        Args:
            scaler_dir (str): Directory to save or load scalers.
        """
        self.scaler_dir = scaler_dir
        self.scalers: dict[str, MinMaxScaler | StandardScaler] = {}
        if not os.path.exists(self.scaler_dir):
            os.makedirs(self.scaler_dir)
            logger.info(f"Created scaler directory at: {self.scaler_dir}")

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
            if logger.isEnabledFor(logging.DEBUG):
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
        """
        data_transformed = data.copy()
        for feature, scaler in self.scalers.items():
            if feature in data_transformed.columns:
                feature_data = data_transformed[[feature]].dropna()
                if not feature_data.empty:
                    data_transformed.loc[feature_data.index, feature] = (
                        scaler.transform(feature_data).ravel()
                    )
                    if logger.isEnabledFor(logging.DEBUG):
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
                if logger.isEnabledFor(logging.DEBUG):
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

    def load_scalers(self, features: list[str] | None = None):
        """
        Loads scalers from disk. If features are specified, loads only those.
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
                trusted_path = resolve_trusted_artifact_path(
                    scaler_path,
                    allowed_roots=(self.scaler_dir,),
                    allowed_suffixes={'.joblib'},
                    must_exist=True,
                )
                scaler = joblib.load(trusted_path)  # audit-ignore: UNSAFE_MODEL_OR_PICKLE_LOAD
                self.scalers[feature_name] = scaler
                logger.info(f"Successfully loaded scaler for '{feature_name}'.")
            except FileNotFoundError:
                logger.warning(f"Scaler file not found for feature '{feature_name}' at {scaler_path}.")
            except Exception as e:
                logger.error(f"Failed to load scaler for '{feature_name}': {e}")
