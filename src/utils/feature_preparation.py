# src/utils/feature_preparation.py
"""
Utility functions for feature preparation and model-feature alignment.
Ensures consistency between raw datasets and architecture-specific input requirements.
"""


import numpy as np
import pandas as pd

from src.core.logging.logger import ProjectLogger

# Initialize standardized project logger
logger = ProjectLogger.get_logger("FeaturePreparation")

def prepare_features_for_training(
    features_df: pd.DataFrame,
    remove_metadata: bool = True,
    fill_na: bool = True,
    verbose: bool = False
) -> tuple[pd.DataFrame, list[str]]:
    """
    Prepares a features DataFrame for model training ingestion.
    Filters out non-numeric metadata and handles data integrity.

    Args:
        features_df: Input DataFrame containing features and metadata.
        remove_metadata: If True, drops predefined non-feature columns.
        fill_na: If True, replaces missing values with zeros.
        verbose: If True, logs detailed diagnostic information.

    Returns:
        Tuple containing the processed numeric DataFrame and the list of feature labels.
    """
    df_clean = _initialize_clean_dataframe(features_df)
    df_clean = _process_metadata_columns(df_clean, remove_metadata, verbose)
    df_numeric, numeric_cols = _extract_and_clean_features(df_clean, fill_na, verbose)

    if verbose:
        _audit_primary_signals(numeric_cols)

    return df_numeric, numeric_cols

def _initialize_clean_dataframe(features_df: pd.DataFrame) -> pd.DataFrame:
    """Initialize a clean copy of the features DataFrame."""
    return features_df.copy()

def _process_metadata_columns(df: pd.DataFrame, remove_metadata: bool, verbose: bool) -> pd.DataFrame:
    """Process metadata columns based on the remove_metadata flag."""
    if remove_metadata:
        return _prune_metadata_columns(df, verbose)
    return df

def _extract_and_clean_features(df: pd.DataFrame, fill_na: bool, verbose: bool) -> tuple[pd.DataFrame, list[str]]:
    """Extract numeric features and clean the data."""
    df_numeric, numeric_cols = _extract_numeric_features(df, verbose)
    df_numeric = _clean_numeric_data(df_numeric, fill_na, verbose)
    return df_numeric, numeric_cols

def _prune_metadata_columns(df: pd.DataFrame, verbose: bool) -> pd.DataFrame:
        """Remove predefined metadata columns from DataFrame."""
        metadata_cols = [
            'news_id',           # News record identifier
            'news_title',        # Raw headline text
            'ticker',            # Asset ticker symbol
            'datetime',          # Primary temporal anchor
            'published_at',      # Source publication timestamp
        ]

        cols_to_drop = [c for c in metadata_cols if c in df.columns]
        if cols_to_drop:
            df = df.drop(columns=cols_to_drop)
            if verbose:
                logger.debug(f"Metadata pruning: Dropped {len(cols_to_drop)} columns ({cols_to_drop})")
        return df

def _extract_numeric_features(df: pd.DataFrame, verbose: bool) -> tuple[pd.DataFrame, list[str]]:
        """Select exclusively numeric features for the core model payload."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        df_numeric = df[numeric_cols]

        if verbose:
            logger.info(f"Feature Vectorization: Identified {len(numeric_cols)} numeric signals.")
            logger.debug(f"Pre-cleansing shape: {df_numeric.shape}")

        return df_numeric, numeric_cols

def _clean_numeric_data(df: pd.DataFrame, fill_na: bool, verbose: bool) -> pd.DataFrame:
    """Handle missing values and infinity in numeric data using smart filling."""
    # Handle missing values with intelligent filling
    if fill_na:
        try:
            # Use SmartMissingDataHandler instead of destructive zero-filling
            from .smart_missing_data_handler import SmartMissingDataHandler

            smart_handler = SmartMissingDataHandler()
            df = smart_handler.handle_missing_data(df, verbose=verbose)

            if verbose:
                # Get fill statistics for monitoring
                stats = smart_handler.get_fill_statistics(
                    df.copy().fillna(np.nan),  # Original with NaNs
                    df  # Filled version
                )
                logger.info(f"SmartMissingDataHandler: Fill efficiency {stats['fill_efficiency']:.2%}")
                logger.info(f"SmartMissingDataHandler: Filled {stats['columns_filled']}/{stats['total_columns']} columns")

                # Detect fill anomalies for quality monitoring
                try:
                    from .missing_data_anomaly_detector import MissingDataAnomalyDetector
                    anomaly_detector = MissingDataAnomalyDetector()
                    anomalies = anomaly_detector.detect_fill_anomalies(
                        df.copy().fillna(np.nan),  # Original with NaNs
                        df  # Filled version
                    )

                    if anomalies:
                        anomaly_report = anomaly_detector.generate_anomaly_report(anomalies)
                        logger.warning(f"SmartMissingDataHandler: Detected {len(anomalies)} fill anomalies")
                        logger.warning(f"Quality score: {anomaly_report['quality_score']:.1f}/100")

                        # Log top anomalies
                        for anomaly in anomalies[:3]:
                            logger.warning(f"  {anomaly['type']} in {anomaly['column']}: {anomaly.get('severity', 'unknown')}")

                        # Log recommendations
                        for rec in anomaly_report['recommendations'][:3]:
                            logger.info(f"  Recommendation: {rec}")
                    else:
                        logger.info("SmartMissingDataHandler: No fill anomalies detected - quality is excellent")

                except ImportError:
                    logger.debug("MissingDataAnomalyDetector not available - skipping anomaly detection")
                except Exception as e:
                    logger.error(f"Error in anomaly detection: {e}")

        except ImportError:
            # Fallback to original zero-fill if smart handler not available
            logger.warning("SmartMissingDataHandler not available, using zero-fill fallback")
            df = df.fillna(0.0)
        except Exception as e:
            logger.error(f"SmartMissingDataHandler failed: {e}, using zero-fill fallback")
            df = df.fillna(0.0)

    # Clamp infinity values to maintain numeric stability during gradient updates
    df = df.replace([np.inf, -np.inf], 0.0)

    if verbose:
        logger.debug(f"Post-cleansing shape: {df.shape}")
        total_missing = df.isna().sum().sum()
        total_inf = np.isinf(df.values).sum()

        if total_missing > 0 or total_inf > 0:
            logger.warning(f"Integrity Guard: Detected {total_missing} NaNs and {total_inf} Infs post-cleansing.")

    return df

def _audit_primary_signals(numeric_cols: list[str]) -> None:
        """Audit critical feature presence for logging."""
        primary_signals = ['news_sentiment', 'AMD_15m_close', 'AMD_1h_close']
        for signal in primary_signals:
            if signal in numeric_cols:
                logger.debug(f"Signal Audit: {signal} validated and included.")
            else:
                logger.debug(f"Signal Audit: {signal} is absent from the current batch mapping.")


def align_features_with_model(
    X: pd.DataFrame,
    model_feature_names: list[str]
) -> pd.DataFrame:
    """
    Aligns the current feature matrix with the specific column set expected by a trained architecture.
    Ensures input consistency during inference cycles.

    Args:
        X: Active feature DataFrame.
        model_feature_names: Exact list of features the model was trained on.

    Returns:
        DataFrame containing only the relevant features in the required architectural order.
    """

    # Filter for intersection between current state and model requirements
    available_features = [f for f in model_feature_names if f in X.columns]

    # Log potential feature drift or missing inputs
    if len(available_features) < len(model_feature_names):
        missing = set(model_feature_names) - set(available_features)
        logger.warning(
            f"Feature Alignment Discrepancy: Model expected {len(model_feature_names)} signals, "
            f"but only found {len(available_features)}. Missing: {list(missing)[:5]}..."
        )

    # Enforce strict column order to match model weight expectations
    return X[available_features]
