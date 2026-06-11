import json
import os
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression

from src.core.logging.logger import ProjectLogger

# Early logger definition for Colab compatibility
logger = ProjectLogger.get_logger(__name__)

# Enhanced fallback for Colab sys.path issues
def _safe_import_with_fallback(module_path: str, function_name: str, fallback_func):
    """Safely import with fallback for Colab path issues"""
    try:
        # Try direct import first
        module = __import__(module_path, fromlist=[function_name])
        return getattr(module, function_name)
    except (ImportError, AttributeError) as e:
        # Use print instead of logger since logger might not be defined yet
        print(f"Warning: Could not import {function_name} from {module_path}: {e}")
        return fallback_func

# Fallback for check_freshness_quick if import fails
try:
    from src.monitoring.data_freshness_monitor import check_freshness_quick
except ImportError:
    print("Warning: Could not import check_freshness_quick, using fallback")
    def check_freshness_quick(data_source: str, max_age_hours: int = 24) -> bool:
        """Fallback implementation - always returns True"""
        return True

# Fallback for check_feature_drift if import fails
try:
    from src.monitoring.feature_drift_monitor import check_feature_drift
except ImportError:
    print("Warning: Could not import check_feature_drift, using fallback")
    def check_feature_drift(X: Any, feature_names: list[str] = [], threshold: float = 0.1) -> dict[str, dict[str, Any]]:  # type: ignore[misc]
        """Fallback implementation - always returns empty dict"""
        return {}

# Additional fallback for path-related import issues
def _ensure_monitoring_functions():
    """Ensure monitoring functions are available with multiple fallback strategies"""
    global check_freshness_quick, check_feature_drift

    # Try to import if not already available
    if 'check_freshness_quick' not in globals() or check_freshness_quick is None:
        check_freshness_quick = _safe_import_with_fallback(
            'src.monitoring.data_freshness_monitor',
            'check_freshness_quick',
            lambda data_source, max_age_hours=24: True
        )

    if 'check_feature_drift' not in globals() or check_feature_drift is None:
        check_feature_drift = _safe_import_with_fallback(
            'src.monitoring.feature_drift_monitor',
            'check_feature_drift',
            lambda *args, **kwargs: True
        )

# Ensure functions are available at module level
_ensure_monitoring_functions()

class SmartFeatureSelector:
    """
    Selects features based on a voting ensemble of methods, now with regime-specific caching.
    """

    def __init__(self, storage_path: str | None = None, min_volatility: float = 0.0001):
        self.logger = logger  # Initialize logger from module level

        # Load configuration from YAML
        try:
            from src.config.unified_config_manager import get_current_config
            config_manager = get_current_config()
            smart_config = config_manager.get('features', {}).get('smart_selection', {})

            # Use YAML config or defaults
            self.min_volatility = smart_config.get('min_volatility', min_volatility)
            self.correlation_method = smart_config.get('correlation_method', 'spearman')
            self.mi_random_state = smart_config.get('mi_random_state', 42)
            self.lgbm_params = smart_config.get('lgbm_params', {
                'objective': 'binary',  # Will be overridden based on task type
                'verbosity': -1,
                'force_col_wise': True,
                'num_leaves': 31,
                'learning_rate': 0.05,
                'num_boost_round': 50
            })
            self.rf_params = smart_config.get('rf_params', {
                'n_estimators': 50,
                'max_depth': 10,
                'random_state': 42,
                'n_jobs': -1
            })
            self.variance_epsilon = smart_config.get('variance_epsilon', 1e-10)

            # Storage path
            if storage_path is None:
                storage_path = smart_config.get('cache_path', 'data/cache/smart_features.json')

            self.logger.info("✅ SmartFeatureSelector loaded config from YAML")

        except Exception as e:
            self.logger.warning(f"Could not load YAML config, using defaults: {e}")
            # Fallback to defaults
            self.min_volatility = min_volatility
            self.correlation_method = 'spearman'
            self.mi_random_state = 42
            self.lgbm_params = {
                'objective': 'binary',
                'verbosity': -1,
                'force_col_wise': True,
                'num_leaves': 31,
                'learning_rate': 0.05,
                'num_boost_round': 50
            }
            self.rf_params = {
                'n_estimators': 50,
                'max_depth': 10,
                'random_state': 42,
                'n_jobs': -1
            }
            self.variance_epsilon = 1e-10

            if storage_path is None:
                storage_path = os.path.join('data', 'cache', 'selected_features.json')

        self.storage_path = storage_path

        os.makedirs(os.path.dirname(self.storage_path), exist_ok=True)
        self.cache = self._load_storage()

    def _load_storage(self) -> dict[str, Any]:
        if os.path.exists(self.storage_path):
            try:
                with open(self.storage_path) as f:
                    data = json.load(f)
                    if isinstance(data, dict):
                        return data
                    return {}
            except (OSError, json.JSONDecodeError) as e:
                logger.error(f"Failed to load feature cache from {self.storage_path}: {e}")
        return {}

    def _save_storage(self):
        try:
            os.makedirs(os.path.dirname(self.storage_path), exist_ok=True)
            with open(self.storage_path, 'w') as f:
                json.dump(self.cache, f, indent=4)
        except OSError as e:
            logger.error(f"Failed to save feature cache to {self.storage_path}: {e}")

    def select(self, features_df: pd.DataFrame, target_series: pd.Series, context_id: str,
               is_classification: bool = True, market_regime: str = "normal",
               force_recalculate: bool = False, max_features: int | None = None) -> list[str]:
        """
        Selects features, caching them based on both context, market regime, and max_features.

        Args:
            features_df: Feature dataframe
            target_series: Target series
            context_id: Context identifier (e.g., "AMD_target_return_1d_mlp")
            is_classification: Whether this is a classification task
            market_regime: Market regime (normal, volatile, trending)
            force_recalculate: Force recalculation even if cached
            max_features: Maximum number of features to select (None = no limit)
        """
        # Include max_features in cache key so each model gets its own features
        regime_context_id = f"{context_id}_{market_regime}_{max_features}"

        if not force_recalculate and regime_context_id in self.cache:
            cached_data = self.cache[regime_context_id]
            # Basic validation: ensure input columns match cached ones.
            if set(cached_data.get("input_features", [])) == set(features_df.columns):
                logger.info(f"Using cached features for {regime_context_id}...")
                selected_features = cached_data.get("selected_features", [])
                if isinstance(selected_features, list):
                    return [str(f) for f in selected_features]
                return []

        logger.info(f"Running dynamic feature selection for {regime_context_id}...")

        # Adaptive volatility threshold based on dataset size
        data_size = len(target_series)
        adaptive_min_volatility = self.min_volatility

        # For small datasets, use lower threshold
        if data_size < 30:
            adaptive_min_volatility = self.min_volatility * 0.1  # 10x lower for small datasets
        elif data_size < 50:
            adaptive_min_volatility = self.min_volatility * 0.3  # 3x lower for medium datasets

        # Pre-filtering and cleaning with adaptive threshold
        if target_series.std() < adaptive_min_volatility:
            logger.warning(f"Target volatility {target_series.std():.6f} is below adaptive threshold {adaptive_min_volatility:.6f} for {regime_context_id} (size: {data_size}). Using all features.")
            # Don't skip - just log and continue with all features
        else:
            logger.info(f"Target volatility {target_series.std():.6f} is above threshold {adaptive_min_volatility:.6f} for {regime_context_id} (size: {data_size}).")

        features_clean = self._clean_data(features_df)
        if features_clean.empty:
            logger.warning(f"No valid features after cleaning for {regime_context_id}.")
            return []

        # Dynamic voting based on regime
        methods = self._get_methods_for_regime(market_regime)
        scores = pd.Series(0.0, index=features_clean.columns, dtype=float)  #

        for method_func, weight in methods.items():
            try:
                ranked_features = method_func(features_clean, target_series, is_classification)
                if ranked_features is None:
                    logger.debug(f"Method {method_func.__name__} returned None, skipping")
                    continue # Method was disabled
                top_n = max(1, int(len(ranked_features) * 0.3))
                scores.loc[ranked_features.head(top_n).index] += weight
                logger.debug(f"Method {method_func.__name__} added weight {weight} to {top_n} features")
            except Exception as e:
                logger.error(f"Feature selection method {method_func.__name__} failed for {regime_context_id}: {e}", exc_info=True)

        # Select features with a score above the median score of features that were selected at all
        if scores.sum() == 0:
            logger.warning(f"No features were selected for {regime_context_id}. Returning empty list.")
            return []

        positive_scores = scores[scores > 0]
        selection_threshold = positive_scores.median()
        selected = positive_scores[positive_scores >= selection_threshold].index.tolist()

        # Apply max_features limit if specified
        if max_features is not None and len(selected) > max_features:
            # Sort by score and keep top max_features
            selected_with_scores = positive_scores[positive_scores >= selection_threshold].sort_values(ascending=False)
            selected = selected_with_scores.head(max_features).index.tolist()
            logger.info(f"Limited selected features to {max_features} (from {len(positive_scores[positive_scores >= selection_threshold])})")

        self.cache[regime_context_id] = {
            "selected_features": selected,
            "feature_scores": scores.to_dict(),
            "input_features": features_df.columns.tolist(),
            "market_regime": market_regime,
            "max_features": max_features,
            "timestamp": datetime.now().isoformat(),
            "selection_threshold": selection_threshold
        }
        self._save_storage()

        logger.info(f"Selected {len(selected)} features for {regime_context_id} with threshold {selection_threshold:.2f}")
        if isinstance(selected, list):
            return [str(f) for f in selected]
        return []

    def _clean_data(self, features_df: pd.DataFrame) -> pd.DataFrame:
        features_clean = features_df.replace([np.inf, -np.inf], np.nan).dropna(axis=1, how='all')
        features_clean = features_clean.loc[:, features_clean.nunique() > 1] # Drop constant columns
        # Simple median imputation
        return features_clean.fillna(features_clean.median())

    def _get_methods_for_regime(self, regime: str) -> dict[Any, float]:
        """ Returns a dictionary of {method: weight} for the given regime.

        Total 6 feature selection methods:
        1. Correlation (Spearman) - correlation with target
        2. Mutual Information - mutual information
        3. LightGBM importance - feature importance from model
        4. Random Forest importance - feature importance from RF
        5. Variance threshold - filter out low-variance features
        6. Chi-squared (for classification) - statistical test
        """
        base_methods = {
            self._correlation_filter: 1.0,
            self._mutual_info_filter: 1.0,
            self._variance_filter: 0.5,
            self._random_forest_filter: 1.0,
        }

        if regime == 'volatile':
            base_methods[self._lgbm_filter] = 1.5  # Emphasize model-based features
            base_methods[self._random_forest_filter] = 1.5
        elif regime == 'trending':
            base_methods[self._correlation_filter] = 1.5  # Emphasize correlation
        else:  # Normal regime
            base_methods[self._lgbm_filter] = 1.0

        return base_methods

    # --- Filter Methods ---
    def _correlation_filter(self, features_df, target_series, is_classification) -> pd.Series:
        """Spearman correlation with target."""
        return features_df.apply(lambda x: x.corr(target_series, method=self.correlation_method)).abs().sort_values(ascending=False)

    def _mutual_info_filter(self, features_df, target_series, is_classification) -> pd.Series:
        """Mutual Information - measures dependence between features and target."""
        mi_func = mutual_info_classif if is_classification else mutual_info_regression
        mi = mi_func(features_df, target_series, random_state=self.mi_random_state)
        return pd.Series(mi, index=features_df.columns).sort_values(ascending=False)

    def _lgbm_filter(self, features_df, target_series, is_classification) -> pd.Series | None:
        """LightGBM feature importance - feature importance from gradient boosting."""
        try:
            import lightgbm as lgb
        except ImportError:
            logger.warning("Library 'lightgbm' not found. Skipping LGBM filter.")
            return None

        try:
            params = self.lgbm_params.copy()
            params["objective"] = "binary" if is_classification else "regression"

            model = lgb.train(params, lgb.Dataset(features_df, label=target_series),
                            num_boost_round=params.pop('num_boost_round', 50))
            return pd.Series(model.feature_importance(importance_type='gain'),
                          index=features_df.columns).sort_values(ascending=False)
        except Exception as e:
            logger.error(f"LGBM filter failed: {e}")
            return None

    def _random_forest_filter(self, features_df, target_series, is_classification) -> pd.Series | None:
        """Random Forest feature importance - feature importance from random forest."""
        try:
            from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
        except ImportError:
            logger.warning("sklearn not found. Skipping Random Forest filter.")
            return None

        try:
            params = self.rf_params.copy()
            # Ensure all required hyperparameters are explicitly set
            params['random_state'] = params.get('random_state', 42)
            params['min_samples_leaf'] = params.get('min_samples_leaf', 1)
            max_features_value = params.get('max_features', 'sqrt')
            if isinstance(max_features_value, int):
                params['max_features'] = max_features_value
            else:
                params['max_features'] = 'sqrt'

            # Explicitly create models with all required parameters
            if is_classification:
                model = RandomForestClassifier(
                    n_estimators=params.get('n_estimators', 50),
                    max_depth=params.get('max_depth', 10),
                    random_state=params['random_state'],
                    min_samples_leaf=params['min_samples_leaf'],
                    max_features=params['max_features'],
                    n_jobs=params.get('n_jobs', -1)
                )
            else:
                model = RandomForestRegressor(
                    n_estimators=params.get('n_estimators', 50),
                    max_depth=params.get('max_depth', 10),
                    random_state=params['random_state'],
                    min_samples_leaf=params['min_samples_leaf'],
                    max_features=params['max_features'],
                    n_jobs=params.get('n_jobs', -1)
                )

            model.fit(features_df, target_series)
            return pd.Series(model.feature_importances_, index=features_df.columns).sort_values(ascending=False)
        except Exception as e:
            logger.error(f"Random Forest filter failed: {e}")
            return None

    def _variance_filter(self, features_df, target_series, is_classification) -> pd.Series | None:
        """Variance threshold - filters out low-variance features."""
        try:
            variances = features_df.var()

            # Filter out non-numeric variances and convert to float
            numeric_variances = variances[variances.apply(lambda x: isinstance(x, (int, float)))]
            if numeric_variances.empty:
                logger.warning("No numeric variances found, returning all features")
                return pd.Series([1.0] * len(features_df.columns), index=features_df.columns)

            # Normalize variances for comparison
            var_min = numeric_variances.min()
            var_max = numeric_variances.max()
            epsilon = float(self.variance_epsilon)
            normalized_var = (numeric_variances - var_min) / (var_max - var_min + epsilon)

            # Return full series with NaN for non-numeric features
            result = pd.Series(index=features_df.columns, dtype=float)
            result.update(normalized_var)
            result.fillna(1.0, inplace=True)  # Give non-numeric features max score

            return result.sort_values(ascending=False)
        except Exception as e:
            logger.error(f"Variance filter failed: {e}")
            return None
