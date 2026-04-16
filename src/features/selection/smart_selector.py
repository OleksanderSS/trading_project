import pandas as pd
import numpy as np
import json
import os
from datetime import datetime
from typing import List, Dict, Any, Optional
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression

from src.core.logging.logger import ProjectLogger

logger = ProjectLogger.get_logger(__name__)

class SmartFeatureSelector:
    """
    Selects features based on a voting ensemble of methods, now with regime-specific caching.
    """

    def __init__(self, storage_path: str = None, min_volatility: float = 0.0001):
        if storage_path is None:
            storage_path = os.path.join('data', 'cache', 'selected_features.json')
        self.storage_path = storage_path
        self.min_volatility = min_volatility
        os.makedirs(os.path.dirname(self.storage_path), exist_ok=True)
        self.cache = self._load_storage()

    def _load_storage(self) -> Dict[str, Any]:
        if os.path.exists(self.storage_path):
            try:
                with open(self.storage_path, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                logger.error(f"Failed to load feature cache from {self.storage_path}: {e}")
        return {}

    def _save_storage(self):
        try:
            os.makedirs(os.path.dirname(self.storage_path), exist_ok=True)
            with open(self.storage_path, 'w') as f:
                json.dump(self.cache, f, indent=4)
        except IOError as e:
            logger.error(f"Failed to save feature cache to {self.storage_path}: {e}")

    def select(self, X: pd.DataFrame, y: pd.Series, context_id: str,
               is_classification: bool = True, market_regime: str = "normal",
               force_recalculate: bool = False, max_features: int = None) -> List[str]:
        """
        Selects features, caching them based on both context, market regime, and max_features.
        
        Args:
            X: Feature dataframe
            y: Target series
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
            if set(cached_data.get("input_features", [])) == set(X.columns):
                logger.info(f"Using cached features for {regime_context_id}...")
                return cached_data["selected_features"]

        logger.info(f"Running dynamic feature selection for {regime_context_id}...")

        # Pre-filtering and cleaning
        if y.std() < self.min_volatility:
            logger.warning(f"Target volatility is below threshold for {regime_context_id}. Skipping.")
            return []
        
        X_clean = self._clean_data(X)
        if X_clean.empty:
            logger.warning(f"No valid features after cleaning for {regime_context_id}.")
            return []

        # Dynamic voting based on regime
        methods = self._get_methods_for_regime(market_regime)
        scores = pd.Series(0.0, index=X_clean.columns, dtype=float)  # ✅ FIX: Явно вказуємо float dtype

        for method_func, weight in methods.items():
            try:
                ranked_features = method_func(X_clean, y, is_classification)
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
            "input_features": X.columns.tolist(),
            "market_regime": market_regime,
            "max_features": max_features,
            "timestamp": datetime.now().isoformat(),
            "selection_threshold": selection_threshold
        }
        self._save_storage()

        logger.info(f"Selected {len(selected)} features for {regime_context_id} with threshold {selection_threshold:.2f}")
        return selected

    def _clean_data(self, X: pd.DataFrame) -> pd.DataFrame:
        X_clean = X.replace([np.inf, -np.inf], np.nan).dropna(axis=1, how='all')
        X_clean = X_clean.loc[:, X_clean.nunique() > 1] # Drop constant columns
        # Simple median imputation
        return X_clean.fillna(X_clean.median())

    def _get_methods_for_regime(self, regime: str) -> Dict[Any, float]:
        """ Returns a dictionary of {method: weight} for the given regime. 
        
        Всього 6 методів вибору фіч:
        1. Correlation (Spearman) - кореляція з таргетом
        2. Mutual Information - взаємна інформація
        3. LightGBM importance - важливість фіч з моделі
        4. Random Forest importance - важливість фіч з RF
        5. Variance threshold - відсів низько-варіативних фіч
        6. Chi-squared (для класифікації) - статистичний тест
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
    def _correlation_filter(self, X, y, is_classification) -> pd.Series:
        """Spearman correlation з таргетом."""
        return X.apply(lambda x: x.corr(y, method='spearman')).abs().sort_values(ascending=False)

    def _mutual_info_filter(self, X, y, is_classification) -> pd.Series:
        """Mutual Information - вимірює залежність між фічами та таргетом."""
        mi_func = mutual_info_classif if is_classification else mutual_info_regression
        mi = mi_func(X, y, random_state=42)
        return pd.Series(mi, index=X.columns).sort_values(ascending=False)

    def _lgbm_filter(self, X, y, is_classification) -> Optional[pd.Series]:
        """LightGBM feature importance - важливість фіч з градієнтного бустингу."""
        try:
            import lightgbm as lgb
        except ImportError:
            logger.warning("Library 'lightgbm' not found. Skipping LGBM filter.")
            return None
            
        try:
            params = {
                "objective": "binary" if is_classification else "regression", 
                "verbosity": -1,
                "force_col_wise": True,
                "num_leaves": 31,
                "learning_rate": 0.05
            }
            model = lgb.train(params, lgb.Dataset(X, label=y), num_boost_round=50)
            return pd.Series(model.feature_importance(importance_type='gain'), index=X.columns).sort_values(ascending=False)
        except Exception as e:
            logger.error(f"LGBM filter failed: {e}")
            return None
    
    def _random_forest_filter(self, X, y, is_classification) -> Optional[pd.Series]:
        """Random Forest feature importance - важливість фіч з випадкового лісу."""
        try:
            from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
        except ImportError:
            logger.warning("sklearn not found. Skipping Random Forest filter.")
            return None
        
        try:
            if is_classification:
                model = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1)
            else:
                model = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1)
            
            model.fit(X, y)
            return pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
        except Exception as e:
            logger.error(f"Random Forest filter failed: {e}")
            return None
    
    def _variance_filter(self, X, y, is_classification) -> Optional[pd.Series]:
        """Variance threshold - відсіває низько-варіативні фічі."""
        try:
            variances = X.var()
            # Нормалізуємо варіанси для порівняння
            normalized_var = (variances - variances.min()) / (variances.max() - variances.min() + 1e-10)
            return normalized_var.sort_values(ascending=False)
        except Exception as e:
            logger.error(f"Variance filter failed: {e}")
            return None