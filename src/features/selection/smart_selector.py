import json
import logging
import os
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression

from src.core.logging.logger import ProjectLogger
from src.pipeline.target_column_utils import is_target_like_column

logger = ProjectLogger.get_logger(__name__)

class SmartFeatureSelector:
    """
    Selects features based on a voting ensemble of methods, now with regime-specific caching.
    """

    def __init__(self, storage_path: str | None = None, min_volatility: float = 0.0001):
        self.logger = logger  # Initialize logger from module level
        if storage_path is None:
            storage_path = os.path.join('data', 'cache', 'selected_features.json')
        self.storage_path = storage_path
        self.min_volatility = min_volatility

        # Feature selection hyperparameters
        self.correlation_method = 'spearman'
        self.mi_random_state = 42
        self.lgbm_params = {
            'objective': 'binary',  # Will be overridden based on task type
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
                logger.exception(f"Failed to load feature cache from {self.storage_path}: {e}")
        return {}

    def _save_storage(self):
        try:
            os.makedirs(os.path.dirname(self.storage_path), exist_ok=True)
            with open(self.storage_path, 'w') as f:
                json.dump(self.cache, f, indent=4, default=str)
        except OSError as e:
            logger.exception(f"Failed to save feature cache to {self.storage_path}: {e}")

    def _check_cache(self, regime_context_id: str, features_df: pd.DataFrame, force_recalculate: bool) -> list[str] | None:
        """Check cache and return cached features if valid."""
        if not force_recalculate and regime_context_id in self.cache:
            cached_data = self.cache[regime_context_id]
            if set(cached_data.get("input_features", [])) == set(features_df.columns):
                logger.info(f"Using cached features for {regime_context_id}...")
                selected_features = cached_data.get("selected_features", [])
                if isinstance(selected_features, list):
                    return [str(f) for f in selected_features]
        return None

    def _pre_filter_data(self, features_df: pd.DataFrame, target_series: pd.Series, regime_context_id: str) -> pd.DataFrame | None:
        """Pre-filter and clean data.

        Strips all target-like columns before any voting method sees the data.
        This is the single, authoritative leakage-prevention gate for the
        selector — individual filter methods must not duplicate this check.
        """
        if target_series.std() < self.min_volatility:
            logger.warning(f"Target volatility is below threshold for {regime_context_id}. Skipping.")
            return None

        # Remove target-like columns using the canonical utility (covers
        # target_* prefixed cols AND derived/state columns that carry forward
        # information about the target window).
        target_cols_present = [c for c in features_df.columns if is_target_like_column(c)]
        
        # Remove metadata columns to prevent them from being selected as features
        metadata_cols = ['ticker', 'datetime', 'date', 'interval', 'timeframe', 'hash', 'symbol']
        meta_cols_present = [c for c in metadata_cols if c in features_df.columns]
        
        cols_to_drop = target_cols_present + meta_cols_present
        if cols_to_drop:
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    f"[{regime_context_id}] Dropping {len(cols_to_drop)} "
                    f"target-like and metadata column(s) before feature selection: "
                    f"{cols_to_drop[:5]}{'...' if len(cols_to_drop) > 5 else ''}"
                )
            features_df = features_df.drop(columns=cols_to_drop, errors='ignore')

        features_clean = self._clean_data(features_df)
        if features_clean.empty:
            logger.warning(f"No valid features after cleaning for {regime_context_id}.")
            return None

        return features_clean

    def _run_feature_selection_methods(self, features_clean: pd.DataFrame, target_series: pd.Series,
                                       is_classification: bool, market_regime: str, regime_context_id: str) -> pd.Series:
        """Run feature selection methods and return scores."""
        methods = self._get_methods_for_regime(market_regime)
        scores = pd.Series(0.0, index=features_clean.columns, dtype=float)

        for method_func, weight in methods.items():
            try:
                ranked_features = method_func(features_clean, target_series, is_classification)
                if ranked_features is None:
                    if logger.isEnabledFor(logging.DEBUG):
                        logger.debug(f"Method {method_func.__name__} returned None, skipping")
                    continue
                top_n = max(1, int(len(ranked_features) * 0.3))
                # Borda count, not a flat vote.
                #
                # This used to be `scores.loc[head(top_n).index] += weight`,
                # which gives the method's BEST feature and its thirtieth-best
                # the identical score. Each method produces a RANKING and the
                # vote threw the ranking away, keeping only membership of the
                # top 30% — so a feature all three methods rank first and a
                # feature all three rank thirtieth came out tied, and the
                # threshold that picks the final set could not separate them.
                #
                # Borda keeps the order: rank 1 of n scores n, rank n scores 1,
                # scaled to [0, 1] so the method weights still mean what they
                # say and no method's vote depends on how many features it was
                # handed.
                chosen = ranked_features.head(top_n)
                borda = np.linspace(1.0, 1.0 / len(chosen), len(chosen))
                scores.loc[chosen.index] += weight * borda
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(
                        f"Method {method_func.__name__} spread weight {weight} "
                        f"over {top_n} features by rank"
                    )
            except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError) as e:
                logger.exception(f"Feature selection method {method_func.__name__} failed for {regime_context_id}: {e}")

        return scores

    def _select_features_by_threshold(self, scores: pd.Series) -> tuple:
        """Select features with score above threshold."""
        if scores.sum() == 0:
            return [], 0.0

        positive_scores = scores[scores > 0]
        selection_threshold = positive_scores.median()
        selected = positive_scores[positive_scores >= selection_threshold].index.tolist()
        return selected, selection_threshold

    def _apply_max_features_limit(self, selected: list, scores: pd.Series, selection_threshold: float,
                                   max_features: int | None) -> list:
        """Apply max_features limit if specified."""
        if max_features is None or len(selected) <= max_features:
            return selected

        positive_scores = scores[scores > 0]
        selected_with_scores = positive_scores[positive_scores >= selection_threshold].sort_values(ascending=False)
        selected = selected_with_scores.head(max_features).index.tolist()
        logger.info(f"Limited selected features to {max_features} (from {len(positive_scores[positive_scores >= selection_threshold])})")
        return selected

    def _cache_results(self, regime_context_id: str, selected: list, scores: pd.Series,
                       features_df: pd.DataFrame, market_regime: str, max_features: int | None,
                       selection_threshold: float) -> None:
        """Cache selection results."""
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

        # Check cache
        cached_features = self._check_cache(regime_context_id, features_df, force_recalculate)
        if cached_features is not None:
            return cached_features

        logger.info(f"Running dynamic feature selection for {regime_context_id}...")

        # Pre-filtering and cleaning
        features_clean = self._pre_filter_data(features_df, target_series, regime_context_id)
        if features_clean is None:
            return []

        # Dynamic voting based on regime
        scores = self._run_feature_selection_methods(features_clean, target_series, is_classification, market_regime, regime_context_id)

        # Select features with score above threshold
        selected, selection_threshold = self._select_features_by_threshold(scores)

        if not selected:
            logger.warning(f"No features were selected for {regime_context_id}. Returning empty list.")
            return []

        # Apply max_features limit if specified
        selected = self._apply_max_features_limit(selected, scores, selection_threshold, max_features)

        # Cache results
        self._cache_results(regime_context_id, selected, scores, features_df, market_regime, max_features, selection_threshold)

        logger.info(f"Selected {len(selected)} features for {regime_context_id} with threshold {selection_threshold:.2f}")
        return [str(f) for f in selected]

    def _clean_data(self, features_df: pd.DataFrame) -> pd.DataFrame:
        features_clean = features_df.replace([np.inf, -np.inf], np.nan).dropna(axis=1, how='all')
        features_clean = features_clean.loc[:, features_clean.nunique() > 1] # Drop constant columns
        # Simple median imputation
        return features_clean.fillna(features_clean.median())

    def _get_methods_for_regime(self, regime: str) -> dict[Any, float]:
        """ Returns a dictionary of {method: weight} for the given regime. """
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
        else:
            # Normal regime
            base_methods[self._lgbm_filter] = 1.0

        return base_methods

    # --- Filter Methods ---
    def _correlation_filter(self, features_df, target_series, is_classification) -> pd.Series:
        """Spearman correlation with target.

        `corrwith` rather than `apply(lambda x: x.corr(...))`, but NOT for the
        reason the audit that prompted this gave. It claimed the per-column
        form is "100-200x slower". Measured on this project's real shape —
        3,000 rows by 2,203 columns — the two are the same: corrwith 4.1s
        against ~4s extrapolated, because pandas loops internally for Spearman
        too. The speed claim does not survive measurement and is not why this
        changed.

        What it does buy is correctness at the edges. A zero-variance column
        yields NaN, and the old form sorted that NaN into the ranking, where it
        occupied a slot in the top 30% while meaning nothing — this batch
        carries 47 constant columns. `.dropna()` here removes them from
        contention instead.
        """
        return (
            features_df.corrwith(target_series, method=self.correlation_method)
            .abs()
            .dropna()
            .sort_values(ascending=False)
        )

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
        except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError) as e:
            logger.exception(f"LGBM filter failed: {e}")
            raise RuntimeError("LGBM feature importance filter failed") from e

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

            # Leakage protection: drop target columns (centrally handled in
            # _pre_filter_data, but kept here as a defensive second layer).
            features_clean = features_df.drop(
                columns=[c for c in features_df.columns if is_target_like_column(c)],
                errors='ignore'
            )
            model.fit(features_clean, target_series)
            return pd.Series(model.feature_importances_, index=features_clean.columns).sort_values(ascending=False)
        except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError) as e:
            logger.exception(f"Random Forest filter failed: {e}")
            raise RuntimeError("Random Forest feature importance filter failed") from e

    def _variance_filter(self, features_df, target_series, is_classification) -> pd.Series | None:
        """Variance threshold - filters out low-variance features."""
        try:
            variances = features_df.var()
            # Normalize variances for comparison
            normalized_var = (variances - variances.min()) / (variances.max() - variances.min() + self.variance_epsilon)
            return normalized_var.sort_values(ascending=False)
        except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError) as e:
            logger.exception(f"Variance filter failed: {e}")
            raise RuntimeError("Variance feature filter failed") from e
