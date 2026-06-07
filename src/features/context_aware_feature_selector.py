"""
Context-Aware Feature Selector with importance analysis.

Integrates context map (state_* features) into feature selection
and provides analysis of context feature importance.
"""
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.core.logging.logger import ProjectLogger

logger = ProjectLogger.get_logger(__name__)


class ContextAwareFeatureSelector:
    """
    Feature selector that understands context features (state_*).

    Features are categorized into:
    - Base features: OHLCV, indicators, macro, news (253 features)
    - Context features: state_* columns (147 features)
    - Temporal features: hour, day_of_week, etc.
    """

    def __init__(self, method: str = 'mutual_info', top_k: int = 50):
        """
        Initialize selector.

        Args:
            method: Selection method ('mutual_info', 'f_regression', 'random_forest')
            top_k: Number of features to select
        """
        self.method = method
        self.top_k = top_k
        self.selector = None
        self._init_selector()

    def _init_selector(self):
        """Initialize feature selector based on method."""
        try:
            from sklearn.ensemble import RandomForestRegressor
            from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression

            if self.method == 'mutual_info':
                self.selector = SelectKBest(mutual_info_regression, k=self.top_k)
            elif self.method == 'f_regression':
                self.selector = SelectKBest(f_regression, k=self.top_k)
            elif self.method == 'random_forest':
                self.selector = RandomForestRegressor(
                    n_estimators=50,
                    max_depth=10,
                    random_state=42,
                    n_jobs=-1
                )
            else:
                logger.warning(f"Unknown method '{self.method}', using mutual_info")
                self.selector = SelectKBest(mutual_info_regression, k=self.top_k)

            logger.info(f"✅ Initialized {self.method} feature selector (top_k={self.top_k})")
        except ImportError as e:
            logger.exception(f"Failed to import sklearn: {e}")
            self.selector = None

    def select_features(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        feature_names: list[str] | None = None
    ) -> tuple[list[str], dict[str, Any]]:
        """
        Select top features with context awareness.

        Args:
            X: Features DataFrame
            y: Target Series
            feature_names: Optional list of feature names (uses X.columns if None)

        Returns:
            Tuple of (selected_feature_names, analysis_metadata)
        """
        if feature_names is None:
            feature_names = list(X.columns)

        # Categorize features
        feature_categories = self._categorize_features(feature_names)

        # Handle NaN values
        X_clean, y_clean = self._clean_data(X, y)

        if len(X_clean) < 10:
            logger.warning(f"Insufficient data after cleaning: {len(X_clean)} rows")
            return feature_names[:self.top_k], {}

        # Perform selection
        if self.method == 'random_forest':
            selected_features, importances = self._select_with_random_forest(
                X_clean, y_clean, feature_names
            )
        else:
            selected_features, importances = self._select_with_sklearn(
                X_clean, y_clean, feature_names
            )

        # Analyze context feature importance
        analysis = self._analyze_context_importance(
            selected_features, importances, feature_categories
        )

        logger.info(f"✅ Selected {len(selected_features)} features:")
        logger.info(f"   Base: {analysis['base_count']}, Context: {analysis['context_count']}, Temporal: {analysis['temporal_count']}")

        return selected_features, analysis

    def _categorize_features(self, feature_names: list[str]) -> dict[str, list[str]]:
        """Categorize features into base, context, and temporal."""
        categories = {
            'base': [],
            'context': [],
            'temporal': []
        }

        temporal_keywords = ['hour', 'day', 'week', 'month', 'quarter', 'year', 'weekend']

        for name in feature_names:
            if name.startswith('state_'):
                categories['context'].append(name)
            elif any(kw in name.lower() for kw in temporal_keywords):
                categories['temporal'].append(name)
            else:
                categories['base'].append(name)

        return categories

    def _clean_data(self, X: pd.DataFrame, y: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
        """Clean data by removing NaN and inf values."""
        # Remove rows with NaN in target
        valid_mask = ~y.isna()
        X_clean = X[valid_mask].copy()
        y_clean = y[valid_mask].copy()

        # Fill NaN in features with median
        X_clean = X_clean.fillna(X_clean.median())

        # Replace inf with large values
        X_clean = X_clean.replace([np.inf, -np.inf], [1e10, -1e10])

        return X_clean, y_clean

    def _select_with_sklearn(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        feature_names: list[str]
    ) -> tuple[list[str], dict[str, float]]:
        """Select features using sklearn SelectKBest."""
        try:
            self.selector.fit(X, y)

            # Get scores
            scores = self.selector.scores_

            # Get selected indices
            selected_indices = self.selector.get_support(indices=True)
            selected_features = [feature_names[i] for i in selected_indices]

            # Create importance dict
            importances = {
                feature_names[i]: float(scores[i])
                for i in range(len(feature_names))
            }

            return selected_features, importances

        except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError) as e:
            logger.exception(f"Feature selection failed: {e}")
            # Fallback: return first top_k features
            return feature_names[:self.top_k], {}

    def _select_with_random_forest(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        feature_names: list[str]
    ) -> tuple[list[str], dict[str, float]]:
        """Select features using Random Forest feature importance."""
        try:
            self.selector.fit(X, y)

            # Get feature importances
            importances_array = self.selector.feature_importances_

            # Create importance dict
            importances = {
                name: float(imp)
                for name, imp in zip(feature_names, importances_array, strict=False)
            }

            # Sort by importance and select top_k
            sorted_features = sorted(
                importances.items(),
                key=lambda x: x[1],
                reverse=True
            )
            selected_features = [name for name, _ in sorted_features[:self.top_k]]

            return selected_features, importances

        except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError) as e:
            logger.exception(f"Random Forest selection failed: {e}")
            return feature_names[:self.top_k], {}

    def _analyze_context_importance(
        self,
        selected_features: list[str],
        importances: dict[str, float],
        feature_categories: dict[str, list[str]]
    ) -> dict[str, Any]:
        """Analyze importance of context features."""
        # Count selected features by category
        selected_base = [f for f in selected_features if f in feature_categories['base']]
        selected_context = [f for f in selected_features if f in feature_categories['context']]
        selected_temporal = [f for f in selected_features if f in feature_categories['temporal']]

        # Calculate average importance by category
        def avg_importance(features):
            if not features or not importances:
                return 0.0
            return np.mean([importances.get(f, 0.0) for f in features])

        # Top context features
        context_importances = {
            f: importances.get(f, 0.0)
            for f in selected_context
        }
        top_context = sorted(
            context_importances.items(),
            key=lambda x: x[1],
            reverse=True
        )[:10]

        analysis = {
            'base_count': len(selected_base),
            'context_count': len(selected_context),
            'temporal_count': len(selected_temporal),
            'base_avg_importance': avg_importance(selected_base),
            'context_avg_importance': avg_importance(selected_context),
            'temporal_avg_importance': avg_importance(selected_temporal),
            'top_context_features': [
                {'name': name, 'importance': float(imp)}
                for name, imp in top_context
            ],
            'uses_context': len(selected_context) > 0,
            'context_ratio': len(selected_context) / len(selected_features) if selected_features else 0.0
        }

        return analysis

    def save_analysis(self, analysis: dict[str, Any], output_path: Path):
        """Save feature analysis to JSON file."""
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                json.dump(analysis, f, indent=2)
            logger.info(f"✅ Saved feature analysis to {output_path}")
        except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError) as e:
            logger.exception(f"Failed to save analysis: {e}")
