"""
Integration module for Context-Aware Feature Selection in Colab.

This module provides a drop-in replacement for ColabFeatureSelector
that includes context awareness and feature importance analysis.
"""
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.core.logging.logger import ProjectLogger

logger = ProjectLogger.get_logger('ColabContextIntegration')


class ContextAwareColabFeatureSelector:
    """
    Enhanced feature selector for Colab with context awareness.

    Automatically detects and analyzes context features (state_*)
    during feature selection.
    """

    def __init__(self, project_path: str):
        """Initialize selector with project path."""
        self.project_path = Path(project_path)
        self.feature_selector = None
        self.uses_context_aware = False
        self._init_selector()

    def _init_selector(self):
        """Initialize Context-Aware FeatureSelector."""
        try:
            import sys
            src_path = self.project_path / 'src'
            if str(src_path) not in sys.path:
                sys.path.insert(0, str(src_path))
            try:
                from src.features.context_aware_feature_selector import ContextAwareFeatureSelector
                self.feature_selector = ContextAwareFeatureSelector(method=
                    'mutual_info', top_k=50)
                print('✅ Context-Aware FeatureSelector initialized')
                self.uses_context_aware = True
                return
            except ImportError as e:
                print(f'⚠️ Context-Aware selector not available: {e}')
            try:
                from src.features.selection.smart_selector import SmartFeatureSelector
                cache_path = (self.project_path / 'data' / 'cache' /
                    'selected_features.json')
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                self.feature_selector = SmartFeatureSelector(storage_path=
                    str(cache_path))
                print('✅ SmartFeatureSelector initialized (fallback)')
                self.uses_context_aware = False
                return
            except ImportError as e:
                print(f'⚠️ SmartFeatureSelector not available: {e}')
            from colab_clean_cell import SimpleFeatureSelector
            self.feature_selector = SimpleFeatureSelector()
            print('✅ SimpleFeatureSelector initialized (basic fallback)')
            self.uses_context_aware = False
        except Exception as e:
            logger.error(f'Виникла помилка: {e}', exc_info=True)
            print(f'❌ Error initializing feature selector: {e}')
            self.feature_selector = None
            self.uses_context_aware = False
            raise

    def select_features(self, features_df: pd.DataFrame, targets_df: pd.
        DataFrame, ticker: str, target_col: str, model_type: str='mlp'
        ) ->tuple[np.ndarray, list[str], dict[str, Any]]:
        """
        Select features with context awareness.

        Args:
            features_df: Features DataFrame
            targets_df: Targets DataFrame
            ticker: Ticker symbol
            target_col: Target column name
            model_type: Model type for max features

        Returns:
            Tuple of (selected_features_array, feature_names, analysis_metadata)
        """
        ticker_features = self._filter_for_ticker(features_df, ticker)
        ticker_targets = self._filter_for_ticker(targets_df, ticker)
        if len(ticker_features) < 10:
            raise ValueError(
                f'Insufficient data for {ticker}: {len(ticker_features)} rows')
        if target_col not in ticker_targets.columns:
            raise ValueError(f"Target '{target_col}' not found in targets")
        y = ticker_targets[target_col]
        numeric_features = ticker_features.select_dtypes(include=[np.number])
        feature_names = list(numeric_features.columns)
        if self.uses_context_aware and hasattr(self.feature_selector,
            'select_features'):
            selected_names, analysis = self.feature_selector.select_features(
                numeric_features, y, feature_names)
        else:
            max_features = self._get_model_max_features(model_type)
            selected_names = feature_names[:min(max_features, len(
                feature_names))]
            analysis = {'base_count': len(selected_names), 'context_count':
                0, 'temporal_count': 0, 'uses_context': False}
        selected_indices = [feature_names.index(name) for name in
            selected_names]
        selected_array = np.array(numeric_features.iloc[:, selected_indices])
        print(f'✅ Selected {len(selected_names)} features for {ticker}:')
        print(
            f"   Base: {analysis.get('base_count', 0)}, Context: {analysis.get('context_count', 0)}, Temporal: {analysis.get('temporal_count', 0)}"
            )
        if analysis.get('top_context_features'):
            print('   Top context features:')
            for feat in analysis['top_context_features'][:5]:
                print(f"      - {feat['name']}: {feat['importance']:.4f}")
        return selected_array, selected_names, analysis

    def _filter_for_ticker(self, df: pd.DataFrame, ticker: str) ->pd.DataFrame:
        """Filter DataFrame for specific ticker."""
        if 'ticker' in df.columns:
            return df[df['ticker'] == ticker].copy()
        elif hasattr(df.index, 'levels') and 'ticker' in df.index.names:
            try:
                return df.xs(ticker, level='ticker')
            except KeyError as e:
                logger.debug(f"Ticker {ticker} not found in index, returning full dataframe: {e}")
        return df

    def _get_model_max_features(self, model_type: str) ->int:
        """Get maximum features for model type."""
        max_features_map = {'mlp': 256, 'lstm': 128, 'gru': 128, 'cnn': 64,
            'transformer': 128, 'tabnet': 256, 'autoencoder': 128,
            'random_forest': 256}
        return max_features_map.get(model_type.lower(), 128)


def save_feature_analysis(analysis: dict[str, Any], ticker: str, target:
    str, model_type: str, output_dir: Path):
    """
    Save feature analysis to JSON file.

    Args:
        analysis: Analysis metadata from feature selection
        ticker: Ticker symbol
        target: Target column name
        model_type: Model type
        output_dir: Output directory for analysis files
    """
    import json
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        filename = f'feature_analysis_{ticker}_{target}_{model_type}.json'
        output_path = output_dir / filename
        full_analysis = {'ticker': ticker, 'target': target, 'model_type':
            model_type, **analysis}
        with open(output_path, 'w') as f:
            json.dump(full_analysis, f, indent=2)
        print(f'✅ Saved feature analysis to {output_path}')
    except Exception as e:
        logger.error(f'Виникла помилка: {e}', exc_info=True)
        print(f'⚠️ Failed to save feature analysis: {e}')
        raise


def visualize_context_importance(analysis: dict[str, Any], output_path:
    Path=None):
    """
    Create visualization of context feature importance.

    Args:
        analysis: Analysis metadata with feature importances
        output_path: Optional path to save plot
    """
    try:
        import matplotlib.pyplot as plt
        top_context = analysis.get('top_context_features', [])
        if not top_context:
            print('⚠️ No context features to visualize')
            return
        names = [f['name'].replace('state_', '') for f in top_context]
        importances = [f['importance'] for f in top_context]
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(names, importances)
        ax.set_xlabel('Feature Importance')
        ax.set_title('Top Context Features Importance')
        ax.invert_yaxis()
        plt.tight_layout()
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f'✅ Saved visualization to {output_path}')
        else:
            plt.show()
        plt.close()
    except ImportError:
        print('⚠️ matplotlib not available, skipping visualization')
    except Exception as e:
        logger.error(f'Виникла помилка: {e}', exc_info=True)
        print(f'⚠️ Failed to create visualization: {e}')
        raise
