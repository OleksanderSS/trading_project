"""
Model Training Orchestrator: Orchestrates light model training for different contexts (tickers, targets, models).
Extracted from HybridOrchestrator to improve code organization and testability.
"""
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error, r2_score

from src.core.logging.logger import ProjectLogger


class ModelTrainingOrchestrator:
    """Orchestrates model training for different contexts and calculates metrics."""

    def __init__(self, config_manager: Any = None):
        self.logger = ProjectLogger.get_logger(__name__)
        self.config_manager = config_manager

    def train_models_for_contexts(self, selected_feature_contexts: dict[str, dict[str, Any]],
                                  features_df: pd.DataFrame, targets_df: pd.DataFrame,
                                  ticker_col: str | None, batch_dir: Path,
                                  light_trainer: Any) -> tuple[dict[str, Any], int]:
        """Train models for all contexts and return metadata and count."""
        models_metadata = {}
        models_trained = 0

        for _context_id, context_data in selected_feature_contexts.items():
            metadata_updates, count = self._process_training_context(
                context_data, features_df, targets_df, ticker_col, batch_dir, light_trainer
            )
            models_metadata.update(metadata_updates)
            models_trained += count

        self.logger.info(f"✅ Trained {models_trained} models across {len(selected_feature_contexts)} contexts")
        return models_metadata, models_trained

    def _process_training_context(self, context_data: dict[str, Any], features_df: pd.DataFrame,
                                 targets_df: pd.DataFrame, ticker_col: str | None,
                                 batch_dir: Path, light_trainer: Any) -> tuple[dict[str, Any], int]:
        """Processes a single training context (ticker/model combination)."""
        model_name = context_data.get('model_name')
        if not model_name:
            self.logger.warning("⚠️ Skipping context: missing model_name")
            return {}, 0

        c_features_df, c_targets_df, available_features, resolved_ticker, timeframe = self._prepare_training_data(
            context_data, features_df, targets_df, ticker_col
        )

        if c_features_df is None or c_targets_df is None or not available_features:
            return {}, 0

        context_info = self._analyze_context_features(available_features)
        context_metadata = {}
        count = 0

        for target_col in context_data['targets']:
            if target_col not in c_targets_df.columns:
                continue

            metadata = self._train_single_model(
                light_trainer, c_features_df, c_targets_df, available_features,
                target_col, model_name, resolved_ticker, timeframe, batch_dir
            )

            if metadata:
                metadata.update(context_info)
                context_metadata[f"{resolved_ticker}_{target_col}_{model_name}"] = metadata
                count += 1

        return context_metadata, count

    def _analyze_context_features(self, features: list[str]) -> dict[str, Any]:
        """Analyzes which features are state-based context features."""
        context_features = [f for f in features if f.startswith('state_')]
        return {
            'uses_context_states': len(context_features) > 0,
            'context_features_count': len(context_features),
            'context_features': context_features[:10]
        }

    def _train_single_model(self, light_trainer: Any, c_features_df: pd.DataFrame, c_targets_df: pd.DataFrame,
                           available_features: list[str], target_col: str, model_name: str,
                           resolved_ticker: str, timeframe: str, batch_dir: Path) -> dict[str, Any] | None:
        """
        Train a single model and return metadata.
        High-level orchestrator for single model training process.
        """
        try:
            # 1. Prepare data split
            split_data = self._prepare_chronological_split(c_features_df, c_targets_df, available_features, target_col)
            if split_data is None:
                return None

            X_train, X_test, y_train, y_test = split_data

            # 2. Train
            task_type = self._resolve_target_task_type(target_col)
            result = self._execute_training(
                light_trainer, X_train, y_train, model_name, resolved_ticker, timeframe, target_col, task_type
            )

            if not result or result.get('status') != 'success':
                return None

            # 3. Evaluate
            metrics = self._evaluate_model(light_trainer, result['model_key'], X_test, y_test, task_type)

            # 4. Save
            model_path = self._save_trained_model(light_trainer, result, batch_dir, model_name, resolved_ticker, target_col)

            # 5. Metadata
            return self._create_model_metadata(resolved_ticker, target_col, model_name, metrics, model_path, available_features)

        except Exception as e:
            self.logger.error(f"❌ Error training {model_name} for {target_col}: {e}", exc_info=True)
            raise RuntimeError(
                f"Training failed for model={model_name}, ticker={resolved_ticker}, target={target_col}"
            ) from e

    def _prepare_chronological_split(self, features_df: pd.DataFrame, targets_df: pd.DataFrame,
                                 feature_cols: list[str], target_col: str) -> tuple | None:
        """Prepares train/test split with validation mask and chronological sorting."""
        f_sorted, t_sorted = self._sort_training_frames(features_df, targets_df)

        X, y = f_sorted[feature_cols].copy(), t_sorted[target_col].copy()
        valid_mask = y.notna() & X.notna().all(axis=1)
        X, y = X[valid_mask], y[valid_mask]

        if len(y) < 5:
            self.logger.warning(f"⚠️ Insufficient data: {len(y)} samples")
            return None

        split_idx = self._calculate_split_index(len(X))
        return X.iloc[:split_idx], X.iloc[split_idx:], y.iloc[:split_idx], y.iloc[split_idx:]

    def _execute_training(self, light_trainer: Any, X_train: pd.DataFrame, y_train: pd.Series,
                         model_name: str, ticker: str, timeframe: str, target: str, task_type: str) -> dict[str, Any] | None:
        """Executes the training call on the light trainer."""
        train_df = X_train.copy()
        train_df[target] = y_train.values

        config = {
            'model_type': model_name,
            'ticker': ticker,
            'timeframe': timeframe,
            'target_col': target,
            'task_type': task_type
        }
        return cast(dict[str, Any] | None, light_trainer.train_light_model(train_df, config))

    def _evaluate_model(self, light_trainer: Any, model_key: str, X_test: pd.DataFrame,
                       y_test: pd.Series, task_type: str) -> dict[str, float]:
        """Performs prediction and calculates metrics."""
        preds = light_trainer.predict(model_key, X_test)
        return self._calculate_metrics(y_test, preds, task_type)

    def _calculate_split_index(self, total_len: int) -> int:
        """Calculate 80/20 train/test split index."""
        return min(max(1, int(total_len * 0.8)), total_len - 1)

    def _sort_training_frames(self, features_df: pd.DataFrame, targets_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Sort feature and target frames by time before chronological splitting."""
        sort_cols = [col for col in ["datetime", "timestamp", "date"] if col in features_df.columns and col in targets_df.columns]
        if not sort_cols:
            return features_df, targets_df

        order = features_df[sort_cols[0]].sort_values().index
        return features_df.loc[order].reset_index(drop=True), targets_df.loc[order].reset_index(drop=True)

    def _train_with_config(self, light_trainer: Any, train_df: pd.DataFrame, model_name: str,
                          resolved_ticker: str, timeframe: str, target_col: str, task_type: str) -> dict[str, Any] | None:
        """Train model with configuration."""
        config = {
            'model_type': model_name,
            'ticker': resolved_ticker,
            'timeframe': timeframe,
            'target_col': target_col,
            'task_type': task_type
        }
        return cast(dict[str, Any] | None, light_trainer.train_light_model(train_df, config))

    def _calculate_metrics(self, y_test: pd.Series, preds: np.ndarray, task_type: str) -> dict[str, float]:
        """Calculate metrics based on task type."""
        if task_type == 'regression':
            mse = mean_squared_error(y_test, preds)
            return {
                'mse': float(mse),
                'rmse': float(np.sqrt(mse)),
                'mae': float(mean_absolute_error(y_test, preds)),
                'r2': float(r2_score(y_test, preds)),
                'score': float(r2_score(y_test, preds))
            }
        else:
            acc = accuracy_score(y_test, preds)
            return {'accuracy': float(acc), 'score': float(acc)}

    def _save_trained_model(self, light_trainer: Any, result: dict[str, Any], batch_dir: Path,
                           model_name: str, resolved_ticker: str, target_col: str) -> Path:
        """Save trained model to disk."""
        models_dir = batch_dir / 'models'
        models_dir.mkdir(parents=True, exist_ok=True)
        model_path = models_dir / f"{model_name}_{resolved_ticker}_{target_col}.joblib"
        light_trainer.save_model_to_disk(result['model_key'], str(model_path))
        if self.logger.isEnabledFor(logging.DEBUG):
            self.logger.debug(f"💾 Model saved: {model_path}")
        return model_path

    def _create_model_metadata(self, resolved_ticker: str, target_col: str, model_name: str,
                              metrics: dict[str, float], model_path: Path, available_features: list[str]) -> dict[str, Any]:
        """Create model metadata dictionary."""
        return {
            'ticker': resolved_ticker,
            'target': target_col,
            'model_type': model_name,
            'model_category': 'light',
            'source': 'local',
            'timestamp': datetime.now().isoformat(),
            'metrics': metrics,
            'model_path': str(model_path),
            'selected_features': available_features,
            'trained': True
        }

    def _resolve_ticker(self, context_ticker: str, c_features_df: pd.DataFrame, ticker_col: str | None) -> str | None:
        """Resolve ticker symbol."""
        if context_ticker:
            return context_ticker
        if ticker_col and ticker_col in c_features_df.columns:
            return str(c_features_df[ticker_col].iloc[-1]).upper()
        return None

    def _resolve_target_task_type(self, target_name: str) -> str:
        """Maps configured targets to task type (regression/classification)."""
        if not self.config_manager:
            # Default to regression if no config manager
            return 'regression'

        targets_config = self.config_manager.get_config('targets', {})
        if hasattr(targets_config, 'as_dict'):
            targets_config = targets_config.as_dict()

        target_definitions = targets_config.get('targets', targets_config)
        target_meta = target_definitions.get(target_name, {}) if isinstance(target_definitions, dict) else {}
        configured_type = str(target_meta.get('type', '')).lower()

        return self._determine_task_type(configured_type, target_name)

    def _determine_task_type(self, configured_type: str, target_name: str) -> str:
        """Determine task type from configuration."""
        if configured_type in {'regression', 'indicator_prediction'}:
            return 'regression'
        if configured_type.startswith('classification'):
            return 'classification'

        fallback_name = str(target_name).lower()
        if 'return' in fallback_name or 'price' in fallback_name or '_f' in fallback_name:
            return 'regression'
        return 'classification'

    def _prepare_training_data(self, context_data: dict[str, Any], features_df: pd.DataFrame,
                               targets_df: pd.DataFrame, ticker_col: str | None) -> tuple:
        """Placeholder for data preparation logic."""
        ticker = context_data.get('ticker')
        timeframe = context_data.get('timeframe')
        available_features = context_data.get('features', [])

        if ticker_col and ticker:
            ticker_mask = features_df[ticker_col] == ticker
            c_features_df = features_df[ticker_mask]
            c_targets_df = targets_df[ticker_mask]
        else:
            c_features_df = features_df
            c_targets_df = targets_df

        return c_features_df, c_targets_df, available_features, ticker, timeframe
