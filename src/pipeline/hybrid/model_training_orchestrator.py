"""
Model Training Orchestrator: Orchestrates light model training for different contexts (tickers, targets, models).
Extracted from HybridOrchestrator to improve code organization and testability.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, cast
from datetime import datetime
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score

from src.core.logging.logger import ProjectLogger


class ModelTrainingOrchestrator:
    """Orchestrates model training for different contexts and calculates metrics."""
    
    def __init__(self, config_manager: Any = None):
        self.logger = ProjectLogger.get_logger(__name__)
        self.config_manager = config_manager
    
    def train_models_for_contexts(self, selected_feature_contexts: Dict[str, Dict[str, Any]],
                                  features_df: pd.DataFrame, targets_df: pd.DataFrame,
                                  ticker_col: Optional[str], batch_dir: Path,
                                  light_trainer: Any) -> Tuple[Dict[str, Any], int]:
        """Train models for all contexts and return metadata and count."""
        models_metadata = {}
        models_trained = 0
        
        for context_id, context_data in selected_feature_contexts.items():
            model_name = context_data.get('model_name')
            if not isinstance(model_name, str) or not model_name:
                self.logger.warning(f"⚠️ Skipping context {context_id}: missing model_name")
                continue
            
            # Prepare training data
            c_features_df, c_targets_df, available_features, resolved_ticker, timeframe = self._prepare_training_data(
                context_data, features_df, targets_df, ticker_col
            )
            
            if c_features_df is None or c_targets_df is None or not available_features or not resolved_ticker or not timeframe:
                continue
            
            # Аналіз використання context features
            context_features = [f for f in available_features if f.startswith('state_')]
            
            # Train for each target
            for target_col in context_data['targets']:
                if target_col not in c_targets_df.columns:
                    continue
                
                metadata = self._train_single_model(
                    light_trainer, c_features_df, c_targets_df, available_features,
                    target_col, model_name, resolved_ticker, timeframe, batch_dir
                )
                
                if metadata:
                    # Додаємо інформацію про context features
                    metadata['uses_context_states'] = len(context_features) > 0
                    metadata['context_features_count'] = len(context_features)
                    metadata['context_features'] = context_features[:10]  # Топ-10 для логу
                    
                    models_metadata[f"{resolved_ticker}_{target_col}_{model_name}"] = metadata
                    models_trained += 1
        
        self.logger.info(f"✅ Trained {models_trained} models across {len(selected_feature_contexts)} contexts")
        return models_metadata, models_trained
    
    def _prepare_training_data(self, context_data: Dict[str, Any], features_df: pd.DataFrame,
                              targets_df: pd.DataFrame, ticker_col: Optional[str]) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[List[str]], Optional[str], Optional[str]]:
        """Prepare training data for a specific context."""
        context_ticker = context_data['ticker']
        
        c_features_df, c_targets_df = features_df.copy(), targets_df.copy()
        if context_ticker and ticker_col:
            mask = features_df[ticker_col].str.upper() == context_ticker.upper()
            c_features_df, c_targets_df = features_df[mask].copy(), targets_df[mask].copy()
        
        if c_features_df.empty:
            self.logger.warning(f"⚠️ No data for ticker {context_ticker}")
            return None, None, None, None, None
        
        available_features = [f for f in context_data['selected_features'] if f in c_features_df.columns]
        if not available_features:
            self.logger.warning(f"⚠️ No selected features found in data for {context_ticker}")
            return None, None, None, None, None
        
        resolved_ticker = self._resolve_ticker(context_ticker, c_features_df, ticker_col)
        if not resolved_ticker:
            self.logger.warning(f"⚠️ Could not resolve ticker from {context_ticker}")
            return None, None, None, None, None
        
        timeframe = str(c_features_df['timeframe'].iloc[-1]) if 'timeframe' in c_features_df.columns else '1d'
        
        return c_features_df, c_targets_df, available_features, resolved_ticker, timeframe
    
    def _train_single_model(self, light_trainer: Any, c_features_df: pd.DataFrame, c_targets_df: pd.DataFrame,
                           available_features: List[str], target_col: str, model_name: str,
                           resolved_ticker: str, timeframe: str, batch_dir: Path) -> Optional[Dict[str, Any]]:
        """Train a single model and return metadata."""
        try:
            c_features_df, c_targets_df = self._sort_training_frames(c_features_df, c_targets_df)
            X, y = c_features_df[available_features].copy(), c_targets_df[target_col].copy()
            valid_mask = y.notna() & X.notna().all(axis=1)
            X, y = X[valid_mask], y[valid_mask]
            
            if len(y) < 5:
                self.logger.warning(f"⚠️ Insufficient data for {model_name}: {len(y)} samples")
                return None
            
            split_idx = self._calculate_split_index(len(X))
            X_train, X_test, y_train, y_test = X.iloc[:split_idx], X.iloc[split_idx:], y.iloc[:split_idx], y.iloc[split_idx:]
            
            train_df = X_train.copy()
            train_df[target_col] = y_train.values
            task_type = self._resolve_target_task_type(target_col)
            
            result = self._train_with_config(light_trainer, train_df, model_name, resolved_ticker, timeframe, target_col, task_type)
            if not result or result.get('status') != 'success':
                self.logger.warning(f"⚠️ Training failed for {model_name}")
                return None
            
            preds = light_trainer.predict(result['model_key'], X_test)
            metrics = self._calculate_metrics(y_test, preds, task_type)
            
            model_path = self._save_trained_model(light_trainer, result, batch_dir, model_name, resolved_ticker, target_col)
            
            return self._create_model_metadata(resolved_ticker, target_col, model_name, metrics, model_path, available_features)
        except Exception as e:
            self.logger.error(f"❌ Error training {model_name} for {target_col}: {e}", exc_info=True)
            return None
    
    def _calculate_split_index(self, total_len: int) -> int:
        """Calculate 80/20 train/test split index."""
        return min(max(1, int(total_len * 0.8)), total_len - 1)

    def _sort_training_frames(self, features_df: pd.DataFrame, targets_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Sort feature and target frames by time before chronological splitting."""
        sort_cols = [col for col in ["datetime", "timestamp", "date"] if col in features_df.columns and col in targets_df.columns]
        if not sort_cols:
            return features_df, targets_df

        order = features_df[sort_cols[0]].sort_values().index
        return features_df.loc[order].reset_index(drop=True), targets_df.loc[order].reset_index(drop=True)
    
    def _train_with_config(self, light_trainer: Any, train_df: pd.DataFrame, model_name: str,
                          resolved_ticker: str, timeframe: str, target_col: str, task_type: str) -> Optional[Dict[str, Any]]:
        """Train model with configuration."""
        config = {
            'model_type': model_name,
            'ticker': resolved_ticker,
            'timeframe': timeframe,
            'target_col': target_col,
            'task_type': task_type
        }
        return cast(Optional[Dict[str, Any]], light_trainer.train_light_model(train_df, config))
    
    def _calculate_metrics(self, y_test: pd.Series, preds: np.ndarray, task_type: str) -> Dict[str, float]:
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
    
    def _save_trained_model(self, light_trainer: Any, result: Dict[str, Any], batch_dir: Path,
                           model_name: str, resolved_ticker: str, target_col: str) -> Path:
        """Save trained model to disk."""
        models_dir = batch_dir / 'models'
        models_dir.mkdir(parents=True, exist_ok=True)
        model_path = models_dir / f"{model_name}_{resolved_ticker}_{target_col}.joblib"
        light_trainer.save_model_to_disk(result['model_key'], str(model_path))
        self.logger.debug(f"💾 Model saved: {model_path}")
        return model_path
    
    def _create_model_metadata(self, resolved_ticker: str, target_col: str, model_name: str,
                              metrics: Dict[str, float], model_path: Path, available_features: List[str]) -> Dict[str, Any]:
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
    
    def _resolve_ticker(self, context_ticker: str, c_features_df: pd.DataFrame, ticker_col: Optional[str]) -> Optional[str]:
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
