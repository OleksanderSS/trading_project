
"""
Stage 4: Modeling and Selection (Unified Advanced ML Arena)

Implements a hybrid workflow using all available architectures,
automated training management, and deep context analytics through the UnifiedTrainingManager.
"""

import logging
import os
import datetime
import json
from typing import Optional, Any, Dict, Tuple, List
from pathlib import Path

import pandas as pd
import psutil

from src.pipeline.stages.base_stage import BaseStage
from src.config.unified_config_manager import UnifiedConfigManager
from src.core.logging.logger import ProjectLogger
from src.training.unified_training_manager import UnifiedTrainingManager, UnifiedConfig, TrainingStrategy
from src.models.adapters.data_preparation import prepare_data_for_models
from src.analytics.analyzers.model_comparison_analyzer import ModelComparisonAnalyzer
from src.features.utils.datetime_utils import ensure_datetime_column, normalize_metadata_columns
from src.training.constants import (
    BATCH_TRAINER_DEFAULT_BATCH_SIZE,
    BATCH_TRAINER_DEFAULT_MAX_MEMORY_GB,
    DEFAULT_TEST_SIZE
)

logger = ProjectLogger.get_logger("ModelingStage")

class ModelingStage(BaseStage):
    """
    Modeling stage: uses UnifiedTrainingManager for training orchestration
    and prepare_data_for_models for unified dataset formatting.
    Supports dynamic switching between Light (local) and Heavy (Colab) training.
    """
    def __init__(self, config_manager: UnifiedConfigManager, brain: Dict[str, Any], **kwargs):
        super().__init__(config_manager, brain)
        self.modeling_config = self.config_manager.get_config('modeling') or {}
        self.system_config = self.config_manager.get_config('system') or {}
        
        # Initialize strategy based on config
        strategy_str = self.modeling_config.get('strategy', 'hybrid').upper()
        strategy = TrainingStrategy[strategy_str] if strategy_str in TrainingStrategy.__members__ else TrainingStrategy.HYBRID
        
        training_config = UnifiedConfig(
            strategy=strategy,
            batch_size=self.modeling_config.get('batch_size', BATCH_TRAINER_DEFAULT_BATCH_SIZE),
            max_memory_gb=self.modeling_config.get('max_memory_gb', BATCH_TRAINER_DEFAULT_MAX_MEMORY_GB)
        )
        
        self.training_manager = UnifiedTrainingManager(training_config)
        self.comparison_analyzer = ModelComparisonAnalyzer()
        
        # Get paths from config using centralized getters
        self.models_dir = self.config_manager.get_models_path()
        self.diary_path = Path(self.system_config.get('diary_path', 'logs/experience_diary.csv'))

        self._init_infrastructure()

    def _init_infrastructure(self):
        """Initializes the environment for training artifacts."""
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        if not self.diary_path.exists():
            self.diary_path.parent.mkdir(parents=True, exist_ok=True)
            columns = [
                'timestamp', 'ticker', 'tf', 'target', 'model_name', 'context_fingerprint', 
                'is_champion', 'cpu_usage', 'ram_usage'
            ]
            pd.DataFrame(columns=columns).to_csv(self.diary_path, index=False)

    async def run(self, **kwargs) -> Dict[str, Any]:
        """Runs the full training cycle using UnifiedTrainingManager."""
        enriched_data = kwargs.get('enriched_data')
        if not enriched_data:
            logger.error("Enriched data not found in pipeline_data. Skipping Modeling Stage.")
            return {}

        # ✅ CRITICAL FIX: Normalize datetime columns at stage entry
        if isinstance(enriched_data, pd.DataFrame):
            enriched_data = normalize_metadata_columns(enriched_data)
            logger.info(f"✅ Normalized enriched_data at stage entry")

        champions = {}
        logger.info("--- [Modeling Stage] Starting Unified Training Flow ---")

        ticker_groups = enriched_data.groupby('ticker') if isinstance(enriched_data, pd.DataFrame) else enriched_data.items()

        for ticker, df in ticker_groups:
            try:
                target_cols = [c for c in df.columns if c.startswith('target_')]
                timeframe = df['timeframe'].iloc[-1] if 'timeframe' in df.columns else "1d"
                
                for target_name in target_cols:
                    context_key = f"{ticker}_{target_name}"
                    
                    # 1. Unified ML Data Preparation
                    prepared_data = prepare_data_for_models(
                        df=df,
                        ticker=ticker,
                        timeframe=timeframe,
                        target_cols=[target_name],
                        test_size=self.modeling_config.get('test_size', DEFAULT_TEST_SIZE)
                    )

                    if not prepared_data:
                        logger.warning(f"Data preparation failed for {context_key}. Skipping.")
                        continue

                    # 2. Execute Training via Manager 
                    training_results = self.training_manager.execute_unified_training(
                        tickers=[ticker], 
                        data_context=prepared_data
                    )
                    
                    # 3. Model Comparison and Champion Selection
                    comparison_report = self.comparison_analyzer.compare_models(
                        training_results, 
                        market_context=self.brain.get('market_regime', 'neutral')
                    )
                    
                    # 4. Process Results
                    ticker_result = training_results.get('tickers_results', {}).get(ticker, {})
                    if ticker_result.get('status') == 'success':
                        context_fingerprint = df['context_fingerprint'].iloc[-1] if 'context_fingerprint' in df.columns else "unknown"
                        
                        # ✅ Витягуємо metrics з результатів тренування
                        winner_name = ticker_result.get('winner')
                        all_metrics = ticker_result.get('metrics', {})
                        winner_metrics = ticker_result.get('winner_metrics', all_metrics.get(winner_name, {}))
                        
                        # ✅ Читаємо selected_features з файлів Colab
                        selected_features = []
                        batch_dir = self._resolve_selected_features_batch_dir()
                        
                        # Читаємо selected_features з файлу, підтримуємо кілька шаблонів імен
                        model_type = winner_name or 'unknown'
                        selected_features = []
                        file_candidates = [
                            batch_dir / f"selected_features_{model_type}_{ticker}_{target_name}.json",
                            batch_dir / f"selected_features_{model_type}_{ticker}.json",
                            batch_dir / f"selected_features_{model_type}_{target_name}.json",
                            batch_dir / f"selected_features_{model_type}.json"
                        ]
                        for candidate in file_candidates:
                            if candidate.exists():
                                try:
                                    with open(candidate, 'r', encoding='utf-8') as f:
                                        feat_data = json.load(f)
                                        selected_features = feat_data.get('selected_features', [])
                                        if selected_features:
                                            logger.info(f"✅ Завантажено {len(selected_features)} фіч для {model_type} з {candidate.name}")
                                            break
                                except Exception as e:
                                    logger.warning(f"⚠️ Не вдалося завантажити фічи з {candidate}: {e}")

                        if not selected_features:
                            glob_candidates = list(batch_dir.glob(f"selected_features_{model_type}*.json"))
                            if glob_candidates:
                                for candidate in glob_candidates:
                                    try:
                                        with open(candidate, 'r', encoding='utf-8') as f:
                                            feat_data = json.load(f)
                                            selected_features = feat_data.get('selected_features', [])
                                            if selected_features:
                                                logger.info(f"✅ Завантажено {len(selected_features)} фіч для {model_type} з {candidate.name} (glob fallback)")
                                                break
                                    except Exception as e:
                                        logger.warning(f"⚠️ Не вдалося завантажити фічи з {candidate}: {e}")

                        if not selected_features:
                            logger.warning(f"⚠️ Не знайдено selected_features для {model_type}. Використовуємо fallback-фічи.")
                        selected_features = prepared_data.get('light_models', {}).get('feature_names', [])
                        
                        feature_count = len(selected_features)
                        
                        # ✅ DEBUG: Логуємо інформацію про фічи та metrics
                        logger.info(f"📊 Stage 4 - {context_key}:")
                        logger.info(f"   winner: {winner_name}")
                        logger.info(f"   winner_metrics: {winner_metrics}")
                        logger.info(f"   all_metrics: {all_metrics}")
                        logger.info(f"   selected_features: {len(selected_features)} фіч")
                        logger.info(f"   Перші 5 фіч: {selected_features[:5]}")
                        
                        # ✅ Витягуємо market regime з контексту
                        market_regime = self.brain.get('market_regime', 'neutral')
                        volatility_regime = self.brain.get('volatility_regime', 'normal')
                        
                        # ✅ Створюємо context map
                        context_map = {
                            'context_fingerprint': context_fingerprint,
                            'market_regime': market_regime,
                            'volatility_regime': volatility_regime,
                            'timestamp': datetime.datetime.now().isoformat()
                        }
                        
                        champion_info = {
                            "ticker": ticker,
                            "target": target_name,
                            "winner": comparison_report.get('champion_model', winner_name),
                            "champion_reason": comparison_report.get('selection_reason', 'Top accuracy'),
                            "context": context_fingerprint,
                            "context_map": context_map,  # ✅ Додаємо context map
                            "market_regime": market_regime,  # ✅ Додаємо market regime
                            "timestamp": datetime.datetime.now().isoformat(),
                            "metrics": winner_metrics,  # ✅ Metrics чемпіона
                            "all_models_metrics": all_metrics,  # ✅ Metrics всіх моделей
                            "model_path": ticker_result.get('model_path'),
                            "selected_features": selected_features,
                            "feature_count": feature_count
                        }
                        champions[context_key] = champion_info
                        logger.info(f"✅ Збережено heavy model champion_info з {len(selected_features)} фічами та metrics")
                        
                        # 5. Train Light Models Locally
                        logger.info(f"🚀 Starting local light model training for {context_key}...")
                        light_models_trained = self._train_light_models_locally(
                            ticker=ticker,
                            target_name=target_name,
                            timeframe=timeframe,
                            prepared_data=prepared_data,
                            batch_dir=batch_dir,
                            context_fingerprint=context_fingerprint,
                            market_regime=market_regime,
                            volatility_regime=volatility_regime
                        )
                        
                        # Додаємо light моделі до champions
                        champions.update(light_models_trained)
                        logger.info(f"✅ Trained {len(light_models_trained)} light models locally")
                        
                        # 6. Log to Experience Diary
                        self._log_to_diary(champion_info, timeframe)
                        
                        # Log light models to diary
                        for light_key, light_info in light_models_trained.items():
                            self._log_to_diary(light_info, timeframe)

            except Exception as e:
                logger.error(f"Error during modeling for ticker {ticker}: {e}", exc_info=True)

        logger.info(f"Modeling Stage complete. Trained {len(champions)} champion models.")
        
        return {
            'models_metadata': champions,
            'processed_data': enriched_data
        }

    def _log_to_diary(self, info: Dict[str, Any], tf: str):
        """Records champion selection metadata."""
        entry = {
            'timestamp': info['timestamp'],
            'ticker': info['ticker'], 
            'tf': tf,
            'target': info['target'], 
            'model_name': info['winner'],
            'context_fingerprint': info['context'], 
            'is_champion': True,
            'cpu_usage': psutil.cpu_percent(), 
            'ram_usage': psutil.virtual_memory().percent
        }
        pd.DataFrame([entry]).to_csv(self.diary_path, mode='a', header=False, index=False)

    def _train_light_models_locally(
        self, 
        ticker: str, 
        target_name: str, 
        timeframe: str,
        prepared_data: Dict[str, Any],
        batch_dir: Path,
        context_fingerprint: str,
        market_regime: str,
        volatility_regime: str
    ) -> Dict[str, Dict[str, Any]]:
        """
        Trains light models locally using selected features from Colab.
        
        Returns:
            Dict of {context_key: champion_info} for each trained light model
        """
        from src.training.light_model_trainer import LightModelTrainer
        from sklearn.metrics import mean_squared_error, accuracy_score
        import numpy as np
        
        light_models = {}
        light_trainer = LightModelTrainer()
        
        # Light model types to train (from models.yaml categories.light)
        light_model_types = ['catboost', 'lightgbm', 'xgboost', 'random_forest', 'linear', 'svm', 'knn']
        
        # Get prepared data
        X_train = prepared_data.get('light_models', {}).get('X_train')
        y_train = prepared_data.get('light_models', {}).get('y_train')
        X_test = prepared_data.get('light_models', {}).get('X_test')
        y_test = prepared_data.get('light_models', {}).get('y_test')
        
        if X_train is None or y_train is None:
            logger.warning(f"⚠️ No training data available for light models")
            return light_models
        
        # Determine task type
        task_type = 'regression' if 'return' in target_name or 'price' in target_name else 'classification'
        
        for model_type in light_model_types:
            try:
                # Load selected features for this model from Colab using flexible file patterns
                selected_features = None
                file_candidates = [
                    batch_dir / f"selected_features_{model_type}_{ticker}_{target_name}.json",
                    batch_dir / f"selected_features_{model_type}_{ticker}.json",
                    batch_dir / f"selected_features_{model_type}_{target_name}.json",
                    batch_dir / f"selected_features_{model_type}.json"
                ]

                for candidate in file_candidates:
                    if candidate.exists():
                        try:
                            with open(candidate, 'r', encoding='utf-8') as f:
                                feat_data = json.load(f)
                                selected_features = feat_data.get('selected_features', [])
                                if selected_features:
                                    logger.info(f"✅ Завантажено {len(selected_features)} фіч для {model_type} з {candidate.name}")
                                    break
                        except Exception as e:
                            logger.warning(f"⚠️ Failed to load selected features file {candidate}: {e}")

                if not selected_features:
                    glob_candidates = list(batch_dir.glob(f"selected_features_{model_type}*.json"))
                    for candidate in glob_candidates:
                        try:
                            with open(candidate, 'r', encoding='utf-8') as f:
                                feat_data = json.load(f)
                                selected_features = feat_data.get('selected_features', [])
                                if selected_features:
                                    logger.info(f"✅ Loaded {len(selected_features)} features for {model_type} from {candidate.name} (glob fallback)")
                                    break
                        except Exception as e:
                            logger.warning(f"⚠️ Failed to load selected features file {candidate}: {e}")

                # Fallback: use all features if no selected features were found
                if not selected_features:
                    logger.info(f"ℹ️ No selected features file for {model_type}, using all available features as fallback")
                    exclude_cols = ['datetime', 'ticker', 'published_at', 'news_id', 'news_title', 'news_sentiment'] + [c for c in X_train.columns if c.startswith('target_')]
                    selected_features = [c for c in X_train.columns if c not in exclude_cols]
                
                if not selected_features:
                    logger.warning(f"⚠️ No features available for {model_type}, skipping")
                    continue
                
                logger.info(f"🔧 Training {model_type} with {len(selected_features)} features...")
                
                # Filter data to selected features
                X_train_filtered = X_train[selected_features]
                X_test_filtered = X_test[selected_features]
                
                # Prepare DataFrame for training
                train_df = X_train_filtered.copy()
                train_df[target_name] = y_train.values
                
                # Train model
                result = light_trainer.train_light_model(
                    features_df=train_df,
                    model_type=model_type,
                    ticker=ticker,
                    timeframe=timeframe,
                    target_col=target_name,
                    task_type=task_type
                )
                
                # Make predictions
                predictions = light_trainer.predict(result['model_key'], X_test_filtered)
                
                # Calculate metrics
                if task_type == 'regression':
                    mse = mean_squared_error(y_test, predictions)
                    score = -mse  # Negative MSE for consistency (higher is better)
                    metrics = {
                        'mse': float(mse),
                        'rmse': float(np.sqrt(mse)),
                        'score': float(score)
                    }
                else:
                    accuracy = accuracy_score(y_test, predictions)
                    score = accuracy
                    metrics = {
                        'accuracy': float(accuracy),
                        'score': float(score)
                    }
                
                # Save model to disk
                model_path = self.models_dir / f"{model_type}_{ticker}_{target_name}.joblib"
                light_trainer.save_model_to_disk(result['model_key'], str(model_path))
                
                # Create context map
                context_map = {
                    'context_fingerprint': context_fingerprint,
                    'market_regime': market_regime,
                    'volatility_regime': volatility_regime,
                    'timestamp': datetime.datetime.now().isoformat()
                }
                
                # Create champion info
                context_key = f"{ticker}_{target_name}_{model_type}"
                light_models[context_key] = {
                    "ticker": ticker,
                    "target": target_name,
                    "winner": model_type,
                    "model_type": model_type,  # ✅ ADD для сумісності з Stage 5
                    "champion_reason": f"Light model trained locally with {len(selected_features)} features",
                    "context": context_fingerprint,
                    "context_map": context_map,
                    "market_regime": market_regime,
                    "timestamp": datetime.datetime.now().isoformat(),
                    "metrics": metrics,
                    "model_path": str(model_path),
                    "model_key": result['model_key'],
                    "selected_features": selected_features,
                    "feature_count": len(selected_features),
                    "model_category": "light"
                }
                
                logger.info(f"✅ {model_type}: score={score:.4f}, features={len(selected_features)}")
                
            except Exception as e:
                logger.error(f"❌ Failed to train {model_type}: {e}", exc_info=True)
                continue
        
        return light_models

    def _resolve_selected_features_batch_dir(self) -> Path:
        """
        Resolves the batch directory for selected features.
        Priority:
        1. data/runtime/runtime_params.json
        2. src/config/runtime_params.json (legacy fallback)
        3. data/colab/accumulated/main_database (default)
        """
        # Try to get batch_name from runtime params
        runtime_params_path = self.config_manager.get_runtime_params_path()
        batch_name = 'main_database'
        
        if runtime_params_path.exists():
            try:
                with open(runtime_params_path, 'r') as f:
                    runtime_params = json.load(f)
                    batch_name = runtime_params.get('batch', {}).get('batch_name', 'main_database')
                    logger.debug(f"✅ Resolved batch_name from {runtime_params_path}: {batch_name}")
            except Exception as e:
                logger.warning(f"⚠️ Failed to load batch_name from runtime_params.json: {e}")
        
        # Construct batch directory
        accumulated_dir = Path(self.config_manager.get('system.accumulation.output_dir', 'data/colab/accumulated'))
        batch_dir = accumulated_dir / batch_name
        
        return batch_dir
