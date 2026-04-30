# src/pipeline/stages/stage_4_modeling.py

"""
Stage 4: Modeling and Selection (Unified Advanced ML Arena)

Implements a hybrid workflow using all available architectures,
automated training management, and deep context analytics through the UnifiedTrainingManager.
"""

import os
import datetime
import json
import aiofiles
from typing import Optional, Any, Dict, Tuple, List
from pathlib import Path
from dataclasses import dataclass

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

@dataclass
class TargetProcessingConfig:
    """Configuration for target processing."""
    ticker: str
    df: Any
    target_name: str
    timeframe: str
    champions: Dict[str, Any]

@dataclass
class FeatureLoadingConfig:
    """Configuration for feature loading."""
    model_type: str
    ticker: str
    target_name: str
    batch_dir: Path
    prepared_data: Dict[str, Any]

@dataclass
class TrainingDebugInfo:
    """Training debug information."""
    context_key: str
    winner_name: str
    winner_metrics: Dict[str, Any]
    all_metrics: Dict[str, Any]
    selected_features: List[str]

@dataclass
class SyncFeatureLoadingConfig:
    """Configuration for synchronous feature loading."""
    model_type: str
    ticker: str
    target_name: str
    batch_dir: Path
    x_train: Any

@dataclass
class SuccessfulTrainingConfig:
    """Configuration for successful training processing."""
    ticker: str
    target_name: str
    timeframe: str
    prepared_data: Dict[str, Any]
    ticker_result: Dict[str, Any]
    comparison_report: Dict[str, Any]
    champions: Dict[str, Any]

@dataclass
class ChampionInfoConfig:
    """Configuration for champion info creation."""
    ticker: str
    target_name: str
    winner_name: str
    comparison_report: Dict[str, Any]
    context_fingerprint: str
    market_regime: str
    winner_metrics: Dict[str, Any]
    all_metrics: Dict[str, Any]
    ticker_result: Dict[str, Any]
    selected_features: List[str]

@dataclass
class SingleModelTrainingConfig:
    """Configuration for single light model training."""
    model_type: str
    ticker: str
    target_name: str
    batch_dir: Path
    x_train: Any
    y_train: Any
    x_test: Any
    y_test: Any
    task_type: str
    light_trainer: Any
    context_fingerprint: str
    market_regime: str
    volatility_regime: str

@dataclass
class LightModelChampionConfig:
    """Configuration for light model champion info creation."""
    ticker: str
    target_name: str
    model_type: str
    model_key: str
    selected_features: List[str]
    metrics: Dict[str, Any]
    model_path: Path
    context_fingerprint: str
    market_regime: str
    volatility_regime: str

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

    def _validate_and_normalize_data(self, enriched_data) -> bool:
        """Validate and normalize input data."""
        if not enriched_data:
            logger.error("Enriched data not found in pipeline_data. Skipping Modeling Stage.")
            return False
        
        if isinstance(enriched_data, pd.DataFrame):
            enriched_data = normalize_metadata_columns(enriched_data)
            logger.info("Normalized enriched_data at stage entry")
        
        return True
    

    async def run(self, **kwargs) -> Dict[str, Any]:
        """Runs the full training cycle using UnifiedTrainingManager."""
        enriched_data = kwargs.get('enriched_data')
        
        if not self._validate_and_normalize_data(enriched_data):
            return {}
        
        champions = {}
        logger.info("--- [Modeling Stage] Starting Unified Training Flow ---")
        
        ticker_groups = enriched_data.groupby('ticker') if isinstance(enriched_data, pd.DataFrame) else enriched_data.items()
        
        for ticker, df in ticker_groups:
            await self._process_ticker_with_async(ticker, df, champions)

        logger.info(f"Modeling Stage complete. Trained {len(champions)} champion models.")
        
        return {
            'models_metadata': champions,
            'processed_data': enriched_data
        }

    async def _process_ticker_with_async(self, ticker, df, champions):
        """Process data for a single ticker with async operations."""
        try:
            target_cols = [c for c in df.columns if c.startswith('target_')]
            timeframe = df['timeframe'].iloc[-1] if 'timeframe' in df.columns else "1d"
            
            for target_name in target_cols:
                config = TargetProcessingConfig(
                    ticker=ticker,
                    df=df,
                    target_name=target_name,
                    timeframe=timeframe,
                    champions=champions
                )
                await self._process_target_with_async(config)
                
        except Exception as e:
            self.handle_stage_error(e, context=f"Modeling-{ticker}", severity="error")

    async def _process_target_with_async(self, config: TargetProcessingConfig):
        """Process data for a single target with async operations."""
        context_key = f"{config.ticker}_{config.target_name}"
        
        # 1. Unified ML Data Preparation
        prepared_data = prepare_data_for_models(
            df=config.df,
            ticker=config.ticker,
            timeframe=config.timeframe,
            target_cols=[config.target_name],
            test_size=self.modeling_config.get('test_size', DEFAULT_TEST_SIZE)
        )

        if not prepared_data:
            logger.warning(f"Data preparation failed for {context_key}. Skipping.")
            return

        # 2. Execute Training via Manager 
        training_results = self.training_manager.execute_unified_training(
            tickers=[config.ticker], 
            data_context=prepared_data
        )
        
        # 3. Model Comparison and Champion Selection
        comparison_report = self.comparison_analyzer.compare_models(
            training_results, 
            market_context=self.brain.get('market_regime', 'neutral')
        )
        
        # 4. Process Results and Load Features
        ticker_result = training_results.get('tickers_results', {}).get(config.ticker, {})
        if ticker_result.get('status') == 'success':
            training_config = SuccessfulTrainingConfig(
                ticker=config.ticker,
                target_name=config.target_name,
                timeframe=config.timeframe,
                prepared_data=prepared_data,
                ticker_result=ticker_result,
                comparison_report=comparison_report,
                champions=config.champions
            )
            await self._process_successful_training(training_config)

    async def _process_successful_training(self, config: SuccessfulTrainingConfig):
        """Process successful training results and load features."""
        context_key = f"{config.ticker}_{config.target_name}"
        context_fingerprint = self._get_context_fingerprint(config.ticker_result)
        
        # Extract metrics
        winner_name = config.ticker_result.get('winner')
        all_metrics = config.ticker_result.get('metrics', {})
        winner_metrics = config.ticker_result.get('winner_metrics', all_metrics.get(winner_name, {}))
        
        # Load selected features
        batch_dir = self._resolve_selected_features_batch_dir()
        feature_config = FeatureLoadingConfig(
            model_type=winner_name,
            ticker=config.ticker,
            target_name=config.target_name,
            batch_dir=batch_dir,
            prepared_data=config.prepared_data
        )
        selected_features = await self._load_selected_features_async(feature_config)
        
        # Log debug information
        debug_info = TrainingDebugInfo(
            context_key=context_key,
            winner_name=winner_name,
            winner_metrics=winner_metrics,
            all_metrics=all_metrics,
            selected_features=selected_features
        )
        self._log_training_debug_info(debug_info)
        
        # Create context and champion info
        market_regime = self.brain.get('market_regime', 'neutral')
        volatility_regime = self.brain.get('volatility_regime', 'normal')
        
        champion_config = ChampionInfoConfig(
            ticker=config.ticker,
            target_name=config.target_name,
            winner_name=winner_name,
            comparison_report=config.comparison_report,
            context_fingerprint=context_fingerprint,
            market_regime=market_regime,
            winner_metrics=winner_metrics,
            all_metrics=all_metrics,
            ticker_result=config.ticker_result,
            selected_features=selected_features
        )
        champion_info = self._create_champion_info(champion_config)
        
        config.champions[context_key] = champion_info
        logger.info(f"✅ Saved heavy model champion_info with {len(selected_features)} features and metrics")
        
        # Train light models locally
        light_models_trained = self._train_light_models_locally(
            config.ticker,
            config.target_name,
            config.prepared_data,
            batch_dir,
            context_fingerprint,
            market_regime,
            volatility_regime
        )
        
        config.champions.update(light_models_trained)
        logger.info(f"✅ Trained {len(light_models_trained)} light models locally")
        
        # Log to experience diary
        self._log_to_diary(champion_info, config.timeframe)
        for light_key, light_info in light_models_trained.items():
            self._log_to_diary(light_info, config.timeframe)

    def _get_context_fingerprint(self, ticker_result):
        """Get context fingerprint from ticker result."""
        # This would need access to df, but for now return a default
        return ticker_result.get('context_fingerprint', 'unknown')

    async def _load_selected_features_async(self, config: FeatureLoadingConfig):
        """Load selected features from files with async operations."""
        selected_features = []
        
        # Try specific file patterns
        file_candidates = [
            config.batch_dir / f"selected_features_{config.model_type}_{config.ticker}_{config.target_name}.json",
            config.batch_dir / f"selected_features_{config.model_type}_{config.ticker}.json",
            config.batch_dir / f"selected_features_{config.model_type}_{config.target_name}.json",
            config.batch_dir / f"selected_features_{config.model_type}.json"
        ]
        
        for candidate in file_candidates:
            if candidate.exists():
                selected_features = await self._try_load_features_file_async(candidate, config.model_type)
                if selected_features:
                    break
        
        # Try glob pattern if no specific files found
        if not selected_features:
            glob_candidates = list(config.batch_dir.glob(f"selected_features_{config.model_type}*.json"))
            for candidate in glob_candidates:
                selected_features = await self._try_load_features_file_async(candidate, config.model_type, is_glob=True)
                if selected_features:
                    break
        
        # Fallback to prepared data features
        if not selected_features:
            logger.warning(f"⚠️ No selected_features found for {config.model_type}. Using fallback features.")
            selected_features = config.prepared_data.get('light_models', {}).get('feature_names', [])
        
        return selected_features

    async def _try_load_features_file_async(self, candidate, model_type, is_glob=False):
        """Try to load features from a single file asynchronously."""
        try:
            async with aiofiles.open(candidate, 'r', encoding='utf-8') as f:
                content = await f.read()
                feat_data = json.loads(content)
                selected_features = feat_data.get('selected_features', [])
                if selected_features:
                    log_msg = f"✅ Loaded {len(selected_features)} features for {model_type} from {candidate.name}"
                    if is_glob:
                        log_msg += " (glob fallback)"
                    logger.info(log_msg)
                    return selected_features
        except Exception as e:
            context = f"LoadFeaturesGlob-{candidate.name}" if is_glob else f"LoadFeatures-{candidate.name}"
            self.handle_stage_error(e, context=context, severity="warning")
            logger.warning(f"⚠️ Failed to load features from {candidate}: {e}")
        
        return []

    def _log_training_debug_info(self, debug_info: TrainingDebugInfo):
        """Log debug information about training results."""
        logger.info(f"📊 Stage 4 - {debug_info.context_key}:")
        logger.info(f"   winner: {debug_info.winner_name}")
        logger.info(f"   winner_metrics: {debug_info.winner_metrics}")
        logger.info(f"   all_metrics: {debug_info.all_metrics}")
        logger.info(f"   selected_features: {len(debug_info.selected_features)} features")
        logger.info(f"   First 5 features: {debug_info.selected_features[:5]}")

    def _create_champion_info(self, config: ChampionInfoConfig):
        """Create champion info dictionary."""
        context_map = {
            'context_fingerprint': config.context_fingerprint,
            'market_regime': config.market_regime,
            'volatility_regime': self.brain.get('volatility_regime', 'normal'),
            'timestamp': datetime.datetime.now().isoformat()
        }
        
        return {
            "ticker": config.ticker,
            "target": config.target_name,
            "winner": config.comparison_report.get('champion_model', config.winner_name),
            "champion_reason": config.comparison_report.get('selection_reason', 'Top accuracy'),
            "context": config.context_fingerprint,
            "context_map": context_map,
            "market_regime": config.market_regime,
            "timestamp": datetime.datetime.now().isoformat(),
            "metrics": config.winner_metrics,
            "all_models_metrics": config.all_metrics,
            "model_path": config.ticker_result.get('model_path'),
            "selected_features": config.selected_features,
            "feature_count": len(config.selected_features)
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
        
        light_models = {}
        light_trainer = LightModelTrainer()
        
        # Get prepared data and validate
        training_data = self._get_light_model_training_data(prepared_data)
        if not training_data:
            return light_models
        
        x_train, y_train, x_test, y_test = training_data
        task_type = self._determine_task_type(target_name)
        light_model_types = self._get_light_model_types()
        
        for model_type in light_model_types:
            try:
                training_config = SingleModelTrainingConfig(
                    model_type=model_type,
                    ticker=ticker,
                    target_name=target_name,
                    batch_dir=batch_dir,
                    x_train=x_train,
                    y_train=y_train,
                    x_test=x_test,
                    y_test=y_test,
                    task_type=task_type,
                    light_trainer=light_trainer,
                    context_fingerprint=context_fingerprint,
                    market_regime=market_regime,
                    volatility_regime=volatility_regime
                )
                model_result = self._train_single_light_model(training_config)
                if model_result:
                    context_key = f"{ticker}_{target_name}_{model_type}"
                    light_models[context_key] = model_result
                    
            except Exception as e:
                logger.error(f"❌ Failed to train {model_type}: {e}", exc_info=True)
                continue
        
        return light_models

    def _get_light_model_training_data(self, prepared_data):
        """Extract and validate training data for light models."""
        X_train = prepared_data.get('light_models', {}).get('X_train')
        y_train = prepared_data.get('light_models', {}).get('y_train')
        X_test = prepared_data.get('light_models', {}).get('X_test')
        y_test = prepared_data.get('light_models', {}).get('y_test')
        
        if X_train is None or y_train is None:
            logger.warning("⚠️ No training data available for light models")
            return None
        
        return X_train, y_train, X_test, y_test

    def _determine_task_type(self, target_name):
        """Determine if this is a regression or classification task."""
        return 'regression' if 'return' in target_name or 'price' in target_name else 'classification'

    def _get_light_model_types(self):
        """Get list of light model types to train."""
        return ['catboost', 'lightgbm', 'xgboost', 'random_forest', 'linear', 'svm', 'knn']

    def _train_single_light_model(self, config: SingleModelTrainingConfig):
        """Train a single light model and return its champion info."""
        # Load selected features
        sync_config = SyncFeatureLoadingConfig(
            model_type=config.model_type,
            ticker=config.ticker,
            target_name=config.target_name,
            batch_dir=config.batch_dir,
            x_train=config.x_train
        )
        selected_features = self._load_selected_features_sync(sync_config)
        
        if not selected_features:
            logger.warning(f"⚠️ No features available for {config.model_type}, skipping")
            return None
        
        logger.info(f"🔧 Training {config.model_type} with {len(selected_features)} features...")
        
        # Filter data to selected features
        x_train_filtered = config.x_train[selected_features]
        x_test_filtered = config.x_test[selected_features]
        
        # Train model
        train_df = x_train_filtered.copy()
        train_df[config.target_name] = config.y_train.values
        
        result = config.light_trainer.train_light_model(
            features_df=train_df,
            target_col=config.target_name
        )
        
        # Make predictions and calculate metrics
        predictions = config.light_trainer.predict(result['model_key'], x_test_filtered)
        metrics = self._calculate_model_metrics(config.y_test, predictions, config.task_type)
        
        # Save model to disk
        model_path = self.models_dir / f"{config.model_type}_{config.ticker}_{config.target_name}.joblib"
        config.light_trainer.save_model_to_disk(result['model_key'], str(model_path))
        
        # Create champion info
        champion_config = LightModelChampionConfig(
            ticker=config.ticker,
            target_name=config.target_name,
            model_type=config.model_type,
            model_key=result['model_key'],
            selected_features=selected_features,
            metrics=metrics,
            model_path=model_path,
            context_fingerprint=config.context_fingerprint,
            market_regime=config.market_regime,
            volatility_regime=config.volatility_regime
        )
        champion_info = self._create_light_model_champion_info(champion_config)
        
        logger.info(f"✅ {config.model_type}: score={metrics['score']:.4f}, features={len(selected_features)}")
        return champion_info

    def _load_selected_features_sync(self, config: SyncFeatureLoadingConfig):
        """Load selected features synchronously for light models."""
        # Try specific file patterns
        file_candidates = [
            config.batch_dir / f"selected_features_{config.model_type}_{config.ticker}_{config.target_name}.json",
            config.batch_dir / f"selected_features_{config.model_type}_{config.ticker}.json",
            config.batch_dir / f"selected_features_{config.model_type}_{config.target_name}.json",
            config.batch_dir / f"selected_features_{config.model_type}.json"
        ]

        for candidate in file_candidates:
            if candidate.exists():
                selected_features = self._try_load_features_file_sync(candidate, config.model_type)
                if selected_features:
                    return selected_features

        # Try glob pattern if no specific files found
        glob_candidates = list(config.batch_dir.glob(f"selected_features_{config.model_type}*.json"))
        for candidate in glob_candidates:
            selected_features = self._try_load_features_file_sync(candidate, config.model_type, is_glob=True)
            if selected_features:
                return selected_features

        # Fallback: use all available features
        logger.info(f"ℹ️ No selected features file for {config.model_type}, using all available features as fallback")
        exclude_cols = ['datetime', 'ticker', 'published_at', 'news_id', 'news_title', 'news_sentiment'] + [c for c in config.x_train.columns if c.startswith('target_')]
        return [c for c in config.x_train.columns if c not in exclude_cols]

    def _try_load_features_file_sync(self, candidate, model_type, is_glob=False):
        """Try to load features from a single file synchronously."""
        try:
            with open(candidate, 'r', encoding='utf-8') as f:
                feat_data = json.load(f)
                selected_features = feat_data.get('selected_features', [])
                if selected_features:
                    log_msg = f"✅ Loaded {len(selected_features)} features for {model_type} from {candidate.name}"
                    if is_glob:
                        log_msg += " (glob fallback)"
                    logger.info(log_msg)
                    return selected_features
        except Exception as e:
            logger.warning(f"⚠️ Failed to load selected features file {candidate}: {e}")
        
        return []

    def _calculate_model_metrics(self, y_test, predictions, task_type):
        """Calculate metrics for the trained model."""
        from sklearn.metrics import mean_squared_error, accuracy_score
        import numpy as np
        
        if task_type == 'regression':
            mse = mean_squared_error(y_test, predictions)
            score = -mse  # Negative MSE for consistency (higher is better)
            return {
                'mse': float(mse),
                'rmse': float(np.sqrt(mse)),
                'score': float(score)
            }
        else:
            accuracy = accuracy_score(y_test, predictions)
            return {
                'accuracy': float(accuracy),
                'score': float(accuracy)
            }

    def _create_light_model_champion_info(self, config: LightModelChampionConfig):
        """Create champion info for light model."""
        context_map = {
            'context_fingerprint': config.context_fingerprint,
            'market_regime': config.market_regime,
            'volatility_regime': config.volatility_regime,
            'timestamp': datetime.datetime.now().isoformat()
        }
        
        return {
            "ticker": config.ticker,
            "target": config.target_name,
            "winner": config.model_type,
            "model_type": config.model_type,
            "champion_reason": f"Light model trained locally with {len(config.selected_features)} features",
            "context": config.context_fingerprint,
            "context_map": context_map,
            "market_regime": config.market_regime,
            "timestamp": datetime.datetime.now().isoformat(),
            "metrics": config.metrics,
            "model_path": str(config.model_path),
            "model_key": config.model_key,
            "selected_features": config.selected_features,
            "feature_count": len(config.selected_features),
            "model_category": "light"
        }

    def _resolve_selected_features_batch_dir(self) -> Path:
        """
        Resolves the batch directory for selected features.
        Priority:
        1. runtime params configured in paths/system
        2. runtime params found inside accumulated batch folders
        3. data/colab/accumulated/main_database (default)
        """
        runtime_params_path = self.config_manager.get_runtime_params_path()
        accumulated_dir = Path(self.config_manager.get('system.accumulation.output_dir', 'data/colab/accumulated'))
        
        # Try to get batch_name from main runtime params
        batch_name = self._try_get_batch_name_from_runtime_params(runtime_params_path)
        
        # If not found, search nested accumulated batch dirs
        if not batch_name:
            batch_name = self._search_nested_runtime_params(accumulated_dir)
        
        # Default to main_database if still not found
        if not batch_name:
            batch_name = 'main_database'
            logger.warning(f"⚠️ No batch_name found in runtime params, defaulting to {batch_name}")
        
        return accumulated_dir / batch_name

    def _try_get_batch_name_from_runtime_params(self, runtime_params_path):
        """Try to get batch_name from the main runtime params file."""
        if not runtime_params_path.exists():
            return None
        
        try:
            with open(runtime_params_path, 'r') as f:
                runtime_params = json.load(f)
                batch_name = runtime_params.get('batch', {}).get('batch_name')
                if batch_name:
                    logger.debug(f"✅ Resolved batch_name from {runtime_params_path}: {batch_name}")
                return batch_name
        except Exception as e:
            logger.warning(f"⚠️ Failed to load batch_name from runtime_params.json: {e}")
            return None

    def _search_nested_runtime_params(self, accumulated_dir):
        """Search nested accumulated batch dirs for runtime_params.json with batch_name."""
        try:
            runtime_files = self._get_sorted_runtime_files(accumulated_dir)
            return self._find_first_batch_name_in_files(runtime_files)
        except Exception as e:
            logger.warning(f"⚠️ Failed to search nested accumulated runtime_params files: {e}")
            return None

    def _get_sorted_runtime_files(self, accumulated_dir):
        """Get runtime_params.json files sorted by modification time (newest first)."""
        return sorted(
            accumulated_dir.glob('**/runtime_params.json'),
            key=lambda p: p.stat().st_mtime,
            reverse=True
        )

    def _find_first_batch_name_in_files(self, runtime_files):
        """Find the first runtime_params.json file that contains a batch_name."""
        for runtime_file in runtime_files:
            batch_name = self._try_get_batch_name_from_file(runtime_file)
            if batch_name:
                logger.debug(f"✅ Resolved batch_name from nested runtime_params: {runtime_file} -> {batch_name}")
                return batch_name
        return None

    def _try_get_batch_name_from_file(self, runtime_file):
        """Try to get batch_name from a specific runtime_params.json file."""
        try:
            with open(runtime_file, 'r') as f:
                runtime_params = json.load(f)
                return runtime_params.get('batch', {}).get('batch_name')
        except Exception:
            return None
