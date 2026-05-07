# src/pipeline/hybrid_orchestrator.py
"""
Hybrid Pipeline Orchestrator:
- Local: Parsing, feature selection, light models
- Colab: Heavy models, heavy analyzers
- State persistence for long-running sessions
"""

import asyncio
import json
import os
import pickle
import time
import shutil
import aiofiles
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Callable, cast
from datetime import datetime
from dataclasses import dataclass

@dataclass
class ModelTrainingContext:
    """Context object for training models."""
    features_df: pd.DataFrame
    targets_df: pd.DataFrame
    ticker_col: Optional[str]
    batch_dir: Path
    light_trainer: Any

@dataclass
class ColabBatchRequest:
    """Data package request for Colab training."""
    features_df: pd.DataFrame
    targets_df: pd.DataFrame
    tickers: List[str]
    timeframes: List[str]
    batch_name: Optional[str] = None
    accumulate: bool = True
    check_feature_selection: bool = True
    force_feature_selection: bool = False

@dataclass
class HybridPipelineRequest:
    """Request for running full hybrid pipeline."""
    tickers: Optional[List[str]] = None
    timeframes: Optional[List[str]] = None
    accumulate: bool = True
    force_training: bool = False
    skip_colab: bool = False
    force_feature_selection: bool = False

from src.config.unified_config_manager import UnifiedConfigManager
from src.core.logging.logger import ProjectLogger
from src.pipeline.pipeline_orchestrator import PipelineOrchestrator
from src.data.management.data_manager import DataManager

# Import hybrid pipeline components
from src.pipeline.hybrid.data_manager import HybridDataManager
from src.pipeline.hybrid.feature_processor import FeatureProcessor
from src.pipeline.hybrid.cache_manager import CacheManager
from src.pipeline.hybrid.colab_manager import ColabManager
from src.pipeline.hybrid.pipeline_manager import PipelineManager
from src.pipeline.hybrid.orchestrator_config import OrchestratorConfigManager, PipelineConfig
from src.pipeline.hybrid.storage_manager import StorageManager
from src.pipeline.hybrid.pipeline_executor import PipelineExecutor
from src.pipeline.hybrid.feature_selection_manager import FeatureSelectionManager
from src.pipeline.hybrid.data_utils import DataUtils
from src.pipeline.hybrid.metadata_manager import MetadataManager
from src.pipeline.hybrid.orchestrator_interface import OrchestratorInterface
from src.pipeline.hybrid.pipeline_runner import PipelineRunner
from src.pipeline.hybrid.light_models_trainer import LightModelsTrainer
from src.pipeline.hybrid.colab_workflow_manager import ColabWorkflowManager
from src.pipeline.hybrid.final_stages_executor import FinalStagesExecutor
from src.pipeline.hybrid.data_processor import DataProcessor
from src.pipeline.stages.stage_0_data_generation import DataGenerator

# Import refactored managers
from src.pipeline.hybrid.data_batch_manager import DataBatchManager
from src.pipeline.hybrid.model_training_orchestrator import ModelTrainingOrchestrator
from src.pipeline.hybrid.context_builder import ContextBuilder
from src.pipeline.hybrid.feature_selection_validator import MockFeaturesRequest, FeatureSelectionValidator
from src.pipeline.hybrid.selected_features_processor import FeatureLoadRequest
from src.pipeline.hybrid.results_processor import ResultsProcessor
from src.pipeline.hybrid.data_cache_manager import DataCacheManager

# Import Phase 2 managers
from src.pipeline.hybrid.test_mode_manager import TestModeManager
from src.pipeline.hybrid.selected_features_processor import SelectedFeaturesProcessor
from src.pipeline.hybrid.pipeline_metadata_manager import PipelineMetadataManager
from src.pipeline.hybrid.final_stages_orchestrator import FinalStagesOrchestrator
from src.pipeline.hybrid.colab_workflow_manager import ColabWorkflowManager

logger = ProjectLogger.get_logger(__name__)

# Constants to avoid duplication
FEATURES_FILE = "features.parquet"
TARGETS_FILE = "targets.parquet"
BATCH_METADATA_FILE = "batch_metadata.json"
SELECTED_FEATURES_PATTERN = "selected_features_*.json"

# Additional file constants to avoid duplication
FEATURES_PARQUET = "features.parquet"
TARGETS_PARQUET = "targets.parquet"

# Google Drive API (optional)
try:
    from google.oauth2.credentials import Credentials
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaFileUpload, MediaIoBaseDownload
    GDRIVE_AVAILABLE = True
except ImportError:
    GDRIVE_AVAILABLE = False
    logger.warning("Google Drive API not installed. Use manual transfer.")

@dataclass
class HybridPipelineConfig:
    """Configuration for pipeline execution."""
    output_dir: Path
    models_dir: Path
    light_models: List[str]
    heavy_models: List[str]
    gdrive_enabled: bool = False


@dataclass
class HybridFinalStagesRequest:
    """Request for running final stages."""
    features_df: Optional[pd.DataFrame]
    targets_df: Optional[pd.DataFrame]
    colab_results: Optional[Dict[str, Any]] = None
    light_results: Optional[Dict[str, Any]] = None
    tickers: Optional[List[str]] = None
    timeframes: Optional[List[str]] = None
    batch_name: Optional[str] = None

@dataclass
class HybridMockFeaturesRequest:
    """Request for creating mock selected features for testing."""
    batch_dir: Path
    test_ticker: str
    test_target: str
    light_models: List[str]
    features_df: pd.DataFrame


class HybridOrchestrator:
    """
    Hybrid orchestrator for distributed pipeline execution.
    
    Acts as a coordination node between modern PipelineOrchestrator
    and Colab-oriented workflows. Local part performs Stages 0-3
    via PipelineOrchestrator, while final stages 4-7 are also
    delegated to the same modern orchestration.
    
    Locally:
    - Stage 0-3: Data collection, cleaning, feature engineering, feature selection
    - Light models (CatBoost, LightGBM, XGBoost, RF, Linear, SVM, KNN)
    
    In Colab:
    - Heavy models (LSTM, GRU, Transformer, TabNet, CNN, Autoencoder, MLP)
    - Heavy analyzers
    
    Persistence:
    - Intermediate results after each stage
    - Ability to restore after session disconnection
    """
    
    def __init__(self, config_manager: UnifiedConfigManager, batch_name: str = "main_database"):
        self.config_manager = config_manager
        self.logger = ProjectLogger.get_logger(__name__)
        self.batch_name = batch_name
        
        # Initialize configuration
        self.orchestrator_config_manager = OrchestratorConfigManager(config_manager)
        self.config = self.orchestrator_config_manager.build_pipeline_config(batch_name)
        
        # Initialize specialized components
        self._init_components()
        
        self.logger.info(f"HybridOrchestrator initialized for batch: {self.batch_name}")
    
    def _init_components(self):
        """Initialize specialized components."""
        self.data_manager = HybridDataManager(self.config.output_dir)
        self.feature_processor = FeatureProcessor()
        self.cache_manager = CacheManager(self.config.output_dir, self.batch_name)
        self.colab_manager = ColabManager(self.config.output_dir, self.batch_name)
        self.pipeline_manager = PipelineManager(self)
        self.storage_manager = StorageManager(self.config)
        self.pipeline_executor = PipelineExecutor(self.config_manager, self.config.output_dir, self.feature_processor)
        self.feature_selection_manager = FeatureSelectionManager(self.config.output_dir)
        self.data_utils = DataUtils()
        self.metadata_manager = MetadataManager(self.config_manager)
        self.interface = OrchestratorInterface(self)
        
        # New specialized components
        self.pipeline_runner = PipelineRunner(
            self.config_manager, str(self.config.output_dir), self.batch_name,
            self.feature_processor, self.metadata_manager
        )
        self.light_models_trainer = LightModelsTrainer({
            'config_manager': self.config_manager,
            'output_dir': self.config.output_dir,
            'batch_name': self.batch_name,
            'light_models': self.config.light_models,
            'models_config': self.config.models_config
        })
        self.colab_workflow_manager = ColabWorkflowManager(
            self.config.output_dir, self.batch_name, self.config.light_models
        )
        self.final_stages_executor = FinalStagesExecutor(
            self.config_manager, str(self.config.output_dir), self.batch_name
        )
        self.data_processor = DataProcessor(self.data_utils)
        
        # Initialize refactored managers
        self.data_batch_manager = DataBatchManager()
        self.model_training_orchestrator = ModelTrainingOrchestrator(self.config_manager)
        self.feature_selection_validator = FeatureSelectionValidator()
        self.results_processor = ResultsProcessor()
        self.data_cache_manager = DataCacheManager()
        
        # Initialize Phase 2 managers
        self.test_mode_manager = TestModeManager()
        self.context_builder = ContextBuilder(self.test_mode_manager)
        self.selected_features_processor = SelectedFeaturesProcessor(
            self.context_builder, self.feature_selection_validator
        )
        self.pipeline_metadata_manager = PipelineMetadataManager(
            self.config.output_dir, self.batch_name, 
            self.config.light_models, self.config.heavy_models
        )
        self.final_stages_orchestrator = FinalStagesOrchestrator(
            self.config_manager, self.config.output_dir, self.batch_name
        )
        
        # Initialize storage
        self.storage_manager.initialize_storage()
        
        # Initialize storage fallback settings
        self.use_s3 = self.config.use_s3
        self.use_gcs = self.config.use_gcs
        self.storage_fallback = self.config.storage_fallback or {}
        
        # Initialize missing components
        self.data_generator = DataGenerator(self.config_manager)
        self.feature_engineer = FeatureProcessor()
        self.heavy_models_trainer = None  # Will be initialized when needed
        
        self._log_initialization_status()
    
    def _log_initialization_status(self):
        """Log initialization status."""
        self.logger.info("🚀 Hybrid Orchestrator initialized")
        self.logger.info(f"📁 Output directory: {self.config.output_dir}")
        self.logger.info(f"💡 Light models: {self.config.light_models}")
        self.logger.info(f"🔥 Heavy models: {self.config.heavy_models}")
        
        if self.config.gdrive_enabled:
            self.logger.info("☁️ Google Drive: ✅ Enabled")
        else:
            self.logger.info("☁️ Google Drive: ❌ Disabled")
    
    def _init_gdrive(self):
        """Initializes Google Drive API."""
        try:
            from google.oauth2.credentials import Credentials
            from googleapiclient.discovery import build
            self.logger.info("✅ Google Drive API initialized")
        except Exception as e:
            self.logger.error(f"❌ Google Drive initialization error: {e}")
            if self._init_fallback_storage():
                self.logger.info("✅ Fallback storage initialized")
            else:
                raise
    
    def _init_fallback_storage(self) -> bool:
        """Initializes fallback storage (S3 or GCS)."""
        if self.use_s3:
            return self._init_s3_storage()
        elif self.use_gcs:
            return self._init_gcs_storage()
        
        self.logger.warning("⚠️ No fallback storage available, using manual transfer")
        return False
    
    def _init_s3_storage(self) -> bool:
        """Initialize S3 storage."""
        try:
            import boto3
            s3_config = self.storage_fallback.get('s3', {})
            self.s3_client = boto3.client(
                's3',
                aws_access_key_id=s3_config.get('access_key'),
                aws_secret_access_key=s3_config.get('secret_key'),
                region_name=s3_config.get('region', 'us-east-1')
            )
            self.s3_bucket = s3_config.get('bucket')
            self.logger.info(f"✅ S3 fallback initialized: {self.s3_bucket}")
            return True
        except Exception as e:
            self.logger.warning(f"⚠️ S3 fallback failed: {e}")
            return False
    
    def _init_gcs_storage(self) -> bool:
        """Initialize GCS storage."""
        try:
            from google.cloud import storage
            gcs_config = self.storage_fallback.get('gcs', {})
            self.gcs_client = storage.Client(project=gcs_config.get('project_id'))
            self.gcs_bucket = self.gcs_client.bucket(gcs_config.get('bucket'))
            self.logger.info(f"✅ GCS fallback initialized: {gcs_config.get('bucket')}")
            return True
        except Exception as e:
            self.logger.warning(f"⚠️ GCS fallback failed: {e}")
            return False
    
    def _resolve_target_task_type(self, target_name: str) -> str:
        """Maps configured targets to task type. Delegates to ModelTrainingOrchestrator."""
        return self.model_training_orchestrator._resolve_target_task_type(target_name)

    def _create_mock_selected_features_for_test(self, request: MockFeaturesRequest) -> List[Path]:
        """Create mock selected_features files for testing. Delegates to FeatureSelectionValidator."""
        return self.feature_selection_validator.create_mock_selected_features_for_test(request)

    async def run_local_pipeline(
        self, 
        tickers: Optional[List[str]] = None,
        timeframes: Optional[List[str]] = None,
        stages_to_run: Optional[List[int]] = None
    ) -> Dict[str, Any]:
        """Execute local pipeline with specified parameters using PipelineOrchestrator"""
        logger.info(f"🚀 Starting local pipeline for tickers: {tickers}, timeframes: {timeframes}")
        
        # Use default stages if not specified
        if stages_to_run is None:
            stages_to_run = [0, 1, 2, 3]  # Data generation, cleaning, feature engineering, preparation
        
        # Validate inputs
        if not tickers:
            return {'status': 'error', 'message': 'No tickers provided'}
        
        # Use PipelineOrchestrator for stages 0-3
        try:
            pipeline_orchestrator = PipelineOrchestrator(
                config_manager=self.config_manager,
                stages_to_run=stages_to_run
            )
            
            # Execute pipeline with proper data
            results = await pipeline_orchestrator.run(
                tickers=tickers,
                timeframes=timeframes,
                run_mode='train',
                batch_name=self.batch_name  # ✅ Передаємо batch_name
            )
            
            return {
                'status': 'completed',
                'results': results,
                'stages_run': stages_to_run
            }
            
        except Exception as e:
            logger.error(f"PipelineOrchestrator failed: {e}")
            return {
                'status': 'error', 
                'message': str(e),
                'stages_run': stages_to_run
            }
    
    async def _run_stage_0_data_generation(self) -> Dict[str, Any]:
        """Execute stage 0: Data generation"""
        logger.info("📊 Stage 0: Data generation")
        result = self.data_generator.generate_synthetic_data()
        return cast(Dict[str, Any], result)
    
    async def _run_stage_1_data_cleaning(self) -> Dict[str, Any]:
        """Execute stage 1: Data cleaning"""
        logger.info("🧹 Stage 1: Data cleaning")
        result = self.data_processor.clean_dataframe(pd.DataFrame())  # Pass empty DataFrame for now
        return cast(Dict[str, Any], result)
    
    async def _run_stage_2_feature_engineering(self) -> Dict[str, Any]:
        """Execute stage 2: Feature engineering"""
        logger.info("🔧 Stage 2: Feature engineering")
        # Feature engineering is handled by enrichers in stage 1
        # This stage is a placeholder for additional feature processing if needed
        return cast(Dict[str, Any], {"status": "completed"})
    
    
    async def _run_stage_5_heavy_model_training(self) -> Dict[str, Any]:
        """Execute stage 5: Heavy model training"""
        logger.info("🚀 Stage 5: Heavy model training")
        if self.heavy_models_trainer is None:
            self.heavy_models_trainer = self.light_models_trainer  # Fallback to light models
        result = self.heavy_models_trainer.train_models()
        return cast(Dict[str, Any], result)

    def _get_datetime_column(self, df: pd.DataFrame) -> Optional[str]:
        """Find datetime column."""
        return self.data_processor.get_datetime_column(df)

    def _filter_data_for_test_mode(
        self,
        features_df: pd.DataFrame,
        targets_df: pd.DataFrame,
        test_ticker: Optional[str],
        test_target: Optional[str]
    ) -> Tuple[pd.DataFrame, pd.DataFrame, List[str], Optional[str]]:
        """Filter data for test mode. Delegates to TestModeManager."""
        return self.test_mode_manager._filter_data_for_test_mode(
            features_df, targets_df, test_ticker, test_target
        )
    
    async def run_light_models(
        self,
        tickers: List[str],
        test_ticker: Optional[str] = None,
        test_target: Optional[str] = None,
        batch_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """Run light models with specified parameters."""
        self.logger.info(f"🤖 Running light models for tickers: {tickers}")
        
        # Load data for the batch
        # Note: self.config.output_dir already includes batch_name!
        # If batch_name parameter is provided, we need to rebuild the path
        if batch_name and batch_name != self.batch_name:
            # Rebuild config for different batch
            batch_config = self.orchestrator_config_manager.build_pipeline_config(batch_name)
            batch_dir = batch_config.output_dir
        else:
            # Use existing config
            batch_dir = self.config.output_dir
        
        features_path = batch_dir / FEATURES_PARQUET
        targets_path = batch_dir / TARGETS_PARQUET
        
        features_df = None
        targets_df = None
        
        if features_path.exists() and targets_path.exists():
            features_df = pd.read_parquet(features_path)
            targets_df = pd.read_parquet(targets_path)
            self.logger.info(f"✅ Loaded data: features={features_df.shape}, targets={targets_df.shape}")
        else:
            self.logger.error("❌ No features/targets files found")
            return {}
        
        # Filter data if test parameters are provided
        if test_ticker or test_target:
            features_df, targets_df, _, _ = self.test_mode_manager._filter_data_for_test_mode(
                features_df, targets_df, test_ticker, test_target
            )
            self.logger.info(f"🔍 Filtered data for test mode: {features_df.shape}")
        
        # Use light_models_trainer with loaded data
        return await self.light_models_trainer.run_light_models(
            features_df=features_df,
            targets_df=targets_df,
            tickers=tickers
        )
    
    async def run_light_models_with_selected_features(
        self,
        features_df: pd.DataFrame,
        targets_df: pd.DataFrame,
        batch_name: str
    ) -> Dict[str, Any]:
        """Train light models locally using Colab-selected features."""
        from src.training.light_model_trainer import LightModelTrainer

        self.logger.info("Starting light model training with selected features...")
        # If batch_name is different from self.batch_name, rebuild config
        if batch_name != self.batch_name:
            batch_config = self.orchestrator_config_manager.build_pipeline_config(batch_name)
            batch_dir = batch_config.output_dir
        else:
            batch_dir = self.config.output_dir
        
        # Load test mode configuration
        test_ticker, test_target, test_model = await self.test_mode_manager._load_test_mode_config(batch_dir)
        
        # Filter data for test mode
        features_df, targets_df, target_cols, ticker_col = self._filter_data_for_test_mode(
            features_df, targets_df, test_ticker, test_target
        )
        
        # Determine models to train
        light_models_to_train = [test_model] if test_model and test_model in self.config.light_models else self.config.light_models
        
        # Load selected features files
        load_request = FeatureLoadRequest(
            batch_dir=batch_dir,
            test_ticker=test_ticker,
            test_target=test_target,
            test_model=test_model,
            light_models_to_train=light_models_to_train,
            target_cols=target_cols,
            features_df=features_df
        )
        selected_feature_contexts = await self.selected_features_processor._load_selected_features_files(load_request)
        
        if not selected_feature_contexts:
            return {'status': 'error', 'message': 'No selected_features files found.', 'models_trained': 0}

        # Train models for all contexts
        light_trainer = LightModelTrainer()
        context = ModelTrainingContext(
            features_df=features_df,
            targets_df=targets_df,
            ticker_col=ticker_col,
            batch_dir=batch_dir,
            light_trainer=light_trainer
        )
        models_metadata, models_trained = self._train_models_for_contexts(
            selected_feature_contexts, context
        )

        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        light_results_path = batch_dir / f"light_models_results_{timestamp}.json"
        async with aiofiles.open(light_results_path, 'w', encoding='utf-8') as f:
            content = json.dumps({
                'timestamp': timestamp, 
                'batch_name': batch_name, 
                'models_trained': models_trained, 
                'models_metadata': models_metadata, 
                'status': 'success'
            }, indent=2, default=str)
            await f.write(content)

        return {
            'status': 'success', 
            'models_trained': models_trained, 
            'models_metadata': models_metadata, 
            'saved_path': str(light_results_path), 
            'timestamp': timestamp
        }


    def _train_models_for_contexts(self, selected_feature_contexts: Dict[str, Dict[str, Any]],
                                   context: ModelTrainingContext) -> Tuple[Dict[str, Any], int]:
        """Train models for all contexts. Delegates to ModelTrainingOrchestrator."""
        return self.model_training_orchestrator.train_models_for_contexts(
            selected_feature_contexts, context.features_df, context.targets_df, 
            context.ticker_col, context.batch_dir, context.light_trainer
        )

    def check_if_feature_selection_needed(self, batch_dir: Path, new_rows_count: int, force: bool = False) -> Dict[str, Any]:
        """Check if new feature selection is required. Delegates to FeatureSelectionValidator."""
        return self.interface.check_if_feature_selection_needed(batch_dir, new_rows_count, force)

    def _check_feature_selection(self, batch_dir: Path, features_df: pd.DataFrame, 
                                check_feature_selection: bool, force_feature_selection: bool) -> Optional[Dict[str, Any]]:
        """Check if feature selection is needed and log warning if so."""
        fs_check = None
        if check_feature_selection:
            fs_check = self.check_if_feature_selection_needed(batch_dir, len(features_df), force_feature_selection)
            if fs_check and fs_check['needed']:
                self.logger.warning(f"🔄 NEW FEATURE SELECTION NEEDED: {fs_check['reason']}")
        return fs_check

    async def prepare_colab_batch(self, request: ColabBatchRequest) -> Dict[str, Any]:
        """Prepares a data package for Colab training."""
        return await self.interface.prepare_colab_data(
            request.features_df, request.targets_df, request.tickers, request.timeframes, 
            request.batch_name or None, 
            str(request.accumulate) if request.accumulate else None,
            str(request.check_feature_selection) if request.check_feature_selection else None
        )

    async def prepare_colab_data(self, tickers: List[str], timeframes: List[str], **kwargs) -> Dict[str, Any]:
        """
        Prepare data for Colab training.
        
        Workflow:
        1. Run local pipeline (stages 0-3) to compute features and targets
        2. Prepare the computed data for Colab training
        3. Package for manual transfer to Colab
        
        Args:
            tickers: List of ticker symbols
            timeframes: List of timeframes
            **kwargs: Additional parameters including test_ticker, test_target, test_model, epochs, max_iterations
            
        Returns:
            Dict with preparation results including batch directory and metadata
        """
        logger.info("📦 Preparing data for Colab training...")
        logger.info(f"🔍 DEBUG: prepare_colab_data received tickers: {tickers}")
        logger.info("   STEP 1: Running local pipeline (stages 0-3) to compute features/targets...")
        
        # Step 1: Run local pipeline to compute features and targets
        logger.info(f"🔍 DEBUG: About to call run_local_pipeline with tickers: {tickers}, timeframes: {timeframes}")
        local_res = await self.run_local_pipeline(tickers, timeframes)
        logger.info(f"🔍 DEBUG: run_local_pipeline returned: {type(local_res)}")
        
        if not local_res:
            logger.error("❌ Local pipeline failed to compute features/targets")
            return {'status': 'error', 'message': 'Local pipeline computation failed'}
        
        # Step 2: Extract features and targets
        try:
            features_df, targets_df = self._extract_features_and_targets(local_res)
        except Exception as e:
            logger.error(f"❌ Failed to extract features and targets: {e}")
            return {'status': 'error', 'message': str(e)}
        
        # Step 2.5: Enrich with context map (CRITICAL for Colab)
        logger.info("   STEP 2.5: Computing context map for Colab...")
        features_df = self._enrich_with_context_map(features_df)
        
        # Step 3: Prepare computed data for Colab
        logger.info("   STEP 3: Preparing data package for Colab...")
        logger.info(f"   DEBUG: About to call prepare_colab_data with features={features_df.shape}, targets={targets_df.shape}")
        result = await self.interface.prepare_colab_data(
            features_df=features_df,
            targets_df=targets_df,
            tickers=tickers,
            timeframes=timeframes,
            test_ticker=kwargs.get('test_ticker'),
            test_target=kwargs.get('test_target'),
            test_model=kwargs.get('test_model'),
            epochs=kwargs.get('epochs'),
            max_iterations=kwargs.get('max_iterations'),
            batch_name=self.batch_name
        )
        logger.info(f"   DEBUG: prepare_colab_data returned: {result}")
        return result
    
    def _enrich_with_context_map(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """
        Enrich features with context map for Colab training.
        
        Args:
            features_df: Features DataFrame
            
        Returns:
            Enriched DataFrame with context_fingerprint column
        """
        try:
            from src.features.enrichers.context_map_enricher import ContextMapEnricher
            
            # Get context map configuration
            enrichers_config = self.config_manager.get('enrichment', {})
            context_config = enrichers_config.get('context_map', {})
            
            # Create and run enricher
            context_enricher = ContextMapEnricher(context_config)
            enriched_df = context_enricher.enrich(features_df)
            
            # Validate context fingerprint was added
            if 'context_fingerprint' in enriched_df.columns:
                unique_contexts = enriched_df['context_fingerprint'].nunique()
                logger.info(f"   ✅ Context map computed: {unique_contexts} unique contexts")
            else:
                logger.warning("   ⚠️ Context fingerprint not added by enricher")
                
            return enriched_df
            
        except Exception as e:
            logger.error(f"   ❌ Failed to compute context map: {e}")
            logger.warning("   🔧 Continuing without context map")
            return features_df

    def _extract_features_from_outputs(self, stage_outputs: Dict[str, Any]) -> pd.DataFrame:
        """Extract features from stage outputs"""
        features_df = (
            stage_outputs.get('features_df') or
            stage_outputs.get('features') or
            stage_outputs.get('enriched_data') or
            stage_outputs.get('enriched_prices') or
            pd.DataFrame()
        )
        return features_df
    
    def _extract_targets_from_outputs(self, stage_outputs: Dict[str, Any]) -> pd.DataFrame:
        """Extract targets from stage outputs"""
        targets_df = (
            stage_outputs.get('targets_df') or
            stage_outputs.get('targets') or
            stage_outputs.get('target_data') or
            stage_outputs.get('enriched_prices') or
            pd.DataFrame()
        )
        return targets_df
    
    def _extract_features_and_targets(self, local_result: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Extract features and targets from local pipeline result"""
        stage_outputs = local_result.get('results', {})
        
        # Safe extraction with DataFrame validation
        try:
            # ✅ FIX: Stage 3 returns enriched_prices and all_targets dicts, need to extract and combine
            # First try direct keys (for backward compatibility)
            features_df = stage_outputs.get('features_df')
            targets_df = stage_outputs.get('targets_df')
            
            # If not found, try to extract from enriched_prices and all_targets
            if features_df is None or (hasattr(features_df, 'empty') and features_df.empty):
                logger.info("   🔍 Extracting features from enriched_prices...")
                enriched_prices = stage_outputs.get('enriched_prices', {})
                
                if enriched_prices:
                    # Combine all timeframes
                    features_dfs = []
                    for tf, df in enriched_prices.items():
                        if isinstance(df, pd.DataFrame) and not df.empty:
                            # Ensure datetime column exists
                            if 'datetime' not in df.columns:
                                if isinstance(df.index, pd.DatetimeIndex):
                                    df = df.reset_index()
                                    if 'index' in df.columns:
                                        df = df.rename(columns={'index': 'datetime'})
                            features_dfs.append(df)
                    
                    features_df = pd.concat(features_dfs, ignore_index=True) if features_dfs else pd.DataFrame()
                    logger.info(f"   ✅ Extracted features from enriched_prices: {features_df.shape}")
                else:
                    # Try combined_features as fallback
                    features_df = stage_outputs.get('combined_features', pd.DataFrame())
                    logger.info(f"   ✅ Using combined_features: {features_df.shape}")
            
            # Extract targets
            if targets_df is None or (hasattr(targets_df, 'empty') and targets_df.empty):
                logger.info("   🔍 Extracting targets from all_targets...")
                all_targets = stage_outputs.get('all_targets', {})
                
                if all_targets:
                    # Combine all timeframes
                    targets_dfs = []
                    for tf, df in all_targets.items():
                        if isinstance(df, pd.DataFrame) and not df.empty:
                            # Ensure datetime column exists
                            if 'datetime' not in df.columns:
                                if isinstance(df.index, pd.DatetimeIndex):
                                    df = df.reset_index()
                                    if 'index' in df.columns:
                                        df = df.rename(columns={'index': 'datetime'})
                            targets_dfs.append(df)
                    
                    targets_df = pd.concat(targets_dfs, ignore_index=True) if targets_dfs else pd.DataFrame()
                    logger.info(f"   ✅ Extracted targets from all_targets: {targets_df.shape}")
                else:
                    targets_df = pd.DataFrame()
                    logger.warning("   ⚠️ No targets found in stage outputs")
            
            # Validate DataFrames before logging
            features_shape = getattr(features_df, 'shape', (0, 0)) if hasattr(features_df, 'shape') else (0, 0)
            targets_shape = getattr(targets_df, 'shape', (0, 0)) if hasattr(targets_df, 'shape') else (0, 0)
            
            logger.info(f"   ✅ Final computed features: {features_shape}")
            logger.info(f"   ✅ Final computed targets: {targets_shape}")
            
            # Validate we have data
            if features_shape[0] == 0 or targets_shape[0] == 0:
                logger.error("   ❌ Extracted empty features or targets")
                logger.error(f"   Available stage_outputs keys: {list(stage_outputs.keys())}")
                return pd.DataFrame(), pd.DataFrame()
            
            return features_df, targets_df
            
        except Exception as e:
            logger.error(f"❌ Error extracting features/targets: {e}")
            logger.error(f"   Stage outputs keys: {list(stage_outputs.keys())}")
            import traceback
            logger.error(f"   Traceback: {traceback.format_exc()}")
            return pd.DataFrame(), pd.DataFrame()
    
    def _combine_timeframe_features(self, features_df: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Combine features from multiple timeframes, excluding target columns"""
        all_features = []
        for tf, df in features_df.items():
            if isinstance(df, pd.DataFrame) and not df.empty:
                # ✅ Виключаємо колонки таргетів (починаються з 'target_')
                feature_cols = [col for col in df.columns if not col.startswith('target_')]
                df_features_only = df[feature_cols].copy()
                all_features.append(df_features_only)
        
        if all_features:
            combined = pd.concat(all_features, ignore_index=True)
            logger.info(f"   ✅ Combined features: {combined.shape} (excluded target columns)")
            return combined
        return pd.DataFrame()
    
    def _load_fallback_data(self) -> Dict[str, Any]:
        """Load fallback data when extraction fails"""
        logger.warning("   🔧 Using fallback data generation")
        
        from src.pipeline.stages.stage_0_data_generation import DataGenerator
        generator = DataGenerator(self.config_manager)
        
        features_df = generator.generate_synthetic_features()
        targets_df = generator.generate_synthetic_targets()
        
        return {
            'status': 'success',
            'features_df': features_df,
            'targets_df': targets_df,
            'message': 'Generated fallback data'
        }
    
    def _load_fallback_data_from_files(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load features and targets from saved files as fallback.
        
        Returns:
            Tuple of (features_df, targets_df)
        """
        logger.warning("⚠️ Could not extract features/targets from stage outputs")
        
        # Try to load from saved files
        # self.config.output_dir already includes batch_name
        batch_dir = self.config.output_dir
        features_path = batch_dir / FEATURES_FILE
        targets_path = batch_dir / TARGETS_FILE
        
        if features_path.exists() and targets_path.exists():
            try:
                features_df = pd.read_parquet(features_path)
                targets_df = pd.read_parquet(targets_path)
                logger.info(f"   ✅ Loaded features from saved file: {features_df.shape}")
                logger.info(f"   ✅ Loaded targets from saved file: {targets_df.shape}")
                return features_df, targets_df
            except Exception as e:
                logger.error(f"❌ Failed to load saved features/targets: {e}")
                raise ValueError('Could not load computed data')
        
        logger.error("❌ Local pipeline produced empty features or targets")
        raise ValueError('Empty data from local pipeline')

    def _load_features_data(self) -> pd.DataFrame:
        """Load features data from parquet file."""
        return self.data_manager.load_features_data()
    
    def _load_targets_data(self) -> pd.DataFrame:
        """Load targets data from parquet file."""
        return self.data_manager.load_targets_data()

    def load_colab_results(self, batch_name: str) -> Dict[str, Any]:
        """Loads training results from Colab. Delegates to ResultsProcessor."""
        return self.results_processor.load_colab_results(batch_name, self.config.output_dir)

    async def run_full_hybrid_pipeline(self, request: HybridPipelineRequest) -> Dict[str, Any]:
        """Full hybrid pipeline with smart caching logic."""
        return await self.interface.run_full_hybrid_pipeline(
            request.tickers, request.timeframes, request.accumulate, request.force_training, 
            request.skip_colab, request.force_feature_selection
        )
    
    async def _collect_local_data(self, tickers: Optional[List[str]], 
                                  timeframes: Optional[List[str]]) -> Dict[str, Any]:
        """Collect local pipeline data."""
        self.logger.info("📊 STEP 1: Collecting new data...")
        results = await self.run_local_pipeline(tickers, timeframes)
        return results
    
    def _handle_data_caching(self, local_res: Dict[str, Any], force_training: bool) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
        """Handle data caching logic. Delegates to DataCacheManager."""
        return self.data_cache_manager.handle_data_caching(
            local_res, force_training, self.batch_name, self.config.output_dir
        )
    def _build_models_metadata(self, colab_results: Dict[str, Any],
                              light_results: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Build models metadata from colab and light results. Delegates to ResultsProcessor."""
        return self.results_processor.build_models_metadata(colab_results, light_results)

    async def run_final_stages(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Runs final stages 4-7 after Colab results are loaded."""
        return await self.interface.run_final_stages(
            request.get('features_df'), request.get('targets_df'), request.get('colab_results'), 
            request.get('light_results'), request.get('tickers'), request.get('timeframes'), request.get('batch_name')
        )

    def _save_dataframe(self, df: pd.DataFrame, path) -> None:
        """Saves DataFrame to parquet. Delegates to DataProcessor."""
        self.data_processor.save_dataframe(df, path)

    async def run_calibration(
        self,
        test_ticker: Optional[str] = None,
        test_target: Optional[str] = None,
        n_trials: int = 50
    ) -> Dict[str, Any]:
        """
        Run DEAN hyperparameter calibration.

        Args:
            test_ticker: Optional ticker for filtering
            test_target: Optional target for filtering
            n_trials: Number of Optuna trials

        Returns:
            Dict with calibration results
        """
        from src.calibration import CalibrationEngine

        logger.info("🎯 Starting DEAN calibration...")

        # Initialize calibration engine
        engine = CalibrationEngine(
            real_data_path="data/duckdb/trading.db",
            synthetic_data_path="data/synthetic/",
            n_trials=n_trials,
            metric="sharpe_ratio",
            batch_name=self.batch_name
        )

        # Run calibration
        results = engine.run_calibration(
            test_ticker=test_ticker,
            test_target=test_target
        )

        logger.info(f"✅ Calibration completed: {results.get('status')}")
        return cast(Dict[str, Any], results)
