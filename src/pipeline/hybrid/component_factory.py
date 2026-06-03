from typing import Any, Dict, List, Optional
from pathlib import Path
from src.core.logging.logger import ProjectLogger
from src.config.unified_config_manager import UnifiedConfigManager
from src.core.exceptions import DataProcessingError

# Import all specialized components
from .data_manager import HybridDataManager
from .feature_processor import FeatureProcessor
from .cache_manager import CacheManager
from .colab_manager import ColabManager
from .pipeline_manager import PipelineManager
from .storage_manager import StorageManager
from .pipeline_executor import PipelineExecutor
from .feature_selection_manager import FeatureSelectionManager
from .data_utils import DataUtils
from .metadata_manager import MetadataManager
from .orchestrator_interface import OrchestratorInterface
from .pipeline_runner import PipelineRunner
from .light_models_trainer import LightModelsTrainer
from .colab_workflow_manager import ColabWorkflowManager
from .final_stages_executor import FinalStagesExecutor
from .data_processor import DataProcessor
from .data_batch_manager import DataBatchManager
from .model_training_orchestrator import ModelTrainingOrchestrator
from .feature_selection_validator import FeatureSelectionValidator
from .results_processor import ResultsProcessor
from .data_cache_manager import DataCacheManager
from .test_mode_manager import TestModeManager
from .context_builder import ContextBuilder
from .selected_features_processor import SelectedFeaturesProcessor
from .pipeline_metadata_manager import PipelineMetadataManager
from .final_stages_orchestrator import FinalStagesOrchestrator
from src.pipeline.stages.stage_0_data_generation import DataGenerator

logger = ProjectLogger.get_logger('OrchestratorFactory')

class OrchestratorComponentFactory:
    """Factory for initializing Hybrid Orchestrator components."""
    
    @staticmethod
    def initialize_components(orchestrator: Any):
        """Initializes all specialized components for the orchestrator."""
        config = orchestrator.config
        config_manager = orchestrator.config_manager
        batch_name = orchestrator.batch_name
        output_dir = config.output_dir
        
        components = {}
        
        try:
            components['data_manager'] = HybridDataManager(output_dir)
            components['feature_processor'] = FeatureProcessor()
            components['cache_manager'] = CacheManager(output_dir, batch_name)
            components['colab_manager'] = ColabManager(output_dir, batch_name)
            components['pipeline_manager'] = PipelineManager(orchestrator)
            components['storage_manager'] = StorageManager(config)
            components['pipeline_executor'] = PipelineExecutor(config_manager, output_dir, components['feature_processor'])
            components['feature_selection_manager'] = FeatureSelectionManager(output_dir)
            components['data_utils'] = DataUtils()
            components['metadata_manager'] = MetadataManager(config_manager)
            components['interface'] = OrchestratorInterface(orchestrator)
            components['pipeline_runner'] = PipelineRunner(config_manager, str(output_dir), batch_name, components['feature_processor'], components['metadata_manager'])
            
            components['light_models_trainer'] = LightModelsTrainer({
                'config_manager': config_manager,
                'output_dir': output_dir,
                'batch_name': batch_name,
                'light_models': config.light_models,
                'models_config': config.models_config
            })
            
            components['colab_workflow_manager'] = ColabWorkflowManager(output_dir, batch_name, config.light_models)
            components['final_stages_executor'] = FinalStagesExecutor(config_manager, str(output_dir), batch_name)
            components['data_processor'] = DataProcessor(components['data_utils'])
            components['data_batch_manager'] = DataBatchManager()
            components['model_training_orchestrator'] = ModelTrainingOrchestrator(config_manager)
            components['feature_selection_validator'] = FeatureSelectionValidator()
            components['results_processor'] = ResultsProcessor()
            components['data_cache_manager'] = DataCacheManager()
            components['test_mode_manager'] = TestModeManager()
            components['context_builder'] = ContextBuilder(components['test_mode_manager'])
            components['selected_features_processor'] = SelectedFeaturesProcessor(components['context_builder'], components['feature_selection_validator'])
            
            components['pipeline_metadata_manager'] = PipelineMetadataManager(
                output_dir, batch_name, config.light_models, config.heavy_models
            )
            
            components['final_stages_orchestrator'] = FinalStagesOrchestrator(config_manager, output_dir, batch_name)
            components['data_generator'] = DataGenerator(config_manager)
            
            # Apply components to orchestrator
            for name, comp in components.items():
                setattr(orchestrator, name, comp)
                
            components['storage_manager'].initialize_storage()
            
            logger.info("✅ All orchestrator components initialized via factory")
            return components
            
        except Exception as e:
            logger.error(f"Failed to initialize orchestrator components: {e}", exc_info=True)
            raise DataProcessingError(f"Component initialization failed: {e}") from e
