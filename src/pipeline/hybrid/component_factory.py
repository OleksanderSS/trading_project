from typing import Any

from src.core.exceptions import DataProcessingError
from src.core.logging.logger import ProjectLogger
from src.pipeline.stages.stage_0_data_generation import DataGenerator

from .colab_manager import ColabManager
from .data_cache_manager import DataCacheManager

# Import all specialized components
from .data_manager import HybridDataManager
from .data_utils import DataUtils
from .feature_processor import FeatureProcessor
from .final_stages_orchestrator import FinalStagesOrchestrator
from .light_models_trainer import LightModelsTrainer
from .metadata_manager import MetadataManager
from .pipeline_manager import PipelineManager
from .pipeline_runner import PipelineRunner
from .results_processor import ResultsProcessor
from .storage_manager import StorageManager

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
            from src.data.management.data_manager import DataManager
            components['data_manager'] = HybridDataManager(output_dir)
            components['db_data_manager'] = DataManager(config_manager)
            components['feature_processor'] = FeatureProcessor()
            components['colab_manager'] = ColabManager(output_dir, batch_name)
            components['pipeline_manager'] = PipelineManager(orchestrator)
            components['storage_manager'] = StorageManager(config)
            components['data_utils'] = DataUtils()
            components['metadata_manager'] = MetadataManager(config_manager)
            components['pipeline_runner'] = PipelineRunner(config_manager, str(output_dir), batch_name, components['feature_processor'], components['metadata_manager'])

            components['light_models_trainer'] = LightModelsTrainer({
                'config_manager': config_manager,
                'output_dir': output_dir,
                'batch_name': batch_name,
                'light_models': config.light_models,
                'models_config': config.models_config,
                'data_manager': components['db_data_manager']
            })

            components['results_processor'] = ResultsProcessor()
            components['data_cache_manager'] = DataCacheManager()

            # NOTE: 13 further components used to be constructed here and
            # attached to the orchestrator via setattr, but nothing ever called
            # them -- HybridOrchestrator's public API only touches
            # pipeline_runner, pipeline_manager, colab_manager and
            # light_models_trainer. They are archived under
            # src/archive/pipeline_hybrid_dormant/ (see MANIFEST). Each
            # duplicated a responsibility a live component already handles;
            # pipeline_executor's stage methods were literally
            # "# Implementation would go here" stubs superseded by
            # pipeline_runner. data_manager.py and data_utils.py stayed put --
            # they have real behavioural test coverage.
            components['final_stages_orchestrator'] = FinalStagesOrchestrator(config_manager, output_dir, batch_name)
            components['data_generator'] = DataGenerator(config_manager)

            # Apply components to orchestrator
            for name, comp in components.items():
                setattr(orchestrator, name, comp)

            components['storage_manager'].initialize_storage()

            logger.info("✅ All orchestrator components initialized via factory")
            return components

        except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError) as e:
            logger.exception(f"Failed to initialize orchestrator components: {e}")
            raise DataProcessingError(f"Component initialization failed: {e}") from e
