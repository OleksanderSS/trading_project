from functools import cached_property
from pathlib import Path
from typing import Any


class OrchestratorContext:
    """
    Container for orchestrator dependencies with lazy initialization to reduce coupling.

    Components are initialized on-demand to avoid unnecessary imports and reduce
    the dependency graph. Only components that are actually used are instantiated.
    """

    def __init__(self, config_manager: Any):
        self.config_manager = config_manager
        self._output_dir: Path | None = None
        self._models_dir: Path | None = None
        self._batch_name = "main_database"
        self._pipeline_config = None

    @property
    def output_dir(self) -> Path:
        """Lazy initialization of output directory path."""
        if self._output_dir is None:
            self._output_dir = Path(self.config_manager.get("paths.output", "data/output"))
        return self._output_dir

    @property
    def models_dir(self) -> Path:
        """Lazy initialization of models directory path."""
        if self._models_dir is None:
            self._models_dir = Path(self.config_manager.get("paths.models", "trained_models"))
        return self._models_dir

    @property
    def batch_name(self) -> str:
        """Batch name for this orchestrator context."""
        return self._batch_name

    @cached_property
    def orchestrator_config(self):
        """Lazy initialization of orchestrator config manager."""
        from src.pipeline.hybrid.orchestrator_config import OrchestratorConfigManager
        return OrchestratorConfigManager(self.config_manager)

    @property
    def pipeline_config(self):
        """Lazy initialization of pipeline configuration."""
        if self._pipeline_config is None:
            self._pipeline_config = self.orchestrator_config.build_pipeline_config(self.batch_name)
        return self._pipeline_config

    @cached_property
    def data_components(self):
        """Lazy initialization of data components context."""
        from src.pipeline.hybrid.data_components_context import DataComponentsContext
        return DataComponentsContext(self.config_manager, self.output_dir, self.models_dir)

    @cached_property
    def feature_processor(self):
        """Lazy initialization of feature processor."""
        from src.pipeline.hybrid.feature_processor import FeatureProcessor
        return FeatureProcessor()

    @cached_property
    def cache_manager(self):
        """Lazy initialization of cache manager."""
        from src.pipeline.hybrid.cache_manager import CacheManager
        return CacheManager(output_dir=self.output_dir, batch_name=self.batch_name)

    @cached_property
    def colab_manager(self):
        """Lazy initialization of Colab manager."""
        from src.pipeline.hybrid.colab_manager import ColabManager
        return ColabManager(output_dir=self.output_dir, batch_name=self.batch_name)

    @cached_property
    def pipeline_executor(self):
        """Lazy initialization of pipeline executor."""
        from src.pipeline.hybrid.pipeline_executor import PipelineExecutor
        return PipelineExecutor(
            config=self.pipeline_config,
            data_manager=self.data_components.data_manager,
            feature_processor=self.feature_processor
        )

    @cached_property
    def feature_selection_manager(self):
        """Lazy initialization of feature selection manager."""
        from src.pipeline.hybrid.feature_selection_manager import FeatureSelectionManager
        return FeatureSelectionManager(config=self.pipeline_config)

    @cached_property
    def metadata_manager(self):
        """Lazy initialization of metadata manager."""
        from src.pipeline.hybrid.metadata_manager import MetadataManager
        return MetadataManager(config=self.pipeline_config)

    @cached_property
    def light_models_trainer(self):
        """Lazy initialization of light models trainer."""
        from src.pipeline.hybrid.light_models_trainer import LightModelsTrainer
        trainer_config = {
            "config_manager": self.config_manager,
            "output_dir": self.output_dir,
            "batch_name": self.batch_name,
            "light_models": self.pipeline_config.light_models,
            "models_config": self.config_manager.get("models", {})
        }
        return LightModelsTrainer(trainer_config=trainer_config)

    @cached_property
    def colab_workflow_manager(self):
        """Lazy initialization of Colab workflow manager."""
        from src.pipeline.hybrid.colab_workflow_manager import ColabWorkflowManager
        return ColabWorkflowManager(
            output_dir=self.output_dir,
            batch_name=self.batch_name,
            light_models=self.pipeline_config.light_models
        )

    @cached_property
    def final_stages_executor(self):
        """Lazy initialization of final stages executor."""
        from src.pipeline.hybrid.final_stages_executor import FinalStagesExecutor
        return FinalStagesExecutor(
            config_manager=self.config_manager,
            output_dir=str(self.output_dir),
            batch_name=self.batch_name
        )

    @cached_property
    def data_batch_manager(self):
        """Lazy initialization of data batch manager."""
        from src.pipeline.hybrid.data_batch_manager import DataBatchManager
        return DataBatchManager()

    @cached_property
    def model_training_orchestrator(self):
        """Lazy initialization of model training orchestrator."""
        from src.pipeline.hybrid.model_training_orchestrator import ModelTrainingOrchestrator
        return ModelTrainingOrchestrator()

    @cached_property
    def test_mode_manager(self):
        """Lazy initialization of test mode manager."""
        from src.pipeline.hybrid.test_mode_manager import TestModeManager
        return TestModeManager()

    @cached_property
    def context_builder(self):
        """Lazy initialization of context builder."""
        from src.pipeline.hybrid.context_builder import ContextBuilder
        return ContextBuilder(test_mode_manager=self.test_mode_manager)

    @cached_property
    def feature_selection_validator(self):
        """Lazy initialization of feature selection validator."""
        from src.pipeline.hybrid.feature_selection_validator import FeatureSelectionValidator
        return FeatureSelectionValidator()

    @cached_property
    def selected_features_processor(self):
        """Lazy initialization of selected features processor."""
        from src.pipeline.hybrid.selected_features_processor import SelectedFeaturesProcessor
        return SelectedFeaturesProcessor(
            context_builder=self.context_builder,
            feature_selection_validator=self.feature_selection_validator
        )

    @cached_property
    def results_processor(self):
        """Lazy initialization of results processor."""
        from src.pipeline.hybrid.results_processor import ResultsProcessor
        return ResultsProcessor()

    @cached_property
    def pipeline_runner(self):
        """Lazy initialization of pipeline runner."""
        from src.pipeline.hybrid.pipeline_runner import PipelineRunner
        return PipelineRunner(
            config_manager=self.config_manager,
            output_dir=str(self.output_dir),
            batch_name=self.batch_name,
            feature_processor=self.feature_processor,
            metadata_manager=self.metadata_manager
        )

    @cached_property
    def pipeline_metadata_manager(self):
        """Lazy initialization of pipeline metadata manager."""
        from src.pipeline.hybrid.pipeline_metadata_manager import PipelineMetadataManager
        return PipelineMetadataManager(
            output_dir=self.output_dir,
            batch_name=self.batch_name,
            light_models=self.pipeline_config.light_models,
            heavy_models=self.pipeline_config.heavy_models
        )

    @cached_property
    def final_stages_orchestrator(self):
        """Lazy initialization of final stages orchestrator."""
        from src.pipeline.hybrid.final_stages_orchestrator import FinalStagesOrchestrator
        return FinalStagesOrchestrator(
            config_manager=self.config_manager,
            output_dir=self.output_dir,
            batch_name=self.batch_name
        )

    # Pipeline manager is set externally due to circular dependency
    def set_pipeline_manager(self, pipeline_manager):
        """Set pipeline manager (called after HybridOrchestrator initialization)."""
        self._pipeline_manager = pipeline_manager

    @property
    def pipeline_manager(self):
        """Get pipeline manager (must be set externally)."""
        return getattr(self, '_pipeline_manager', None)
