# audit-ignore: ARCHITECTURAL_USAGE
"""
Orchestrator Configuration and Initialization Manager.
Handles all configuration building and initialization logic.
"""

from dataclasses import dataclass
from importlib.util import find_spec
from pathlib import Path

from src.config.unified_config_manager import UnifiedConfigManager
from src.core.logging.logger import ProjectLogger

logger = ProjectLogger.get_logger(__name__)

GDRIVE_AVAILABLE = all(
    find_spec(module_name) is not None
    for module_name in (
        "google.oauth2.credentials",
        "googleapiclient.discovery",
        "googleapiclient.http",
    )
)


@dataclass
class PipelineConfig:
    """Configuration for hybrid pipeline."""
    output_dir: Path
    models_dir: Path
    light_models: list
    heavy_models: list
    models_config: dict
    system_config: dict
    gdrive_enabled: bool
    gdrive_folder_id: str | None = None
    gdrive_service: Any = None
    storage_fallback: dict | None = None
    use_s3: bool = False
    use_gcs: bool = False


class OrchestratorConfigManager:
    """Manages configuration and initialization for Hybrid Orchestrator."""

    def __init__(self, config_manager: UnifiedConfigManager):
        self.config_manager = config_manager
        self.logger = ProjectLogger.get_logger(__name__)

    def build_pipeline_config(self, batch_name: str = "main_database") -> PipelineConfig:
        """Build complete pipeline configuration."""
        # Base output directory
        output_dir = Path(self.config_manager.get('system.accumulation.output_dir', 'data/colab/accumulated')) / batch_name
        output_dir.mkdir(parents=True, exist_ok=True)

        # Models directory
        system_config = self.config_manager.get_config('system') or {}
        models_dir = Path(system_config.get('models_path', 'trained_models'))
        models_dir.mkdir(parents=True, exist_ok=True)

        # Model configurations
        models_config = self.config_manager.get_config('models') or {}

        # Try to get models from different possible locations
        light_models = []
        heavy_models = []

        # Try dual_model_manager first
        dual_config = models_config.get('dual_model_manager', {})
        if dual_config:
            light_models = dual_config.get('light_models', [])
            heavy_models = dual_config.get('heavy_models', [])

        # Fallback to pipeline config
        if not light_models and not heavy_models:
            pipeline_config = models_config.get('pipeline', {})
            light_models = pipeline_config.get('light_models', [])
            heavy_models = pipeline_config.get('heavy_models', [])

        # Final fallback to categories
        if not light_models and not heavy_models:
            light_models = models_config.get('categories', {}).get('light', [])
            heavy_models = models_config.get('categories', {}).get('heavy', [])

        # Log what we found
        self.logger.info(f"🔍 Models config found: light_models={len(light_models)}, heavy_models={len(heavy_models)}")
        self.logger.info(f"💡 Light models: {light_models}")
        self.logger.info(f"🔥 Heavy models: {heavy_models}")

        # Google Drive configuration
        gdrive_enabled = GDRIVE_AVAILABLE and system_config.get('google_drive', {}).get('enabled', False)
        gdrive_folder_id = system_config.get('google_drive', {}).get('folder_id')

        # Fallback storage options
        storage_fallback = system_config.get('storage_fallback', {})

        # Storage configuration
        use_s3 = storage_fallback.get('use_s3', False)
        use_gcs = storage_fallback.get('use_gcs', False)

        return PipelineConfig(
            output_dir=output_dir,
            models_dir=models_dir,
            light_models=light_models,
            heavy_models=heavy_models,
            models_config=models_config,
            system_config=system_config,
            gdrive_enabled=gdrive_enabled,
            gdrive_folder_id=gdrive_folder_id,
            storage_fallback=storage_fallback,
            use_s3=use_s3,
            use_gcs=use_gcs
        )

    def resolve_target_task_type(self, target_name: str) -> str:
        """Maps configured targets to the trainer's regression/classification contract."""
        targets_config = self.config_manager.get_config('targets', {})
        if hasattr(targets_config, 'as_dict'):
            targets_config = targets_config.as_dict()

        configured_type = targets_config.get(target_name, {}).get('type', 'regression')
        return self._determine_task_type(configured_type, target_name)

    def _determine_task_type(self, configured_type: str, target_name: str) -> str:
        """Determine task type from configuration."""
        if configured_type in {'regression', 'indicator_prediction'}:
            return 'regression'
        elif configured_type in {'classification', 'binary_classification'}:
            return 'classification'
        else:
            # Default to regression for unknown types
            self.logger.warning(f"Unknown target type '{configured_type}' for {target_name}, defaulting to regression")
            return 'regression'
