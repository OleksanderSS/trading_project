"""
StageLoader

Encapsulates stage-loading logic extracted from PipelineOrchestrator.
"""
import logging
from typing import Any

from src.core.logging.logger import ProjectLogger
from src.utils.dynamic_module_loader import DynamicModuleLoader


class StageLoader:
    def __init__(self, config_manager, logger=None):
        self.config_manager = config_manager
        self.logger = logger or ProjectLogger.get_logger(__name__)

    def load_stages(self, stages_to_run: list[int] | None = None, dependencies: dict[str, Any] | None = None) -> list:
        """Load and instantiate pipeline stages according to configuration.

        Args:
            stages_to_run: Optional list of stage indices to limit loading.
            dependencies: Dependency map forwarded to stage constructors.

        Returns:
            List of instantiated stage objects.
        """
        stages_config = self.config_manager.get_config("training_pipeline", [])
        dependencies = dependencies or {}

        loaded_stages = []
        for i, stage_info in enumerate(stages_config):
            if not stage_info.get("enabled", False):
                self.logger.info(f"Stage '{stage_info.get('name')}' disabled. Skipping.")
                continue

            if stages_to_run is not None and i not in stages_to_run:
                if self.logger.isEnabledFor(logging.DEBUG):
                    self.logger.debug(f"Stage {i} ('{stage_info.get('name')}') not requested. Skipping.")
                continue

            try:
                # Support legacy config with 'module' + 'class' keys
                cfg = dict(stage_info)
                if "class_path" not in cfg:
                    module = cfg.get("module")
                    class_name = cfg.get("class")
                    if module and class_name:
                        cfg["class_path"] = f"{module}.{class_name}"

                stage_instance = DynamicModuleLoader.load_instance(cfg, **dependencies)
                # attach index for traceability
                stage_instance._pipeline_stage_index = i
                loaded_stages.append(stage_instance)
                self.logger.info(f"Stage '{stage_info.get('name')}' loaded (index={i}).")
            except (ImportError, AttributeError, TypeError, ValueError) as e:
                self.logger.error(f"Failed to load stage '{stage_info.get('name')}': {e}", exc_info=True)
                raise

        return loaded_stages
