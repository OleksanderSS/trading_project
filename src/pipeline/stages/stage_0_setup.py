"""
Stage 0: Environment Setup

Responsible for system initialization, directory creation, and verification
of environment readiness before the pipeline execution.
"""
import logging
import os
from typing import Any

from src.config.unified_config_manager import UnifiedConfigManager
from src.core.error_handling.error_handler import ErrorHandler
from src.core.logging.logger import ProjectLogger
from src.pipeline.stages.base_stage import BaseStage


class Stage0Setup(BaseStage):
    """
    Stage responsible for preparing the working environment by ensuring
    necessary infrastructure and configurations are in place.
    """

    def __init__(self, config_manager: UnifiedConfigManager, error_handler:
        ErrorHandler, **kwargs):
        super().__init__(config_manager, error_handler, **kwargs)
        self.logger = ProjectLogger.get_logger('Stage0Setup')

    async def run(self, **kwargs) ->dict[str, Any]:
        """
        Executes environment setup by creating required system directories.

        Uses paths defined in 'paths.yaml' to initialize folders for data,
        models, logs, and temporary artifacts.

        Args:
            **kwargs: Arbitrary keyword arguments (not used in this stage).

        Returns:
            An empty dictionary, as this stage does not produce output for subsequent stages.

        Raises:
            KeyError: If required keys are missing from the configuration.
            Exception: For general setup failures.
        """
        self.logger.info('Initializing environment setup...')
        try:
            paths_config = self.config_manager.get_config('paths')
            if not paths_config:
                self.logger.critical(
                    "Critical error: 'paths' configuration section is missing. Pipeline cannot proceed."
                    )
                raise KeyError("Missing 'paths' configuration section.")
            required_paths = {'data': paths_config.get('root'), 'models':
                paths_config.get('models'), 'logs': paths_config.get('logs'
                ), 'temp': paths_config.get('temp')}
            created_dirs = []
            for name, path in required_paths.items():
                if not path:
                    self.logger.warning(
                        f"Path for '{name}' is not defined in configuration.")
                    continue
                if not os.path.exists(path):
                    self.logger.info(f'Creating directory: {path}')
                    os.makedirs(path, exist_ok=True)
                    created_dirs.append(path)
                else:
                    if self.logger.isEnabledFor(logging.DEBUG):
                        self.logger.debug(f'Directory already exists: {path}')
            summary = ', '.join(created_dirs
                ) if created_dirs else 'none (all existed)'
            self.logger.info(
                f'Environment setup successfully completed. New directories: {summary}'
                )
        except KeyError as e:
            self.handle_stage_error(e, context='ConfigKey-paths', severity=
                'error', should_raise=True)
        except Exception as e:
            self.logger.error(f'Виникла помилка: {e}', exc_info=True)
            self.handle_stage_error(e, context='EnvironmentSetup', severity
                ='error', should_raise=True)
            raise
        return {}
