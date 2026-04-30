"""
Argument validation utilities for hybrid pipeline.
"""

from typing import Dict, Any, List
from src.core.logging.logger import ProjectLogger

logger = ProjectLogger.get_logger(__name__)


class ArgumentValidator:
    """Validates command line arguments for the hybrid pipeline."""
    
    @staticmethod
    def validate_arguments(args, config_manager):
        """
        Validates command line arguments.

        Checks:
        - Tickers exist in config
        - Targets exist in config
        - Models exist in config
        - Execution mode and other parameters are valid
        """
        errors, warnings = [], []

        # Get configuration data
        config_data = ArgumentValidator._get_config_data(config_manager)

        # Validate individual parameters
        ArgumentValidator._validate_test_ticker(args, config_data, errors)
        ArgumentValidator._validate_test_target(args, config_data, errors)
        ArgumentValidator._validate_test_model(args, config_data, errors)
        ArgumentValidator._validate_mode(args, errors)
        ArgumentValidator._validate_numeric_params(args, errors)
        ArgumentValidator._validate_stages(args, errors, warnings)

        # Report results
        ArgumentValidator._report_validation_results(errors, warnings)

    @staticmethod
    def _get_config_data(config_manager) -> Dict[str, Any]:
        """Gets configuration data."""
        assets_config = config_manager.get_config('assets') or {}
        targets_config = config_manager.get_config('targets') or {}
        models_config = config_manager.get_config('models') or {}

        # Load all unique tickers from all sectors
        sectors = assets_config.get('sectors', {})
        all_tickers = set()
        for sector_name, sector_config in sectors.items():
            sector_assets = sector_config.get('assets', [])
            all_tickers.update(sector_assets)
        
        available_tickers = sorted(all_tickers)

        available_targets = list(targets_config.keys())
        available_models = (
            list(models_config.get('model_definitions', {}).keys()) or
            list(models_config.get('available', {}).keys())
        )

        return {
            'available_tickers': available_tickers,
            'available_targets': available_targets,
            'available_models': available_models
        }

    @staticmethod
    def _validate_test_ticker(args: Any, config_data: Dict[str, Any], errors: List[str]) -> None:
        """Validates test ticker."""
        if args.test_ticker and args.test_ticker not in config_data['available_tickers']:
            errors.append(
                "❌ Ticker '{}' not found. Available: {}".format(
                    args.test_ticker, ', '.join(config_data['available_tickers'])
                )
            )

    @staticmethod
    def _validate_test_target(args: Any, config_data: Dict[str, Any], errors: List[str]) -> None:
        """Validates test target."""
        if args.test_target:
            target_name = args.test_target
            if not target_name.startswith('target_'):
                target_name = "target_{}".format(target_name)

            if target_name not in config_data['available_targets']:
                errors.append(
                    "❌ Target '{}' not found. Available: {}... (total {})".format(
                        args.test_target,
                        ', '.join(config_data['available_targets'][:5]),
                        len(config_data['available_targets'])
                    )
                )

    @staticmethod
    def _validate_test_model(args: Any, config_data: Dict[str, Any], errors: List[str]) -> None:
        """Validates test model."""
        if args.test_model and args.test_model not in config_data['available_models']:
            errors.append(
                "❌ Model '{}' not found. Available: {}... (total {})".format(
                    args.test_model,
                    ', '.join(config_data['available_models'][:5]),
                    len(config_data['available_models'])
                )
            )

    @staticmethod
    def _validate_mode(args: Any, errors: List[str]) -> None:
        """Validates execution mode."""
        valid_modes = ['local', 'full', 'prepare', 'light', 'continue']
        if args.mode not in valid_modes:
            errors.append(
                "❌ Invalid mode '{}'. Available modes: {}".format(
                    args.mode, ', '.join(valid_modes)
                )
            )

    @staticmethod
    def _validate_numeric_params(args: Any, errors: List[str]) -> None:
        """Validates numeric parameters."""
        if args.max_iterations < 1:
            errors.append(
                "❌ max_iterations must be >= 1, got: {}".format(args.max_iterations)
            )

    @staticmethod
    def _validate_stages(args: Any, errors: List[str], warnings: List[str]) -> None:
        """Validates stage parameters."""
        if args.stages:
            for stage in args.stages:
                if not ArgumentValidator._is_valid_stage(stage):
                    errors.append(
                        "❌ Invalid stage number: {}. Valid range: 4-7".format(stage)
                    )
            
            if len(args.stages) > 1:
                warnings.append(
                    "⚠️ Multiple stages specified. Will run stages: {}".format(
                        ', '.join(map(str, sorted(args.stages)))
                    )
                )

    @staticmethod
    def _is_valid_stage(stage: int) -> bool:
        """
        Check if stage number is valid
        
        Args:
            stage (int): Stage number to validate
            
        Returns:
            bool: True if stage is valid, False otherwise
        """
        return 4 <= stage <= 7

    @staticmethod
    def _report_validation_results(errors: List[str], warnings: List[str]) -> None:
        """
        Reports validation results using structured logging.
        
        Args:
            errors (List[str]): List of validation error messages.
            warnings (List[str]): List of validation warning messages.
            
        Returns:
            None
        """
        if errors:
            logger.error("❌ VALIDATION ERRORS (%d found):", len(errors))
            for error in errors:
                logger.error("   %s", error)
            logger.error("💡 Use --force to override validation errors")
        
        if warnings:
            logger.warning("⚠️ VALIDATION WARNINGS (%d found):", len(warnings))
            for warning in warnings:
                logger.warning("   %s", warning)
        
        if not errors and not warnings:
            logger.info("✅ Arguments validated successfully")
