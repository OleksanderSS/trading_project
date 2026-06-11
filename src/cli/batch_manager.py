"""
Batch management utilities for hybrid pipeline.
"""

from pathlib import Path

from src.core.logging.logger import ProjectLogger

logger = ProjectLogger.get_logger(__name__)


class BatchManager:
    """Manages batch naming and operations for the hybrid pipeline."""

    @staticmethod
    def generate_batch_name(args) -> str:
        """Generate batch name based on test parameters."""
        has_test_params = args.test_ticker or args.test_target or args.test_model

        if has_test_params:
            return BatchManager._generate_test_batch_name(args)
        else:
            return BatchManager._generate_full_batch_name(args)

    @staticmethod
    def _generate_test_batch_name(args) -> str:
        """Generate batch name for test mode."""
        parts = []

        if args.test_ticker:
            parts.append(f"ticker_{args.test_ticker}")
        if args.test_target:
            target_name = args.test_target
            if not target_name.startswith('target_'):
                target_name = f"target_{target_name}"
            parts.append(target_name)
        if args.test_model:
            parts.append(f"model_{args.test_model}")

        return BatchManager._handle_continue_mode(parts, args)

    @staticmethod
    def _generate_full_batch_name(args) -> str:
        """Generate batch name for full mode.

        Full mode (без test параметрів) завжди використовує 'main_database'.
        Це дозволяє:
        - Полегшеному режиму (test mode) створювати окремі підпапки
        - Повноцінному режиму (full mode) накопичувати дані в одну папку
        """
        return "main_database"

    @staticmethod
    def _handle_continue_mode(parts, args) -> str:
        """Handle batch name generation for continue mode."""
        base_pattern = "test_" + "_".join(parts) if parts else "manual_run"

        if args.mode == 'continue':
            # Find existing batch directory in the Colab accumulated data root
            output_dir = Path("data/colab/accumulated")
            if output_dir.exists():
                existing_batches = [
                    d.name for d in output_dir.iterdir()
                    if d.is_dir() and d.name.startswith(base_pattern)
                ]
                if existing_batches:
                    return max(existing_batches)  # Return latest

        return base_pattern

    @staticmethod
    def sanitize_path_input(path_input: str) -> str:
        """
        Sanitize path input to prevent path traversal attacks.

        Args:
            path_input: Input string that will be used in file paths

        Returns:
            Sanitized string safe for path construction
        """
        import re

        if not path_input:
            return ""

        # Remove path traversal characters
        sanitized = re.sub(r'[./\\]', '_', path_input)

        # Remove null bytes and other dangerous characters
        sanitized = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', sanitized)

        # Limit length to prevent path overflow
        sanitized = sanitized[:100]

        return sanitized
