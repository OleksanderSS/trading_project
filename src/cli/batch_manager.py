"""
Batch management utilities for hybrid pipeline.
"""

import re
from pathlib import Path

from src.core.logging.logger import ProjectLogger
from src.utils.path_utils import sanitize_path_input

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
            # audit-ignore: ARCHITECTURAL_USAGE
            target_name = args.test_target
            # audit-ignore: ARCHITECTURAL_USAGE
            if not target_name.startswith('target_'):
                # audit-ignore: ARCHITECTURAL_USAGE
                target_name = f"target_{target_name}"
            # audit-ignore: ARCHITECTURAL_USAGE
            parts.append(target_name)
        if args.test_model:
            parts.append(f"model_{args.test_model}")

        return BatchManager._handle_continue_mode(parts, args)

    @staticmethod
    def _generate_full_batch_name(args) -> str:
        """Generate batch name for full mode."""
        return "main_database"

    @staticmethod
    def _handle_continue_mode(parts, args) -> str:
        """Handle batch name generation for continue mode."""
        base_pattern = "test_" + "_".join(parts) if parts else "manual_run"

        if args.mode == 'continue':
            output_dir = Path("outputs")
            if output_dir.exists():
                existing_batches = [
                    d.name for d in output_dir.iterdir()
                    if d.is_dir() and d.name.startswith(base_pattern)
                ]
                if existing_batches:
                    return max(existing_batches)

        return base_pattern

    @staticmethod
    def sanitize_path_input(path_input: str) -> str:
        """
        Sanitize path input to prevent path traversal attacks.
        Uses centralized path utility.
        """
        # Centralized sanitization
        sanitized = sanitize_path_input(path_input)

        # Remove null bytes and other dangerous characters
        sanitized = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', sanitized)

        # Limit length to prevent path overflow
        return sanitized[:100]
