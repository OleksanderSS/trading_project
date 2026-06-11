"""
Argument parsing utilities for hybrid pipeline.
"""

import argparse


def create_argument_parser() -> argparse.ArgumentParser:
    """Create and configure argument parser."""
    parser = argparse.ArgumentParser(description='Hybrid Trading Pipeline')
    parser.add_argument(
        '--mode',
        choices=['local', 'full', 'prepare', 'light', 'continue', 'calibrate'],
        default='local',
        help='Pipeline execution mode'
    )
    parser.add_argument(
        '--tickers',
        nargs='+',
        default=None,
        help='List of tickers to process (if not specified, uses assets preset)'
    )
    parser.add_argument(
        '--test-ticker',
        help='Test with specific ticker'
    )
    parser.add_argument(
        '--test-target',
        help='Test with specific target'
    )
    parser.add_argument(
        '--test-model',
        help='Test with specific model'
    )
    parser.add_argument(
        '--batch-name',
        help='Custom batch name'
    )
    parser.add_argument(
        '--max-iterations',
        type=int,
        default=None,
        help='Maximum iterations for training'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=None,
        help='Number of epochs for training (test mode only)'
    )
    parser.add_argument(
        '--stages',
        nargs='+',
        type=int,
        help='Specific stages to run (4-7)'
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Force execution even if validation fails'
    )
    parser.add_argument(
        '--skip-colab',
        action='store_true',
        help='Skip Colab stage and run final stages locally using fallback features'
    )
    parser.add_argument(
        '--force-training',
        action='store_true',
        help='Force retraining and refresh cached local data'
    )
    parser.add_argument(
        '--force-feature-selection',
        action='store_true',
        help='Force feature selection even if existing selection is available'
    )
    parser.add_argument(
        '--n-trials',
        type=int,
        default=50,
        help='Number of Optuna trials for calibration mode'
    )
    # CLI-options for internal flags (Point 4)
    parser.add_argument(
        '--max-models',
        type=int,
        default=50,
        help='Maximum models to keep in PersistentModelPool (default: 50)'
    )
    parser.add_argument(
        '--drift-threshold',
        type=float,
        default=0.3,
        help='Drift detection threshold for ModelQualityController (default: 0.3)'
    )
    parser.add_argument(
        '--min-quality',
        type=float,
        default=0.5,
        help='Minimum acceptable model quality score (default: 0.5)'
    )
    return parser
