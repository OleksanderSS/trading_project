"""
Argument parsing utilities for hybrid pipeline.
"""

import argparse


def create_argument_parser() -> argparse.ArgumentParser:
    """Create and configure argument parser."""
    parser = argparse.ArgumentParser(description='Hybrid Trading Pipeline')
    parser.add_argument(
        '--mode',
        # 'calibrate' removed: it was advertised here and dispatched in
        # run_hybrid_pipeline.py to an executor method that does not exist.
        choices=['local', 'full', 'prepare', 'light', 'continue'],
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
        default=100,
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
        '--n-trials',
        type=int,
        default=50,
        help='Number of Optuna trials for calibration mode'
    )
    return parser
