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
        # Each mode SAYS which stages it runs, because the one defect this
        # switch has produced was a mode whose name implied more than it did.
        # `--mode light` runs `stages_to_run=[4]` and nothing else
        # (`light_models_trainer.py`), and on 2026-09-01 a twelve-hour run of
        # it was twice recommended as a way to exercise stages 5-7, which it
        # has never touched (REGISTER #205, CLAIMS P8). The help string then
        # read "Pipeline execution mode" -- five words that could not be
        # wrong, and could not be checked either.
        #
        # A claim written here is a claim a test can hold to the code, which
        # is the whole content of scan unit P4 mode A: a switch with no stated
        # promise cannot be found to have broken one.
        help=(
            "Which stages to run. "
            "local: stages 0-3, ending with the enriched frame. "
            "prepare: stages 0-3 and then write the Colab batch. "
            "light: STAGE 4 ONLY -- trains models and stops; it does not "
            "predict, trade or evaluate. "
            "full: the whole hybrid preparation flow, pausing for Colab. "
            "continue: stages 5-7 on results already on disk; add "
            "--skip-training to reuse the champions instead of retraining."
        ),
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
        '--allow-missing-timeframes',
        action='store_true',
        help=(
            'Prepare mode: succeed even when a requested timeframe produced no '
            'rows. Without this the run fails, because a cadence can be dropped '
            'anywhere in stages 2-3 and every downstream stage then reports '
            'success on a smaller scope than was asked for. Pass it when the '
            'gap is known and intended, so that the intent is in the command '
            'rather than in nobody memory.'
        )
    )
    parser.add_argument(
        '--skip-training',
        action='store_true',
        help=(
            'Continue mode only: run stages 5-7 on the champions already on '
            'disk instead of training again. Verifying twenty minutes of '
            'final-stage work cost ten hours of re-training without this, '
            'which is why those stages stayed the least-tested part of the '
            'system.'
        )
    )
    parser.add_argument(
        '--n-trials',
        type=int,
        default=50,
        help='Number of Optuna trials for calibration mode'
    )
    return parser
