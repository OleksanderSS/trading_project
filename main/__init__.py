"""
Main package - рефакторинг main.py
"""

from .orchestrator import TradingOrchestrator
from .cli import create_cli_parser, run_cli

__all__ = ['TradingOrchestrator', 'create_cli_parser', 'run_cli']
