#!/usr/bin/env python3
"""
Trading Package Initialization
"""

from .real_data_collector import RealDataCollector
from .virtual_portfolio import VirtualPortfolio
from .real_trading_system import RealTradingSystem

__all__ = [
    'RealDataCollector',
    'VirtualPortfolio', 
    'RealTradingSystem'
]

__version__ = '1.0.0'
__description__ = 'Real Trading System with Virtual Portfolio'
__author__ = 'Trading System Development Team'
