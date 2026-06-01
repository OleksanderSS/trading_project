#!/usr/bin/env python3
"""
Decay Function Registry - Decay Function Management
Manages decay function definitions and news type configurations.
"""

import numpy as np
from typing import Dict, Any, Callable

from src.core.logging.logger import ProjectLogger

logger = ProjectLogger.get_logger("DecayFunctionRegistry")


class DecayFunctionRegistry:
    """
    Decay function registry for managing decay function definitions.
    
    Handles:
    - Decay function definitions
    - News type configurations
    - Function parameter management
    """
    
    def __init__(self):
        """Initialize Decay Function Registry."""
        self.logger = logger
        self.decay_functions = self._initialize_decay_functions()
        self.news_types = self._initialize_news_types()
        self.logger.info("✅ DecayFunctionRegistry initialized")
    
    def _initialize_decay_functions(self) -> Dict[str, Dict[str, Any]]:
        """Initialize decay function definitions."""
        return {
            'exponential': {
                'description': 'Exponential decay with half-life parameter',
                'params': ['half_life_hours', 'initial_impact'],
                'function': lambda t, hl, init: init * np.exp(-np.log(2) * t / hl) if hl > 0 else init
            },
            'linear': {
                'description': 'Linear decay over time',
                'params': ['max_impact_hours', 'initial_impact'],
                'function': lambda t, max_h, init: max(0, init * (1 - t / max_h) if max_h > 0 else init)
            },
            'logarithmic': {
                'description': 'Logarithmic decay with scale factor',
                'params': ['scale_factor', 'initial_impact'],
                'function': lambda t, scale, init: init / (1 + t / scale) if scale > 0 else init
            },
            'step_function': {
                'description': 'Step function with immediate impact and gradual decay',
                'params': ['immediate_hours', 'decay_rate', 'initial_impact'],
                'function': lambda t, imm_h, rate, init: init if t < imm_h else init * np.exp(-rate * (t - imm_h))
            },
            'power_law': {
                'description': 'Power law decay with exponent parameter',
                'params': ['exponent', 'scale_hours', 'initial_impact'],
                'function': lambda t, exp, scale, init: init / (1 + (t / scale) ** exp) if scale > 0 else init
            }
        }
    
    def _initialize_news_types(self) -> Dict[str, Dict[str, Any]]:
        """Initialize news type configurations."""
        return {
            'earnings': {
                'description': 'Earnings announcements',
                'typical_decay_hours': 48,
                'typical_function': 'exponential',
                'impact_duration': 'long'
            },
            'macro': {
                'description': 'Macroeconomic data releases',
                'typical_decay_hours': 24,
                'typical_function': 'step_function',
                'impact_duration': 'medium'
            },
            'sector': {
                'description': 'Sector-specific news',
                'typical_decay_hours': 12,
                'typical_function': 'exponential',
                'impact_duration': 'medium'
            },
            'company_specific': {
                'description': 'Company-specific news',
                'typical_decay_hours': 6,
                'typical_function': 'linear',
                'impact_duration': 'short'
            },
            'market_sentiment': {
                'description': 'General market sentiment',
                'typical_decay_hours': 8,
                'typical_function': 'logarithmic',
                'impact_duration': 'short'
            }
        }
    
    def get_decay_function(self, function_name: str) -> Dict[str, Any]:
        """Get decay function configuration by name."""
        return self.decay_functions.get(function_name, {})
    
    def get_news_type_config(self, news_type: str) -> Dict[str, Any]:
        """Get news type configuration by name."""
        return self.news_types.get(news_type, {})
    
    def get_all_decay_functions(self) -> Dict[str, Dict[str, Any]]:
        """Get all decay function configurations."""
        return self.decay_functions.copy()
    
    def get_all_news_types(self) -> Dict[str, Dict[str, Any]]:
        """Get all news type configurations."""
        return self.news_types.copy()
