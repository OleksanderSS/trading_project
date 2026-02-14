# core/analysis/context_mapper.py
"""Context Mapper for Smart Switcher"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class ContextMapper:
    """Maps market context to optimal model selection"""
    
    def __init__(self):
        self.context_rules = {
            'bull_market': {
                'preferred_models': ['lgbm', 'xgboost', 'catboost'],
                'risk_tolerance': 0.03,
                'confidence_threshold': 0.6
            },
            'bear_market': {
                'preferred_models': ['lstm', 'gru', 'transformer'],
                'risk_tolerance': 0.01,
                'confidence_threshold': 0.7
            },
            'sideways': {
                'preferred_models': ['rf', 'svm', 'knn'],
                'risk_tolerance': 0.02,
                'confidence_threshold': 0.65
            },
            'high_volatility': {
                'preferred_models': ['cnn', 'tabnet', 'autoencoder'],
                'risk_tolerance': 0.015,
                'confidence_threshold': 0.75
            }
        }
        
    def analyze_market_context(self, data: pd.DataFrame) -> str:
        """Analyze market data to determine context"""
        try:
            if 'close' not in data.columns or len(data) < 50:
                return 'sideways'
            
            # Calculate basic indicators
            returns = data['close'].pct_change().dropna()
            volatility = returns.std() * np.sqrt(252)  # Annualized volatility
            trend = (data['close'].iloc[-1] / data['close'].iloc[-20] - 1)  # 20-day trend
            
            # Determine context
            if volatility > 0.3:
                return 'high_volatility'
            elif trend > 0.05:
                return 'bull_market'
            elif trend < -0.05:
                return 'bear_market'
            else:
                return 'sideways'
                
        except Exception as e:
            logger.error(f"Error analyzing market context: {e}")
            return 'sideways'
    
    def get_context_config(self, context: str) -> Dict[str, Any]:
        """Get configuration for specific context"""
        return self.context_rules.get(context, self.context_rules['sideways'])
    
    def map_models_to_context(self, context: str, available_models: List[str]) -> List[str]:
        """Map available models to context preferences"""
        config = self.get_context_config(context)
        preferred = config['preferred_models']
        
        # Filter available models by preference
        mapped_models = []
        for model in preferred:
            if model in available_models:
                mapped_models.append(model)
        
        # Add remaining models
        for model in available_models:
            if model not in mapped_models:
                mapped_models.append(model)
        
        return mapped_models
    
    def get_risk_parameters(self, context: str) -> Dict[str, float]:
        """Get risk parameters for context"""
        config = self.get_context_config(context)
        return {
            'risk_tolerance': config['risk_tolerance'],
            'confidence_threshold': config['confidence_threshold']
        }
