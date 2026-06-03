#!/usr/bin/env python3
"""
Decay Predictor - Prediction Logic
Handles news impact prediction using trained decay models.
"""

import pandas as pd
import numpy as np
from datetime import timedelta
from typing import Optional, Dict, Any

from src.core.logging.logger import ProjectLogger

logger = ProjectLogger.get_logger("DecayPredictor")


class DecayPredictor:
    """
    Decay predictor for news impact prediction.
    
    Handles:
    - News impact prediction
    - Time-series impact forecasting
    - Model-based prediction
    """
    
    def __init__(self, decay_functions: Dict[str, Dict[str, Any]]):
        """
        Initialize Decay Predictor.
        
        Args:
            decay_functions: Dictionary of decay function configurations
        """
        self.logger = logger
        self.decay_functions = decay_functions
        self.logger.info("✅ DecayPredictor initialized")
    
    def predict_news_impact(self,
                           news_data: pd.DataFrame,
                           trained_models: Dict[str, Any],
                           model_name: Optional[str] = None,
                           news_type: Optional[str] = None,
                           decay_parameters: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        """
        Predict news impact using trained decay models.
        
        Args:
            news_data: DataFrame with news timestamps
            trained_models: Dictionary of trained models
            model_name: Specific decay model to use (uses best if None)
            news_type: News type for model selection
            decay_parameters: Additional decay parameters
            
        Returns:
            DataFrame with predicted impact over time
        """
        try:
            if not trained_models:
                self.logger.warning('No trained models available. Please fit models first.')
                return pd.DataFrame()
            
            if model_name and model_name in trained_models:
                model_info = trained_models[model_name]
            elif news_type and decay_parameters and news_type in decay_parameters:
                model_info = decay_parameters[news_type]
            else:
                model_info = trained_models.get('best_overall_model', {})
            
            if not model_info or 'parameters' not in model_info:
                self.logger.error('No valid model found for prediction')
                return pd.DataFrame()
            
            predictions = []
            function_config = self.decay_functions.get(model_info.get('function_name'))
            
            if not function_config:
                return pd.DataFrame()
            
            for _, news_row in news_data.iterrows():
                news_time = pd.to_datetime(news_row['timestamp'])
                hours_range = np.arange(0, 73, 1)
                impact_predictions = function_config['function'](
                    hours_range,
                    *list(model_info['parameters'].values())
                )
                
                for hour, impact in zip(hours_range, impact_predictions):
                    prediction_time = news_time + timedelta(hours=hour)
                    predictions.append({
                        'news_timestamp': news_time,
                        'prediction_timestamp': prediction_time,
                        'hours_since_news': hour,
                        'predicted_impact': impact,
                        'model_name': model_info.get('function_name'),
                        'news_type': news_type
                    })
            
            return pd.DataFrame(predictions)
        except Exception as e:
            self.logger.error(f'Error predicting news impact: {e}')
            return pd.DataFrame()
