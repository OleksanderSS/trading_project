# core/pipeline/context_enhanced_modeling.py

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from utils.logger import ProjectLogger
from utils.trading_calendar import TradingCalendar
from core.pipeline.context_features import build_context_features
from core.trading_advisor import TradingAdvisor

logger = ProjectLogger.get_logger(__name__)

class ContextEnhancedModeling:
    """
    Enhanced modeling that adds macro/context features AFTER model training.
    This treats models as "eyes" and context as "expanded awareness".
    """
    
    def __init__(self, calendar: Optional[TradingCalendar] = None):
        self.calendar = calendar
        self.logger = logger
        self.context_cache = {}  # Cache context features by date
        
    def enhance_model_predictions(self, 
                                model_predictions: Dict[str, np.ndarray],
                                df_features: pd.DataFrame,
                                df_context: pd.DataFrame,
                                context_weights: Optional[Dict[str, float]] = None) -> Dict[str, np.ndarray]:
        """
        Enhance model predictions with context features.
        
        Args:
            model_predictions: Raw predictions from models (heavy/light)
            df_features: Feature dataframe used for predictions
            df_context: Context features (macro, news, market regime)
            context_weights: Weights for different context features
            
        Returns:
            Enhanced predictions with context applied
        """
        if context_weights is None:
            context_weights = {
                'market_phase': 0.3,
                'trend_alignment': 0.25,
                'macro_bias': 0.2,
                'volatility': 0.15,
                'news_sentiment': 0.1
            }
        
        enhanced_predictions = {}
        
        for model_name, predictions in model_predictions.items():
            # Get context features aligned with predictions
            aligned_context = self._align_context_with_predictions(
                df_context, df_features, predictions
            )
            
            # Apply context enhancement
            enhanced = self._apply_context_enhancement(
                predictions, aligned_context, context_weights
            )
            
            enhanced_predictions[model_name] = enhanced
            
        return enhanced_predictions
    
    def _align_context_with_predictions(self, 
                                     df_context: pd.DataFrame,
                                     df_features: pd.DataFrame,
                                     predictions: np.ndarray) -> pd.DataFrame:
        """Align context features with prediction timeline"""
        # Ensure same index
        if not df_context.index.equals(df_features.index):
            # Try to align by date
            if 'date' in df_features.columns:
                df_context_aligned = df_context.reindex(df_features['date'], method='nearest')
            else:
                df_context_aligned = df_context.reindex(df_features.index, method='nearest')
        else:
            df_context_aligned = df_context.copy()
            
        # Ensure same length as predictions
        if len(df_context_aligned) != len(predictions):
            # Trim or pad to match
            if len(df_context_aligned) > len(predictions):
                df_context_aligned = df_context_aligned.iloc[:len(predictions)]
            else:
                # Pad with last values
                padding = len(predictions) - len(df_context_aligned)
                last_row = df_context_aligned.iloc[-1:]
                df_context_aligned = pd.concat([
                    df_context_aligned,
                    pd.concat([last_row] * padding, ignore_index=True)
                ], ignore_index=True)
                
        return df_context_aligned
    
    def _apply_context_enhancement(self, 
                                 predictions: np.ndarray,
                                 context_features: pd.DataFrame,
                                 weights: Dict[str, float]) -> np.ndarray:
        """Apply context enhancement to predictions"""
        enhanced = predictions.copy()
        
        # Market phase enhancement
        if 'market_phase' in context_features.columns:
            phase_multiplier = context_features['market_phase'].map({
                'bull': 1.1,
                'bear': 0.9,
                'neutral': 1.0
            }).fillna(1.0)
            enhanced *= phase_multiplier.values * weights.get('market_phase', 0.3)
            
        # Trend alignment enhancement
        if 'trend_alignment' in context_features.columns:
            trend_boost = 1.0 + context_features['trend_alignment'] * 0.1
            enhanced *= trend_boost.values * weights.get('trend_alignment', 0.25)
            
        # Macro bias enhancement
        if 'macro_bias' in context_features.columns:
            # Normalize macro bias to [-0.1, 0.1]
            macro_normalized = np.tanh(context_features['macro_bias'] / 2.0) * 0.1
            enhanced += macro_normalized.values * weights.get('macro_bias', 0.2)
            
        # Volatility adjustment
        if 'macro_volatility' in context_features.columns:
            # High volatility reduces confidence
            vol_factor = 1.0 / (1.0 + context_features['macro_volatility'] * 0.01)
            enhanced *= vol_factor.values * weights.get('volatility', 0.15)
            
        # News sentiment enhancement
        if 'phase_weighted_score' in context_features.columns:
            news_boost = 1.0 + context_features['phase_weighted_score'] * 0.05
            enhanced *= news_boost.values * weights.get('news_sentiment', 0.1)
            
        return enhanced
    
    def build_enhanced_features(self, 
                               df_base: pd.DataFrame,
                               news_data: Optional[pd.DataFrame] = None,
                               macro_data: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Build enhanced features combining base features with context.
        This is where we add the "expanded awareness" to model inputs.
        """
        # Build context features
        df_context = build_context_features(df_base, self.calendar)
        
        # Add news context if available
        if news_data is not None:
            df_context = self._add_news_context(df_context, news_data)
            
        # Add macro context if available
        if macro_data is not None:
            df_context = self._add_macro_context(df_context, macro_data)
            
        # Combine base features with context
        df_enhanced = pd.concat([df_base, df_context], axis=1)
        
        return df_enhanced
    
    def _add_news_context(self, df_context: pd.DataFrame, news_data: pd.DataFrame) -> pd.DataFrame:
        """Add news-related context features"""
        # Aggregate news by date
        if 'date' in news_data.columns and 'sentiment' in news_data.columns:
            news_agg = news_data.groupby('date').agg({
                'sentiment': ['mean', 'count', 'std'],
                'title': 'count'
            }).fillna(0)
            
            news_agg.columns = ['news_sentiment_avg', 'news_count', 'news_sentiment_std', 'news_title_count']
            
            # Merge with context
            df_context = df_context.merge(news_agg, left_index=True, right_index=True, how='left')
            df_context.fillna(0, inplace=True)
            
        return df_context
    
    def _add_macro_context(self, df_context: pd.DataFrame, macro_data: pd.DataFrame) -> pd.DataFrame:
        """Add macro-economic context features"""
        # Assuming macro_data has date index and macro indicators
        macro_cols = ['GDP', 'CPI', 'VIX', 'FEDFUNDS', 'UNRATE']
        available_macros = [col for col in macro_cols if col in macro_data.columns]
        
        if available_macros:
            # Forward-fill macro data
            macro_ffilled = macro_data[available_macros].fillna(method='ffill')
            
            # Merge with context
            df_context = df_context.merge(macro_ffilled, left_index=True, right_index=True, how='left')
            df_context.fillna(method='ffill', inplace=True)
            
        return df_context
    
    def train_with_context(self,
                         df_features: pd.DataFrame,
                         df_targets: pd.DataFrame,
                         model_type: str = 'light',
                         context_weight: float = 0.3) -> Dict[str, Any]:
        """
        Train models with context enhancement.
        The context is applied AFTER initial model training.
        """
        # Split data
        train_size = int(len(df_features) * 0.8)
        X_train = df_features.iloc[:train_size]
        X_test = df_features.iloc[train_size:]
        y_train = df_targets.iloc[:train_size]
        y_test = df_targets.iloc[train_size:]
        
        # Build context features
        df_context = self.build_enhanced_features(df_features)
        
        # Train base models (the "eyes")
        base_models = self._train_base_models(X_train, y_train, model_type)
        
        # Get base predictions
        base_predictions = {}
        for name, model in base_models.items():
            base_predictions[name] = model.predict(X_test)
        
        # Enhance predictions with context (the "expanded awareness")
        context_features = df_context.iloc[train_size:]
        enhanced_predictions = self.enhance_model_predictions(
            base_predictions, X_test, context_features
        )
        
        # Evaluate both base and enhanced
        results = {
            'base_models': base_models,
            'base_predictions': base_predictions,
            'enhanced_predictions': enhanced_predictions,
            'base_performance': self._evaluate_predictions(base_predictions, y_test),
            'enhanced_performance': self._evaluate_predictions(enhanced_predictions, y_test),
            'context_features': context_features
        }
        
        return results
    
    def _train_base_models(self, X_train, y_train, model_type: str) -> Dict[str, Any]:
        """Train base models (heavy or light)"""
        # This would integrate with existing model training logic
        # For now, return placeholder
        return {'placeholder_model': None}
    
    def _evaluate_predictions(self, predictions: Dict[str, np.ndarray], y_true: pd.DataFrame) -> Dict[str, float]:
        """Evaluate prediction performance"""
        from sklearn.metrics import mean_absolute_error, r2_score
        
        performance = {}
        for name, pred in predictions.items():
            if len(pred) == len(y_true):
                performance[name] = {
                    'mae': mean_absolute_error(y_true, pred),
                    'r2': r2_score(y_true, pred)
                }
        return performance
