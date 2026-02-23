# core/pipeline/context_aware_modeling.py

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from utils.logger import ProjectLogger
from config.feature_layers import (
    LOCAL_FEATURES, SHORT_TERM_FEATURES, TREND_CONTEXT_FEATURES,
    MACRO_CONTEXT_FEATURES, NEWS_CONTEXT_FEATURES, REVERSE_IMPACT_FEATURES
)

logger = ProjectLogger.get_logger(__name__)

class ContextAwareModeling:
    """
    Context-aware modeling that treats features in layers:
    - Core features: OHLCV + basic technical
    - Context layers: Macro, news, market regime
    - Result: Model prediction + full context table for real-time comparison
    """
    
    def __init__(self):
        self.logger = logger
        self.feature_layers = {
            'core': LOCAL_FEATURES + SHORT_TERM_FEATURES,
            'trend': TREND_CONTEXT_FEATURES,
            'macro': MACRO_CONTEXT_FEATURES,
            'news': NEWS_CONTEXT_FEATURES,
            'impact': REVERSE_IMPACT_FEATURES
        }
        self.context_history = []
        
    def create_context_aware_dataset(self,
                                    df_base: pd.DataFrame,
                                    df_macro: Optional[pd.DataFrame] = None,
                                    df_news: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Create a dataset where each row contains:
        1. Core features for model training
        2. All context features for real-time comparison
        3. Context metadata for analysis
        """
        # Start with base features
        df_context_aware = df_base.copy()
        
        # Add macro context
        if df_macro is not None:
            df_context_aware = self._add_macro_context(df_context_aware, df_macro)
            
        # Add news context
        if df_news is not None:
            df_context_aware = self._add_news_context(df_context_aware, df_news)
            
        # Add context metadata
        df_context_aware = self._add_context_metadata(df_context_aware)
        
        # Create feature layer mapping
        df_context_aware.attrs['feature_layers'] = self.feature_layers
        df_context_aware.attrs['layer_importance'] = {}
        
        return df_context_aware
    
    def _add_macro_context(self, df: pd.DataFrame, df_macro: pd.DataFrame) -> pd.DataFrame:
        """Add macro-economic context with proper alignment"""
        # Ensure datetime index
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        if not isinstance(df_macro.index, pd.DatetimeIndex):
            df_macro.index = pd.to_datetime(df_macro.index)
            
        # Forward-fill macro data to align with daily data
        macro_aligned = df_macro.reindex(df.index, method='ffill')
        
        # Add macro features with context prefix
        for col in df_macro.columns:
            if col in MACRO_CONTEXT_FEATURES:
                df[f'macro_{col}'] = macro_aligned[col]
                
        # Calculate macro context interactions
        if 'macro_VIX' in df.columns and 'macro_FEDFUNDS' in df.columns:
            df['macro_volatility_monetary'] = df['macro_VIX'] * df['macro_FEDFUNDS']
            
        if 'macro_GDP' in df.columns and 'macro_UNRATE' in df.columns:
            df['macro_economic_health'] = df['macro_GDP'] / (df['macro_UNRATE'] + 1)
            
        return df
    
    def _add_news_context(self, df: pd.DataFrame, df_news: pd.DataFrame) -> pd.DataFrame:
        """Add news context with temporal aggregation"""
        # Ensure datetime index
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        if not isinstance(df_news.index, pd.DatetimeIndex):
            df_news.index = pd.to_datetime(df_news.index)
            
        # Aggregate news by date with lookback windows
        for date in df.index:
            # 7-day lookback window
            start_date = date - timedelta(days=7)
            end_date = date + timedelta(days=1)  # Include same day
            
            window_news = df_news[(df_news.index >= start_date) & (df_news.index <= end_date)]
            
            if not window_news.empty:
                # News sentiment statistics
                if 'sentiment' in window_news.columns:
                    sentiments = window_news['sentiment'].dropna()
                    if not sentiments.empty:
                        df.loc[date, 'news_sentiment_mean'] = sentiments.mean()
                        df.loc[date, 'news_sentiment_std'] = sentiments.std()
                        df.loc[date, 'news_sentiment_trend'] = np.polyfit(range(len(sentiments)), sentiments, 1)[0] if len(sentiments) > 1 else 0
                        
                # News volume
                df.loc[date, 'news_count_7d'] = len(window_news)
                
                # News impact
                if 'impact_score' in window_news.columns:
                    df.loc[date, 'news_impact_mean'] = window_news['impact_score'].mean()
                    df.loc[date, 'news_impact_max'] = window_news['impact_score'].max()
            else:
                # Default values when no news
                df.loc[date, 'news_sentiment_mean'] = 0.0
                df.loc[date, 'news_sentiment_std'] = 0.0
                df.loc[date, 'news_sentiment_trend'] = 0.0
                df.loc[date, 'news_count_7d'] = 0
                df.loc[date, 'news_impact_mean'] = 0.0
                df.loc[date, 'news_impact_max'] = 0.0
                
        return df
    
    def _add_context_metadata(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add metadata about context quality and availability"""
        # Context availability flags
        df['ctx_has_macro'] = int(df.columns.intersection(['macro_VIX', 'macro_FEDFUNDS']).size > 0)
        df['ctx_has_news'] = int(df.columns.intersection(['news_sentiment_mean']).size > 0)
        df['ctx_has_volume'] = int('volume' in df.columns and df['volume'].notna().any())
        
        # Context quality scores
        df['ctx_completeness'] = df[['ctx_has_macro', 'ctx_has_news', 'ctx_has_volume']].sum(axis=1) / 3
        
        # Context recency (how recent is the data)
        if 'date' in df.columns:
            df['ctx_recency_days'] = (datetime.now() - pd.to_datetime(df['date'])).dt.days
        else:
            df['ctx_recency_days'] = 0
            
        # Context volatility (how stable is the context)
        if 'news_sentiment_std' in df.columns and 'macro_VIX' in df.columns:
            df['ctx_stability'] = 1.0 / (1.0 + df['news_sentiment_std'].fillna(0) + df['macro_VIX'].fillna(0) * 0.01)
        else:
            df['ctx_stability'] = 1.0
            
        return df
    
    def train_with_context_layers(self,
                                df_features: pd.DataFrame,
                                df_targets: pd.DataFrame,
                                model_type: str = 'light') -> Dict[str, Any]:
        """
        Train models using core features but track context layers for analysis.
        """
        # Identify core features (exclude context features)
        core_features = [col for col in df_features.columns if any(
            col.startswith(prefix) for prefix in ['open', 'high', 'low', 'close', 'volume', 'SMA_', 'RSI_', 'MACD_', 'ATR_']
        )]
        
        context_features = [col for col in df_features.columns if col not in core_features]
        
        # Split data
        train_size = int(len(df_features) * 0.8)
        X_train_core = df_features[core_features].iloc[:train_size]
        X_test_core = df_features[core_features].iloc[train_size:]
        
        X_train_context = df_features[context_features].iloc[:train_size]
        X_test_context = df_features[context_features].iloc[train_size:]
        
        y_train = df_targets.iloc[:train_size]
        y_test = df_targets.iloc[train_size:]
        
        # Train models on core features only
        models = self._train_models(X_train_core, y_train, model_type)
        
        # Get predictions
        predictions = {}
        for name, model in models.items():
            predictions[name] = model.predict(X_test_core)
        
        # Create context-aware results table
        results_table = self._create_results_table(
            X_test_core, X_test_context, predictions, y_test
        )
        
        # Analyze context impact
        context_analysis = self._analyze_context_impact(
            results_table, context_features
        )
        
        return {
            'models': models,
            'predictions': predictions,
            'results_table': results_table,
            'context_analysis': context_analysis,
            'core_features': core_features,
            'context_features': context_features
        }
    
    def _create_results_table(self,
                             X_core: pd.DataFrame,
                             X_context: pd.DataFrame,
                             predictions: Dict[str, np.ndarray],
                             y_true: pd.DataFrame) -> pd.DataFrame:
        """
        Create comprehensive results table with predictions and full context.
        This is what will be used for real-time comparison.
        """
        results = pd.DataFrame(index=X_core.index)
        
        # Add core features
        for col in X_core.columns:
            results[f'core_{col}'] = X_core[col]
            
        # Add context features
        for col in X_context.columns:
            results[f'ctx_{col}'] = X_context[col]
            
        # Add predictions
        for model_name, pred in predictions.items():
            results[f'pred_{model_name}'] = pred
            
        # Add actual values
        for col in y_true.columns:
            results[f'target_{col}'] = y_true[col]
            
        # Add prediction metadata
        results['prediction_date'] = results.index
        results['prediction_confidence'] = self._calculate_prediction_confidence(predictions)
        
        return results
    
    def _calculate_prediction_confidence(self, predictions: Dict[str, np.ndarray]) -> pd.Series:
        """Calculate confidence based on prediction consensus"""
        if len(predictions) == 0:
            return pd.Series(0.0, index=next(iter(predictions.values())).shape[0])
            
        # Convert predictions to DataFrame
        pred_df = pd.DataFrame(predictions)
        
        # Calculate standard deviation across models (lower = more confident)
        confidence = 1.0 / (1.0 + pred_df.std(axis=1))
        
        return confidence
    
    def _analyze_context_impact(self,
                               results_table: pd.DataFrame,
                               context_features: List[str]) -> Dict[str, Any]:
        """Analyze how context features impact prediction accuracy"""
        analysis = {
            'feature_importance': {},
            'context_correlations': {},
            'regime_analysis': {}
        }
        
        # Calculate correlations between context features and prediction errors
        for model_col in [col for col in results_table.columns if col.startswith('pred_')]:
            target_col = model_col.replace('pred_', 'target_')
            if target_col in results_table.columns:
                prediction_error = results_table[target_col] - results_table[model_col]
                
                for ctx_col in context_features:
                    full_ctx_col = f'ctx_{ctx_col}'
                    if full_ctx_col in results_table.columns:
                        correlation = results_table[full_ctx_col].corr(prediction_error)
                        if not np.isnan(correlation):
                            analysis['context_correlations'][f'{model_col}_{ctx_col}'] = correlation
                            
        return analysis
    
    def get_real_time_context_summary(self, current_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Get context summary for real-time trading decisions.
        This compares current situation to historical patterns.
        """
        summary = {
            'current_context': {},
            'historical_similarity': {},
            'recommendations': {}
        }
        
        # Current context snapshot
        summary['current_context'] = {
            'market_phase': self._detect_market_phase(current_data),
            'macro_environment': self._summarize_macro_context(current_data),
            'news_sentiment': self._summarize_news_context(current_data),
            'volatility_regime': self._detect_volatility_regime(current_data)
        }
        
        return summary
    
    def _detect_market_phase(self, df: pd.DataFrame) -> str:
        """Detect current market phase"""
        if 'SMA_50' in df.columns and 'SMA_200' in df.columns:
            latest_sma_50 = df['SMA_50'].iloc[-1] if not df['SMA_50'].empty else 0
            latest_sma_200 = df['SMA_200'].iloc[-1] if not df['SMA_200'].empty else 0
            latest_close = df['close'].iloc[-1] if 'close' in df.columns and not df['close'].empty else 0
            
            if latest_close > latest_sma_50 > latest_sma_200:
                return 'bull'
            elif latest_close < latest_sma_50 < latest_sma_200:
                return 'bear'
            else:
                return 'neutral'
        return 'unknown'
    
    def _summarize_macro_context(self, df: pd.DataFrame) -> Dict[str, float]:
        """Summarize macro-economic context"""
        macro_summary = {}
        
        macro_cols = ['macro_VIX', 'macro_FEDFUNDS', 'macro_GDP', 'macro_UNRATE']
        for col in macro_cols:
            if col in df.columns:
                latest_value = df[col].iloc[-1] if not df[col].empty else 0
                macro_summary[col.replace('macro_', '')] = latest_value
                
        return macro_summary
    
    def _summarize_news_context(self, df: pd.DataFrame) -> Dict[str, float]:
        """Summarize news context"""
        news_summary = {}
        
        news_cols = ['news_sentiment_mean', 'news_count_7d', 'news_impact_mean']
        for col in news_cols:
            if col in df.columns:
                latest_value = df[col].iloc[-1] if not df[col].empty else 0
                news_summary[col.replace('news_', '')] = latest_value
                
        return news_summary
    
    def _detect_volatility_regime(self, df: pd.DataFrame) -> str:
        """Detect current volatility regime"""
        if 'ATR_14' in df.columns and not df['ATR_14'].empty:
            latest_atr = df['ATR_14'].iloc[-1]
            atr_history = df['ATR_14'].dropna()
            
            if len(atr_history) > 20:
                atr_percentile = (latest_atr > atr_history).mean()
                if atr_percentile > 0.8:
                    return 'high'
                elif atr_percentile < 0.2:
                    return 'low'
                else:
                    return 'normal'
                    
        return 'unknown'
    
    def _train_models(self, X_train, y_train, model_type: str) -> Dict[str, Any]:
        """Train models - placeholder for actual implementation"""
        # This would integrate with existing model training logic
        return {'placeholder_model': None}
