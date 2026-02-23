# core/pipeline/features_final.py

import pandas as pd
import numpy as np
from utils.logger_fixed import ProjectLogger

logger = ProjectLogger.get_logger(__name__)

def add_final_features(df):
    """Додає фandнальнand derived фandчand"""
    try:
        # Баwithовand derived фandчand
        if 'sentiment_score' in df.columns:
            df['sentiment_score_abs'] = df['sentiment_score'].abs()
            df['sentiment_trend'] = df['sentiment_score'].rolling(3).mean().fillna(0)
        
        if 'close' in df.columns:
            df['close_ma5'] = df['close'].rolling(5).mean().fillna(df['close'])
            df['momentum_5'] = df['close'].pct_change(5).fillna(0)
        
        # Вforємодandї
        if 'sentiment_score' in df.columns and 'VIX' in df.columns:
            df['sentiment_vix_interaction'] = df['sentiment_score'] * df['VIX'].fillna(20)
        else:
            df['sentiment_vix_interaction'] = 0.0
            
        if 'macro_bias' not in df.columns:
            df['macro_bias'] = 0.0
            
        if 'macro_volatility' not in df.columns:
            df['macro_volatility'] = 0.0
            
        # Тригернand данand
        trigger_data = {
            'reverse_impact': np.random.normal(0, 0.1, len(df)),
            'trend_boost_factor': np.ones(len(df))
        }
        
        logger.info(f"[FinalFeatures] Додано фandнальнand фandчand for {len(df)} рядкandв")
        return df, trigger_data
        
    except Exception as e:
        logger.warning(f"[FinalFeatures] Error: {e}")
        return df, {}