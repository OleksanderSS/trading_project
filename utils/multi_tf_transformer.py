# utils/multi_tf_transformer.py

import pandas as pd
import numpy as np
from typing import Dict, Optional, List
from utils.logger import ProjectLogger

logger = ProjectLogger.get_logger(__name__)

class MultiTFTransformer:
    """Multi-timeframe data transformer with alignment and validation"""
    
    def __init__(self, base_timeframe: str = "1d"):
        self.base_timeframe = base_timeframe
        self.logger = logger
        
    def transform_all_timeframes(self, 
                               df_dict: Dict[str, pd.DataFrame], 
                               strict: bool = True) -> Dict[str, pd.DataFrame]:
        """Transform all timeframes with validation"""
        df_features_all = {}
        
        for tf, df in df_dict.items():
            try:
                # Validate input dataframe
                df_validated = validate_dataframe(df, tf, strict=strict)
                
                # Apply timeframe-specific transformations
                df_transformed = self._transform_single_timeframe(df_validated, tf)
                
                # Merge with existing features if any
                if tf in df_features_all:
                    df_merged = pd.concat([df_features_all[tf], df_transformed], axis=1)
                else:
                    df_merged = df_transformed
                
                # Final validation
                df_merged = validate_dataframe(df_merged, f"{tf}_merged", strict=strict)
                df_features_all[tf] = df_merged
                logger.info(f"[MultiTFTransformer] Processed timeframe {tf}, columns={len(df_merged.columns)}")
                
            except Exception as e:
                logger.error(f"[MultiTFTransformer] Error processing {tf}: {e}")
                if strict:
                    raise
                df_features_all[tf] = pd.DataFrame()
                
        return df_features_all
    
    def _transform_single_timeframe(self, df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        """Apply transformations specific to timeframe"""
        # Add timeframe-specific features here
        df_transformed = df.copy()
        
        # Example: add rolling averages based on timeframe
        if timeframe == "1d":
            df_transformed = self._add_daily_features(df_transformed)
        elif timeframe == "1h":
            df_transformed = self._add_hourly_features(df_transformed)
        elif timeframe == "15m":
            df_transformed = self._add_intraday_features(df_transformed)
            
        return df_transformed
    
    def _add_daily_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add daily-specific features"""
        if 'close' in df.columns:
            df['sma_20'] = df['close'].rolling(20).mean()
            df['sma_50'] = df['close'].rolling(50).mean()
        return df
    
    def _add_hourly_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add hourly-specific features"""
        if 'close' in df.columns:
            df['sma_12'] = df['close'].rolling(12).mean()
            df['sma_24'] = df['close'].rolling(24).mean()
        return df
    
    def _add_intraday_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add intraday-specific features"""
        if 'close' in df.columns:
            df['sma_5'] = df['close'].rolling(5).mean()
            df['sma_10'] = df['close'].rolling(10).mean()
        return df

def validate_dataframe(df: pd.DataFrame, name: str, strict: bool = True) -> pd.DataFrame:
    """Validate dataframe structure and content"""
    if df.empty:
        msg = f"Empty dataframe: {name}"
        if strict:
            raise ValueError(msg)
        else:
            logger.warning(msg)
            return pd.DataFrame()
    
    # Check for required columns
    if 'close' not in df.columns:
        msg = f"Missing 'close' column in {name}"
        if strict:
            raise ValueError(msg)
        else:
            logger.warning(msg)
    
    # Remove infinite values
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)
    
    # Fill NaN values
    df[numeric_cols] = df[numeric_cols].fillna(method='ffill').fillna(0)
    
    return df
