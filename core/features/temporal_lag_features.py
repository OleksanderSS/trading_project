"""
Temporal and Lag Features for максимальної точностand прогноwithування
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class TemporalLagFeatureEngine:
    """Створення часових and лагових фandчей"""
    
    def __init__(self):
        self.feature_names = []
        
    def add_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Додати часовand фandчand"""
        
        logger.info("Adding temporal features...")
        
        df = df.copy()
        
        # Перетворення дати якщо потрandбно
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            date_col = 'date'
        elif 'trade_date' in df.columns:
            df['trade_date'] = pd.to_datetime(df['trade_date'])
            date_col = 'trade_date'
        else:
            logger.warning("No date column found")
            return df
        
        # 1. Day of week
        df['day_of_week'] = df[date_col].dt.dayofweek
        self.feature_names.append('day_of_week')
        
        # 2. Month
        df['month'] = df[date_col].dt.month
        self.feature_names.append('month')
        
        # 3. Quarter
        df['quarter'] = df[date_col].dt.quarter
        self.feature_names.append('quarter')
        
        # 4. Day of month
        df['day_of_month'] = df[date_col].dt.day
        self.feature_names.append('day_of_month')
        
        # 5. Week of year
        df['week_of_year'] = df[date_col].dt.isocalendar().week
        self.feature_names.append('week_of_year')
        
        # 6. Is month end (potentially important for trading)
        df['is_month_end'] = df[date_col].dt.is_month_end.astype(int)
        self.feature_names.append('is_month_end')
        
        # 7. Is month start
        df['is_month_start'] = df[date_col].dt.is_month_start.astype(int)
        self.feature_names.append('is_month_start')
        
        # 8. Days to month end
        next_month_end = df[date_col] + pd.offsets.MonthEnd(0)
        df['days_to_month_end'] = (next_month_end - df[date_col]).dt.days
        self.feature_names.append('days_to_month_end')
        
        # 9. Hour (if intraday data)
        if df[date_col].dt.hour.nunique() > 1:
            df['hour'] = df[date_col].dt.hour
            self.feature_names.append('hour')
            
            # 10. Is trading session (9:30 AM - 4:00 PM)
            df['is_trading_session'] = ((df[date_col].dt.hour >= 9) & 
                                      (df[date_col].dt.hour < 16) & 
                                      (df[date_col].dt.weekday < 5)).astype(int)
            self.feature_names.append('is_trading_session')
            
            # 11. Is pre-market (before 9:30 AM)
            df['is_premarket'] = ((df[date_col].dt.hour < 9) & 
                                 (df[date_col].dt.weekday < 5)).astype(int)
            self.feature_names.append('is_premarket')
            
            # 12. Is after-hours (after 4:00 PM)
            df['is_after_hours'] = ((df[date_col].dt.hour >= 16) & 
                                   (df[date_col].dt.weekday < 5)).astype(int)
            self.feature_names.append('is_after_hours')
        
        # 13. Cyclical encoding for day of week
        df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        self.feature_names.extend(['day_of_week_sin', 'day_of_week_cos'])
        
        # 14. Cyclical encoding for month
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        self.feature_names.extend(['month_sin', 'month_cos'])
        
        # 15. Season (winter, spring, summer, fall)
        df['season'] = df['month'].map({12: 0, 1: 0, 2: 0,  # Winter
                                        3: 1, 4: 1, 5: 1,  # Spring
                                        6: 2, 7: 2, 8: 2,  # Summer
                                        9: 3, 10: 3, 11: 3})  # Fall
        self.feature_names.append('season')
        
        logger.info(f"Added {len([f for f in self.feature_names if 'temporal' in f or f in ['day_of_week',
            'month',
            'quarter']])} temporal features")
        
        return df
    
    def add_lag_features(self, df: pd.DataFrame, price_cols: List[str], 
                        lags: List[int] = [1, 2, 3, 5, 10]) -> pd.DataFrame:
        """Додати лаговand фandчand for цandнових колонок"""
        
        logger.info(f"Adding lag features for {len(price_cols)} price columns...")
        
        df = df.copy()
        
        for col in price_cols:
            if col not in df.columns:
                logger.warning(f"Column {col} not found, skipping")
                continue
                
            # Lagged values
            for lag in lags:
                lag_col = f"{col}_lag_{lag}"
                df[lag_col] = df[col].shift(lag)
                self.feature_names.append(lag_col)
            
            # Lagged returns
            for lag in lags:
                if lag == 1:
                    return_col = f"{col}_return_lag_{lag}"
                else:
                    return_col = f"{col}_return_lag_{lag}"
                
                df[return_col] = df[col].pct_change(lag).shift(lag)
                self.feature_names.append(return_col)
            
            # Moving averages
            for window in [5, 10, 20, 50]:
                if len(df) >= window:
                    ma_col = f"{col}_ma_{window}"
                    df[ma_col] = df[col].rolling(window=window).mean()
                    self.feature_names.append(ma_col)
                    
                    # Price relative to moving average
                    pr_ma_col = f"{col}_pct_above_ma_{window}"
                    df[pr_ma_col] = (df[col] - df[ma_col]) / df[ma_col]
                    self.feature_names.append(pr_ma_col)
            
            # Exponential moving averages
            for span in [12, 26]:
                ema_col = f"{col}_ema_{span}"
                df[ema_col] = df[col].ewm(span=span).mean()
                self.feature_names.append(ema_col)
                
                # Price relative to EMA
                pr_ema_col = f"{col}_pct_above_ema_{span}"
                df[pr_ema_col] = (df[col] - df[ema_col]) / df[ema_col]
                self.feature_names.append(pr_ema_col)
        
        logger.info(f"Added lag features for {len(price_cols)} columns")
        
        return df
    
    def add_volatility_features(self, df: pd.DataFrame, price_cols: List[str]) -> pd.DataFrame:
        """Додати волатильнand фandчand"""
        
        logger.info("Adding volatility features...")
        
        df = df.copy()
        
        for col in price_cols:
            if col not in df.columns:
                continue
            
            # Rolling volatility (standard deviation of returns)
            for window in [5, 10, 20]:
                if len(df) >= window:
                    vol_col = f"{col}_vol_{window}"
                    returns = df[col].pct_change()
                    df[vol_col] = returns.rolling(window=window).std() * np.sqrt(252)  # Annualized
                    self.feature_names.append(vol_col)
            
            # True Range (for intraday data)
            if len(df) >= 2:
                high_col = col.replace('close', 'high')
                low_col = col.replace('close', 'low')
                
                if high_col in df.columns and low_col in df.columns:
                    tr_col = f"{col}_true_range"
                    df[tr_col] = np.maximum(df[high_col] - df[low_col], 
                                           np.maximum(abs(df[high_col] - df[col].shift(1)),
                                                   abs(df[col].shift(1) - df[low_col])))
                    self.feature_names.append(tr_col)
                    
                    # Average True Range
                    atr_col = f"{col}_atr_14"
                    df[atr_col] = df[tr_col].rolling(window=14).mean()
                    self.feature_names.append(atr_col)
        
        # VIX-like volatility index (if multiple tickers)
        close_cols = [col for col in price_cols if 'close' in col]
        if len(close_cols) > 1:
            # Cross-asset volatility
            for window in [5, 10]:
                vol_cols = []
                for col in close_cols:
                    returns = df[col].pct_change()
                    vol = returns.rolling(window=window).std()
                    vol_cols.append(vol)
                
                if vol_cols:
                    vol_matrix = pd.concat(vol_cols, axis=1)
                    avg_vol = vol_matrix.mean(axis=1)
                    df[f'cross_asset_vol_{window}'] = avg_vol
                    self.feature_names.append(f'cross_asset_vol_{window}')
        
        logger.info("Added volatility features")
        
        return df
    
    def add_momentum_features(self, df: pd.DataFrame, price_cols: List[str]) -> pd.DataFrame:
        """Додати momentum фandчand"""
        
        logger.info("Adding momentum features...")
        
        df = df.copy()
        
        for col in price_cols:
            if col not in df.columns:
                continue
            
            # Price momentum over different periods
            for period in [1, 5, 10, 20]:
                if len(df) >= period + 1:
                    mom_col = f"{col}_momentum_{period}"
                    df[mom_col] = (df[col] - df[col].shift(period)) / df[col].shift(period)
                    self.feature_names.append(mom_col)
            
            # Rate of change (ROC)
            for period in [5, 10, 20]:
                if len(df) >= period + 1:
                    roc_col = f"{col}_roc_{period}"
                    df[roc_col] = ((df[col] - df[col].shift(period)) / df[col].shift(period)) * 100
                    self.feature_names.append(roc_col)
            
            # Acceleration (change in momentum)
            for period in [5, 10]:
                if len(df) >= period + 2:
                    acc_col = f"{col}_acceleration_{period}"
                    momentum = df[col].pct_change(period)
                    df[acc_col] = momentum - momentum.shift(period)
                    self.feature_names.append(acc_col)
        
        logger.info("Added momentum features")
        
        return df
    
    def add_interaction_features(self, df: pd.DataFrame, feature_pairs: List[tuple]) -> pd.DataFrame:
        """Додати фandчand вforємодandї"""
        
        logger.info("Adding interaction features...")
        
        df = df.copy()
        
        for feat1, feat2 in feature_pairs[:20]:  # Обмежуємо кandлькandсть
            if feat1 in df.columns and feat2 in df.columns:
                # Multiplication interaction
                interaction_col = f"{feat1}_x_{feat2}"
                df[interaction_col] = df[feat1] * df[feat2]
                self.feature_names.append(interaction_col)
                
                # Ratio interaction (if no zeros)
                if (df[feat2] != 0).all():
                    ratio_col = f"{feat1}_div_{feat2}"
                    df[ratio_col] = df[feat1] / df[feat2]
                    self.feature_names.append(ratio_col)
        
        logger.info("Added interaction features")
        
        return df
    
    def add_all_features(self, df: pd.DataFrame, price_cols: List[str] = None) -> pd.DataFrame:
        """Додати all фandчand"""
        
        if price_cols is None:
            # Автоматично withнаходимо цandновand колонки
            price_cols = [col for col in df.columns if any(x in col.lower() 
                         for x in ['close', 'open', 'high', 'low'])]
        
        logger.info(f"Starting feature engineering for {len(price_cols)} price columns...")
        
        # 1. Temporal features
        df = self.add_temporal_features(df)
        
        # 2. Lag features
        df = self.add_lag_features(df, price_cols)
        
        # 3. Volatility features
        df = self.add_volatility_features(df, price_cols)
        
        # 4. Momentum features
        df = self.add_momentum_features(df, price_cols)
        
        # 5. Interaction features (selective)
        if len(self.feature_names) > 10:
            # Беремо першand 10 фandчей for вforємодandї
            base_features = self.feature_names[:10]
            feature_pairs = [(base_features[i], base_features[j]) 
                           for i in range(len(base_features)) 
                           for j in range(i+1, min(i+4, len(base_features)))]
            df = self.add_interaction_features(df, feature_pairs)
        
        # Заповнюємо NaN values
        df = df.fillna(0)
        
        logger.info(f"Feature engineering completed. Total features: {len(self.feature_names)}")
        
        return df


def create_enhanced_features(df: pd.DataFrame, price_cols: List[str] = None) -> pd.DataFrame:
    """Створення покращених фandчей"""
    
    feature_engine = TemporalLagFeatureEngine()
    return feature_engine.add_all_features(df, price_cols)


if __name__ == "__main__":
    # Тестування
    df = pd.read_parquet('c:/trading_project/data/stages/merged_full.parquet')
    
    # Знаходимо цandновand колонки
    price_cols = [col for col in df.columns if any(x in col.lower() 
                 for x in ['close', 'open', 'high', 'low'])]
    
    logger.info(f"Found {len(price_cols)} price columns")
    logger.info(f"Original shape: {df.shape}")
    
    # Додаємо фandчand
    df_enhanced = create_enhanced_features(df, price_cols)
    
    logger.info(f"Enhanced shape: {df_enhanced.shape}")
    logger.info(f"New features added: {df_enhanced.shape[1] - df.shape[1]}")
    
    # Зберandгаємо
    output_path = 'c:/trading_project/data/stages/enhanced_with_temporal_features.parquet'
    df_enhanced.to_parquet(output_path)
    logger.info(f"Saved to: {output_path}")