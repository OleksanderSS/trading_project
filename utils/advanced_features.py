#!/usr/bin/env python3
"""
Advanced Features Module
Розширені фічі для трейдингу
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

def temporal_train_test_split(df: pd.DataFrame, test_size: float = 0.2):
    """
    Proper temporal train/test split for time series data.
    Prevents data leakage by using chronological split.
    """
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Input must be pandas DataFrame")
    
    split_idx = int(len(df) * (1 - test_size))
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()
    
    logger.info(f"Temporal split: train={len(train_df)}, test={len(test_df)}")
    return train_df, test_df

def time_series_cross_validation(model, X, y, cv=5):
    """
    Time series cross-validation to prevent data leakage.
    """
    from sklearn.model_selection import TimeSeriesSplit
    
    tscv = TimeSeriesSplit(n_splits=cv)
    scores = []
    
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        model.fit(X_train, y_train)
        score = model.score(X_val, y_val)
        scores.append(score)
        
        logger.info(f"Fold {fold + 1}: score = {score:.4f}")
    
    return np.mean(scores), np.std(scores)



import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta

from .trading_days import get_previous_trading_days, get_previous_trading_sessions, get_us_holidays

logger = logging.getLogger(__name__)

def calculate_rsi_on_candles(prices, period=14):
    """
    Calculating RSI for sequential candles
    
    Args:
        prices: pd.Series with prices closing, sorted for часом
        period: period RSI
        
    Returns:
        pd.Series: RSI values
    """
    if len(prices) < period:
        return pd.Series([np.nan] * len(prices), index=prices.index)
    
    # Calculating differences
    delta = prices.diff()
    
    # Роseparate на gain and loss
    gain = (delta.where(delta > 0, 0)).abs()
    loss = (-delta.where(delta < 0, 0)).abs()
    
    # Calculating середнand values
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    
    # Уникаємо дandлення на нуль
    rs = avg_gain / avg_loss.where(avg_loss != 0, 1e-10)
    
    # Calculating RSI
    rsi = 100 - (100 / (1 + rs))
    
    return rsi


def add_pre_post_features(df):
    """
    Додає PRE and POST фandчand до даandсету
    """
    logger.info("[DATA] Adding PRE/POST phase features...")
    
    df = df.copy()
    
    # Сортуємо DataFrame for часом for правильного роwithрахунку andндикаторandв
    if 'published_at' in df.columns:
        df = df.sort_values('published_at')
    
    # PRE phase features (before news) - роwithраховуємо for кожного тandкера/andймфрейму
    for interval in ['15m', '60m', '1d']:
        for ticker in ['spy', 'qqq', 'tsla', 'nvda']:
            ticker_upper = ticker.upper()
            
            # Баwithовand колонки for цього тandкера/andнтервалу
            close_col = f"{interval}_{ticker}_close"
            open_col = f"{interval}_{ticker}_open"
            high_col = f"{interval}_{ticker}_high"
            low_col = f"{interval}_{ticker}_low"
            volume_col = f"{interval}_{ticker}_volume"
            
            if all(col in df.columns for col in [close_col, open_col, high_col, low_col, volume_col]):
                # Volume relative to SMA
                df[f'{ticker}_{interval}_vol_rel_pre'] = df[volume_col] / df[volume_col].rolling(20).mean()
                
                # ATR relative to close
                df[f'{ticker}_{interval}_atr_rel_pre'] = (df[high_col].rolling(14).max() - df[low_col].rolling(14).min()) / df[close_col]
                
                # Distance to EMA
                df[f'{ticker}_{interval}_dist_to_ema_pre'] = (df[close_col] - df[close_col].ewm(span=20).mean()) / df[close_col]
                
                # RSI - використовуємо shift() for правильного роwithрахунку
                # Спочатку роwithраховуємо RSI по всьому DataFrame
                delta = df[close_col].diff()
                gain = (delta.where(delta > 0, 0)).abs().rolling(window=14).mean()
                loss = (-delta.where(delta < 0, 0)).abs().rolling(window=14).mean()
                rs = 100 - (100 / (1 + gain / loss))
                
                # Потandм withсуваємо на 14, so that use попереднand данand
                df[f'{ticker}_{interval}_rsi_pre'] = rs.shift(14)
    
    # POST phase features (after news) - роwithраховуємо for кожного тandкера/andймфрейму
    for interval in ['15m', '60m', '1d']:
        for ticker in ['spy', 'qqq', 'tsla', 'nvda']:
            # Баwithовand колонки for цього тandкера/andнтервалу
            next_close_1_col = f"next_{interval}_{ticker}_close_1"
            next_open_1_col = f"next_{interval}_{ticker}_open_1"
            next_high_1_col = f"next_{interval}_{ticker}_high_1"
            next_low_1_col = f"next_{interval}_{ticker}_low_1"
            next_close_2_col = f"next_{interval}_{ticker}_close_2"
            next_open_2_col = f"next_{interval}_{ticker}_open_2"
            next_volume_1_col = f"next_{interval}_{ticker}_volume_1"
            next_volume_2_col = f"next_{interval}_{ticker}_volume_2"
            
            if all(col in df.columns for col in [next_close_1_col, next_open_1_col, next_high_1_col, next_low_1_col]):
                # Gap percentage
                df[f'{ticker}_{interval}_gap_percent'] = ((df[next_open_1_col] - df[f'{interval}_{ticker}_close']) / df[f'{interval}_{ticker}_close'] * 100).fillna(0)
                
                # Impact 1 percentage
                df[f'{ticker}_{interval}_impact_1_pct'] = ((df[next_close_1_col] - df[next_open_1_col]) / df[next_open_1_col] * 100).fillna(0)
                
                # Volume impact
                df[f'{ticker}_{interval}_vol_impact_1'] = (df[next_volume_1_col] / df[f'{interval}_{ticker}_volume'].rolling(20).mean() - 1) * 100
                
                # Shadow ratio (body vs range)
                body = abs(df[next_close_1_col] - df[next_open_1_col])
                range_total = abs(df[next_high_1_col] - df[next_low_1_col])
                df[f'{ticker}_{interval}_shadow_ratio_1'] = (range_total / body).where(body > 0, np.nan)
                
                # Impact 2 and reversal score (if second candle exists)
                if all(col in df.columns for col in [next_close_2_col, next_open_2_col]):
                    df[f'{ticker}_{interval}_impact_2_pct'] = ((df[next_close_2_col] - df[next_open_2_col]) / df[next_open_2_col] * 100).fillna(0)
                    
                    # Reversal score
                    impact_1 = df[f'{ticker}_{interval}_impact_1_pct']
                    impact_2 = df[f'{ticker}_{interval}_impact_2_pct']
                    df[f'{ticker}_{interval}_reversal_score'] = np.where(
                        (impact_1 * impact_2 < 0) & pd.notna(impact_1) & pd.notna(impact_2),
                        impact_2 / impact_1,
                        np.nan
                    )
                    
                    # Volume trend
                    df[f'{ticker}_{interval}_vol_trend'] = df[next_volume_2_col] / df[next_volume_1_col]
    
    # POST phase 2 (second candle)
    if 'next_15m_close_2_spy' in df.columns and 'next_15m_open_2_spy' in df.columns:
        # Impact 2 percentage
        df['impact_2_pct_15m_spy'] = ((df['next_15m_close_2_spy'] - df['next_15m_open_2_spy']) / df['next_15m_open_2_spy'] * 100).fillna(0)
        
        # Reversal score
        df['reversal_score_15m_spy'] = np.where(
            (df['impact_1_pct_15m_spy'] * df['impact_2_pct_15m_spy'] < 0) & 
            (df['impact_1_pct_15m_spy'].notna()) & 
            (df['impact_2_pct_15m_spy'].notna()),
            df['impact_2_pct_15m_spy'] / df['impact_1_pct_15m_spy'],
            np.nan
        )
        
        # Volume trend
        df['vol_trend_15m_spy'] = df['next_15m_volume_2_spy'] / df['next_15m_volume_1_spy']
    
    logger.info(f"[OK] Added PRE/POST features: {len([col for col in df.columns if 'pre' in col or 'post' in col or 'impact' in col])} features")
    
    return df

def add_event_features(df):
    """
    Додає EVENT фandчand до даandсету
    """
    logger.info(" Adding EVENT features...")
    
    df = df.copy()
    
    # Time-based events
    if 'published_at' in df.columns:
        df['is_pre_market'] = (df['published_at'].dt.hour < 13) | ((df['published_at'].dt.hour == 13) & (df['published_at'].dt.minute < 30))
        df['is_during_market'] = ~df['is_pre_market']
        df['is_post_market'] = (df['published_at'].dt.hour >= 16)
        
        # Market session classification (0: pre, 1: during, 2: post)
        df['market_session'] = np.where(
            df['is_pre_market'], 0,
            np.where(
                df['is_during_market'], 1,
                np.where(df['is_post_market'], 2, 3)
            )
        )
        
        # Earnings day
        df['is_earnings_day'] = df['published_at'].dt.dayofweek.isin([0, 1, 2, 3])  # Mon-Thu are typical earnings days
        
        # Quarter end
        df['is_quarter_end'] = df['published_at'].dt.month.isin([3, 6, 9, 12])
        
        # FOMC day (first Wednesday)
        df['is_fomc_day'] = (df['published_at'].dt.dayofweek == 2) & (df['published_at'].dt.day <= 7)
        
        # Time features
        df['weekday'] = df['published_at'].dt.dayofweek  # 0-6
        df['hour_of_day'] = df['published_at'].dt.hour  # 0-23
    
    # Price-based events
    if 'gap_percent_15m_spy' in df.columns:
        df['is_gap_up'] = df['gap_percent_15m_spy'] > 0.5
        df['is_gap_down'] = df['gap_percent_15m_spy'] < -0.5
        df['is_large_gap'] = df['gap_percent_15m_spy'].abs() > 2.0
        
        # Breakout detection
        if '15m_high_spy' in df.columns and '15m_low_spy' in df.columns:
            df['is_breakout_up'] = df['next_15m_high_1_spy'] > df['15m_high_spy'].rolling(20).max()
            df['is_breakout_down'] = df['next_15m_low_1_spy'] < df['15m_low_spy'].rolling(20).min()
    
    # Volume events
    if '15m_volume_spy' in df.columns:
        vol_sma = df['15m_volume_spy'].rolling(20).mean()
        vol_std = df['15m_volume_spy'].rolling(20).std()
        df['is_volume_spike'] = (df['15m_volume_spy'] > (vol_sma + 2 * vol_std))
        df['is_volume_crush'] = (df['15m_volume_spy'] < (vol_sma - 2 * vol_std))
    
    # Sentiment events
    if 'sentiment_score' in df.columns:
        sent_mean = df['sentiment_score'].rolling(10).mean()
        sent_std = df['sentiment_score'].rolling(10).std()
        df['is_high_sentiment'] = df['sentiment_score'] > (sent_mean + sent_std)
        df['is_low_sentiment'] = df['sentiment_score'] < (sent_mean - sent_std)
        df['is_sentiment_change'] = df['sentiment_score'].diff().abs() > 0.5
    
    # Technical events
    if 'rsi_pre_spy' in df.columns:
        df['is_rsi_overbought'] = df['rsi_pre_spy'] > 70
        df['is_rsi_oversold'] = df['rsi_pre_spy'] < 30
        
        if '15m_close_spy' in df.columns:
            ema_short = df['15m_close_spy'].ewm(span=12).mean()
            ema_long = df['15m_close_spy'].ewm(span=26).mean()
            df['is_macd_bullish'] = ema_short > ema_long
            df['is_macd_bearish'] = ema_short < ema_long
    
    logger.info(f"[OK] Added EVENT features: {len([col for col in df.columns if 'is_' in col])} features")
    
    return df

def add_context_features(df):
    """
    Додає CONTEXT фandчand до даandсету
    """
    logger.info(" Adding CONTEXT features...")
    
    df = df.copy()
    
    # Sentiment context
    if 'sentiment_score' in df.columns:
        df['sentiment_volatility'] = df['sentiment_score'].rolling(20).std()
        df['sentiment_trend_3d'] = df['sentiment_score'].diff(3).rolling(20).mean()
    
    # Macro context (placeholder - requires VIX data)
    df['vix_level'] = np.nan  # Will be filled with actual VIX data
    df['vix_change'] = np.nan
    df['rate_environment'] = np.nan
    df['macro_event_intensity'] = np.nan
    df['fear_greed_index'] = np.nan
    
    # News context
    if 'published_at' in df.columns:
        # News density (news per hour)
        hour_counts = df.groupby(df['published_at'].dt.hour).size()
        df['news_density'] = df['published_at'].dt.hour.map(hour_counts).fillna(0)
        
        # News frequency score
        df['news_frequency_score'] = df['published_at'].dt.hour.map(lambda x: 1 if 9 <= x <= 17 else 0.5)  # Higher during market hours
        
        # Breaking news flag (high sentiment + high volume)
        if 'sentiment_score' in df.columns and '15m_volume_spy' in df.columns:
            df['breaking_news_flag'] = ((df['sentiment_score'].abs() > 0.8) & 
                                       (df['15m_volume_spy'] > df['15m_volume_spy'].quantile(0.9)))
    
    logger.info(f"[OK] Added CONTEXT features: {len([col for col in df.columns if any(x in col for x in ['sentiment', 'vix', 'news', 'density', 'frequency', 'breaking'])])} features")
    
    return df


def calculate_rsi_with_trading_days(df, close_col, interval, ticker):
    """
    Роwithраховує RSI with урахуванням попереднandх торгових днandв
    
    Args:
        df: DataFrame with даними
        close_col: Наwithва колонки with prices closing
        interval: Таймфрейм ('15m', '60m', '1d')
        ticker: Тandкер
    
    Returns:
        pd.Series: RSI values with урахуванням торгових днandв
    """
    rsi_values = pd.Series(index=df.index, dtype=float)
    
    # Отримуємо свяand США for рокandв у даandсетand
    years = df['published_at'].dt.year.unique()
    all_holidays = []
    for year in years:
        all_holidays.extend(get_us_holidays(year))
    
    # Calculating RSI for кожного рядка
    for idx, row in df.iterrows():
        news_time = row['published_at']
        
        if interval == '1d':
            # Для whereнних candles - шукаємо 14 попереднandх торгових днandв
            trading_days = get_previous_trading_days(news_time.date(), 14, all_holidays)
            
            # Збираємо цandни closing for цand днand
            close_prices = []
            for trading_day in trading_days:
                # Шукаємо рядки with цandєю датою
                day_data = df[df['published_at'].dt.date == trading_day]
                if not day_data.empty and close_col in day_data.columns:
                    # Беремо осandнню цandну closing дня
                    day_close = day_data[close_col].dropna().iloc[-1] if not day_data[close_col].dropna().empty else None
                    if day_close is not None:
                        close_prices.append(day_close)
            
            # Calculating RSI якщо є досandтньо data
            if len(close_prices) >= 14:
                rsi_values[idx] = calculate_simple_rsi(close_prices)
                
        elif interval == '60m':
            # Для годинних candles - шукаємо 14 попереднandх годинних сесandй
            trading_sessions = get_previous_trading_sessions(news_time, 14)
            
            # Збираємо цandни closing for цand сесandї
            close_prices = []
            for session_time in trading_sessions:
                # Шукаємо рядки with цandєю годиною
                hour_data = df[(df['published_at'].dt.hour == session_time.hour) & 
                              (df['published_at'].dt.date == session_time.date())]
                if not hour_data.empty and close_col in hour_data.columns:
                    hour_close = hour_data[close_col].dropna().iloc[-1] if not hour_data[close_col].dropna().empty else None
                    if hour_close is not None:
                        close_prices.append(hour_close)
            
            # Calculating RSI якщо є досandтньо data
            if len(close_prices) >= 14:
                rsi_values[idx] = calculate_simple_rsi(close_prices)
                
        elif interval == '15m':
            # Для 15-хвилинних candles - шукаємо 14 попереднandх 15-хвилинних candles
            # This приблиwithно 3.5 години попереднього торгового дня
            
            # Шукаємо осandннandй торговий whereнь
            last_trading_day = get_previous_trading_days(news_time.date(), 1, all_holidays)[0]
            
            # Збираємо 15-хвилиннand свandчки with осandннього торгового дня
            day_data = df[df['published_at'].dt.date == last_trading_day]
            
            if not day_data.empty and close_col in day_data.columns:
                # Беремо осandннand 14 candles дня
                day_closes = day_data[close_col].dropna().tail(14)
                
                if len(day_closes) >= 14:
                    rsi_values[idx] = calculate_simple_rsi(day_closes.tolist())
    
    return rsi_values


def calculate_simple_rsi(prices, period=14):
    """
    Простий роwithрахунок RSI for списку цandн
    
    Args:
        prices: Список цandн closing
        period: Перandод RSI
    
    Returns:
        float: RSI values
    """
    if len(prices) < period:
        return np.nan
    
    prices = np.array(prices)
    deltas = np.diff(prices)
    
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    
    avg_gain = np.mean(gains[-period:])
    avg_loss = np.mean(losses[-period:])
    
    if avg_loss == 0:
        return 100.0
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    return rsi


def add_pre_post_features(df):
    """
    Додає PRE and POST фandчand до даandсету
    """
    logger.info("[DATA] Adding PRE/POST phase features...")
    
    df = df.copy()
    
    # PRE phase features (before news) - роwithраховуємо for кожного тandкера/andймфрейму
    for interval in ['15m', '60m', '1d']:
        for ticker in ['spy', 'qqq', 'tsla', 'nvda']:
            ticker_upper = ticker.upper()
            
            # Баwithовand колонки for цього тandкера/andнтервалу
            close_col = f"{interval}_{ticker}_close"
            open_col = f"{interval}_{ticker}_open"
            high_col = f"{interval}_{ticker}_high"
            low_col = f"{interval}_{ticker}_low"
            volume_col = f"{interval}_{ticker}_volume"
            
            if all(col in df.columns for col in [close_col, open_col, high_col, low_col, volume_col]):
                # Volume relative to SMA
                df[f'{ticker}_{interval}_vol_rel_pre'] = df[volume_col] / df[volume_col].rolling(20).mean()
                
                # ATR relative to close
                df[f'{ticker}_{interval}_atr_rel_pre'] = (df[high_col].rolling(14).max() - df[low_col].rolling(14).min()) / df[close_col]
                
                # Distance to EMA
                df[f'{ticker}_{interval}_dist_to_ema_pre'] = (df[close_col] - df[close_col].ewm(span=20).mean()) / df[close_col]
                
                # RSI - правильний роwithрахунок по свandчках
                rsi_col = f'{ticker}_{interval}_rsi_pre'
                
                # Створюємо DataFrame with даними candles for цього тandкера/andнтервалу
                candle_data = pd.DataFrame({
                    'close': df[close_col],
                    'original_index': df.index
                }).dropna()
                
                if len(candle_data) >= 14:
                    # Calculating RSI по sequential свandчках
                    candle_data = candle_data.sort_values('original_index')
                    candle_data['rsi'] = calculate_rsi_on_candles(candle_data['close'])
                    
                    # Поверandємо RSI в оригandнальний DataFrame
                    rsi_map = dict(zip(candle_data['original_index'], candle_data['rsi']))
                    df[rsi_col] = df.index.map(rsi_map)
                else:
                    df[rsi_col] = np.nan
    
    # POST phase features (after news) - роwithраховуємо for кожного тandкера/andймфрейму
    for interval in ['15m', '60m', '1d']:
        for ticker in ['spy', 'qqq', 'tsla', 'nvda']:
            # Баwithовand колонки for цього тandкера/andнтервалу
            next_close_1_col = f"next_{interval}_{ticker}_close_1"
            next_open_1_col = f"next_{interval}_{ticker}_open_1"
            next_high_1_col = f"next_{interval}_{ticker}_high_1"
            next_low_1_col = f"next_{interval}_{ticker}_low_1"
            next_close_2_col = f"next_{interval}_{ticker}_close_2"
            next_open_2_col = f"next_{interval}_{ticker}_open_2"
            next_high_2_col = f"next_{interval}_{ticker}_high_2"
            next_low_2_col = f"next_{interval}_{ticker}_low_2"
            next_volume_1_col = f"next_{interval}_{ticker}_volume_1"
            next_volume_2_col = f"next_{interval}_{ticker}_volume_2"
            
            if all(col in df.columns for col in [next_close_1_col, next_open_1_col, next_high_1_col, next_low_1_col]):
                # Gap percentage
                df[f'{ticker}_{interval}_gap_percent'] = ((df[next_open_1_col] - df[f'{interval}_{ticker}_close']) / df[f'{interval}_{ticker}_close'] * 100).fillna(0)
                
                # Impact 1 percentage
                df[f'{ticker}_{interval}_impact_1_pct'] = ((df[next_close_1_col] - df[next_open_1_col]) / df[next_open_1_col] * 100).fillna(0)
                
                # Volume impact 1
                df[f'{ticker}_{interval}_vol_impact_1'] = (df[next_volume_1_col] / df[f'{interval}_{ticker}_volume'].rolling(20).mean() - 1) * 100
                
                # Shadow ratio 1
                range_total = df[next_high_1_col] - df[next_low_1_col]
                body = abs(df[next_close_1_col] - df[next_open_1_col])
                df[f'{ticker}_{interval}_shadow_ratio_1'] = (range_total / body).where(body > 0, np.nan)
                
                # Impact 2 percentage
                if next_close_2_col in df.columns and next_open_2_col in df.columns:
                    df[f'{ticker}_{interval}_impact_2_pct'] = ((df[next_close_2_col] - df[next_open_2_col]) / df[next_open_2_col] * 100).fillna(0)
                    
                    # Reversal score
                    df[f'{ticker}_{interval}_reversal_score'] = np.where(
                        (df[f'{ticker}_{interval}_impact_1_pct'] * df[f'{ticker}_{interval}_impact_2_pct'] < 0) & 
                        (df[f'{ticker}_{interval}_impact_1_pct'].notna()) & 
                        (df[f'{ticker}_{interval}_impact_2_pct'].notna()),
                        1, 0
                    )
                    
                    # Volume trend
                    if next_volume_2_col in df.columns:
                        df[f'{ticker}_{interval}_vol_trend'] = df[next_volume_2_col] / df[next_volume_1_col]
    
    logger.info(f"[OK] Added PRE/POST features: {len([col for col in df.columns if any(x in col for x in ['_pre', '_gap', '_impact', '_reversal', '_vol_trend'])])} features")
    
    return df
