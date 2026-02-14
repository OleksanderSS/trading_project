# utils/technical_features.py

import pandas as pd
import numpy as np
import ta
from utils.logger import ProjectLogger
from config.technical_config import TECHNICAL_WINDOWS

logger = ProjectLogger.get_logger("TradingProjectLogger")

def _apply_all_ta(df: pd.DataFrame) -> pd.DataFrame:
    """Розраховує всі технічні індикатори для одного DataFrame."""
    required_cols = ['open', 'high', 'low', 'close', 'volume']
    if not all(col in df.columns for col in required_cols):
        logger.warning(f"[technical_features] Пропуск, відсутні необхідні колонки.")
        return df

    # Basic Indicators
    for w in TECHNICAL_WINDOWS.get("sma", [14, 50, 200]):
        df[f"SMA_{w}"] = df["close"].rolling(window=w, min_periods=1).mean()
    for w in TECHNICAL_WINDOWS.get("ema", [12, 26]):
        df[f"EMA_{w}"] = df["close"].ewm(span=w, adjust=False, min_periods=1).mean()
    for w in TECHNICAL_WINDOWS.get("rsi", [14]):
        df[f"RSI_{w}"] = ta.momentum.RSIIndicator(close=df["close"], window=w).rsi()
    for slow, fast, sign in TECHNICAL_WINDOWS.get("macd", [(26, 12, 9)]):
        macd = ta.trend.MACD(close=df["close"], window_slow=slow, window_fast=fast, window_sign=sign)
        df[f'MACD_{slow}_{fast}_{sign}'] = macd.macd()
        df[f'MACD_signal_{slow}_{fast}_{sign}'] = macd.macd_signal()
        df[f'MACD_diff_{slow}_{fast}_{sign}'] = macd.macd_diff()

    df['gap_size'] = df['open'] - df['close'].shift(1)
    df['gap_size_pct'] = df['gap_size'] / df['close'].shift(1)
    df['gap_signal'] = np.select(
        [df['gap_size_pct'] > 0.01, df['gap_size_pct'] < -0.01], [1, -1], default=0
    )

    # Advanced Indicators
    bb_indicator = ta.volatility.BollingerBands(close=df['close'], window=20, window_dev=2)
    df['BB_upper'] = bb_indicator.bollinger_hband()
    df['BB_middle'] = bb_indicator.bollinger_mavg()
    df['BB_lower'] = bb_indicator.bollinger_lband()
    df['ATR'] = ta.volatility.AverageTrueRange(high=df['high'], low=df['low'], close=df['close'], window=14).average_true_range()
    stoch_indicator = ta.momentum.StochasticOscillator(high=df['high'], low=df['low'], close=df['close'], window=14, smooth_window=3)
    df['Stoch_K'] = stoch_indicator.stoch()
    df['Stoch_D'] = stoch_indicator.stoch_signal()
    df['Williams_R'] = ta.momentum.WilliamsRIndicator(high=df['high'], low=df['low'], close=df['close'], lbp=14).williams_r()
    df['CCI'] = ta.trend.CCIIndicator(high=df['high'], low=df['low'], close=df['close'], window=20).cci()
    df['OBV'] = ta.volume.OnBalanceVolumeIndicator(close=df['close'], volume=df['volume']).on_balance_volume()
    adx_indicator = ta.trend.ADXIndicator(high=df['high'], low=df['low'], close=df['close'], window=14)
    df['ADX'] = adx_indicator.adx()
    df['DI_plus'] = adx_indicator.adx_pos()
    df['DI_minus'] = adx_indicator.adx_neg()
    df['MFI'] = ta.volume.MFIIndicator(high=df['high'], low=df['low'], close=df['close'], volume=df['volume'], window=14).money_flow_index()
    df['VPT'] = ta.volume.VolumePriceTrendIndicator(close=df['close'], volume=df['volume']).volume_price_trend()
    df['TRIX'] = ta.trend.TRIXIndicator(close=df['close'], window=15).trix()
    df['Mass_Index'] = ta.trend.MassIndex(high=df['high'], low=df['low']).mass_index()
    vortex_indicator = ta.trend.VortexIndicator(high=df['high'], low=df['low'], close=df['close'], window=14)
    df['Vortex_Plus'] = vortex_indicator.vortex_indicator_pos()
    df['Vortex_Minus'] = vortex_indicator.vortex_indicator_neg()
    kst_indicator = ta.trend.KSTIndicator(close=df['close'])
    df['KST'] = kst_indicator.kst()
    df['KST_Signal'] = kst_indicator.kst_sig()
    df['Ultimate_Osc'] = ta.momentum.UltimateOscillator(high=df['high'], low=df['low'], close=df['close']).ultimate_oscillator()
    df['DPO'] = ta.trend.DPOIndicator(close=df['close'], window=20).dpo()
    kc_indicator = ta.volatility.KeltnerChannel(high=df['high'], low=df['low'], close=df['close'])
    df['KC_Upper'] = kc_indicator.keltner_channel_hband()
    df['KC_Middle'] = kc_indicator.keltner_channel_mband()
    df['KC_Lower'] = kc_indicator.keltner_channel_lband()
    dc_indicator = ta.volatility.DonchianChannel(high=df['high'], low=df['low'], close=df['close'])
    df['DC_Upper'] = dc_indicator.donchian_channel_hband()
    df['DC_Middle'] = dc_indicator.donchian_channel_mband()
    df['DC_Lower'] = dc_indicator.donchian_channel_lband()
    ichimoku_indicator = ta.trend.IchimokuIndicator(high=df['high'], low=df['low'])
    df['Ichimoku_Tenkan'] = ichimoku_indicator.ichimoku_conversion_line()
    df['Ichimoku_Kijun'] = ichimoku_indicator.ichimoku_base_line()
    df['Ichimoku_Senkou_A'] = ichimoku_indicator.ichimoku_a()
    df['Ichimoku_Senkou_B'] = ichimoku_indicator.ichimoku_b()

    # Heikin-Ashi Calculation (vectorized approximation)
    df['HA_Close'] = (df['open'] + df['high'] + df['low'] + df['close']) / 4
    # Approximate HA_Open for vectorization. Correct for all but the first value in a group.
    # The first value is handled by fillna, taking its own open/close average.
    ha_open_shifted = (df['open'].shift(1) + df['close'].shift(1)) / 2
    df['HA_Open'] = ha_open_shifted.fillna((df['open'] + df['close']) / 2)
    df['HA_High'] = df[['high', 'HA_Open', 'HA_Close']].max(axis=1)
    df['HA_Low'] = df[['low', 'HA_Open', 'HA_Close']].min(axis=1)


    return df

def add_all_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Додає всі технічні ознаки, групуючи за тікером."""
    if 'ticker' not in df.columns:
        logger.error("[technical_features] ❌ Для розрахунку індикаторів потрібна колонка 'ticker'")
        return df
    if df.empty:
        logger.warning("[technical_features] ⚠️ Початковий DataFrame порожній.")
        return df

    # Застосовуємо розрахунок індикаторів до кожної групи тікерів
    # The logic inside _apply_all_ta is designed to work on groups.
    df_featured = df.groupby('ticker').apply(_apply_all_ta)
    
    # Після apply індекс може стати багаторівневим, тому скидаємо його
    df_featured = df_featured.reset_index(drop=True)
    
    logger.info(f"[technical_features] ✅ Added all technical features. Original columns: {len(df.columns)}, New columns: {len(df_featured.columns)}")
    return df_featured
