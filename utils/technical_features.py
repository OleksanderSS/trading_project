# utils/technical_features.py

import pandas as pd
import numpy as np
import ta
from utils.logger import ProjectLogger
from config.technical_config import TECHNICAL_WINDOWS

logger = ProjectLogger.get_logger("TradingProjectLogger")

def _calculate_ta_for_group(df: pd.DataFrame) -> pd.DataFrame:
    """Розраховує всі технічні індикатори для одного DataFrame."""
    required_cols = ['open', 'high', 'low', 'close', 'volume']
    if not all(col in df.columns for col in required_cols):
        logger.warning(f"[technical_features] Пропуск, відсутні необхідні колонки.")
        return df

    df_ta = df.copy()

    # Basic Indicators
    for w in TECHNICAL_WINDOWS.get("sma", [14, 50, 200]):
        df_ta[f"SMA_{w}"] = df_ta["close"].rolling(window=w, min_periods=1).mean()
    for w in TECHNICAL_WINDOWS.get("ema", [12, 26]):
        df_ta[f"EMA_{w}"] = df_ta["close"].ewm(span=w, adjust=False, min_periods=1).mean()
    for w in TECHNICAL_WINDOWS.get("rsi", [14]):
        df_ta[f"RSI_{w}"] = ta.momentum.RSIIndicator(close=df_ta["close"], window=w).rsi()
    for slow, fast, sign in TECHNICAL_WINDOWS.get("macd", [(26, 12, 9)]):
        macd = ta.trend.MACD(close=df_ta["close"], window_slow=slow, window_fast=fast, window_sign=sign)
        df_ta[f'MACD_{slow}_{fast}_{sign}'] = macd.macd()
        df_ta[f'MACD_signal_{slow}_{fast}_{sign}'] = macd.macd_signal()
        df_ta[f'MACD_diff_{slow}_{fast}_{sign}'] = macd.macd_diff()

    df_ta['gap_size'] = df_ta['open'] - df_ta['close'].shift(1)
    df_ta['gap_size_pct'] = df_ta['gap_size'] / df_ta['close'].shift(1)
    df_ta['gap_signal'] = np.select(
        [df_ta['gap_size_pct'] > 0.01, df_ta['gap_size_pct'] < -0.01], [1, -1], default=0
    )

    # Advanced Indicators
    bb_indicator = ta.volatility.BollingerBands(close=df_ta['close'], window=20, window_dev=2)
    df_ta['BB_upper'] = bb_indicator.bollinger_hband()
    df_ta['BB_middle'] = bb_indicator.bollinger_mavg()
    df_ta['BB_lower'] = bb_indicator.bollinger_lband()
    df_ta['ATR'] = ta.volatility.AverageTrueRange(high=df_ta['high'], low=df_ta['low'], close=df_ta['close'], window=14).average_true_range()
    stoch_indicator = ta.momentum.StochasticOscillator(high=df_ta['high'], low=df_ta['low'], close=df_ta['close'], window=14, smooth_window=3)
    df_ta['Stoch_K'] = stoch_indicator.stoch()
    df_ta['Stoch_D'] = stoch_indicator.stoch_signal()
    df_ta['Williams_R'] = ta.momentum.WilliamsRIndicator(high=df_ta['high'], low=df_ta['low'], close=df_ta['close'], lbp=14).williams_r()
    df_ta['CCI'] = ta.trend.CCIIndicator(high=df_ta['high'], low=df_ta['low'], close=df_ta['close'], window=20).cci()
    df_ta['OBV'] = ta.volume.OnBalanceVolumeIndicator(close=df_ta['close'], volume=df_ta['volume']).on_balance_volume()
    adx_indicator = ta.trend.ADXIndicator(high=df_ta['high'], low=df_ta['low'], close=df_ta['close'], window=14)
    df_ta['ADX'] = adx_indicator.adx()
    df_ta['DI_plus'] = adx_indicator.adx_pos()
    df_ta['DI_minus'] = adx_indicator.adx_neg()
    df_ta['MFI'] = ta.volume.MFIIndicator(high=df_ta['high'], low=df_ta['low'], close=df_ta['close'], volume=df_ta['volume'], window=14).money_flow_index()
    df_ta['VPT'] = ta.volume.VolumePriceTrendIndicator(close=df_ta['close'], volume=df_ta['volume']).volume_price_trend()
    df_ta['TRIX'] = ta.trend.TRIXIndicator(close=df_ta['close'], window=15).trix()
    df_ta['Mass_Index'] = ta.trend.MassIndex(high=df_ta['high'], low=df_ta['low']).mass_index()
    vortex_indicator = ta.trend.VortexIndicator(high=df_ta['high'], low=df_ta['low'], close=df_ta['close'], window=14)
    df_ta['Vortex_Plus'] = vortex_indicator.vortex_indicator_pos()
    df_ta['Vortex_Minus'] = vortex_indicator.vortex_indicator_neg()
    kst_indicator = ta.trend.KSTIndicator(close=df_ta['close'])
    df_ta['KST'] = kst_indicator.kst()
    df_ta['KST_Signal'] = kst_indicator.kst_sig()
    df_ta['Ultimate_Osc'] = ta.momentum.UltimateOscillator(high=df_ta['high'], low=df_ta['low'], close=df_ta['close']).ultimate_oscillator()
    df_ta['DPO'] = ta.trend.DPOIndicator(close=df_ta['close'], window=20).dpo()
    kc_indicator = ta.volatility.KeltnerChannel(high=df_ta['high'], low=df_ta['low'], close=df_ta['close'])
    df_ta['KC_Upper'] = kc_indicator.keltner_channel_hband()
    df_ta['KC_Middle'] = kc_indicator.keltner_channel_mband()
    df_ta['KC_Lower'] = kc_indicator.keltner_channel_lband()
    dc_indicator = ta.volatility.DonchianChannel(high=df_ta['high'], low=df_ta['low'], close=df_ta['close'])
    df_ta['DC_Upper'] = dc_indicator.donchian_channel_hband()
    df_ta['DC_Middle'] = dc_indicator.donchian_channel_mband()
    df_ta['DC_Lower'] = dc_indicator.donchian_channel_lband()
    ichimoku_indicator = ta.trend.IchimokuIndicator(high=df_ta['high'], low=df_ta['low'])
    df_ta['Ichimoku_Tenkan'] = ichimoku_indicator.ichimoku_conversion_line()
    df_ta['Ichimoku_Kijun'] = ichimoku_indicator.ichimoku_base_line()
    df_ta['Ichimoku_Senkou_A'] = ichimoku_indicator.ichimoku_a()
    df_ta['Ichimoku_Senkou_B'] = ichimoku_indicator.ichimoku_b()

    # Heikin-Ashi Calculation (vectorized approximation)
    df_ta['HA_Close'] = (df_ta['open'] + df_ta['high'] + df_ta['low'] + df_ta['close']) / 4
    # Approximate HA_Open for vectorization. Correct for all but the first value in a group.
    # The first value is handled by fillna, taking its own open/close average.
    ha_open_shifted = (df_ta['open'].shift(1) + df_ta['close'].shift(1)) / 2
    df_ta['HA_Open'] = ha_open_shifted.fillna((df_ta['open'] + df_ta['close']) / 2)
    df_ta['HA_High'] = df_ta[['high', 'HA_Open', 'HA_Close']].max(axis=1)
    df_ta['HA_Low'] = df_ta[['low', 'HA_Open', 'HA_Close']].min(axis=1)


    return df_ta

def add_all_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Додає всі технічні ознаки, групуючи за тікером та інтервалом."""
    if 'ticker' not in df.columns or 'interval' not in df.columns:
        logger.error("[technical_features] ❌ Для розрахунку індикаторів потрібні колонки 'ticker' та 'interval'")
        return df
    if df.empty:
        logger.warning("[technical_features] ⚠️ Початковий DataFrame порожній.")
        return df

    # Створюємо список для зберігання оброблених груп
    processed_groups = []

    # Групуємо за тікером та інтервалом
    grouped = df.groupby(['ticker', 'interval'])

    # Обробляємо кожну групу окремо
    for name, group in grouped:
        # Розраховуємо технічні індикатори для групи
        processed_group = _calculate_ta_for_group(group)
        # Додаємо оброблену групу до списку
        processed_groups.append(processed_group)

    # Об'єднуємо всі оброблені групи в один DataFrame
    df_featured = pd.concat(processed_groups, ignore_index=True)
    
    logger.info(f"[technical_features] ✅ Added all technical features. Original columns: {len(df.columns)}, New columns: {len(df_featured.columns)}")
    return df_featured
