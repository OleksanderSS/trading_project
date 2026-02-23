# config/technical_config.py

"""
Конфігурація для технічних індикаторів.
"""

TECHNICAL_WINDOWS = {
    # БАЗОВІ СЕРЕДНІ
    "sma": [14, 50, 200],           # Simple Moving Averages
    "ema": [12, 26],                # Exponential Moving Averages
    
    # АДАПТИВНІ СЕРЕДНІ (НОВІ)
    "kama": [20],                   # Kaufman's Adaptive Moving Average
    "hma": [20],                    # Hull Moving Average
    
    # МОМЕНТУМ
    "rsi": [14],                    # Relative Strength Index
    "macd": [(12, 26, 9)],          # MACD parameters
    "roc": [10],                    # Rate of Change
    "ao": [(5, 34)],                # Awesome Oscillator
    "ac": [(5, 34)],                # Accelerator Oscillator
    "demark": [13],                 # DeMarker Indicator
    "fisher": [10],                 # Fisher Transform
    
    # ВОЛАТИЛЬНІСТЬ
    "atr": [14],                    # Average True Range
    "bollinger": [(20, 2)],         # Bollinger Bands (window, std_dev)
    "donchian": [20],               # Donchian Channels
    "keltner": [(20, 2)],           # Keltner Channels (window, multiplier)
    "historical_vol": [20],         # Historical Volatility
    
    # ОСЦИЛЯТОРИ
    "stochastic": [(14, 3)],       # Stochastic (k_window, d_window)
    "williams_r": [14],             # Williams %R
    "cci": [20],                    # Commodity Channel Index
    "trix": [15],                   # TRIX
    "mass_index": [(9, 25)],       # Mass Index (ema_window, sum_window)
    
    # ТРЕНД
    "adx": [14],                    # Average Directional Index
    
    # ОБСЯГ
    "obv": [],                      # On-Balance Volume
    "mfi": [14],                    # Money Flow Index
    "cmf": [20],                    # Chaikin Money Flow
    "vpt": [],                      # Volume Price Trend
    "vwap": [],                     # Volume Weighted Average Price
    
    # ГЕПИ
    "gap": [],                      # Gap features
}

# ПРІОРИТЕТНІ ІНДИКАТОРИ ДЛЯ ВИКОРИСТАННЯ
PRIORITY_INDICATORS = {
    "trend": ["SMA_50", "EMA_12", "KAMA", "HMA", "ADX"],
    "momentum": ["RSI_14", "MACD_12_26_9", "AO", "AC", "DeMarker"],
    "volatility": ["ATR", "Bollinger", "Donchian", "Keltner"],
    "volume": ["OBV", "CMF", "VWAP", "MFI"],
    "oscillators": ["Stochastic", "Williams_R", "CCI", "Fisher"]
}

# ОПТИМАЛЬНІ ПАРАМЕТРИ ДЛЯ РІЗНИХ ТАЙМФРЕЙМІВ
TIMEFRAME_OPTIMIZATION = {
    "5m": {                         # Ультра-короткий - тільки базові індикатори
        "rsi": [14],               # Стандартний RSI (коротші занадто шумові)
        "sma": [20],               # Один SMA (багато SMA = шум)
        "ema": [12],               # Один EMA (швидкий але стабільний)
        "atr": [14],               # Стандартний ATR
        "bollinger": [(20, 2)],   # Bollinger Bands (стабільні)
        "macd": [(12, 26, 9)],    # MACD (стандартний)
        "adx": [14],               # ADX (тренд)
        "obv": [],                 # OBV (об'єм)
        "vwap": [],               # VWAP (ціна/об'єм)
        # ВИКЛЮЧАЄМО: stochastic, williams_r, cci (занадто шумові для 5m)
        # ВИКЛЮЧАЄМО: kama, hma, trix, mass_index (занадто складні для 5m)
    },
    "15m": {                        # Інтрадей - швидкі сигнали
        "rsi": [7, 14],            # Коротші періоди для швидкості
        "sma": [10, 20],           # Коротші середні
        "ema": [5, 10],            # Швидкі EMA
        "atr": [7, 14],            # Коротший ATR
        "kama": [10],              # Швидший KAMA
        "hma": [10],               # Швидший HMA
        "stochastic": [(10, 3)],   # Швидший stochastic
        "williams_r": [10],        # Коротший Williams %R
        "cci": [10],               # Швидкий CCI
        "bollinger": [(20, 2)],   # Bollinger Bands
        "macd": [(12, 26, 9)],    # MACD
        "adx": [14],               # ADX тренд
        "obv": [],                 # OBV об'єм
        "mfi": [14],               # Money Flow Index
        "vwap": [],               # VWAP ціна/об'єм
        "donchian": [20],          # Donchian канали
        "keltner": [(20, 2)],     # Keltner канали
        "ao": [(5, 34)],          # Awesome Oscillator
        "demark": [13],            # DeMarker
        # ВИКЛЮЧАЄМО: trix, mass_index, fisher (занадто складні для 15m)
    },
    "60m": {                        # Середньострок - баланс
        "rsi": [14],               # Стандартний RSI
        "sma": [20, 50],           # Середні середні
        "ema": [12, 26],           # Стандартні EMA
        "atr": [14],               # Стандартний ATR
        "kama": [20],              # Стандартний KAMA
        "hma": [20],               # Стандартний HMA
        "stochastic": [(14, 3)],   # Стандартний stochastic
        "williams_r": [14],        # Стандартний Williams %R
        "cci": [20],               # Стандартний CCI
        "bollinger": [(20, 2)],   # Bollinger Bands
        "macd": [(12, 26, 9)],    # MACD
        "adx": [14],               # ADX тренд
        "obv": [],                 # OBV об'єм
        "mfi": [14],               # Money Flow Index
        "vwap": [],               # VWAP ціна/об'єм
        "donchian": [20],          # Donchian канали
        "keltner": [(20, 2)],     # Keltner канали
        "ao": [(5, 34)],          # Awesome Oscillator
        "demark": [13],            # DeMarker
        "trix": [15],             # TRIX (для 60m вже нормально)
        # ВИКЛЮЧАЄМО: mass_index, fisher (дуже складні)
    },
    "1d": {                         # Довгострок - стабільність
        "rsi": [14],               # Стандартний RSI
        "sma": [50, 200],          # Довгі середні
        "ema": [12, 26],           # Стандартні EMA
        "atr": [14],               # Стандартний ATR
        "kama": [20],              # Стандартний KAMA
        "hma": [20],               # Стандартний HMA
        "stochastic": [(14, 3)],   # Стандартний stochastic
        "williams_r": [14],        # Стандартний Williams %R
        "cci": [20],               # Стандартний CCI
        "bollinger": [(20, 2)],   # Bollinger Bands
        "macd": [(12, 26, 9)],    # MACD
        "adx": [14],               # ADX тренд
        "obv": [],                 # OBV об'єм
        "mfi": [14],               # Money Flow Index
        "vwap": [],               # VWAP ціна/об'єм
        "donchian": [20],          # Donchian канали
        "keltner": [(20, 2)],     # Keltner канали
        "ao": [(5, 34)],          # Awesome Oscillator
        "demark": [13],            # DeMarker
        "trix": [15],             # TRIX
        "mass_index": [(9, 25)],   # Mass Index (для денних OK)
        "fisher": [10],            # Fisher Transform (для денних OK)
        # ВСІ індикатори для денних даних
    }
}