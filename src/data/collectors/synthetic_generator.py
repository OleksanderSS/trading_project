import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class SyntheticMarketGenerator:
    """
    Генерує синтетичні ринкові сценарії для стрес-тестування (Flash Crash, Parabolic Run, etc.)
    """
    def generate_flash_crash(self, base_df, drop_percent=0.1, duration_bars=5):
        df = base_df.copy()
        crash_start = len(df) // 2
        for i in range(duration_bars):
            df.iloc[crash_start + i, df.columns.get_loc('close')] *= (1 - drop_percent / duration_bars)
        return df

    def generate_high_volatility(self, base_df, noise_level=0.05):
        df = base_df.copy()
        df['close'] *= (1 + np.random.normal(0, noise_level, len(df)))
        return df
