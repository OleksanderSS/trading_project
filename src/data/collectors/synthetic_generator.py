import numpy as np

from .base_collector import BaseCollector


class SyntheticGenerator(BaseCollector):
    """Generates synthetic market scenarios for stress-testing.

    Supports Flash Crash, Parabolic Run, and other scenarios.
    """

    def generate_flash_crash(
        self, base_df, drop_percent=0.1, duration_bars=5
    ):
        df = base_df.copy()
        crash_start = len(df) // 2
        for i in range(duration_bars):
            idx = crash_start + i
            col_idx = df.columns.get_loc('close')
            df.iloc[idx, col_idx] *= (1 - drop_percent / duration_bars)
        return df

    def generate_high_volatility(
        self, base_df, noise_level=0.05
    ):
        df = base_df.copy()
        noise = np.random.normal(0, noise_level, len(df))
        df['close'] *= (1 + noise)
        return df
