# utils/gdelt_signals.py

import pandas as pd


def compute_impulse_from_gdelt(df: pd.DataFrame, strong:int=10, medium:int=5) -> pd.DataFrame:
    """
    Агрегує withгадки по date/ticker and класифandкує силу сигналу:
      - >= strong: 1.0
      - >= medium: 0.5
      - >= 1: 0.1
      - 0: 0.0
    """
    if df is None or df.empty or "date" not in df.columns or "ticker" not in df.columns:
        return pd.DataFrame(columns=["date","ticker","mention_score","signal_strength","gdelt_weighted"])

    counts = df.groupby(["date", "ticker"])["mention_score"].sum().reset_index()

    def classify(count: int) -> float:
        if count >= strong: return 1.0
        if count >= medium: return 0.7
        if count >= 5: return 0.3
        if count >= 1: return 0.1
        return 0.0

    counts["signal_strength"] = counts["mention_score"].apply(classify)
    # Поки беwith forтухання по новинах: вага = сила сигналу на whereнь
    counts["gdelt_weighted"] = counts["signal_strength"]
    return counts