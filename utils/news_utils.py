#!/usr/bin/env python3
"""Утиліти для обробки новин"""

import pandas as pd
from typing import Dict

def expand_news_to_all_tickers(df_news: pd.DataFrame, tickers: Dict) -> pd.DataFrame:
    """
    Розширює DataFrame з новинами, створюючи копію кожної новини для кожного тікера.
    Це дозволяє асоціювати загальні ринкові новини з кожним конкретним активом.

    Args:
        df_news (pd.DataFrame): DataFrame з новинами, де може бути колонка 'ticker'.
        tickers (Dict): Словник тікерів для розширення.

    Returns:
        pd.DataFrame: Розширений DataFrame.
    """
    if df_news.empty:
        return pd.DataFrame()

    expanded = []
    for _, row in df_news.iterrows():
        direct_ticker = row.get("ticker", "GENERAL")
        for t in tickers.keys():
            new_row = row.copy()
            new_row["ticker"] = t
            # Позначаємо, чи була новина специфічною для цього тікера
            new_row["is_direct_match"] = (t == direct_ticker)
            new_row["news_type"] = "DIRECT" if t == direct_ticker else "GENERAL"
            expanded.append(new_row)
            
    return pd.DataFrame(expanded)

