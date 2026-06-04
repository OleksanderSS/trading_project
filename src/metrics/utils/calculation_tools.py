#!/usr/bin/env python3
"""
Calculation Tools - Універсальні допоміжні функції для розрахунку фінансових метрик та обробки часових рядів.
"""

import numpy as np
import pandas as pd

from src.core.logging.logger import ProjectLogger

logger = ProjectLogger.get_logger("CalculationTools")

def adjust_for_risk_free_rate(returns: pd.Series, rf_rate: float) -> pd.Series:
    """
    Коригує дохідність на безризикову ставку.

    Args:
        returns: Серія доходностей.
        rf_rate: Річна безризикова ставка (наприклад, 0.05 для 5%).

    Returns:
        pd.Series: Надлишкова дохідність (Excess Returns).
    """
    # Припускаємо що вхідні дані щоденні, тому ділимо річну ставку на кількість торгових днів
    daily_rf = (1 + rf_rate) ** (1/252) - 1
    return returns - daily_rf

def annualize_returns(returns: pd.Series, periods: int = 252) -> float:
    """
    Розраховує річну дохідність на основі періодичних даних.

    Args:
        returns: Серія доходностей за період.
        periods: Кількість періодів у році (252 для днів, 252*6.5*4 для 15м тощо).

    Returns:
        float: Річна дохідність.
    """
    if returns.empty:
        return 0.0

    total_return = (1 + returns).prod()
    n_periods = len(returns)

    if n_periods == 0:
        return 0.0

    return (total_return ** (periods / n_periods)) - 1

def calculate_rolling_volatility(returns: pd.Series, window: int = 20) -> pd.Series:
    """
    Розраховує ковзну волатильність.

    Args:
        returns: Серія доходностей.
        window: Вікно для розрахунку (кількість барів).

    Returns:
        pd.Series: Серія значень волатильності.
    """
    return returns.rolling(window=window, min_periods=1).std() * np.sqrt(252)

def calculate_drawdown_series(equity_curve: pd.Series) -> pd.Series:
    """
    Розраховує серію просідань для кривої капіталу.

    Args:
        equity_curve: Серія значень капіталу (Equity Curve).

    Returns:
        pd.Series: Серія відсоткових просідань від піку.
    """
    running_max = equity_curve.cummax()
    drawdown = (equity_curve - running_max) / running_max
    return drawdown
