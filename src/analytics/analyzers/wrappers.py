from typing import Any, Dict

import pandas as pd

from src.analytics.interfaces import IAnalyzer
from src.analytics.calculators.drawdown_calculator import DrawdownCalculator
from src.analytics.calculators.fama_french_factors import FamaFrenchFactors
from src.analytics.calculators.volatility_calculator import VolatilityCalculator


class DrawdownAnalyzer(IAnalyzer):
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}

    def analyze(self, data: Any, **kwargs) -> Dict[str, Any]:
        price_col = kwargs.get("price_col", self.config.get("price_col", "close"))
        high_col = kwargs.get("high_col", self.config.get("high_col", "high"))
        window = int(kwargs.get("window", self.config.get("window", 20)))
        if not isinstance(data, pd.DataFrame):
            raise TypeError("DrawdownAnalyzer expects a pandas DataFrame")

        calc = DrawdownCalculator()
        drawdown = calc.calculate_max_drawdown_from_prices(data, price_col=price_col, high_col=high_col)
        rolling_drawdown = calc.calculate_rolling_drawdown(data, window=window, price_col=price_col, high_col=high_col)
        underwater_duration = calc.calculate_underwater_duration(data, price_col=price_col, high_col=high_col)

        clean_drawdown = drawdown.dropna()
        return {
            "drawdown": drawdown,
            "rolling_drawdown": rolling_drawdown,
            "underwater_duration": underwater_duration,
            "max_drawdown": float(clean_drawdown.min()) if not clean_drawdown.empty else 0.0,
            "window": window,
        }


class FamaFrenchAnalyzer(IAnalyzer):
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.calculator = FamaFrenchFactors()

    def analyze(self, data: Any, **kwargs) -> Dict[str, Any]:
        start_date = kwargs.get("start_date", self.config.get("start_date", "2020-01-01"))
        end_date = kwargs.get("end_date", self.config.get("end_date", "2020-12-31"))
        return {
            "factors": self.calculator.get_factors(start_date, end_date),
            "start_date": start_date,
            "end_date": end_date,
        }


class VolatilityAnalyzer(IAnalyzer):
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}

    def analyze(self, data: Any, **kwargs) -> Dict[str, Any]:
        window = int(kwargs.get("window", self.config.get("window", 20)))
        periods_per_year = int(kwargs.get("periods_per_year", self.config.get("periods_per_year", 252)))
        returns = self._extract_returns(data, kwargs)

        calc = VolatilityCalculator()
        rolling_volatility = calc.calculate_rolling_volatility(
            returns,
            window=window,
            periods_per_year=periods_per_year,
        )
        realized_volatility = calc.calculate_realized_volatility(
            returns,
            window=window,
            periods_per_year=periods_per_year,
        )
        clean_volatility = rolling_volatility.dropna()
        return {
            "rolling_volatility": rolling_volatility,
            "realized_volatility": realized_volatility,
            "latest_volatility": float(clean_volatility.iloc[-1]) if not clean_volatility.empty else 0.0,
            "window": window,
            "periods_per_year": periods_per_year,
        }

    def _extract_returns(self, data: Any, kwargs: Dict[str, Any]) -> pd.Series:
        if isinstance(data, pd.Series):
            return data.astype(float)
        if not isinstance(data, pd.DataFrame):
            raise TypeError("VolatilityAnalyzer expects a pandas Series or DataFrame")

        returns_col = kwargs.get("returns_col", self.config.get("returns_col"))
        if returns_col and returns_col in data.columns:
            return data[returns_col].astype(float)

        price_col = kwargs.get("price_col", self.config.get("price_col", "close"))
        if price_col not in data.columns:
            raise ValueError(f"Neither returns_col nor price column '{price_col}' found")

        return data[price_col].astype(float).pct_change(fill_method=None).dropna()
