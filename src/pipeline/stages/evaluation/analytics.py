from typing import Any

import numpy as np
import pandas as pd

from src.core.logging.logger import ProjectLogger

logger = ProjectLogger.get_logger(__name__)


def calculate_financial_metrics(metrics_calculator, portfolio_history: pd.DataFrame) -> dict[str, Any]:
    if portfolio_history is None or portfolio_history.empty:
        return {"error": "empty_portfolio_history"}

    if "total_value" not in portfolio_history.columns:
        return {"error": "missing_total_value_column"}

    try:
        return metrics_calculator.calculate(portfolio_history["total_value"])
    except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError) as e:
        logger.exception(f"Error calculating financial metrics: {e}")
        return {"error": "calculation_failed", "details": str(e)}


def run_deep_analysis(
    analytics_engine,
    signals_df: pd.DataFrame,
    portfolio_history: pd.DataFrame,
    enriched_data: pd.DataFrame | None = None,
    brain: dict | None = None,
) -> dict[str, Any]:
    if analytics_engine is None:
        return {"error": "missing_analytics_engine"}

    data_map = _build_data_map(signals_df, portfolio_history, enriched_data, brain)

    try:
        return analytics_engine.run_full_analysis(data_map)
    except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError) as e:
        logger.exception(f"Error running deep analysis: {e}")
        return {"error": "analysis_failed", "details": str(e)}

def _build_data_map(signals_df, portfolio_history, enriched_data, brain) -> dict[str, Any]:
    price_data = signals_df[["price"]].copy() if "price" in signals_df.columns else pd.DataFrame()
    if "close" not in price_data.columns and "price" in price_data.columns:
        price_data["close"] = price_data["price"]
    if "volume" not in price_data.columns:
        price_data["volume"] = 0

    returns = pd.Series(dtype=float)
    if isinstance(portfolio_history, pd.DataFrame) and "returns" in portfolio_history.columns:
        returns = portfolio_history["returns"].dropna()

    market_data = pd.DataFrame(
        {
            "price": signals_df["price"] if "price" in signals_df.columns else 0,
            "volume": signals_df.get("volume", 0),
            "returns": returns,
        }
    )

    predictions = pd.Series(dtype=float)
    prediction_signal_available = pd.Series(dtype=int)
    if "signal" in signals_df.columns:
        mapped_signals = signals_df["signal"].map({"BUY": 1, "SELL": -1, "HOLD": 0})
        prediction_signal_available = mapped_signals.notna().astype(int)
        predictions = mapped_signals.where(mapped_signals.notna(), 0)

    return {
        "price_data": price_data,
        "market_data": market_data,
        "signals": signals_df["signal"] if "signal" in signals_df.columns else None,
        "returns": returns,
        "portfolio_returns": pd.DataFrame({"Strategy": returns}) if not returns.empty else pd.DataFrame(),
        "benchmark_returns": pd.DataFrame(
            {
                "Benchmark": price_data["close"]
                .pct_change(fill_method=None)
                .replace([np.inf, -np.inf], np.nan)
            }
        ) if "close" in price_data.columns else pd.DataFrame(),
        "portfolio_data": portfolio_history,
        "news_data": brain.get("news_data") if brain else None,
        "macro_data": brain.get("macro_data") if brain else None,
        "market_indicators": None,
        "economic_data": brain.get("macro_data") if brain else None,
        "historical_economic_data": brain.get("macro_data") if brain else None,
        "predictions": predictions,
        "prediction_signal_available": prediction_signal_available,
        "performance_metrics": {},
        "features_data": enriched_data if enriched_data is not None else pd.DataFrame(),
        "target_series": signals_df["signal"] if "signal" in signals_df.columns else pd.Series(dtype=float),
        "causal_series": price_data["close"] if "close" in price_data.columns else pd.Series(dtype=float),
        "models_metadata": brain.get("models_metadata", {}) if brain else {},
    }


def create_evaluation_summary(
    financial_metrics: dict[str, Any], backtest_results: dict[str, Any], analysis_results: dict[str, Any]
) -> dict[str, Any]:
    return {
        "metrics": financial_metrics,
        "backtest_stats": backtest_results.get("performance", {}),
        "analysis": analysis_results,
        "timestamp": pd.Timestamp.now().isoformat(),
    }
