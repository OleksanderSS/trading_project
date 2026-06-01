import asyncio
import logging
from functools import partial
from typing import Any, Dict

import pandas as pd

logger = logging.getLogger(__name__)


def prepare_pivot(signals_df: pd.DataFrame):
    # Minimal pivot logic: produce price_pivot and signal_pivot similar to previous implementation
    if "ticker" in signals_df.columns:
        if "timestamp" in signals_df.columns and signals_df["timestamp"].notna().any():
            price_pivot = signals_df.pivot_table(index="timestamp", columns="ticker", values="price", aggfunc="mean")
            sig_numeric = signals_df.copy()
            sig_numeric["sig_val"] = sig_numeric["signal"].map({"BUY": 1, "SELL": -1, "HOLD": 0})
            signal_pivot = sig_numeric.pivot_table(
                index="timestamp", columns="ticker", values="sig_val", aggfunc="mean"
            )
        else:
            price_agg = signals_df.groupby("ticker")["price"].mean()
            price_pivot = price_agg.to_frame().T
            price_pivot.index = [pd.Timestamp.now()]

            sig_numeric = signals_df.copy()
            sig_numeric["sig_val"] = sig_numeric["signal"].map({"BUY": 1, "SELL": -1, "HOLD": 0})
            signal_agg = sig_numeric.groupby("ticker")["sig_val"].mean()
            signal_pivot = signal_agg.to_frame().T
            signal_pivot.index = price_pivot.index
    else:
        price_pivot = signals_df[["price"]].copy()
        price_pivot.index = [pd.Timestamp.now()]
        sig_numeric = signals_df.copy()
        sig_numeric["sig_val"] = sig_numeric["signal"].map({"BUY": 1, "SELL": -1, "HOLD": 0})
        signal_pivot = sig_numeric[["sig_val"]].copy()
        signal_pivot.index = price_pivot.index

    return price_pivot, signal_pivot


async def run_backtest(backtester, signals_df) -> Dict[str, Any]:
    price_pivot, signal_pivot = prepare_pivot(signals_df)
    if price_pivot.empty or signal_pivot.empty:
        return {}

    if price_pivot.select_dtypes(include="number").shape[1] == 0:
        return {}
    if signal_pivot.select_dtypes(include="number").shape[1] == 0:
        return {}

    method = None
    if hasattr(backtester, "run_comprehensive_backtest"):
        method = backtester.run_comprehensive_backtest
    elif hasattr(backtester, "run"):
        method = backtester.run
    elif hasattr(backtester, "simulate"):
        method = backtester.simulate
    else:
        return {}

    try:
        if asyncio.iscoroutinefunction(method):
            return await method(price_pivot=price_pivot, signals=signal_pivot)
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, partial(method, price_pivot, signal_pivot))
    except TypeError:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, partial(method, price_pivot, signal_pivot))
    except Exception as e:
        logger.error(f"Backtest execution failed: {e}", exc_info=True)
        raise
