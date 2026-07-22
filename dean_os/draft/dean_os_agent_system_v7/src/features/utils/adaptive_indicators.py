"""
Adaptive Technical Indicators
==============================
Indicators whose internal period/bandwidth automatically widens or narrows
based on the *trailing* volatility and trend-strength of the time series.

Core idea
---------
A fixed RSI(14) treats a calm +0.1 %/day market the same as a crash day.
Adaptive indicators scale their lookback so that:
  • high-volatility → shorter period → faster signal response
  • low-volatility  → longer period  → smoother, less noisy signal

All calculations are strictly point-in-time (no look-ahead):
  • `vol_window` / `regime_window` define *trailing* windows used to assess
    the current volatility/trend regime before computing the indicator.
  • No centred rolling windows are used.

Usage
-----
    from src.features.utils.adaptive_indicators import AdaptiveIndicators

    ai = AdaptiveIndicators()
    df["ARSI_14"]   = ai.adaptive_rsi(df["close"])
    df["ABB_upper"], df["ABB_mid"], df["ABB_lower"] = ai.adaptive_bollinger(df["close"])
    summary = ai.get_regime_summary(df["close"])
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Safe quantile helper — avoids ZeroDivisionError when the series is very
# short or has zero variance.
# ---------------------------------------------------------------------------

def _safe_quantile(series: pd.Series, q: float, fallback: float = 1.0) -> float:
    """Return the q-quantile of *series*, falling back to *fallback* when the
    result would be 0 or NaN (prevents division by zero downstream)."""
    val = series.quantile(q)
    if pd.isna(val) or val == 0.0:
        val2 = series.mean()
        return float(val2) if (not pd.isna(val2) and val2 != 0.0) else fallback
    return float(val)


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class AdaptiveIndicators:
    """
    Adaptive technical indicators with volatility-/trend-aware period scaling.

    Parameters
    ----------
    vol_window : int
        Rolling window (bars) used to estimate the current volatility level.
        Default 20.
    regime_window : int
        Rolling window (bars) used to estimate the current trend strength.
        Default 50.
    min_period_ratio : float
        Lower bound for the period multiplier. 0.4 → the adaptive period
        is never shorter than 40 % of the base period.  Default 0.4.
    max_period_ratio : float
        Upper bound for the period multiplier.  2.5 → the adaptive period
        is never longer than 250 % of the base period.  Default 2.5.
    """

    def __init__(
        self,
        vol_window: int = 20,
        regime_window: int = 50,
        min_period_ratio: float = 0.4,
        max_period_ratio: float = 2.5,
    ) -> None:
        self.vol_window = int(vol_window)
        self.regime_window = int(regime_window)
        self.min_period_ratio = float(min_period_ratio)
        self.max_period_ratio = float(max_period_ratio)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _returns(self, prices: pd.Series) -> pd.Series:
        return prices.pct_change(fill_method=None).replace(
            [np.inf, -np.inf], np.nan
        )

    def _vol_multiplier(self, prices: pd.Series) -> pd.Series:
        """
        Per-bar multiplier in [min_ratio, max_ratio] derived from trailing
        volatility relative to its 90th-percentile baseline.

        High vol  → multiplier < 1  → shorter (faster) period.
        Low vol   → multiplier > 1  → longer  (smoother) period.
        """
        ret = self._returns(prices)
        vol = ret.rolling(self.vol_window, min_periods=2).std()
        # Use 90th-percentile as "normal" volatility baseline so the
        # multiplier is centred around 1 in typical conditions.
        baseline = _safe_quantile(vol.dropna(), 0.90, fallback=vol.mean() or 1e-6)
        # Invert: high vol → smaller period → multiplier < 1
        raw = baseline / vol.clip(lower=1e-9)
        return raw.clip(lower=self.min_period_ratio, upper=self.max_period_ratio)

    def _trend_multiplier(self, prices: pd.Series) -> pd.Series:
        """
        Per-bar multiplier based on the absolute rolling mean return
        (proxy for trend strength).

        Strong trend → multiplier > 1 → longer period (follow the trend).
        Choppy / flat → multiplier < 1 → shorter period (react faster).
        """
        ret = self._returns(prices)
        trend = ret.rolling(self.regime_window, min_periods=5).mean().abs()
        baseline = _safe_quantile(trend.dropna(), 0.90, fallback=trend.mean() or 1e-6)
        raw = trend / max(float(baseline), 1e-9)
        return raw.clip(lower=self.min_period_ratio, upper=self.max_period_ratio)

    def _adaptive_period(
        self, prices: pd.Series, base_period: int, use_trend: bool = False
    ) -> pd.Series:
        """
        Return a per-bar float series of effective periods.

        Combines vol_multiplier (and optionally trend_multiplier) with the
        base period, then clips to [max(2, base*min_ratio), base*max_ratio].
        """
        vm = self._vol_multiplier(prices)
        if use_trend:
            tm = self._trend_multiplier(prices)
            # Average of the two signals
            effective = (vm + tm) / 2.0
        else:
            effective = vm
        period_series = (base_period * effective).round().clip(
            lower=max(2, int(base_period * self.min_period_ratio)),
            upper=int(base_period * self.max_period_ratio),
        )
        return period_series

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def adaptive_rsi(
        self,
        prices: pd.Series,
        base_period: int = 14,
    ) -> pd.Series:
        """
        RSI with a dynamically adjusted lookback.

        High-volatility regimes → shorter period → faster signal.
        Low-volatility regimes  → longer period  → smoother signal.

        Returns
        -------
        pd.Series  (same index as *prices*)
            Named ``ARSI_{base_period}``.
        """
        periods = self._adaptive_period(prices, base_period)
        result = pd.Series(np.nan, index=prices.index, name=f"ARSI_{base_period}")
        ret = prices.diff()

        for i in range(len(prices)):
            p = int(periods.iloc[i]) if not pd.isna(periods.iloc[i]) else base_period
            if i < p:
                continue
            window = ret.iloc[i - p : i + 1]
            gain = window.clip(lower=0).mean()
            loss = -window.clip(upper=0).mean()
            if loss == 0:
                result.iloc[i] = 100.0
            else:
                rs = gain / loss
                result.iloc[i] = 100.0 - (100.0 / (1.0 + rs))

        return result

    def adaptive_macd(
        self,
        prices: pd.Series,
        base_fast: int = 12,
        base_slow: int = 26,
        base_signal: int = 9,
    ) -> tuple[pd.Series, pd.Series, pd.Series]:
        """
        MACD where both fast and slow EMA spans scale with volatility.

        Returns
        -------
        (macd_line, signal_line, histogram)  all pd.Series.
        """
        vm = self._vol_multiplier(prices)
        fast_spans = (base_fast * vm).round().clip(lower=2, upper=base_fast * 3).astype(int)
        slow_spans = (base_slow * vm).round().clip(lower=3, upper=base_slow * 3).astype(int)

        macd_vals = pd.Series(np.nan, index=prices.index, name="AMACD")
        for i in range(1, len(prices)):
            fs = int(fast_spans.iloc[i])
            ss = int(slow_spans.iloc[i])
            if i < ss:
                continue
            fast_ema = prices.iloc[: i + 1].ewm(span=fs, adjust=False).mean().iloc[-1]
            slow_ema = prices.iloc[: i + 1].ewm(span=ss, adjust=False).mean().iloc[-1]
            macd_vals.iloc[i] = fast_ema - slow_ema

        macd_clean = macd_vals.dropna()
        if len(macd_clean) >= base_signal:
            signal_vals = macd_vals.ewm(span=base_signal, adjust=False, min_periods=base_signal).mean()
        else:
            signal_vals = pd.Series(np.nan, index=prices.index, name="AMACD_Signal")

        histogram = macd_vals - signal_vals
        signal_vals.name = "AMACD_Signal"
        histogram.name = "AMACD_Hist"
        return macd_vals, signal_vals, histogram

    def adaptive_bollinger(
        self,
        prices: pd.Series,
        base_period: int = 20,
        n_std: float = 2.0,
    ) -> tuple[pd.Series, pd.Series, pd.Series]:
        """
        Bollinger Bands with adaptive lookback period.

        High vol → shorter window → tighter, faster-moving bands.
        Low vol  → longer window  → wider, slower-moving bands.

        Returns
        -------
        (upper, middle, lower)  all pd.Series.
        """
        periods = self._adaptive_period(prices, base_period)
        upper = pd.Series(np.nan, index=prices.index, name="ABB_Upper")
        middle = pd.Series(np.nan, index=prices.index, name="ABB_Mid")
        lower = pd.Series(np.nan, index=prices.index, name="ABB_Lower")

        for i in range(len(prices)):
            p = int(periods.iloc[i]) if not pd.isna(periods.iloc[i]) else base_period
            if i < p - 1:
                continue
            window = prices.iloc[i - p + 1 : i + 1]
            mu = window.mean()
            sigma = window.std(ddof=1) if len(window) > 1 else 0.0
            if pd.isna(sigma) or sigma < 0:
                sigma = 0.0
            middle.iloc[i] = mu
            upper.iloc[i] = mu + n_std * sigma
            lower.iloc[i] = mu - n_std * sigma

        return upper, middle, lower

    def adaptive_atr(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        base_period: int = 14,
    ) -> pd.Series:
        """
        Average True Range with adaptive smoothing period.

        Returns
        -------
        pd.Series  named ``AATR_{base_period}``.
        """
        periods = self._adaptive_period(close, base_period)
        tr = pd.concat(
            [
                high - low,
                (high - close.shift(1)).abs(),
                (low - close.shift(1)).abs(),
            ],
            axis=1,
        ).max(axis=1)

        result = pd.Series(np.nan, index=close.index, name=f"AATR_{base_period}")
        for i in range(len(close)):
            p = int(periods.iloc[i]) if not pd.isna(periods.iloc[i]) else base_period
            if i < p:
                continue
            result.iloc[i] = tr.iloc[i - p + 1 : i + 1].mean()

        return result

    def adaptive_moving_average(
        self,
        prices: pd.Series,
        base_period: int = 20,
        ma_type: str = "ema",
    ) -> pd.Series:
        """
        SMA or EMA with adaptive lookback.

        Parameters
        ----------
        ma_type : {"ema", "sma"}
        """
        periods = self._adaptive_period(prices, base_period, use_trend=True)
        result = pd.Series(np.nan, index=prices.index, name=f"A{ma_type.upper()}_{base_period}")

        for i in range(len(prices)):
            p = int(periods.iloc[i]) if not pd.isna(periods.iloc[i]) else base_period
            if i < p - 1:
                continue
            window = prices.iloc[i - p + 1 : i + 1]
            if ma_type == "ema":
                result.iloc[i] = window.ewm(span=p, adjust=False).mean().iloc[-1]
            else:
                result.iloc[i] = window.mean()

        return result

    # ------------------------------------------------------------------
    # Diagnostics / monitoring
    # ------------------------------------------------------------------

    def get_regime_summary(self, prices: pd.Series) -> dict[str, Any]:
        """
        Return a dict describing the *current* (most recent bar) adaptive
        regime parameters for monitoring/logging purposes.
        """
        if prices.empty or len(prices) < self.vol_window:
            return {"status": "insufficient_data", "min_bars_needed": self.vol_window}

        vm = self._vol_multiplier(prices)
        tm = self._trend_multiplier(prices)
        ret = self._returns(prices)
        vol = ret.rolling(self.vol_window, min_periods=2).std()

        last_vm = float(vm.iloc[-1]) if not pd.isna(vm.iloc[-1]) else 1.0
        last_tm = float(tm.iloc[-1]) if not pd.isna(tm.iloc[-1]) else 1.0
        last_vol = float(vol.iloc[-1]) if not pd.isna(vol.iloc[-1]) else 0.0

        return {
            "status": "ok",
            "current_volatility": round(last_vol, 6),
            "vol_multiplier": round(last_vm, 3),
            "trend_multiplier": round(last_tm, 3),
            "effective_rsi14_period": int(round(14 * last_vm)),
            "effective_bb20_period": int(round(20 * last_vm)),
            "effective_atr14_period": int(round(14 * last_vm)),
            "regime_label": (
                "high_volatility" if last_vm < 0.8
                else "low_volatility" if last_vm > 1.4
                else "normal"
            ),
        }


# ---------------------------------------------------------------------------
# Module-level singleton for use by TechnicalAnalysisEnricher
# ---------------------------------------------------------------------------

_instance: AdaptiveIndicators | None = None


def get_adaptive_indicators() -> AdaptiveIndicators:
    """Return a shared module-level instance (lazy init)."""
    global _instance
    if _instance is None:
        _instance = AdaptiveIndicators()
    return _instance
