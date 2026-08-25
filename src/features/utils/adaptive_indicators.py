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

    # ------------------------------------------------------------------
    # Window loops
    # ------------------------------------------------------------------
    #
    # These four indicators take a window whose LENGTH CHANGES PER ROW, so
    # none of them can be a `.rolling(p)` call -- that is the whole point of
    # the module. What they can avoid is paying pandas for the slicing.
    #
    # Measured on the 110-ticker rebuild of 2026-08-24, daily frame, 7,507
    # rows per ticker: the six adaptive features cost 290.3 s per ticker,
    # 76% of the entire technical-analysis step, while SMA, EMA, RSI, MACD,
    # Bollinger, ATR, stochastic, Williams %R and CCI together cost 0.4 s.
    # The arithmetic was never the cost. `ret.iloc[i - p : i + 1]` builds a
    # new Series with its own index on every one of 7,507 iterations, four
    # times over, and that is where the four and a half minutes went.
    #
    # The arithmetic below is deliberately unchanged, down to pandas'
    # NaN-skipping (`np.nanmean` for `.mean()`, `np.nanstd(ddof=1)` for
    # `.std()`) and to the off-by-one difference between the RSI/ATR guard
    # (`i < p`) and the Bollinger/MA guard (`i < p - 1`), which is original.
    # `tests/unit/test_adaptive_indicators_equivalence.py` holds the previous
    # implementations verbatim and asserts the outputs are identical, so this
    # is a speed change and nothing else.

    @staticmethod
    def _periods_array(periods: pd.Series, base_period: int) -> np.ndarray:
        """Per-row window lengths as ints, NaN falling back to the base."""
        raw = periods.to_numpy(dtype=float)
        return np.where(np.isnan(raw), float(base_period), raw).astype(np.intp)

    @staticmethod
    def _mean_or_nan(window: np.ndarray) -> float:
        """`Series.mean()`: skip NaN, and give NaN when everything is NaN."""
        if window.size == 0 or bool(np.isnan(window).all()):
            return float("nan")
        return float(np.nanmean(window))

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
        result = np.full(len(prices), np.nan, dtype=float)
        ret = prices.diff().to_numpy(dtype=float)
        window_lengths = self._periods_array(periods, base_period)

        for i in range(len(prices)):
            p = int(window_lengths[i])
            if i < p:
                continue
            window = ret[i - p : i + 1]
            gain = self._mean_or_nan(np.clip(window, 0.0, None))
            loss = -self._mean_or_nan(np.clip(window, None, 0.0))
            if loss == 0:
                result[i] = 100.0
            else:
                rs = gain / loss
                result[i] = 100.0 - (100.0 / (1.0 + rs))

        return pd.Series(result, index=prices.index, name=f"ARSI_{base_period}")

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
        values = prices.to_numpy(dtype=float)
        upper_v = np.full(len(prices), np.nan, dtype=float)
        middle_v = np.full(len(prices), np.nan, dtype=float)
        lower_v = np.full(len(prices), np.nan, dtype=float)
        window_lengths = self._periods_array(periods, base_period)

        for i in range(len(prices)):
            p = int(window_lengths[i])
            if i < p - 1:
                continue
            window = values[i - p + 1 : i + 1]
            mu = self._mean_or_nan(window)
            # `Series.std(ddof=1)` is NaN when fewer than two values are
            # present, and the guard below turns that into 0.0 -- as before.
            if window.size > 1 and int(np.count_nonzero(~np.isnan(window))) > 1:
                sigma = float(np.nanstd(window, ddof=1))
            else:
                sigma = 0.0
            if np.isnan(sigma) or sigma < 0:
                sigma = 0.0
            middle_v[i] = mu
            upper_v[i] = mu + n_std * sigma
            lower_v[i] = mu - n_std * sigma

        index = prices.index
        return (
            pd.Series(upper_v, index=index, name="ABB_Upper"),
            pd.Series(middle_v, index=index, name="ABB_Mid"),
            pd.Series(lower_v, index=index, name="ABB_Lower"),
        )

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

        result = np.full(len(close), np.nan, dtype=float)
        true_range = tr.to_numpy(dtype=float)
        window_lengths = self._periods_array(periods, base_period)

        for i in range(len(close)):
            p = int(window_lengths[i])
            if i < p:
                continue
            result[i] = self._mean_or_nan(true_range[i - p + 1 : i + 1])

        return pd.Series(result, index=close.index, name=f"AATR_{base_period}")

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
        values = prices.to_numpy(dtype=float)
        result = np.full(len(prices), np.nan, dtype=float)
        window_lengths = self._periods_array(periods, base_period)

        for i in range(len(prices)):
            p = int(window_lengths[i])
            if i < p - 1:
                continue
            window = values[i - p + 1 : i + 1]
            if ma_type != "ema":
                result[i] = self._mean_or_nan(window)
                continue
            # `ewm(span=p, adjust=False).mean()` is the recursion
            # s_k = a * x_k + (1 - a) * s_(k-1), seeded with s_0 = x_0. Run
            # it directly here; a window holding NaN goes back to pandas,
            # whose NaN handling in `ewm` is more intricate than it looks and
            # is not worth reproducing for a case that costs nothing.
            if bool(np.isnan(window).any()):
                result[i] = float(
                    pd.Series(window).ewm(span=p, adjust=False).mean().to_numpy()[-1]
                )
                continue
            alpha = 2.0 / (float(p) + 1.0)
            smoothed = float(window[0])
            for step in range(1, window.size):
                smoothed = alpha * float(window[step]) + (1.0 - alpha) * smoothed
            result[i] = smoothed

        return pd.Series(
            result, index=prices.index, name=f"A{ma_type.upper()}_{base_period}"
        )

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
