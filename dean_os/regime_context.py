from __future__ import annotations

from math import sqrt
from typing import Any

from dean_os.schemas import MarketRegimeSnapshot
from dean_os.utils import clamp


REGIME_TAG_MAP: dict[str, list[str]] = {
    "BULL_MARKET": ["rising_market"],
    "TRENDING_UP": ["rising_market"],
    "MOMENTUM": ["rising_market", "momentum"],
    "BREAKOUT": ["rising_market", "breakout"],
    "BEAR_MARKET": ["falling_market"],
    "TRENDING_DOWN": ["falling_market"],
    "RANGING": ["calm_market", "range_bound"],
    "SIDEWAYS": ["calm_market", "range_bound"],
    "NORMAL": ["calm_market"],
    "MEAN_REVERSION": ["calm_market", "mean_reversion"],
    "VOLATILE": ["volatility_spike"],
    "CRISIS": ["crisis", "volatility_spike"],
}


def normalize_context_tags(tags: list[str] | tuple[str, ...] | None) -> list[str]:
    """Normalize context labels so memory lookup is stable across CLIs and agents."""

    normalized: list[str] = []
    seen = set()
    for tag in tags or []:
        value = str(tag).strip().lower().replace("-", "_").replace(" ", "_")
        if value and value not in seen:
            normalized.append(value)
            seen.add(value)
    return normalized


class RegimeContextBuilder:
    """Builds DEAN-OS regime context without starting the trading pipeline."""

    def from_analyzer_result(
        self,
        result: dict[str, Any] | None,
        source: str = "market_regime_analyzer",
    ) -> MarketRegimeSnapshot:
        result = result or {}
        regime = str(result.get("regime") or "UNKNOWN").upper()
        confidence = _safe_float(result.get("confidence"), default=0.0)
        direct_tags = normalize_context_tags(result.get("context_tags") or result.get("tags") or [])
        mapped_tags = REGIME_TAG_MAP.get(regime, [])
        metrics = {
            key: value
            for key, value in result.items()
            if key not in {"regime", "confidence", "context_tags", "tags"}
        }
        warnings = []
        if regime == "UNKNOWN":
            warnings.append("Market regime analyzer returned UNKNOWN.")
        return MarketRegimeSnapshot(
            regime=regime,
            confidence=clamp(confidence, 0.0, 1.0),
            context_tags=normalize_context_tags([*mapped_tags, *direct_tags]),
            source=source,
            metrics=metrics,
            warnings=warnings,
        )

    def from_project_analyzer(self, data: Any) -> MarketRegimeSnapshot:
        """Use the existing project MarketRegimeAnalyzer, then normalize its output."""

        try:
            from src.analytics.context.market_regime_analyzer import MarketRegimeAnalyzer

            result = MarketRegimeAnalyzer().analyze(data)
            return self.from_analyzer_result(result, source="project_market_regime_analyzer")
        except Exception as exc:
            return MarketRegimeSnapshot(
                source="project_market_regime_analyzer",
                warnings=[f"Project market regime analyzer failed: {type(exc).__name__}: {exc}"],
            )

    def from_price_frame(
        self,
        data: Any,
        close_col: str = "close",
        volume_col: str | None = "volume",
        min_returns: int = 10,
    ) -> MarketRegimeSnapshot:
        """Fallback OHLCV classifier for safe local tests and small research runs."""

        try:
            import pandas as pd
        except Exception as exc:
            return MarketRegimeSnapshot(
                source="fallback_ohlcv",
                warnings=[f"pandas is required for fallback OHLCV regime detection: {exc}"],
            )

        if not hasattr(data, "columns") or close_col not in data.columns:
            return MarketRegimeSnapshot(
                source="fallback_ohlcv",
                warnings=[f"Missing close column: {close_col}"],
            )

        close = pd.to_numeric(data[close_col], errors="coerce").dropna()
        returns = close.pct_change(fill_method=None).replace([float("inf"), float("-inf")], pd.NA).dropna()
        if len(returns) < min_returns:
            return MarketRegimeSnapshot(
                source="fallback_ohlcv",
                warnings=[f"Insufficient returns for regime detection: {len(returns)} < {min_returns}"],
            )

        window_20 = min(20, len(returns))
        window_60_prices = min(60, len(close))
        trend_20 = _window_return(close, window_20 + 1)
        trend_60 = _window_return(close, min(61, len(close)))
        vol_5 = float(returns.tail(min(5, len(returns))).std() * sqrt(252))
        vol_20 = float(returns.tail(window_20).std() * sqrt(252))
        vol_ratio = vol_5 / vol_20 if vol_20 else 0.0
        recent_prices = close.tail(window_60_prices)
        drawdown_60 = float((recent_prices / recent_prices.cummax() - 1.0).min()) if len(recent_prices) else 0.0
        shock_down = float(returns.tail(5).min()) if len(returns) else 0.0
        volume_ratio = self._volume_ratio(data, volume_col)

        tags: list[str] = []
        if drawdown_60 <= -0.20 or shock_down <= -0.07:
            tags.extend(["crisis", "volatility_spike"])
            regime = "CRISIS"
        elif drawdown_60 <= -0.12 and vol_ratio >= 1.4:
            tags.extend(["crisis", "volatility_spike"])
            regime = "CRISIS"
        elif vol_ratio >= 1.6 or vol_20 >= 0.45:
            tags.append("volatility_spike")
            regime = "VOLATILE"
        elif trend_20 >= 0.08 or trend_60 >= 0.12:
            tags.append("rising_market")
            regime = "TRENDING_UP"
        elif trend_20 <= -0.08 or trend_60 <= -0.12:
            tags.append("falling_market")
            regime = "TRENDING_DOWN"
        else:
            tags.append("calm_market")
            regime = "NORMAL"

        if volume_ratio is not None and volume_ratio >= 1.5:
            tags.append("volume_expansion")

        confidence = 0.52
        confidence += min(abs(trend_20), 0.25) * 1.2
        confidence += min(max(vol_ratio - 1.0, 0.0), 1.5) * 0.12
        confidence += 0.15 if regime == "CRISIS" else 0.0

        return MarketRegimeSnapshot(
            regime=regime,
            confidence=clamp(confidence, 0.1, 0.95),
            context_tags=normalize_context_tags(tags),
            source="fallback_ohlcv",
            metrics={
                "trend_20": trend_20,
                "trend_60": trend_60,
                "volatility_5": vol_5,
                "volatility_20": vol_20,
                "volatility_ratio": vol_ratio,
                "drawdown_60": drawdown_60,
                "shock_down_5d": shock_down,
                "volume_ratio_20": volume_ratio,
            },
        )

    def _volume_ratio(self, data: Any, volume_col: str | None) -> float | None:
        if not volume_col or not hasattr(data, "columns") or volume_col not in data.columns:
            return None
        volume = data[volume_col]
        try:
            volume = volume.astype(float).dropna()
            if len(volume) < 20:
                return None
            recent = float(volume.tail(5).mean())
            baseline = float(volume.tail(20).mean())
            return recent / baseline if baseline else None
        except Exception:
            return None


def _window_return(series: Any, window: int) -> float:
    if len(series) < 2:
        return 0.0
    window = min(max(window, 2), len(series))
    start = float(series.iloc[-window])
    end = float(series.iloc[-1])
    return end / start - 1.0 if start else 0.0


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default
