from __future__ import annotations

from typing import Any

import pandas as pd

from dean_os.base import BaseAgent
from dean_os.schemas import MarketContext, PipelineReport
from dean_os.utils import clamp


class RiskAgent(BaseAgent):
    version = "0.1.0"
    branch = "pipeline"

    async def run(self, context: MarketContext) -> PipelineReport:
        max_drawdown_limit = float(self.config.get("max_drawdown_limit", 0.20))
        max_daily_var_95 = float(self.config.get("max_daily_var_95", 0.08))
        max_gross_exposure = float(self.config.get("max_gross_exposure", 1.0))
        risk = self._risk_snapshot(context.returns, context.positions)

        evidence = [
            self.evidence("metric", "risk", "max_drawdown", risk["max_drawdown"]),
            self.evidence("metric", "risk", "daily_var_95", risk["daily_var_95"]),
            self.evidence("metric", "risk", "gross_exposure", risk["gross_exposure"]),
        ]

        if not risk["has_inputs"]:
            verdict = "caution"
            reasons = ["No returns or positions supplied to RiskAgent"]
            risks = ["Risk gate cannot validate drawdown, VaR, or exposure before pipeline execution"]
            signal_strength = 0.0
            confidence = 0.65
            quality = 0.35
        elif risk["gross_exposure"] > max_gross_exposure:
            verdict = "blocked"
            reasons = [f"Gross exposure {risk['gross_exposure']:.2f} exceeds {max_gross_exposure:.2f}"]
            risks = ["Exposure limit breach"]
            signal_strength = -1.0
            confidence = 0.95
            quality = 0.9
        elif abs(risk["max_drawdown"]) > max_drawdown_limit:
            verdict = "blocked"
            reasons = [f"Drawdown {risk['max_drawdown']:.2%} exceeds {max_drawdown_limit:.2%}"]
            risks = ["Drawdown limit breach"]
            signal_strength = -1.0
            confidence = 0.95
            quality = 0.9
        elif risk["daily_var_95"] > max_daily_var_95:
            verdict = "blocked"
            reasons = [f"Daily VaR95 {risk['daily_var_95']:.2%} exceeds {max_daily_var_95:.2%}"]
            risks = ["Tail-risk limit breach"]
            signal_strength = -0.9
            confidence = 0.9
            quality = 0.85
        else:
            verdict = "clear"
            reasons = ["Risk checks passed"]
            risks = []
            pressure = max(
                abs(risk["max_drawdown"]) / max_drawdown_limit if max_drawdown_limit else 0.0,
                risk["daily_var_95"] / max_daily_var_95 if max_daily_var_95 else 0.0,
                risk["gross_exposure"] / max_gross_exposure if max_gross_exposure else 0.0,
            )
            signal_strength = clamp(1.0 - pressure, -1.0, 1.0)
            confidence = 0.85
            quality = 0.85

        return PipelineReport(
            agent_name=self.name,
            agent_version=self.version,
            verdict=verdict,
            confidence=confidence,
            data_quality_score=quality,
            signal_strength=signal_strength,
            reasons=reasons,
            risks=risks,
            blind_spots=["Risk snapshot does not include broker-side liquidity or real execution impact"],
            evidence=evidence,
            input_hash=self.context_hash(context),
            metrics_snapshot=risk,
            risk_context=risk,
        )

    def _risk_snapshot(self, returns: Any, positions: dict[str, float]) -> dict[str, Any]:
        series = self._to_return_series(returns)
        gross_exposure = float(sum(abs(value) for value in positions.values())) if positions else 0.0
        if series is None or series.empty:
            return {
                "has_inputs": bool(positions),
                "max_drawdown": 0.0,
                "daily_var_95": 0.0,
                "gross_exposure": gross_exposure,
                "sample_count": 0,
            }
        equity = (1.0 + series.fillna(0.0)).cumprod()
        drawdown = equity / equity.cummax() - 1.0
        daily_var_95 = abs(float(series.quantile(0.05)))
        return {
            "has_inputs": True,
            "max_drawdown": float(drawdown.min()),
            "daily_var_95": daily_var_95,
            "gross_exposure": gross_exposure,
            "sample_count": int(series.shape[0]),
        }

    def _to_return_series(self, returns: Any) -> pd.Series | None:
        if returns is None:
            return None
        if isinstance(returns, pd.Series):
            return returns.astype(float)
        if isinstance(returns, pd.DataFrame):
            numeric = returns.select_dtypes(include="number")
            if numeric.empty:
                return None
            return numeric.mean(axis=1).astype(float)
        try:
            return pd.Series(returns, dtype="float64")
        except Exception:
            return None
