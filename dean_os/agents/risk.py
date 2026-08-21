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
            self.evidence("metric", "risk", "return_sample_count", risk["sample_count"]),
            self.evidence(
                "metric",
                "risk",
                "returns_offline_only",
                bool(context.metadata.get("returns_offline_only")),
            ),
        ]

        if (
            context.phase == "pre_trade"
            and context.metadata.get("returns_offline_only") is True
        ):
            verdict = "blocked"
            reasons = [
                "Supervised target labels cannot serve as realized returns in pre-trade risk review"
            ]
            risks = [
                "Target leakage would make drawdown and VaR evidence invalid"
            ]
            signal_strength = -1.0
            confidence = 1.0
            quality = 0.0
        elif not risk["has_inputs"]:
            # For pre_trade phase, missing risk data is a hard block
            if context.phase == "pre_trade":
                verdict = "blocked"
                reasons = ["No returns or positions supplied to RiskAgent in pre_trade phase"]
                risks = ["Risk gate cannot validate drawdown, VaR, or exposure before trade execution - hard block"]
                signal_strength = -1.0
                confidence = 1.0
                quality = 0.0
            else:
                verdict = "caution"
                reasons = ["No returns or positions supplied to RiskAgent"]
                risks = ["Risk gate cannot validate drawdown, VaR, or exposure before pipeline execution"]
                signal_strength = 0.0
                confidence = 0.65
                quality = 0.35
        elif not risk["returns_measurable"]:
            # Positions exist but the return history is too short to say
            # anything about drawdown or tail risk. Previously this state was
            # written as 0.0 and read as "no risk", which is the one reading
            # the evidence does not support.
            shortfall = (
                f"{risk['sample_count']} return observations, "
                f"{risk['min_return_samples']} needed"
            )
            if context.phase == "pre_trade":
                verdict = "blocked"
                reasons = [f"Drawdown and VaR are unmeasurable before trade execution: {shortfall}"]
                risks = ["Risk gate cannot validate drawdown or tail risk - hard block"]
                signal_strength = -1.0
                confidence = 1.0
                quality = 0.0
            else:
                verdict = "caution"
                reasons = [f"Drawdown and VaR are unmeasurable: {shortfall}"]
                risks = ["Exposure was checked; drawdown and tail risk were not"]
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

    #: A 95th-percentile tail needs enough observations that the quantile lands
    #: on real data rather than interpolating between the only two points there
    #: are. Twenty is the smallest sample where the 5% tail touches an
    #: observation at all.
    MIN_RETURN_SAMPLES = 20

    def _risk_snapshot(self, returns: Any, positions: dict[str, float]) -> dict[str, Any]:
        """Measure drawdown, tail risk and exposure -- or report that we cannot.

        The gate used to answer every question with 0.0. Handed the single
        placeholder return the orchestrator supplies (``{"SPY": 0.0}``) it
        reported a drawdown of 0%, a VaR-95 of 0%, and a verdict of "Risk
        checks passed" at 0.85 confidence -- a clean bill of health computed
        from one invented number. A quantile over one observation is not a
        measurement, and a zero that means "nothing was measured" is
        indistinguishable here from a zero that means "no risk", except that
        the second one opens the gate.

        So the unmeasurable metrics come back as None and the caller has to
        decide what to do about not knowing.
        """
        min_samples = int(self.config.get("min_return_samples", self.MIN_RETURN_SAMPLES))
        series = self._to_return_series(returns)
        gross_exposure = float(sum(abs(value) for value in positions.values())) if positions else 0.0

        if series is not None and not series.empty:
            series = series.replace([float('inf'), float('-inf')], float('nan')).dropna()
        sample_count = 0 if series is None else int(series.shape[0])

        if sample_count < min_samples:
            return {
                "has_inputs": bool(positions),
                "returns_measurable": False,
                "max_drawdown": None,
                "daily_var_95": None,
                "gross_exposure": gross_exposure,
                "sample_count": sample_count,
                "min_return_samples": min_samples,
            }

        equity = (1.0 + series).cumprod()
        drawdown = equity / equity.cummax() - 1.0
        return {
            "has_inputs": True,
            "returns_measurable": True,
            "max_drawdown": float(drawdown.min()),
            "daily_var_95": abs(float(series.quantile(0.05))),
            "gross_exposure": gross_exposure,
            "sample_count": sample_count,
            "min_return_samples": min_samples,
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
