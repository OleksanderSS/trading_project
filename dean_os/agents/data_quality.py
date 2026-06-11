from __future__ import annotations

from typing import Any

from dean_os.base import BaseAgent
from dean_os.schemas import MarketContext, PipelineReport
from dean_os.utils import clamp


class DataQualityAgent(BaseAgent):
    version = "0.1.0"
    branch = "pipeline"

    async def run(self, context: MarketContext) -> PipelineReport:
        max_missing_ratio = float(self.config.get("max_missing_ratio", 0.25))
        metrics = self._inspect_frames(context.dataframes)
        evidence = [
            self.evidence("dataframe_check", "context.dataframes", "frame_count", metrics["frame_count"]),
            self.evidence("dataframe_check", "context.dataframes", "max_missing_ratio", metrics["max_missing_ratio"]),
            self.evidence("dataframe_check", "context.dataframes", "empty_frames", metrics["empty_frames"]),
        ]

        if metrics["frame_count"] == 0:
            return PipelineReport(
                agent_name=self.name,
                agent_version=self.version,
                verdict="caution",
                confidence=0.7,
                data_quality_score=0.35,
                signal_strength=0.0,
                reasons=["No DataFrame inputs supplied to DataQualityAgent"],
                risks=["Data quality gate cannot validate missingness, staleness, or synthetic fallback"],
                blind_spots=["The main pipeline may still load data internally after this preflight check"],
                evidence=evidence,
                input_hash=self.context_hash(context),
                metrics_snapshot=metrics,
            )

        if metrics["empty_frames"]:
            verdict = "blocked"
            reasons = [f"Empty frames detected: {', '.join(metrics['empty_frames'])}"]
            risks = ["Model input can silently collapse if empty frames pass downstream"]
            signal_strength = -1.0
        elif metrics["max_missing_ratio"] > max_missing_ratio:
            verdict = "blocked"
            reasons = [f"Missing ratio {metrics['max_missing_ratio']:.2%} exceeds {max_missing_ratio:.2%}"]
            risks = ["High missingness can fabricate or suppress signals after imputation"]
            signal_strength = -0.8
        elif metrics["synthetic_flags"]:
            verdict = "caution"
            reasons = [f"Synthetic data markers detected: {', '.join(metrics['synthetic_flags'])}"]
            risks = ["Synthetic data should not be used for production decisions"]
            signal_strength = -0.2
        else:
            verdict = "clear"
            reasons = ["DataFrame quality checks passed"]
            risks = []
            signal_strength = 0.5

        quality_score = clamp(1.0 - metrics["max_missing_ratio"], 0.0, 1.0)
        return PipelineReport(
            agent_name=self.name,
            agent_version=self.version,
            verdict=verdict,
            confidence=0.85,
            data_quality_score=quality_score,
            signal_strength=signal_strength,
            reasons=reasons,
            risks=risks,
            blind_spots=["This check does not prove temporal alignment or publication-time correctness"],
            evidence=evidence,
            input_hash=self.context_hash(context),
            metrics_snapshot=metrics,
        )

    def _inspect_frames(self, frames: dict[str, Any]) -> dict[str, Any]:
        metrics: dict[str, Any] = {
            "frame_count": len(frames),
            "row_counts": {},
            "missing_ratios": {},
            "max_missing_ratio": 0.0,
            "empty_frames": [],
            "synthetic_flags": [],
        }
        for name, frame in frames.items():
            row_count = self._safe_len(frame)
            metrics["row_counts"][name] = row_count
            if row_count == 0:
                metrics["empty_frames"].append(name)
            missing_ratio = self._missing_ratio(frame)
            metrics["missing_ratios"][name] = missing_ratio
            metrics["max_missing_ratio"] = max(metrics["max_missing_ratio"], missing_ratio)
            if self._has_synthetic_marker(frame):
                metrics["synthetic_flags"].append(name)
        return metrics

    def _safe_len(self, value: Any) -> int:
        try:
            return int(len(value))
        except Exception:
            return 0

    def _missing_ratio(self, frame: Any) -> float:
        if not hasattr(frame, "isna"):
            return 0.0
        try:
            missing = frame.isna().sum().sum()
            total = frame.shape[0] * frame.shape[1]
            return float(missing / total) if total else 0.0
        except Exception:
            return 0.0

    def _has_synthetic_marker(self, frame: Any) -> bool:
        columns = [str(column).lower() for column in getattr(frame, "columns", [])]
        if any("synthetic" in column or "simulated" in column for column in columns):
            return True
        attrs = getattr(frame, "attrs", {})
        return bool(attrs.get("synthetic") or attrs.get("is_synthetic"))
