from __future__ import annotations

from dean_os.schemas import AnalyticalReport, ConsensusDecision, EvidenceItem, PipelineReport
from dean_os.utils import clamp, sha256_json


class ConsensusEngine:
    hard_veto_agents = {"pipeline_audit", "data_quality", "risk"}

    def __init__(self, analytical_step: float = 0.08):
        self.analytical_step = analytical_step

    def combine(
        self,
        pipeline_reports: list[PipelineReport],
        pipeline_result: dict,
        analytical_reports: list[AnalyticalReport],
    ) -> ConsensusDecision:
        all_reports = [*pipeline_reports, *analytical_reports]
        report_hashes = {report.agent_name: sha256_json(report) for report in all_reports}

        hard_veto = self._find_hard_veto(pipeline_reports)
        if hard_veto is not None:
            return ConsensusDecision(
                decision_id=self._decision_id(pipeline_result, all_reports),
                decision="blocked",
                final_score=-1.0,
                confidence=hard_veto.confidence,
                blocking_agents=[hard_veto.agent_name],
                reasons=hard_veto.reasons,
                risks=hard_veto.risks,
                blind_spots=hard_veto.blind_spots,
                evidence=hard_veto.evidence,
                risk_context=hard_veto.risk_context,
                agent_report_hashes=report_hashes,
                config_hash=sha256_json({"analytical_step": self.analytical_step}),
            )

        pipeline_score = self._pipeline_score(pipeline_reports, pipeline_result)
        modifier = self._analytical_modifier(analytical_reports, pipeline_result.get("timeframe"))
        final_score = clamp(pipeline_score * modifier, -1.0, 1.0)
        decision = self._map_score_to_decision(final_score, pipeline_reports, analytical_reports)

        return ConsensusDecision(
            decision_id=self._decision_id(pipeline_result, all_reports),
            decision=decision,
            final_score=final_score,
            confidence=self._confidence(pipeline_reports, analytical_reports),
            supporting_agents=[r.agent_name for r in analytical_reports if r.position_bias == "bullish"],
            opposing_agents=[r.agent_name for r in analytical_reports if r.position_bias == "bearish"],
            reasons=self._collect_reasons(pipeline_reports, analytical_reports),
            risks=self._collect_risks(pipeline_reports, analytical_reports),
            blind_spots=self._collect_blind_spots(pipeline_reports, analytical_reports),
            evidence=self._collect_evidence(pipeline_reports, analytical_reports),
            risk_context=self._risk_context(pipeline_reports),
            agent_report_hashes=report_hashes,
            config_hash=sha256_json({"analytical_step": self.analytical_step}),
        )

    def _find_hard_veto(self, reports: list[PipelineReport]) -> PipelineReport | None:
        for report in reports:
            if report.agent_name in self.hard_veto_agents and report.verdict == "blocked":
                return report
        return None

    def _pipeline_score(self, reports: list[PipelineReport], pipeline_result: dict) -> float:
        model_score = float(pipeline_result.get("model_score", pipeline_result.get("score", 0.0)))
        risk_score = self._report_score(reports, "risk", default=0.4)
        regime_score = self._report_score(reports, "regime", default=0.0)
        return clamp(0.40 * model_score + 0.35 * risk_score + 0.25 * regime_score, -1.0, 1.0)

    def _report_score(self, reports: list[PipelineReport], agent_name: str, default: float) -> float:
        for report in reports:
            if report.agent_name == agent_name:
                if report.signal_strength is not None:
                    return report.signal_strength
                if report.verdict == "clear":
                    return 0.5
                if report.verdict == "caution":
                    return 0.0
        return default

    def _analytical_modifier(self, reports: list[AnalyticalReport], timeframe: str | None) -> float:
        max_modifier = 0.20
        if timeframe in {"15m", "30m", "1h"}:
            max_modifier = 0.05
        modifier = 1.0
        for report in reports:
            if report.data_quality == "weak":
                continue
            weight = report.confidence * self._horizon_discount(report.horizon_years, timeframe)
            if report.position_bias == "bullish":
                modifier += self.analytical_step * weight
            elif report.position_bias == "bearish":
                modifier -= self.analytical_step * weight
        return clamp(modifier, 1.0 - max_modifier, 1.0 + max_modifier)

    def _horizon_discount(self, horizon_years: float, timeframe: str | None) -> float:
        if timeframe in {"15m", "30m", "1h"}:
            return 0.05
        if timeframe in {"1d", "daily"}:
            return 0.35 if horizon_years >= 1 else 0.60
        return 0.70 if horizon_years <= 2 else 0.45

    def _map_score_to_decision(
        self,
        score: float,
        pipeline_reports: list[PipelineReport],
        analytical_reports: list[AnalyticalReport],
    ) -> str:
        if any(report.verdict == "caution" for report in pipeline_reports):
            return "watchlist"
        if any(report.watchlist_score >= 0.75 for report in analytical_reports):
            return "watchlist"
        if score > 0.70:
            return "candidate_long"
        if score < -0.70:
            return "candidate_short"
        if abs(score) > 0.40:
            return "watchlist"
        return "no_trade"

    def _confidence(self, pipeline_reports: list[PipelineReport], analytical_reports: list[AnalyticalReport]) -> float:
        reports = [*pipeline_reports, *analytical_reports]
        if not reports:
            return 0.0
        return clamp(sum(report.confidence * report.data_quality_score for report in reports) / len(reports), 0.0, 1.0)

    def _collect_reasons(self, pipeline_reports: list[PipelineReport], analytical_reports: list[AnalyticalReport]) -> list[str]:
        return [reason for report in [*pipeline_reports, *analytical_reports] for reason in report.reasons]

    def _collect_risks(self, pipeline_reports: list[PipelineReport], analytical_reports: list[AnalyticalReport]) -> list[str]:
        return [risk for report in [*pipeline_reports, *analytical_reports] for risk in report.risks]

    def _collect_blind_spots(
        self, pipeline_reports: list[PipelineReport], analytical_reports: list[AnalyticalReport]
    ) -> list[str]:
        return [spot for report in [*pipeline_reports, *analytical_reports] for spot in report.blind_spots]

    def _collect_evidence(
        self, pipeline_reports: list[PipelineReport], analytical_reports: list[AnalyticalReport]
    ) -> list[EvidenceItem]:
        return [item for report in [*pipeline_reports, *analytical_reports] for item in report.evidence]

    def _risk_context(self, pipeline_reports: list[PipelineReport]) -> dict | None:
        for report in pipeline_reports:
            if report.risk_context:
                return report.risk_context
        return None

    def _decision_id(self, pipeline_result: dict, reports: list) -> str:
        seed = {
            "tickers": pipeline_result.get("tickers", []),
            "timeframe": pipeline_result.get("timeframe"),
            "reports": [report.agent_name for report in reports],
        }
        return sha256_json(seed)[:16]
