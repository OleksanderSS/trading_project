"""AnxietyKillSwitch — автоматичний запобіжник при ринковій невизначеності.

Якщо ринок демонструє підвищену волатильність, агенти показують низьку
впевненість, або причинно-наслідкові графи виявляють багато нерозв'язаних
шоків — система автоматично переходить в режим `review_only`, де жодне
рішення `candidate_long` / `candidate_short` неможливе.

Це реалізація концепції "anxiety mode" з архітектурного документу
DEAN_OS_World_Model_Architecture_Principles_v2.

Логіка Stage 6:
  Normal → [тригер] → review_only (paper_trade_only або нижче)
  review_only → [умови нормалізувалися N послідовних разів] → Normal

Налаштування через agent_registry.yaml або передачу AnxietyConfig.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from dean_os.schemas import ConsensusDecision, MarketContext


@dataclass
class AnxietyConfig:
    """Пороги для запуску kill-switch. Всі пороги налаштовуються."""

    # Денна реалізована волатильність (std returns)
    max_realized_vol_daily: float = 0.035  # 3.5%

    # Мінімальна впевненість консенсусу
    min_consensus_confidence: float = 0.40

    # Максимальний просідання (як частка, від'ємне число)
    max_drawdown_threshold: float = -0.15  # -15%

    # Кількість нерозв'язаних шоків у причинному графі
    max_causal_unresolved_shocks: int = 3

    # Мінімальна кількість активних агентів для ухвалення рішення
    min_active_agents: int = 3

    # Рішення, на які замінюємо "небезпечні" при спрацюванні
    safe_decision_override: str = "paper_trade_only"

    # Дозволити override лише якщо рішення було "агресивним"
    override_only_aggressive: bool = True
    aggressive_decisions: frozenset[str] = field(
        default_factory=lambda: frozenset({"candidate_long", "candidate_short"})
    )


@dataclass
class KillSwitchResult:
    """Результат оцінки kill-switch для поточного рішення."""

    triggered: bool
    mode: str  # "normal" або "review_only"
    reasons: list[str]
    metrics: dict[str, Any]


class AnxietyKillSwitch:
    """Оцінює поточний стан і вирішує, чи активувати режим тривожності.

    Використання::

        ks = AnxietyKillSwitch()
        result = ks.evaluate(context, decision)
        if result.triggered:
            decision = ks.apply(decision, result)
    """

    def __init__(self, config: AnxietyConfig | None = None):
        self.config = config or AnxietyConfig()

    def evaluate(
        self,
        context: MarketContext,
        decision: ConsensusDecision,
    ) -> KillSwitchResult:
        """Перевіряє всі тригери і повертає результат.

        Не мутує decision — тільки оцінює. Для застосування — `apply()`.
        """
        reasons: list[str] = []
        metrics: dict[str, Any] = {}

        # Trigger 1: Realized volatility
        vol = self._get_realized_vol(context)
        metrics["realized_vol_daily"] = vol
        if vol is not None and vol > self.config.max_realized_vol_daily:
            reasons.append(
                f"Realized volatility {vol:.2%} exceeds threshold "
                f"{self.config.max_realized_vol_daily:.2%}"
            )

        # Trigger 2: Low consensus confidence
        metrics["consensus_confidence"] = decision.confidence
        if decision.confidence < self.config.min_consensus_confidence:
            reasons.append(
                f"Consensus confidence {decision.confidence:.2f} below minimum "
                f"{self.config.min_consensus_confidence:.2f}"
            )

        # Trigger 3: Risk agent drawdown
        drawdown = self._get_max_drawdown(decision)
        metrics["max_drawdown"] = drawdown
        if drawdown is not None and drawdown < self.config.max_drawdown_threshold:
            reasons.append(
                f"Drawdown {drawdown:.2%} exceeds threshold "
                f"{self.config.max_drawdown_threshold:.2%}"
            )

        # Trigger 4: Unresolved shocks in causal graph
        shocks = self._get_causal_shocks(context)
        metrics["causal_unresolved_shocks"] = shocks
        if shocks is not None and shocks >= self.config.max_causal_unresolved_shocks:
            reasons.append(
                f"Unresolved causal shocks: {shocks} "
                f"(limit: {self.config.max_causal_unresolved_shocks})"
            )

        # Trigger 5: Too few agents responded
        # decision_influencing_agent_count (not len(agent_report_hashes),
        # which counts every report including review-only domain analysts
        # that always run and never move the score) -- otherwise this check
        # would almost never fire even when every decision-relevant guardian
        # is missing or failing, since 8+ review-only agents alone already
        # clear most min_active_agents thresholds.
        active_agents = decision.decision_influencing_agent_count
        metrics["active_agents"] = active_agents
        if active_agents < self.config.min_active_agents:
            reasons.append(
                f"Only {active_agents} active agents "
                f"(minimum: {self.config.min_active_agents}) - insufficient for decision"
            )

        triggered = len(reasons) > 0
        mode = "review_only" if triggered else "normal"
        return KillSwitchResult(
            triggered=triggered,
            mode=mode,
            reasons=reasons,
            metrics=metrics,
        )

    def apply(
        self,
        decision: ConsensusDecision,
        result: KillSwitchResult,
    ) -> ConsensusDecision:
        """Застосовує kill-switch до рішення.

        Якщо kill-switch спрацював і рішення "агресивне" (candidate_long/short),
        замінює його на безпечний override (paper_trade_only за замовчуванням).

        Завжди додає kill_switch_reasons і anxiety_kill_switch_triggered=True.
        """
        if not result.triggered:
            return decision

        # Визначаємо нове рішення
        new_decision = decision.decision
        if self.config.override_only_aggressive:
            if decision.decision in self.config.aggressive_decisions:
                new_decision = self.config.safe_decision_override
        else:
            new_decision = self.config.safe_decision_override  # type: ignore[assignment]

        # Оновлюємо рішення (Pydantic model_copy для незмінності)
        return decision.model_copy(update={
            "decision": new_decision,
            "anxiety_kill_switch_triggered": True,
            "kill_switch_reasons": result.reasons,
            "reasons": [
                *decision.reasons,
                f"[KILL_SWITCH] Anxiety mode activated: {'; '.join(result.reasons)}",
            ],
        })

    # ── Приватні хелпери для читання метрик з контексту ─────────────────────

    def _get_realized_vol(self, context: MarketContext) -> float | None:
        """Читає волатильність із pipeline_result або metadata."""
        vol = context.pipeline_result.get("realized_vol_daily")
        if vol is not None:
            return float(vol)
        vol = context.metadata.get("realized_vol_daily")
        if vol is not None:
            return float(vol)
        # Обчислюємо зі series returns якщо є
        if context.returns is not None:
            try:
                import pandas as pd
                series = context.returns
                if not isinstance(series, pd.Series):
                    if hasattr(series, "mean"):
                        series = series.mean(axis=1)
                if isinstance(series, pd.Series) and not series.empty:
                    return float(series.std())
            except Exception:
                pass
        return None

    def _get_max_drawdown(self, decision: ConsensusDecision) -> float | None:
        """Читає max_drawdown з risk_context рішення."""
        if decision.risk_context:
            dd = decision.risk_context.get("max_drawdown")
            if dd is not None:
                return float(dd)
        return None

    def _get_causal_shocks(self, context: MarketContext) -> int | None:
        """Читає кількість нерозв'язаних шоків із metadata."""
        shocks = context.metadata.get("causal_unresolved_shocks")
        if shocks is not None:
            return int(shocks)
        # Якщо є causal_graph у metadata
        cg = context.metadata.get("causal_graph", {})
        if isinstance(cg, dict):
            unresolved = cg.get("unresolved_shocks")
            if unresolved is not None:
                return int(unresolved)
        return None


def build_kill_switch_from_yaml(agent_config: dict) -> AnxietyKillSwitch:
    """Будує AnxietyKillSwitch із секції agent_registry.yaml.

    Очікуваний формат в YAML::

        anxiety_kill_switch:
          max_realized_vol_daily: 0.035
          min_consensus_confidence: 0.40
          max_drawdown_threshold: -0.15
          max_causal_unresolved_shocks: 3
          min_active_agents: 3
          safe_decision_override: paper_trade_only
    """
    ks_config = agent_config.get("anxiety_kill_switch", {})
    if not ks_config:
        return AnxietyKillSwitch()
    return AnxietyKillSwitch(config=AnxietyConfig(
        max_realized_vol_daily=float(ks_config.get("max_realized_vol_daily", 0.035)),
        min_consensus_confidence=float(ks_config.get("min_consensus_confidence", 0.40)),
        max_drawdown_threshold=float(ks_config.get("max_drawdown_threshold", -0.15)),
        max_causal_unresolved_shocks=int(ks_config.get("max_causal_unresolved_shocks", 3)),
        min_active_agents=int(ks_config.get("min_active_agents", 3)),
        safe_decision_override=str(ks_config.get("safe_decision_override", "paper_trade_only")),
    ))


__all__ = [
    "AnxietyConfig",
    "AnxietyKillSwitch",
    "KillSwitchResult",
    "build_kill_switch_from_yaml",
]
