"""
dean_os/observability/safety_counters.py

Лічильники безпеки системи. Відстежують будь-які спроби генерації заборонених виходів
(торгові сигнали, цінові цілі, прямі рекомендації) під час щоденного запуску.
Codex Phase 6 — Safety Counters.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class SafetyCounters:
    """
    Лічильники безпеки для щоденного запуску пайплайну.
    Якщо будь-який лічильник > 0 — запуск повинен бути позначений як порушення,
    а відповідний артефакт — заблокований.
    """
    buy_sell_hold_generated: int = 0
    price_target_generated: int = 0
    trade_signal_generated: int = 0
    broker_call_attempted: int = 0
    production_config_mutation_attempted: int = 0
    model_promotion_attempted: int = 0

    def is_clean(self) -> bool:
        """Повертає True, якщо жодного порушення не виявлено."""
        return (
            self.buy_sell_hold_generated == 0
            and self.price_target_generated == 0
            and self.trade_signal_generated == 0
            and self.broker_call_attempted == 0
            and self.production_config_mutation_attempted == 0
            and self.model_promotion_attempted == 0
        )

    def violations(self) -> list[str]:
        """Повертає список назв порушених лічильників."""
        result = []
        if self.buy_sell_hold_generated:
            result.append(f"buy_sell_hold_generated={self.buy_sell_hold_generated}")
        if self.price_target_generated:
            result.append(f"price_target_generated={self.price_target_generated}")
        if self.trade_signal_generated:
            result.append(f"trade_signal_generated={self.trade_signal_generated}")
        if self.broker_call_attempted:
            result.append(f"broker_call_attempted={self.broker_call_attempted}")
        if self.production_config_mutation_attempted:
            result.append(f"production_config_mutation_attempted={self.production_config_mutation_attempted}")
        if self.model_promotion_attempted:
            result.append(f"model_promotion_attempted={self.model_promotion_attempted}")
        return result

    def as_dict(self) -> dict:
        return {
            "buy_sell_hold_generated": self.buy_sell_hold_generated,
            "price_target_generated": self.price_target_generated,
            "trade_signal_generated": self.trade_signal_generated,
            "broker_call_attempted": self.broker_call_attempted,
            "production_config_mutation_attempted": self.production_config_mutation_attempted,
            "model_promotion_attempted": self.model_promotion_attempted,
        }
