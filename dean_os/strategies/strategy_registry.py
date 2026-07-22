"""
dean_os/strategies/strategy_registry.py

Реєстр стратегій системи DEAN-OS.
Зберігає, знаходить, та перевіряє стратегічні плейбуки.
Відповідає інтеграційному порядку Codex Phase 7.

Правило: жодна стратегія не може перейти у "live" рівень без проходження всіх gates.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

from dean_os.execution.maturity_gates import verify_gate_receipt

from dean_os.strategies.strategy_playbook import (
    MaturityLevel,
    StrategyPlaybook,
    StrategyStatus,
)

_REGISTRY_DIR = Path("configs/strategies")


class StrategyNotFound(KeyError):
    pass


class StrategyRegistry:
    """
    In-memory реєстр стратегічних плейбуків з підтримкою
    збереження/завантаження з диску.
    """

    def __init__(self, registry_dir: Path | str = _REGISTRY_DIR):
        self._registry_dir = Path(registry_dir)
        self._playbooks: dict[str, StrategyPlaybook] = {}

    # ── CRUD ──────────────────────────────────────────────────────────────────

    def register(self, playbook: StrategyPlaybook) -> None:
        """Реєструє або оновлює стратегічний плейбук."""
        self._playbooks[playbook.strategy_id] = playbook

    def get(self, strategy_id: str) -> StrategyPlaybook:
        if strategy_id not in self._playbooks:
            raise StrategyNotFound(f"Strategy '{strategy_id}' not found in registry.")
        return self._playbooks[strategy_id]

    def all(self) -> list[StrategyPlaybook]:
        return list(self._playbooks.values())

    def by_status(self, status: StrategyStatus) -> list[StrategyPlaybook]:
        return [p for p in self._playbooks.values() if p.status == status]

    def allowed_for_regime(self, current_regime: str) -> list[StrategyPlaybook]:
        return [
            p for p in self._playbooks.values()
            if p.is_regime_allowed(current_regime)
        ]

    # ── Promotion Gate ─────────────────────────────────────────────────────────

    def request_promotion(
        self,
        strategy_id: str,
        target_level: MaturityLevel,
        approver: str | None = None,
        gate_receipt: dict | None = None,
    ) -> dict:
        """
        Запит на промоцію стратегії. Перевіряє всі gate-умови.
        Повертає рішення: blocked | review_required | approved.
        Схема відповідає STRATEGY_PROMOTION_GATE_TEMPLATE.
        """
        playbook = self.get(strategy_id)
        receipt_ok, receipt_issues = verify_gate_receipt(
            gate_receipt,
            expected_strategy_id=strategy_id,
            expected_target_gate=target_level.value,
        )
        can_promote, issues = playbook.can_promote_to(
            target_level,
            approval_present=receipt_ok,
        )
        issues.extend(receipt_issues)
        if approver and receipt_ok and gate_receipt.get("approver") != approver:
            issues.append("approver_does_not_match_gate_receipt")

        # Правила з Codex: no_direct_research_to_live
        current = playbook.promotion_policy.current_maturity_level
        live_levels = {MaturityLevel.SHADOW, MaturityLevel.SUPERVISED_LIVE, MaturityLevel.CONSTRAINED_AUTONOMOUS}
        if current == MaturityLevel.RESEARCH and target_level in live_levels:
            issues.append("no_direct_research_to_live: must pass replay and paper first")
            can_promote = False

        if issues or not receipt_ok:
            decision_status = "blocked"
        else:
            decision_status = "approved"

        return {
            "strategy_id": strategy_id,
            "from_level": current.value,
            "to_level": target_level.value,
            "decision": {
                "status": decision_status,
                "issues": issues,
                "approver": gate_receipt.get("approver") if receipt_ok else None,
                "gate_receipt_sha256": (
                    gate_receipt.get("receipt_sha256") if receipt_ok else None
                ),
            },
        }

    # ── Block / Deprecation ────────────────────────────────────────────────────

    def block(self, strategy_id: str, reason: str) -> None:
        """Блокує стратегію (встановлює статус REJECTED)."""
        playbook = self.get(strategy_id)
        playbook.status = StrategyStatus.REJECTED
        # Зберігаємо причину в description.thesis
        playbook.description.thesis = f"[BLOCKED: {reason}] " + playbook.description.thesis

    def deprecate(self, strategy_id: str, reason: str) -> None:
        """Позначає стратегію як застарілу."""
        playbook = self.get(strategy_id)
        playbook.status = StrategyStatus.DEPRECATED
        playbook.description.thesis = f"[DEPRECATED: {reason}] " + playbook.description.thesis

    # ── Persistence ───────────────────────────────────────────────────────────

    def save_to_disk(self) -> None:
        """Зберігає всі плейбуки у JSON-файли на диску."""
        self._registry_dir.mkdir(parents=True, exist_ok=True)
        for strategy_id, playbook in self._playbooks.items():
            path = self._registry_dir / f"{strategy_id}.json"
            path.write_text(playbook.model_dump_json(indent=2), encoding="utf-8")

    def load_from_disk(self) -> int:
        """Завантажує всі плейбуки з диску. Повертає кількість завантажених."""
        if not self._registry_dir.exists():
            return 0
        count = 0
        for path in self._registry_dir.glob("*.json"):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                playbook = StrategyPlaybook.model_validate(data)
                self._playbooks[playbook.strategy_id] = playbook
                count += 1
            except Exception:
                pass
        return count

    def summary(self) -> dict:
        status_counts: dict[str, int] = {}
        for p in self._playbooks.values():
            status_counts[p.status.value] = status_counts.get(p.status.value, 0) + 1
        return {
            "total_strategies": len(self._playbooks),
            "by_status": status_counts,
        }
