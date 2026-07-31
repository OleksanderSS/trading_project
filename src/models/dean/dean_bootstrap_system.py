import logging
import time
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any

import numpy as np

"""
!!! ПОТРІБНО ДООПРАЦЮВАТИ !!!
Цей файл містить сиру, експериментальну логіку, натхненну принципами Станіслава Деана.
Він ще не інтегрований в основний робочий процес і потребує подальшого розвитку та тестування.
"""

"""
DEAN BOOTSTRAP SYSTEM
Система на основі принципів Станіслава Деана: бутстреп, критика, внутрішня симуляція
"""

logger = logging.getLogger(__name__)

class ModelRole(Enum):
    """Ролі моделей в системі Деана"""
    ACTOR = "actor"  # Модель, що діє
    CRITIC = "critic"  # Модель, що критикує
    SIMULATOR = "simulator"  # Модель, що симулює
    ADVERSARY = "adversary"  # Модель, що заважає розвитку

@dataclass
class DeanAction:
    """Дія в системі Деана"""
    action_id: str
    action_type: str
    parameters: dict[str, Any]
    confidence: float
    timestamp: datetime
    context: dict[str, Any]

@dataclass
class DeanCritique:
    """Критика дії в системі Деана"""
    action_id: str
    critique_score: float  # -1 до 1
    critique_points: list[str]
    alternative_suggestions: list[dict[str, Any]]
    confidence: float

@dataclass
class DeanSimulation:
    """Внутрішня симуляція ситуації"""
    scenario_id: str
    initial_conditions: dict[str, Any]
    predicted_outcomes: list[dict[str, Any]]
    confidence_distribution: list[float]
    simulation_steps: list[dict[str, Any]]

class DeanBootstrapSystem:
    """
    Основна система на основі принципів Деана:
    1. Бутстреп - одночасна дія та критика
    2. Адверсаріальний розвиток - одна розвивається, інша заважає
    3. Внутрішня симуляція - прогнозування ситуацій
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.models = {}
        self.action_history = {} # Keyed by action_id
        self.critique_history = {} # Keyed by action_id
        self.simulation_history = []
        self.reward_history = []
        self.evolution_metrics = {
            'actor_performance': [],
            'critic_accuracy': [],
            'simulation_precision': [],
            'adversarial_pressure': [],
            'actor_rewards': [],
            'critic_rewards': []
        }
        # ✅ Integrated: security constraint validation for agent actions
        try:
            from src.meta_learning.security.constraint_engine import get_security_constraint_engine
            self.security_engine = get_security_constraint_engine()
            self.logger.info("✅ SecurityConstraintEngine integrated in DeanBootstrapSystem")
        except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError) as e:
            self.security_engine = None
            self.logger.warning(f"SecurityConstraintEngine not available: {e}")

    def register_model(self, model_id: str, role: ModelRole, model_instance: Any):
        """Реєстрація моделі в системі"""
        self.models[model_id] = {
            'role': role,
            'instance': model_instance,
            'performance_history': [],
            'evolution_stage': 0
        }
        self.logger.info(f"[BRAIN] Registered {role.value} model: {model_id}")

    def bootstrap_action_critique(self, context: dict[str, Any]) -> tuple[DeanAction, DeanCritique]:
        """
        Бутстреп: одночасно дія та критика з механізмом зворотного зв'язку
        """
        actor_models = [m for m in self.models.values() if m['role'] == ModelRole.ACTOR]
        critic_models = [m for m in self.models.values() if m['role'] == ModelRole.CRITIC]

        if not actor_models or not critic_models:
            raise ValueError("Need both actor and critic models for bootstrap")

        # 1. Актор генерує дію
        actor = actor_models[0]['instance']
        action = self._generate_action(actor, context)

        # 2. Критик одночасно аналізує дію
        critic = critic_models[0]['instance']
        critique = self._generate_critique(critic, action, context)

        # Записуємо в історію
        self.action_history[action.action_id] = action
        self.critique_history[action.action_id] = critique

        self.logger.info(f"[DRAMA] Bootstrap: Action={action.action_type} (ID: {action.action_id}), Critique={critique.critique_score:.2f}")

        return action, critique

    def critique_existing_action(
        self,
        action_type: str,
        confidence: float,
        context: dict[str, Any],
        features: Any = None,
    ) -> tuple[DeanAction, DeanCritique]:
        """Critique a decision that has ALREADY been made elsewhere.

        `bootstrap_action_critique` requires a registered ACTOR model, because
        it generates the action itself. On the live path that actor does not
        exist and should not: by the time ConsensusEngine reaches its critic
        filter the signal is already decided, so the consensus IS the actor.
        Registering a second, separate DeanActor purely to satisfy that guard
        would be architecture by accident.

        This wraps the existing decision in a DeanAction so the reward loop
        (`calculate_reward`) keys off the same `action_id` as the bootstrap
        path, then runs only the critic.
        """
        critic_models = [m for m in self.models.values() if m['role'] == ModelRole.CRITIC]
        if not critic_models:
            raise ValueError("No critic model registered")

        action = DeanAction(
            action_id=f"act_{int(time.time() * 1000)}",
            action_type=str(action_type).lower(),
            parameters={'ticker': context.get('ticker')},
            confidence=float(confidence),
            timestamp=datetime.now(),
            context=context,
        )

        critic = critic_models[0]['instance']
        critique_data = critic.critique_action(action, context, features)
        critique = DeanCritique(
            action_id=action.action_id,
            critique_score=critique_data['score'],
            critique_points=critique_data['points'],
            alternative_suggestions=critique_data['alternatives'],
            confidence=critique_data['confidence'],
        )

        self.action_history[action.action_id] = action
        self.critique_history[action.action_id] = critique
        return action, critique

    def calculate_reward(self, action_id: str, outcome: dict[str, Any], confidence_bonus: bool = True):
        """
        Розрахунок винагороди для Актора та Критика на основі результату
        """
        if action_id not in self.action_history or action_id not in self.critique_history:
            self.logger.warning(f"Action {action_id} not found in history. Cannot calculate reward.")
            return

        action = self.action_history[action_id]
        critique = self.critique_history[action_id]

        pnl = outcome.get('pnl', 0.0)
        outcome.get('risk_reduced', False)

        # 1. Винагорода Актора
        actor_reward = pnl
        if pnl > 0 and confidence_bonus:
            actor_reward *= (1 + action.confidence)

        # 2. Винагорода Критика
        critic_reward = 0.0

        # Випадок А: Критик був правий, попередивши про збиток (Critique Score низький, PnL був би негативний)
        if critique.critique_score < 0 and pnl <= 0:
            critic_reward = abs(pnl) * critique.confidence # Нагорода за врятовані кошти

        # Випадок Б: Критик помилився, заблокувавши прибутковий трейд (False Negative)
        elif critique.critique_score < 0 and pnl > 0:
            critic_reward = -pnl # Штраф за втрачений прибуток

        # Випадок В: Критик підтримав прибутковий трейд
        elif critique.critique_score > 0 and pnl > 0:
            critic_reward = pnl * 0.5 # Нагорода за підтвердження правильного рішення

        # Випадок Г: Критик підтримав збитковий трейд (False Positive)
        elif critique.critique_score > 0 and pnl < 0:
            critic_reward = pnl * 1.5 # Сильний штраф за пропуск ризику

        reward_data = {
            'action_id': action_id,
            'actor_reward': actor_reward,
            'critic_reward': critic_reward,
            'pnl': pnl,
            'timestamp': datetime.now()
        }

        self.reward_history.append(reward_data)
        self.evolution_metrics['actor_rewards'].append(actor_reward)
        self.evolution_metrics['critic_rewards'].append(critic_reward)

        self.logger.info(f"[BRAIN] Reward Calculated for {action_id}: Actor={actor_reward:.4f}, Critic={critic_reward:.4f}")
        return reward_data

    def adversarial_evolution(self, training_data: dict[str, Any]) -> dict[str, float]:
        """
        Адверсаріальний розвиток: одна model розвивається, інша заважає
        """
        actor_models = [m for m in self.models.values() if m['role'] == ModelRole.ACTOR]
        adversary_models = [m for m in self.models.values() if m['role'] == ModelRole.ADVERSARY]

        evolution_results = {}

        for actor in actor_models:
            actor_id = actor['instance'].get_id()

            # 1. Актор намагається покращитися
            actor_improvement = self._train_actor(actor['instance'], training_data)

            # 2. Адверсарій намагається заважати
            adversary_pressure = 0.0
            for adversary in adversary_models:
                pressure = self._apply_adversarial_pressure(adversary['instance'], actor['instance'])
                adversary_pressure += pressure

            # 3. Результат еволюції
            net_improvement = actor_improvement - adversary_pressure
            evolution_results[actor_id] = net_improvement

            # 4. Оновлюємо метрики
            self.evolution_metrics['actor_performance'].append(actor_improvement)
            self.evolution_metrics['adversarial_pressure'].append(adversary_pressure)

            self.logger.info(f" Evolution: {actor_id} improvement={actor_improvement:.3f}, pressure={adversary_pressure:.3f}")

        return evolution_results

    def internal_simulation(self, scenario: dict[str, Any]) -> DeanSimulation:
        """
        Внутрішня симуляція: як мозок людини прогнозує ситуації
        """
        simulator_models = [m for m in self.models.values() if m['role'] == ModelRole.SIMULATOR]

        if not simulator_models:
            raise ValueError("Need simulator model for internal simulation")

        simulator = simulator_models[0]['instance']

        # 1. Створюємо сценарій симуляції
        simulation_id = f"sim_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # 2. Симулюємо множинні майбутні сценарії
        predicted_outcomes = []
        confidence_distribution = []
        simulation_steps = []

        # Симуляція різних часових горизонтів
        time_horizons = [1, 5, 10, 20]  # дні

        for horizon in time_horizons:
            outcome = self._simulate_scenario(simulator, scenario, horizon)
            predicted_outcomes.append(outcome)
            confidence_distribution.append(outcome['confidence'])
            simulation_steps.append({
                'horizon': horizon,
                'state': outcome['state'],
                'key_factors': outcome['key_factors']
            })

        # 3. Створюємо об'єкт симуляції
        simulation = DeanSimulation(
            scenario_id=simulation_id,
            initial_conditions=scenario,
            predicted_outcomes=predicted_outcomes,
            confidence_distribution=confidence_distribution,
            simulation_steps=simulation_steps
        )

        self.simulation_history.append(simulation)

        self.logger.info(f" Internal simulation: {len(predicted_outcomes)} outcomes, avg_confidence={np.mean(confidence_distribution):.2f}")

        return simulation

    def _generate_action(self, actor_model, context: dict[str, Any]) -> DeanAction:
        """Генерація дії актором"""
        action_data = actor_model.decide_action(context)
        action_id = f"act_{int(time.time() * 1000)}"

        return DeanAction(
            action_id=action_id,
            action_type=action_data['type'],
            parameters=action_data['parameters'],
            confidence=action_data['confidence'],
            timestamp=datetime.now(),
            context=context
        )

    def _generate_critique(self, critic_model, action: DeanAction, context: dict[str, Any]) -> DeanCritique:
        """Генерація критики критиком"""
        critique_data = critic_model.critique_action(action, context)

        return DeanCritique(
            action_id=action.action_id,
            critique_score=critique_data['score'],
            critique_points=critique_data['points'],
            alternative_suggestions=critique_data['alternatives'],
            confidence=critique_data['confidence']
        )

    def _train_actor(self, actor_model, training_data: dict[str, Any]) -> float:
        """Тренування актора"""
        improvement = actor_model.train(training_data)
        return improvement

    def _apply_adversarial_pressure(self, adversary_model, actor_model) -> float:
        """Застосування адверсаріального тиску"""
        pressure = adversary_model.apply_pressure(actor_model)
        return pressure

    def _simulate_scenario(self, simulator_model, scenario: dict[str, Any], horizon: int) -> dict[str, Any]:
        """Симуляція сценарію на певному часовому горизонті"""
        outcome = simulator_model.simulate(scenario, horizon)
        return outcome

    def get_evolution_summary(self) -> dict[str, Any]:
        """Отримати звіт про еволюцію системи"""
        return {
            'total_actions': len(self.action_history),
            'total_critiques': len(self.critique_history),
            'total_rewards': len(self.reward_history),
            'evolution_metrics': self.evolution_metrics,
            'current_performance': self._calculate_current_performance(),
            'learning_rate': self._calculate_learning_rate()
        }

    def _calculate_current_performance(self) -> float:
        """Розрахунок поточної продуктивності"""
        if not self.evolution_metrics['actor_performance']:
            return 0.0
        return np.mean(self.evolution_metrics['actor_performance'][-10:])  # останні 10

    def _calculate_learning_rate(self) -> float:
        """Розрахунок швидкості навчання"""
        if len(self.evolution_metrics['actor_performance']) < 2:
            return 0.0

        recent = self.evolution_metrics['actor_performance'][-5:]
        older = self.evolution_metrics['actor_performance'][-10:-5]

        return np.mean(recent) - np.mean(older)


# Глобальна система Деана
_dean_system = None

def get_dean_system() -> DeanBootstrapSystem:
    """Отримати глобальну систему Деана"""
    global _dean_system
    if _dean_system is None:
        _dean_system = DeanBootstrapSystem()
    return _dean_system

def get_dean_bootstrap_system() -> DeanBootstrapSystem:
    """Отримати глобальну систему Деана (alias for compatibility)"""
    return get_dean_system()
