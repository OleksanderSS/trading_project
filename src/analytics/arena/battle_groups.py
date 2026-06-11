from dataclasses import dataclass
from typing import Any

from src.models.registry.model_registry import ModelRegistry


@dataclass
class BattleGroup:
    """Конфігурація групи боїв"""
    name: str
    description: str
    models: list[str]
    max_battles_per_model: int
    battle_format: str
    scoring_weights: dict[str, float]


# Use ModelRegistry to define groups dynamically where possible
_all_models = ModelRegistry.get_all_model_names()
_light_models = ModelRegistry.get_models_by_type('light')
_heavy_models = ModelRegistry.get_models_by_type('heavy')
_enhanced_models = ModelRegistry.get_models_by_type('enhanced')

BATTLE_GROUPS = {
    'traditional_vs_enhanced': BattleGroup(
        name='Traditional vs Enhanced',
        description='Бої між традиційними ML моделями та Enhanced моделями',
        models=_light_models + _enhanced_models,
        max_battles_per_model=3,
        battle_format='round_robin',
        scoring_weights={'accuracy': 0.3, 'sharpe_ratio': 0.25, 'win_rate': 0.2, 'max_drawdown': 0.15, 'confidence_score': 0.1}
    ),
    'light_vs_heavy': BattleGroup(
        name='Light vs Heavy Models',
        description='Бої між легкими та важкими моделями',
        models=_all_models,
        max_battles_per_model=4,
        battle_format='tournament',
        scoring_weights={'accuracy': 0.25, 'sharpe_ratio': 0.3, 'win_rate': 0.2, 'max_drawdown': 0.15, 'confidence_score': 0.1}
    ),
    'all_models': BattleGroup(
        name='All Models Battle Royale',
        description='Бої між всіма доступними моделями',
        models=_all_models,
        max_battles_per_model=2,
        battle_format='elimination',
        scoring_weights={'accuracy': 0.2, 'sharpe_ratio': 0.35, 'win_rate': 0.25, 'max_drawdown': 0.15, 'confidence_score': 0.05}
    ),
    'deep_learning_battle': BattleGroup(
        name='Deep Learning Battle',
        description='Бої між глибокими моделями навчання',
        models=_heavy_models,
        max_battles_per_model=4,
        battle_format='round_robin',
        scoring_weights={'accuracy': 0.2, 'sharpe_ratio': 0.4, 'win_rate': 0.2, 'max_drawdown': 0.15, 'confidence_score': 0.05}
    ),
    # ... (other groups can be refactored similarly)
}


class BattleGroupManager:
    """Менеджер груп боїв"""

    def __init__(self):
        self.available_groups = BATTLE_GROUPS
        self.custom_groups = {}

    def get_group(self, group_name: str) ->BattleGroup:
        """Отримати групу боїв"""
        if group_name in self.available_groups:
            return self.available_groups[group_name]
        elif group_name in self.custom_groups:
            return self.custom_groups[group_name]
        else:
            raise ValueError(f"Battle group '{group_name}' not found")

    def list_groups(self) ->list[str]:
        """Список всіх доступних груп"""
        return list(self.available_groups.keys()) + list(self.custom_groups
            .keys())

    def create_custom_group(self, name: str, models: list[str], description:
        str='', battle_format: str='round_robin', max_battles: int=3
        ) ->BattleGroup:
        """Створити власну групу боїв"""
        custom_group = BattleGroup(name=name, description=description or
            f'Custom group: {name}', models=models, max_battles_per_model=
            max_battles, battle_format=battle_format, scoring_weights={
            'accuracy': 0.3, 'sharpe_ratio': 0.25, 'win_rate': 0.2,
            'max_drawdown': 0.15, 'confidence_score': 0.1})
        self.custom_groups[name] = custom_group
        return custom_group

    def get_recommended_groups(self, available_models: list[str]) ->list[str]:
        """Отримати рекомендовані групи на основі доступних моделей"""
        recommended = []
        for group_name, group in self.available_groups.items():
            available_in_group = [model for model in group.models if model in
                available_models]
            if len(available_in_group) >= 2:
                recommended.append(group_name)
        return recommended

    def generate_battle_schedule(self, group_name: str, available_models:
        list[str]) ->list[tuple]:
        """Згенерувати розклад боїв для групи"""
        group = self.get_group(group_name)
        available_in_group = [model for model in group.models if model in
            available_models]
        if len(available_in_group) < 2:
            raise ValueError(
                f"Not enough models for group '{group_name}': {len(available_in_group)}"
                )
        battle_format = group.battle_format or 'round_robin'
        if battle_format == 'round_robin':
            return self._generate_round_robin_battles(available_in_group)
        elif battle_format == 'tournament':
            return self._generate_tournament_battles(available_in_group,
                group.max_battles_per_model)
        elif battle_format == 'elimination':
            return self._generate_elimination_battles(available_in_group)
        else:
            return self._generate_round_robin_battles(available_in_group)

    def _generate_round_robin_battles(self, available_models: list[str]
        ) ->list[tuple]:
        """Згенерувати бої в форматі round-robin"""
        battles = []
        for i in range(len(available_models)):
            for j in range(i + 1, len(available_models)):
                battles.append((available_models[i], available_models[j]))
        return battles

    def _generate_tournament_battles(self, available_models: list[str],
        max_battles_per_model: int) ->list[tuple]:
        """Згенерувати бої в турнірному форматі"""
        battles = []
        for i in range(len(available_models)):
            for j in range(i + 1, min(i + max_battles_per_model + 1, len(
                available_models))):
                battles.append((available_models[i], available_models[j]))
        return battles

    def _generate_elimination_battles(self, available_models: list[str]
        ) ->list[tuple]:
        """Згенерувати бої в форматі вибування"""
        battles = []
        for i in range(len(available_models)):
            for j in range(i + 1, min(i + 2, len(available_models))):
                battles.append((available_models[i], available_models[j]))
        return battles

    def get_group_info(self, group_name: str) ->dict[str, Any]:
        """Отримати детальну інформацію про групу"""
        try:
            group = self.get_group(group_name)
            return {'name': group.name, 'description': group.description,
                'models_count': len(group.models), 'models': group.models,
                'max_battles_per_model': group.max_battles_per_model,
                'battle_format': group.battle_format, 'scoring_weights':
                group.scoring_weights, 'total_possible_battles': len(group.
                models) * (len(group.models) - 1) // 2}
        except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError) as e:
            self.logger.error(f'Виникла помилка: {e}', exc_info=True)
            return {'error': str(e)}


def get_battle_group_manager() ->BattleGroupManager:
    """Отримати глобальний менеджер груп боїв"""
    global _battle_group_manager
    if '_battle_group_manager' not in globals():
        _battle_group_manager = BattleGroupManager()
    return _battle_group_manager


def get_all_battle_groups() ->dict[str, BattleGroup]:
    """Отримати всі групи боїв"""
    manager = get_battle_group_manager()
    return {**manager.available_groups, **manager.custom_groups}


def get_popular_groups() ->list[str]:
    """Отримати популярні групи боїв"""
    return ['traditional_vs_enhanced', 'light_vs_heavy',
        'enhanced_showdown', 'quick_test']
