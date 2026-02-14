# models/arena/battle_groups.py - Battle Group Configurations

from typing import List, Dict, Any
from dataclasses import dataclass

@dataclass
class BattleGroup:
    """Конфігурація групи боїв"""
    name: str
    description: str
    models: List[str]
    max_battles_per_model: int
    battle_format: str  # "round_robin", "tournament", "elimination"
    scoring_weights: Dict[str, float]

# Попередньо визначені групи боїв
BATTLE_GROUPS = {
    "traditional_vs_enhanced": BattleGroup(
        name="Traditional vs Enhanced",
        description="Бої між традиційними ML моделями та Enhanced моделями (Dean RL, Sentiment)",
        models=[
            # Traditional Models
            "lgbm", "rf", "xgboost", "catboost", "linear", "mlp", "svm", "knn",
            # Enhanced Models  
            "dean_ensemble", "sentiment", "lgbm_bayesian"
        ],
        max_battles_per_model=3,
        battle_format="round_robin",
        scoring_weights={
            "accuracy": 0.3,
            "sharpe_ratio": 0.25,
            "win_rate": 0.2,
            "max_drawdown": 0.15,
            "confidence_score": 0.1
        }
    ),
    
    "light_vs_heavy": BattleGroup(
        name="Light vs Heavy Models",
        description="Бої між легкими та важкими моделями",
        models=[
            # Light Models
            "lgbm", "rf", "xgboost", "catboost", "linear", "mlp", "svm", "knn",
            # Heavy Models
            "lstm", "gru", "transformer", "cnn", "tabnet", "autoencoder"
        ],
        max_battles_per_model=4,
        battle_format="tournament",
        scoring_weights={
            "accuracy": 0.25,
            "sharpe_ratio": 0.3,
            "win_rate": 0.2,
            "max_drawdown": 0.15,
            "confidence_score": 0.1
        }
    ),
    
    "all_models": BattleGroup(
        name="All Models Battle Royale",
        description="Бої між всіма доступними моделями",
        models=[
            "lgbm", "rf", "xgboost", "catboost", "linear", "mlp", "svm", "knn",
            "lstm", "gru", "transformer", "cnn", "tabnet", "autoencoder",
            "dean_ensemble", "sentiment", "lgbm_bayesian"
        ],
        max_battles_per_model=2,
        battle_format="elimination",
        scoring_weights={
            "accuracy": 0.2,
            "sharpe_ratio": 0.35,
            "win_rate": 0.25,
            "max_drawdown": 0.15,
            "confidence_score": 0.05
        }
    ),
    
    "enhanced_showdown": BattleGroup(
        name="Enhanced Models Showdown",
        description="Бої між Enhanced моделями",
        models=[
            "dean_ensemble", "sentiment", "lgbm_bayesian"
        ],
        max_battles_per_model=5,
        battle_format="round_robin",
        scoring_weights={
            "accuracy": 0.25,
            "sharpe_ratio": 0.3,
            "win_rate": 0.25,
            "max_drawdown": 0.15,
            "confidence_score": 0.05
        }
    ),
    
    "traditional_championship": BattleGroup(
        name="Traditional Models Championship",
        description="Чемпіонат серед традиційних моделей",
        models=[
            "lgbm", "rf", "xgboost", "catboost", "linear", "mlp", "svm", "knn"
        ],
        max_battles_per_model=3,
        battle_format="tournament",
        scoring_weights={
            "accuracy": 0.35,
            "sharpe_ratio": 0.25,
            "win_rate": 0.2,
            "max_drawdown": 0.15,
            "confidence_score": 0.05
        }
    ),
    
    "deep_learning_battle": BattleGroup(
        name="Deep Learning Battle",
        description="Бої між глибокими моделями навчання",
        models=[
            "lstm", "gru", "transformer", "cnn", "tabnet", "autoencoder"
        ],
        max_battles_per_model=4,
        battle_format="round_robin",
        scoring_weights={
            "accuracy": 0.2,
            "sharpe_ratio": 0.4,
            "win_rate": 0.2,
            "max_drawdown": 0.15,
            "confidence_score": 0.05
        }
    ),
    
    "quick_test": BattleGroup(
        name="Quick Test Battle",
        description="Швидкий тест між кількома моделями",
        models=["lgbm", "rf", "xgboost", "dean_ensemble"],
        max_battles_per_model=2,
        battle_format="round_robin",
        scoring_weights={
            "accuracy": 0.3,
            "sharpe_ratio": 0.3,
            "win_rate": 0.2,
            "max_drawdown": 0.15,
            "confidence_score": 0.05
        }
    )
}

class BattleGroupManager:
    """Менеджер груп боїв"""
    
    def __init__(self):
        self.available_groups = BATTLE_GROUPS
        self.custom_groups = {}
    
    def get_group(self, group_name: str) -> BattleGroup:
        """Отримати групу боїв"""
        if group_name in self.available_groups:
            return self.available_groups[group_name]
        elif group_name in self.custom_groups:
            return self.custom_groups[group_name]
        else:
            raise ValueError(f"Battle group '{group_name}' not found")
    
    def list_groups(self) -> List[str]:
        """Список всіх доступних груп"""
        return list(self.available_groups.keys()) + list(self.custom_groups.keys())
    
    def create_custom_group(self, name: str, models: List[str], description: str = "", 
                          battle_format: str = "round_robin", max_battles: int = 3) -> BattleGroup:
        """Створити власну групу боїв"""
        custom_group = BattleGroup(
            name=name,
            description=description or f"Custom group: {name}",
            models=models,
            max_battles_per_model=max_battles,
            battle_format=battle_format,
            scoring_weights={
                "accuracy": 0.3,
                "sharpe_ratio": 0.25,
                "win_rate": 0.2,
                "max_drawdown": 0.15,
                "confidence_score": 0.1
            }
        )
        
        self.custom_groups[name] = custom_group
        return custom_group
    
    def get_recommended_groups(self, available_models: List[str]) -> List[str]:
        """Отримати рекомендовані групи на основі доступних моделей"""
        recommended = []
        
        for group_name, group in self.available_groups.items():
            # Перевіряємо, чи є доступні моделі для цієї групи
            available_in_group = [model for model in group.models if model in available_models]
            
            if len(available_in_group) >= 2:  # Мінімум 2 моделі для боїв
                recommended.append(group_name)
        
        return recommended
    
    def generate_battle_schedule(self, group_name: str, available_models: List[str]) -> List[tuple]:
        """Згенерувати розклад боїв для групи"""
        group = self.get_group(group_name)
        available_in_group = [model for model in group.models if model in available_models]
        
        if len(available_in_group) < 2:
            raise ValueError(f"Not enough models for group '{group_name}': {len(available_in_group)}")
        
        battles = []
        
        if group.battle_format == "round_robin":
            # Кожен з кожним
            for i in range(len(available_in_group)):
                for j in range(i + 1, len(available_in_group)):
                    battles.append((available_in_group[i], available_in_group[j]))
        
        elif group.battle_format == "tournament":
            # Турнірний формат
            battles = []
            for i in range(len(available_in_group)):
                for j in range(i + 1, min(i + group.max_battles_per_model + 1, len(available_in_group))):
                    battles.append((available_in_group[i], available_in_group[j]))
        
        elif group.battle_format == "elimination":
            # Формат вибування
            battles = []
            for i in range(len(available_in_group)):
                for j in range(i + 1, min(i + 2, len(available_in_group))):
                    battles.append((available_in_group[i], available_in_group[j]))
        
        else:
            # За замовчуванням round_robin
            for i in range(len(available_in_group)):
                for j in range(i + 1, len(available_in_group)):
                    battles.append((available_in_group[i], available_in_group[j]))
        
        return battles
    
    def get_group_info(self, group_name: str) -> Dict[str, Any]:
        """Отримати детальну інформацію про групу"""
        try:
            group = self.get_group(group_name)
            
            return {
                'name': group.name,
                'description': group.description,
                'models_count': len(group.models),
                'models': group.models,
                'max_battles_per_model': group.max_battles_per_model,
                'battle_format': group.battle_format,
                'scoring_weights': group.scoring_weights,
                'total_possible_battles': len(group.models) * (len(group.models) - 1) // 2
            }
            
        except Exception as e:
            return {'error': str(e)}

# Глобальні функції
def get_battle_group_manager() -> BattleGroupManager:
    """Отримати глобальний менеджер груп боїв"""
    global _battle_group_manager
    if '_battle_group_manager' not in globals():
        _battle_group_manager = BattleGroupManager()
    return _battle_group_manager

def get_all_battle_groups() -> Dict[str, BattleGroup]:
    """Отримати всі групи боїв"""
    manager = get_battle_group_manager()
    return {**manager.available_groups, **manager.custom_groups}

def get_popular_groups() -> List[str]:
    """Отримати популярні групи боїв"""
    return [
        "traditional_vs_enhanced",
        "light_vs_heavy", 
        "enhanced_showdown",
        "quick_test"
    ]
