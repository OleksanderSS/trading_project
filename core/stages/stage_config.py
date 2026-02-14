# core/stages/stage_config.py - Конфandгурацandя and сandндартиforцandя еandпandв

from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum

class StageType(Enum):
    """Типи еandпandв обробки"""
    DATA_COLLECTION = "data_collection"
    DATA_ENRICHMENT = "data_enrichment"
    FEATURE_ENGINEERING = "feature_engineering"
    MODEL_TRAINING = "model_training"
    SIGNAL_GENERATION = "signal_generation"

class ModelType(Enum):
    """Типи моwhereлей"""
    LIGHT = "light"  # Легкand моwhereлand - локальnot тренування
    HEAVY = "heavy"  # Важкand моwhereлand - Colab тренування

@dataclass
class StageConfig:
    """Конфandгурацandя еandпу"""
    name: str
    stage_type: StageType
    dependencies: List[str]  # Еandпи, вandд яких forлежить
    cache_enabled: bool = True
    max_cache_age_hours: int = 24
    parallel_execution: bool = False
    required_inputs: List[str] = None
    
    def __post_init__(self):
        if self.required_inputs is None:
            self.required_inputs = []

# Сandндартна конфandгурацandя еandпandв
STAGE_CONFIGS = {
    "stage_0": StageConfig(
        name="environment_init",
        stage_type=StageType.DATA_COLLECTION,
        dependencies=[],
        cache_enabled=False,
        parallel_execution=False
    ),
    
    "stage_1": StageConfig(
        name="data_collection",
        stage_type=StageType.DATA_COLLECTION,
        dependencies=["stage_0"],
        cache_enabled=True,
        max_cache_age_hours=6,
        parallel_execution=True,
        required_inputs=["tickers", "timeframes", "date_range"]
    ),
    
    "stage_2": StageConfig(
        name="data_enrichment",
        stage_type=StageType.DATA_ENRICHMENT,
        dependencies=["stage_1"],
        cache_enabled=True,
        max_cache_age_hours=12,
        parallel_execution=False,
        required_inputs=["raw_news", "price_data", "macro_data"]
    ),
    
    "stage_3": StageConfig(
        name="feature_engineering",
        stage_type=StageType.FEATURE_ENGINEERING,
        dependencies=["stage_2"],
        cache_enabled=True,
        max_cache_age_hours=24,
        parallel_execution=True,
        required_inputs=["enriched_data", "technical_indicators"]
    ),
    
    "stage_4_light": StageConfig(
        name="light_model_training",
        stage_type=StageType.MODEL_TRAINING,
        dependencies=["stage_3"],
        cache_enabled=True,
        max_cache_age_hours=48,
        parallel_execution=True,
        required_inputs=["features_data"]
    ),
    
    "stage_4_heavy": StageConfig(
        name="heavy_model_training",
        stage_type=StageType.MODEL_TRAINING,
        dependencies=["stage_3"],
        cache_enabled=True,
        max_cache_age_hours=168,  # 7 днandв
        parallel_execution=False,
        required_inputs=["features_data"]
    ),
    
    "stage_5": StageConfig(
        name="signal_generation",
        stage_type=StageType.SIGNAL_GENERATION,
        dependencies=["stage_4_light", "stage_4_heavy"],
        cache_enabled=True,
        max_cache_age_hours=1,
        parallel_execution=False,
        required_inputs=["light_models", "heavy_models", "current_data"]
    )
}

class StageValidator:
    """Валandдатор еandпandв"""
    
    @staticmethod
    def validate_stage_config(stage_name: str, config: StageConfig) -> bool:
        """Валandдує конфandгурацandю еandпу"""
        if not config.name:
            return False
        
        if not config.stage_type:
            return False
        
        if not isinstance(config.dependencies, list):
            return False
        
        return True
    
    @staticmethod
    def validate_dependencies(stage_name: str, dependencies: List[str]) -> bool:
        """Перевandряє чи all forлежностand andснують"""
        for dep in dependencies:
            if dep not in STAGE_CONFIGS:
                return False
        return True
    
    @staticmethod
    def get_execution_order(stage_names: List[str]) -> List[str]:
        """Поверandє правильний порядок виконання еandпandв"""
        ordered = []
        remaining = stage_names.copy()
        
        while remaining:
            added = False
            for stage in remaining[:]:
                config = STAGE_CONFIGS.get(stage)
                if config:
                    deps_met = all(dep in ordered for dep in config.dependencies)
                    if deps_met:
                        ordered.append(stage)
                        remaining.remove(stage)
                        added = True
                        break
            
            if not added:
                raise ValueError(f"Circular dependency detected in stages: {remaining}")
        
        return ordered

def get_stage_config(stage_name: str) -> Optional[StageConfig]:
    """Отримати конфandгурацandю еandпу"""
    return STAGE_CONFIGS.get(stage_name)

def get_all_stage_configs() -> Dict[str, StageConfig]:
    """Отримати all конфandгурацandї еandпandв"""
    return STAGE_CONFIGS.copy()

def get_light_models() -> List[str]:
    """Список легких моwhereлей (локальnot тренування)"""
    return [
        "linear_regression",
        "random_forest", 
        "xgboost",
        "lightgbm",
        "catboost",
        "svm",
        "knn"
    ]

def get_heavy_models() -> List[str]:
    """Список важких моwhereлей (Colab тренування)"""
    return [
        "lstm",
        "gru",
        "transformer",
        "cnn",
        "autoencoder",
        "tabnet",
        "deep_ensemble"
    ]
