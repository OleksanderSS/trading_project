import logging
import asyncio
from typing import Any
import pandas as pd

from src.main.modes.base import BaseMode
from src.core.logging.logger import ProjectLogger
from src.data.collectors.synthetic_generator import SyntheticGenerator, ScenarioConfig, BUILTIN_SCENARIOS, GeneratorConfig
from src.analytics.arena.arena_battle import TradingModelArena
import joblib
from pathlib import Path

class ShadowBattleMode(BaseMode):
    """
    Режим Shadow Battle: запускає синтетичні стрес-сценарії (Black Swan, Flash Crash)
    і перевіряє, як на них реагують різні моделі через модуль Arena.
    """
    
    def __init__(self, config_manager: Any):
        self.config_manager = config_manager
        self.logger = ProjectLogger.get_logger(__name__)
        
    def run(self, **kwargs) -> dict[str, Any]:
        self.logger.info("--- Starting SHADOW BATTLE Mode ---")
        try:
            # 1. Initialize Arena
            arena = TradingModelArena()

            # 2. Generate Black Swan data
            self.logger.info("[ShadowBattle] Generating Black Swan scenario...")
            gen_config = GeneratorConfig(random_seed=42)
            generator = SyntheticGenerator(config=gen_config)
            scenario_data = generator.generate_scenarios(scenario_names=['black_swan'])
            # Extract the first generated path for black swan
            price_df = scenario_data['black_swan'][0]
            
            # 4. We need to create dummy targets and basic features for the models to predict
            # Just computing SMA and basic returns for testing
            features_df = price_df.copy()
            features_df['returns'] = features_df['close'].pct_change().fillna(0)
            features_df['volatility_10'] = features_df['returns'].rolling(10).std().fillna(0)
            
            # 5. Load pre-trained Champions
            trained_models_dir = Path("d:/trading_project/data/trained_models")
            
            # Load models
            models_to_test = {}
            for model_path in trained_models_dir.glob("model_*_target_*.joblib"):
                try:
                    model = joblib.load(model_path)
                    model_name = model_path.stem.split("_")[-1]  # Extract algorithm name (xgboost, linear, etc.)
                    models_to_test[model_name] = model
                except Exception as e:
                    self.logger.warning(f"Could not load model {model_path}: {e}")
                    
            if not models_to_test:
                self.logger.warning("No pre-trained models found for SPY. Creating dummy model for test.")
                class DummyModel:
                    def predict(self, df): return [1] * len(df)
                    def predict_proba(self, df): return [[0.5, 0.5]] * len(df)
                models_to_test['dummy'] = DummyModel()
            
            # 6. Run Arena Battle
            self.logger.info("[ShadowBattle] Commencing Arena Battles on Synthetic Data...")
            
            battle_results = []
            
            # Dummy actual targets (assume market goes down in black swan)
            actual_targets = pd.Series(-1, index=features_df.index)
            
            for name, model in models_to_test.items():
                self.logger.info(f"Registering candidate: {name}")
                arena.register_model(name, model)
                
                try:
                    result = arena.run_blind_challenge(name, features_df, actual_targets)
                    battle_results.append({
                        "model": name,
                        "alignment": result.get("structural_alignment", 0),
                        "gap": result.get("realization_gap", 0)
                    })
                except Exception as e:
                    self.logger.error(f"Error during battle for {name}: {e}")
                    
            self.logger.info(f"Battle Results: {battle_results}")
            
            return {"status": "success", "results": battle_results}
            
        except Exception as e:
            self.logger.exception(f"[ShadowBattle] Error: {e}")
            return {"status": "failed", "error": str(e)}
