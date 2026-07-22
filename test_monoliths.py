from src.pipeline.stages.stage_1_collection import CollectionStage
from src.pipeline.stages.stage_5_prediction import PredictionStage
from src.pipeline.stages.stage_6_trading_execution import TradingExecutionStage
from src.pipeline.stages.stage_7_evaluation import EvaluationStage
from src.config.unified_config_manager import UnifiedConfigManager
import asyncio

async def test_imports():
    try:
        config = UnifiedConfigManager()
        s1 = CollectionStage(config_manager=config)
        s5 = PredictionStage(config_manager=config)
        s6 = TradingExecutionStage(config_manager=config)
        s7 = EvaluationStage(config_manager=config)
        print("All monolith facades successfully instantiated!")
    except Exception as e:
        print(f"Error instantiating stages: {e}")

if __name__ == "__main__":
    asyncio.run(test_imports())
