import sys
import asyncio
from src.config.unified_config_manager import UnifiedConfigManager
from src.pipeline.stages.stage_4_modeling import ModelingStage

async def test_modeling_stage():
    try:
        config_manager = UnifiedConfigManager()
        stage = ModelingStage(config_manager=config_manager)
        print("ModelingStage successfully instantiated!")
    except Exception as e:
        print(f"Error instantiating ModelingStage: {e}")

if __name__ == "__main__":
    asyncio.run(test_modeling_stage())
