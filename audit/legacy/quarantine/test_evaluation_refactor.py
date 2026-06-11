import asyncio
import pandas as pd
import numpy as np
from src.pipeline.stages.stage_7_evaluation import EvaluationStage
from src.config.unified_config_manager import UnifiedConfigManager
from src.core.error_handling.error_handler import ErrorHandler

async def test_evaluation_stage_refactor():
    print("Testing modular EvaluationStage...")
    
    config_manager = UnifiedConfigManager()
    error_handler = ErrorHandler()
    
    stage = EvaluationStage(config_manager, error_handler)
    
    # Create mock signals data
    dates = pd.date_range(start='2024-01-01', periods=100)
    signals_df = pd.DataFrame({
        'timestamp': dates,
        'ticker': 'BTC',
        'price': np.random.randn(100).cumsum() + 100,
        'predictions': np.random.randn(100)
    })
    
    # Run stage
    print("Running stage.run()...")
    results = await stage.run(signals=signals_df)
    
    print(f"Stage Results keys: {list(results.keys())}")
    assert 'evaluation_summary' in results
    
    print("✅ Modular EvaluationStage tests passed!")

if __name__ == "__main__":
    asyncio.run(test_evaluation_stage_refactor())
