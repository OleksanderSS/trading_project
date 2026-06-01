import sys
from unittest.mock import MagicMock

# Додаємо шлях до проекту
sys.path.append("D:\\trading_project")

# Mocking dependencies
mock_config = MagicMock()

try:
    from src.pipeline.hybrid_orchestrator import HybridOrchestrator
    
    print("Testing HybridOrchestrator initialization...")
    # Initialize with mock config
    ho = HybridOrchestrator(mock_config, batch_name="test_batch")
    
    print("Orchestrator initialized successfully.")
    
    # Check if methods still exist
    required_methods = ['run_local_pipeline', 'run_light_models', 'prepare_colab_data']
    for method in required_methods:
        if hasattr(ho, method):
            print(f"Method '{method}' exists.")
        else:
            print(f"Method '{method}' MISSING!")
            sys.exit(1)
            
    print("Integrity validation passed.")
    sys.exit(0)

except Exception as e:
    print(f"Validation FAILED: {e}")
    sys.exit(1)
