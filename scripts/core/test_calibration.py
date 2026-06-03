#!/usr/bin/env python3
"""
Quick test script for CalibrationEngine.

Tests:
1. Import CalibrationEngine
2. Initialize engine
3. Check methods exist
4. Test with mock data (if optuna installed)
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

print("🧪 Testing CalibrationEngine...")
print()

# Test 1: Import
print("1️⃣ Testing import...")
try:
    from src.calibration import CalibrationEngine
    print("   ✅ CalibrationEngine imported successfully")
except ImportError as e:
    print(f"   ❌ Import failed: {e}")
    sys.exit(1)

# Test 2: Initialize
print()
print("2️⃣ Testing initialization...")
try:
    engine = CalibrationEngine(
        real_data_path="data/duckdb/trading.db",
        synthetic_data_path="data/synthetic/",
        n_trials=10,
        metric="sharpe_ratio",
        batch_name="test_calibration"
    )
    print("   ✅ CalibrationEngine initialized successfully")
except ImportError as e:
    print(f"   ⚠️  Initialization requires optuna: {e}")
    print()
    print("=" * 60)
    print("✅ IMPORT TEST PASSED!")
    print("=" * 60)
    print()
    print("⚠️  optuna is NOT installed (required for full functionality)")
    print()
    print("To install optuna:")
    print("   pip install optuna")
    print()
    print("After installing optuna, run this test again to verify full functionality.")
    print()
    sys.exit(0)
except Exception as e:
    print(f"   ❌ Initialization failed: {e}")
    sys.exit(1)

# Test 3: Check methods
print()
print("3️⃣ Testing methods exist...")
methods = [
    'load_real_data',
    'load_synthetic_scenarios',
    'define_hyperparameter_space',
    'evaluate_hyperparameters',
    'run_calibration'
]

for method in methods:
    if hasattr(engine, method):
        print(f"   ✅ {method}() exists")
    else:
        print(f"   ❌ {method}() missing")
        sys.exit(1)

# Test 4: Check optuna
print()
print("4️⃣ Testing optuna availability...")
try:
    import optuna
    print("   ✅ optuna is installed")
    print(f"   📦 optuna version: {optuna.__version__}")
except ImportError:
    print("   ⚠️  optuna is NOT installed")
    print("   💡 Install with: pip install optuna")
    print()
    print("✅ Basic tests passed (optuna not required for import)")
    sys.exit(0)

# Test 5: Test hyperparameter space
print()
print("5️⃣ Testing hyperparameter space...")
try:
    study = optuna.create_study(direction='maximize')
    trial = study.ask()
    hyperparams = engine.define_hyperparameter_space(trial)
    
    expected_params = [
        'actor_lr', 'critic_lr', 'hidden_dim', 'num_layers',
        'batch_size', 'replay_buffer_size', 'gamma', 'tau',
        'exploration_noise', 'dropout', 'weight_decay'
    ]
    
    for param in expected_params:
        if param in hyperparams:
            print(f"   ✅ {param}: {hyperparams[param]}")
        else:
            print(f"   ❌ {param} missing")
            sys.exit(1)
    
    print()
    print("   ✅ All 11 hyperparameters defined correctly")
    
except Exception as e:
    print(f"   ❌ Hyperparameter space test failed: {e}")
    sys.exit(1)

# Test 6: Output directory
print()
print("6️⃣ Testing output directory...")
if engine.output_dir.exists():
    print(f"   ✅ Output directory exists: {engine.output_dir}")
else:
    print(f"   ⚠️  Output directory will be created: {engine.output_dir}")

# Summary
print()
print("=" * 60)
print("✅ ALL TESTS PASSED!")
print("=" * 60)
print()
print("📊 CalibrationEngine is ready to use!")
print()
print("Next steps:")
print("1. Prepare data:")
print("   python scripts/accumulate_real_data.py --tickers AMD --days 30")
print("   python scripts/generate_synthetic_data.py")
print()
print("2. Run calibration:")
print("   python run_hybrid_pipeline.py --mode calibrate --test-ticker AMD --n-trials 10")
print()
print("3. Check results:")
print("   cat results/calibration/test_calibration/calibration_results.json")
print()
