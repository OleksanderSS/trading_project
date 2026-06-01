import importlib
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.analytics.interfaces import IAnalyzer
from src.analytics.unified_analytics_engine import UnifiedAnalyticsEngine
from src.config.unified_config_manager import UnifiedConfigManager

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger("AnalyticsDiagnostics")

def scan_analytics_directory(base_path: Path) -> List[Dict[str, Any]]:
    """Scans the analytics directory for potential analyzer classes."""
    potential_analyzers = []
    
    for py_file in base_path.glob("**/*.py"):
        if py_file.name == "__init__.py" or py_file.name == "interfaces.py":
            continue
            
        module_path = str(py_file.relative_to(project_root)).replace(os.path.sep, ".").replace(".py", "")
        
        try:
            module = importlib.import_module(module_path)
            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                if isinstance(attr, type) and issubclass(attr, IAnalyzer) and attr is not IAnalyzer:
                    potential_analyzers.append({
                        "name": attr_name,
                        "module": module_path,
                        "class": attr,
                        "file": str(py_file.relative_to(project_root))
                    })
        except Exception as e:
            # logger.debug(f"Could not load module {module_path}: {e}")
            pass
            
    return potential_analyzers

def main():
    logger.info("Starting Analytics System Diagnostics...")
    
    config_manager = UnifiedConfigManager()
    engine = UnifiedAnalyticsEngine(config_manager)
    
    # 1. Check loaded analyzers
    loaded = engine.get_registered_components()['analyzers']
    logger.info(f"Analyzers loaded in engine: {len(loaded)}")
    for name in loaded:
        logger.info(f"  - [OK] {name}")
        
    # 2. Check for missing/failed analyzers in config
    logger.info("Checking configuration paths...")
    analysis_engine_root = config_manager.get('engine')
    analysis_engine_nested = config_manager.get('analysis.engine')
    
    if analysis_engine_root:
        logger.info(f"Found 'engine' at root with {len(analysis_engine_root.get('analyzers', []))} analyzers.")
    if analysis_engine_nested:
        logger.info(f"Found 'analysis.engine' with {len(analysis_engine_nested.get('analyzers', []))} analyzers.")
        
    # Check calculators
    calc_root = config_manager.get('calculators_config')
    calc_nested = config_manager.get('analysis.calculators_config')
    
    if calc_root:
        logger.info(f"Found 'calculators_config' at root with {len(calc_root)} calculators.")
    if calc_nested:
        logger.info(f"Found 'analysis.calculators_config' with {len(calc_nested)} calculators.")

    engine_config = config_manager.get('analysis.engine', {})
    configured_analyzers = engine_config.get('analyzers', [])
    configured_names = [c.get('name', c.get('class', '').lower()) for c in configured_analyzers]
    
    failed_to_load = [name for name in configured_names if name not in loaded]
    if failed_to_load:
        logger.error(f"Analyzers in config but NOT loaded: {failed_to_load}")
    else:
        logger.info("All configured analyzers loaded successfully.")
        
    # 3. Scan for "orphan" analyzers (implemented but not configured)
    logger.info("Scanning filesystem for unconfigured analyzers...")
    all_potential = scan_analytics_directory(project_root / "src" / "analytics")
    
    orphan_analyzers = []
    for pot in all_potential:
        # Check if this class/module combo is in engine.analyzers
        is_loaded = False
        for name, instance in engine.analyzers.items():
            if instance.__class__.__name__ == pot['name']:
                is_loaded = True
                break
        
        if not is_loaded:
            orphan_analyzers.append(pot)
            
    if orphan_analyzers:
        logger.warning(f"Found {len(orphan_analyzers)} analyzers NOT registered in UnifiedAnalyticsEngine:")
        for orphan in orphan_analyzers:
            logger.warning(f"  - {orphan['name']} ({orphan['module']})")
    else:
        logger.info("No orphan analyzers found in src/analytics.")
        
    # 4. Check Calculators
    logger.info("Checking Calculators configuration...")
    calculators_config = config_manager.get('analysis.calculators_config', {})
    logger.info(f"Calculators configured: {len(calculators_config)}")
    for name, cfg in calculators_config.items():
        try:
            module = importlib.import_module(cfg['module'])
            getattr(module, cfg['class'])
            logger.info(f"  - [OK] {name}")
        except Exception as e:
            logger.error(f"  - [FAIL] {name}: {e}")

    # 5. Minimal Execution Test
    logger.info("Running minimal engine execution test...")
    
    # Create dummy data
    dates = pd.date_range(start="2024-01-01", periods=100, freq='D')
    dummy_price_data = pd.DataFrame({
        'open': np.random.randn(100).cumsum() + 100,
        'high': np.random.randn(100).cumsum() + 105,
        'low': np.random.randn(100).cumsum() + 95,
        'close': np.random.randn(100).cumsum() + 100,
        'volume': np.random.randint(1000, 5000, 100)
    }, index=dates)
    
    data_map = {
        "price_data": dummy_price_data,
        "market_indicators": pd.DataFrame({'vol': np.random.randn(100)}, index=dates),
        "news_data": [],
        "target_series": pd.Series(np.random.randn(100), index=dates),
        "causal_series": pd.Series(np.random.randn(100), index=dates),
        "features_data": pd.DataFrame(np.random.randn(100, 5), index=dates),
        "macro_data": pd.DataFrame({'gdp': [1, 2, 3]}, index=pd.date_range("2024-01-01", periods=3, freq='ME')),
        "models_metadata": {},
        "portfolio_data": pd.DataFrame()
    }
    
    try:
        results = engine.run_full_analysis(data_map)
        logger.info("Engine execution completed.")
        for name, res in results.items():
            status = "SUCCESS" if "error" not in res else f"FAILED: {res.get('error')}"
            logger.info(f"  - {name}: {status}")
    except Exception as e:
        logger.error(f"Engine execution failed: {e}", exc_info=True)

if __name__ == "__main__":
    main()
