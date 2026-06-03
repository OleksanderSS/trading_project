import importlib
import logging
import pkgutil
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.config.unified_config_manager import UnifiedConfigManager
from src.features.enrichers.base import BaseEnricher
from src.features.feature_orchestrator import FeatureOrchestrator

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger("FeatureDiagnostics")

def scan_enrichers_directory(base_path: Path) -> List[Dict[str, Any]]:
    """Scans the enrichers directory for potential enricher classes."""
    potential_enrichers = []
    package_path = str(base_path)
    package_name = 'src.features.enrichers'
    
    for _, module_name, _ in pkgutil.iter_modules([package_path]):
        full_module_name = f"{package_name}.{module_name}"
        try:
            module = importlib.import_module(full_module_name)
            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                if isinstance(attr, type) and issubclass(attr, BaseEnricher) and attr is not BaseEnricher:
                    # Instantiate to get name and priority
                    try:
                        # Try with empty config
                        instance = attr({})
                    except:
                        try:
                            instance = attr()
                        except:
                            instance = None
                            
                    potential_enrichers.append({
                        "class_name": attr_name,
                        "module": full_module_name,
                        "name": instance.name if instance else "Unknown",
                        "priority": instance.priority if instance else 99,
                        "file": str(Path(module.__file__).relative_to(project_root))
                    })
        except Exception as e:
            pass
            
    return potential_enrichers

def main():
    logger.info("Starting Feature System Diagnostics...")
    
    config_manager = UnifiedConfigManager()
    orchestrator = FeatureOrchestrator.create_from_config(config_manager)
    
    # 1. Check loaded enrichers
    loaded_names = [e.name for e in orchestrator.enrichers]
    logger.info(f"Enrichers loaded in orchestrator: {len(loaded_names)}")
    for e in orchestrator.enrichers:
        logger.info(f"  - [OK] {e.name} (Priority: {e.priority}, Class: {e.__class__.__name__})")
        
    # 2. Check configuration
    feature_config = config_manager.get_config('features', {})
    enabled_in_config = feature_config.get('enabled_enrichers', {})
    
    logger.info("Checking configuration consistency...")
    for name, enabled in enabled_in_config.items():
        if enabled and name not in loaded_names:
            logger.error(f"  - [FAIL] {name} is ENABLED in config but NOT loaded by orchestrator!")
        elif not enabled and name in loaded_names:
            logger.warning(f"  - [WARN] {name} is DISABLED in config but LOADED by orchestrator? (Check logic)")
            
    # 3. Scan for "orphan" enrichers
    logger.info("Scanning filesystem for unconfigured/disabled enrichers...")
    all_potential = scan_enrichers_directory(project_root / "src" / "features" / "enrichers")
    
    orphans = []
    for pot in all_potential:
        if pot['name'] not in loaded_names:
            orphans.append(pot)
            
    if orphans:
        logger.warning(f"Found {len(orphans)} enrichers NOT currently active:")
        for orphan in orphans:
            logger.warning(f"  - {orphan['name']} (Class: {orphan['class_name']}, File: {orphan['file']})")
    else:
        logger.info("No orphan enrichers found.")
        
    # 4. Minimal Execution Test
    logger.info("Running minimal orchestration test...")
    
    dates = pd.date_range(start="2024-01-01", periods=100, freq='D')
    df = pd.DataFrame({
        'open': np.random.randn(100).cumsum() + 100,
        'high': np.random.randn(100).cumsum() + 105,
        'low': np.random.randn(100).cumsum() + 95,
        'close': np.random.randn(100).cumsum() + 100,
        'volume': np.random.randint(1000, 5000, 100),
        'ticker': 'TEST',
        'datetime': dates
    })
    
    # Mock data for enrichers that need it
    cleaned_data = {
        'news': pd.DataFrame(columns=['datetime', 'ticker', 'title', 'sentiment']),
        'macro_data': pd.DataFrame(columns=['date', 'series_id', 'value']),
        'market_sentiment': pd.DataFrame(columns=['date', 'sentiment_score'])
    }
    
    try:
        enriched_df = orchestrator.run(df, **cleaned_data)
        logger.info(f"Orchestration completed. Initial columns: {len(df.columns)}, Final columns: {len(enriched_df.columns)}")
        logger.info(f"Added columns: {set(enriched_df.columns) - set(df.columns)}")
    except Exception as e:
        logger.error(f"Orchestration failed: {e}", exc_info=True)

if __name__ == "__main__":
    main()
