from typing import Any

import pandas as pd

from src.core.logging.logger import ProjectLogger
from src.features.feature_orchestrator import FeatureOrchestrator

logger = ProjectLogger.get_logger("FeatureEngineeringOrchestratorManager")

class FeatureEngineeringOrchestratorManager:
    def __init__(self, config_manager: Any):
        self.orchestrator = FeatureOrchestrator.create_from_config(config_manager)
        logger.info("✅ Feature orchestrator initialized")

    def run_enrichment(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame | None:
        logger.info("🔄 Computing features via Orchestrator...")
        return self.orchestrator.run(df, **kwargs)

    def get_config_hash(self) -> str:
        return self.orchestrator.get_config_hash() if hasattr(self.orchestrator, "get_config_hash") else "default"
