#!/usr/bin/env python3
"""
Runner for the Shadow Battle.
"""

import sys
from src.config.unified_config_manager import get_current_config
from src.main.modes.shadow_battle import ShadowBattleMode
from src.core.logging.logger import ProjectLogger

logger = ProjectLogger.get_logger("RunShadowBattle")

def main():
    logger.info("Initializing Shadow Battle runner...")
    config = get_current_config()
    mode = ShadowBattleMode(config)
    result = mode.run()
    logger.info(f"Shadow Battle finished with status: {result.get('status')}")

if __name__ == "__main__":
    main()
