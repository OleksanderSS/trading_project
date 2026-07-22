#!/usr/bin/env python3
"""
Runner for the Monster Test (Stress Test) Mode.
"""

import sys
import asyncio
from src.config.unified_config_manager import get_current_config
from src.main.modes.monster_test import MonsterTestMode
from src.core.logging.logger import ProjectLogger

logger = ProjectLogger.get_logger("RunMonsterTest")

def main():
    logger.info("Initializing Monster Test runner...")
    config = get_current_config()
    mode = MonsterTestMode(config)
    result = mode.run()
    logger.info(f"Monster Test finished with status: {result.get('status')}")

if __name__ == "__main__":
    main()
