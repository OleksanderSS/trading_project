import sys
from pathlib import Path
from src.core.logging.logger import ProjectLogger
from src.config.unified_config_manager import UnifiedConfigManager
from src.main.modes.historical_replay import HistoricalEventReplayMode

import logging

def main():
    logger = logging.getLogger("RunHistoricalReplay")
    ProjectLogger() # Initialize project logger once
    logger.info("Initializing Historical Event Replay runner...")

    try:
        config_manager = UnifiedConfigManager()
        global_config = config_manager.merged_config

        mode_config = {
            'event_type': 'sharp_drop',
            'threshold': -0.05,
            'context_bars_before': 20,
            'context_bars_after': 10
        }

        # Accept override from command line if any
        if len(sys.argv) > 1:
            mode_config['event_type'] = sys.argv[1]

        mode = HistoricalEventReplayMode(mode_config=mode_config, config_manager=config_manager)
        mode.run()
        
        logger.info("Historical Replay finished successfully.")
    except Exception as e:
        logger.error(f"Error during historical replay: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
