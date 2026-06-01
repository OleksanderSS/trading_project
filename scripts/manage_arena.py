import os

import pandas as pd

from src.analytics.arena.arena_battle import get_trading_arena
from src.analytics.arena.performance_tracker import get_performance_tracker
from src.core.logging.logger import ProjectLogger

logger = ProjectLogger.get_logger("ArenaManagement")

def manage_arena():
    arena = get_trading_arena()
    tracker = get_performance_tracker()
    
    # 1. State Persistence Demo
    state_file = "arena_state.json"
    perf_file = "performance_data.json"
    if os.path.exists(state_file):
        logger.info(f"Loading previous arena state from {state_file}")
        arena.load_arena_state(state_file)
    if os.path.exists(perf_file):
        logger.info(f"Loading performance data from {perf_file}")
        tracker.load_performance_data(perf_file)
    
    # 2. Automated Battle Group Creation
    groups = arena.get_recommended_battle_groups()
    logger.info(f"Available battle groups: {groups}")
    
    for group in groups:
        count = arena.create_battles_from_group(group)
        logger.info(f"Created {count} battles for group: {group}")
        
    # 3. Analytics Reporting
    logger.info("--- Arena Leaderboard Report ---")
    categories = tracker.get_leaderboard_categories()
    for cat, entries in categories.items():
        if entries:
            top = entries[0]
            logger.info(f"Category: {cat} | Top Model: {top['model_name']} | Points: {top['points']}")

    # 4. Autonomous Recovery Service
    run_recovery_service(arena)

    # 5. Save state
    if arena.save_arena_state(state_file):
        logger.info(f"Arena state saved to {state_file}")
    if tracker.save_performance_data(perf_file):
        logger.info(f"Performance data saved to {perf_file}")

def run_recovery_service(arena):
    """Checks all models in COOLDOWN and attempts automated recovery."""
    logger.info("--- Running Autonomous Recovery Service ---")
    for name, info in arena.models.items():
        if info.get('status') == 'COOLDOWN':
            # Run recovery check (requires mock data or recent window)
            # In a real pipeline, 'test_data' would be the latest market window
            success = arena.check_cooldown_recovery(name, pd.DataFrame(), pd.Series())
            if success:
                logger.info(f"Recovery service successfully restored {name}")
            else:
                logger.info(f"Recovery service: {name} remains in COOLDOWN")

if __name__ == "__main__":
    manage_arena()
