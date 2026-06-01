import logging

from src.analytics.arena.arena_battle import TradingModelArena as ArenaBattle
from src.analytics.arena.arena_orchestrator import ArenaOrchestrator
from src.analytics.arena.battle_groups import BattleGroupManager as BattleGroups
from src.analytics.arena.performance_tracker import ModelPerformanceTracker

# Setup basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ArenaTest")

def test_arena():
    logger.info("Initializing Arena components for testing...")

    # Initialize components
    arena = ArenaBattle()
    groups = BattleGroups()
    tracker = ModelPerformanceTracker()
    orchestrator = ArenaOrchestrator(arena, groups, tracker)

    # Run a test battle for a mock ticker
    logger.info("Executing recommended battles...")
    results = orchestrator.run_recommended_battles(ticker="AMD", target="return_1d")

    logger.info(f"Arena test complete. Ran {len(results)} battles.")
    leaderboard = orchestrator.get_leaderboard()
    logger.info(f"Leaderboard: {leaderboard}")

if __name__ == "__main__":
    test_arena()
