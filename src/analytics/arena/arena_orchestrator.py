import time
from typing import Any

import pandas as pd

from src.analytics.arena.arena_battle import TradingModelArena as ArenaBattle
from src.analytics.arena.battle_groups import BattleGroupManager as BattleGroups
from src.analytics.arena.ensemble_performance_bridge import EnsemblePerformanceBridge
from src.analytics.arena.performance_tracker import ModelPerformanceTracker
from src.core.logging.logger import ProjectLogger

logger = ProjectLogger.get_logger("ArenaOrchestrator")


class ArenaOrchestrator:
    """
    Orchestrates battles between models in the Arena and tracks their performance.
    """

    def __init__(
        self,
        arena_battle: ArenaBattle,
        battle_groups: BattleGroups,
        tracker: ModelPerformanceTracker,
        ensemble_bridge: EnsemblePerformanceBridge = None,
    ):
        self.arena = arena_battle
        self.groups = battle_groups
        self.tracker = tracker
        self.bridge = ensemble_bridge

    def run_recommended_battles(self, ticker: str, target: str) -> dict[str, Any]:
        """Runs battles and records performance to the tracker."""
        logger.info(f"Starting arena battles for {ticker}...")

        start_time = time.time()
        # Use injected groups manager instead of global function
        groups = self.groups.list_groups()

        results = {}
        for group_name in groups:
            # Conduct battle
            battle_res = self.arena.conduct_battle(
                ticker=ticker,
                target=target,
                candidate_name=group_name,
                test_data=pd.DataFrame(),
                actual_targets=pd.Series(),
            )

            # Record performance using tracker
            self.tracker.record_battle_performance({"model_name": group_name, "ticker": ticker, "results": battle_res})

            execution_time = time.time() - start_time
            vote_count = battle_res.get("votes", 0)

            logger.info(f"Battle {group_name} finished in {execution_time:.2f}s with {vote_count} votes.")
            results[group_name] = battle_res

        return results

    def create_custom_battle_group(self, name: str, models: list[str], description: str = ""):
        """API wrapper for creating custom battle groups."""
        return self.groups.create_custom_group(name, models, description)

    def get_battle_group_details(self, group_name: str) -> dict[str, Any]:
        """API wrapper for getting battle group info."""
        return self.groups.get_group_info(group_name)

    def get_leaderboard(self):
        """Expose top performers."""
        return self.tracker.get_top_performers()

    def get_unified_arena_report(self) -> dict[str, Any]:
        """Expose bridge performance report."""
        if self.bridge:
            return self.bridge.get_unified_performance_view()
        return {"error": "Ensemble bridge not configured."}
