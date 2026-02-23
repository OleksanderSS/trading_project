# models/arena/__init__.py - Arena Mode Package

from .arena_battle import TradingModelArena, get_trading_arena, BattleResult, BattleMetrics, Battle
from .battle_groups import BattleGroup, BattleGroupManager, get_battle_group_manager, get_all_battle_groups, get_popular_groups
from .performance_tracker import ModelPerformanceTracker, ModelPerformanceRecord, LeaderboardEntry, get_performance_tracker

__all__ = [
    # Core Arena Classes
    'TradingModelArena',
    'get_trading_arena',
    'BattleResult',
    'BattleMetrics', 
    'Battle',
    
    # Battle Groups
    'BattleGroup',
    'BattleGroupManager',
    'get_battle_group_manager',
    'get_all_battle_groups',
    'get_popular_groups',
    
    # Performance Tracking
    'ModelPerformanceTracker',
    'ModelPerformanceRecord',
    'LeaderboardEntry',
    'get_performance_tracker'
]

# Version
__version__ = '1.0.0'

# Package Info
__author__ = 'Trading Project Team'
__description__ = 'Arena Mode for Trading Model Comparison'
__email__ = 'trading@project.com'

# Configuration
ARENA_CONFIG = {
    'default_battle_groups': ['traditional_vs_enhanced', 'light_vs_heavy'],
    'max_battles_per_session': 50,
    'leaderboard_update_frequency': 'auto',
    'performance_history_limit': 1000,
    'auto_save_interval': 300  # seconds
}
