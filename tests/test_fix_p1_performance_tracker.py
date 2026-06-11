import pytest
from src.analytics.arena.performance_tracker import get_performance_tracker
from unittest.mock import MagicMock

def test_get_top_performers_raises_runtime_error_on_failure():
    tracker = get_performance_tracker()
    
    # Force a failure by manipulating internal state
    tracker.leaderboard = None 
    
    with pytest.raises(RuntimeError, match="Failed to get top performers"):
        tracker.get_top_performers(metric='points')
