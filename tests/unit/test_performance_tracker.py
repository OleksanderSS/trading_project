import pytest
from src.analytics.arena.performance_tracker import ModelPerformanceTracker

def test_leaderboard_update():
    tracker = ModelPerformanceTracker()
    battle_data = {
        "battle_id": 1,
        "model1": "lgbm",
        "model2": "rf",
        "winner": "lgbm",
        "model1_metrics": {"accuracy": 0.8, "sharpe_ratio": 1.5, "win_rate": 0.9, "confidence_score": 0.9, "execution_time": 0.1},
        "model2_metrics": {"accuracy": 0.7, "sharpe_ratio": 1.2, "win_rate": 0.5, "confidence_score": 0.8, "execution_time": 0.2}
    }
    
    tracker.record_battle_performance(battle_data)
    
    assert len(tracker.leaderboard) > 0
    assert tracker.leaderboard[0].model_name == "lgbm"
    assert tracker.leaderboard[0].wins == 1
    assert tracker.leaderboard[0].points == 3
