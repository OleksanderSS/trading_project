"""Tests for AdaptiveModelSelector."""

import pytest
import tempfile
import json
from pathlib import Path

from src.models.model_selector.adaptive_selector import AdaptiveModelSelector


@pytest.fixture
def temp_leaderboard():
    """Create temporary leaderboard file."""
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
        filepath = f.name
    yield filepath
    Path(filepath).unlink(missing_ok=True)


@pytest.fixture
def selector(temp_leaderboard):
    """Create selector instance."""
    return AdaptiveModelSelector(
        fallback="lightgbm",
        leaderboard_path=temp_leaderboard,
        learning_rate=0.1
    )


def test_selector_creation(temp_leaderboard):
    """Test selector creation."""
    selector = AdaptiveModelSelector(
        fallback="catboost",
        leaderboard_path=temp_leaderboard,
        learning_rate=0.2
    )
    assert selector.fallback == "catboost"
    assert selector.learning_rate == 0.2
    assert len(selector.arena_leaderboard) == 0


def test_select_best_model_adaptive_fallback(selector):
    """Test selection with empty leaderboard (fallback)."""
    model = selector.select_best_model_adaptive("1|0|-1|1|0")
    assert model == "lightgbm"  # Fallback


def test_update_from_feedback(selector):
    """Test updating leaderboard from feedback."""
    context = "1|0|-1|1|0"
    model_id = "catboost_v1"
    
    # Provide feedback
    selector.update_from_feedback(
        model_id, context,
        actual_return=0.05,
        predicted_return=0.04
    )
    
    # Check leaderboard updated
    assert context in selector.arena_leaderboard
    assert model_id in selector.arena_leaderboard[context]
    assert selector.arena_leaderboard[context][model_id]['total_predictions'] == 1


def test_online_learning(selector):
    """Test online learning updates win_rate."""
    context = "1|0|-1|1|0"
    model_id = "catboost_v1"
    
    # First feedback (good prediction)
    selector.update_from_feedback(
        model_id, context,
        actual_return=0.05,
        predicted_return=0.05
    )
    
    win_rate_1 = selector.arena_leaderboard[context][model_id]['win_rate']
    
    # Second feedback (bad prediction)
    selector.update_from_feedback(
        model_id, context,
        actual_return=0.05,
        predicted_return=-0.05
    )
    
    win_rate_2 = selector.arena_leaderboard[context][model_id]['win_rate']
    
    # Win rate should decrease
    assert win_rate_2 < win_rate_1


def test_leaderboard_persistence(temp_leaderboard):
    """Test leaderboard persists across instances."""
    # Create selector and add data
    selector1 = AdaptiveModelSelector(leaderboard_path=temp_leaderboard)
    selector1.update_from_feedback(
        "model1", "1|0|-1|1|0",
        actual_return=0.05,
        predicted_return=0.04
    )
    
    # Create new selector instance
    selector2 = AdaptiveModelSelector(leaderboard_path=temp_leaderboard)
    
    # Check data persisted
    assert "1|0|-1|1|0" in selector2.arena_leaderboard
    assert "model1" in selector2.arena_leaderboard["1|0|-1|1|0"]


def test_recent_performance_tracking(selector):
    """Test recent performance tracking."""
    model_id = "catboost_v1"
    context = "1|0|-1|1|0"
    
    # Add multiple feedbacks
    for i in range(5):
        selector.update_from_feedback(
            model_id, context,
            actual_return=0.05,
            predicted_return=0.05 + i * 0.01
        )
    
    # Check performance tracked
    assert model_id in selector.performance_tracker
    assert len(selector.performance_tracker[model_id]) == 5


def test_alternative_model_selection(selector):
    """Test alternative model selection."""
    context = "1|0|-1|1|0"
    
    # Add two models with different performance
    selector.update_from_feedback(
        "model1", context,
        actual_return=0.05,
        predicted_return=0.05
    )
    selector.arena_leaderboard[context]["model1"]["win_rate"] = 0.9
    
    selector.update_from_feedback(
        "model2", context,
        actual_return=0.05,
        predicted_return=0.04
    )
    selector.arena_leaderboard[context]["model2"]["win_rate"] = 0.7
    
    # Get alternative (should be model2, second best)
    alternative = selector._get_alternative_model(context)
    assert alternative == "model2"


def test_selection_history(selector):
    """Test selection history tracking."""
    # Make selections
    for i in range(3):
        selector.select_best_model_adaptive(f"1|0|-1|{i}|0")
    
    # Check history
    assert len(selector.selection_history) == 3
    assert 'timestamp' in selector.selection_history[0]
    assert 'context' in selector.selection_history[0]
    assert 'selected_model' in selector.selection_history[0]


def test_get_leaderboard_summary(selector):
    """Test leaderboard summary."""
    # Add some data
    selector.update_from_feedback(
        "model1", "1|0|-1|1|0",
        actual_return=0.05,
        predicted_return=0.04
    )
    selector.update_from_feedback(
        "model2", "1|0|-1|1|0",
        actual_return=0.05,
        predicted_return=0.04
    )
    
    summary = selector.get_leaderboard_summary()
    
    assert summary['total_contexts'] == 1
    assert summary['total_models'] == 2
    assert 'model1' in summary['models']
    assert 'model2' in summary['models']


def test_export_history(selector, temp_leaderboard):
    """Test exporting history."""
    # Add data
    selector.update_from_feedback(
        "model1", "1|0|-1|1|0",
        actual_return=0.05,
        predicted_return=0.04
    )
    selector.select_best_model_adaptive("1|0|-1|1|0")
    
    # Export
    export_path = str(Path(temp_leaderboard).parent / "history.json")
    selector.export_history(export_path)
    
    # Check file created
    assert Path(export_path).exists()
    
    # Check content
    with open(export_path) as f:
        data = json.load(f)
    assert 'selection_history' in data
    assert 'performance_tracker' in data
    
    # Cleanup
    Path(export_path).unlink()


def test_get_model_performance(selector):
    """Test getting model performance stats."""
    model_id = "catboost_v1"
    context = "1|0|-1|1|0"
    
    # Add feedbacks
    for i in range(10):
        selector.update_from_feedback(
            model_id, context,
            actual_return=0.05,
            predicted_return=0.05 + i * 0.001
        )
    
    perf = selector.get_model_performance(model_id)
    
    assert perf['model_id'] == model_id
    assert perf['total_predictions'] == 10
    assert 'avg_accuracy' in perf
    assert 'recent_accuracy' in perf
    assert 'std_accuracy' in perf


def test_get_model_performance_no_data(selector):
    """Test getting performance for model with no data."""
    perf = selector.get_model_performance("nonexistent")
    assert perf['status'] == 'no_data'


def test_learning_rate_effect(temp_leaderboard):
    """Test learning rate affects adaptation speed."""
    # High learning rate (more reactive)
    selector_high = AdaptiveModelSelector(
        leaderboard_path=temp_leaderboard,
        learning_rate=0.9
    )
    selector_high.update_from_feedback(
        "model1", "1|0|-1|1|0",
        actual_return=0.05,
        predicted_return=0.05
    )
    selector_high.update_from_feedback(
        "model1", "1|0|-1|1|0",
        actual_return=0.05,
        predicted_return=-0.05
    )
    
    # Low learning rate (more stable)
    selector_low = AdaptiveModelSelector(
        leaderboard_path=str(Path(temp_leaderboard).parent / "lb2.json"),
        learning_rate=0.1
    )
    selector_low.update_from_feedback(
        "model1", "1|0|-1|1|0",
        actual_return=0.05,
        predicted_return=0.05
    )
    selector_low.update_from_feedback(
        "model1", "1|0|-1|1|0",
        actual_return=0.05,
        predicted_return=-0.05
    )
    
    # High LR should change more
    wr_high = selector_high.arena_leaderboard["1|0|-1|1|0"]["model1"]["win_rate"]
    wr_low = selector_low.arena_leaderboard["1|0|-1|1|0"]["model1"]["win_rate"]
    
    assert wr_high < wr_low
    
    # Cleanup
    Path(str(Path(temp_leaderboard).parent / "lb2.json")).unlink(missing_ok=True)


def test_poor_performance_triggers_alternative(selector):
    """Test poor recent performance triggers alternative selection."""
    context = "1|0|-1|1|0"
    
    # Setup: model1 with poor recent performance
    selector.performance_tracker["model1"] = [0.2, 0.1, 0.15, 0.2, 0.1]  # Poor
    selector.arena_leaderboard[context] = {
        "model1": {"points": 10, "win_rate": 0.8, "total_predictions": 10},
        "model2": {"points": 8, "win_rate": 0.7, "total_predictions": 10}
    }
    
    # Select (should switch to model2 due to poor recent performance)
    selected = selector.select_best_model_adaptive(context)
    
    # Should select alternative due to poor recent performance
    assert selected == "model2"
