from datetime import datetime, timedelta

from src.models.ensemble.weight_stability.visualizer import WeightStabilityVisualizer


def test_weight_stability_visualizer_writes_nonempty_plot(tmp_path):
    visualizer = WeightStabilityVisualizer(config={})
    start = datetime(2026, 1, 1)
    weight_history = [
        {"timestamp": start, "weights": {"m1": 0.6, "m2": 0.4}},
        {"timestamp": start + timedelta(days=1), "weights": {"m1": 0.55, "m2": 0.45}},
        {"timestamp": start + timedelta(days=2), "weights": {"m1": 0.5, "m2": 0.5}},
    ]
    weight_changes = [
        {"m1": -0.05, "m2": 0.05},
        {"m1": -0.05, "m2": 0.05},
    ]
    output = tmp_path / "stability.png"

    visualizer.plot_stability_metrics(weight_history, weight_changes, ["m1", "m2"], str(output))

    assert output.exists()
    assert output.stat().st_size > 0
