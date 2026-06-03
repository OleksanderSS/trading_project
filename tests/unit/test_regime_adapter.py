from src.features.analysis.regime_adapter import RegimeAdapter


def test_volatile_recommendation_changes_method_weights():
    adapter = RegimeAdapter()

    base = adapter.update_method_weights("normal", [])
    adjusted = adapter.update_method_weights("normal", ["volatile regime detected"])

    assert adjusted["lgbm"] > base["lgbm"]
    assert adjusted["correlation"] < base["correlation"]
    assert abs(sum(adjusted.values()) - 1.0) < 1e-9
