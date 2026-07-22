from src.pipeline.target_column_utils import (
    is_direct_target_column,
    is_target_like_column,
    split_model_features_and_targets,
)


def test_target_column_utils_split_direct_targets_from_target_derived_features():
    columns = [
        "datetime",
        "feature_a",
        "target_up_1d",
        "TARGET_RETURN_1P",
        "state_TARGET_RETURN_1P",
        "state_state_TARGET_DIRECTION_5P",
    ]

    features, targets, dropped = split_model_features_and_targets(columns)

    assert features == ["datetime", "feature_a"]
    assert targets == ["target_up_1d", "TARGET_RETURN_1P"]
    assert dropped == ["state_TARGET_RETURN_1P", "state_state_TARGET_DIRECTION_5P"]


def test_target_column_utils_classifies_case_insensitive_targets():
    assert is_direct_target_column("TARGET_DIRECTION_5P")
    assert is_target_like_column("state_TARGET_DIRECTION_5P")
    assert not is_target_like_column("targeted_marketing_spend")
