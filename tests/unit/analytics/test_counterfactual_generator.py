import numpy as np
import pandas as pd

from src.analytics.context.counterfactual_generator import CounterfactualGenerator


def test_propensity_ate_uses_matched_outcomes_not_random_values():
    generator = CounterfactualGenerator()
    matched = {"pairs": [(0, 2), (1, 3)]}
    data = pd.DataFrame({
        "outcome": [1.0, 2.0, 0.4, 1.4],
        "treatment": [1, 1, 0, 0],
    })

    result = generator._calculate_ate(matched, "outcome", data)

    assert np.isclose(result["ate"], 0.6)
    assert np.allclose(result["treatment_effects"], [0.6, 0.6])


def test_linear_causal_estimation_recovers_data_driven_effect():
    generator = CounterfactualGenerator()
    data = pd.DataFrame({
        "treatment": [0, 0, 0, 1, 1, 1, 0, 1],
        "x": [0.0, 1.0, 2.0, 0.0, 1.0, 2.0, 3.0, 3.0],
    })
    data["outcome"] = 1.5 * data["treatment"] + 0.2 * data["x"] + [0.01, -0.02, 0.01, 0.03, -0.01, 0.02, 0.0, -0.02]

    result = generator.estimate_causal_effects(
        data=data,
        treatment_col="treatment",
        outcome_col="outcome",
        covariates=["x"],
        method="linear",
    )

    effect = result["causal_effects"]["treatment_effect"]
    low, high = result["confidence_intervals"]
    assert effect > 1.4
    assert low <= effect <= high
    assert 0.0 <= result["causal_effects"]["p_value"] <= 1.0


def test_difference_in_differences_reports_real_significance_fields():
    generator = CounterfactualGenerator()
    index = pd.date_range("2026-01-01", periods=6, freq="D")
    treatment = pd.DataFrame({"outcome": [1.0, 1.1, 1.2, 2.0, 2.2, 2.4]}, index=index)
    control = pd.DataFrame({"outcome": [1.0, 1.05, 1.1, 1.2, 1.25, 1.3]}, index=index)

    result = generator.run_difference_in_differences(
        treatment_group=treatment,
        control_group=control,
        pre_period=(index[0], index[2]),
        post_period=(index[3], index[5]),
        outcome_col="outcome",
    )

    assert result["treatment_effect"] > 0.7
    assert result["significance"]["confidence_interval"][0] < result["treatment_effect"]
    assert result["treatment_effect"] < result["significance"]["confidence_interval"][1]
    assert 0.0 <= result["significance"]["p_value"] <= 1.0
    assert 0.0 <= result["parallel_trends_test"]["p_value"] <= 1.0


def test_match_balance_uses_sample_size_weighted_pooled_variance():
    generator = CounterfactualGenerator()
    data = pd.DataFrame(
        {
            "x": [10.0, 12.0, 9.0, 1.0, 2.0, 3.0],
            "treatment": [1, 1, 1, 0, 0, 0],
            "outcome": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        }
    )
    matched = {"pairs": [(0, 3), (1, 4), (2, 5)]}

    result = generator._check_covariate_balance(matched, ["x"], data)

    treatment_values = data.loc[[0, 1, 2], "x"]
    control_values = data.loc[[3, 4, 5], "x"]
    pooled_var = (
        ((len(treatment_values) - 1) * treatment_values.var(ddof=1))
        + ((len(control_values) - 1) * control_values.var(ddof=1))
    ) / (len(treatment_values) + len(control_values) - 2)
    expected_smd = (treatment_values.mean() - control_values.mean()) / np.sqrt(pooled_var)

    assert np.isclose(result["covariate_balance"]["x"]["standardized_mean_diff"], expected_smd)
