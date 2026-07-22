from __future__ import annotations

import pytest

from dean_os.bayesian_scenario_update import (
    ScenarioLikelihood,
    ScenarioPrior,
    update_mutually_exclusive_scenarios,
)


def test_bayesian_update_keeps_probability_confidence_and_impact_separate() -> None:
    priors = [
        ScenarioPrior(
            scenario_id="priced_in",
            prior_probability=0.45,
            prior_kind="base_rate",
            prior_source="reviewed export-control base rate",
            estimate_confidence=0.7,
            impact=-0.1,
            market_reaction=-0.05,
            fundamental_change=-0.02,
        ),
        ScenarioPrior(
            scenario_id="medium_revenue_impact",
            prior_probability=0.35,
            prior_kind="base_rate",
            prior_source="reviewed export-control base rate",
            estimate_confidence=0.5,
            impact=-0.5,
            market_reaction=-0.3,
            fundamental_change=-0.4,
        ),
        ScenarioPrior(
            scenario_id="regime_shift",
            prior_probability=0.20,
            prior_kind="review_prior",
            prior_source="analyst scenario review",
            estimate_confidence=0.3,
            impact=-0.9,
            market_reaction=-0.6,
            fundamental_change=-0.8,
        ),
    ]
    likelihoods = [
        ScenarioLikelihood(
            scenario_id="priced_in",
            evidence_id="policy-1",
            likelihood=0.8,
            likelihood_source="reviewed policy scope",
        ),
        ScenarioLikelihood(
            scenario_id="medium_revenue_impact",
            evidence_id="policy-1",
            likelihood=0.5,
            likelihood_source="reviewed policy scope",
        ),
        ScenarioLikelihood(
            scenario_id="regime_shift",
            evidence_id="policy-1",
            likelihood=0.2,
            likelihood_source="reviewed policy scope",
        ),
    ]

    update = update_mutually_exclusive_scenarios(priors, likelihoods)
    by_id = {item.scenario_id: item for item in update.posteriors}

    assert update.probability_mass_valid is True
    assert update.posterior_mass == pytest.approx(1.0)
    assert by_id["priced_in"].posterior_probability > 0.45
    assert by_id["priced_in"].estimate_confidence == 0.7
    assert by_id["priced_in"].impact == -0.1
    assert by_id["priced_in"].market_reaction == -0.05
    assert by_id["priced_in"].fundamental_change == -0.02
    assert update.calibration_status == "uncalibrated"


def test_bayesian_update_rejects_bad_probability_mass() -> None:
    priors = [
        ScenarioPrior(
            scenario_id="a",
            prior_probability=0.6,
            prior_source="review",
        ),
        ScenarioPrior(
            scenario_id="b",
            prior_probability=0.3,
            prior_source="review",
        ),
    ]
    likelihoods = [
        ScenarioLikelihood(
            scenario_id="a", evidence_id="e", likelihood=0.5, likelihood_source="r"
        ),
        ScenarioLikelihood(
            scenario_id="b", evidence_id="e", likelihood=0.5, likelihood_source="r"
        ),
    ]

    with pytest.raises(ValueError, match="sum to 1.0"):
        update_mutually_exclusive_scenarios(priors, likelihoods)
