from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator


class ScenarioPrior(BaseModel):
    scenario_id: str
    prior_probability: float = Field(gt=0.0, lt=1.0)
    prior_kind: Literal[
        "base_rate",
        "review_prior",
        "market_implied",
        "calibrated_model",
    ] = "review_prior"
    prior_source: str
    estimate_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    impact: float | None = Field(default=None, ge=-1.0, le=1.0)
    market_reaction: float | None = Field(default=None, ge=-1.0, le=1.0)
    fundamental_change: float | None = Field(default=None, ge=-1.0, le=1.0)


class ScenarioLikelihood(BaseModel):
    scenario_id: str
    evidence_id: str
    likelihood: float = Field(gt=0.0, le=1.0)
    likelihood_source: str
    conditional_independence_assumed: bool = False


class ScenarioPosterior(BaseModel):
    scenario_id: str
    prior_probability: float
    likelihood: float
    bayes_factor_vs_alternatives: float
    posterior_probability: float
    estimate_confidence: float
    impact: float | None = None
    market_reaction: float | None = None
    fundamental_change: float | None = None


class BayesianScenarioUpdate(BaseModel):
    update_contract: str = "dean_bayesian_scenario_update_v1"
    evidence_id: str
    posteriors: list[ScenarioPosterior]
    posterior_mass: float
    probability_mass_valid: bool
    calibration_status: Literal["uncalibrated", "replay_calibrated"] = "uncalibrated"
    limitations: list[str] = Field(default_factory=list)


def update_mutually_exclusive_scenarios(
    priors: list[ScenarioPrior],
    likelihoods: list[ScenarioLikelihood],
    *,
    calibration_status: Literal["uncalibrated", "replay_calibrated"] = "uncalibrated",
) -> BayesianScenarioUpdate:
    """Bayesian update for one evidence item across exclusive scenarios.

    Confidence and impact fields are carried through unchanged. Bayes updates
    scenario probability only; they do not make the estimate more trustworthy
    and do not imply a market or fundamental effect.
    """

    if len(priors) < 2:
        raise ValueError("at least two mutually exclusive scenarios are required")
    prior_ids = [item.scenario_id for item in priors]
    if len(prior_ids) != len(set(prior_ids)):
        raise ValueError("scenario priors must have unique scenario_id values")
    prior_mass = sum(item.prior_probability for item in priors)
    if abs(prior_mass - 1.0) > 1e-6:
        raise ValueError("scenario prior probabilities must sum to 1.0")

    likelihood_by_id = {item.scenario_id: item for item in likelihoods}
    if set(likelihood_by_id) != set(prior_ids):
        raise ValueError("one likelihood is required for every scenario")
    evidence_ids = {item.evidence_id for item in likelihoods}
    if len(evidence_ids) != 1:
        raise ValueError("all likelihoods must condition on the same evidence item")

    evidence_probability = sum(
        prior.prior_probability * likelihood_by_id[prior.scenario_id].likelihood
        for prior in priors
    )
    if evidence_probability <= 0:
        raise ValueError("evidence probability must be positive")

    posteriors: list[ScenarioPosterior] = []
    for prior in priors:
        likelihood = likelihood_by_id[prior.scenario_id].likelihood
        posterior = prior.prior_probability * likelihood / evidence_probability
        alternative_prior_mass = 1.0 - prior.prior_probability
        alternative_likelihood = sum(
            other.prior_probability * likelihood_by_id[other.scenario_id].likelihood
            for other in priors
            if other.scenario_id != prior.scenario_id
        ) / alternative_prior_mass
        bayes_factor = likelihood / alternative_likelihood
        posteriors.append(
            ScenarioPosterior(
                scenario_id=prior.scenario_id,
                prior_probability=prior.prior_probability,
                likelihood=likelihood,
                bayes_factor_vs_alternatives=bayes_factor,
                posterior_probability=posterior,
                estimate_confidence=prior.estimate_confidence,
                impact=prior.impact,
                market_reaction=prior.market_reaction,
                fundamental_change=prior.fundamental_change,
            )
        )

    posterior_mass = sum(item.posterior_probability for item in posteriors)
    limitations = [
        "Posterior quality depends on the prior and likelihood estimates",
        "Bayesian updating does not establish causal identification",
        "Confidence and impact are not probabilities and were not updated",
    ]
    if any(item.conditional_independence_assumed for item in likelihoods):
        limitations.append(
            "At least one likelihood assumes conditional independence"
        )
    return BayesianScenarioUpdate(
        evidence_id=next(iter(evidence_ids)),
        posteriors=posteriors,
        posterior_mass=posterior_mass,
        probability_mass_valid=abs(posterior_mass - 1.0) <= 1e-6,
        calibration_status=calibration_status,
        limitations=limitations,
    )


__all__ = [
    "BayesianScenarioUpdate",
    "ScenarioLikelihood",
    "ScenarioPosterior",
    "ScenarioPrior",
    "update_mutually_exclusive_scenarios",
]
