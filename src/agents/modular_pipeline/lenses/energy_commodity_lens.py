from src.agents.modular_pipeline.lenses._prompted_lens import PromptedLens


class EnergyCommodityLens(PromptedLens):
    LENS_NAME = "EnergyCommodityLens"
    SUPPORTED_TAGS = ("energy", "oil_gas", "renewables", "commodities")
    PERSPECTIVE = "energy supply, demand, policy, and commodity transmission"
