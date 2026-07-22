from src.agents.modular_pipeline.lenses._prompted_lens import PromptedLens


class RealEstateConstructionLens(PromptedLens):
    LENS_NAME = "RealEstateConstructionLens"
    SUPPORTED_TAGS = ("real_estate", "construction")
    PERSPECTIVE = "housing, commercial real estate, construction, and financing"
