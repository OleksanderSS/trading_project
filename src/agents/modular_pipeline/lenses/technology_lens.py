from src.agents.modular_pipeline.lenses._prompted_lens import PromptedLens


class TechnologySectorLens(PromptedLens):
    LENS_NAME = "TechnologySectorLens"
    SUPPORTED_TAGS = (
        "technology",
        "semiconductors",
        "hardware",
        "software",
        "ai_infrastructure",
    )
    PERSPECTIVE = "technology, semiconductors, and AI infrastructure"
