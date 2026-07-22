from src.agents.modular_pipeline.lenses._prompted_lens import PromptedLens


class CrossPlatformInnovationLens(PromptedLens):
    LENS_NAME = "CrossPlatformInnovationLens"
    SUPPORTED_TAGS = (
        "technology",
        "ai_infrastructure",
        "healthcare",
        "robotics",
        "ev_manufacturing",
    )
    PERSPECTIVE = "cross-sector innovation and technology transmission"
