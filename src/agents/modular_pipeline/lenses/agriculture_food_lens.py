from src.agents.modular_pipeline.lenses._prompted_lens import PromptedLens


class AgricultureFoodLens(PromptedLens):
    LENS_NAME = "AgricultureFoodLens"
    SUPPORTED_TAGS = ("agriculture", "food_supply")
    PERSPECTIVE = "agriculture, food supply, crop yields, and input costs"
