from src.agents.modular_pipeline.lenses._prompted_lens import PromptedLens


class MetalsMiningLens(PromptedLens):
    LENS_NAME = "MetalsMiningLens"
    SUPPORTED_TAGS = ("metals", "mining", "raw_materials")
    PERSPECTIVE = "metals demand, mining supply, inventories, and project capacity"
