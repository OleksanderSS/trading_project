from src.agents.modular_pipeline.lenses._prompted_lens import PromptedLens


class ConsumerRetailLens(PromptedLens):
    LENS_NAME = "ConsumerRetailLens"
    SUPPORTED_TAGS = ("retail", "consumer_goods", "e_commerce")
    PERSPECTIVE = "consumer demand, retail inventories, margins, and credit"
