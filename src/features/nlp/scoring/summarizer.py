
# src/feature_engineering/nlp/summarizer.py

import logging
import torch
from transformers import pipeline, Pipeline
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class Summarizer:
    """
    Generates text summaries using a pre-trained model from the Hugging Face Hub.
    Features lazy loading and robust error handling.
    """

    def __init__(self, model_config: Dict[str, Any]):
        """
        Initializes the Summarizer with a specific configuration.

        Args:
            model_config (Dict[str, Any]): Configuration dictionary containing:
                - 'model_name': Hugging Face model identifier (e.g., 't5-small').
                - 'max_length': The maximum length of the generated summary.
                - 'min_length': The minimum length of the generated summary.
                - 'input_max_len': The maximum length of the input text to process.
                - 'min_words_for_summary': The minimum number of words for text to be summarized.
                - 'device': Optional device override ('cuda', 'cpu').
        """
        if not model_config or 'model_name' not in model_config:
            raise ValueError("Summarizer configuration must contain a 'model_name'.")

        self.model_name = model_config.get('model_name', 't5-small')
        self.max_length = model_config.get('max_length', 120)
        self.min_length = model_config.get('min_length', 30)
        self.input_max_len = model_config.get('input_max_len', 1024) # Max tokens for model input
        self.min_words_for_summary = model_config.get('min_words_for_summary', 20)
        
        # Determine the device to run on
        device_override = model_config.get('device')
        if device_override:
            self.device = device_override
            device_index = -1 if device_override == 'cpu' else 0 # pipeline expects index
        else:
            is_cuda = torch.cuda.is_available()
            self.device = 'cuda' if is_cuda else 'cpu'
            device_index = 0 if is_cuda else -1
        self.pipeline_device_index = device_index

        self.summarization_pipeline: Optional[Pipeline] = None
        logger.info(f"Summarizer initialized for model '{self.model_name}' on device '{self.device}'.")

    def _load_model(self):
        """
        Lazily loads the summarization pipeline from Hugging Face.
        This is called on the first summarization request.
        """
        if self.summarization_pipeline is not None:
            return

        logger.info(f"Lazy loading summarization model '{self.model_name}'...")
        try:
            self.summarization_pipeline = pipeline(
                "summarization",
                model=self.model_name,
                device=self.pipeline_device_index,
                framework="pt"
            )
            logger.info(f"Summarization model '{self.model_name}' loaded successfully.")
        except Exception as e:
            logger.critical(
                f"Fatal error: Failed to load summarization model '{self.model_name}'. "
                f"Summarization will be unavailable. Reason: {e}", exc_info=True
            )
            self.summarization_pipeline = None # Ensure it remains None on failure

    def summarize(self, text: str) -> str:
        """
        Generates a summary for the given text, with fallbacks for short text or errors.

        Args:
            text (str): The input text to summarize.

        Returns:
            str: The generated summary, or a truncated portion of the original text as a fallback.
        """
        if not text or not isinstance(text, str) or not text.strip():
            return ""

        # Fallback for short texts that don't need summarization
        if len(text.split()) < self.min_words_for_summary:
            logger.debug("Text is too short for summarization, returning original text.")
            return text

        # Lazy load the model; if it fails, summarization_pipeline remains None
        self._load_model()

        # If model loading failed, provide a fallback summary (truncated text)
        if self.summarization_pipeline is None:
            logger.warning("Summarization model is unavailable. Returning a truncated portion of the text.")
            return text[:self.max_length]

        # Prepare the input text
        truncated_text = text[:self.input_max_len]
        # T5-based models often benefit from a task-specific prefix
        input_text_with_prefix = f"summarize: {truncated_text}"

        try:
            # Generate summary using the pipeline
            result = self.summarization_pipeline(
                input_text_with_prefix,
                max_length=self.max_length,
                min_length=self.min_length,
                truncation=True,
                do_sample=False
            )

            summary = result[0]['summary_text'].strip() if result and isinstance(result, list) else ""
            
            if not summary:
                logger.warning("Model generated an empty summary. Providing fallback.")
                return truncated_text.split('.')[0] + '.' # Return first sentence as a fallback

            return summary

        except Exception as e:
            logger.error(f"Error during summarization pipeline: {e}", exc_info=True)
            # Final fallback in case of runtime error
            return truncated_text[:self.max_length]

