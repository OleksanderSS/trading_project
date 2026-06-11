
# src/feature_engineering/nlp/roberta_sentiment.py

import logging
from typing import Any

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

logger = logging.getLogger(__name__)

class RobertaSentimentAnalyzer:
    """
    Performs sentiment analysis using a pre-trained RoBERTa-based model from Hugging Face.
    Supports lazy loading and batch processing.
    """

    def __init__(self, model_config: dict[str, Any]):
        """
        Initializes the analyzer with model configuration.

        Args:
            model_config (Dict[str, Any]): Configuration dictionary containing:
                - 'model_name': Hugging Face model identifier.
                - 'confidence_threshold': Minimum score for a label to be accepted.
                - 'device': The device to run the model on ('cuda', 'cpu', 'mps').
        """
        if not model_config or 'model_name' not in model_config:
            raise ValueError("Sentiment model configuration must contain a 'model_name'.")

        self.model_name = model_config['model_name']
        self.confidence_threshold = model_config.get('confidence_threshold', 0.6)

        # Determine device, defaulting to CUDA if available
        self.device = model_config.get('device')
        if not self.device:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.tokenizer: AutoTokenizer | None = None
        self.model: AutoModelForSequenceClassification | None = None
        logger.info(f"RobertaSentimentAnalyzer initialized for model '{self.model_name}' on device '{self.device}'.")

    def _load_model(self):
        """
        Lazily loads the tokenizer and model from Hugging Face into memory.
        This method is called automatically on the first analysis request.
        """
        if self.model is not None:
            return

        logger.info(f"Lazy loading sentiment model '{self.model_name}'...")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            self.model.to(self.device)
            self.model.eval()  # Set model to evaluation mode
            logger.info(f"Model '{self.model_name}' loaded successfully on '{self.device}'.")
        except Exception as e:
            logger.critical(f"Fatal error: Failed to load sentiment model '{self.model_name}'. Reason: {e}", exc_info=True)
            # Set to None to prevent further attempts and allow graceful failure
            self.tokenizer = None
            self.model = None
            # Re-raise to signal a critical failure in initialization
            raise

    def analyze(self, text: str) -> dict[str, Any]:
        """
        Analyzes the sentiment of a single piece of text.

        Args:
            text (str): The input text to analyze.

        Returns:
            Dict[str, Any]: A dictionary containing the sentiment label, confidence score,
                            and detailed scores for all classes.
        """
        # Default result for invalid input or model failure
        default_result = {'label': 'neutral', 'score': 0.0, 'details': {}}

        if not text or not isinstance(text, str) or not text.strip():
            return default_result

        # Ensure the model is loaded before proceeding
        if self.model is None or self.tokenizer is None:
            try:
                self._load_model()
            except Exception:
                # _load_model already logged the critical error
                return default_result

        # This check is necessary in case loading failed
        if self.model is None or self.tokenizer is None:
             return default_result

        try:
            # Tokenize the input text
            inputs = self.tokenizer(text, return_tensors='pt', truncation=True, max_length=512, padding=True)
            inputs = {key: val.to(self.device) for key, val in inputs.items()}

            # Perform inference
            with torch.no_grad():
                outputs = self.model(**inputs)
                # Apply softmax to convert logits to probabilities
                scores = torch.nn.functional.softmax(outputs.logits, dim=-1)[0]

            # Map scores to labels based on the model's configuration
            id2label = self.model.config.id2label
            score_map = {id2label[i]: scores[i].item() for i in range(scores.shape[0])}

            # Determine the label with the highest score
            max_score_label = max(score_map, key=lambda k: score_map[k])
            max_score_value = score_map[max_score_label]

            # Apply confidence threshold: if confidence is too low, default to neutral
            final_label = max_score_label if max_score_value >= self.confidence_threshold else 'neutral'

            return {
                'label': final_label,
                'score': max_score_value,
                'details': {k: round(v, 4) for k, v in score_map.items()}
            }
        except Exception as e:
            logger.error(f"Error during sentiment analysis for text snippet: '{text[:80]}...'. Error: {e}", exc_info=True)
            return default_result
