import logging
from typing import Any

# ✅ Lazy imports — transformers/torch are heavy; loaded only when model is instantiated
_torch = None
_transformers = None

def _get_torch():
    global _torch
    if _torch is None:
        import torch
        _torch = torch
    return _torch

def _get_transformers():
    global _transformers
    if _transformers is None:
        import transformers
        _transformers = transformers
    return _transformers

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
            raise ValueError(
                "Sentiment model configuration must contain a 'model_name'.")
        self.model_name = model_config['model_name']
        self.confidence_threshold = model_config.get('confidence_threshold',
            0.6)
        self.device = model_config.get('device')
        if not self.device:
            self.device = 'cuda' if _get_torch().cuda.is_available() else 'cpu'
        self.tokenizer: Any = None  # AutoTokenizer — lazy loaded
        self.model: Any = None  # AutoModelForSequenceClassification — lazy loaded
        logger.info(
            f"RobertaSentimentAnalyzer initialized for model '{self.model_name}' on device '{self.device}'."
            )

    def _load_model(self):
        """
        Lazily loads the tokenizer and model from Hugging Face into memory.
        This method is called automatically on the first analysis request.
        """
        if self.model is not None:
            return
        logger.info(f"Lazy loading sentiment model '{self.model_name}'...")
        try:
            tf_mod = _get_transformers()
            self.tokenizer = tf_mod.AutoTokenizer.from_pretrained(self.model_name)
            self.model = tf_mod.AutoModelForSequenceClassification.from_pretrained(
                self.model_name)
            self.model.to(self.device)
            self.model.eval()
            logger.info(
                f"Model '{self.model_name}' loaded successfully on '{self.device}'."
                )
        except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError) as e:
            logger.critical(
                f"Fatal error: Failed to load sentiment model '{self.model_name}'. Reason: {e}"
                , exc_info=True)
            self.tokenizer = None
            self.model = None
            raise

    def analyze(self, text: str) ->dict[str, Any]:
        """
        Analyzes the sentiment of a single piece of text.

        Args:
            text (str): The input text to analyze.

        Returns:
            Dict[str, Any]: A dictionary containing the sentiment label, confidence score,
                            and detailed scores for all classes.
        """
        default_result = {'label': 'neutral', 'score': 0.0, 'details': {}}
        if not text or not isinstance(text, str) or not text.strip():
            return default_result
        if self.model is None or self.tokenizer is None:
            try:
                self._load_model()
            except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError) as e:
                logger.exception('Виникла помилка при завантаженні моделі')
                return default_result
        if self.model is None or self.tokenizer is None:
            return default_result
        try:
            inputs = self.tokenizer(text, return_tensors='pt', truncation=
                True, max_length=512, padding=True)
            inputs = {key: val.to(self.device) for key, val in inputs.items()}
            with _get_torch().no_grad():
                outputs = self.model(**inputs)
                scores = _get_torch().nn.functional.softmax(outputs.logits, dim=-1)[0]
            id2label = self.model.config.id2label
            score_map = {id2label[i]: scores[i].item() for i in range(
                scores.shape[0])}
            max_score_label = max(score_map, key=lambda k: score_map[k])
            max_score_value = score_map[max_score_label]
            final_label = (max_score_label if max_score_value >= self.
                confidence_threshold else 'neutral')
            return {'label': final_label, 'score': max_score_value,
                'details': {k: round(v, 4) for k, v in score_map.items()}}
        except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError) as e:
            logger.error(
                f"Error during sentiment analysis for text snippet: '{text[:80]}...'. Error: {e}"
                , exc_info=True)
            return default_result
