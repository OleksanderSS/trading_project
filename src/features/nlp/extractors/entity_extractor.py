
# src/feature_engineering/nlp/entity_extractor.py

from typing import List, Dict, Any, Optional, Iterable
import logging
from functools import lru_cache

logger = logging.getLogger(__name__)

class EntityExtractor:
    """
    Extracts named entities from text using a pre-trained spaCy model.
    """

    def __init__(self, entity_config: Dict[str, Any]):
        """
        Initializes the EntityExtractor with a specific configuration.

        Args:
            entity_config (Dict[str, Any]): Configuration dictionary containing:
                - 'spacy_model': The name of the spaCy model to use (e.g., 'en_core_web_sm').
                - 'disable_components': A list of spaCy pipeline components to disable.
        """
        if not entity_config:
            raise ValueError("Entity configuration cannot be empty.")

        self.model_name = entity_config.get("spacy_model", "en_core_web_sm")
        self.disable_components = entity_config.get("disable_components", ["tagger", "parser", "attribute_ruler", "lemmatizer"])
        
        self.nlp = self._load_model()

    def _load_model(self) -> Optional[Any]:
        """
        Loads the configured spaCy model, handling potential errors.
        """
        try:
            import spacy

            # Efficiently load the model with disabled components
            nlp = spacy.load(self.model_name, disable=self.disable_components)
            logger.info(f"Successfully loaded spaCy model: '{self.model_name}'")
            return nlp
        except OSError:
            logger.error(
                f"Could not find spaCy model '{self.model_name}'. "
                f"To fix, run: python -m spacy download {self.model_name}"
            )
            # Return None to indicate failure, allowing graceful degradation
            return None

    @lru_cache(maxsize=1024)
    def extract(self, text: str, entity_types: Optional[List[str]] = None) -> List[str]:
        """
        Extracts named entities from a given text.

        Args:
            text (str): The input text to analyze.
            entity_types (Optional[List[str]]): A list of specific entity labels to filter for 
                                               (e.g., ['ORG', 'GPE']). If None, all entities are returned.

        Returns:
            List[str]: A list of unique, stripped entity texts.
        """
        if not self.nlp or not text or not isinstance(text, str) or not text.strip():
            return []

        try:
            doc = self.nlp(text)
            # Use a set for automatic deduplication, then convert to sorted list
            entities = {ent.text.strip() for ent in doc.ents if not entity_types or ent.label_ in entity_types}
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"Extracted {len(entities)} entities from text.")
            return sorted(entities)
        except Exception as e:
            logger.error(f"An unexpected error occurred during entity extraction: {e}", exc_info=True)
            raise RuntimeError("Entity extraction failed") from e

    def extract_batch(self, texts: Iterable[str], entity_types: Optional[List[str]] = None) -> List[List[str]]:
        """
        Extracts named entities from a batch of texts using spaCy's pipe.
        """
        if not self.nlp:
            return [[] for _ in texts]
            
        results = []
        try:
            # Use spaCy's pipe for efficient batch processing
            for doc in self.nlp.pipe(texts):
                entities = {ent.text.strip() for ent in doc.ents if not entity_types or ent.label_ in entity_types}
                results.append(sorted(entities))
        except Exception as e:
            logger.error(f"An error occurred during batch entity extraction: {e}", exc_info=True)
            return [[] for _ in texts]
        return results
