# enrichment/entity_extractor.py

# import spacy
import re
from typing import List, Optional, Set
import logging
from enrichment.enrichment_config import EnrichmentConfig

logger = logging.getLogger(__name__)

class EntityExtractor:
    def __init__(self, model: str = None):
        # if model is None:
        #     model = EnrichmentConfig.ENTITY_EXTRACTION["spacy_model"]
        
        # self.model_name = model
        # self.disable_components = EnrichmentConfig.ENTITY_EXTRACTION["disable_components"]
        self.ticker_map = EnrichmentConfig.ENTITY_EXTRACTION["ticker_patterns"]
        
        # try:
        #     # Завантажуємо модель, але вимикаємо непотрібні компоненти для швидкості
        #     self.nlp = spacy.load(model, disable=self.disable_components)
        #     logger.info(f"Loaded spaCy model: {model} (optimized mode)")
        #     self._model_loaded = True
        # except OSError:
        #     logger.error(f"spaCy model {model} not found.")
        #     self.nlp = None
        #     self._model_loaded = False
        logger.warning("Spacy is temporarily disabled. Entity extraction will be limited.")
        self.nlp = None

    def quick_match(self, text: str) -> Set[str]:
        """Миттєвий пошук тикерів через Regex (дуже швидко)"""
        found = set()
        text_lower = text.lower()
        for ticker, pattern in self.ticker_map.items():
            if re.search(pattern, text_lower):
                found.add(ticker)
        return found

    def extract_entities(self, text: str, entity_types: Optional[List[str]] = None) -> List[str]:
        if not text or not text.strip():
            return []

        # 1. Спершу робимо швидкий пошук. 
        # Якщо це проста новина про Теслу, нам може і не знадобитися spaCy.
        quick_entities = self.quick_match(text)
        
        # 2. Якщо потрібен глибший аналіз (наприклад, для пошуку організацій ORG)
        # if self.nlp:
        #     try:
        #         doc = self.nlp(text)
        #         spacy_ents = [ent.text.strip() for ent in doc.ents 
        #                      if not entity_types or ent.label_ in entity_types]
                
        #         # Об'єднуємо результати
        #         all_entities = list(quick_entities.union(set(spacy_ents)))
        #         return all_entities
        #     except Exception as e:
        #         logger.error(f"spaCy error: {e}")
        #         return list(quick_entities)
        
        return list(quick_entities)