import logging
import re
from typing import List, Dict

from src.core.logging.logger import ProjectLogger

logger = ProjectLogger.get_logger("EntityLinker")

class EntityLinker:
    """
    Мапить згадки компаній та активів у тексті новин на конкретні Exposure Tags.
    Вирішує проблему перетину сфер (напр., Tesla = EV + AI + Energy).
    """
    
    def __init__(self):
        # В реальності це може вантажитися з yaml, але для швидкодії тримаємо як dict
        self.entity_graph = {
            "NVDA": {
                "keywords": ["nvidia", "jensen huang", "rtx", "h100"],
                "tags": ["semiconductors", "ai_infrastructure", "hardware", "data_centers", "technology"]
            },
            "TSLA": {
                "keywords": ["tesla", "elon musk", "model y", "cybertruck", "autopilot"],
                "tags": ["ev_manufacturing", "ai_software", "energy_storage", "robotics", "consumer_discretionary"]
            },
            "AAPL": {
                "keywords": ["apple", "iphone", "tim cook", "ios"],
                "tags": ["consumer_electronics", "software", "technology"]
            },
            "MSFT": {
                "keywords": ["microsoft", "azure", "windows", "satya nadella"],
                "tags": ["software", "cloud", "ai_infrastructure", "technology"]
            },
            "XOM": {
                "keywords": ["exxon", "exxonmobil", "oil"],
                "tags": ["energy", "oil_gas", "commodities"]
            },
            # Generic commodities
            "OIL": {
                "keywords": ["crude oil", "brent", "wti", "opec"],
                "tags": ["oil_gas", "commodities", "energy", "inflation_driver"]
            }
        }
        
    def extract_tags(self, text: str) -> List[str]:
        """
        Знаходить компанії/сутності в тексті та повертає їх теги впливу.
        """
        text_lower = text.lower()
        extracted_tags = set()
        found_entities = []
        
        for entity_code, data in self.entity_graph.items():
            # Шукаємо за ключовими словами
            for keyword in data["keywords"]:
                # Використовуємо межі слів для уникнення хибних спрацьовувань
                pattern = r'\b' + re.escape(keyword) + r'\b'
                if re.search(pattern, text_lower):
                    extracted_tags.update(data["tags"])
                    if entity_code not in found_entities:
                        found_entities.append(entity_code)
                    break # Переходимо до наступної сутності
                    
        if found_entities:
            logger.info(f"EntityLinker found entities: {found_entities} -> Tags: {list(extracted_tags)}")
            
        return list(extracted_tags)
