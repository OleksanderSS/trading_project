"""
Когнітивний Пайплайн (Cognitive Pipeline)

Цей скрипт демонструє, як об'єднати:
1. Текст (стаття, книга, новина)
2. Когнітивний Шаблон (Cognitive Extractor)
3. Universal Toolbox (Доступ до всіх інструментів)

Для інтеграції в dean_os цей пайплайн можна викликати як окремий етап (Stage),
або як вбудований метод в UnifiedResearchAgent.
"""

import asyncio
from typing import Dict, Any
import logging
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.stdout.reconfigure(encoding='utf-8')

# Імпорт наших нових модулів
from src.agents.templates.cognitive_extractor import get_cognitive_prompt
from src.agents.tools.universal_registry import toolbox

logger = logging.getLogger(__name__)

class CognitiveAnalyst:
    def __init__(self, model_name: str = "gpt-4o"):
        self.model_name = model_name
        self.tools = toolbox.get_all_tools()
        
    async def analyze_document(self, text: str, metadata: dict = None) -> str:
        """
        Аналізує документ, використовуючи когнітивний промпт.
        Тут має бути виклик LLM (через openai, anthropic, чи внутрішній dean_os клієнт).
        """
        system_prompt = get_cognitive_prompt(document_metadata=metadata)
        
        logger.info(f"Запуск когнітивного аналізу. Довжина тексту: {len(text)} символів.")
        
        # Симуляція виклику LLM (замініть на реальний виклик вашої LLM)
        # prompt = system_prompt + "\n\nTEXT TO ANALYZE:\n" + text
        # response = await llm_client.generate(prompt)
        
        # Simulated response for demonstration
        simulated_response = f"""
[CORE MECHANISM]: The text describes a scenario where systemic fragility is exposed by a minor trigger, leading to cascaded failures across sectors.
[CAUSAL CHAIN]: Minor Trigger -> Institutional Failure -> Supply Chain Collapse -> Inflationary Spike
[BEHAVIORAL/PSYCHOLOGICAL DRIVERS]: Widespread panic hoarding driven by sudden loss of trust in fiat stability.
[MODERN ANALOGIES]: Similar to the Arab Spring where localized food inflation triggered massive regime changes. Currently applicable to the XYZ sector.
[MARKET/SYSTEMIC VULNERABILITIES]: High exposure in logistics and regional banking.
        """
        
        return simulated_response

    async def cross_domain_research(self, query: str) -> Dict[str, Any]:
        """
        Приклад того, як аналітик може використати Універсальну Скриньку
        для перевірки гіпотези, яка виникла під час читання.
        """
        logger.info(f"Перевірка гіпотези: {query}")
        
        # Якщо LLM вирішила, що треба перевірити погоду та ціни на нафту:
        results = {}
        
        try:
            # Викликаємо інструменти з реєстру
            weather_func = self.tools.get("check_weather")
            if weather_func:
                results["weather"] = await weather_func(29.76, -95.36, 1) # Техас
                
            eia_func = self.tools.get("get_oil_prices")
            if eia_func:
                results["oil"] = await eia_func(days_back=2)
                
        except Exception as e:
            logger.error(f"Помилка під час крос-доменного пошуку: {e}")
            
        return results

async def demo():
    analyst = CognitiveAnalyst()
    
    # 1. Читання "складного" тексту
    sample_text = "В країні Х розпочалася сильна посуха, що призвело до дефіциту зерна. Уряд спробував зафіксувати ціни, що викликало чорний ринок..."
    metadata = {"Title": "Economic Origins of Dictatorship", "Author": "Acemoglu"}
    
    analysis = await analyst.analyze_document(sample_text, metadata)
    print("=== COGNITIVE EXTRACTION RESULT ===")
    print(analysis)
    
    # 2. Перевірка через Universal Toolbox
    print("\n=== CROSS-DOMAIN RESEARCH ===")
    research = await analyst.cross_domain_research("Перевірити поточні ціни на нафту та погоду в ключових регіонах")
    print(research)

if __name__ == "__main__":
    asyncio.run(demo())
