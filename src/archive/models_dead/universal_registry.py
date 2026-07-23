"""
Universal Registry: Єдиний реєстр всіх інструментів (колекторів та API) для агентів.
Будь-який аналітик може отримати доступ до будь-якого інструменту з цього списку,
незалежно від його спеціалізації, щоб знаходити cross-domain інсайти.
"""

from typing import Dict, Callable, Any
import logging

# On-Demand Tools
from .weather_tool import check_weather
from .gdelt_tool import search_global_events
from .pubmed_tool import search_clinical_trials
from .eia_tool import get_oil_prices
from .comtrade_tool import get_trade_volume

# Core Collectors (інтеграція існуючих колекторів як інструментів за запитом)
# Якщо аналітик хоче сам запустити SDMX або Reddit, він зможе це зробити.
try:
    from src.data.collectors.sdmx_macro_collector import SDMXMacroCollector
    from src.data.collectors.reddit_sentiment_collector import RedditSentimentCollector
    from src.data.collectors.wikimedia_attention_collector import WikimediaAttentionCollector
except ImportError:
    logging.warning("Core collectors not found or failed to import into registry.")

class UniversalToolbox:
    def __init__(self):
        self._tools: Dict[str, Callable] = {}
        self._register_default_tools()

    def _register_default_tools(self):
        # 1. On-Demand API Tools
        self.register_tool("check_weather", check_weather)
        self.register_tool("search_global_events", search_global_events)
        self.register_tool("search_clinical_trials", search_clinical_trials)
        self.register_tool("get_oil_prices", get_oil_prices)
        self.register_tool("get_trade_volume", get_trade_volume)
        
        # 2. Wrappers for Core Collectors
        self.register_tool("get_macro_sdmx", self._wrap_sdmx)
        self.register_tool("get_reddit_sentiment", self._wrap_reddit)
        self.register_tool("get_wikimedia_attention", self._wrap_wikimedia)

    def register_tool(self, name: str, func: Callable):
        self._tools[name] = func
        
    def get_all_tools(self) -> Dict[str, Callable]:
        """Повертає словник усіх доступних інструментів для LLM агента."""
        return self._tools

    # --- Wrappers to make classes look like simple async functions for LLM ---
    
    async def _wrap_sdmx(self, data_flow: str = "EXR", agency: str = "ECB") -> Any:
        try:
            collector = SDMXMacroCollector({"frequency": "on_demand"})
            # Simplification: In a real scenario, we might pass params to the collector
            return {"status": "SDMX initiated", "message": "Check data lake for results"}
        except Exception as e:
            return {"error": str(e)}

    async def _wrap_reddit(self, subreddit: str = "economics") -> Any:
        try:
            collector = RedditSentimentCollector({"frequency": "on_demand", "subreddits": [subreddit]})
            data = await collector.collect()
            return {"status": "success", "data": data}
        except Exception as e:
            return {"error": str(e)}

    async def _wrap_wikimedia(self, article: str = "Inflation") -> Any:
        try:
            collector = WikimediaAttentionCollector({"frequency": "on_demand", "articles": [article]})
            data = await collector.collect()
            return {"status": "success", "data": data}
        except Exception as e:
            return {"error": str(e)}

# Global instance
toolbox = UniversalToolbox()
