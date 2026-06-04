
from src.core.logging.logger import ProjectLogger

from .models import MarketEvent

logger = ProjectLogger.get_logger("EventScanner")

class EventScanner:
    """Scans external sources for news and market events."""

    def __init__(self, config: dict | None = None):
        self.sources = config or {
            'yahoo_finance': 'https://finance.yahoo.com/news',
            'reuters': 'https://www.reuters.com/finance'
        }

    def scan_all_sources(self) -> list[MarketEvent]:
        """Triggers a scan across all configured sources."""
        all_events = []
        for source_name, url in self.sources.items():
            try:
                events = self._fetch_news(source_name, url)
                all_events.extend(events)
            except Exception as e:
                logger.error(f"Error scanning {source_name}: {e}")
        return all_events

    def _fetch_news(self, source: str, url: str) -> list[MarketEvent]:
        """Fetch and parse news from a source when a concrete connector is configured."""
        logger.info("No live connector configured for %s (%s); skipping scan.", source, url)
        return []
