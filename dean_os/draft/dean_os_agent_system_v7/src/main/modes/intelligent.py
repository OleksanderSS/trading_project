#!/usr/bin/env python3
"""
Intelligent Mode - Інтелектуальний режим з усіма розширеними модулями
"""

import logging
from typing import Any, cast

from src.main.system_orchestrator import SystemOrchestrator


class IntelligentMode:
    """Інтелектуальний режим з усіма розширеними модулями"""

    def __init__(self, args):
        self.args = args
        self.logger = logging.getLogger(__name__)

        # Ініціалізуємо System Orchestrator
        self.orchestrator = SystemOrchestrator()

    def run(self) -> dict[str, Any]:
        """Запуск інтелектуального режиму"""
        self.logger.info("🧠 Starting INTELLIGENT Mode...")

        # Отримуємо тікери з аргументів
        tickers = self._get_tickers()

        # Запускаємо інтелектуальний pipeline через SystemOrchestrator
        import asyncio
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        if loop.is_running():
            # If we're already in an event loop (e.g. pytest or another async run)
            import nest_asyncio
            nest_asyncio.apply()
            results = loop.run_until_complete(self.orchestrator.run_mode(mode='intelligent', tickers=tickers))
        else:
            results = asyncio.run(self.orchestrator.run_mode(mode='intelligent', tickers=tickers))

        # Додаємо інформацію про режим
        results['mode'] = 'intelligent'
        results['system_version'] = '2.0.0'

        return cast(dict[str, Any], results)

    def _get_tickers(self) -> list[str]:
        """Отримати тікери з аргументів"""
        if self.args.tickers:
            return [t.strip().upper() for t in self.args.tickers.split(',')]
        else:
            return ["SPY", "QQQ", "TSLA", "NVDA"]

