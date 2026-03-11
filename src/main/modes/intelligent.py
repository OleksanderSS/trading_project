#!/usr/bin/env python3
"""
Intelligent Mode - Інтелектуальний режим з усіма розширеними модулями
"""

import logging
from typing import Dict, Any, List, Optional
from pathlib import Path

from main.unified_system import UnifiedTradingSystem


class IntelligentMode:
    """Інтелектуальний режим з усіма розширеними модулями"""
    
    def __init__(self, args):
        self.args = args
        self.logger = logging.getLogger(__name__)
        
        # Ініціалізуємо Unified System
        self.unified_system = UnifiedTradingSystem(enable_all_features=True)
    
    def run(self) -> Dict[str, Any]:
        """Запуск інтелектуального режиму"""
        self.logger.info("🧠 Starting INTELLIGENT Mode...")
        
        # Отримуємо тікери з аргументів
        tickers = self._get_tickers()
        
        # Запускаємо інтелектуальний pipeline
        results = self.unified_system.run_intelligent_pipeline(tickers=tickers)
        
        # Додаємо інформацію про режим
        results['mode'] = 'intelligent'
        results['system_version'] = '2.0.0'
        
        return results
    
    def _get_tickers(self) -> List[str]:
        """Отримати тікери з аргументів"""
        if self.args.tickers:
            return [t.strip().upper() for t in self.args.tickers.split(',')]
        else:
            return ["SPY", "QQQ", "TSLA", "NVDA"]
