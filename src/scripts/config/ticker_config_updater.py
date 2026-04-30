"""
Ticker Configuration Updater
Утиліта для синхронізації тікерів між різними конфігураціями.
"""

import json
import os
import sys
from pathlib import Path
from typing import List, Dict, Any

from src.core.logging.logger import ProjectLogger
from config.tickers import get_tickers, get_category_stats

class TickerConfigUpdater:
    """Клас для оновлення конфігурацій тікерів"""
    
    def __init__(self, project_root: str = None):
        ProjectLogger.setup_logging()
        self.logger = ProjectLogger.get_logger("TickerConfigUpdater")
        self.project_root = Path(project_root) if project_root else Path(__file__).parent.parent.parent.parent
        self.collectors_config_path = self.project_root / "collectors" / "collectors_config.json"
        
    def update_collectors_config(self, category: str = "all") -> bool:
        """
        Оновити target_tickers в collectors_config.json
        """
        try:
            # Перевіряємо шлях
            if not self.collectors_config_path.exists():
                self.logger.error(f"Configuration file not found at {self.collectors_config_path}")
                return False

            # Завантажуємо конфігурацію
            with open(self.collectors_config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            # Отримуємо тікери
            tickers = get_tickers(category)
            
            # Оновлюємо target_tickers для YF колектора
            if 'collectors' in config and 'yf' in config['collectors']:
                config['collectors']['yf']['additional_params']['target_tickers'] = tickers
                self.logger.info(f"Updated YF collector with {len(tickers)} tickers from category '{category}'")
            else:
                self.logger.error("YF collector configuration not found")
                return False
            
            # Зберігаємо конфігурацію
            with open(self.collectors_config_path, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"Configuration saved to {self.collectors_config_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error updating collectors config: {e}")
            return False
    
    def get_current_config_tickers(self) -> List[str]:
        """
        Отримати поточний список тікерів з конфігурації
        """
        try:
            if not self.collectors_config_path.exists():
                return []
            with open(self.collectors_config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            return config['collectors']['yf']['additional_params']['target_tickers']
        except Exception as e:
            self.logger.error(f"Error reading current config: {e}")
            return []
    
    def compare_configs(self, category: str = "all") -> Dict[str, List[str]]:
        """
        Порівняти поточну конфігурацію з централізованим списком
        """
        current = set(self.get_current_config_tickers())
        centralized = set(get_tickers(category))
        
        return {
            "current": sorted(current),
            "centralized": sorted(centralized),
            "missing": sorted(centralized - current),
            "extra": sorted(current - centralized)
        }
    
    def sync_all_configs(self, category: str = "all") -> bool:
        """
        Синхронізувати усі конфігурації з централізованим списком
        """
        self.logger.info(f"Syncing all ticker configurations with category '{category}'...")
        success = self.update_collectors_config(category)
        if success:
            self.logger.info("All configurations synchronized successfully")
        else:
            self.logger.error("Failed to synchronize configurations")
        return success
    
    def print_status(self):
        """Вивести статус поточної конфігурації"""
        self.logger.info("=== Ticker Configuration Status ===")
        current = self.get_current_config_tickers()
        stats = get_category_stats()
        
        self.logger.info(f"Current config tickers: {len(current)}")
        self.logger.info(f"Available categories: {len(stats)}")
        
        self.logger.info("--- Category Statistics ---")
        for category, count in stats.items():
            self.logger.info(f"{category}: {count} tickers")
        
        self.logger.info("--- Comparison with 'all' category ---")
        comparison = self.compare_configs("all")
        self.logger.info(f"Current: {len(comparison['current'])}")
        self.logger.info(f"Centralized: {len(comparison['centralized'])}")
        self.logger.info(f"Missing: {len(comparison['missing'])}")
        self.logger.info(f"Extra: {len(comparison['extra'])}")

def main():
    """CLI Entry Point"""
    import argparse
    parser = argparse.ArgumentParser(description='Update ticker configurations')
    parser.add_argument('--category', default='all', choices=['core', 'all', 'tech', 'etf', 'sp500', 'finance', 'healthcare', 'energy', 'consumer', 'industrial', 'materials', 'utilities', 'realestate', 'communication', 'international', 'crypto'], help='Ticker category to use')
    parser.add_argument('--status', action='store_true', help='Show current status')
    parser.add_argument('--sync', action='store_true', help='Sync configurations')
    parser.add_argument('--compare', action='store_true', help='Compare configurations')
    
    args = parser.parse_args()
    updater = TickerConfigUpdater()
    
    if args.status:
        updater.print_status()
    elif args.sync:
        updater.sync_all_configs(args.category)
    elif args.compare:
        comp = updater.compare_configs(args.category)
        updater.logger.info(f"=== Comparison with '{args.category}' category ===")
        updater.logger.info(f"Current: {len(comp['current'])}")
        updater.logger.info(f"Centralized: {len(comp['centralized'])}")
        updater.logger.info(f"Missing: {len(comp['missing'])}")
        updater.logger.info(f"Extra: {len(comp['extra'])}")
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
