"""
Ticker Configuration Updater
Утилandand for синхронandforцandї тandкерandв мandж рandwithними конфandгурацandями
"""

import json
import os
import sys
from pathlib import Path
from typing import List, Dict, Any

# Додаємо шлях до проекту
current_dir = Path(__file__).parent
project_root = current_dir.parent
sys.path.append(str(project_root))

from config.tickers import get_tickers, get_category_stats

class TickerConfigUpdater:
    """Клас for оновлення конфandгурацandй тandкерandв"""
    
    def __init__(self, project_root: str = None):
        self.project_root = Path(project_root) if project_root else Path(__file__).parent.parent
        self.collectors_config_path = self.project_root / "collectors" / "collectors_config.json"
        
    def update_collectors_config(self, category: str = "all") -> bool:
        """
        Оновити target_tickers в collectors_config.json
        
        Args:
            category: Категорandя тandкерandв (core, all, tech, etf, etc.)
            
        Returns:
            bool: Успandшнandсть оновлення
        """
        try:
            # Заванandжуємо конфandгурацandю
            with open(self.collectors_config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            # Отримуємо тandкери
            tickers = get_tickers(category)
            
            # Оновлюємо target_tickers for YF колектора
            if 'collectors' in config and 'yf' in config['collectors']:
                config['collectors']['yf']['additional_params']['target_tickers'] = tickers
                print(f"[SUCCESS] Updated YF collector with {len(tickers)} tickers from category '{category}'")
            else:
                print("[ERROR] YF collector configuration not found")
                return False
            
            # Зберandгаємо конфandгурацandю
            with open(self.collectors_config_path, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
            
            print(f"[SUCCESS] Configuration saved to {self.collectors_config_path}")
            return True
            
        except Exception as e:
            print(f"[ERROR] Error updating collectors config: {e}")
            return False
    
    def get_current_config_tickers(self) -> List[str]:
        """
        Отримати поточний список тandкерandв with конфandгурацandї
        
        Returns:
            List[str]: Поточний список тandкерandв
        """
        try:
            with open(self.collectors_config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            return config['collectors']['yf']['additional_params']['target_tickers']
            
        except Exception as e:
            print(f"[ERROR] Error reading current config: {e}")
            return []
    
    def compare_configs(self, category: str = "all") -> Dict[str, List[str]]:
        """
        Порandвняти поточну конфandгурацandю with централandwithованим списком
        
        Args:
            category: Категорandя for порandвняння
            
        Returns:
            Dict[str, List[str]]: {
                "current": поточнand тandкери,
                "centralized": централandwithованand тandкери,
                "missing": вandдсутнand в поточнandй,
                "extra": forйвand в поточнandй
            }
        """
        current = set(self.get_current_config_tickers())
        centralized = set(get_tickers(category))
        
        return {
            "current": sorted(list(current)),
            "centralized": sorted(list(centralized)),
            "missing": sorted(list(centralized - current)),
            "extra": sorted(list(current - centralized))
        }
    
    def sync_all_configs(self, category: str = "all") -> bool:
        """
        Синхронandwithувати all конфandгурацandї with централandwithованим списком
        
        Args:
            category: Категорandя тandкерandв for синхронandforцandї
            
        Returns:
            bool: Успandшнandсть синхронandforцandї
        """
        print(f"[SYNC] Syncing all ticker configurations with category '{category}'...")
        
        # Оновлюємо collectors config
        success = self.update_collectors_config(category)
        
        if success:
            print("[SUCCESS] All configurations synchronized successfully")
        else:
            print("[ERROR] Failed to synchronize configurations")
        
        return success
    
    def print_status(self):
        """Вивести сandтус поточної конфandгурацandї"""
        print("=== Ticker Configuration Status ===")
        
        current = self.get_current_config_tickers()
        stats = get_category_stats()
        
        print(f"Current config tickers: {len(current)}")
        print(f"Available categories: {len(stats)}")
        
        print("\n=== Category Statistics ===")
        for category, count in stats.items():
            print(f"{category}: {count} tickers")
        
        print("\n=== Comparison with 'all' category ===")
        comparison = self.compare_configs("all")
        
        print(f"Current: {len(comparison['current'])}")
        print(f"Centralized: {len(comparison['centralized'])}")
        print(f"Missing: {len(comparison['missing'])}")
        print(f"Extra: {len(comparison['extra'])}")
        
        if comparison['missing']:
            print(f"\nMissing tickers: {comparison['missing'][:10]}...")
        if comparison['extra']:
            print(f"Extra tickers: {comparison['extra'][:10]}...")

def main():
    """Основна функцandя for CLI"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Update ticker configurations')
    parser.add_argument('--category', default='all', 
                       choices=['core', 'all', 'tech', 'etf', 'sp500', 'finance', 
                               'healthcare', 'energy', 'consumer', 'industrial',
                               'materials', 'utilities', 'realestate', 'communication',
                               'international', 'crypto'],
                       help='Ticker category to use')
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
        comparison = updater.compare_configs(args.category)
        print(f"=== Comparison with '{args.category}' category ===")
        print(f"Current: {len(comparison['current'])}")
        print(f"Centralized: {len(comparison['centralized'])}")
        print(f"Missing: {len(comparison['missing'])}")
        print(f"Extra: {len(comparison['extra'])}")
        
        if comparison['missing']:
            print(f"\nMissing: {comparison['missing']}")
        if comparison['extra']:
            print(f"\nExtra: {comparison['extra']}")
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
