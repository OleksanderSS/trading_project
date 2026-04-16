#!/usr/bin/env python3
"""
Скрипт для запуску етапів 5-7 (Прогнози + Торгівля + Оцінка) з існуючими моделями.

Використання:
    python scripts/run_stages_5_7.py --batch test_ticker_amd_target_return_1d_ep5_iter5
"""

import asyncio
import argparse
import sys
import json
from pathlib import Path
from datetime import datetime

# Додаємо кореневу директорію проєкту до sys.path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.config.unified_config_manager import UnifiedConfigManager
from src.pipeline.hybrid_orchestrator import HybridOrchestrator
from src.core.logging.logger import ProjectLogger
import pandas as pd

logger = ProjectLogger.get_logger(__name__)


async def main():
    parser = argparse.ArgumentParser(description='Запуск етапів 5-7 (Прогнози + Торгівля + Оцінка)')
    parser.add_argument(
        '--batch',
        required=True,
        help='Назва батчу (наприклад, test_ticker_amd_target_return_1d_ep5_iter5)'
    )
    parser.add_argument(
        '--base-dir',
        default='data/colab/accumulated',
        help='Базова директорія для батчів'
    )
    
    args = parser.parse_args()
    
    batch_name = args.batch
    base_dir = Path(args.base_dir)
    batch_dir = base_dir / batch_name
    
    logger.info(f"🚀 Запуск етапів 5-7 для батчу: {batch_name}")
    logger.info(f"📁 Директорія батчу: {batch_dir}")
    
    # Перевіряємо, чи існує батч
    if not batch_dir.exists():
        logger.error(f"❌ Батч не знайдено: {batch_dir}")
        return
    
    # Перевіряємо, чи існують необхідні файли
    features_path = batch_dir / "features.parquet"
    targets_path = batch_dir / "targets.parquet"
    colab_results_path = batch_dir / "colab_results_summary.json"
    
    if not features_path.exists():
        logger.error(f"❌ Не знайдено features.parquet: {features_path}")
        return
    
    if not targets_path.exists():
        logger.error(f"❌ Не знайдено targets.parquet: {targets_path}")
        return
    
    if not colab_results_path.exists():
        logger.error(f"❌ Не знайдено colab_results_summary.json: {colab_results_path}")
        return
    
    logger.info("✅ Всі необхідні файли знайдено")
    
    # Завантажуємо дані
    logger.info("📥 Завантаження даних...")
    features_df = pd.read_parquet(features_path)
    targets_df = pd.read_parquet(targets_path)
    
    with open(colab_results_path, 'r') as f:
        colab_results = json.load(f)
    
    logger.info(f"✅ Дані завантажено:")
    logger.info(f"   Features: {features_df.shape}")
    logger.info(f"   Targets: {targets_df.shape}")
    logger.info(f"   Colab результати: {len(colab_results.get('ticker_results', {}))} тікерів")
    
    # Ініціалізуємо конфіг та оркестратор
    config_manager = UnifiedConfigManager()
    orchestrator = HybridOrchestrator(config_manager, batch_name=batch_name)
    
    # Запускаємо етапи 5-7
    logger.info("\n" + "🚀 "*40)
    logger.info("🚀 ЗАПУСК ЕТАПІВ 5-7")
    logger.info("🚀 "*40)
    
    try:
        # Запускаємо фінальні етапи 4-7
        logger.info("\n🎯 ЕТАПИ 4-7: Запуск тренування легких моделей та аналізу результатів...")
        final_results = await orchestrator.run_final_stages(
            features_df=features_df,
            targets_df=targets_df,
            colab_results=colab_results,
            tickers=None,  # Використовуємо всі тікери з даних
            timeframes=None,  # Використовуємо всі таймфрейми з даних
            batch_name=args.batch_name  # ✅ ADD batch_name
        )
        
        # Виводимо результати
        logger.info("\n" + "="*80)
        logger.info("📊 РЕЗУЛЬТАТИ ЕТАПІВ 5-7")
        logger.info("="*80)
        
        if final_results:
            logger.info(f"✅ Етапи 5-7 завершено")
            logger.info(f"📋 Статус: {final_results.get('status')}")
            
            if 'predictions' in final_results:
                logger.info(f"💡 Прогнози: {len(final_results.get('predictions', []))} прогнозів")
            
            if 'trading_activity' in final_results:
                logger.info(f"🎲 Торгівельна активність: {len(final_results.get('trading_activity', []))} операцій")
            
            if 'portfolio_summary' in final_results:
                portfolio = final_results.get('portfolio_summary', {})
                logger.info(f"💰 Портфель: {portfolio.get('total_value', 0):.2f}")
            
            if 'evaluation_metrics' in final_results:
                metrics = final_results.get('evaluation_metrics', {})
                logger.info(f"📈 Метрики: {metrics}")
        
        logger.info("\n✅ Етапи 5-7 завершено успішно!")
        
    except Exception as e:
        logger.error(f"❌ Помилка при запуску етапів 5-7: {e}", exc_info=True)
        return


if __name__ == '__main__':
    asyncio.run(main())
