#!/usr/bin/env python3
"""
Автоматизований скрипт для повного пайплайну
Запускає: prepare → важкі моделі → continue

Використання:
    python scripts/run_full_pipeline_automated.py --batch-name main_database
"""

import sys
import os
import asyncio
import argparse
from pathlib import Path

# Додати проект до шляху
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.cli.pipeline_executor import PipelineExecutor
from src.config.unified_config_manager import UnifiedConfigManager
from src.pipeline.hybrid_orchestrator import HybridOrchestrator
from src.core.logging.logger import ProjectLogger

logger = ProjectLogger.get_logger(__name__)


async def run_heavy_models_locally(batch_name: str):
    """
    Запуск важких моделей локально (симуляція Colab)
    Використовує той самий логік, що й colab_clean_cell.py
    """
    logger.info("=" * 80)
    logger.info("🚀 ЗАПУСК ВАЖКИХ МОДЕЛЕЙ ЛОКАЛЬНО (СИМУЛЯЦІЯ COLAB)")
    logger.info("=" * 80)
    
    try:
        # Встановлення змінної середовища для MLflow
        import os
        os.environ['MLFLOW_ALLOW_FILE_STORE'] = 'true'
        logger.info("✅ MLFLOW_ALLOW_FILE_STORE=true встановлено")
        
        # Імпорт компонентів для важких моделей
        from scripts.colab.colab_clean_cell import ColabTrainingController
        
        # Ініціалізація контролера
        controller = ColabTrainingController()
        controller.initialize()
        
        # Запуск тренування
        controller.run_training_pipeline()
        
        logger.info("✅ Важкі моделі завершено успішно")
        return True
        
    except Exception as e:
        logger.exception(f"❌ Помилка при запуску важких моделей: {e}")
        return False


async def main():
    """Головна функція для автоматизованого запуску"""
    parser = argparse.ArgumentParser(description='Автоматизований повний пайплайн')
    parser.add_argument('--batch-name', default='main_database', help='Назва батчу')
    parser.add_argument('--skip-prepare', action='store_true', help='Пропустити prepare mode')
    parser.add_argument('--skip-heavy', action='store_true', help='Пропустити важкі моделі')
    parser.add_argument('--skip-continue', action='store_true', help='Пропустити continue mode')
    
    args = parser.parse_args()
    
    logger.info("=" * 80)
    logger.info("🚀 АВТОМАТИЗОВАНИЙ ПОВНИЙ ПАЙПЛАЙН")
    logger.info(f"   Batch: {args.batch_name}")
    logger.info("=" * 80)
    
    # Крок 1: Prepare mode (етапи 0-3)
    if not args.skip_prepare:
        logger.info("\n" + "=" * 80)
        logger.info("КРОК 1: Prepare Mode (етапи 0-3)")
        logger.info("=" * 80)
        
        try:
            config_manager = UnifiedConfigManager()
            orchestrator = HybridOrchestrator(config_manager, batch_name=args.batch_name)
            
            # Визначення тікерів та таймфреймів
            assets_config = config_manager.get_config('assets', {})
            tickers = assets_config.get('tickers', ['AMD', 'NVDA', 'AAPL'])
            timeframes = ['15m', '1h', '1d']
            
            logger.info(f"📊 Тікери: {tickers}")
            logger.info(f"⏱️  Таймфрейми: {timeframes}")
            
            # Запуск prepare mode
            results = await PipelineExecutor.execute_prepare_mode(
                orchestrator, tickers, timeframes,
                test_ticker=None,
                test_target=None,
                test_model=None,
                epochs=1,
                max_iterations=1
            )
            
            logger.info("✅ Prepare mode завершено успішно")
            
        except Exception as e:
            logger.exception(f"❌ Помилка в prepare mode: {e}")
            return 1
    else:
        logger.info("\n⏭️  Prepare mode пропущено (--skip-prepare)")
    
    # Крок 2: Важкі моделі (симуляція Colab)
    if not args.skip_heavy:
        logger.info("\n" + "=" * 80)
        logger.info("КРОК 2: Важкі моделі (симуляція Colab)")
        logger.info("=" * 80)
        
        success = await run_heavy_models_locally(args.batch_name)
        
        if not success:
            logger.error("❌ Важкі моделі завершилися з помилкою")
            return 1
    else:
        logger.info("\n⏭️  Важкі моделі пропущено (--skip-heavy)")
    
    # Крок 3: Continue mode (етапи 6-7)
    if not args.skip_continue:
        logger.info("\n" + "=" * 80)
        logger.info("КРОК 3: Continue Mode (етапи 6-7)")
        logger.info("=" * 80)
        
        try:
            config_manager = UnifiedConfigManager()
            orchestrator = HybridOrchestrator(config_manager, batch_name=args.batch_name)
            
            # Створення args об'єкта для continue mode
            class Args:
                def __init__(self, batch_name):
                    self.batch_name = batch_name
                    self.test_ticker = None
                    self.test_target = None
                    self.test_model = None
                    self.epochs = 1
                    self.max_iterations = 1
                    self.stages = None
            
            args_obj = Args(args.batch_name)
            
            # Запуск continue mode
            results = await PipelineExecutor.execute_continue_mode(
                orchestrator, args_obj
            )
            
            logger.info("✅ Continue mode завершено успішно")
            
        except Exception as e:
            logger.exception(f"❌ Помилка в continue mode: {e}")
            return 1
    else:
        logger.info("\n⏭️  Continue mode пропущено (--skip-continue)")
    
    logger.info("\n" + "=" * 80)
    logger.info("✅ АВТОМАТИЗОВАНИЙ ПОВНИЙ ПАЙПЛАЙН ЗАВЕРШЕНО УСПІШНО")
    logger.info("=" * 80)
    
    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
