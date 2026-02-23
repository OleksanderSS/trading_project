# data_integration_helper.py - Інтеграція нової системи накопичення

import logging
from typing import Dict, Any
import pandas as pd

logger = logging.getLogger(__name__)

def integrate_with_stage_manager():
    """
    Інтегрує EnhancedDataAccumulator з існуючим StageManager
    """
    try:
        from utils.enhanced_data_accumulator import enhanced_accumulator
        
        # Модифікуємо StageManager для використання нового акумулятора
        from core.stages import stage_manager
        
        # Зберігаємо оригінальний метод
        original_accumulate = stage_manager.StageManager._accumulate_stage2_data
        
        def enhanced_accumulate_stage2_data(self, raw_news, merged_df, pivots):
            """Покращена версія накопичення data"""
            logger.info("[StageManager] Використання покращеного накопичення...")
            
            # Викликаємо оригінальний метод для сумісності
            try:
                original_accumulate(self, raw_news, merged_df, pivots)
            except Exception as e:
                logger.warning(f"[StageManager] Оригінальне накопичення не вдалося: {e}")
            
            # Використовуємо нову систему
            try:
                saved_files = enhanced_accumulator.accumulate_stage2_data(merged_df)
                logger.info(f"[StageManager] Покращене накопичення: {saved_files}")
                
                # Створюємо пакети для Colab
                packages = enhanced_accumulator.create_colab_packages()
                logger.info(f"[StageManager] Створено пакети для Colab: {list(packages.keys())}")
                
            except Exception as e:
                logger.error(f"[StageManager] Помилка покращеного накопичення: {e}")
        
        # Замінюємо метод
        stage_manager.StageManager._accumulate_stage2_data = enhanced_accumulate_stage2_data
        
        logger.info("[DataIntegration] Enhanced accumulator інтегровано")
        return True
        
    except Exception as e:
        logger.error(f"[DataIntegration] Помилка інтеграції: {e}")
        return False

def setup_stage1_accumulation():
    """
    Налаштовує накопичення для Stage 1
    """
    try:
        from utils.enhanced_data_accumulator import enhanced_accumulator
        # ВИПРАВЛЕНО: використовуємо StageManager замість видаленої функції
        from core.stages.stage_manager import StageManager
        
        # Зберігаємо оригінальну функцію
        original_collect = run_stage_1_collect
        
        def enhanced_collect():
            """Покращена версія збору data з накопиченням"""
            logger.info("[Stage1] Використання покращеного збору з накопиченням...")
            
            # Викликаємо оригінальну функцію
            stage1_data = original_collect()
            
            # Накопичуємо дані
            try:
                saved_files = enhanced_accumulator.save_stage1_data(stage1_data)
                logger.info(f"[Stage1] Накопичено Stage 1: {saved_files}")
            except Exception as e:
                logger.error(f"[Stage1] Помилка накопичення Stage 1: {e}")
            
            return stage1_data
        
        # Замінюємо функцію
        import core.stages.stage_1_collectors_layer
        core.stages.stage_1_collectors_layer.run_stage_1_collect = enhanced_collect
        
        logger.info("[DataIntegration] Stage 1 accumulation налаштовано")
        return True
        
    except Exception as e:
        logger.error(f"[DataIntegration] Помилка налаштування Stage 1: {e}")
        return False

def get_accumulation_status():
    """
    Повертає статус системи накопичення
    """
    try:
        from utils.enhanced_data_accumulator import enhanced_accumulator
        return enhanced_accumulator.get_accumulation_stats()
    except Exception as e:
        logger.error(f"[DataIntegration] Помилка отримання статусу: {e}")
        return {}

def create_manual_packages():
    """
    Створює пакети для Colab вручну
    """
    try:
        from utils.enhanced_data_accumulator import enhanced_accumulator
        
        packages = enhanced_accumulator.create_colab_packages()
        logger.info(f"[DataIntegration] Створено пакети: {list(packages.keys())}")
        
        return packages
    except Exception as e:
        logger.error(f"[DataIntegration] Помилка створення пакетів: {e}")
        return {}

# Автоматична інтеграція при імпорті
def auto_integrate():
    """Автоматично інтегрує покращення"""
    logger.info("[DataIntegration] Автоматична інтеграція...")
    
    success = True
    success &= integrate_with_stage_manager()
    success &= setup_stage1_accumulation()
    
    if success:
        logger.info("[DataIntegration] [OK] Всі покращення інтегровано")
    else:
        logger.warning("[DataIntegration] [WARN] Деякі покращення не вдалося інтегрувати")
    
    return success

# Запускаємо автоматичну інтеграцію
if __name__ != "__main__":
    auto_integrate()
