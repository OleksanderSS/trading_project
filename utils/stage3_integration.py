# stage3_integration.py - Інтеграція накопичення Stage 3 в існуючу систему

import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

def integrate_stage3_accumulation():
    """
    Інтегрує Stage3Accumulator з існуючою системою
    """
    try:
        from utils.stage3_accumulator import stage3_accumulator
        from core.stages import stage_manager
        
        # Модифікуємо StageManager для використання накопичення Stage 3
        original_run_stage_3 = stage_manager.StageManager.run_stage_3
        
        def enhanced_run_stage_3(self, merged_df, force_refresh=False):
            """Покращена версія Stage 3 з накопиченням"""
            logger.info("[StageManager] Використання покращеного Stage 3 з накопиченням...")
            
            # Викликаємо оригінальний метод
            try:
                merged_stage3, context_df, features_df, trigger_data = original_run_stage_3(self, merged_df, force_refresh)
                
                # Накопичуємо дані Stage 3
                try:
                    saved_files = stage3_accumulator.accumulate_stage3_data(
                        features_df, context_df, trigger_data
                    )
                    logger.info(f"[StageManager] Stage 3 накопичено: {saved_files}")
                    
                    # Створюємо пакети для Colab
                    packages = stage3_accumulator.create_colab_packages()
                    logger.info(f"[StageManager] Створено пакети для Colab: {list(packages.keys())}")
                    
                except Exception as e:
                    logger.error(f"[StageManager] Помилка накопичення Stage 3: {e}")
                
                return merged_stage3, context_df, features_df, trigger_data
                
            except Exception as e:
                logger.error(f"[StageManager] Помилка оригінального Stage 3: {e}")
                raise
        
        # Замінюємо метод
        stage_manager.StageManager.run_stage_3 = enhanced_run_stage_3
        
        logger.info("[Stage3Integration] Stage 3 accumulator інтегровано")
        return True
        
    except Exception as e:
        logger.error(f"[Stage3Integration] Помилка інтеграції: {e}")
        return False

def integrate_with_colab_utils():
    """
    Інтегрує з існуючими Colab utils
    """
    try:
        from utils.stage3_accumulator import stage3_accumulator
        from utils.colab_utils import ColabUtils
        
        # Зберігаємо оригінальний метод
        original_export_stage3 = ColabUtils.export_stage3_data
        
        def enhanced_export_stage3_data(self, features_df, context_df=None, trigger_data=None, filename=None):
            """Покращена версія експорту з накопиченням"""
            logger.info("[ColabUtils] Використання покращеного експорту з накопиченням...")
            
            # Викликаємо оригінальний метод
            try:
                original_path = original_export_stage3_data(self, features_df, context_df, trigger_data, filename)
            except Exception as e:
                logger.warning(f"[ColabUtils] Оригінальний експорт не вдалося: {e}")
                original_path = None
            
            # Накопичуємо дані
            try:
                saved_files = stage3_accumulator.accumulate_stage3_data(
                    features_df, context_df, trigger_data
                )
                logger.info(f"[ColabUtils] Накопичено через експорт: {saved_files}")
                
            except Exception as e:
                logger.error(f"[ColabUtils] Помилка накопичення через експорт: {e}")
            
            return original_path
        
        # Замінюємо метод
        ColabUtils.export_stage3_data = enhanced_export_stage3_data
        
        logger.info("[Stage3Integration] Colab utils інтегровано")
        return True
        
    except Exception as e:
        logger.error(f"[Stage3Integration] Помилка інтеграції Colab utils: {e}")
        return False

def setup_auto_integration():
    """
    Налаштовує автоматичну інтеграцію
    """
    logger.info("[Stage3Integration] Налаштування автоматичної інтеграції...")
    
    success = True
    success &= integrate_stage3_accumulation()
    success &= integrate_with_colab_utils()
    
    if success:
        logger.info("[Stage3Integration] [OK] Всі інтеграції Stage 3 успішні")
    else:
        logger.warning("[Stage3Integration] [WARN] Деякі інтеграції Stage 3 не вдалося")
    
    return success

def get_stage3_status():
    """
    Повертає статус системи накопичення Stage 3
    """
    try:
        from utils.stage3_accumulator import stage3_accumulator
        return stage3_accumulator.get_accumulation_stats()
    except Exception as e:
        logger.error(f"[Stage3Integration] Помилка отримання статусу Stage 3: {e}")
        return {}

def create_manual_stage3_packages():
    """
    Створює пакети для Colab вручну
    """
    try:
        from utils.stage3_accumulator import stage3_accumulator
        
        packages = stage3_accumulator.create_colab_packages()
        logger.info(f"[Stage3Integration] Створено Stage 3 пакети: {list(packages.keys())}")
        
        return packages
    except Exception as e:
        logger.error(f"[Stage3Integration] Помилка створення Stage 3 пакетів: {e}")
        return {}

# Автоматична інтеграція при імпорті
def auto_integrate():
    """Автоматично інтегрує покращення Stage 3"""
    logger.info("[Stage3Integration] Автоматична інтеграція Stage 3...")
    
    success = setup_auto_integration()
    
    return success

# Запускаємо автоматичну інтеграцію
if __name__ != "__main__":
    auto_integrate()
