# integrate_simple_accumulator.py - Інтеграція простої системи накопичення

import logging

logger = logging.getLogger(__name__)

def integrate_simple_accumulation():
    """
    Інтегрує просту систему накопичення в існуючий pipeline
    """
    try:
        from utils.simple_accumulator import simple_accumulator
        from core.stages import stage_manager
        
        # Інтегруємо в Stage 1
        original_run_stage_1 = stage_manager.StageManager.run_stage_1
        
        def enhanced_run_stage_1(self, debug_no_network=False, force_refresh=False):
            logger.info("[StageManager] Stage 1 з накопиченням...")
            
            # Викликаємо оригінальний метод
            stage1_data = original_run_stage_1(self, debug_no_network, force_refresh)
            
            # Зберігаємо дані
            try:
                saved_files = simple_accumulator.save_stage1(stage1_data)
                logger.info(f"[StageManager] Stage 1 saved: {saved_files}")
            except Exception as e:
                logger.error(f"[StageManager] Помилка збереження Stage 1: {e}")
            
            return stage1_data
        
        # Інтегруємо в Stage 2
        original_run_stage_2 = stage_manager.StageManager.run_stage_2
        
        def enhanced_run_stage_2(self, stage1_data, tickers=None, time_frames=None, force_refresh=False):
            logger.info("[StageManager] Stage 2 з накопиченням...")
            
            # Викликаємо оригінальний метод
            raw_news, merged_df, pivots = original_run_stage_2(self, stage1_data, tickers, time_frames, force_refresh)
            
            # Зберігаємо дані
            try:
                saved_files = simple_accumulator.save_stage2(merged_df)
                logger.info(f"[StageManager] Stage 2 saved: {saved_files}")
            except Exception as e:
                logger.error(f"[StageManager] Помилка збереження Stage 2: {e}")
            
            return raw_news, merged_df, pivots
        
        # Інтегруємо в Stage 3
        original_run_stage_3 = stage_manager.StageManager.run_stage_3
        
        def enhanced_run_stage_3(self, merged_df, force_refresh=False):
            logger.info("[StageManager] Stage 3 з накопиченням...")
            
            # Викликаємо оригінальний метод
            merged_stage3, context_df, features_df, trigger_data = original_run_stage_3(self, merged_df, force_refresh)
            
            # Зберігаємо дані
            try:
                saved_files = simple_accumulator.save_stage3(features_df)
                logger.info(f"[StageManager] Stage 3 saved: {saved_files}")
            except Exception as e:
                logger.error(f"[StageManager] Помилка збереження Stage 3: {e}")
            
            return merged_stage3, context_df, features_df, trigger_data
        
        # Замінюємо методи
        stage_manager.StageManager.run_stage_1 = enhanced_run_stage_1
        stage_manager.StageManager.run_stage_2 = enhanced_run_stage_2
        stage_manager.StageManager.run_stage_3 = enhanced_run_stage_3
        
        logger.info("[SimpleIntegration] [OK] Проста система накопичення інтегрована")
        return True
        
    except Exception as e:
        logger.error(f"[SimpleIntegration] Помилка інтеграції: {e}")
        return False

def get_accumulation_status():
    """Повертає статус накопичення"""
    try:
        from utils.simple_accumulator import simple_accumulator
        return simple_accumulator.get_status()
    except Exception as e:
        logger.error(f"[SimpleIntegration] Помилка отримання статусу: {e}")
        return {}

# Автоматична інтеграція
if __name__ != "__main__":
    integrate_simple_accumulation()
