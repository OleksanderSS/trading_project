# integrate_ticker_batch.py - Інтеграція пакетного тренування в pipeline

import logging

logger = logging.getLogger(__name__)

def integrate_ticker_batch_training():
    """
    Інтегрує пакетне тренування по тікерах в pipeline
    """
    try:
        from utils.ticker_batch_trainer import ticker_batch_trainer
        from core.stages import stage_manager
        
        # Модифікуємо Stage 3 для пакетного тренування
        original_run_stage_3 = stage_manager.StageManager.run_stage_3
        
        def enhanced_run_stage_3_with_batch_training(self, merged_df, force_refresh=False):
            """Stage 3 з пакетним тренуванням по тікерах"""
            logger.info("[StageManager] Stage 3 з пакетним тренуванням...")
            
            # Викликаємо оригінальний метод
            merged_stage3, context_df, features_df, trigger_data = original_run_stage_3(self, merged_df, force_refresh)
            
            # Пакетне тренування по тікерах
            try:
                logger.info("[StageManager] Початок пакетного тренування по тікерах...")
                
                # Тренуємо моделі по тікерах
                batch_results = ticker_batch_trainer.train_all_ticker_batches(
                    features_df, 
                    models_to_train=['random_forest', 'xgboost', 'lightgbm']
                )
                
                logger.info(f"[StageManager] Пакетне тренування завершено: {len(batch_results.get('ticker_results', {}))} тікерів")
                
                # Зберігаємо результати в контекст
                if 'context' not in locals():
                    context = {}
                context['ticker_batch_results'] = batch_results
                
            except Exception as e:
                logger.error(f"[StageManager] Помилка пакетного тренування: {e}")
            
            return merged_stage3, context_df, features_df, trigger_data
        
        # Замінюємо метод
        stage_manager.StageManager.run_stage_3 = enhanced_run_stage_3_with_batch_training
        
        logger.info("[TickerBatchIntegration] [OK] Пакетне тренування інтегровано")
        return True
        
    except Exception as e:
        logger.error(f"[TickerBatchIntegration] Помилка інтеграції: {e}")
        return False

def create_ticker_batch_stage():
    """
    Створює окремий етап для пакетного тренування
    """
    try:
        from utils.ticker_batch_trainer import ticker_batch_trainer
        from core.stages import stage_manager
        
        def run_ticker_batch_training(self, features_df, models_to_train=None):
            """
            Окремий етап пакетного тренування по тікерах
            """
            logger.info("[StageManager] Запуск окремого етапу пакетного тренування...")
            
            if features_df.empty:
                logger.warning("[StageManager] Порожні фічі для пакетного тренування")
                return {}
            
            # Тренуємо моделі по тікерах
            batch_results = ticker_batch_trainer.train_all_ticker_batches(
                features_df, models_to_train
            )
            
            return batch_results
        
        # Додаємо новий метод в StageManager
        stage_manager.StageManager.run_ticker_batch_training = run_ticker_batch_training
        
        logger.info("[TickerBatchIntegration] [OK] Окремий етап created")
        return True
        
    except Exception as e:
        logger.error(f"[TickerBatchIntegration] Помилка створення етапу: {e}")
        return False

def setup_ticker_batch_integration():
    """
    Налаштовує інтеграцію пакетного тренування
    """
    logger.info("[TickerBatchIntegration] Налаштування пакетного тренування...")
    
    success = True
    success &= integrate_ticker_batch_training()
    success &= create_ticker_batch_stage()
    
    if success:
        logger.info("[TickerBatchIntegration] [OK] Всі інтеграції успішні")
    else:
        logger.warning("[TickerBatchIntegration] [WARN] Деякі інтеграції не вдалося")
    
    return success

def get_ticker_batch_status():
    """
    Повертає статус пакетного тренування
    """
    try:
        from utils.ticker_batch_trainer import ticker_batch_trainer
        
        # Перевіряємо наявність результатів
        results_dir = ticker_batch_trainer.results_dir
        result_files = list(results_dir.glob("ticker_batch_results_*.json"))
        
        status = {
            'results_count': len(result_files),
            'latest_result': None,
            'results_dir': str(results_dir)
        }
        
        if result_files:
            latest_file = max(result_files, key=lambda x: x.stat().st_mtime)
            status['latest_result'] = str(latest_file)
            
            # Завантажуємо останні результати
            import json
            with open(latest_file, 'r') as f:
                results = json.load(f)
            
            status['summary'] = results.get('summary', {})
            status['ticker_count'] = len(results.get('ticker_results', {}))
        
        return status
        
    except Exception as e:
        logger.error(f"[TickerBatchIntegration] Помилка отримання статусу: {e}")
        return {}

# Автоматична інтеграція
if __name__ != "__main__":
    setup_ticker_batch_integration()
