#!/usr/bin/env python3
"""
Progressive mode - прогресивне навчання з повним pipeline
"""

from .base import BaseMode
from typing import Dict, Any
import sys
from pathlib import Path

# Додаємо шлях до core modules
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from core.training.progressive_trainer import ProgressiveTrainer, ProgressiveConfig
from core.stages.stage_manager import StageManager
from config.tickers import get_tickers, get_ticker_categories


class ProgressiveMode(BaseMode):
    """Режим прогресивного навчання з повним pipeline"""
    
    def run(self) -> Dict[str, Any]:
        """Запуск прогресивного навчання"""
        self.logger.info("Starting progressive mode with full pipeline...")
        
        try:
            # Отримуємо тікери з CLI або з конфігурації
            tickers = getattr(self, 'tickers', None) or getattr(self.config, 'tickers', None)
            if tickers:
                ticker_list = [t.strip() for t in tickers.split(',')] if isinstance(tickers, str) else tickers
            else:
                # Використовуємо всі доступні тікери
                ticker_list = get_tickers()[:50]  # Обмежуємо для тестування
            
            self.logger.info(f"Processing {len(ticker_list)} tickers: {ticker_list[:5]}...")
            
            # Створюємо конфігурацію прогресивного тренування
            config = ProgressiveConfig(
                initial_batch_size=5,
                max_batch_size=15,
                growth_factor=1.3,
                min_accuracy_threshold=0.75,
                enable_adaptive_batching=True,
                enable_quality_filtering=True,
                enable_smart_scheduling=True,
                save_intermediate_results=True
            )
            
            # Ініціалізуємо прогресивний тренер
            trainer = ProgressiveTrainer(config)
            
            # Запускаємо прогресивне тренування
            results = trainer.execute_progressive_training(ticker_list)
            
            return {
                'status': 'success',
                'mode': 'progressive',
                'message': 'Progressive training completed successfully',
                'training_summary': results.get('training_summary', {}),
                'total_tickers_processed': len(ticker_list),
                'pipeline_stages': ['stage1_collection', 'stage2_enrichment', 'stage3_features', 'stage4_modeling'],
                'models_trained': ['lgbm', 'xgboost', 'rf', 'mlp', 'cnn', 'lstm', 'transformer'],
                'features_engineered': [
                    'technical_indicators', 'news_impact', 'sentiment_analysis',
                    'macro_economic', 'multi_timeframe', 'linguistic_dna',
                    'volatility_features', 'momentum_features', 'volume_features'
                ],
                'targets_created': [
                    'price_direction', 'volatility_target', 'regime_target',
                    'momentum_target', 'mean_reversion_target'
                ]
            }
            
        except Exception as e:
            self.logger.error(f"Progressive mode failed: {e}")
            return {
                'status': 'failed',
                'mode': 'progressive',
                'error': str(e),
                'message': 'Progressive training failed'
            }
