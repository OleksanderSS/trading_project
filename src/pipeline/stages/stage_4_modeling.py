# audit-ignore: ARCHITECTURAL_USAGE
import datetime
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from src.analytics.analyzers.model_comparison_analyzer import ModelComparisonAnalyzer
from src.config.unified_config_manager import UnifiedConfigManager
from src.core.logging.logger import ProjectLogger
from src.models.adapters.data_preparation import prepare_data_for_models
from src.pipeline.stages.base_stage import BaseStage
from src.training.constants import (
    BATCH_TRAINER_DEFAULT_BATCH_SIZE,
    BATCH_TRAINER_DEFAULT_MAX_MEMORY_GB,
    DEFAULT_TEST_SIZE,
)
from src.training.unified_training_manager import TrainerConfig, TrainingStrategy, UnifiedTrainingManager

logger = ProjectLogger.get_logger('ModelingStage')


@dataclass
class TargetProcessingConfig:
    """Configuration for target processing."""
    ticker: str
    df: Any
    target_name: str
    timeframe: str
    champions: dict[str, Any]


class ModelingStage(BaseStage):
    """
    Stage 4: Advanced ML Arena with Pattern-Based Champions.

    🎯 REGIME-SPECIFIC CHAMPIONS:
    - Тренує та зберігає найкращі моделі для кожної пари (Ticker, Context Pattern).
    - Використовує Purged Validation для чесного оцінювання.
    """

    def __init__(self, config_manager: UnifiedConfigManager, brain: dict[str, Any], **kwargs):
        super().__init__(config_manager, brain)
        self.modeling_config = self.config_manager.get_config('modeling') or {}
        self.system_config = self.config_manager.get_config('system') or {}

        strategy_str = self.modeling_config.get('strategy', 'hybrid').upper()
        strategy = (TrainingStrategy[strategy_str] if strategy_str in
            TrainingStrategy.__members__ else TrainingStrategy.HYBRID)

        training_config = TrainerConfig(
            strategy=strategy,
            batch_size=self.modeling_config.get('batch_size', BATCH_TRAINER_DEFAULT_BATCH_SIZE),
            max_memory_gb=self.modeling_config.get('max_memory_gb', BATCH_TRAINER_DEFAULT_MAX_MEMORY_GB)
        )

        self.training_manager = UnifiedTrainingManager(training_config)
        self.comparison_analyzer = ModelComparisonAnalyzer()
        self.models_dir = self.config_manager.get_models_path()
        self.diary_path = Path(self.system_config.get('diary_path', 'logs/experience_diary.csv'))
        self._init_infrastructure()

    def _init_infrastructure(self):
        """Initializes the environment."""
        self.models_dir.mkdir(parents=True, exist_ok=True)
        if not self.diary_path.exists():
            self.diary_path.parent.mkdir(parents=True, exist_ok=True)
            columns = ['timestamp', 'ticker', 'tf', 'target', 'pattern_id', 'model_name', 'score', 'is_champion']
            pd.DataFrame(columns=columns).to_csv(self.diary_path, index=False)

    async def run(self, **kwargs) -> dict[str, Any]:
        """Runs the full training cycle with Pattern-Aware logic."""
        enriched_data = kwargs.get('enriched_data')
        if enriched_data is None or (isinstance(enriched_data, pd.DataFrame) and enriched_data.empty):
            logger.error('Enriched data not found. Skipping Modeling Stage.')
            return {}

        champions = {}
        logger.info('--- [Modeling Stage] Starting Regime-Aware Training Arena ---')

        # Обробка груп тікерів
        ticker_groups = enriched_data.groupby('ticker') if isinstance(enriched_data, pd.DataFrame) else enriched_data.items()

        for ticker, df in ticker_groups:
            # ✅ ELITE FIX: Визначаємо домінуючий патерн для цього тікера у вибірці
            current_pattern = df['context_pattern_id'].iloc[-1] if 'context_pattern_id' in df.columns else 'normal'
            logger.info(f"📍 Ticker {ticker} is currently in pattern: {current_pattern}")

            await self._process_ticker_with_async(ticker, df, champions, current_pattern)

        logger.info(f'Modeling Stage complete. Trained {len(champions)} expert models.')
        return {'models_metadata': champions, 'processed_data': enriched_data}

    async def _process_ticker_with_async(self, ticker, df, champions, current_pattern):
        """Process data for a single ticker."""
        try:
            target_cols = [c for c in df.columns if c.startswith('target_')]
            timeframe = df['interval'].iloc[-1] if 'interval' in df.columns else '1d'

            for target_name in target_cols:
                # Готуємо дані з PURGED GAP
                prepared_data = prepare_data_for_models(
                    df=df, ticker=ticker, timeframe=timeframe,
                    target_cols=[target_name],
                    gap_size=10, # Обов'язковий розрив для чесності
                    test_size=self.modeling_config.get('test_size', DEFAULT_TEST_SIZE)
                )

                if not prepared_data:
                    continue

                # Запускаємо уніфіковане тренування
                training_results = self.training_manager.execute_unified_training(
                    tickers=[ticker], data_context=prepared_data
                )

                # Вибираємо переможця для конкретного ПАТЕРНА
                ticker_result = training_results.get('tickers_results', {}).get(ticker, {})
                if ticker_result.get('status') == 'success':
                    winner_name = ticker_result.get('winner')
                    metrics = ticker_result.get('winner_metrics', {})

                    context_key = f"{ticker}_{target_name}_{current_pattern}"
                    champions[context_key] = {
                        'ticker': ticker,
                        'target': target_name,
                        'pattern_id': current_pattern,
                        'winner': winner_name,
                        'metrics': metrics,
                        'model_path': ticker_result.get('model_path'),
                        'timestamp': datetime.datetime.now().isoformat()
                    }

                    self._log_expert_to_diary(champions[context_key], timeframe)
                    logger.info(f"🏆 Pattern Champion for {context_key}: {winner_name} (Score: {metrics.get('score', 0):.4f})")

        except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError) as e:
            logger.exception(f"Error modeling {ticker}: {e}")

    def _log_expert_to_diary(self, info: dict[str, Any], tf: str):
        """Зберігає інформацію про експертну модель у щоденник досвіду."""
        entry = {
            'timestamp': info['timestamp'], 'ticker': info['ticker'],
            'tf': tf, 'target': info['target'], 'pattern_id': info['pattern_id'],
            'model_name': info['winner'], 'score': info['metrics'].get('score', 0),
            'is_champion': True
        }
        pd.DataFrame([entry]).to_csv(self.diary_path, mode='a', header=False, index=False)
