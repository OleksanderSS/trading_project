"""
Unified Training Manager for Large Ticker Sets
"""
from __future__ import annotations

import json
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any

from src.config.unified_config_manager import get_current_config
from src.core.logging.logger import ProjectLogger

logger = ProjectLogger.get_logger('UnifiedTrainingManager')
if TYPE_CHECKING:
    from src.training.base_trainer import TrainerConfig


def __getattr__(name: str) -> Any:
    if name == 'TrainerConfig':
        from src.training.base_trainer import TrainerConfig
        return TrainerConfig
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


class TrainingStrategy(Enum):
    BATCH = 'batch'
    PROGRESSIVE = 'progressive'
    HYBRID = 'hybrid'


class UnifiedTrainingManager:
    """
    Unified manager for coordinating different training strategies (Batch, Progressive, Hybrid).
    Acts as a high-level orchestrator for the Modeling Stage.
    """

    def __init__(self, config: TrainerConfig | None = None):
        if config is None:
            from src.training.base_trainer import TrainerConfig
            config = TrainerConfig()
        self.config = config
        self.config_manager = get_current_config()
        self.logger = logger
        system_config = self.config_manager.get_config('system', {})
        self.base_dir = Path(system_config.get('unified_models_path',
            'models/unified'))
        self.plans_dir = self.base_dir / 'plans'
        self.results_dir = self.base_dir / 'results'
        self.checkpoints_dir = self.base_dir / 'checkpoints'
        for dir_path in [self.base_dir, self.plans_dir, self.results_dir,
            self.checkpoints_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        self.trainers: dict[str, Any] = {}
        from src.analytics.arena.arena_battle import get_trading_arena
        from src.analytics.context.contextual_model_selector import ContextualModelSelector
        all_models = self._get_available_model_names()
        self.context_selector = ContextualModelSelector(available_models=
            all_models)
        self.arena = get_trading_arena()
        self._initialize_trainers()

    def _initialize_trainers(self):
        """Initialize all available trainer implementations."""
        from src.training.base_trainer import TrainerConfig
        from src.training.batch_trainer import BatchTrainer
        from src.training.progressive_trainer import ProgressiveTrainer
        batch_config = TrainerConfig(batch_size=self.config.batch_size,
            max_memory_gb=self.config.max_memory_gb)
        self.trainers[TrainingStrategy.BATCH.value] = BatchTrainer(batch_config
            )
        progressive_config = TrainerConfig(initial_batch_size=self.config.
            initial_batch_size, max_batch_size=self.config.max_batch_size,
            growth_factor=self.config.growth_factor, min_accuracy_threshold
            =self.config.min_accuracy_threshold, max_loss_threshold=self.
            config.max_loss_threshold, max_time_hours=self.config.
            max_time_hours, enable_adaptive_batching=self.config.
            enable_adaptive_batching)
        self.trainers[TrainingStrategy.PROGRESSIVE.value] = ProgressiveTrainer(
            progressive_config)

    def execute_unified_training(self, tickers: list[str], data_context:
        dict[str, Any]) ->dict[str, Any]:
        """
        Execute the training cycle based on the best strategy for the given ticker set.
        """
        self.logger.info(
            f'🚀 Starting unified training for {len(tickers)} tickers')
        plan = self.create_unified_plan(tickers)
        self.save_unified_plan(plan)
        strategy_val = plan['strategy']
        strategy = TrainingStrategy(strategy_val)
        for ticker in tickers:
            models_to_train = self._select_models_for_ticker(ticker,
                data_context)
            plan['ticker_plans'][ticker] = {'models': models_to_train}
        results: dict[str, Any] = {'strategy': strategy.value,
            'tickers_results': {}, 'training_summary': {}, 'timestamp':
            datetime.now().isoformat()}
        try:
            if strategy == TrainingStrategy.BATCH:
                results.update(self.trainers[strategy.value].
                    execute_batch_training(plan, data_context))
            elif strategy == TrainingStrategy.PROGRESSIVE:
                results.update(self.trainers[strategy.value].
                    execute_progressive_training(plan, data_context))
            elif strategy == TrainingStrategy.HYBRID:
                results.update(self._execute_hybrid_training(plan,
                    data_context))
        except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError) as e:
            self.logger.error(f'❌ Training execution failed: {e}', exc_info
                =True)
            results['status'] = 'failed'
            results['error'] = str(e)
            return results
        if results.get('tickers_results') and 'y_test' in data_context:
            self.logger.info('⚔️ Initiating Arena Battle for benchmarking...')
            try:
                battle_results = self.arena.run_battle(results['tickers_results'], actual_targets=data_context['y_test'])
                results['arena_rankings'] = battle_results
                self.logger.info('✅ Arena Battle completed.')
            except Exception as e:
                self.logger.warning(f'⚠️ Arena battle failed: {e}', exc_info=True)
                results['arena_error'] = str(e)
        self.save_unified_results(results)
        return results

    def _select_models_for_ticker(self, ticker: str, data: (dict[str, Any] |
        None)=None) ->list[str]:
        """Select optimal models for a ticker using contextual analysis."""
        context_fingerprint = data.get('context_fingerprint', 'default'
            ) if data else 'default'

        self.logger.info(f"_select_models_for_ticker called for ticker={ticker}, context_fingerprint={context_fingerprint}")

        # Check if model categories are configured (light/heavy) first
        # This takes precedence over contextual selection
        models_config = self.config_manager.get_config('models', {})
        categories = models_config.get('categories', {})
        self.logger.info(f"Model categories config: {categories}")
        if categories:
            # Use models from configured categories (light or heavy)
            # If 'light' category exists, use only light models
            if 'light' in categories and isinstance(categories['light'], list):
                self.logger.info(f"Using light models from config: {categories['light']}")
                return categories['light']
            # If 'heavy' category exists, use only heavy models
            elif 'heavy' in categories and isinstance(categories['heavy'], list):
                self.logger.info(f"Using heavy models from config: {categories['heavy']}")
                return categories['heavy']
            # Otherwise, use all models from all categories
            else:
                all_category_models = []
                for category_name, models in categories.items():
                    if isinstance(models, list):
                        all_category_models.extend(models)
                if all_category_models:
                    self.logger.info(f"Using all models from categories: {all_category_models}")
                    return all_category_models

        # Only use contextual selection if no categories are configured
        try:
            select_models = getattr(self.context_selector, 'select_models',
                None)
            if callable(select_models):
                # Prefer contextual selection when caller provided the required inputs.
                recommended = select_models(ticker, context_fingerprint, data=data)
                if recommended:
                    self.logger.info(f"Using contextual selection for {ticker}: {recommended}")
                    return recommended
        except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError) as e:
            self.logger.warning(
                f'Contextual model selection failed for {ticker}: {e}. Falling back to configured models.'
                )

        enabled_types = models_config.get('enabled_types')
        if enabled_types:
            self.logger.info(f"Using enabled_types from config: {enabled_types}")
            return enabled_types

        available_models = self._get_available_model_names()
        self.logger.info(f"Using all available models: {available_models}")
        return available_models

    def _get_available_model_names(self) ->list[str]:
        from src.factories.model_factory import ModelFactory
        return ModelFactory.get_available_models()

    def create_unified_plan(self, tickers: list[str]) ->dict[str, Any]:
        """Analyze the task and create a training plan."""
        analysis = self._analyze_ticker_set(tickers)
        strategy_str = self.config.strategy if hasattr(self.config, 'strategy'
            ) and self.config.strategy else analysis['recommended_strategy']
        if isinstance(strategy_str, TrainingStrategy):
            strategy_str = strategy_str.value
        strategy = TrainingStrategy(strategy_str)
        if strategy == TrainingStrategy.BATCH:
            plan = self.trainers[strategy.value]._prepare_ticker_groups({
                'tickers': tickers})
            plan = {'strategy': 'batch', 'tickers': tickers, 'groups': plan}
        elif strategy == TrainingStrategy.PROGRESSIVE:
            plan = self._create_progressive_plan(tickers)
        else:
            plan = {'strategy': strategy.value, 'tickers': tickers}
        plan.update({'analysis': analysis, 'strategy': strategy.value,
            'timestamp': datetime.now().isoformat(), 'ticker_plans': {}})
        return plan

    def _analyze_ticker_set(self, tickers: list[str]) ->dict[str, Any]:
        """Analyze the ticker set to recommend the best strategy."""
        count = len(tickers)
        recommended = 'progressive' if count > 5 else 'batch'
        return {'ticker_count': count, 'recommended_strategy': recommended,
            'complexity_estimate': 'high' if count > 20 else 'medium'}

    def _create_progressive_plan(self, tickers: list[str]) ->dict[str, Any]:
        trainer = self.trainers[TrainingStrategy.PROGRESSIVE.value]
        batches = trainer.create_progressive_batches(tickers)
        return {'total_tickers': len(tickers), 'total_batches': len(batches
            ), 'strategy': 'progressive', 'batches': [{'batch_id': i + 1,
            'tickers': b} for i, b in enumerate(batches)]}

    def _create_hybrid_plan(self, tickers: list[str]) ->dict[str, Any]:
        return {'strategy': 'hybrid', 'phases': [], 'tickers': tickers}

    def _execute_hybrid_training(self, plan: dict[str, Any], data_context:
        dict[str, Any]) ->dict[str, Any]:
        """Execute hybrid training. Автоматично вибирає кращу локальну стратегію."""
        num_tickers = len(plan.get('ticker_plans', {}))
        strategy = (TrainingStrategy.BATCH if num_tickers < 50 else
            TrainingStrategy.PROGRESSIVE)
        self.logger.info(
            f'Executing HYBRID training: використовую {strategy.value} для локальної оптимізації.'
            )
        result = self.trainers[strategy.value].execute_batch_training(plan,
            data_context=data_context)
        return dict(result) if result else {}

    def save_unified_plan(self, plan: dict[str, Any]) ->str:
        filepath = (self.plans_dir /
            f"unified_plan_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        with open(filepath, 'w') as f:
            json.dump(plan, f, indent=2, default=str)
        return str(filepath)

    def save_unified_results(self, results: dict[str, Any]) ->str:
        filepath = (self.results_dir /
            f"unified_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        return str(filepath)
