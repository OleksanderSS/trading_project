"""
Real-Time Learning Loop
Автоматичне навчання та адаптація системи на основі результатів торгів
"""

from datetime import datetime, timedelta
from typing import Any
import asyncio

import numpy as np
import pandas as pd

from src.core.logging.logger import ProjectLogger

logger = ProjectLogger.get_logger(__name__)

class RealTimeLearning:
    """
    Система реального часу для автоматичного навчання та адаптації
    на основі результатів торгів та ринкових умов
    """

    def __init__(self, config_manager):
        self.config_manager = config_manager
        self.logger = logger

        # Історія торгів та навчання
        self.trade_history = []
        self.performance_history = []
        self.model_performance = {}

        # Моделі для адаптації
        self.strategy_adapter = StrategyAdaptator()
        self.performance_tracker = PerformanceTracker()
        self.meta_learner = MetaLearner()

        # Параметри навчання
        self.min_trades_for_learning = 50
        self.learning_frequency = 100  # Кожні 100 торгів
        self.adaptation_threshold = 0.05
        self.is_learning = False  # Флаг стану

    async def update_and_adapt(self, new_trade_results: list[dict[str, Any]]) -> dict[str, Any]:
        """
        Асинхронно оновлює дані та адаптує стратегії на основі результатів торгів.

        Args:
            new_trade_results: Нові результати торгів

        Returns:
            Звіт про адаптацію та рекомендації
        """
        self.logger.info(f"🔄 Processing {len(new_trade_results)} new trade results...")

        # 1. Оновлення історії
        self._update_trade_history(new_trade_results)

        # 2. Аналіз продуктивності
        performance_analysis = self._analyze_performance()

        # 3. Запуск мета-навчання асинхронно, якщо не запущено
        if not self.is_learning and len(self.trade_history) >= self.min_trades_for_learning:
            self.is_learning = True
            asyncio.create_task(self._run_meta_learning_async())

        # 4. Виявлення проблемних моделей
        problematic_models = self._identify_problematic_models(performance_analysis)

        # 5. Адаптація стратегій
        adaptation_results = {}
        if len(self.trade_history) >= self.min_trades_for_learning:
            adaptation_results = self._perform_adaptation(problematic_models, performance_analysis)

        # 6. Оновлення метаданих
        self._update_metadata(performance_analysis, adaptation_results)

        return {
            'performance_analysis': performance_analysis,
            'problematic_models': problematic_models,
            'adaptation_results': adaptation_results,
            'recommendations': self._generate_recommendations(performance_analysis),
            'next_adaptation_due': self._calculate_next_adaptation(),
            'status': 'adaptation_in_progress' if self.is_learning else 'monitoring'
        }

    async def _run_meta_learning_async(self) -> None:
        """Асинхронний процес оновлення мета-моделі."""
        try:
            self.meta_learner.update_meta_model(self.trade_history)
            self.logger.info("🧠 Meta-model updated asynchronously.")
        except Exception as e:
            self.logger.error(f"Error in async meta-learning: {e}")
        finally:
            self.is_learning = False

    def _perform_adaptation(self, problematic_models: list[dict[str, Any]], performance_analysis: dict[str, Any]) -> dict[str, Any]:
        """Виконує адаптацію стратегій та ризиків."""
        self.logger.info("🔄 Performing real-time strategy adaptation...")
        
        weight_adjustments = self.strategy_adapter.adapt_model_weights(problematic_models, performance_analysis)
        risk_adjustments = self.strategy_adapter.adapt_risk_parameters(performance_analysis)
        strategy_adjustments = self.strategy_adapter.adapt_entry_exit_strategies(performance_analysis)
        
        return {
            'weight_adjustments': weight_adjustments,
            'risk_adjustments': risk_adjustments,
            'strategy_adjustments': strategy_adjustments,
            'timestamp': datetime.now()
        }

    def _update_trade_history(self, new_trade_results: list[dict[str, Any]]) -> None:
        """Оновлює історію торгів"""
        for trade in new_trade_results:
            # Валідація та стандартизація даних
            standardized_trade = self._standardize_trade_data(trade)
            self.trade_history.append(standardized_trade)

            # Оновлення продуктивності моделей
            if 'model_name' in trade and 'profit_loss' in trade:
                model_name = trade['model_name']
                if model_name not in self.model_performance:
                    self.model_performance[model_name] = []
                self.model_performance[model_name].append(trade['profit_loss'])

        # Обмеження історії
        if len(self.trade_history) > 1000:
            self.trade_history = self.trade_history[-1000:]

        self.logger.info(f"📊 Trade history updated: {len(self.trade_history)} total trades")

    def _standardize_trade_data(self, trade: dict[str, Any]) -> dict[str, Any]:
        """Стандартизує дані торгів"""
        standardized = {
            'timestamp': trade.get('timestamp', datetime.now()),
            'ticker': trade.get('ticker', 'unknown'),
            'signal': trade.get('signal', 0),
            'profit_loss': trade.get('profit_loss', 0),
            'model_name': trade.get('model_name', 'unknown'),
            'confidence': trade.get('confidence', 0.5),
            'market_volatility': trade.get('market_volatility', 0.02),
            'position_size': trade.get('position_size', 0.02),
            'exit_reason': trade.get('exit_reason', 'unknown'),
            'market_regime': trade.get('market_regime', 'neutral')
        }

        # Додаткові метрики
        if 'entry_price' in trade and 'exit_price' in trade:
            standardized['return'] = (trade['exit_price'] - trade['entry_price']) / trade['entry_price']

        return standardized

    def _analyze_performance(self) -> dict[str, Any]:
        """Аналізує продуктивність системи"""
        if len(self.trade_history) < 10:
            return {'status': 'insufficient_data'}

        # Конвертація в DataFrame
        df = pd.DataFrame(self.trade_history)

        # Загальні метрики
        total_trades = len(df)
        winning_trades = len(df[df['profit_loss'] > 0])
        win_rate = winning_trades / total_trades

        # Фінансові метрики
        total_pnl = df['profit_loss'].sum()
        avg_pnl = df['profit_loss'].mean()
        max_drawdown = self._calculate_max_drawdown(df['profit_loss'])

        # Метрики по моделях
        model_performance = {}
        for model_name, trades in df.groupby('model_name'):
            model_performance[model_name] = {
                'trades': len(trades),
                'win_rate': len(trades[trades['profit_loss'] > 0]) / len(trades),
                'avg_pnl': trades['profit_loss'].mean(),
                'sharpe_ratio': self._calculate_sharpe_ratio(trades['profit_loss'])
            }

        # Метрики по ринкових умовах
        regime_performance = {}
        for regime, trades in df.groupby('market_regime'):
            regime_performance[regime] = {
                'trades': len(trades),
                'win_rate': len(trades[trades['profit_loss'] > 0]) / len(trades),
                'avg_pnl': trades['profit_loss'].mean()
            }

        analysis = {
            'total_trades': total_trades,
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'avg_pnl': avg_pnl,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': self._calculate_sharpe_ratio(df['profit_loss']),
            'model_performance': model_performance,
            'regime_performance': regime_performance,
            'performance_trend': self._calculate_performance_trend(df),
            'last_update': datetime.now()
        }

        self.logger.info(f"📈 Performance: Win Rate {win_rate:.2%}, Total PnL {total_pnl:.2f}")
        return analysis

    def _identify_problematic_models(self, performance_analysis: dict[str, Any]) -> list[dict[str, Any]]:
        """Виявляє проблемні моделі"""
        problematic: list[dict[str, Any]] = []

        if 'model_performance' not in performance_analysis:
            return problematic

        for model_name, metrics in performance_analysis['model_performance'].items():
            # Критерії проблемної моделі
            issues = []

            if metrics['win_rate'] < 0.45:
                issues.append('low_win_rate')

            if metrics['avg_pnl'] < -0.01:
                issues.append('negative_avg_pnl')

            if metrics['sharpe_ratio'] < 0.5:
                issues.append('low_sharpe_ratio')

            if metrics['trades'] < 10:
                issues.append('insufficient_trades')

            if issues:
                problematic.append({
                    'model': model_name,
                    'issues': issues,
                    'metrics': metrics
                })

        self.logger.info(f"⚠️ Found {len(problematic)} problematic models")
        return problematic



    def _generate_recommendations(self, performance_analysis: dict[str, Any]) -> list[str]:
        """Генерує рекомендації на основі аналізу"""
        recommendations = []

        if performance_analysis.get('win_rate', 0) < 0.5:
            recommendations.append("Consider reducing position sizes due to low win rate")

        if performance_analysis.get('sharpe_ratio', 0) < 1.0:
            recommendations.append("Improve risk management to increase Sharpe ratio")

        if performance_analysis.get('max_drawdown', 0) > 0.1:
            recommendations.append("Implement stricter stop-loss to reduce drawdown")

        # Рекомендації по моделях
        if 'model_performance' in performance_analysis:
            for model_name, metrics in performance_analysis['model_performance'].items():
                if metrics['win_rate'] < 0.4:
                    recommendations.append(f"Consider retraining or replacing {model_name} model")

        return recommendations

    def _calculate_next_adaptation(self) -> datetime:
        """Розраховує час наступної адаптації"""
        trades_until_adaptation = self.learning_frequency - (len(self.trade_history) % self.learning_frequency)

        # Приблизний час наступної адаптації
        next_adaptation = datetime.now() + timedelta(hours=trades_until_adaptation * 0.5)

        return next_adaptation

    def _calculate_max_drawdown(self, pnl_series: pd.Series) -> float:
        """Розраховує максимальний drawdown"""
        if len(pnl_series) < 2:
            return 0.0

        cumulative = pnl_series.cumsum()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max

        return float(drawdown.min())

    def _calculate_sharpe_ratio(self, pnl_series: pd.Series) -> float:
        """Розраховує Sharpe ratio (per trade)"""
        if len(pnl_series) < 2:
            return 0.0

        returns = pnl_series.dropna()
        if len(returns) < 2:
            return 0.0

        std = returns.std()
        if std == 0:
            return 0.0

        return float(returns.mean() / std)

    def _calculate_performance_trend(self, df: pd.DataFrame) -> str:
        """Розраховує тренд продуктивності"""
        if len(df) < 50:
            return 'insufficient_data'

        # Порівняємо останні 25% торгів з попередніми 25%
        split_point = int(len(df) * 0.75)

        recent_trades = df.iloc[split_point:]
        older_trades = df.iloc[:split_point]

        recent_win_rate = len(recent_trades[recent_trades['profit_loss'] > 0]) / len(recent_trades)
        older_win_rate = len(older_trades[older_trades['profit_loss'] > 0]) / len(older_trades)

        if recent_win_rate > older_win_rate * 1.1:
            return 'improving'
        elif recent_win_rate < older_win_rate * 0.9:
            return 'declining'
        else:
            return 'stable'

    def _update_metadata(self, performance_analysis: dict[str, Any], adaptation_results: dict[str, Any]) -> None:
        """Оновлює метадані системи"""
        # Оновлення продуктивності
        self.performance_history.append({
            'timestamp': datetime.now(),
            'performance': performance_analysis,
            'adaptations': adaptation_results
        })

        # Обмеження історії
        if len(self.performance_history) > 100:
            self.performance_history = self.performance_history[-100:]

    def get_learning_status(self) -> dict[str, Any]:
        """Повертає статус системи навчання"""
        return {
            'total_trades': len(self.trade_history),
            'models_tracked': len(self.model_performance),
            'last_adaptation': self.performance_history[-1]['timestamp'] if self.performance_history else None,
            'next_adaptation_due': self._calculate_next_adaptation(),
            'learning_ready': len(self.trade_history) >= self.min_trades_for_learning,
            'performance_trend': self._calculate_performance_trend(pd.DataFrame(self.trade_history)) if self.trade_history else 'no_data'
        }


class StrategyAdaptator:
    """Адаптор стратегій для динамічного налаштування"""

    def __init__(self):
        self.logger = ProjectLogger.get_logger(__name__)

    def adapt_model_weights(self, problematic_models: list[dict], performance_analysis: dict[str, Any]) -> dict[str, Any]:
        """Адаптує ваги моделей"""
        self.logger.info("🔄 Adapting model weights...")

        adapted_weights = {}

        for model_info in problematic_models:
            model_name = model_info['model']
            issues = model_info['issues']

            # Зменшення ваги для проблемних моделей
            weight_reduction = 0.5
            for issue in issues:
                if issue == 'low_win_rate':
                    weight_reduction *= 0.7
                elif issue == 'negative_avg_pnl':
                    weight_reduction *= 0.5
                elif issue == 'low_sharpe_ratio':
                    weight_reduction *= 0.8

            adapted_weights[model_name] = weight_reduction

        return {
            'adapted_weights': adapted_weights,
            'models_affected': len(adapted_weights)
        }

    def adapt_risk_parameters(self, performance_analysis: dict[str, Any]) -> dict[str, Any]:
        """Адаптує параметри ризику"""
        self.logger.info("🛡️ Adapting risk parameters...")

        win_rate = performance_analysis.get('win_rate', 0.5)
        max_drawdown = performance_analysis.get('max_drawdown', 0.05)

        # Адаптація на основі продуктивності
        if win_rate < 0.45:
            risk_adjustment = 0.7  # Зменшити ризик
        elif win_rate > 0.6:
            risk_adjustment = 1.2  # Збільшити ризик
        else:
            risk_adjustment = 1.0

        if max_drawdown > 0.1:
            risk_adjustment *= 0.8  # Додаткове зменшення ризику

        return {
            'risk_adjustment': risk_adjustment,
            'base_risk_level': 0.02 * risk_adjustment,
            'max_position_size': 0.05 * risk_adjustment
        }

    def adapt_entry_exit_strategies(self, performance_analysis: dict[str, Any]) -> dict[str, Any]:
        """Адаптує стратегії входу/виходу"""
        self.logger.info("🎯 Adapting entry/exit strategies...")

        win_rate = performance_analysis.get('win_rate', 0.5)

        # Адаптація стоп-лосс та тейк-профіт
        if win_rate < 0.45:
            # Зменшуємо стоп-лосс для швидшого виходу
            stop_loss_adjustment = 0.8
            take_profit_adjustment = 0.9
        elif win_rate > 0.6:
            # Збільшуємо тейк-профіт для більшого прибутку
            stop_loss_adjustment = 1.1
            take_profit_adjustment = 1.3
        else:
            stop_loss_adjustment = 1.0
            take_profit_adjustment = 1.0

        return {
            'stop_loss_adjustment': stop_loss_adjustment,
            'take_profit_adjustment': take_profit_adjustment
        }


class PerformanceTracker:
    """Трекер продуктивності для аналізу тенденцій"""

    def __init__(self):
        self.logger = ProjectLogger.get_logger(__name__)

    def track_performance(self, trade_results: list[dict]) -> dict[str, Any]:
        """Відстежує продуктивність"""
        if not trade_results:
            return {'status': 'no_data'}

        df = pd.DataFrame(trade_results)

        # Розрахунок метрик
        metrics = {
            'total_trades': len(df),
            'win_rate': len(df[df['profit_loss'] > 0]) / len(df),
            'avg_pnl': df['profit_loss'].mean(),
            'volatility': df['profit_loss'].std()
        }

        return metrics


class MetaLearner:
    """Мета-навчання для покращення стратегій"""

    def __init__(self):
        self.logger = ProjectLogger.get_logger(__name__)
        self.meta_model = None

    def update_meta_model(self, trade_history: list[dict]) -> dict[str, Any]:
        """Оновлює мета-модель"""
        self.logger.info("🧠 Updating meta-model...")

        if len(trade_history) < 100:
            return {'status': 'insufficient_data'}

        # Тут має бути реальна логіка мета-навчання
        # Для прикладу використовуємо просту модель

        df = pd.DataFrame(trade_history)

        # Підготовка фіч для мета-моделі
        features = self._prepare_meta_features(df)

        # Тренування мета-моделі
        from sklearn.ensemble import RandomForestClassifier

        X = features.drop(['profit_loss_positive'], axis=1)
        y = features['profit_loss_positive']

        self.meta_model = RandomForestClassifier(n_estimators=50, random_state=42)
        
        # ✅ Recency weighting: exponential weight decay based on trade sequence (half-life of 100 trades)
        total_samples = len(df)
        trade_indices = np.arange(total_samples)
        half_life_trades = 100
        sample_weights = 2.0 ** (-(total_samples - 1 - trade_indices) / half_life_trades)
        
        self.meta_model.fit(X, y, sample_weight=sample_weights)

        return {
            'status': 'updated',
            'accuracy': self.meta_model.score(X, y),
            'feature_importance': dict(zip(X.columns, self.meta_model.feature_importances_, strict=False))
        }

    def _prepare_meta_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Готує фічі для мета-моделі"""
        features = pd.DataFrame()

        # Базові фічі
        features['signal_strength'] = df['signal'].abs()
        features['confidence'] = df['confidence']
        features['market_volatility'] = df['market_volatility']
        features['position_size'] = df['position_size']

        # Цільова змінна
        features['profit_loss_positive'] = (df['profit_loss'] > 0).astype(int)

        return features
