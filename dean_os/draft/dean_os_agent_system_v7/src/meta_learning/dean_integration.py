# src/meta_learning/dean_integration.py - ІНТЕГРАЦІЯ DEAN TRADING MODELS В PIPELINE

from typing import Any

import numpy as np
import pandas as pd

from src.analytics.context.causal_engine import CausalEngine
from src.core.logging.logger import Logger as ProjectLogger
from src.models.model_selector.heavy_light_comparator import HeavyLightModelComparator

from .dean_trading_models import DeanActor, DeanCritic, DeanSimulator

logger = ProjectLogger.get_logger(__name__)

class DeanModelIntegrator:
    """Інтеграція Dean Trading Models в основний pipeline"""

    def __init__(self, config_manager=None):
        self.config_manager = config_manager
        self.actor_model = None
        self.critic_model = None
        self.simulator_model = None
        self.causal_engine = CausalEngine()
        self.model_comparator = HeavyLightModelComparator()
        self.is_initialized = False

    def initialize_models(self):
        """Ініціалізація всіх Dean моделей"""
        try:
            # Create default base model for DeanActor
            from sklearn.ensemble import RandomForestClassifier

            # Simple default model for Actor
            default_base_model = RandomForestClassifier(n_estimators=10, random_state=42)

            # Default rules config for DeanCritic
            default_rules_config = {
                'high_vol_threshold': 0.05,
                'max_position_size': 0.2,
                'min_confidence': 0.6
            }

            # Default similarity finder for DeanSimulator
            default_similarity_finder = self.model_comparator.similarity_finder if hasattr(self.model_comparator, 'similarity_finder') else None
            if default_similarity_finder is None:
                from src.analytics.analyzers.knn_similarity_finder import KnnSimilarityFinder
                default_similarity_finder = KnnSimilarityFinder()

            self.actor_model = DeanActor(default_base_model, "dean_actor_001")
            self.critic_model = DeanCritic(default_rules_config, "dean_critic_001")
            self.simulator_model = DeanSimulator(default_similarity_finder, "dean_simulator_001")

            self.is_initialized = True
            logger.info("[DEAN] Всі моделі (Actor, Critic, Simulator) successfully ініціалізовані")
            return True
        except Exception as e:
            logger.error(f"[DEAN] Помилка ініціалізації моделей: {e}")
            return False

    def create_market_context(self, data: pd.DataFrame, ticker: str, timeframe: str) -> dict[str, Any]:
        """Створення контексту для Dean моделей з data pipeline"""
        try:
            # Отримуємо останні дані
            latest_data = data.iloc[-1] if len(data) > 0 else None
            if latest_data is None:
                return self._create_default_context()

            # Технічні індикатори та базовий контекст
            context = {
                'ticker': ticker,
                'timeframe': timeframe,
                'current_price': latest_data.get('close', 100),
                'trend': self._determine_trend(data),
                'volatility': self._calculate_volatility(data),
                'volume': latest_data.get('volume', 1.0),
                'momentum': self._calculate_momentum(data),
                'support_resistance': self._find_support_resistance(data),
                'market_sentiment': self._get_market_sentiment(data),
                'technical_signals': self._extract_technical_signals(latest_data),
                'risk_metrics': self._calculate_risk_metrics(data)
            }

            # Інтеграція CausalEngine для проекцій майбутніх наслідків
            trigger_event = context['technical_signals'].get('rsi_signal') or context['trend']
            context['causal_projections'] = self.causal_engine.generate_projections(trigger_event)

            return context

        except Exception as e:
            logger.error(f"[DEAN] Помилка створення контексту: {e}")
            return self._create_default_context()

    def _determine_trend(self, data: pd.DataFrame) -> str:
        """Визначення тренду"""
        try:
            if len(data) < 10:
                return "neutral"

            # Проста логіка визначення тренду
            recent_prices = data['close'].tail(10)
            if recent_prices.iloc[-1] > recent_prices.iloc[0] * 1.02:
                return "bullish"
            elif recent_prices.iloc[-1] < recent_prices.iloc[0] * 0.98:
                return "bearish"
            else:
                return "neutral"
        except Exception as e:
            logger.warning(f"Error determining trend: {e}")
            return "neutral"

    def _calculate_volatility(self, data: pd.DataFrame) -> float:
        """Розрахунок волатильності"""
        try:
            if len(data) < 20:
                return 0.02

            returns = data['close'].pct_change(fill_method=None).fillna(0).tail(20)
            return returns.std()
        except Exception as e:
            logger.warning(f"Error calculating volatility: {e}")
            return 0.02

    def _calculate_momentum(self, data: pd.DataFrame) -> float:
        """Розрахунок моментуму"""
        try:
            if len(data) < 10:
                return 0.0

            current_price = data['close'].iloc[-1]
            prev_price = data['close'].iloc[-10]
            return (current_price - prev_price) / prev_price
        except Exception as e:
            logger.warning(f"Error calculating momentum: {e}")
            return 0.0

    def _find_support_resistance(self, data: pd.DataFrame) -> dict[str, float]:
        """Пошук підтримки та опору"""
        try:
            if len(data) < 20:
                return {'support': 95, 'resistance': 105}

            prices = data['close'].tail(20)
            support = prices.min()
            resistance = prices.max()

            return {'support': float(support), 'resistance': float(resistance)}
        except Exception as e:
            logger.warning(f"Error finding support/resistance: {e}")
            return {'support': 95, 'resistance': 105}

    def _get_market_sentiment(self, data: pd.DataFrame) -> str:
        """Отримання сентименту ринку"""
        try:
            # Проста логіка на основі ціни та обсягу
            if len(data) < 5:
                return "neutral"

            recent_change = data['close'].pct_change(fill_method=None).tail(5).mean()
            if recent_change > 0.01:
                return "positive"
            elif recent_change < -0.01:
                return "negative"
            else:
                return "neutral"
        except Exception as e:
            logger.warning(f"Error getting market sentiment: {e}")
            return "neutral"

    def _extract_technical_signals(self, latest_data: pd.Series) -> dict[str, Any]:
        """Вилучення технічних сигналів"""
        try:
            signals = {}

            # RSI сигнал
            if 'rsi' in latest_data:
                rsi = latest_data['rsi']
                if rsi > 70:
                    signals['rsi_signal'] = 'overbought'
                elif rsi < 30:
                    signals['rsi_signal'] = 'oversold'
                else:
                    signals['rsi_signal'] = 'neutral'

            # MACD сигнал
            if 'macd' in latest_data and 'macd_signal' in latest_data:
                macd = latest_data['macd']
                macd_signal = latest_data['macd_signal']
                if macd > macd_signal:
                    signals['macd_signal'] = 'bullish'
                else:
                    signals['macd_signal'] = 'bearish'

            return signals
        except Exception as e:
            logger.warning(f"Error extracting technical signals: {e}")
            return {}

    def _calculate_risk_metrics(self, data: pd.DataFrame) -> dict[str, float]:
        """Розрахунок ризикових метрик"""
        try:
            if len(data) < 20:
                return {'var_95': 0.02, 'max_drawdown': 0.05}

            returns = data['close'].pct_change(fill_method=None).fillna(0).tail(20)

            # Value at Risk (95%)
            var_95 = returns.quantile(0.05)

            # Max Drawdown
            cumulative = (1 + returns).cumprod()
            running_max = cumulative.expanding().max()
            drawdown = (cumulative - running_max) / running_max
            max_drawdown = drawdown.min()

            return {
                'var_95': float(abs(var_95)),
                'max_drawdown': float(abs(max_drawdown))
            }
        except Exception as e:
            logger.warning(f"Error calculating risk metrics: {e}")
            return {'var_95': 0.02, 'max_drawdown': 0.05}

    def _create_default_context(self) -> dict[str, Any]:
        """Створення контексту за замовчуванням"""
        return {
            'ticker': 'UNKNOWN',
            'timeframe': '1d',
            'current_price': 100.0,
            'trend': 'neutral',
            'volatility': 0.02,
            'volume': 1.0,
            'momentum': 0.0,
            'support_resistance': {'support': 95.0, 'resistance': 105.0},
            'market_sentiment': 'neutral',
            'technical_signals': {},
            'risk_metrics': {'var_95': 0.02, 'max_drawdown': 0.05},
            'causal_projections': []
        }

    def generate_trading_signals(self, context: dict[str, Any]) -> dict[str, Any]:
        """Генерація торгових сигналів на основі контексту"""
        try:
            if not self.is_initialized:
                return self._create_default_signals()

            # Використовуємо HeavyLightModelComparator для вибору типу моделі для Актора
            model_type_recommendation = self.model_comparator.recommend_model_type(context)
            logger.info(f"[DEAN] Рекомендований тип моделі для виконання: {model_type_recommendation}")

            # Використовуємо Actor модель для генерації сигналів
            if self.actor_model:
                action_data = self.actor_model.decide_action(context, model_type=model_type_recommendation)

                # Критик оцінює запропоновану дію
                critique = self.critic_model.critique_action(action_data, context)

                # Симулятор прогнозує результат
                simulation = self.simulator_model.simulate(action_data, context)

                return {
                    'action': action_data.get('type', 'HOLD'),
                    'confidence': action_data.get('confidence', 0.5),
                    'critique_score': critique.get('score', 0),
                    'projected_outcome': simulation.get('summary'),
                    'position_size': self._calculate_position_size(context, action_data.get('confidence', 0.5)),
                    'stop_loss': self._calculate_stop_loss(context),
                    'take_profit': self._calculate_take_profit(context)
                }
            else:
                return self._create_default_signals()

        except Exception as e:
            logger.error(f"[DEAN] Помилка генерації сигналів: {e}")
            return self._create_default_signals()

    def _calculate_position_size(self, context: dict[str, Any], confidence: float) -> float:
        """Розрахунок розміру позиції"""
        try:
            base_size = 0.1  # 10% базова позиція
            volatility_adjustment = min(1.0, 0.02 / max(context.get('volatility', 0.02), 0.001))

            return base_size * confidence * volatility_adjustment
        except Exception as e:
            logger.warning(f"Error calculating position size: {e}")
            return 0.1

    def _calculate_stop_loss(self, context: dict[str, Any]) -> float:
        """Розрахунок stop loss"""
        try:
            current_price = context.get('current_price', 100)
            volatility = context.get('volatility', 0.02)

            # 2% від ціни або 2x волатильність
            stop_loss_pct = max(0.02, volatility * 2)
            return current_price * (1 - stop_loss_pct)
        except Exception as e:
            logger.warning(f"Error calculating stop loss: {e}")
            return 98.0

    def _calculate_take_profit(self, context: dict[str, Any]) -> float:
        """Розрахунок take profit"""
        try:
            current_price = context.get('current_price', 100)
            volatility = context.get('volatility', 0.02)

            # 3% від ціни або 3x волатильність
            take_profit_pct = max(0.03, volatility * 3)
            return current_price * (1 + take_profit_pct)
        except Exception as e:
            logger.warning(f"Error calculating take profit: {e}")
            return 103.0

    def _create_default_signals(self) -> dict[str, Any]:
        """Створення сигналів за замовчуванням"""
        return {
            'action': 'HOLD',
            'confidence': 0.5,
            'position_size': 0.1,
            'stop_loss': 98.0,
            'take_profit': 103.0
        }

    def evaluate_performance(self, trades: list[dict[str, Any]]) -> dict[str, float]:
        """Оцінка продуктивності торгівлі"""
        try:
            if not trades:
                return {
                    'total_trades': 0,
                    'win_rate': 0.0,
                    'profit_factor': 0.0,
                    'sharpe_ratio': 0.0,
                    'max_drawdown': 0.0
                }

            total_trades = len(trades)
            winning_trades = sum(1 for trade in trades if trade.get('pnl', 0) > 0)
            win_rate = winning_trades / total_trades if total_trades > 0 else 0.0

            total_profit = sum(trade.get('pnl', 0) for trade in trades if trade.get('pnl', 0) > 0)
            total_loss = abs(sum(trade.get('pnl', 0) for trade in trades if trade.get('pnl', 0) < 0))
            profit_factor = total_profit / total_loss if total_loss > 0 else 0.0

            # Уніфікований Sharpe ratio (анналізований, risk_free_rate = 0.0 для trade PnL)
            returns = [trade.get('pnl', 0) for trade in trades]
            if returns:
                std_ret = np.std(returns)
                sharpe_ratio = (np.mean(returns) / std_ret * np.sqrt(252)) if std_ret > 1e-6 else 0.0
            else:
                sharpe_ratio = 0.0

            return {
                'total_trades': total_trades,
                'win_rate': win_rate,
                'profit_factor': profit_factor,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': 0.0  # Спрощено
            }

        except Exception as e:
            logger.error(f"[DEAN] Помилка оцінки продуктивності: {e}")
            return {
                'total_trades': 0,
                'win_rate': 0.0,
                'profit_factor': 0.0,
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.0
            }


# Alias for backward compatibility
DeanIntegration = DeanModelIntegrator
