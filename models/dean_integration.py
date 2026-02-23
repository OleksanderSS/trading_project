# models/dean_integration.py - ІНТЕГРАЦІЯ DEAN TRADING MODELS В PIPELINE

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
from utils.logger import ProjectLogger
from models.dean_trading_models import DeanActorModel, DeanCriticModel, DeanAdversaryModel, DeanSimulatorModel

logger = ProjectLogger.get_logger(__name__)

class DeanModelIntegrator:
    """Інтеграція Dean Trading Models в основний pipeline"""
    
    def __init__(self):
        self.actor_model = None
        self.critic_model = None
        self.adversary_model = None
        self.simulator_model = None
        self.is_initialized = False
        
    def initialize_models(self):
        """Ініціалізація всіх Dean моделей"""
        try:
            self.actor_model = DeanActorModel("dean_actor_001")
            self.critic_model = DeanCriticModel("dean_critic_001")
            self.adversary_model = DeanAdversaryModel("dean_adversary_001")
            self.simulator_model = DeanSimulatorModel("dean_simulator_001")
            
            self.is_initialized = True
            logger.info("[DEAN] Всі моделі successfully ініціалізовані")
            return True
        except Exception as e:
            logger.error(f"[DEAN] Помилка ініціалізації моделей: {e}")
            return False
    
    def create_market_context(self, data: pd.DataFrame, ticker: str, timeframe: str) -> Dict[str, Any]:
        """Створення контексту для Dean моделей з data pipeline"""
        try:
            # Отримуємо останні дані
            latest_data = data.iloc[-1] if len(data) > 0 else None
            if latest_data is None:
                return self._create_default_context()
            
            # Технічні індикатори
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
        except:
            return "neutral"
    
    def _calculate_volatility(self, data: pd.DataFrame) -> float:
        """Розрахунок волатильності"""
        try:
            if len(data) < 20:
                return 0.02
            
            returns = data['close'].pct_change().tail(20)
            return returns.std()
        except:
            return 0.02
    
    def _calculate_momentum(self, data: pd.DataFrame) -> float:
        """Розрахунок моментуму"""
        try:
            if len(data) < 10:
                return 0.0
            
            current_price = data['close'].iloc[-1]
            prev_price = data['close'].iloc[-10]
            return (current_price - prev_price) / prev_price
        except:
            return 0.0
    
    def _find_support_resistance(self, data: pd.DataFrame) -> Dict[str, float]:
        """Пошук підтримки та опору"""
        try:
            if len(data) < 20:
                return {'support': 95, 'resistance': 105}
            
            prices = data['close'].tail(20)
            support = prices.min()
            resistance = prices.max()
            
            return {'support': float(support), 'resistance': float(resistance)}
        except:
            return {'support': 95, 'resistance': 105}
    
    def _get_market_sentiment(self, data: pd.DataFrame) -> str:
        """Отримання сентименту ринку"""
        try:
            # Проста логіка на основі ціни та обсягу
            if len(data) < 5:
                return "neutral"
            
            recent_change = data['close'].pct_change().tail(5).mean()
            if recent_change > 0.01:
                return "positive"
            elif recent_change < -0.01:
                return "negative"
            else:
                return "neutral"
        except:
            return "neutral"
    
    def _extract_technical_signals(self, latest_data: pd.Series) -> Dict[str, Any]:
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
        except:
            return {}
    
    def _calculate_risk_metrics(self, data: pd.DataFrame) -> Dict[str, float]:
        """Розрахунок ризикових метрик"""
        try:
            if len(data) < 20:
                return {'var_95': 0.02, 'max_drawdown': 0.05}
            
            returns = data['close'].pct_change().tail(20)
            
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
        except:
            return {'var_95': 0.02, 'max_drawdown': 0.05}
    
    def _create_default_context(self) -> Dict[str, Any]:
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
            'risk_metrics': {'var_95': 0.02, 'max_drawdown': 0.05}
        }
    
    def generate_trading_signals(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Генерація торгових сигналів на основі контексту"""
        try:
            if not self.is_initialized:
                return self._create_default_signals()
            
            # Використовуємо Actor модель для генерації сигналів
            if self.actor_model:
                action = self.actor_model.predict_action(context)
                confidence = self.actor_model.get_confidence()
                
                return {
                    'action': action,
                    'confidence': confidence,
                    'position_size': self._calculate_position_size(context, confidence),
                    'stop_loss': self._calculate_stop_loss(context),
                    'take_profit': self._calculate_take_profit(context)
                }
            else:
                return self._create_default_signals()
                
        except Exception as e:
            logger.error(f"[DEAN] Помилка генерації сигналів: {e}")
            return self._create_default_signals()
    
    def _calculate_position_size(self, context: Dict[str, Any], confidence: float) -> float:
        """Розрахунок розміру позиції"""
        try:
            base_size = 0.1  # 10% базова позиція
            volatility_adjustment = min(1.0, 0.02 / max(context.get('volatility', 0.02), 0.001))
            
            return base_size * confidence * volatility_adjustment
        except:
            return 0.1
    
    def _calculate_stop_loss(self, context: Dict[str, Any]) -> float:
        """Розрахунок stop loss"""
        try:
            current_price = context.get('current_price', 100)
            volatility = context.get('volatility', 0.02)
            
            # 2% від ціни або 2x волатильність
            stop_loss_pct = max(0.02, volatility * 2)
            return current_price * (1 - stop_loss_pct)
        except:
            return 98.0
    
    def _calculate_take_profit(self, context: Dict[str, Any]) -> float:
        """Розрахунок take profit"""
        try:
            current_price = context.get('current_price', 100)
            volatility = context.get('volatility', 0.02)
            
            # 3% від ціни або 3x волатильність
            take_profit_pct = max(0.03, volatility * 3)
            return current_price * (1 + take_profit_pct)
        except:
            return 103.0
    
    def _create_default_signals(self) -> Dict[str, Any]:
        """Створення сигналів за замовчуванням"""
        return {
            'action': 'HOLD',
            'confidence': 0.5,
            'position_size': 0.1,
            'stop_loss': 98.0,
            'take_profit': 103.0
        }
    
    def evaluate_performance(self, trades: List[Dict[str, Any]]) -> Dict[str, float]:
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
            
            # Спрощений Sharpe ratio
            returns = [trade.get('pnl', 0) for trade in trades]
            if returns:
                sharpe_ratio = np.mean(returns) / (np.std(returns) + 1e-6)
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
