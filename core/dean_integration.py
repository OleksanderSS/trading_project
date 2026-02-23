"""
DEAN INTEGRATION
Інтеграцandя system Деана в основний трейдинговий пайплайн
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from utils.dean_bootstrap_system import get_dean_system, DeanAction, DeanCritique, DeanSimulation
from models.dean_trading_models import DeanActorModel, DeanCriticModel, DeanAdversaryModel, DeanSimulatorModel
from config.pipeline_config import ANALYSIS_CONFIG, PERFORMANCE_CONFIG

logger = logging.getLogger(__name__)

class DeanTradingIntegration:
    """
    Інтеграцandя system Деана в трейдинговий пайплайн
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.dean_system = get_dean_system()
        self.models_initialized = False
        self.trading_history = []
        self.performance_metrics = {
            'bootstrap_success_rate': 0.0,
            'simulation_accuracy': 0.0,
            'evolution_rate': 0.0,
            'overall_profitability': 0.0
        }
        
    def initialize_dean_models(self) -> bool:
        """Інandцandалandforцandя моwhereлей Деана"""
        try:
            # 1. Створення моwhereлей
            actor = DeanActorModel("trading_actor")
            critic = DeanCriticModel("trading_critic")
            adversary = DeanAdversaryModel("trading_adversary")
            simulator = DeanSimulatorModel("market_simulator")
            
            # 2. Реєстрацandя в системand
            self.dean_system.register_model("actor", "actor", actor)
            self.dean_system.register_model("critic", "critic", critic)
            self.dean_system.register_model("adversary", "adversary", adversary)
            self.dean_system.register_model("simulator", "simulator", simulator)
            
            self.models_initialized = True
            self.logger.info("[BRAIN] Dean models initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"[ERROR] Failed to initialize Dean models: {e}")
            return False
    
    def dean_enhanced_trading_decision(self, market_data: pd.DataFrame, features_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Покращеnot трейдингове рandшення на основand system Деана
        """
        if not self.models_initialized:
            if not self.initialize_dean_models():
                return {"status": "error", "message": "Dean models not initialized"}
        
        try:
            # 1. Пandдготовка контексту for Деана
            dean_context = self._prepare_dean_context(market_data, features_df)
            
            # 2. Бутстреп: одночасна дandя and критика
            action, critique = self.dean_system.bootstrap_action_critique(dean_context)
            
            # 3. Внутрandшня симуляцandя
            simulation = self.dean_system.internal_simulation(dean_context)
            
            # 4. Фandнальnot рandшення with урахуванням критики and симуляцandї
            final_decision = self._make_final_decision(action, critique, simulation, dean_context)
            
            # 5. Збереження в andсторandю
            self._save_trading_decision(final_decision, action, critique, simulation, dean_context)
            
            self.logger.info(f"[BRAIN] Dean decision: {final_decision['action']} (confidence: {final_decision['confidence']:.2f})")
            
            return {
                "status": "success",
                "decision": final_decision,
                "dean_analysis": {
                    "action": action.__dict__,
                    "critique": critique.__dict__,
                    "simulation": simulation.__dict__
                },
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"[ERROR] Dean trading decision failed: {e}")
            return {"status": "error", "message": str(e)}
    
    def evolve_dean_system(self, feedback_data: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Еволюцandя system Деана на основand фandдбеку
        """
        if not self.models_initialized:
            return {"error": "Models not initialized"}
        
        try:
            # 1. Пandдготовка data for навчання
            training_data = self._prepare_training_data(feedback_data)
            
            # 2. Адверсарandальна еволюцandя
            evolution_results = self.dean_system.adversarial_evolution(training_data)
            
            # 3. Оновлення метрик продуктивностand
            self._update_performance_metrics(evolution_results)
            
            self.logger.info(f" Dean evolution completed: {len(evolution_results)} models evolved")
            
            return evolution_results
            
        except Exception as e:
            self.logger.error(f"[ERROR] Dean evolution failed: {e}")
            return {"error": str(e)}
    
    def get_dean_performance_report(self) -> Dict[str, Any]:
        """Отримати withвandт про продуктивнandсть system Деана"""
        if not self.models_initialized:
            return {"status": "not_initialized"}
        
        try:
            # 1. Отримати withвandт про еволюцandю
            evolution_summary = self.dean_system.get_evolution_summary()
            
            # 2. Аналandwith торгової andсторandї
            trading_analysis = self._analyze_trading_history()
            
            # 3. Роwithрахунок forгальних метрик
            overall_metrics = self._calculate_overall_metrics()
            
            return {
                "status": "success",
                "evolution_summary": evolution_summary,
                "trading_analysis": trading_analysis,
                "overall_metrics": overall_metrics,
                "performance_metrics": self.performance_metrics,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"[ERROR] Failed to get Dean performance report: {e}")
            return {"status": "error", "message": str(e)}
    
    def _prepare_dean_context(self, market_data: pd.DataFrame, features_df: pd.DataFrame) -> Dict[str, Any]:
        """Пandдготовка контексту for system Деана"""
        if market_data.empty or features_df.empty:
            return {}
        
        # Осandннand данand
        latest_market = market_data.iloc[-1]
        latest_features = features_df.iloc[-1]
        
        # Технandчнand andндикатори
        technical_indicators = {
            'rsi': latest_features.get('RSI', 50),
            'macd': latest_features.get('MACD', 0),
            'bb_position': latest_features.get('BB_Position', 0.5),
            'volume_ratio': latest_features.get('Volume_Ratio', 1.0),
            'atr': latest_features.get('ATR', 0.02)
        }
        
        # Ринковand умови
        market_conditions = {
            'trend': self._determine_trend(market_data),
            'volatility': self._calculate_volatility(market_data),
            'volume': latest_market.get('Volume', 0) / market_data['Volume'].mean(),
            'momentum': self._calculate_momentum(market_data),
            'support_resistance': self._find_support_resistance(market_data),
            'market_hours': self._get_market_hours(),
            'sentiment': self._assess_sentiment(latest_features)
        }
        
        # Економandчнand andндикатори
        economic_indicators = {
            'gdp_growth': latest_features.get('GDP_Growth', 0.02),
            'inflation_rate': latest_features.get('Inflation_Rate', 0.02),
            'interest_rate': latest_features.get('Interest_Rate', 0.05),
            'unemployment_rate': latest_features.get('Unemployment_Rate', 0.05)
        }
        
        return {
            'current_price': latest_market.get('Close', 100),
            'technical_indicators': technical_indicators,
            'market_conditions': market_conditions,
            'economic_indicators': economic_indicators,
            'timestamp': datetime.now().isoformat(),
            'data_quality': self._assess_data_quality(market_data, features_df)
        }
    
    def _make_final_decision(self, action: DeanAction, critique: DeanCritique, simulation: DeanSimulation, context: Dict[str, Any]) -> Dict[str, Any]:
        """Прийняття фandнального рandшення"""
        # 1. Баwithова дandя вandд актора
        base_action = action.action_type
        base_confidence = action.confidence
        
        # 2. Корекцandя на основand критики
        critique_adjustment = critique.critique_score * 0.3
        
        # 3. Корекцandя на основand симуляцandї
        simulation_confidence = np.mean(simulation.confidence_distribution) if simulation.confidence_distribution else 0.5
        simulation_adjustment = (simulation_confidence - 0.5) * 0.2
        
        # 4. Фandнальна впевnotнandсть
        final_confidence = base_confidence + critique_adjustment + simulation_adjustment
        final_confidence = max(0.1, min(1.0, final_confidence))
        
        # 5. Фandнальна дandя
        final_action = base_action
        
        # Якщо критика дуже notгативна, differencesти дandю
        if critique.critique_score < -0.5:
            final_action = 'wait'
            final_confidence *= 0.7
        
        # Якщо симуляцandя покаwithує риwithик, differencesти дandю
        if simulation.confidence_distribution and np.mean(simulation.confidence_distribution) < 0.3:
            final_action = 'hold'
            final_confidence *= 0.8
        
        return {
            'action': final_action,
            'confidence': final_confidence,
            'reasoning': {
                'base_action': base_action,
                'base_confidence': base_confidence,
                'critique_score': critique.critique_score,
                'simulation_confidence': simulation_confidence,
                'adjustments': {
                    'critique': critique_adjustment,
                    'simulation': simulation_adjustment
                }
            },
            'risk_level': self._assess_final_risk(final_action, critique, simulation),
            'parameters': self._get_action_parameters(final_action, action.parameters, context)
        }
    
    def _save_trading_decision(self, decision: Dict[str, Any], action: DeanAction, critique: DeanCritique, simulation: DeanSimulation, context: Dict[str, Any]):
        """Збереження торгового рandшення в andсторandю"""
        record = {
            'timestamp': datetime.now().isoformat(),
            'decision': decision,
            'action': action.__dict__,
            'critique': critique.__dict__,
            'simulation': simulation.__dict__,
            'context': context
        }
        
        self.trading_history.append(record)
        
        # Обмеження andсторandї
        if len(self.trading_history) > 1000:
            self.trading_history = self.trading_history[-500:]
    
    def _prepare_training_data(self, feedback_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Пandдготовка data for навчання моwhereлей Деана"""
        return {
            'trade_results': feedback_data,
            'critique_results': [],
            'simulation_results': [],
            'pressure_results': []
        }
    
    def _update_performance_metrics(self, evolution_results: Dict[str, float]):
        """Оновлення метрик продуктивностand"""
        if evolution_results:
            avg_improvement = np.mean(list(evolution_results.values()))
            self.performance_metrics['evolution_rate'] = avg_improvement
    
    def _analyze_trading_history(self) -> Dict[str, Any]:
        """Аналandwith торгової andсторandї"""
        if not self.trading_history:
            return {}
        
        # Роwithрахунок метрик
        total_decisions = len(self.trading_history)
        successful_decisions = sum(1 for record in self.trading_history 
                                 if record['decision'].get('confidence', 0) > 0.6)
        
        action_distribution = {}
        for record in self.trading_history:
            action = record['decision'].get('action', 'unknown')
            action_distribution[action] = action_distribution.get(action, 0) + 1
        
        return {
            'total_decisions': total_decisions,
            'successful_decisions': successful_decisions,
            'success_rate': successful_decisions / total_decisions if total_decisions > 0 else 0,
            'action_distribution': action_distribution,
            'avg_confidence': np.mean([record['decision'].get('confidence', 0) for record in self.trading_history])
        }
    
    def _calculate_overall_metrics(self) -> Dict[str, Any]:
        """Роwithрахунок forгальних метрик"""
        return {
            'bootstrap_success_rate': self.performance_metrics['bootstrap_success_rate'],
            'simulation_accuracy': self.performance_metrics['simulation_accuracy'],
            'evolution_rate': self.performance_metrics['evolution_rate'],
            'overall_profitability': self.performance_metrics['overall_profitability'],
            'system_health': self._assess_system_health()
        }
    
    def _determine_trend(self, market_data: pd.DataFrame) -> str:
        """Виvalues тренду"""
        if len(market_data) < 20:
            return 'neutral'
        
        recent_prices = market_data['Close'].tail(20)
        sma_20 = recent_prices.mean()
        current_price = recent_prices.iloc[-1]
        
        if current_price > sma_20 * 1.02:
            return 'bullish'
        elif current_price < sma_20 * 0.98:
            return 'bearish'
        else:
            return 'neutral'
    
    def _calculate_volatility(self, market_data: pd.DataFrame) -> float:
        """Роwithрахунок волатильностand"""
        if len(market_data) < 20:
            return 0.2
        
        returns = market_data['Close'].pct_change().tail(20)
        return returns.std()
    
    def _calculate_momentum(self, market_data: pd.DataFrame) -> float:
        """Роwithрахунок моментуму"""
        if len(market_data) < 10:
            return 0.0
        
        current_price = market_data['Close'].iloc[-1]
        price_10_days_ago = market_data['Close'].iloc[-10]
        
        return (current_price - price_10_days_ago) / price_10_days_ago
    
    def _find_support_resistance(self, market_data: pd.DataFrame) -> Dict[str, float]:
        """Пошук пandдтримки and опору"""
        if len(market_data) < 50:
            return {'support': 0, 'resistance': 0}
        
        recent_prices = market_data['Close'].tail(50)
        support = recent_prices.min()
        resistance = recent_prices.max()
        
        return {'support': support, 'resistance': resistance}
    
    def _get_market_hours(self) -> str:
        """Отримати сandтус ринкових годин"""
        current_hour = datetime.now().hour
        
        if 9 <= current_hour <= 16:
            return 'open'
        else:
            return 'closed'
    
    def _assess_sentiment(self, features: pd.Series) -> str:
        """Оцandнка сентименту"""
        # Спрощена оцandнка на основand технandчних andндикаторandв
        rsi = features.get('RSI', 50)
        macd = features.get('MACD', 0)
        
        if rsi > 70 and macd > 0:
            return 'positive'
        elif rsi < 30 and macd < 0:
            return 'negative'
        else:
            return 'neutral'
    
    def _assess_data_quality(self, market_data: pd.DataFrame, features_df: pd.DataFrame) -> float:
        """Оцandнка якостand data"""
        quality_score = 1.0
        
        # Перевandрка на пропущенand данand
        if market_data.isnull().any().any():
            quality_score -= 0.2
        
        if features_df.isnull().any().any():
            quality_score -= 0.2
        
        # Перевandрка обсягу data
        if len(market_data) < 100:
            quality_score -= 0.3
        
        if len(features_df) < 50:
            quality_score -= 0.3
        
        return max(0.0, quality_score)
    
    def _assess_final_risk(self, action: str, critique: DeanCritique, simulation: DeanSimulation) -> float:
        """Оцandнка фandнального риwithику"""
        base_risk = 0.5
        
        if action in ['buy', 'sell']:
            base_risk = 0.7
        elif action == 'wait':
            base_risk = 0.2
        elif action == 'hold':
            base_risk = 0.3
        
        # Корекцandя на основand критики
        if critique.critique_score < -0.3:
            base_risk += 0.2
        
        # Корекцandя на основand симуляцandї
        if simulation.confidence_distribution and np.mean(simulation.confidence_distribution) < 0.4:
            base_risk += 0.15
        
        return min(1.0, base_risk)
    
    def _get_action_parameters(self, action: str, base_params: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Отримати параметри дandї"""
        if action == 'buy':
            return {
                'position_size': base_params.get('position_size', 0.1),
                'stop_loss': base_params.get('stop_loss', 0.02),
                'take_profit': base_params.get('take_profit', 0.05)
            }
        elif action == 'sell':
            return {
                'position_size': base_params.get('position_size', 0.1),
                'stop_loss': base_params.get('stop_loss', 0.02),
                'take_profit': base_params.get('take_profit', 0.05)
            }
        elif action == 'wait':
            return {
                'reason': 'waiting_for_better_conditions',
                'wait_time': 3600  # 1 година
            }
        else:
            return {}
    
    def _assess_system_health(self) -> float:
        """Оцandнка withдоров'я system"""
        health_score = 1.0
        
        # Перевandрка andнandцandалandforцandї
        if not self.models_initialized:
            health_score -= 0.5
        
        # Перевandрка кandлькостand моwhereлей
        if len(self.dean_system.models) < 4:
            health_score -= 0.3
        
        # Перевandрка andсторandї
        if len(self.trading_history) < 10:
            health_score -= 0.2
        
        return max(0.0, health_score)


# Глобальний екwithемпляр andнтеграцandї
_dean_integration = None

def get_dean_integration() -> DeanTradingIntegration:
    """Отримати глобальну andнтеграцandю Деана"""
    global _dean_integration
    if _dean_integration is None:
        _dean_integration = DeanTradingIntegration()
    return _dean_integration
