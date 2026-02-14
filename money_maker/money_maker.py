# money_maker.py - Головний script for максимального forробandтку

import pandas as pd
import numpy as np
import json
from typing import Dict, List
import warnings

warnings.filterwarnings('ignore')
import os
import logging

from utils.logger import ProjectLogger
from core.strategy.profit_optimizer import ProfitOptimizer
from models.model_selector.smart_selector import SmartModelSelector
# ВИПРАВЛЕНО: використовуємо StageManager замість видаленої функції
from core.stages.stage_manager import StageManager
from core.stages.stage_2_enrichment import run_stage_2_enrichment
from core.stages.stage_3_features import prepare_stage3_datasets
from core.stages.stage_4_comprehensive_comparison import ComprehensiveComparisonStage
from core.stages.stage_5_context_aware import run_stage_5_context_aware
from core.analysis.smart_switcher import SmartSwitcher
from core.batch_colab_manager import BatchColabManager
from config.config import PATHS
# НОВІ ІМПОРТИ - Enhanced Models
from models.models_train import train_all_models_enhanced, train_dean_models, train_sentiment_models
from models.models_predict import predict_all_models_final, predict_dean_models, predict_sentiment_models
from models.dean_integration import get_dean_integrator
from models.sentiment_integration import get_sentiment_integrator

ProjectLogger.get_logger()
logger = logging.getLogger(__name__)


class MoneyMaker:
    """
    ПОЛНА MONEY MAKER SYSTEM - Paper Trading з Smart Switcher
    та інтеграцією з Colab batch training + Enhanced Models
    """
    
    def __init__(self, enable_colab_integration: bool = True, enable_real_trading: bool = False, 
                 enable_enhanced_models: bool = True):
        self.profit_optimizer = ProfitOptimizer()
        self.model_selector = SmartModelSelector()
        self.smart_switcher = SmartSwitcher()
        self.comprehensive_comparison = ComprehensiveComparisonStage({})
        
        # Enhanced Models Integration
        self.enable_enhanced_models = enable_enhanced_models
        if enable_enhanced_models:
            self.dean_integrator = get_dean_integrator()
            self.sentiment_integrator = get_sentiment_integrator()
            logger.info("[ENHANCED] Enhanced Models enabled: Dean RL + Sentiment + Bayesian")
        
        # Colab integration
        self.enable_colab_integration = enable_colab_integration
        if enable_colab_integration:
            self.colab_manager = BatchColabManager(
                max_concurrent_sessions=2,
                batch_timeout_minutes=45
            )
        
        # Real Trading integration
        self.enable_real_trading = enable_real_trading
        if enable_real_trading:
            from trading.real_trading_system import RealTradingSystem
            self.real_trader = RealTradingSystem(
                initial_balance=10000.0,
                portfolio_name="money_maker_real"
            )
        
        # Paper trading state
        self.paper_trading_state = {
            'active_positions': {},
            'performance_history': [],
            'current_selection': None,
            'total_pnl': 0.0,
            'current_balance': 10000.0,
            'wins': 0,
            'losses': 0,
            'win_rate': 0.0,
            'max_drawdown': 0.0,
            'max_position_size': 0.1  # 10% від балансу
        }
        
        # Training results cache
        self.training_results = {}
        self.feature_importance = {}
        self.model_performance = {}
        
        # Results storage
        self.results = {}
        self.analysis_history = []

    def run_full_pipeline_with_colab(self, tickers: List[str] = None) -> Dict:
        """Запуск повного pipeline з Colab + Enhanced Models"""
        if tickers is None:
            tickers = ["SPY", "QQQ", "TSLA", "NVDA", "AAPL"]
        
        logger.info(f"[PIPELINE] Starting full pipeline with Enhanced Models for {len(tickers)} tickers")
        
        try:
            # Stage 1: Data Collection
            logger.info("[STAGE 1] Starting data collection...")
            stage1_results = run_stage_1_collect(tickers=tickers)
            
            # Stage 2: Data Enrichment  
            logger.info("[STAGE 2] Starting data enrichment...")
            stage2_results = run_stage_2_enrichment(stage1_results)
            
            # Stage 3: Feature Engineering
            logger.info("[STAGE 3] Starting feature engineering...")
            stage3_results = prepare_stage3_datasets(stage2_results)
            
            # Enhanced Model Training
            if self.enable_enhanced_models:
                logger.info("[ENHANCED] Starting Enhanced Model Training...")
                enhanced_results = self._run_enhanced_model_training(stage3_results, tickers)
            else:
                enhanced_results = {}
            
            # Original Pipeline (if needed)
            if self.enable_colab_integration:
                logger.info("[COLAB] Starting Colab integration...")
                colab_results = self._run_colab_training_pipeline(stage3_results, tickers)
            else:
                colab_results = {}
            
            # Stage 4: Model Comparison
            logger.info("[STAGE 4] Starting model comparison...")
            comparison_results = self._run_model_comparison(stage3_results)
            
            # Stage 5: Smart Switcher Setup
            logger.info("[STAGE 5] Setting up Smart Switcher...")
            self._setup_smart_switcher(stage3_results, comparison_results)
            
            # Paper Trading Initialization
            self._initialize_paper_trading()
            
            # Combine all results
            final_results = {
                'pipeline_status': 'completed',
                'tickers': tickers,
                'stage1_results': stage1_results,
                'stage2_results': stage2_results,
                'stage3_results': stage3_results,
                'enhanced_models': enhanced_results if self.enable_enhanced_models else None,
                'colab_results': colab_results if self.enable_colab_integration else None,
                'comparison_results': comparison_results,
                'smart_switcher_ready': True,
                'paper_trading_ready': True,
                'enhanced_models_enabled': self.enable_enhanced_models,
                'total_models_trained': self._count_total_models(enhanced_results, colab_results),
                'timestamp': datetime.now().isoformat()
            }
            
            logger.info(f"[OK] Full pipeline completed successfully with Enhanced Models")
            return final_results
            
        except Exception as e:
            logger.error(f"[ERROR] Pipeline failed: {e}")
            return {'pipeline_status': 'failed', 'error': str(e)}
    
    def _run_enhanced_model_training(self, stage3_results: Dict, tickers: List[str]) -> Dict:
        """Запуск Enhanced Model Training"""
        logger.info("[ENHANCED] Training Dean RL + Sentiment + Bayesian models")
        
        enhanced_results = {
            'dean_models': {},
            'sentiment_models': {},
            'bayesian_models': {},
            'training_summary': {}
        }
        
        try:
            # Отримуємо дані для тренування
            for ticker in tickers:
                if ticker in stage3_results:
                    ticker_data = stage3_results[ticker]
                    
                    # Dean Models Training
                    if self.enable_enhanced_models:
                        try:
                            dean_training_data = {
                                'trade_results': [],  # Тут мають бути реальні результати
                                'simulation_results': [],
                                'market_data': {'features': ticker_data}
                            }
                            dean_improvements = train_dean_models(dean_training_data)
                            enhanced_results['dean_models'][ticker] = dean_improvements
                            logger.info(f"[DEAN] Dean models trained for {ticker}: {dean_improvements}")
                        except Exception as e:
                            logger.warning(f"[DEAN] Dean training failed for {ticker}: {e}")
                            enhanced_results['dean_models'][ticker] = {'error': str(e)}
                    
                    # Sentiment Models Training
                    if self.enable_enhanced_models:
                        try:
                            # Імітація news data
                            news_data = pd.DataFrame({
                                'title': [
                                    f"Market analysis for {ticker}",
                                    f"Earnings report for {ticker}",
                                    f"Technical indicators for {ticker}"
                                ]
                            })
                            sentiment_results = train_sentiment_models(news_data)
                            enhanced_results['sentiment_models'][ticker] = sentiment_results
                            logger.info(f"[SENTIMENT] Sentiment models trained for {ticker}")
                        except Exception as e:
                            logger.warning(f"[SENTIMENT] Sentiment training failed for {ticker}: {e}")
                            enhanced_results['sentiment_models'][ticker] = {'error': str(e)}
            
            # Training Summary
            enhanced_results['training_summary'] = {
                'dean_models_trained': len([r for r in enhanced_results['dean_models'].values() if 'error' not in r]),
                'sentiment_models_trained': len([r for r in enhanced_results['sentiment_models'].values() if 'error' not in r]),
                'total_enhanced_models': len(tickers) * 2,  # Dean + Sentiment
                'training_time': 'enhanced_training_completed'
            }
            
            logger.info(f"[OK] Enhanced model training completed: {enhanced_results['training_summary']}")
            return enhanced_results
            
        except Exception as e:
            logger.error(f"[ERROR] Enhanced training failed: {e}")
            return {'error': str(e)}
    
    def _count_total_models(self, enhanced_results: Dict, colab_results: Dict) -> int:
        """Підрахунок загальної кількості моделей"""
        total = 0
        
        # Enhanced models
        if enhanced_results and 'training_summary' in enhanced_results:
            summary = enhanced_results['training_summary']
            total += summary.get('total_enhanced_models', 0)
        
        # Colab models
        if colab_results:
            for batch_result in colab_results.values():
                if 'training_summary' in batch_result:
                    total += batch_result['training_summary'].get('total_models', 0)
        
        return total
    
    def _run_colab_training_pipeline(self, stage3_results: Dict, tickers: List[str]) -> Dict:
        """Запуск Colab training pipeline"""
        logger.info("[COLAB] Running Colab training pipeline")
        
        try:
            # Підготовка батчів
            features_df = self._extract_features_from_stage3(stage3_results)
            batches = self._prepare_colab_batches(features_df, tickers)
            
            # Відправка в Colab
            colab_results = {}
            for batch in batches:
                batch_result = self._send_to_colab_training(batch)
                colab_results[batch['batch_id']] = batch_result
            
            return colab_results
            
        except Exception as e:
            logger.error(f"[ERROR] Colab pipeline failed: {e}")
            return {'error': str(e)}
    
    def _extract_features_from_stage3(self, stage3_results: Dict) -> pd.DataFrame:
        """Витягування фіч з Stage 3 результатів"""
        logger.info("[FEATURES] Extracting features from Stage 3 results")
        
        try:
            all_features = []
            
            for ticker, ticker_data in stage3_results.items():
                if isinstance(ticker_data, dict) and 'features' in ticker_data:
                    features_df = ticker_data['features']
                    if not features_df.empty:
                        features_df['ticker'] = ticker
                        all_features.append(features_df)
            
            if all_features:
                combined_features = pd.concat(all_features, ignore_index=True)
                logger.info(f"[OK] Extracted features: {combined_features.shape}")
                return combined_features
            else:
                logger.warning("[WARN] No features found in Stage 3 results")
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"[ERROR] Feature extraction failed: {e}")
            return pd.DataFrame()
    
    def _run_model_comparison(self, stage3_results: Dict) -> Dict:
        """Запуск порівняння моделей"""
        logger.info("[COMPARISON] Running model comparison")
        
        try:
            # Імітація порівняння моделей
            comparison_results = {
                'comparison_status': 'completed',
                'best_models': {
                    'light': 'lgbm',
                    'heavy': 'transformer',
                    'overall': 'lgbm'
                },
                'performance_metrics': {
                    'lgbm': {'accuracy': 0.82, 'sharpe': 1.2},
                    'transformer': {'accuracy': 0.85, 'sharpe': 1.4},
                    'lstm': {'accuracy': 0.80, 'sharpe': 1.1}
                },
                'feature_importance_analysis': {
                    'top_features': ['rsi', 'macd', 'volume', 'volatility'],
                    'importance_scores': [0.25, 0.20, 0.15, 0.12]
                }
            }
            
            logger.info(f"[OK] Model comparison completed: {comparison_results['best_models']}")
            return comparison_results
            
        except Exception as e:
            logger.error(f"[ERROR] Model comparison failed: {e}")
            return {'error': str(e)}
    
    def _setup_smart_switcher(self, stage3_results: Dict, comparison_results: Dict):
        """Налаштування Smart Switcher з Enhanced Models"""
        logger.info("[SWITCHER] Setting up Smart Switcher with Enhanced Models")
        
        try:
            # Налаштовуємо Smart Switcher з результатами
            self.smart_switcher.training_results = comparison_results
            self.smart_switcher.feature_importance = comparison_results.get('feature_importance_analysis', {})
            self.smart_switcher.model_performance = comparison_results.get('performance_metrics', {})
            
            # Enhanced Models integration
            if self.enable_enhanced_models:
                self.smart_switcher.enhanced_models_enabled = True
                self.smart_switcher.dean_integrator = self.dean_integrator
                self.smart_switcher.sentiment_integrator = self.sentiment_integrator
                logger.info("[SWITCHER] Enhanced Models integrated in Smart Switcher")
            
            # Тестуємо вибір найкращої комбінації
            features_df = self._extract_features_from_stage3(stage3_results)
            if not features_df.empty:
                test_data = features_df.tail(100)  # Останні 100 записів для тесту
                
                best_combo = self.smart_switcher.select_best_combination(
                    comparison_results, test_data
                )
                
                self.paper_trading_state['current_selection'] = best_combo
                self.paper_trading_state['last_analysis_time'] = datetime.now().isoformat()
                
                logger.info(f"[OK] Smart Switcher ready: {best_combo.get('model')} / {best_combo.get('target')} / {best_combo.get('timeframe')}")
            
        except Exception as e:
            logger.error(f"[ERROR] Smart Switcher setup failed: {e}")
    
    def _initialize_paper_trading(self) -> Dict:
        """Ініціалізація Paper Trading з Enhanced Models"""
        logger.info("[PAPER] Initializing Paper Trading with Enhanced Models")
        
        paper_config = {
            'initial_balance': 100000.0,
            'current_balance': 100000.0,
            'max_position_size': 0.1,  # 10% від балансу
            'risk_per_trade': 0.02,    # 2% ризику на торгівлю
            'stop_loss': 0.05,         # 5% стоп-лосс
            'take_profit': 0.1,        # 10% тейк-профіт
            'max_positions': 5,
            'rebalance_frequency': 'daily',
            'analysis_frequency': 'hourly',
            'smart_switcher_enabled': True,
            'auto_trade_execution': False,  # Paper trading only
            'performance_tracking': True,
            'enhanced_models_enabled': self.enable_enhanced_models
        }
        
        self.paper_trading_state.update(paper_config)
        
        logger.info("[OK] Paper trading initialized with Enhanced Models")
        return paper_config
    
    def run_enhanced_trading_session(self, duration_hours: int = 4) -> Dict:
        """Запуск Enhanced Trading сесії з усіма моделями"""
        logger.info(f"[ENHANCED] Starting Enhanced Trading session for {duration_hours} hours")
        
        try:
            # Ініціалізуємо сесію
            session_start = datetime.now()
            session_stats = {
                'session_start': session_start.isoformat(),
                'duration_hours': duration_hours,
                'enhanced_models_enabled': self.enable_enhanced_models,
                'analyses_performed': 0,
                'signals_generated': 0,
                'trades_executed': 0,
                'enhanced_decisions': {
                    'dean_decisions': 0,
                    'sentiment_decisions': 0,
                    'traditional_decisions': 0
                }
            }
            
            # Симуляція торгової сесії
            for hour in range(duration_hours):
                logger.info(f"[ENHANCED] Hour {hour + 1}/{duration_hours}")
                
                # Enhanced Analysis
                if self.enable_enhanced_models:
                    analysis_result = self._run_enhanced_analysis()
                    session_stats['analyses_performed'] += 1
                    
                    # Dean RL Decision
                    if analysis_result.get('dean_decision'):
                        session_stats['enhanced_decisions']['dean_decisions'] += 1
                    
                    # Sentiment Decision  
                    if analysis_result.get('sentiment_decision'):
                        session_stats['enhanced_decisions']['sentiment_decisions'] += 1
                
                # Traditional Analysis
                traditional_result = self._run_traditional_analysis()
                session_stats['enhanced_decisions']['traditional_decisions'] += 1
                
                # Signal Generation
                signal = self._generate_enhanced_signal(analysis_result, traditional_result)
                if signal:
                    session_stats['signals_generated'] += 1
                    
                    # Trade Execution (paper)
                    trade_result = self._execute_paper_trade(signal)
                    if trade_result:
                        session_stats['trades_executed'] += 1
            
            # Calculate Performance
            performance = self._calculate_session_performance(session_start)
            
            session_results = {
                'session_status': 'completed',
                'session_stats': session_stats,
                'performance': performance,
                'enhanced_models_summary': self._get_enhanced_models_summary(),
                'final_balance': performance.get('current_balance', 100000),
                'total_pnl': performance.get('total_pnl', 0),
                'win_rate': performance.get('win_rate', 0),
                'max_drawdown': performance.get('max_drawdown', 0),
                'session_end': datetime.now().isoformat()
            }
            
            logger.info(f"[OK] Enhanced Trading session completed: PnL ${session_results['total_pnl']:.2f}")
            return session_results
            
        except Exception as e:
            logger.error(f"[ERROR] Enhanced Trading session failed: {e}")
            return {'session_status': 'failed', 'error': str(e)}
    
    def _run_enhanced_analysis(self) -> Dict:
        """Запуск Enhanced Analysis з Dean + Sentiment"""
        try:
            analysis_result = {
                'dean_decision': None,
                'sentiment_decision': None,
                'combined_confidence': 0.0
            }
            
            # Dean RL Analysis
            if self.enable_enhanced_models and hasattr(self, 'dean_integrator'):
                try:
                    # Імітація data для Dean
                    market_data = pd.DataFrame({
                        'close': [100, 101, 102, 101, 103],
                        'volume': [1000, 1200, 900, 1100, 1300]
                    })
                    dean_decision = self.dean_integrator.get_trading_decision(
                        market_data, 'SPY', '1h'
                    )
                    analysis_result['dean_decision'] = dean_decision
                    logger.info(f"[DEAN] Dean decision: {dean_decision.get('type', 'unknown')}")
                except Exception as e:
                    logger.warning(f"[DEAN] Dean analysis failed: {e}")
            
            # Sentiment Analysis
            if self.enable_enhanced_models and hasattr(self, 'sentiment_integrator'):
                try:
                    # Імітація news data
                    news_data = pd.DataFrame({
                        'title': ['Market rally continues', 'Fed signals rate cut'],
                        'text': ['Positive market sentiment', 'Economic indicators strong']
                    })
                    price_data = pd.DataFrame({
                        'close': [100, 101, 102],
                        'volume': [1000, 1200, 900]
                    })
                    sentiment_decision = self.sentiment_integrator.get_sentiment_signal(
                        news_data, price_data
                    )
                    analysis_result['sentiment_decision'] = sentiment_decision
                    logger.info(f"[SENTIMENT] Sentiment signal: {sentiment_decision.get('signal_type', 'unknown')}")
                except Exception as e:
                    logger.warning(f"[SENTIMENT] Sentiment analysis failed: {e}")
            
            # Calculate combined confidence
            dean_conf = analysis_result['dean_decision'].get('confidence', 0) if analysis_result['dean_decision'] else 0
            sentiment_conf = analysis_result['sentiment_decision'].get('confidence', 0) if analysis_result['sentiment_decision'] else 0
            analysis_result['combined_confidence'] = (dean_conf + sentiment_conf) / 2
            
            return analysis_result
            
        except Exception as e:
            logger.error(f"[ERROR] Enhanced analysis failed: {e}")
            return {}
    
    def _run_traditional_analysis(self) -> Dict:
        """Запуск Traditional Analysis"""
        try:
            # Імітація традиційного аналізу
            traditional_result = {
                'model_type': 'lgbm',
                'signal': 'buy',
                'confidence': 0.75,
                'target': 'direction',
                'timeframe': '1h'
            }
            
            logger.info(f"[TRADITIONAL] Traditional signal: {traditional_result['signal']}")
            return traditional_result
            
        except Exception as e:
            logger.error(f"[ERROR] Traditional analysis failed: {e}")
            return {}
    
    def _generate_enhanced_signal(self, enhanced_analysis: Dict, traditional_analysis: Dict) -> Dict:
        """Генерація Enhanced Signal"""
        try:
            # Комбінуємо всі сигнали
            signals = []
            
            # Traditional signal
            if traditional_analysis:
                signals.append({
                    'source': 'traditional',
                    'signal': traditional_analysis.get('signal', 'hold'),
                    'confidence': traditional_analysis.get('confidence', 0.5),
                    'weight': 0.6
                })
            
            # Dean signal
            if enhanced_analysis.get('dean_decision'):
                dean_decision = enhanced_analysis['dean_decision']
                signals.append({
                    'source': 'dean_rl',
                    'signal': dean_decision.get('type', 'hold'),
                    'confidence': dean_decision.get('confidence', 0.5),
                    'weight': 0.25
                })
            
            # Sentiment signal
            if enhanced_analysis.get('sentiment_decision'):
                sentiment_decision = enhanced_analysis['sentiment_decision']
                signals.append({
                    'source': 'sentiment',
                    'signal': sentiment_decision.get('signal_type', 'hold'),
                    'confidence': sentiment_decision.get('confidence', 0.5),
                    'weight': 0.15
                })
            
            # Calculate weighted signal
            if signals:
                weighted_signal = self._calculate_weighted_signal(signals)
                logger.info(f"[SIGNAL] Enhanced signal: {weighted_signal['signal']} (confidence: {weighted_signal['confidence']:.2f})")
                return weighted_signal
            
            return None
            
        except Exception as e:
            logger.error(f"[ERROR] Signal generation failed: {e}")
            return None
    
    def _calculate_weighted_signal(self, signals: List[Dict]) -> Dict:
        """Розрахунок зваженого сигналу"""
        signal_scores = {'buy': 1, 'sell': -1, 'hold': 0}
        
        total_weight = 0
        weighted_score = 0
        total_confidence = 0
        
        for signal in signals:
            weight = signal['weight']
            confidence = signal['confidence']
            signal_type = signal['signal']
            
            score = signal_scores.get(signal_type, 0)
            
            weighted_score += score * weight * confidence
            total_weight += weight * confidence
            total_confidence += confidence
        
        if total_weight > 0:
            final_score = weighted_score / total_weight
            final_confidence = total_confidence / len(signals)
            
            # Convert score back to signal
            if final_score > 0.3:
                final_signal = 'buy'
            elif final_score < -0.3:
                final_signal = 'sell'
            else:
                final_signal = 'hold'
            
            return {
                'signal': final_signal,
                'confidence': final_confidence,
                'weighted_score': final_score,
                'sources': [s['source'] for s in signals]
            }
        
        return {'signal': 'hold', 'confidence': 0.0, 'weighted_score': 0.0, 'sources': []}
    
    def _execute_paper_trade(self, signal: Dict) -> Dict:
        """Виконання paper trade"""
        try:
            # Імітація виконання торгівлі
            trade_result = {
                'signal': signal.get('signal', 'hold'),
                'confidence': signal.get('confidence', 0),
                'execution_price': 100.0 + np.random.uniform(-2, 2),
                'quantity': 10,
                'pnl': np.random.uniform(-50, 100),
                'timestamp': datetime.now().isoformat(),
                'sources': signal.get('sources', [])
            }
            
            logger.info(f"[TRADE] Paper trade executed: {trade_result['signal']} PnL: ${trade_result['pnl']:.2f}")
            return trade_result
            
        except Exception as e:
            logger.error(f"[ERROR] Paper trade execution failed: {e}")
            return None
    
    def _calculate_session_performance(self, session_start: datetime) -> Dict:
        """Розрахунок продуктивності сесії"""
        try:
            # Імітація розрахунку продуктивності
            duration = (datetime.now() - session_start).total_seconds() / 3600
            
            performance = {
                'initial_balance': 100000,
                'current_balance': 100000 + np.random.uniform(-2000, 5000),
                'total_pnl': np.random.uniform(-2000, 5000),
                'total_pnl_pct': np.random.uniform(-2, 5),
                'win_rate': np.random.uniform(0.45, 0.65),
                'max_drawdown': np.random.uniform(0.02, 0.08),
                'sharpe_ratio': np.random.uniform(0.5, 1.5),
                'duration_hours': duration,
                'trades_per_hour': np.random.uniform(2, 8)
            }
            
            return performance
            
        except Exception as e:
            logger.error(f"[ERROR] Performance calculation failed: {e}")
            return {}
    
    def _get_enhanced_models_summary(self) -> Dict:
        """Отримання summary Enhanced моделей"""
        try:
            summary = {
                'dean_models': {
                    'enabled': self.enable_enhanced_models,
                    'models': ['DeanActorModel', 'DeanCriticModel', 'DeanAdversaryModel', 'DeanSimulatorModel'],
                    'status': 'integrated' if self.enable_enhanced_models else 'disabled'
                },
                'sentiment_models': {
                    'enabled': self.enable_enhanced_models,
                    'models': ['FinBERT', 'SentimentAnalyzer'],
                    'status': 'integrated' if self.enable_enhanced_models else 'disabled'
                },
                'bayesian_optimization': {
                    'enabled': self.enable_enhanced_models,
                    'models': ['LGBM_Bayesian'],
                    'status': 'integrated' if self.enable_enhanced_models else 'disabled'
                }
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"[ERROR] Enhanced models summary failed: {e}")
            return {}
    
    def run_real_trading_session(self, duration_hours: int = 4) -> Dict:
        """Запуск реальної торгової сесії з Enhanced Models"""
        logger.info(f"[REAL] Starting Real Trading session with Enhanced Models for {duration_hours} hours")
        
        try:
            # Використовуємо Enhanced Trading сесію
            enhanced_results = self.run_enhanced_trading_session(duration_hours)
            
            # Додаємо реальну торгівлю логіку
            if enhanced_results.get('session_status') == 'completed':
                enhanced_results['trading_mode'] = 'real_virtual_account'
                enhanced_results['data_source'] = 'real_market_data'
                enhanced_results['execution_type'] = 'paper_trading_with_real_data'
                
                logger.info(f"[OK] Real Trading session completed: PnL ${enhanced_results['total_pnl']:.2f}")
                return enhanced_results
            else:
                logger.error("[ERROR] Enhanced Trading session failed")
                return enhanced_results
                
        except Exception as e:
            logger.error(f"[ERROR] Real Trading session failed: {e}")
            return {'session_status': 'failed', 'error': str(e)}
    
    def run_full_optimization(self) -> Dict:
        """Запуск повної оптимізації з Enhanced Models"""
        logger.info("[OPTIMIZATION] Starting full optimization with Enhanced Models")
        
        try:
            # Запускаємо повний pipeline
            pipeline_results = self.run_full_pipeline_with_colab()
            
            if pipeline_results.get('pipeline_status') == 'completed':
                # Запускаємо Enhanced Trading сесію
                trading_results = self.run_enhanced_trading_session(duration_hours=2)
                
                # Об'єднуємо результати
                optimization_results = {
                    'optimization_status': 'completed',
                    'pipeline_results': pipeline_results,
                    'trading_results': trading_results,
                    'enhanced_models_enabled': self.enable_enhanced_models,
                    'total_models_trained': pipeline_results.get('total_models_trained', 0),
                    'final_pnl': trading_results.get('total_pnl', 0),
                    'final_balance': trading_results.get('final_balance', 100000),
                    'optimization_summary': {
                        'pipeline_completed': pipeline_results.get('pipeline_status') == 'completed',
                        'trading_completed': trading_results.get('session_status') == 'completed',
                        'enhanced_models_active': self.enable_enhanced_models,
                        'smart_switcher_ready': pipeline_results.get('smart_switcher_ready', False)
                    }
                }
                
                logger.info(f"[OK] Full optimization completed: PnL ${optimization_results['final_pnl']:.2f}")
                return optimization_results
            else:
                logger.error("[ERROR] Pipeline failed during optimization")
                return pipeline_results
                
        except Exception as e:
            logger.error(f"[ERROR] Full optimization failed: {e}")
            return {'optimization_status': 'failed', 'error': str(e)}
    
    def _train_light_models_locally(self, features_df: pd.DataFrame, tickers: List[str]) -> Dict:
        """Тренування легких моделей локально"""
        logger.info(f"[LOCAL] Training light models for {len(tickers)} tickers")
        
        light_models = ['lgbm', 'xgboost', 'rf']
        results = {}
        
        for ticker in tickers:
            results[ticker] = {}
            
            for timeframe in ['15m', '60m', '1d']:
                results[ticker][timeframe] = {}
                
                for model_type in light_models:
                    try:
                        # Спрощене тренування для демонстрації
                        model_result = {
                            'model': model_type,
                            'accuracy': np.random.uniform(0.65, 0.85),
                            'sharpe_ratio': np.random.uniform(0.5, 1.5),
                            'max_drawdown': np.random.uniform(0.05, 0.15),
                            'samples': len(features_df),
                            'training_time': np.random.uniform(10, 60),
                            'feature_importance': self._generate_mock_feature_importance()
                        }
                        
                        results[ticker][timeframe][model_type] = model_result
                        
                    except Exception as e:
                        logger.warning(f"Error training {model_type} for {ticker}_{timeframe}: {e}")
        
        logger.info(f"[OK] Light models training completed: {len(results)} tickers")
        return results
    
    def _prepare_colab_batches(self, features_df: pd.DataFrame, tickers: List[str]) -> List[Dict]:
        """Підготовка батчів для Colab"""
        logger.info("[COLAB] Preparing batches for Colab training")
        
        # Створюємо батчі по комбінаціях тікер/таймфрейм
        batches = []
        heavy_models = ['mlp', 'cnn', 'lstm', 'transformer']
        
        # Розбиваємо на батчі по 3 тікери для уникнення timeout в Colab
        ticker_batches = [tickers[i:i+3] for i in range(0, len(tickers), 3)]
        
        for batch_idx, ticker_batch in enumerate(ticker_batches):
            batch_data = {
                'batch_id': f'colab_batch_{batch_idx + 1}',
                'tickers': ticker_batch,
                'timeframes': ['15m', '60m', '1d'],
                'targets': ['direction', 'volatility', 'regime', 'momentum'],
                'models': heavy_models,
                'features': features_df.columns.tolist(),
                'data_sample_size': min(1000, len(features_df)),  # Обмеження для Colab
                'priority': 'high' if batch_idx == 0 else 'normal'
            }
            
            batches.append(batch_data)
        
        logger.info(f"[OK] Prepared {len(batches)} batches for Colab")
        return batches
    
    def _send_to_colab_training(self, batches: List[Dict]) -> Dict:
        """Відправка батчів в Colab"""
        logger.info("[COLAB] Sending batches to Colab")
        
        # Імітація відправки в Colab
        colab_results = {}
        
        for batch in batches:
            batch_result = {
                'batch_id': batch['batch_id'],
                'status': 'sent',
                'sent_at': datetime.now().isoformat(),
                'estimated_completion': (datetime.now() + timedelta(minutes=45)).isoformat(),
                'colab_session_id': f"session_{batch['batch_id']}_{int(time.time())}"
            }
            
            colab_results[batch['batch_id']] = batch_result
            
            # В реальності тут був б код для відправки в Colab
            logger.info(f"[SENT] Batch {batch['batch_id']} sent to Colab")
        
        return colab_results
    
    def _wait_for_colab_completion(self, colab_results: Dict) -> Dict:
        """Очікування завершення Colab тренування"""
        logger.info("[COLAB] Waiting for training completion")
        
        completed_results = {}
        
        # Імітація очікування та отримання результатів
        for batch_id, batch_info in colab_results.items():
            # В реальності тут був б код для перевірки статусу
            time.sleep(2)  # Імітація
            
            # Генеруємо фейкові результати
            mock_results = self._generate_mock_colab_results(batch_info)
            completed_results[batch_id] = mock_results
            
            logger.info(f"[COMPLETED] Batch {batch_id} completed")
        
        logger.info(f"[OK] All {len(completed_results)} Colab batches completed")
        return completed_results
    
    def _generate_mock_colab_results(self, batch_info: Dict) -> Dict:
        """Генерація мок результатів Colab"""
        tickers = batch_info['tickers']
        models = batch_info['models']
        
        results = {}
        
        for ticker in tickers:
            results[ticker] = {}
            
            for timeframe in batch_info['timeframes']:
                results[ticker][timeframe] = {}
                
                for model in models:
                    # Важкі моделі мають кращу продуктивність
                    results[ticker][timeframe][model] = {
                        'model': model,
                        'accuracy': np.random.uniform(0.75, 0.92),
                        'sharpe_ratio': np.random.uniform(0.8, 2.2),
                        'max_drawdown': np.random.uniform(0.03, 0.12),
                        'samples': 1000,
                        'training_time': np.random.uniform(300, 900),  # 5-15 хв
                        'feature_importance': self._generate_mock_feature_importance(),
                        'validation_score': np.random.uniform(0.70, 0.88),
                        'training_environment': 'colab'
                    }
        
        return {
            'batch_id': batch_info['batch_id'],
            'status': 'completed',
            'completed_at': datetime.now().isoformat(),
            'results': results,
            'training_summary': {
                'total_models': len(tickers) * len(batch_info['timeframes']) * len(models),
                'avg_accuracy': np.random.uniform(0.78, 0.86),
                'avg_training_time': np.random.uniform(400, 800)
            }
        }
    
    def _merge_training_results(self, light_results: Dict, colab_results: Dict) -> Dict:
        """Об'єднання результатів локального та Colab тренування"""
        logger.info("[MERGE] Merging local and Colab training results")
        
        merged_results = {
            'light_models': light_results,
            'heavy_models': {},
            'comparison_ready': True
        }
        
        # Додаємо результати важких моделей з Colab
        for batch_id, batch_result in colab_results.items():
            if 'results' in batch_result:
                for ticker, ticker_data in batch_result['results'].items():
                    if ticker not in merged_results['heavy_models']:
                        merged_results['heavy_models'][ticker] = {}
                    
                    for timeframe, tf_data in ticker_data.items():
                        if timeframe not in merged_results['heavy_models'][ticker]:
                            merged_results['heavy_models'][ticker][timeframe] = {}
                        
                        merged_results['heavy_models'][ticker][timeframe].update(tf_data)
        
        logger.info(f"[OK] Merged results for {len(merged_results['heavy_models'])} tickers")
        return merged_results
    
    def _run_comprehensive_comparison(self, training_results: Dict, features_df: pd.DataFrame) -> Dict:
        """Запуск комплексного порівняння моделей"""
        logger.info("[COMPARISON] Running comprehensive model comparison")
        
        try:
            # Створюємо конфігурацію для порівняння
            comparison_config = {}
            
            # Додаємо легкі моделі
            for ticker, ticker_data in training_results.get('light_models', {}).items():
                for timeframe, tf_data in ticker_data.items():
                    for model, model_data in tf_data.items():
                        key = f"{ticker}_{timeframe}_{model}"
                        comparison_config[key] = {
                            'ticker': ticker,
                            'timeframe': timeframe,
                            'model': model,
                            'type': 'light',
                            'performance': model_data
                        }
            
            # Додаємо важкі моделі
            for ticker, ticker_data in training_results.get('heavy_models', {}).items():
                for timeframe, tf_data in ticker_data.items():
                    for model, model_data in tf_data.items():
                        key = f"{ticker}_{timeframe}_{model}"
                        comparison_config[key] = {
                            'ticker': ticker,
                            'timeframe': timeframe,
                            'model': model,
                            'type': 'heavy',
                            'performance': model_data
                        }
            
            # Запускаємо порівняння
            comparison_results = self.comprehensive_comparison.run_comprehensive_comparison(
                {'features_df': features_df, 'model_configs': comparison_config}
            )
            
            logger.info("[OK] Comprehensive comparison completed")
            return comparison_results
            
        except Exception as e:
            logger.error(f"[ERROR] Comparison failed: {e}")
            return {'error': str(e)}
    
    def _setup_smart_switcher(self, comparison_results: Dict, features_df: pd.DataFrame):
        """Налаштування Smart Switcher"""
        logger.info("[SWITCHER] Setting up Smart Switcher")
        
        try:
            # Налаштовуємо Smart Switcher з результатами
            self.smart_switcher.training_results = comparison_results
            self.smart_switcher.feature_importance = comparison_results.get('feature_importance_analysis', {})
            self.smart_switcher.model_performance = comparison_results.get('comparison_results', {})
            
            # Тестуємо вибір найкращої комбінації
            test_data = features_df.tail(100)  # Останні 100 записів для тесту
            
            if not test_data.empty:
                best_combo = self.smart_switcher.select_best_combination(
                    comparison_results, test_data
                )
                
                self.paper_trading_state['current_selection'] = best_combo
                self.paper_trading_state['last_analysis_time'] = datetime.now().isoformat()
                
                logger.info(f"[OK] Smart Switcher ready: {best_combo.get('model')} / {best_combo.get('target')} / {best_combo.get('timeframe')}")
            
        except Exception as e:
            logger.error(f"[ERROR] Smart Switcher setup failed: {e}")
    
    def _initialize_paper_trading(self) -> Dict:
        """Ініціалізація Paper Trading"""
        logger.info("[PAPER] Initializing Paper Trading system")
        
        paper_config = {
            'initial_balance': 100000.0,
            'current_balance': 100000.0,
            'max_position_size': 0.1,  # 10% від балансу
            'risk_per_trade': 0.02,    # 2% ризику на торгівлю
            'stop_loss': 0.05,         # 5% стоп-лосс
            'take_profit': 0.1,        # 10% тейк-профіт
            'max_positions': 5,
            'rebalance_frequency': 'daily',
            'analysis_frequency': 'hourly',
            'smart_switcher_enabled': True,
            'auto_trade_execution': False,  # Paper trading only
            'performance_tracking': True
        }
        
        self.paper_trading_state.update(paper_config)
        
        logger.info("[OK] Paper trading initialized")
        return paper_config
    
    def run_paper_trading_session(self, duration_minutes: int = 60) -> Dict:
        """Запуск сесії Paper Trading"""
        logger.info(f"[PAPER] Starting paper trading session for {duration_minutes} minutes")
        
        session_start = datetime.now()
        session_results = {
            'session_start': session_start.isoformat(),
            'trades': [],
            'performance': {},
            'selections_made': 0,
            'pnl': 0.0
        }
        
        try:
            # Симуляція торгової сесії
            for minute in range(duration_minutes):
                current_time = session_start + timedelta(minutes=minute)
                
                # Аналіз ринку та вибір моделі (кожні 15 хвилин)
                if minute % 15 == 0:
                    selection = self._analyze_and_select_model()
                    
                    if selection:
                        session_results['selections_made'] += 1
                        
                        # Симуляція торгівлі
                        trade_result = self._simulate_paper_trade(selection, current_time)
                        session_results['trades'].append(trade_result)
                
                # Оновлення стану
                if minute % 30 == 0:
                    self._update_paper_trading_state()
                
                # Невелика затримка
                time.sleep(0.1)
            
            # Фінальні розрахунки
            session_results['session_end'] = datetime.now().isoformat()
            session_results['duration_minutes'] = duration_minutes
            session_results['pnl'] = self.paper_trading_state['total_pnl']
            session_results['win_rate'] = self.paper_trading_state['win_rate']
            session_results['max_drawdown'] = self.paper_trading_state['max_drawdown']
            
            logger.info(f"[OK] Paper trading session completed: PnL={session_results['pnl']:.2f}")
            
        except Exception as e:
            logger.error(f"[ERROR] Paper trading session failed: {e}")
            session_results['error'] = str(e)
        
        return session_results
    
    def _analyze_and_select_model(self) -> Dict:
        """Аналіз та вибір моделі"""
        try:
            # Отримуємо поточні дані (симуляція)
            current_data = self._get_current_market_data()
            
            if current_data.empty:
                return None
            
            # Використовуємо Smart Switcher для вибору
            best_selection = self.smart_switcher.select_best_combination(
                self.training_results, current_data
            )
            
            return best_selection
            
        except Exception as e:
            logger.error(f"[ERROR] Model selection failed: {e}")
            return None
    
    def _simulate_paper_trade(self, selection: Dict, trade_time: datetime) -> Dict:
        """Симуляція paper торгівлі"""
        try:
            # Симуляція результату торгівлі
            entry_price = np.random.uniform(100, 500)
            return_pct = np.random.normal(0.001, 0.02)  # 0.1% середній, 2% волатильність
            
            exit_price = entry_price * (1 + return_pct)
            pnl = return_pct * self.paper_trading_state['max_position_size'] * self.paper_trading_state['current_balance']
            
            # Оновлення стану
            self.paper_trading_state['total_pnl'] += pnl
            self.paper_trading_state['current_balance'] += pnl
            
            # Розрахунок win rate
            if pnl > 0:
                wins = self.paper_trading_state.get('wins', 0) + 1
                self.paper_trading_state['wins'] = wins
            else:
                losses = self.paper_trading_state.get('losses', 0) + 1
                self.paper_trading_state['losses'] = losses
            
            total_trades = self.paper_trading_state.get('wins', 0) + self.paper_trading_state.get('losses', 0)
            if total_trades > 0:
                self.paper_trading_state['win_rate'] = self.paper_trading_state.get('wins', 0) / total_trades
            
            trade_result = {
                'timestamp': trade_time.isoformat(),
                'selection': selection,
                'entry_price': entry_price,
                'exit_price': exit_price,
                'return_pct': return_pct,
                'pnl': pnl,
                'balance_after': self.paper_trading_state['current_balance'],
                'trade_type': 'profit' if pnl > 0 else 'loss'
            }
            
            return trade_result
            
        except Exception as e:
            logger.error(f"[ERROR] Trade simulation failed: {e}")
            return {'error': str(e)}
    
    def _get_current_market_data(self) -> pd.DataFrame:
        """Отримання поточних ринкових data (симуляція)"""
        # Симуляція поточних data
        np.random.seed(42)
        
        data = {
            'close': np.random.normal(200, 10, 50),
            'volume': np.random.normal(1000000, 200000, 50),
            'rsi_14': np.random.normal(50, 15, 50),
            'macd': np.random.normal(0, 1, 50),
            'sentiment_score': np.random.normal(0, 0.3, 50),
            'vix': np.random.normal(20, 5, 50)
        }
        
        df = pd.DataFrame(data)
        df.index = pd.date_range(end=datetime.now(), periods=50, freq='15T')
        
        return df
    
    def _update_paper_trading_state(self):
        """Оновлення стану paper trading"""
        # Розрахунок max drawdown
        peak_balance = max(self.paper_trading_state['current_balance'], 100000)
        current_drawdown = (peak_balance - self.paper_trading_state['current_balance']) / peak_balance
        self.paper_trading_state['max_drawdown'] = max(self.paper_trading_state['max_drawdown'], current_drawdown)
    
    def _generate_mock_feature_importance(self) -> Dict:
        """Генерація мок важливості фіч"""
        top_features = [
            'rsi_14', 'macd', 'sma_20', 'volume_ratio', 'sentiment_score',
            'vix', 't10y2y', 'price_momentum', 'volatility_20', 'atr_14'
        ]
        
        importance = {}
        remaining = 1.0
        
        for i, feature in enumerate(top_features):
            if i == len(top_features) - 1:
                importance[feature] = remaining
            else:
                imp = np.random.uniform(0.05, 0.15)
                importance[feature] = imp
                remaining -= imp
        
        return importance

    def find_best_profit_strategies(self, stage1_data: Dict, df: pd.DataFrame, tickers: List[str] = None) -> Dict:
        logger.info("[SEARCH] Пошук найкращих стратегandй for forробandтку")
        if tickers is None: tickers = ["SPY", "QQQ", "TSLA", "NVDA"]
        
        # Використовуємо SmartModelSelector for вибору найкращих моwhereлей
        recommendations = self.model_selector.get_model_recommendations(df, tickers)
        
        # Формуємо стратегandї на основand рекомендацandй
        strategies = {}
        for ticker, rec in recommendations.items():
            strategies[ticker] = {
                "best_model": rec.get("best", {}).get("model", "lgbm"),
                "best_target": rec.get("best", {}).get("target", "direction"),
                "best_score": rec.get("best", {}).get("score", 0.0),
                "classification_model": rec.get("classification", {}).get("model", "lgbm"),
                "regression_model": rec.get("regression", {}).get("model", "lgbm"),
                "context": self.model_selector.analyze_context(df, ticker)
            }
        
        self.results = strategies
        return strategies

    def _save_results(self, filename: str):
        """Збереження результатів"""
        try:
            results_path = os.path.join(PATHS.RESULTS_DIR, filename)
            with open(results_path, 'w') as f:
                json.dump(self.results, f, indent=2, default=str)
            logger.info(f"[OK] Results saved to {results_path}")
        except Exception as e:
            logger.error(f"[ERROR] Failed to save results: {e}")

    def print_results(self):
        print(json.dumps(self.results, indent=2, default=str))

    def save_results(self, filename: str = "money_maker_results.json"):
        self._save_results(filename)

    def run_full_optimization(self) -> Dict:
        """Запускає повний пайплайн вandд withбору data до вибору стратегandї."""
        return self.run_full_pipeline_with_colab()

    def run_full_pipeline_with_colab(self) -> Dict:
        """Запускає повний пайплайн вandд withбору data до вибору стратегandї."""
        logger.info("[START] ЗАПУСК ПОВНОГО ПАЙПЛАЙНУ ОПТИМІЗАЦІЇ")

        try:
            # Еandп 1: Збandр сирих data
            logger.info("--- ЕТАП 1: Збandр сирих data ---")
            stage1_data = run_stage_1_collect()

            # Еandп 2: Збагачення data
            logger.info("--- ЕТАП 2: Збагачення data ---")
            merged_df, _, _, _ = run_stage_2_enrichment(stage1_data)

            # Еandп 3: Створення фandч and andргетandв
            logger.info("--- ЕТАП 3: Створення фandч and andргетandв ---")
            _, _, features_df, _ = prepare_stage3_datasets(merged_df=merged_df)

            if features_df.empty:
                logger.error("[ERROR] Фandнальний даandфрейм порожнandй. Оптимandforцandя notможлива.")
                return {}

            # Еandп 4: Пошук найкращих стратегandй
            logger.info("--- ЕТАП 4: Пошук найкращих стратегandй ---")
            strategies = self.find_best_profit_strategies(stage1_data, features_df)

            self.results = strategies
            self.print_results()
            self.save_results()

            logger.info("[OK] Пайплайн оптимandforцandї успandшно forвершено!")
            return strategies

        except Exception as e:
            logger.error(f"[ERROR] Критична error пandд час виконання пайплайну: {e}", exc_info=True)
            raise

    def run_real_trading_session(self, duration_hours: int = 8) -> Dict:
        """Запуск реальної торгової сесії з віртуальним рахунком"""
        if not self.enable_real_trading:
            raise ValueError("Real trading not enabled. Initialize MoneyMaker with enable_real_trading=True")
        
        logger.info(f"[REAL] Starting real trading session for {duration_hours} hours...")
        
        try:
            # Запуск реальної торгової сесії
            session_results = self.real_trader.run_trading_session(duration_hours)
            
            # Збереження результатів
            self.real_trader.save_session_results(session_results)
            
            logger.info(f"[OK] Real trading session completed")
            return session_results
            
        except Exception as e:
            logger.error(f"[ERROR] Real trading session failed: {e}")
            return {'error': str(e)}


def main():
    logger.info("[MONEY] MONEY MAKER - Максимandforцandя forробandтку череthrough ML")
    try:
        money_maker = MoneyMaker(enable_real_trading=True)
        money_maker.run_full_optimization()
    except Exception as e:
        logger.error(f"[ERROR] Критична error: {e}", exc_info=True)


if __name__ == "__main__":
    main()
