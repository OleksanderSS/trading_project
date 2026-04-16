"""
Stage 6: Trading Execution

This stage takes the final predictions from Stage 5 and orchestrates the entire
trading process using the refactored trading module.
"""

import pandas as pd
from typing import Dict, Any, List

from src.pipeline.stages.base_stage import BaseStage
from src.core.logging.logger import ProjectLogger

# Import all the refactored trading components
from src.trading.virtual_portfolio import VirtualPortfolio
from src.trading.consensus_engine import ConsensusEngine
from src.trading.post_inference_filter import PostInferenceFilter
from src.trading.portfolio_manager import PortfolioManager
from src.trading.trader import Trader
from src.trading.trading_orchestrator import TradingOrchestrator
from src.analytics.analyzers.news_impact_analyzer import NewsImpactAnalyzer
from src.analytics.analyzers.causal_event_finder import CausalEventFinder

class TradingExecutionStage(BaseStage):
    """
    A pipeline stage to execute the trading logic.
    """
    def __init__(self, config_manager, error_handler, **kwargs):
        super().__init__(config_manager, error_handler, **kwargs)
        self.logger = ProjectLogger.get_logger(self.__class__.__name__)
        self.news_impact_analyzer = NewsImpactAnalyzer(config=self.config_manager.get('analysis.news_impact', {}))
        self.causal_finder = CausalEventFinder(treatment='signal', outcome='predicted_return', common_causes=['confidence', 'anomaly_score'])
        self._initialize_trading_stack()

    def _initialize_trading_stack(self):
        """
        Initializes the full trading stack, wiring all components together.
        """
        self.logger.info("Initializing the complete trading stack...")

        # 1. Initialize the state keeper: Virtual Portfolio
        self.portfolio = VirtualPortfolio()
        self.logger.info(f"Initialized VirtualPortfolio. Cash: {self.portfolio.current_balance:.2f}")

        # 2. Initialize the optional filter
        self.post_inference_filter = PostInferenceFilter()
        self.logger.info("Initialized PostInferenceFilter.")

        # 3. Initialize the decision maker: Consensus Engine
        from src.meta_learning.memory.diary_engine import DiaryEngine
        from src.analytics.analyzers.adaptive_confidence_analyzer import AdaptiveConfidenceAnalyzer
        
        self.diary_engine = DiaryEngine()
        self.threshold_analyzer = AdaptiveConfidenceAnalyzer(config=self.config_manager.get('analysis.adaptive_confidence', {}))

        self.consensus_engine = ConsensusEngine(
            experience_diary=self.diary_engine,
            threshold_analyzer=self.threshold_analyzer,
            config_manager=self.config_manager
        )
        self.logger.info("Initialized ConsensusEngine.")

        # 4. Initialize the risk officer: Portfolio Manager
        self.portfolio_manager = PortfolioManager(
            virtual_portfolio=self.portfolio, 
            config=self.config_manager.get('strategy.risk_management', {})
        )
        self.logger.info("Initialized PortfolioManager.")

        # 5. Initialize the executor: Trader
        self.trader = Trader(paper_trading=True)
        self.logger.info("Initialized Trader.")

        # 6. Initialize the main conductor for the trading process
        self.trading_orchestrator = TradingOrchestrator(
            consensus_engine=self.consensus_engine,
            portfolio_manager=self.portfolio_manager,
            virtual_portfolio=self.portfolio,
            trader=self.trader,
            post_inference_filter=self.post_inference_filter
        )
        self.logger.info("Trading stack initialization complete.")

    async def run(self, **kwargs) -> Dict[str, Any]:
        """
        The entry point for the trading execution stage.

        Args:
            **kwargs: The data dictionary from the previous stage.
                                   Expected to contain 'predictions' and 'current_prices'.

        Returns:
            Dict[str, Any]: The data dictionary, potentially updated with trading results.
        """
        self.logger.info("Starting trading execution stage...")

        predictions = kwargs.get('predictions')
        current_prices = kwargs.get('current_prices')

        # ✅ FIX: Якщо predictions не знайдені, спробуємо завантажити з диска (Stage 5 не запускалась)
        if not predictions:
            self.logger.warning("⚠️ No 'predictions' found in kwargs. Attempting to load from disk...")
            
            # Спробуємо знайти stage_5_results.json
            from pathlib import Path
            import json
            
            # Витягуємо batch_name з kwargs або шукаємо найновіший
            batch_name = kwargs.get('batch_name')
            output_dir = Path(self.config_manager.get('system.accumulation.output_dir', 'data/colab/accumulated'))
            
            if not batch_name:
                # Шукаємо найновіший batch
                batch_dirs = list(output_dir.glob('test_ticker_*'))
                if batch_dirs:
                    batch_name = max(batch_dirs, key=lambda p: p.stat().st_mtime).name
                    self.logger.info(f"🔍 Знайдено найновіший batch: {batch_name}")
            
            if batch_name:
                batch_dir = output_dir / batch_name
                stage_5_file = batch_dir / "stage_5_results.json"
                
                if stage_5_file.exists():
                    try:
                        with open(stage_5_file, 'r') as f:
                            stage_5_results = json.load(f)
                        
                        predictions = stage_5_results.get('predictions', [])
                        current_prices = stage_5_results.get('current_prices', {})
                        
                        self.logger.info(f"✅ Завантажено {len(predictions)} прогнозів з {stage_5_file.name}")
                        self.logger.info(f"✅ Завантажено ціни для {len(current_prices)} тікерів")
                        
                        # Додаємо models_metadata якщо не передана
                        if 'models_metadata' not in kwargs:
                            models_metadata = stage_5_results.get('models_metadata', {})
                            if models_metadata:
                                kwargs['models_metadata'] = models_metadata
                                self.logger.info(f"✅ Завантажено {len(models_metadata)} моделей з метаданими")
                    except Exception as e:
                        self.logger.error(f"❌ Помилка завантаження {stage_5_file}: {e}")
                else:
                    self.logger.warning(f"⚠️ Файл не знайдено: {stage_5_file}")
            else:
                self.logger.warning(f"⚠️ Не вдалося знайти batch_name")

        if not predictions:
            self.logger.warning("❌ No 'predictions' found in the data. Skipping trading execution.")
            return {}
        
        self.logger.info(f"📊 Received {len(predictions)} predictions")
            
        if not current_prices:
            self.logger.warning("No 'current_prices' found. Extracting from predictions...")
            # Витягуємо ціни з predictions якщо немає окремо
            current_prices = {}
            for pred in predictions:
                ticker = pred.get('ticker')
                last_price = pred.get('last_price')
                if ticker and last_price:
                    current_prices[ticker] = last_price
            
            if not current_prices:
                self.logger.error("❌ Cannot extract current_prices. Skipping trading execution.")
                return {}
        
        self.logger.info(f"💰 Current prices for {len(current_prices)} tickers")

        # Let the trading orchestrator handle the entire process
        try:
            self.trading_orchestrator.process_signals(
                raw_predictions=predictions,
                current_prices=current_prices
            )
            
            self.logger.info("✅ Trading execution completed successfully")
        except Exception as e:
            self.handle_stage_error(e, context="TradingOrchestration", severity="error")
            self.logger.error(f"❌ Trading execution failed: {e}", exc_info=True)
            # Even if trading failed, return predictions as signals for Stage 7
            return {
                'signals': predictions,
                'trading_error': str(e)
            }

        # The portfolio state is managed internally. We can add results to the data dict if needed.
        portfolio_summary = self.portfolio.get_portfolio_summary(current_prices)
        
        # ✅ FIX: VirtualPortfolio doesn't have get_trade_history(), use trade_log instead
        trade_history = self.portfolio.trade_log if hasattr(self.portfolio, 'trade_log') else []
        
        self.logger.info(f"📊 Trading summary: {len(trade_history)} trades, portfolio value: {portfolio_summary.get('total_value', 0):.2f}")
        
        # ✅ NEW: Generate analyzer recommendations
        # ✅ FIX: Передаємо models_metadata з kwargs
        models_metadata = kwargs.get('models_metadata', {})
        news_data = kwargs.get('news_data')  # ✅ Передаємо news_data
        analyzer_summary = self._generate_analyzer_recommendations(predictions, current_prices, portfolio_summary, models_metadata, news_data=news_data)
        
        # ✅ NEW: Збереження результатів Stage 6 на диск
        self._save_stage_6_results(
            predictions=predictions,
            current_prices=current_prices,
            portfolio_summary=portfolio_summary,
            trade_history=trade_history,
            analyzer_summary=analyzer_summary,
            kwargs=kwargs
        )
        
        return {
            'trading_activity': trade_history[-5:] if trade_history else [],
            'portfolio_summary': portfolio_summary,
            'signals': predictions,  # ✅ Завжди передаємо signals для Stage 7
            'analyzer_summary': analyzer_summary  # ✅ Додаємо рекомендації аналізаторів
        }
    
    def _save_stage_6_results(self, predictions: List[Dict], current_prices: Dict, portfolio_summary: Dict, trade_history: List, analyzer_summary: Dict, kwargs: Dict) -> None:
        """
        ✅ NEW: Збереження результатів Stage 6 на диск для гнучкого запуску.
        
        Зберігає stage_6_results.json у batch_dir для подальшого використання в Stage 7.
        """
        import json
        from pathlib import Path
        from datetime import datetime
        
        try:
            # Витягуємо batch_name
            batch_name = kwargs.get('batch_name')
            output_dir = Path(self.config_manager.get('system.accumulation.output_dir', 'data/colab/accumulated'))
            
            if not batch_name:
                # Шукаємо найновіший batch
                batch_dirs = list(output_dir.glob('test_ticker_*'))
                if batch_dirs:
                    batch_name = max(batch_dirs, key=lambda p: p.stat().st_mtime).name
            
            if batch_name:
                batch_dir = output_dir / batch_name
                batch_dir.mkdir(parents=True, exist_ok=True)
                
                # Підготовляємо дані для збереження
                stage_6_results = {
                    'timestamp': datetime.now().isoformat(),
                    'batch_name': batch_name,
                    'predictions': predictions,
                    'current_prices': current_prices,
                    'portfolio_summary': portfolio_summary,
                    'trade_history': trade_history,
                    'analyzer_summary': analyzer_summary,
                    'total_trades': len(trade_history),
                    'portfolio_value': portfolio_summary.get('total_value', 0)
                }
                
                # Зберігаємо на диск
                results_file = batch_dir / "stage_6_results.json"
                with open(results_file, 'w') as f:
                    json.dump(stage_6_results, f, indent=2, default=str)
                
                self.logger.info(f"✅ Результати Stage 6 збережені: {results_file.name}")
        except Exception as e:
            self.logger.warning(f"⚠️ Помилка збереження результатів Stage 6: {e}")
    
    def _generate_analyzer_recommendations(self, predictions: List[Dict], current_prices: Dict[str, float], portfolio_summary: Dict, models_metadata: Dict = None, **kwargs) -> Dict[str, Any]:
        """
        Генерує рекомендації на основі аналізу predictions, метрик моделей та поточного стану портфеля.
        
        Логіка вибору чемпіона:
        1. Важкі моделі порівнюються між собою (за accuracy/metrics)
        2. Легкі моделі порівнюються між собою (за accuracy/metrics)
        3. Найкраща важка порівнюється з найкращою легкою
        4. Чемпіон визначається для конкретного таргету
        
        Args:
            predictions: Список прогнозів від моделей
            current_prices: Поточні ціни тікерів
            portfolio_summary: Стан портфеля
            models_metadata: Метадані моделей з accuracy та metrics (передається з kwargs)
        
        Returns:
            Dict з рекомендаціями: buy_recommendations, sell_recommendations, risk_warnings, champion_model
        """
        self.logger.info("🔍 Generating analyzer recommendations with champion selection...")
        
        recommendations = {
            'buy_recommendations': [],
            'sell_recommendations': [],
            'risk_warnings': [],
            'champion_model': None,
            'champion_by_target': {},
            'heavy_models_ranking': [],
            'light_models_ranking': [],
            'model_rankings': [],
            'actor_critic_log': {
                'status': 'fallback',
                'reason': 'DEAN models not trained yet. Need trade history for Actor/Critic training.',
                'trade_count': len(self.portfolio.trade_log) if hasattr(self.portfolio, 'trade_log') else 0,
                'required_trades': 100  # Мінімум для тренування DEAN
            }
        }
        
        try:
            # ✅ 1. Завантажуємо models_metadata (передається через kwargs, не через brain)
            if models_metadata is None:
                models_metadata = {}
            
            # ✅ DEBUG: Логуємо що отримали
            self.logger.info(f"📊 models_metadata: {len(models_metadata)} моделей")
            if models_metadata:
                self.logger.info(f"📊 Перші 3 ключі: {list(models_metadata.keys())[:3]}")
                # Логуємо структуру першої моделі
                first_key = list(models_metadata.keys())[0]
                first_meta = models_metadata[first_key]
                self.logger.info(f"📊 Структура metadata для {first_key}:")
                self.logger.info(f"   - winner: {first_meta.get('winner')}")
                self.logger.info(f"   - ticker: {first_meta.get('ticker')}")
                self.logger.info(f"   - target: {first_meta.get('target')}")
                self.logger.info(f"   - metrics: {first_meta.get('metrics', {}).keys()}")
            
            if not models_metadata:
                self.logger.warning("⚠️ models_metadata не знайдено. Використовую fallback логіку.")
                return self._fallback_recommendations(predictions, current_prices, portfolio_summary)
            
            self.logger.info(f"📊 Завантажено {len(models_metadata)} моделей з метриками")
            
            # ✅ 2. Класифікуємо моделі на важкі та легкі
            HEAVY_MODELS = ["gru", "tabnet", "transformer", "cnn", "lstm", "autoencoder"]
            
            heavy_models = {}
            light_models = {}
            
            for context_id, meta in models_metadata.items():
                model_type = meta.get('winner', meta.get('model_type', ''))
                ticker = meta.get('ticker', '')
                target = meta.get('target', '')
                
                # Витягуємо accuracy з metrics
                # ✅ FIX: Для regression використовуємо r2, для classification - accuracy
                metrics = meta.get('metrics', {})  # ✅ FIXED: було model_meta
                
                # Визначаємо тип задачі
                is_regression = 'r2' in metrics or 'mse' in metrics
                
                if is_regression:
                    # Для regression використовуємо R2 як міру якості
                    accuracy = metrics.get('r2', metrics.get('score', 0.0))
                else:
                    # Для classification використовуємо accuracy
                    accuracy = metrics.get('accuracy', metrics.get('test_accuracy', metrics.get('score', 0.0)))
                
                model_info = {
                    'context_id': context_id,
                    'model_type': model_type,
                    'ticker': ticker,
                    'target': target,
                    'accuracy': accuracy,
                    'metrics': metrics,
                    'context_fingerprint': meta.get('context', 'unknown')
                }
                
                # Класифікуємо
                if any(heavy in model_type.lower() for heavy in HEAVY_MODELS):
                    key = f"{ticker}_{target}"
                    if key not in heavy_models:
                        heavy_models[key] = []
                    heavy_models[key].append(model_info)
                else:
                    key = f"{ticker}_{target}"
                    if key not in light_models:
                        light_models[key] = []
                    light_models[key].append(model_info)
            
            self.logger.info(f"📊 Класифіковано: {len(heavy_models)} груп важких моделей, {len(light_models)} груп легких моделей")
            
            # ✅ 3. Вибираємо найкращу модель для кожного таргету
            for target_key in set(list(heavy_models.keys()) + list(light_models.keys())):
                heavy_group = heavy_models.get(target_key, [])
                light_group = light_models.get(target_key, [])
                
                # Сортуємо за accuracy
                heavy_group.sort(key=lambda x: x['accuracy'], reverse=True)
                light_group.sort(key=lambda x: x['accuracy'], reverse=True)
                
                # Зберігаємо рейтинги
                if heavy_group:
                    recommendations['heavy_models_ranking'].extend(heavy_group)
                if light_group:
                    recommendations['light_models_ranking'].extend(light_group)
                
                # Вибираємо чемпіона: найкраща важка vs найкраща легка
                best_heavy = heavy_group[0] if heavy_group else None
                best_light = light_group[0] if light_group else None
                
                champion = None
                if best_heavy and best_light:
                    # Порівнюємо accuracy
                    if best_heavy['accuracy'] >= best_light['accuracy']:
                        champion = best_heavy
                        reason = f"Heavy model wins: {best_heavy['accuracy']:.4f} vs {best_light['accuracy']:.4f}"
                    else:
                        champion = best_light
                        reason = f"Light model wins: {best_light['accuracy']:.4f} vs {best_heavy['accuracy']:.4f}"
                elif best_heavy:
                    champion = best_heavy
                    reason = "Only heavy model available"
                elif best_light:
                    champion = best_light
                    reason = "Only light model available"
                
                if champion:
                    recommendations['champion_by_target'][target_key] = {
                        'model_type': champion['model_type'],
                        'accuracy': champion['accuracy'],
                        'ticker': champion['ticker'],
                        'target': champion['target'],
                        'reason': reason,
                        'context_fingerprint': champion['context_fingerprint']
                    }
                    
                    self.logger.info(f"🏆 Champion for {target_key}: {champion['model_type']} (accuracy: {champion['accuracy']:.4f})")
            
            # ✅ 4. Оновлюємо confidence в predictions на основі accuracy
            for pred in predictions:
                ticker = pred.get('ticker')
                # Витягуємо target з predictions_by_model ключів
                predictions_by_model = pred.get('predictions_by_model', {})
                if predictions_by_model:
                    # Формат ключа: model_ticker_target
                    first_key = list(predictions_by_model.keys())[0]
                    parts = first_key.split('_')
                    if len(parts) >= 3:
                        target = '_'.join(parts[2:])  # target може містити _
                        target_key = f"{ticker}_{target}"
                        
                        # Знаходимо чемпіона для цього таргету
                        champion_info = recommendations['champion_by_target'].get(target_key)
                        if champion_info:
                            # Оновлюємо confidence на основі accuracy
                            pred['confidence'] = champion_info['accuracy']
                            pred['champion_model'] = champion_info['model_type']
                            pred['context_fingerprint'] = champion_info['context_fingerprint']
            
            # ✅ 5. Генеруємо BUY/SELL рекомендації з корекцією на основі новин
            # Спочатку аналізуємо новини (якщо є)
            news_impact_scores = {}
            news_data = kwargs.get('news_data')  # Отримуємо з kwargs
            
            if news_data is not None and not news_data.empty:
                try:
                    self.logger.info(f"📰 Аналіз впливу новин на {len(news_data)} записів...")
                    news_analysis = self.news_impact_analyzer.analyze(news_data)
                    
                    if news_analysis and 'news_impact_scores' in news_analysis:
                        impact_scores = news_analysis['news_impact_scores']
                        significance_levels = news_analysis.get('news_significance_levels', {})
                        
                        # Конвертуємо в dict для швидкого доступу
                        for ticker in predictions:
                            ticker_name = ticker.get('ticker')
                            if ticker_name:
                                # Беремо останній impact score для тікера
                                ticker_scores = impact_scores[impact_scores.index.str.contains(ticker_name, case=False, na=False)]
                                if not ticker_scores.empty:
                                    news_impact_scores[ticker_name] = {
                                        'score': float(ticker_scores.iloc[-1]),
                                        'significance': str(significance_levels.iloc[-1]) if not significance_levels.empty else 'low'
                                    }
                        
                        self.logger.info(f"✅ News impact аналіз: {len(news_impact_scores)} тікерів з новинами")
                except Exception as e:
                    self.logger.warning(f"⚠️ Помилка аналізу новин: {e}")
            
            for pred in predictions:
                ticker = pred.get('ticker')
                pred_value = pred.get('predictions')
                
                # Конвертуємо prediction в число
                if isinstance(pred_value, (list, tuple)):
                    pred_value = float(pred_value[-1]) if len(pred_value) > 0 else 0.0
                else:
                    pred_value = float(pred_value) if pred_value is not None else 0.0
                
                confidence = pred.get('confidence', 0.5)
                champion_model = pred.get('champion_model', 'unknown')
                
                # ✅ Корекція на основі новин
                news_adjustment = 1.0
                news_warning = None
                
                if ticker in news_impact_scores:
                    news_info = news_impact_scores[ticker]
                    news_score = news_info['score']
                    news_significance = news_info['significance']
                    
                    # Негативні новини зменшують confidence та розмір позиції
                    if news_score < -0.3 and news_significance in ['high', 'medium']:
                        news_adjustment = 0.5  # Зменшуємо на 50%
                        news_warning = f"Negative news impact: {news_score:.2f} ({news_significance})"
                        self.logger.warning(f"⚠️ {ticker}: {news_warning}")
                    # Позитивні новини збільшують confidence
                    elif news_score > 0.3 and news_significance in ['high', 'medium']:
                        news_adjustment = 1.2  # Збільшуємо на 20%
                        self.logger.info(f"✅ {ticker}: Positive news boost: {news_score:.2f} ({news_significance})")
                
                # Застосовуємо корекцію
                adjusted_confidence = confidence * news_adjustment
                
                # BUY якщо прогноз позитивний
                if pred_value > 0.01:
                    recommendations['buy_recommendations'].append({
                        'ticker': ticker,
                        'predicted_return': pred_value,
                        'current_price': current_prices.get(ticker),
                        'confidence': adjusted_confidence,
                        'original_confidence': confidence,
                        'news_adjustment': news_adjustment,
                        'news_warning': news_warning,
                        'champion_model': champion_model,
                        'reason': f"Positive prediction: {pred_value:.4f} (confidence: {adjusted_confidence:.2f})"
                    })
                # SELL якщо прогноз негативний
                elif pred_value < -0.01:
                    recommendations['sell_recommendations'].append({
                        'ticker': ticker,
                        'predicted_return': pred_value,
                        'current_price': current_prices.get(ticker),
                        'confidence': adjusted_confidence,
                        'original_confidence': confidence,
                        'news_adjustment': news_adjustment,
                        'news_warning': news_warning,
                        'champion_model': champion_model,
                        'reason': f"Negative prediction: {pred_value:.4f} (confidence: {adjusted_confidence:.2f})"
                    })
            
            # ✅ 6. Перевіряємо ризики портфеля
            total_value = portfolio_summary.get('total_value', 0)
            cash = portfolio_summary.get('cash', 0)
            positions = portfolio_summary.get('positions', [])
            
            if total_value > 0 and cash < total_value * 0.1:
                recommendations['risk_warnings'].append({
                    'type': 'low_cash',
                    'message': f"Low cash reserves: ${cash:.2f} ({cash/total_value*100:.1f}% of portfolio)",
                    'severity': 'medium'
                })
            
            if len(positions) > 10:
                recommendations['risk_warnings'].append({
                    'type': 'over_diversification',
                    'message': f"Too many positions: {len(positions)}. Consider consolidation.",
                    'severity': 'low'
                })
            
            # ✅ 7. Загальний чемпіон (найкраща модель серед усіх таргетів)
            all_champions = list(recommendations['champion_by_target'].values())
            if all_champions:
                overall_champion = max(all_champions, key=lambda x: x['accuracy'])
                recommendations['champion_model'] = overall_champion['model_type']
                recommendations['champion_accuracy'] = overall_champion['accuracy']
            
            # ✅ 8. Додаємо Context Map та Market Signals
            # Витягуємо з models_metadata
            context_maps = {}
            market_signals = {
                'regimes': {},
                'volatility': {},
                'trends': {}
            }
            
            for context_id, meta in models_metadata.items():
                context_map = meta.get('context_map', {})
                if context_map:
                    ticker = meta.get('ticker', 'unknown')
                    context_maps[ticker] = context_map
                    
                    # Збираємо market signals
                    market_regime = context_map.get('market_regime', 'neutral')
                    volatility_regime = context_map.get('volatility_regime', 'normal')
                    
                    market_signals['regimes'][ticker] = market_regime
                    market_signals['volatility'][ticker] = volatility_regime
            
            recommendations['context_maps'] = context_maps
            recommendations['market_signals'] = market_signals
            recommendations['news_impact'] = news_impact_scores  # ✅ Додаємо news impact
            
            # ✅ 9. Causal Analysis - пояснення прогнозів
            try:
                # Підготуємо дані для causal analysis
                causal_data = pd.DataFrame(predictions)
                
                # Додаємо signal колонку (1 для BUY, -1 для SELL, 0 для HOLD)
                def get_signal(pred):
                    if isinstance(pred, dict):
                        pred_value = pred.get('predictions', 0)
                    else:
                        pred_value = pred
                    
                    if isinstance(pred_value, (list, tuple)):
                        pred_value = pred_value[-1] if len(pred_value) > 0 else 0
                    
                    return 1 if pred_value > 0.01 else (-1 if pred_value < -0.01 else 0)
                
                causal_data['signal'] = causal_data.apply(lambda row: get_signal(row.get('predictions', 0)), axis=1)
                causal_data['predicted_return'] = causal_data['predictions'].apply(
                    lambda x: float(x[-1]) if isinstance(x, (list, tuple)) and len(x) > 0 else float(x) if isinstance(x, (int, float)) else 0.0
                )
                
                # Запускаємо causal analysis
                if len(causal_data) >= 10:  # Мінімум 10 рядків для аналізу
                    causal_result = self.causal_finder.analyze(causal_data)
                    
                    recommendations['causal_analysis'] = {
                        'causal_effect': causal_result.get('causal_effect', 0.0),
                        'status': causal_result.get('status', 'unknown'),
                        'explanation': f"Causal effect of signal on predicted return: {causal_result.get('causal_effect', 0.0):.4f}"
                    }
                    
                    self.logger.info(f"🔍 Causal analysis: effect={causal_result.get('causal_effect', 0.0):.4f}")
                else:
                    recommendations['causal_analysis'] = {
                        'status': 'insufficient_data',
                        'explanation': 'Not enough predictions for causal analysis'
                    }
            except Exception as e:
                self.logger.warning(f"⚠️ Causal analysis failed: {e}")
                recommendations['causal_analysis'] = {
                    'status': 'error',
                    'error': str(e)
                }
            
            self.logger.info(f"✅ Generated {len(recommendations['buy_recommendations'])} BUY and {len(recommendations['sell_recommendations'])} SELL recommendations")
            if recommendations['champion_model']:
                self.logger.info(f"🏆 Overall champion: {recommendations['champion_model']} (accuracy: {recommendations.get('champion_accuracy', 0):.4f})")
            self.logger.info(f"🗺️ Context maps: {len(context_maps)} tickers")
            self.logger.info(f"📊 Market signals: {market_signals['regimes']}")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to generate analyzer recommendations: {e}", exc_info=True)
            recommendations['error'] = str(e)
            # Fallback до простої логіки
            return self._fallback_recommendations(predictions, current_prices, portfolio_summary)
        
        return recommendations
    
    def _fallback_recommendations(self, predictions: List[Dict], current_prices: Dict[str, float], portfolio_summary: Dict) -> Dict[str, Any]:
        """
        Fallback логіка коли models_metadata недоступна.
        Використовує просту логіку на основі сили сигналу.
        """
        self.logger.info("🔄 Using fallback recommendation logic...")
        
        recommendations = {
            'buy_recommendations': [],
            'sell_recommendations': [],
            'risk_warnings': [],
            'champion_model': None,
            'model_rankings': [],
            'fallback_mode': True
        }
        
        # Простий вибір на основі сили сигналу
        model_scores = {}
        for pred in predictions:
            predictions_by_model = pred.get('predictions_by_model', {})
            for model_name, pred_value in predictions_by_model.items():
                try:
                    pred_float = float(pred_value)
                    score = abs(pred_float)
                    if model_name not in model_scores:
                        model_scores[model_name] = []
                    model_scores[model_name].append(score)
                except (ValueError, TypeError):
                    continue
        
        model_rankings = []
        for model_name, scores in model_scores.items():
            avg_score = sum(scores) / len(scores) if scores else 0.0
            model_rankings.append({
                'model': model_name,
                'avg_signal_strength': avg_score,
                'predictions_count': len(scores)
            })
        
        model_rankings.sort(key=lambda x: x['avg_signal_strength'], reverse=True)
        recommendations['model_rankings'] = model_rankings
        
        if model_rankings:
            recommendations['champion_model'] = model_rankings[0]['model']
        
        # Генеруємо рекомендації
        for pred in predictions:
            ticker = pred.get('ticker')
            pred_value = pred.get('predictions')
            
            if isinstance(pred_value, (list, tuple)):
                pred_value = float(pred_value[-1]) if len(pred_value) > 0 else 0.0
            else:
                pred_value = float(pred_value) if pred_value is not None else 0.0
            
            if pred_value > 0.01:
                recommendations['buy_recommendations'].append({
                    'ticker': ticker,
                    'predicted_return': pred_value,
                    'current_price': current_prices.get(ticker),
                    'confidence': 0.5,
                    'reason': f"Positive prediction: {pred_value:.4f}"
                })
            elif pred_value < -0.01:
                recommendations['sell_recommendations'].append({
                    'ticker': ticker,
                    'predicted_return': pred_value,
                    'current_price': current_prices.get(ticker),
                    'confidence': 0.5,
                    'reason': f"Negative prediction: {pred_value:.4f}"
                })
        
        return recommendations
