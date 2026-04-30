# src/pipeline/stages/stage_6_trading_execution.py

"""
Stage 6: Trading Execution

This stage takes the final predictions from Stage 5 and orchestrates the entire
trading process using the refactored trading module.
"""

import pandas as pd
import numpy as np
import json
from typing import Dict, Any, List, Optional
from datetime import datetime
from pathlib import Path

from src.pipeline.stages.base_stage import BaseStage
from src.core.logging.logger import ProjectLogger

# Import all the refactored trading components
from src.core.utils.prediction_utils import normalize_prediction
from src.trading.virtual_portfolio import VirtualPortfolio
from src.trading.consensus_engine import ConsensusEngine, EnhancedConsensusEngine
from src.trading.post_inference_filter import PostInferenceFilter
from src.trading.portfolio_manager import PortfolioManager
from src.trading.trader import Trader
from src.trading.trading_orchestrator import TradingOrchestrator
from src.analytics.analyzers.news_impact_analyzer import NewsImpactAnalyzer
from src.analytics.analyzers.causal_event_finder import CausalEventFinder
from src.algorithms.regime_detector import MarketRegimeDetector
from src.simulation.simulation_engine import SimulationEngine
from src.calibration.confidence_calibrator import ConfidenceCalibrator
from src.trading.adaptive_parameter_manager import AdaptiveParameterManager
from src.calibration.adaptive_confidence_calibrator import AdaptiveConfidenceCalibrator
from src.trading.elite_risk_sizer import EliteRiskSizer
from src.risk.elite_risk_metrics import EliteRiskMetrics
from src.trading.live_adaptive_ensemble import LiveAdaptiveEnsemble
from src.meta_learning.memory.diary_engine import DiaryEngine
from src.analytics.analyzers.adaptive_confidence_analyzer import AdaptiveConfidenceAnalyzer

class TradingExecutionStage(BaseStage):
    """
    A pipeline stage to execute the trading logic.
    """
    def __init__(self, config_manager, error_handler, **kwargs):
        super().__init__(config_manager, error_handler, **kwargs)
        self.logger = ProjectLogger.get_logger(self.__class__.__name__)
        self.news_impact_analyzer = NewsImpactAnalyzer(config=self.config_manager.get('analysis.news_impact', {}))
        self.causal_finder = CausalEventFinder(treatment='signal', outcome='predicted_return', common_causes=['confidence', 'anomaly_score'])
        # Move _initialize_trading_stack call to run() to avoid issues during orchestration setup

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
        self.diary_engine = DiaryEngine()
        self.threshold_analyzer = AdaptiveConfidenceAnalyzer(config=self.config_manager.get('analysis.adaptive_confidence', {}))

        self.consensus_engine = ConsensusEngine(
            experience_diary=self.diary_engine,
            threshold_analyzer=self.threshold_analyzer,
            config_manager=self.config_manager
        )
        self.logger.info("Initialized ConsensusEngine.")

        # EnhancedConsensusEngine for regime-based weighting
        self.enhanced_consensus = EnhancedConsensusEngine()
        self.logger.info("Initialized EnhancedConsensusEngine.")

        # SimulationEngine for Monte Carlo validation
        self.simulation_engine = SimulationEngine()
        self.logger.info("Initialized SimulationEngine for Monte Carlo validation.")

        # ConfidenceCalibrator for confidence calibration
        self.confidence_calibrator = ConfidenceCalibrator(logger=self.logger)
        calibration_path = self.config_manager.get('paths.models', 'data/trained_models') + '/confidence_calibrator.pkl'
        
        if self.confidence_calibrator.load(calibration_path):
            self.logger.info("✅ Loaded existing confidence calibration model")
        else:
            try:
                if self.confidence_calibrator.calibrate_on_history(self.diary_engine, window=500):
                    self.confidence_calibrator.save(calibration_path)
                    self.logger.info("✅ Trained new confidence calibration model")
                else:
                    self.logger.warning("⚠️ Could not train confidence calibration model")
            except Exception as e:
                self.logger.warning(f"⚠️ Confidence calibration training failed: {e}")

        # ELITE COMPONENT #1: AdaptiveParameterManager
        self.param_manager = AdaptiveParameterManager(logger=self.logger)
        self.logger.info("Initialized AdaptiveParameterManager (elite-grade parameters).")

        # ELITE COMPONENT #2: AdaptiveConfidenceCalibrator
        self.adaptive_confidence_calibrator = AdaptiveConfidenceCalibrator(logger=self.logger)
        adaptive_calib_path = self.config_manager.get('paths.models', 'data/trained_models') + '/adaptive_confidence_calibrator.pkl'
        
        if self.adaptive_confidence_calibrator.load(adaptive_calib_path):
            self.logger.info("✅ Loaded existing adaptive confidence calibration model")
        else:
            self.logger.info("ℹ️ Adaptive confidence calibrator initialized (will learn from new trades)")
        self.logger.info("Initialized AdaptiveConfidenceCalibrator (elite-grade confidence).")

        # ELITE COMPONENT #3: EliteRiskSizer
        self.risk_sizer = EliteRiskSizer(logger=self.logger)
        self.logger.info("Initialized EliteRiskSizer (Kelly + correlation-aware sizing).")

        # ELITE COMPONENT #4: EliteRiskMetrics
        self.risk_metrics = EliteRiskMetrics(logger=self.logger)
        self.logger.info("Initialized EliteRiskMetrics (GARCH + Cornish-Fisher VaR).")

        # ELITE COMPONENT #5: LiveAdaptiveEnsemble
        self.live_ensemble = LiveAdaptiveEnsemble(logger=self.logger)
        self.logger.info("Initialized LiveAdaptiveEnsemble (dynamic model weighting).")

        # Initialize Market Regime Detector
        self.regime_detector = MarketRegimeDetector()
        self.logger.info("Initialized MarketRegimeDetector.")

        # 4. Initialize the risk officer: Portfolio Manager
        self.portfolio_manager = PortfolioManager(
            virtual_portfolio=self.portfolio,
            elite_risk_sizer=self.risk_sizer,
            config=self.config_manager.get('strategy.risk_management', {})
        )
        self.logger.info("Initialized PortfolioManager with EliteRiskSizer.")

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
        Runs the full trading cycle.
        """
        # Initialize trading stack if not already done
        if not hasattr(self, 'trading_orchestrator'):
            self._initialize_trading_stack()
        
        # Load predictions and pricesting trading execution stage...
        self.logger.info("Starting trading execution stage...")

        predictions, current_prices = await self._load_or_extract_data(kwargs)
        
        if not predictions:
            self.logger.warning("❌ No 'predictions' found in the data. Skipping trading execution.")
            return {}
        
        trading_result = self._execute_trading(predictions, current_prices)
        if 'trading_error' in trading_result:
            return trading_result
        
        return self._finalize_results(predictions, current_prices, kwargs)

    async def _load_or_extract_data(self, kwargs: Dict) -> tuple:
        """Load predictions and current prices from kwargs or disk."""
        predictions = kwargs.get('predictions')
        current_prices = kwargs.get('current_prices')

        if not predictions:
            load_result = await self._load_predictions_from_disk(kwargs)
            if load_result and len(load_result) >= 2:
                predictions, current_prices, _ = load_result
            else:
                return None, None
        
        if not predictions:
            return None, None
        
        self.logger.info(f"📊 Received {len(predictions)} predictions")
            
        if not current_prices:
            current_prices = self._extract_current_prices(predictions)
            if not current_prices:
                return None, None
        
        self.logger.info(f"💰 Current prices for {len(current_prices)} tickers")
        return predictions, current_prices

    async def _load_predictions_from_disk(self, kwargs: Dict) -> tuple:
        """Load predictions from disk when not provided in kwargs."""
        self.logger.warning("⚠️ No 'predictions' found in kwargs. Attempting to load from disk...")
        
        batch_name = kwargs.get('batch_name')
        output_dir = Path(self.config_manager.get('system.accumulation.output_dir', 'data/colab/accumulated'))
        
        if not batch_name:
            batch_dirs = list(output_dir.glob('test_ticker_*'))
            if batch_dirs:
                batch_name = max(batch_dirs, key=lambda p: p.stat().st_mtime).name
                self.logger.info(f"🔍 Found latest batch: {batch_name}")
        
        if batch_name:
            return await self._process_batch_file(batch_name, output_dir, kwargs)
        
        self.logger.warning("⚠️ Could not find batch_name")
        return None, None, kwargs

    async def _process_batch_file(self, batch_name: str, output_dir: Path, kwargs: Dict) -> tuple:
        """Process batch file for loading."""
        batch_dir = output_dir / batch_name
        stage_5_file = batch_dir / "stage_5_results.json"
        
        if not stage_5_file.exists():
            # Try with potential typo if needed, but primary is results.json
            self.logger.warning(f"⚠️ File not found: {stage_5_file}")
            return None, None, kwargs
            
        try:
            content = await self._read_file_async(stage_5_file)
            stage_5_results = json.loads(content)
            
            predictions = stage_5_results.get('predictions', [])
            current_prices = stage_5_results.get('current_prices', {})
            
            self.logger.info(f"✅ Loaded {len(predictions)} forecasts from {stage_5_file.name}")
            self.logger.info(f"✅ Loaded prices for {len(current_prices)} tickers")
            
            if 'models_metadata' not in kwargs:
                models_metadata = stage_5_results.get('models_metadata', {})
                if models_metadata:
                    kwargs['models_metadata'] = models_metadata
                    self.logger.info(f"✅ Loaded {len(models_metadata)} models with metadata")
            
            return predictions, current_prices, kwargs
        except Exception as e:
            self.logger.error(f"❌ Error loading {stage_5_file}: {e}")
            return None, None, kwargs

    async def _read_file_async(self, file_path: Path) -> str:
        """Read a file asynchronously."""
        import aiofiles
        async with aiofiles.open(file_path, mode='r', encoding='utf-8') as f:
            return await f.read()

    def _extract_current_prices(self, predictions: List[Dict]) -> Optional[Dict]:
        """Extract current prices from predictions when not provided."""
        self.logger.warning("No 'current_prices' found. Extracting from predictions...")
        current_prices = {}
        for pred in predictions:
            ticker = pred.get('ticker')
            last_price = pred.get('last_price')
            if ticker and last_price:
                current_prices[ticker] = last_price
        
        if not current_prices:
            self.logger.error("❌ Cannot extract current_prices. Skipping trading execution.")
            return None
        
        return current_prices

    def _execute_trading(self, predictions: List[Dict], current_prices: Dict) -> Dict[str, Any]:
        """Execute trading using the trading orchestrator."""
        try:
            self.trading_orchestrator.process_signals(
                raw_predictions=predictions,
                current_prices=current_prices
            )
            self.logger.info("✅ Trading execution completed successfully")
            return {}
        except Exception as e:
            self.handle_stage_error(e, context="TradingOrchestration", severity="error")
            self.logger.error(f"❌ Trading execution failed: {e}", exc_info=True)
            return {
                'signals': predictions,
                'trading_error': str(e)
            }

    def _finalize_results(self, predictions: List[Dict], current_prices: Dict, kwargs: Dict) -> Dict[str, Any]:
        """Finalize and return trading results."""
        portfolio_summary = self.portfolio.get_portfolio_summary(current_prices)
        trade_history = getattr(self.portfolio, 'transactions', [])
        
        self.logger.info(f"📊 Trading summary: {len(trade_history)} trades, portfolio value: {portfolio_summary.get('total_value', 0):.2f}")
        
        models_metadata = kwargs.get('models_metadata', {})
        kwargs_pass = kwargs.copy()
        kwargs_pass.pop('models_metadata', None)
        kwargs_pass.pop('predictions', None)
        kwargs_pass.pop('current_prices', None)
        analyzer_summary = self._generate_analyzer_recommendations(predictions, current_prices, models_metadata, **kwargs_pass)
        
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
            'signals': predictions,
            'analyzer_summary': analyzer_summary
        }
    
    def _save_stage_6_results(self, predictions: List[Dict], current_prices: Dict, portfolio_summary: Dict, trade_history: List, analyzer_summary: Dict, kwargs: Dict) -> None:
        """Saves Stage 6 results to disk for flexible runs."""
        try:
            batch_name = kwargs.get('batch_name')
            output_dir = Path(self.config_manager.get('system.accumulation.output_dir', 'data/colab/accumulated'))
            
            if not batch_name:
                batch_dirs = list(output_dir.glob('test_ticker_*'))
                if batch_dirs:
                    batch_name = max(batch_dirs, key=lambda p: p.stat().st_mtime).name
            
            if batch_name:
                batch_dir = output_dir / batch_name
                batch_dir.mkdir(parents=True, exist_ok=True)
                
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
                
                results_file = batch_dir / "stage_6_results.json"
                with open(results_file, 'w', encoding='utf-8') as f:
                    json.dump(stage_6_results, f, indent=2, default=str)
                
                self.logger.info(f"✅ Stage 6 results saved: {results_file.name}")
        except Exception as e:
            self.logger.warning(f"⚠️ Error saving Stage 6 results: {e}")

    def _calculate_robust_confidence(self, ticker, target_key, models_metadata, predictions):
        """Calculates robust confidence score."""
        try:
            champion_data = self._find_champion_model(ticker, target_key, models_metadata)
            base_confidence = self._calculate_base_confidence(champion_data)
            error_ratios = self._calculate_error_ratios(champion_data)
            anomaly_penalty = self._calculate_anomaly_penalty(ticker, predictions)
            consensus_boost = self._calculate_consensus_boost(predictions)
            
            confidence = (base_confidence * 0.4 + error_ratios['mae_ratio'] * 0.3 + error_ratios['rmse_ratio'] * 0.2) * (1 - anomaly_penalty) + consensus_boost
            calibrated_confidence = self.adaptive_confidence_calibrator.calibrate(confidence)
            
            return max(0.01, min(1.0, calibrated_confidence))
        
        except Exception as e:
            self.logger.warning(f"Error calculating robust confidence: {e}")
            return 0.3

    def _find_champion_model(self, ticker: str, target_key: str, models_metadata: Dict) -> Dict:
        """Find the champion model for the given ticker and target."""
        heavy_models_list = []
        
        for context_id, meta in models_metadata.items():
            if meta.get('target') == target_key.split('_', 1)[1]:
                ticker_from_meta = meta.get('ticker', '')
                if ticker_from_meta == ticker:
                    test_metrics = meta.get('test_metrics', {})
                r2 = test_metrics.get('r2', 0.0)
                model_type = meta.get('model_type', '')
                
                heavy_models_list.append({'type': model_type, 'r2': r2, 'meta': meta})
        
        all_models = sorted(heavy_models_list, key=lambda x: x['r2'], reverse=True)
        return all_models[0] if all_models else None

    def _calculate_base_confidence(self, champion_data: Optional[Dict]) -> float:
        """Calculate base confidence from champion model data."""
        if not champion_data:
            return 0.3
        
        r2 = champion_data.get('r2', 0.0)
        if r2 < -2:
            return 0.1
        elif r2 < 0:
            return 0.2 + (r2 / 2) * 0.2 
        else:
            return 0.3 + r2 * 0.7 

    def _calculate_error_ratios(self, champion_data: Dict) -> Dict[str, float]:
        """Calculate error ratios from champion model data."""
        test_metrics = champion_data['meta'].get('test_metrics', {}) if champion_data else {}
        rmse = test_metrics.get('rmse', 1.0)
        mae = test_metrics.get('mae', 1.0)
        mae_ratio = 1.0 / (1.0 + mae * 2)
        rmse_ratio = 1.0 / (1.0 + rmse * 2)
        
        return {'mae_ratio': mae_ratio, 'rmse_ratio': rmse_ratio}

    def _calculate_anomaly_penalty(self, ticker: str, predictions: List[Dict]) -> float:
        """Calculate anomaly penalty from predictions."""
        anomaly_score = 0.5 
        for pred in predictions:
            if pred.get('ticker') == ticker:
                anomaly_score = pred.get('anomaly_score', 0.5)
                break
        
        return anomaly_score * 0.2

    def _calculate_consensus_boost(self, predictions: List[Dict]) -> float:
        """Calculate consensus boost from predictions."""
        positive_votes = len([p for p in predictions if p.get('predictions', 0) > 0.001])
        total_models = len(predictions)
        return (positive_votes / max(total_models, 1)) * 0.1

    def _get_signal_strength(self, confidence):
        """Determines signal strength based on confidence."""
        if confidence >= 0.7: return "very_strong"
        elif confidence >= 0.6: return "strong"
        elif confidence >= 0.5: return "medium"
        elif confidence >= 0.35: return "weak"
        else: return "very_weak"

    def _calculate_risk_score(self, rec):
        """Calculates risk score for a position."""
        base_risk = 0.5
        confidence_risk = (1 - rec['confidence']) * 0.3
        news_risk = 0.1 if rec.get('news_warning') else 0.0
        return max(0.0, min(1.0, base_risk + confidence_risk + news_risk))

    def _validate_with_monte_carlo(self, ticker):
        """Advanced risk validation using EliteRiskMetrics."""
        try:
            var_hist = self.risk_metrics.compute_historical_simulation_var(ticker, confidence_level=0.95, lookback_days=252)
            var_garch = self.risk_metrics.compute_garch_var(ticker, confidence_level=0.95)
            var_cf, _ = self.risk_metrics.compute_cornish_fisher_var(ticker, confidence_level=0.95, lookback_days=252)
            
            var_95 = 0.4 * var_hist + 0.35 * var_garch + 0.25 * var_cf
            var_threshold = 0.05 
            position_size_factor = 1.0
            
            if var_95 > var_threshold:
                excess_var = var_95 - var_threshold
                reduction = (excess_var / var_threshold) * 0.5
                position_size_factor = max(0.1, 1.0 - reduction)
                self.logger.warning(f"⚠️ Elite VaR {var_95:.3f} > {var_threshold:.3f} for {ticker}, factor reduced to {position_size_factor:.1%}")
            
            return 0.5 * position_size_factor, var_95, position_size_factor
        except Exception as e:
            self.logger.warning(f"⚠️ Elite risk validation failed for {ticker}: {e}")
            return 0.5, 0.03, 1.0

    def _create_consolidated_table(self, buy_recs, sell_recs, models_metadata):
        """Creates a consolidated table of all recommendations."""
        consolidated = []
        
        for rec in buy_recs:
            ticker = rec['ticker']
            supporting_models = [m for m in models_metadata.values() if m.get('ticker') == ticker]
            positive_count = len([m for m in supporting_models if m.get('prediction', 0) > 0.01])
            
            consolidated.append({
                "ticker": ticker,
                "action": "BUY",
                "priority": 1,
                "confidence": rec['confidence'],
                "signal_strength": self._get_signal_strength(rec['confidence']),
                "predicted_return": rec['predicted_return'],
                "expected_return_pct": f"{rec['predicted_return']*100:.2f}%",
                "current_price": rec['current_price'],
                "champion_model": rec['champion_model'],
                "supporting_models_count": positive_count,
                "total_models": len(supporting_models),
                "model_consensus": positive_count / max(len(supporting_models), 1),
                "risk_score": self._calculate_risk_score(rec),
                "news_warning": rec.get('news_warning'),
                "var_95": rec.get('var_95', 0.03),
                "position_size_factor": rec.get('position_size_factor', 1.0),
                "reason": rec['reason'],
                "composite_score": rec['confidence'] * (positive_count / max(len(supporting_models), 1)),
                "timestamp": datetime.now().isoformat()
            })
        
        for rec in sell_recs:
            ticker = rec['ticker']
            supporting_models = [m for m in models_metadata.values() if m.get('ticker') == ticker]
            negative_count = len([m for m in supporting_models if m.get('prediction', 0) < -0.01])
            
            consolidated.append({
                "ticker": ticker,
                "action": "SELL",
                "priority": 3,
                "confidence": rec['confidence'],
                "signal_strength": self._get_signal_strength(rec['confidence']),
                "predicted_return": rec['predicted_return'],
                "expected_return_pct": f"{rec['predicted_return']*100:.2f}%",
                "current_price": rec['current_price'],
                "champion_model": rec['champion_model'],
                "supporting_models_count": negative_count,
                "total_models": len(supporting_models),
                "model_consensus": negative_count / max(len(supporting_models), 1),
                "risk_score": self._calculate_risk_score(rec),
                "news_warning": rec.get('news_warning'),
                "var_95": rec.get('var_95', 0.03),
                "position_size_factor": rec.get('position_size_factor', 1.0),
                "reason": rec['reason'],
                "composite_score": rec['confidence'] * (negative_count / max(len(supporting_models), 1)),
                "timestamp": datetime.now().isoformat()
            })
        
        consolidated.sort(key=lambda x: x['composite_score'], reverse=True)
        return consolidated

    def _generate_analyzer_recommendations(self, predictions: List[Dict], current_prices: Dict[str, float], models_metadata: Dict, **kwargs) -> Dict[str, Any]:
        """Generates recommendations based on predictions and model metrics."""
        self.logger.info("🔍 Generating analyzer recommendations...")
        
        recommendations = self._initialize_recommendations_structure()
        
        try:
            if not models_metadata:
                self.logger.warning("⚠️ models_metadata not found. Using fallback logic.")
                return self._fallback_recommendations(predictions, current_prices)
            
            heavy_models, light_models = self._categorize_models(models_metadata)
            self._populate_champion_by_target(recommendations, heavy_models, light_models)
            self._enhance_predictions_with_confidence(predictions, recommendations, models_metadata)
            
            news_impact_scores = self._analyze_news_impact(kwargs) 
            self._generate_trading_recommendations(predictions, current_prices, recommendations, news_impact_scores)
            recommendations['consolidated_table'] = self._create_consolidated_table(
                recommendations['buy_recommendations'], recommendations['sell_recommendations'], models_metadata
            )
            
        except Exception as e:
            self.logger.error(f"❌ Failed to generate recommendations: {e}", exc_info=True)
            return self._fallback_recommendations(predictions, current_prices)
        
        return recommendations

    def _initialize_recommendations_structure(self) -> Dict[str, Any]:
        """Initialize the recommendations dictionary structure."""
        return {
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
                'reason': 'DEAN models not trained yet. Need more history.',
                'trade_count': len(getattr(self.portfolio, 'transactions', []))
            }
        }

    def _categorize_models(self, models_metadata: Dict) -> tuple:
        """Categorize models into heavy and light models."""
        HEAVY_MODELS = {"gru", "tabnet", "transformer", "cnn", "lstm", "autoencoder"}
        heavy_models = {}
        light_models = {}
        
        for context_id, meta in models_metadata.items():
            model_type = meta.get('winner', meta.get('model_type', '')).lower()
            ticker = meta.get('ticker', '')
            target = meta.get('target', '')
            metrics = meta.get('metrics') or {}
            is_regression = isinstance(metrics, dict) and ('r2' in metrics or 'mse' in metrics)
            accuracy = metrics.get('r2', metrics.get('score', 0.0)) if is_regression else metrics.get('accuracy', metrics.get('score', 0.0))
            
            model_info = {
                'context_id': context_id,
                'model_type': model_type,
                'ticker': ticker,
                'target': target,
                'accuracy': accuracy,
                'metrics': metrics
            }
            
            key = f"{ticker}_{target}"
            if any(heavy in model_type for heavy in HEAVY_MODELS):
                heavy_models.setdefault(key, []).append(model_info)
            else:
                light_models.setdefault(key, []).append(model_info)
        
        return heavy_models, light_models

    def _get_champion_model_for_target(self, target_key, heavy_models, light_models):
        """Helper to get champion model for a target."""
        combined_group = heavy_models.get(target_key, []) + light_models.get(target_key, [])
        if not combined_group:
            return None
        
        return max(combined_group, key=lambda x: x['accuracy'])

    def _populate_champion_by_target(self, recommendations: Dict, heavy_models: Dict, light_models: Dict) -> None:
        """Populate champion by target recommendations."""
        regime = 'ranging'  # Default
        all_targets = set(heavy_models.keys()) | set(light_models.keys())
        for target_key in all_targets:
            champion = self._get_champion_model_for_target(target_key, heavy_models, light_models)
            if champion:
                recommendations['champion_by_target'][target_key] = {
                    'model_type': 'live_adaptive_ensemble',
                    'regime': regime,
                    'ticker': champion['ticker'],
                    'accuracy': champion['accuracy']
                }

    def _enhance_predictions_with_confidence(self, predictions: List[Dict], recommendations: Dict, models_metadata: Dict) -> None:
        """Enhance predictions with confidence calculations."""
        for pred in predictions:
            ticker = pred.get('ticker')
            pred_by_model = pred.get('predictions_by_model', {})
            if pred_by_model:
                first_key = list(pred_by_model.keys())[0]
                parts = first_key.split('_')
                if len(parts) >= 3:
                    target = '_'.join(parts[2:])
                    target_key = f"{ticker}_{target}"
                    ensemble_info = recommendations['champion_by_target'].get(target_key)
                    if ensemble_info:
                        pred['confidence'] = self._calculate_robust_confidence(ticker, target_key, models_metadata, [pred])
                        pred['champion_model'] = 'ensemble'

    def _analyze_news_impact(self, news_data: Any) -> Dict:
        """Analyze news impact for predictions."""
        news_impact_scores = {}
        if news_data is not None and hasattr(news_data, 'empty') and not news_data.empty:
            try:
                news_analysis = self.news_impact_analyzer.analyze(news_data)
                if news_analysis and 'news_impact_scores' in news_analysis:
                    return news_analysis['news_impact_scores']
            except Exception as e:
                self.logger.warning(f"News impact analysis error: {e}")
        return news_impact_scores

    def _generate_trading_recommendations(self, predictions: List[Dict], current_prices: Dict, recommendations: Dict, news_impact_scores: Dict) -> None:
        """Generate buy and sell recommendations based on predictions."""
        regime = 'ranging'
        
        for pred in predictions:
            ticker = pred.get('ticker')
            if not ticker:
                continue
                
            asset_class = self._determine_asset_class(ticker)
            adaptive_params = self.param_manager.compute_adaptive_params(
                regime=regime, 
                asset_class=asset_class, 
                volatility_percentile=50
            )
            
            pred_value = self._extract_prediction_value(pred)
            news_warning = self._check_news_warning(ticker, news_impact_scores)
            mc_confidence, var_95, pos_factor = self._validate_with_monte_carlo(ticker)
            
            recommendation = {
                'ticker': ticker, 
                'predicted_return': pred_value, 
                'current_price': current_prices.get(ticker),
                'confidence': mc_confidence, 
                'news_warning': news_warning, 
                'var_95': var_95, 
                'position_size_factor': pos_factor, 
                'champion_model': 'ensemble'
            }
            
            if pred_value > adaptive_params.buy_threshold:
                recommendation.update({'reason': 'Positive prediction'})
                recommendations['buy_recommendations'].append(recommendation)
            elif pred_value < adaptive_params.sell_threshold:
                recommendation.update({'reason': 'Negative prediction'})
                recommendations['sell_recommendations'].append(recommendation)
            else:
                pass

    def _extract_prediction_value(self, pred: Dict) -> float:
        """Extract, normalize and SCALE prediction value to percentage range."""
        predictions = pred.get('predictions', pred.get('prediction', 0))
        val = 0.0
        if isinstance(predictions, (list, np.ndarray)):
            val = float(predictions[-1])
        else:
            val = float(predictions)
        
        # Robust scaling: If value is absurdly large (e.g. > 100 or < -100), 
        # it's likely a raw output. Scale it to a reasonable percentage range (0-5%).
        # This is a safety measure for reporting.
        if abs(val) > 10:
            # Simple log-based scaling or sigmoid-like squash
            scaled_val = np.sign(val) * (np.log1p(abs(val)) / 100.0)
            return float(scaled_val)
            
        return normalize_prediction(val)

    def _check_news_warning(self, ticker: str, news_impact_scores: Dict) -> Optional[str]:
        """Check for news warnings."""
        if ticker in news_impact_scores and news_impact_scores[ticker].get('score', 0) < -0.3:
            return "Negative news impact"
        return None
    
    def _fallback_recommendations(self, predictions: List[Dict], current_prices: Dict[str, float]) -> Dict[str, Any]:
        """Simple fallback recommendation logic with robust scaling."""
        recommendations = {'buy_recommendations': [], 'sell_recommendations': [], 'risk_warnings': [], 'fallback_mode': True}
        for pred in predictions:
            ticker = pred.get('ticker')
            pred_value = self._extract_prediction_value(pred)
            
            if pred_value > 0.01:
                recommendations['buy_recommendations'].append({'ticker': ticker, 'predicted_return': pred_value, 'current_price': current_prices.get(ticker), 'confidence': 0.5, 'reason': 'Positive (Fallback)'})
            elif pred_value < -0.01:
                recommendations['sell_recommendations'].append({'ticker': ticker, 'predicted_return': pred_value, 'current_price': current_prices.get(ticker), 'confidence': 0.5, 'reason': 'Negative (Fallback)'})
        return recommendations
    
    def _determine_asset_class(self, ticker: str) -> str:
        """Determines asset class based on ticker."""
        return 'large_cap'

    def _detect_regime(self) -> str:
        """Detects market regime."""
        return 'ranging' # Simplified

    def _update_live_ensemble_performance(self):
        """Update live ensemble performance tracking."""
        try:
            # Get current timestamp for performance tracking
            import time
            current_time = time.time()
            
            # Update performance tracking
            self.last_performance_update = current_time
            self.current_performance_score = self._calculate_ensemble_performance_score()
            
            self.logger.info(f"✅ Ensemble performance updated: score={self.current_performance_score:.3f}")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to update ensemble performance: {e}")
    
    def _calculate_ensemble_performance_score(self) -> float:
        """Calculate current ensemble performance score."""
        # Base score from ensemble models
        base_score = 0.5
        
        # Add bonus for model diversity
        if hasattr(self, 'ensemble_models') and self.ensemble_models:
            model_types = {model.get('type', 'unknown') for model in self.ensemble_models}
            diversity_bonus = min(len(model_types) * 0.1, 0.3)
            base_score += diversity_bonus
        
        # Add performance bonus from recent predictions
        if hasattr(self, 'recent_predictions') and self.recent_predictions:
            accuracy_bonus = min(len(self.recent_predictions) * 0.05, 0.2)
            base_score += accuracy_bonus
        
        return min(base_score, 1.0)

    def _get_recent_model_returns(self) -> list:
        """Mock returns for models."""
        return [0.01, -0.005, 0.008]
