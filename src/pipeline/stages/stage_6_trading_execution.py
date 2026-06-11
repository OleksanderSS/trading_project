# src/pipeline/stages/stage_6_trading_execution.py

"""
Stage 6: Trading Execution

This stage takes the final predictions from Stage 5 and orchestrates the entire
trading process using the refactored trading module.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

from src.algorithms.adaptive_position_sizer import AdaptivePositionSizer
from src.algorithms.regime_detector import MarketRegimeDetector
from src.algorithms.risk_parity_allocator import RiskParityAllocator
from src.analytics.analyzers.adaptive_confidence_analyzer import AdaptiveConfidenceAnalyzer
from src.analytics.analyzers.causal_event_finder import CausalEventFinder
from src.analytics.analyzers.news_impact_analyzer import NewsImpactAnalyzer
from src.analytics.calculators.drawdown_calculator import DrawdownCalculator
from src.analytics.calculators.risk_reward_calculator import RiskRewardCalculator
from src.analytics.calculators.volatility_calculator import VolatilityCalculator
from src.analytics.context.macro_context_analyzer import MacroContextAnalyzer

# MarketRegimeAnalyzer removed - use MarketRegimeDetector directly
from src.analytics.context.market_phase_analyzer import MarketPhaseAnalyzer
from src.calibration.adaptive_confidence_calibrator import AdaptiveConfidenceCalibrator
from src.core.logging.logger import ProjectLogger

# Import all the refactored trading components
from src.core.utils.prediction_utils import normalize_prediction
from src.meta_learning.memory.diary_engine import DiaryEngine
from src.models.ensemble.confidence_calibrator import ConfidenceCalibrator
from src.pipeline.stages.base_stage import BaseStage
from src.risk.elite_risk_metrics import EliteRiskMetrics
from src.simulation.simulation_engine import SimulationEngine
from src.trading.adaptive_parameter_manager import AdaptiveParameterManager, AssetClass, MarketRegime

# Add ConsensusEngine for signal quality improvement
from src.trading.consensus_engine import ConsensusEngine, EnhancedConsensusEngine
from src.trading.elite_risk_sizer import EliteRiskSizer

# Use superior LiveAdaptiveEnsemble instead of broken AdaptiveEnsemble
from src.trading.live_adaptive_ensemble import LiveAdaptiveEnsemble
from src.trading.portfolio_manager import PortfolioManager
from src.trading.post_inference_filter import PostInferenceFilter
from src.trading.trader import Trader
from src.trading.trading_orchestrator import TradingOrchestrator
from src.trading.virtual_portfolio import VirtualPortfolio


class TradingExecutionStage(BaseStage):
    """
    A pipeline stage to execute the trading logic.
    """

    # Type annotations for optional components
    market_regime_detector: MarketRegimeDetector | None
    market_phase_analyzer: MarketPhaseAnalyzer | None
    macro_context_analyzer: MacroContextAnalyzer | None
    live_adaptive_ensemble: LiveAdaptiveEnsemble | None
    consensus_engine: ConsensusEngine | None
    confidence_calibrator: ConfidenceCalibrator | None
    adaptive_calibrator: AdaptiveConfidenceCalibrator | None

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

        # ConsensusEngine will be initialized later with fallback handling
        self.consensus_engine = None
        self.logger.info("ConsensusEngine will be initialized with fallback handling.")

        # EnhancedConsensusEngine for regime-based weighting
        self.enhanced_consensus = EnhancedConsensusEngine(
            experience_diary=self.diary_engine,
            threshold_analyzer=self.threshold_analyzer
        )
        self.logger.info("Initialized EnhancedConsensusEngine.")

        # SimulationEngine for Monte Carlo validation
        self.simulation_engine = SimulationEngine()
        self.logger.info("Initialized SimulationEngine for Monte Carlo validation.")

        # Initialize advanced trading analytics tools
        self.adaptive_position_sizer = AdaptivePositionSizer()
        self.risk_parity_allocator = RiskParityAllocator()
        self.volatility_calculator = VolatilityCalculator()
        self.drawdown_calculator = DrawdownCalculator()
        self.risk_reward_calculator = RiskRewardCalculator()

        # Initialize advanced context analysis tools with graceful fallback
        # MarketRegimeAnalyzer removed - use MarketRegimeDetector directly
        self.regime_detector = MarketRegimeDetector()
        self.logger.info("✅ MarketRegimeDetector initialized")

        try:
            # MarketPhaseAnalyzer needs configuration, provide default
            phase_config = {
                'indicators': {'rsi': {}, 'macd': {}, 'volume': {}},
                'rules': [
                    {'name': 'bullish', 'conditions': {'rsi': {'<': 30}, 'volume': {'>': 1.2}}},
                    {'name': 'bearish', 'conditions': {'rsi': {'>': 70}, 'volume': {'>': 1.2}}},
                    {'name': 'neutral', 'conditions': {'rsi': {'between': [30, 70]}}}
                ]
            }
            self.market_phase_analyzer = MarketPhaseAnalyzer(phase_config)
            self.logger.info("✅ MarketPhaseAnalyzer initialized with default configuration")
        except Exception as e:
            self.market_phase_analyzer = None
            self.logger.warning(f"⚠️ MarketPhaseAnalyzer failed to initialize: {e}")

        try:
            # MacroContextAnalyzer needs indicators configuration
            macro_config = {
                'indicators': {
                    'vix': {'weight': 0.3, 'direction': 'inverse'},
                    'interest_rates': {'weight': 0.4, 'direction': 'direct'},
                    'inflation': {'weight': 0.3, 'direction': 'inverse'}
                },
                'regime_thresholds': {
                    'bullish': 0.7,
                    'bearish': -0.7,
                    'neutral': (-0.7, 0.7)
                }
            }
            self.macro_context_analyzer = MacroContextAnalyzer(macro_config)
            self.logger.info("✅ MacroContextAnalyzer initialized with default indicators")
        except Exception as e:
            self.macro_context_analyzer = None
            self.logger.warning(f"⚠️ MacroContextAnalyzer failed to initialize: {e}")

        # Initialize superior LiveAdaptiveEnsemble
        try:
            self.live_adaptive_ensemble = LiveAdaptiveEnsemble(logger=self.logger, reweight_interval_days=7)
            self.logger.info("✅ LiveAdaptiveEnsemble initialized with dynamic weighting")
        except Exception as e:
            self.live_adaptive_ensemble = None
            self.logger.warning(f"⚠️ LiveAdaptiveEnsemble failed to initialize: {e}")

        # Initialize ConsensusEngine for signal quality improvement
        try:
            # Ensure threshold_analyzer is not None
            t_analyzer = self.threshold_analyzer
            if t_analyzer is None:
                self.logger.info("🔧 Re-initializing AdaptiveConfidenceAnalyzer for ConsensusEngine")
                t_analyzer = AdaptiveConfidenceAnalyzer(config=self.config_manager.get('analysis.adaptive_confidence', {}))

            self.consensus_engine = ConsensusEngine(
                experience_diary=self.diary_engine,
                threshold_analyzer=t_analyzer,
                config_manager=self.config_manager,
                live_ensemble=self.live_adaptive_ensemble
            )
            self.logger.info("✅ ConsensusEngine initialized for signal quality improvement")
        except Exception as e:
            self.consensus_engine = None
            self.logger.warning(f"⚠️ ConsensusEngine failed to initialize: {e}")

        self.logger.info("✅ Trading analytics tools initialized with superior ensemble and consensus systems")

        # ConfidenceCalibrator enabled - optuna dependency was optional
        try:
            self.confidence_calibrator = ConfidenceCalibrator(method="isotonic")
            self.logger.info("✅ ConfidenceCalibrator initialized")
        except Exception as e:
            self.confidence_calibrator = None
            self.logger.warning(f"⚠️ ConfidenceCalibrator failed: {e}")

        # ELITE COMPONENT #1: AdaptiveParameterManager
        self.param_manager = AdaptiveParameterManager(logger=self.logger)
        self.logger.info("Initialized AdaptiveParameterManager (elite-grade parameters).")

        # ELITE COMPONENT #2: AdaptiveConfidenceCalibrator enabled
        try:
            self.adaptive_calibrator = AdaptiveConfidenceCalibrator()
            self.logger.info("✅ AdaptiveConfidenceCalibrator initialized")
        except Exception as e:
            self.adaptive_calibrator = None
            self.logger.warning(f"⚠️ AdaptiveConfidenceCalibrator failed: {e}")

        # ELITE COMPONENT #3: EliteRiskSizer
        self.risk_sizer = EliteRiskSizer(logger=self.logger)
        self.logger.info("Initialized EliteRiskSizer (Kelly + correlation-aware sizing).")

        # ELITE COMPONENT #4: EliteRiskMetrics
        self.risk_metrics = EliteRiskMetrics(logger=self.logger)
        self.logger.info("Initialized EliteRiskMetrics (GARCH + Cornish-Fisher VaR).")

        # ELITE COMPONENT #5: LiveAdaptiveEnsemble
        # ✅ INT FIX: Reuse the already-initialized live_adaptive_ensemble to avoid two disconnected instances.
        # live_ensemble is now an alias — both ConsensusEngine and the brain['live_ensemble'] sync point
        # reference the same object, so performance updates are consistent.
        if self.live_adaptive_ensemble is not None:
            self.live_ensemble = self.live_adaptive_ensemble
            self.logger.info("✅ live_ensemble unified with live_adaptive_ensemble (single instance)")
        else:
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

        # Initialize the post-inference filter for signal refinement
        self.post_inference_filter = PostInferenceFilter(config=self.config_manager.get('strategy.post_inference', {}))
        self.logger.info("✅ PostInferenceFilter initialized")

        # 6. Initialize the main conductor for the trading process
        self.trading_orchestrator = TradingOrchestrator(
            consensus_engine=self.consensus_engine,
            portfolio_manager=self.portfolio_manager,
            virtual_portfolio=self.portfolio,
            trader=self.trader,
            post_inference_filter=self.post_inference_filter
        )
        self.logger.info("Trading stack initialization complete.")

    async def run(self, **kwargs) -> dict[str, Any]:
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

    async def _load_or_extract_data(self, kwargs: dict) -> tuple:
        """
        Load predictions and current prices from kwargs or disk.
        """
        predictions = kwargs.get('predictions')
        current_prices = kwargs.get('current_prices')

        # 1. Attempt to load from disk if missing in kwargs
        if not predictions:
            predictions, current_prices = await self._load_forecasts_from_disk(kwargs)

        if not predictions:
            return None, None

        self.logger.info(f"📊 Received {len(predictions)} predictions")

        # 2. Extract prices from predictions if still missing
        if not current_prices:
            current_prices = self._extract_current_prices(predictions)
            if not current_prices:
                return None, None

        self.logger.info(f"💰 Current prices for {len(current_prices)} tickers")
        return predictions, current_prices

    async def _load_forecasts_from_disk(self, kwargs: dict) -> tuple:
        """Helper to load predictions and prices from disk."""
        load_result = await self._load_predictions_from_disk(kwargs)
        if not load_result or len(load_result) < 2:
            return None, None
        return load_result[0], load_result[1]

    async def _load_predictions_from_disk(self, kwargs: dict) -> tuple:
        """Load predictions from disk when not provided in kwargs."""
        self.logger.warning("⚠️ No 'predictions' found in kwargs. Attempting to load from disk...")

        batch_name = kwargs.get('batch_name')
        output_dir = Path(self.config_manager.get('system.accumulation.output_dir', 'data/colab/accumulated'))

        if not batch_name:
            # Prioritize main_database over test batches
            if (output_dir / "main_database").exists():
                batch_name = "main_database"
            else:
                batch_dirs = list(output_dir.glob('test_ticker_*'))
                if batch_dirs:
                    batch_name = max(batch_dirs, key=lambda p: p.stat().st_mtime).name

            if batch_name:
                self.logger.info(f"🔍 Using batch: {batch_name}")

        if batch_name:
            return await self._process_batch_file(batch_name, output_dir, kwargs)

        self.logger.warning("⚠️ Could not find batch_name")
        return None, None, kwargs

    async def _process_batch_file(self, batch_name: str, output_dir: Path, kwargs: dict) -> tuple:
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
        async with aiofiles.open(file_path, encoding='utf-8') as f:
            content = await f.read()
            return str(content)

    def _extract_current_prices(self, predictions: list[dict]) -> dict | None:
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

    def _execute_trading(self, predictions: list[dict], current_prices: dict) -> dict[str, Any]:
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

    def _finalize_results(self, predictions: list[dict], current_prices: dict, kwargs: dict) -> dict[str, Any]:
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

        results_bundle = {
            'predictions': predictions,
            'current_prices': current_prices,
            'portfolio_summary': portfolio_summary,
            'trade_history': trade_history,
            'analyzer_summary': analyzer_summary
        }
        self._save_stage_6_results(results_bundle, kwargs)

        return {
            'trading_activity': trade_history[-5:] if trade_history else [],
            'portfolio_summary': portfolio_summary,
            'signals': predictions,
            'analyzer_summary': analyzer_summary
        }

    def _save_stage_6_results(self, results_bundle: dict[str, Any], kwargs: dict) -> None:
        """
        Saves Stage 6 results to disk for flexible runs.
        """
        try:
            batch_name = kwargs.get('batch_name') or self._find_latest_batch_name()
            if not batch_name:
                return

            output_dir = Path(self.config_manager.get('system.accumulation.output_dir', 'data/colab/accumulated'))
            batch_dir = output_dir / batch_name
            batch_dir.mkdir(parents=True, exist_ok=True)

            stage_6_results = {
                'timestamp': datetime.now().isoformat(),
                'batch_name': batch_name,
                **results_bundle,
                'total_trades': len(results_bundle.get('trade_history', [])),
                'portfolio_value': results_bundle.get('portfolio_summary', {}).get('total_value', 0)
            }

            results_file = batch_dir / "stage_6_results.json"
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(stage_6_results, f, indent=2, default=str)

            self.logger.info(f"✅ Stage 6 results saved: {results_file.name}")
        except Exception as e:
            self.logger.warning(f"⚠️ Error saving Stage 6 results: {e}")

    def _find_latest_batch_name(self) -> str | None:
        """Finds the most recent batch directory name.

        ✅ BUG FIX: Previously only searched 'test_ticker_*' — 'main_database' was never found.
        Now tries main_database first (full-mode default), then scans all batch dirs.
        """
        output_dir = Path(self.config_manager.get('system.accumulation.output_dir', 'data/colab/accumulated'))

        # Priority 1: main_database (full-mode default batch name)
        main_db = output_dir / 'main_database'
        if main_db.exists():
            self.logger.debug("Found main_database batch dir — using it")
            return 'main_database'

        # Priority 2: any test_ticker_* dirs (local / test mode)
        batch_dirs = list(output_dir.glob('test_ticker_*'))
        if batch_dirs:
            return max(batch_dirs, key=lambda p: p.stat().st_mtime).name

        # Priority 3: any subdirectory sorted by mtime
        all_dirs = [p for p in output_dir.iterdir() if p.is_dir()]
        if all_dirs:
            return max(all_dirs, key=lambda p: p.stat().st_mtime).name

        return None

    def _calculate_robust_confidence(self, ticker, target_key, models_metadata, predictions):
        """Calculates robust confidence score.

        ✅ BUG FIX 1: weights now sum to 1.0 (was 0.9).
        ✅ BUG FIX 2: always returns a value — previously returned None when calibrator was None.
        """
        try:
            champion_data = self._find_champion_model(ticker, target_key, models_metadata)
            base_confidence = self._calculate_base_confidence(champion_data)
            error_ratios = self._calculate_error_ratios(champion_data)
            anomaly_penalty = self._calculate_anomaly_penalty(ticker, predictions)
            consensus_boost = self._calculate_consensus_boost(predictions)

            # Weights: 0.4 + 0.3 + 0.2 + 0.1 (consensus_boost treated as additive bonus)
            # Combined weighted base, capped before boost so it stays <=1
            weighted_base = (base_confidence * 0.45 + error_ratios['mae_ratio'] * 0.30 + error_ratios['rmse_ratio'] * 0.25)
            adjusted = weighted_base * (1 - anomaly_penalty) + consensus_boost
            confidence = max(0.01, min(1.0, adjusted))

            if self.adaptive_calibrator is not None:
                calibrated = self.adaptive_calibrator.calibrate(confidence)
                return max(0.01, min(1.0, calibrated))

            # ✅ Always return — was missing return when calibrator is None
            return confidence

        except Exception as e:
            self.logger.warning(f"Error calculating robust confidence: {e}")
            return 0.3

    def _find_champion_model(self, ticker: str, target_key: str, models_metadata: dict) -> dict:
        """Find the champion model for the given ticker and target.

        ✅ BUG FIX: Previously r2 and model_type were read outside the if ticker_from_meta == ticker
        block, so they used values from the previous loop iteration when the ticker didn't match.
        """
        heavy_models_list = []

        for _context_id, meta in models_metadata.items():
            if meta.get('target') == target_key.split('_', 1)[1]:
                ticker_from_meta = meta.get('ticker', '')
                if ticker_from_meta == ticker:
                    # ✅ All three lines are now correctly inside the if-block
                    test_metrics = meta.get('test_metrics', {})
                    r2 = test_metrics.get('r2', 0.0)
                    model_type = meta.get('model_type', '')
                    heavy_models_list.append({'type': model_type, 'r2': r2, 'meta': meta})

        all_models = sorted(heavy_models_list, key=lambda x: x['r2'], reverse=True)
        return all_models[0] if all_models else {}

    def _calculate_base_confidence(self, champion_data: dict | None) -> float:
        """Calculate base confidence from champion model data."""
        if not champion_data:
            return 0.3

        r2 = champion_data.get('r2', 0.0)
        if r2 < -2:
            return 0.1
        elif r2 < 0:
            return float(0.2 + (r2 / 2) * 0.2)
        else:
            return float(0.3 + r2 * 0.7)

    def _calculate_error_ratios(self, champion_data: dict) -> dict[str, float]:
        """Calculate error ratios from champion model data."""
        test_metrics = champion_data['meta'].get('test_metrics', {}) if champion_data else {}
        rmse = test_metrics.get('rmse', 1.0)
        mae = test_metrics.get('mae', 1.0)
        mae_ratio = 1.0 / (1.0 + mae * 2)
        rmse_ratio = 1.0 / (1.0 + rmse * 2)

        return {'mae_ratio': mae_ratio, 'rmse_ratio': rmse_ratio}

    def _calculate_anomaly_penalty(self, ticker: str, predictions: list[dict]) -> float:
        """Calculate anomaly penalty from predictions."""
        anomaly_score = 0.5
        for pred in predictions:
            if pred.get('ticker') == ticker:
                anomaly_score = pred.get('anomaly_score', 0.5)
                break

        return anomaly_score * 0.2

    def _calculate_consensus_boost(self, predictions: list[dict]) -> float:
        """Calculate consensus boost from predictions."""
        positive_votes = len([p for p in predictions if p.get('predictions', 0) > 0.001])
        total_models = len(predictions)
        return (positive_votes / max(total_models, 1)) * 0.1

    def _get_signal_strength(self, confidence):
        """Determines signal strength based on confidence."""
        if confidence >= 0.7:
            return "very_strong"
        elif confidence >= 0.6:
            return "strong"
        elif confidence >= 0.5:
            return "medium"
        elif confidence >= 0.35:
            return "weak"
        else:
            return "very_weak"

    def _calculate_risk_score(self, rec):
        """Calculates risk score for a position."""
        base_risk = 0.5
        confidence_risk = (1 - rec['confidence']) * 0.3
        news_risk = 0.1 if rec.get('news_warning') else 0.0
        return max(0.0, min(1.0, base_risk + confidence_risk + news_risk))

    def _validate_with_monte_carlo(self, ticker):
        """Advanced risk validation using EliteRiskMetrics with stress testing and liquidity."""
        try:
            # 1. Ensemble VaR (Historical + GARCH + Cornish-Fisher)
            var_hist = self.risk_metrics.compute_historical_simulation_var(ticker, confidence_level=0.95, lookback_days=252)
            var_garch = self.risk_metrics.compute_garch_var(ticker, confidence_level=0.95)
            var_cf, _ = self.risk_metrics.compute_cornish_fisher_var(ticker, confidence_level=0.95, lookback_days=252)
            var_95 = 0.4 * var_hist + 0.35 * var_garch + 0.25 * var_cf

            # 2. Stress Testing (market crash scenario)
            portfolio = {ticker: 1.0}  # 100% weight for single asset
            stress_result = self.risk_metrics.run_stress_test(portfolio, scenario='market_crash')
            stress_impact = abs(stress_result['portfolio_impact'])

            # 3. Calculate position size factor
            var_threshold = 0.05
            position_size_factor = 1.0

            # Reduce size if VaR exceeds threshold
            if var_95 > var_threshold:
                excess_var = var_95 - var_threshold
                reduction = (excess_var / var_threshold) * 0.5
                position_size_factor = max(0.1, 1.0 - reduction)
                self.logger.warning(f"⚠️ Elite VaR {var_95:.3f} > {var_threshold:.3f} for {ticker}, factor reduced to {position_size_factor:.1%}")

            # Further reduce if stress test shows critical loss
            if stress_impact > 0.1:  # >10% loss in crash
                stress_reduction = min(stress_impact, 0.5)
                position_size_factor *= (1 - stress_reduction)
                self.logger.debug(f"⚠️ Stress test shows {stress_impact:.1%} loss for {ticker}, factor reduced to {position_size_factor:.1%}")

            return 0.5 * position_size_factor, var_95, position_size_factor
        except Exception as e:
            self.logger.warning(f"⚠️ Elite risk validation failed for {ticker}: {e}")
            return 0.5, 0.03, 1.0

    def _create_consolidated_table(self, buy_recs, sell_recs, models_metadata, predictions=None):
        """Creates a consolidated table of all recommendations."""
        consolidated = []
        predictions = predictions or []

        # Build per-ticker prediction map from Stage 5 predictions_by_model
        # Key: ticker -> list of per-model prediction values
        ticker_model_preds: dict[str, list[float]] = {}
        for pred in predictions:
            t = pred.get('ticker')
            if not t:
                continue
            by_model = pred.get('predictions_by_model', {})
            if by_model:
                vals = []
                for v in by_model.values():
                    try:
                        vals.append(float(v))
                    except (TypeError, ValueError):
                        pass
                ticker_model_preds.setdefault(t, []).extend(vals)
            else:
                # Fallback: use single prediction value
                try:
                    ticker_model_preds.setdefault(t, []).append(float(pred.get('predictions', 0)))
                except (TypeError, ValueError):
                    pass

        for rec in buy_recs:
            ticker = rec['ticker']
            supporting_models = [m for m in models_metadata.values() if m.get('ticker') == ticker]
            model_preds = ticker_model_preds.get(ticker, [])
            positive_count = sum(1 for v in model_preds if v > 0.001)
            total_model_count = len(model_preds) if model_preds else max(len(supporting_models), 1)
            consensus = positive_count / total_model_count

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
                "total_models": total_model_count,
                "model_consensus": consensus,
                "risk_score": self._calculate_risk_score(rec),
                "news_warning": rec.get('news_warning'),
                "var_95": rec.get('var_95', 0.03),
                "position_size_factor": rec.get('position_size_factor', 1.0),
                "reason": rec['reason'],
                "composite_score": rec['confidence'] * consensus,
                "timestamp": datetime.now().isoformat()
            })

        for rec in sell_recs:
            ticker = rec['ticker']
            supporting_models = [m for m in models_metadata.values() if m.get('ticker') == ticker]
            model_preds = ticker_model_preds.get(ticker, [])
            negative_count = sum(1 for v in model_preds if v < -0.001)
            total_model_count = len(model_preds) if model_preds else max(len(supporting_models), 1)
            consensus = negative_count / total_model_count

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
                "total_models": total_model_count,
                "model_consensus": consensus,
                "risk_score": self._calculate_risk_score(rec),
                "news_warning": rec.get('news_warning'),
                "var_95": rec.get('var_95', 0.03),
                "position_size_factor": rec.get('position_size_factor', 1.0),
                "reason": rec['reason'],
                "composite_score": rec['confidence'] * consensus,
                "timestamp": datetime.now().isoformat()
            })

        consolidated.sort(key=lambda x: x['composite_score'], reverse=True)
        return consolidated

    def _generate_analyzer_recommendations(self, predictions: list[dict], current_prices: dict[str, float], models_metadata: dict, **kwargs) -> dict[str, Any]:
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
            features_df = kwargs.get('features_df') if kwargs.get('features_df') is not None else kwargs.get('features_data')
            self._generate_trading_recommendations(predictions, current_prices, recommendations, news_impact_scores, features_df=features_df)
            recommendations['consolidated_table'] = self._create_consolidated_table(
                recommendations['buy_recommendations'], recommendations['sell_recommendations'],
                models_metadata, predictions
            )

        except Exception as e:
            self.logger.error(f"❌ Failed to generate recommendations: {e}", exc_info=True)
            return self._fallback_recommendations(predictions, current_prices)

        return recommendations

    def _initialize_recommendations_structure(self) -> dict[str, Any]:
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

    def _categorize_models(self, models_metadata: dict) -> tuple:
        """Categorize models into heavy and light models."""
        HEAVY_MODELS = {"gru", "tabnet", "transformer", "cnn", "lstm", "autoencoder"}
        heavy_models: dict[str, list[dict[str, Any]]] = {}
        light_models: dict[str, list[dict[str, Any]]] = {}

        for context_id, meta in models_metadata.items():
            model_type = meta.get('winner', meta.get('model_type', '')).lower()
            ticker = meta.get('ticker', '')
            target = meta.get('target', '')
            metrics = meta.get('metrics') or {}
            is_regression = isinstance(metrics, dict) and ('r2' in metrics or 'mse' in metrics)
            # Determine accuracy based on model type
            if is_regression:
                accuracy = metrics.get('r2', metrics.get('score', 0.0))
            else:
                accuracy = metrics.get('accuracy', metrics.get('score', 0.0))

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

    def _populate_champion_by_target(self, recommendations: dict, heavy_models: dict, light_models: dict) -> None:
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

    def _enhance_predictions_with_confidence(self, predictions: list[dict], recommendations: dict, models_metadata: dict) -> None:
        """
        Enhance predictions with confidence calculations.

        CodeScene: Deep Nested Complexity acceptable - Prediction enhancement requires nested
        iteration through predictions, models, and targets to match ensemble recommendations
        with confidence scores. This hierarchical structure reflects the data organization.
        """
        for pred in predictions:
            ticker = pred.get('ticker')
            pred_by_model = pred.get('predictions_by_model', {})

            if not pred_by_model:
                continue

            first_key = list(pred_by_model.keys())[0]
            parts = first_key.split('_')
            if len(parts) < 3:
                continue

            target_key = f"{ticker}_{'_'.join(parts[2:])}"
            ensemble_info = recommendations['champion_by_target'].get(target_key)
            if not ensemble_info:
                continue

            pred['confidence'] = self._calculate_robust_confidence(ticker, target_key, models_metadata, [pred])
            pred['champion_model'] = 'ensemble'

    def _analyze_news_impact(self, news_data: Any) -> dict:
        """Analyze news impact for predictions."""
        news_impact_scores: dict[str, Any] = {}
        if news_data is not None and hasattr(news_data, 'empty') and not news_data.empty:
            try:
                news_analysis = self.news_impact_analyzer.analyze(news_data)
                if news_analysis and 'news_impact_scores' in news_analysis:
                    return cast(dict[str, Any], news_analysis['news_impact_scores'])
            except Exception as e:
                self.logger.warning(f"News impact analysis error: {e}")
        return news_impact_scores

    def _generate_trading_recommendations(self, predictions: list[dict], current_prices: dict, recommendations: dict, news_impact_scores: dict, features_df: pd.DataFrame | None = None) -> None:
        """
        Generate buy and sell recommendations based on predictions.
        """
        # Set default regime
        global_regime = 'ranging'

        # Dynamically detect global regime if features are available
        if features_df is not None and not features_df.empty:
            try:
                # Use SPY or the first available ticker for global regime
                ticker = 'SPY' if 'SPY' in features_df['ticker'].values else features_df['ticker'].iloc[0]
                ticker_df = features_df[features_df['ticker'] == ticker] if 'ticker' in features_df.columns else features_df

                if 'close' in ticker_df.columns:
                    returns = ticker_df['close'].pct_change().dropna().values
                    if len(returns) > 30:
                        regime_result = self.regime_detector.detect_regime(returns, data_bundle={'prices': ticker_df['close'].values})
                        detected = regime_result.get('regime', 'NORMAL').lower()
                        # Map detected regime to one of the simple regimes understood by param manager
                        if 'trend' in detected and 'up' in detected:
                            global_regime = 'bull'
                        elif 'trend' in detected and 'down' in detected:
                            global_regime = 'bear'
                        elif 'volatile' in detected or 'crisis' in detected:
                            global_regime = 'volatile'
                        else:
                            global_regime = 'ranging'
                        self.logger.info(f"📊 Dynamically detected global regime: {global_regime} (from {detected})")
            except Exception as e:
                self.logger.warning(f"⚠️ Failed to detect dynamic regime: {e}")

        for pred in predictions:
            ticker = pred.get('ticker')
            if not ticker:
                continue

            asset_class = self._determine_asset_class(ticker)

            # Optionally detect regime per ticker
            regime = global_regime
            if features_df is not None and not features_df.empty and 'ticker' in features_df.columns:
                 ticker_df = features_df[features_df['ticker'] == ticker]
                 if not ticker_df.empty and 'close' in ticker_df.columns:
                     returns = ticker_df['close'].pct_change().dropna().values
                     if len(returns) > 30:
                         try:
                             regime_result = self.regime_detector.detect_regime(returns, data_bundle={'prices': ticker_df['close'].values})
                             detected = regime_result.get('regime', 'NORMAL').lower()
                             if 'trend' in detected and 'up' in detected:
                                 regime = 'bull'
                             elif 'trend' in detected and 'down' in detected:
                                 regime = 'bear'
                             elif 'volatile' in detected or 'crisis' in detected:
                                 regime = 'volatile'
                             else:
                                 regime = 'ranging'
                         except Exception:
                             pass

            adaptive_params = self.param_manager.compute_adaptive_params(
                regime=MarketRegime(regime.lower()),
                asset_class=AssetClass(asset_class.lower()),
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

    def _extract_prediction_value(self, pred: dict) -> float:
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

    def _check_news_warning(self, ticker: str, news_impact_scores: dict) -> str | None:
        """Check for news warnings."""
        if ticker in news_impact_scores and news_impact_scores[ticker].get('score', 0) < -0.3:
            return "Negative news impact"
        return None

    def _fallback_recommendations(self, predictions: list[dict], current_prices: dict[str, float]) -> dict[str, Any]:
        """Simple fallback recommendation logic with robust scaling."""
        recommendations = {'buy_recommendations': [], 'sell_recommendations': [], 'risk_warnings': [], 'fallback_mode': True}
        for pred in predictions:
            ticker = pred.get('ticker')
            pred_value = self._extract_prediction_value(pred)

            if pred_value > 0.01:
                cast(list, recommendations['buy_recommendations']).append({'ticker': str(ticker), 'predicted_return': pred_value, 'current_price': current_prices.get(str(ticker)), 'confidence': 0.5, 'reason': 'Positive (Fallback)'})
            elif pred_value < -0.01:
                cast(list, recommendations['sell_recommendations']).append({'ticker': str(ticker), 'predicted_return': pred_value, 'current_price': current_prices.get(str(ticker)), 'confidence': 0.5, 'reason': 'Negative (Fallback)'})
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
