from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from src.core.exceptions import DataProcessingError
from src.core.logging.logger import ProjectLogger
from src.core.utils.prediction_utils import normalize_prediction
from src.models.registry.model_registry import ModelRegistry
from src.risk.elite_risk_metrics import EliteRiskMetrics
from src.trading.adaptive_parameter_manager import AdaptiveParameterManager, AssetClass, MarketRegime
from src.agents.modular_pipeline.orchestrator import get_default_orchestrator
import asyncio


class TradingRecommendationEngine:

    def __init__(self, logger: ProjectLogger, news_impact_analyzer: (Any |
        None), param_manager: (AdaptiveParameterManager | None),
        regime_detector: (Any | None), risk_metrics: (EliteRiskMetrics |
        None), adaptive_calibrator: (Any | None)):
        self.logger = logger
        self.news_impact_analyzer = news_impact_analyzer
        self.param_manager = param_manager
        self.regime_detector = regime_detector
        self.risk_metrics = risk_metrics
        self.adaptive_calibrator = adaptive_calibrator
        
        # Ініціалізуємо модульного аналітика замість старого veto_system
        self.cognitive_orchestrator = get_default_orchestrator()

    def generate_recommendations(self, predictions: list[dict[str, Any]],
        current_prices: dict[str, float], models_metadata: dict[str, Any],
        news_data: Any=None, features_df: (pd.DataFrame | None)=None) ->dict[
        str, Any]:
        recommendations = self._initialize_recommendations_structure()
        try:
            if not models_metadata:
                self.logger.warning(
                    '⚠️ models_metadata not found. Using fallback logic.')
                return self._fallback_recommendations(predictions,
                    current_prices)
            heavy_models, light_models = self._categorize_models(
                models_metadata)
            self._populate_champion_by_target(recommendations, heavy_models,
                light_models)
            self._enhance_predictions_with_confidence(predictions,
                recommendations, models_metadata)
            news_impact_scores = self._analyze_news_impact(news_data)
            self._generate_trading_recommendations(predictions,
                current_prices, recommendations, news_impact_scores,
                features_df=features_df)
            
            # --- COGNITIVE PIPELINE (Modular Lenses) ---
            try:
                # Отримуємо свіжі новини (якщо є)
                latest_news_text = ""
                if hasattr(news_data, 'to_string'):
                    latest_news_text = news_data.head(5).to_string()
                elif isinstance(news_data, dict):
                    latest_news_text = str(news_data)[:500]

                if latest_news_text:
                    # Запускаємо оркестратор
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        import nest_asyncio
                        nest_asyncio.apply()
                    
                    # Припустимо, ми витягнули теги (можна адаптувати classification_yaml)
                    # Тимчасово прокидаємо wildcard або глобальні теги для аналізу
                    analysis_packet = loop.run_until_complete(
                        self.cognitive_orchestrator.analyze(latest_news_text, affected_tags=['market_wide'])
                    )
                    
                    # Прокидаємо згенеровані сценарії у рекомендації, щоб PortfolioManager міг їх порізати
                    cognitive_scenarios = analysis_packet.get("scenario_nodes", [])
                    if cognitive_scenarios:
                        for rec in recommendations['buy_recommendations']:
                            rec['cognitive_scenarios'] = cognitive_scenarios
                        for rec in recommendations['sell_recommendations']:
                            rec['cognitive_scenarios'] = cognitive_scenarios
                            
            except Exception as e:
                self.logger.error(f"Помилка в Cognitive Pipeline: {e}. Пропускаємо аналіз новин.")
            # ----------------------------------------------------

            recommendations['consolidated_table'
                ] = self._create_consolidated_table(recommendations[
                'buy_recommendations'], recommendations[
                'sell_recommendations'], models_metadata, predictions)
        except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError) as e:
            self.logger.error(f'❌ Failed to generate recommendations: {e}',
                exc_info=True)
            return self._fallback_recommendations(predictions, current_prices)
        return recommendations

    def _initialize_recommendations_structure(self) ->dict[str, Any]:
        return {'buy_recommendations': [], 'sell_recommendations': [],
            'risk_warnings': [], 'champion_model': None,
            'champion_by_target': {}, 'heavy_models_ranking': [],
            'light_models_ranking': [], 'model_rankings': [],
            'actor_critic_log': {'status': 'fallback', 'reason':
            'DEAN models not trained yet. Need more history.',
            'trade_count': 0}}

    def _categorize_models(self, models_metadata: dict[str, Any]) ->tuple[
        dict[str, list[dict[str, Any]]], dict[str, list[dict[str, Any]]]]:
        heavy_model_types = set(ModelRegistry.get_models_by_type('heavy'))
        heavy_models: dict[str, list[dict[str, Any]]] = {}
        light_models: dict[str, list[dict[str, Any]]] = {}
        for context_id, meta in models_metadata.items():
            model_type = meta.get('winner', meta.get('model_type', '')).lower()
            ticker = meta.get('ticker', '')
            target = meta.get('target', '')
            metrics = meta.get('metrics') or {}
            is_regression = isinstance(metrics, dict) and ('r2' in metrics or
                'mse' in metrics)
            if is_regression:
                accuracy = metrics.get('r2', metrics.get('score', 0.0))
            else:
                accuracy = metrics.get('accuracy', metrics.get('score', 0.0))
            model_info = {'context_id': context_id, 'model_type':
                model_type, 'ticker': ticker, 'target': target, 'accuracy':
                accuracy, 'metrics': metrics}
            key = f'{ticker}_{target}'
            if any(heavy in model_type for heavy in heavy_model_types):
                heavy_models.setdefault(key, []).append(model_info)
            else:
                light_models.setdefault(key, []).append(model_info)
        return heavy_models, light_models

    def _get_champion_model_for_target(self, target_key: str, heavy_models:
        dict[str, list[dict[str, Any]]], light_models: dict[str, list[dict[
        str, Any]]]) ->(dict[str, Any] | None):
        combined_group = heavy_models.get(target_key, []) + light_models.get(
            target_key, [])
        if not combined_group:
            return None
        return max(combined_group, key=lambda x: x['accuracy'])

    def _populate_champion_by_target(self, recommendations: dict[str, Any],
        heavy_models: dict[str, list[dict[str, Any]]], light_models: dict[
        str, list[dict[str, Any]]], features_df: pd.DataFrame | None = None) ->None:
        """Populate champion model metadata per target.

        Previously hardcoded regime='ranging' here while _generate_trading_recommendations
        correctly detected the real regime. Now uses _detect_global_regime so the
        champion metadata reflects the actual market state.
        """
        try:
            regime = self._detect_global_regime(features_df) if features_df is not None and not features_df.empty else 'ranging'
        except Exception:  # noqa: BLE001 — regime detection is best-effort for metadata
            regime = 'ranging'

        all_targets = set(heavy_models.keys()) | set(light_models.keys())
        for target_key in all_targets:
            champion = self._get_champion_model_for_target(target_key,
                heavy_models, light_models)
            if champion:
                recommendations['champion_by_target'][target_key] = {
                    'model_type': 'live_adaptive_ensemble', 'regime':
                    regime, 'ticker': champion['ticker'], 'accuracy':
                    champion['accuracy']}

    def _enhance_predictions_with_confidence(self, predictions: list[dict[
        str, Any]], recommendations: dict[str, Any], models_metadata: dict[
        str, Any]) ->None:
        for pred in predictions:
            ticker = pred.get('ticker')
            pred_by_model = pred.get('predictions_by_model', {})
            if not pred_by_model:
                continue
            first_key = next(iter(pred_by_model), '')
            parts = first_key.split('_')
            if len(parts) < 3:
                continue
            target_key = f"{ticker}_{'_'.join(parts[2:])}"
            ensemble_info = recommendations['champion_by_target'].get(
                target_key)
            if not ensemble_info:
                continue
            pred['confidence'] = self._calculate_robust_confidence(ticker,
                target_key, models_metadata, [pred])
            pred['champion_model'] = 'ensemble'

    def _analyze_news_impact(self, news_data: Any) ->dict[str, Any]:
        news_impact_scores: dict[str, Any] = {}
        if news_data is not None:
            if hasattr(news_data, 'empty') and not news_data.empty:
                try:
                    news_analysis = self.news_impact_analyzer.analyze(news_data
                        )
                    if news_analysis and 'news_impact_scores' in news_analysis:
                        return dict(news_analysis['news_impact_scores'])
                except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError) as e:
                    self.logger.error(f'Виникла помилка: {e}', exc_info=True)
                    self.logger.warning(f'News impact analysis error: {e}. Continuing without news impact.')
                    # Graceful degradation: return empty scores instead of raising.
                    # A news analysis failure should not degrade recommendations
                    # for all tickers in the batch.
            elif isinstance(news_data, dict
                ) and 'news_impact_scores' in news_data:
                return dict(news_data['news_impact_scores'])
        return news_impact_scores

    def _detect_global_regime(self, features_df: pd.DataFrame) -> str:
        """Detect global market regime from features."""
        global_regime = 'ranging'
        if features_df is None or features_df.empty:
            return global_regime

        try:
            tickers = features_df['ticker'].unique()
            ticker = 'SPY' if 'SPY' in tickers else tickers[0]
            ticker_df = features_df[features_df['ticker'] == ticker] if 'ticker' in features_df.columns else features_df

            if 'close' not in ticker_df.columns:
                return global_regime

            returns = (
                ticker_df['close']
                .pct_change(fill_method=None)
                .replace([np.inf, -np.inf], np.nan)
                .dropna()
                .values
            )
            if len(returns) > 30 and self.regime_detector is not None:
                regime_result = self.regime_detector.detect_regime(returns, data_bundle={'prices': ticker_df['close'].values})
                detected = regime_result.get('regime', 'NORMAL').lower()

                if 'trend' in detected and 'up' in detected:
                    global_regime = 'bull'
                elif 'trend' in detected and 'down' in detected:
                    global_regime = 'bear'
                elif 'volatile' in detected or 'crisis' in detected:
                    global_regime = 'volatile'
                else:
                    global_regime = 'ranging'

                self.logger.info(f'📊 Dynamically detected global regime: {global_regime} (from {detected})')
        except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError, DataProcessingError) as e:
            self.logger.warning(f'⚠️ Failed to detect dynamic regime: {e}. Falling back to "ranging".')
            # Graceful degradation: regime detection failure must not abort
            # all recommendations for the entire batch of tickers.

        return global_regime

    def _detect_ticker_regime(self, ticker: str, features_df: pd.DataFrame, global_regime: str) -> str:
        """Detect ticker-specific regime."""
        regime = global_regime

        if features_df is None or features_df.empty or 'ticker' not in features_df.columns:
            return regime

        ticker_df = features_df[features_df['ticker'] == ticker]
        if ticker_df.empty or 'close' not in ticker_df.columns:
            return regime

        returns = (
            ticker_df['close']
            .pct_change(fill_method=None)
            .replace([np.inf, -np.inf], np.nan)
            .dropna()
            .values
        )
        if len(returns) <= 30 or self.regime_detector is None:
            return regime

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
        except Exception as e:
            self.logger.warning(f'Regime detection failed for {ticker}: {e}. Using global_regime "{global_regime}".')
            # Graceful degradation: per-ticker regime failure falls back to
            # the already-computed global regime rather than aborting the batch.

        return regime

    def _compute_adaptive_parameters(self, regime: str, asset_class: str) -> object:
        """Compute adaptive parameters for regime and asset class."""
        if self.param_manager is None:
            return type('P', (), {'buy_threshold': 0.01, 'sell_threshold': -0.01})()

        return self.param_manager.compute_adaptive_params(
            regime=MarketRegime(regime.lower()),
            asset_class=AssetClass(asset_class.lower()),
            volatility_percentile=50
        )

    def _build_recommendation(self, ticker: str, pred_value: float, current_price: float,
                             news_warning: str, mc_confidence: float, var_95: float,
                             pos_factor: float, adaptive_params: object) -> dict:
        """Build recommendation dictionary with Stop-Loss and Take-Profit."""
        take_profit_mult = getattr(adaptive_params, 'take_profit_multiplier', 2.0)
        stop_loss_mult = getattr(adaptive_params, 'stop_loss_multiplier', 1.0)

        stop_loss_pct = var_95 * stop_loss_mult
        take_profit_pct = var_95 * take_profit_mult

        return {
            'ticker': ticker,
            'predicted_return': pred_value,
            'current_price': current_price,
            'confidence': mc_confidence,
            'news_warning': news_warning,
            'var_95': var_95,
            'position_size_factor': pos_factor,
            'stop_loss_pct': stop_loss_pct,
            'take_profit_pct': take_profit_pct,
            'champion_model': 'ensemble'
        }

    def _classify_recommendation(self, recommendation: dict, pred_value: float,
                                adaptive_params: object, recommendations: dict) -> None:
        """Classify recommendation as buy or sell based on thresholds."""
        if pred_value > adaptive_params.buy_threshold:
            recommendation.update({'reason': 'Positive prediction'})
            recommendations['buy_recommendations'].append(recommendation)
        elif pred_value < adaptive_params.sell_threshold:
            recommendation.update({'reason': 'Negative prediction'})
            recommendations['sell_recommendations'].append(recommendation)

    def _generate_trading_recommendations(self, predictions: list[dict[str,
        Any]], current_prices: dict[str, float], recommendations: dict[str,
        Any], news_impact_scores: dict[str, Any], features_df: (pd.
        DataFrame | None)=None) ->None:
        global_regime = self._detect_global_regime(features_df)

        for pred in predictions:
            ticker = pred.get('ticker')
            if not ticker:
                continue

            asset_class = self._determine_asset_class(ticker)
            regime = self._detect_ticker_regime(ticker, features_df, global_regime)

            adaptive_params = self._compute_adaptive_parameters(regime, asset_class)

            pred_value = self._extract_prediction_value(pred)
            news_warning = self._check_news_warning(ticker, news_impact_scores)
            mc_confidence, var_95, pos_factor = self._validate_with_monte_carlo(ticker)
            
            final_confidence = pred.get('confidence', mc_confidence)
            
            # Dynamic Position Sizing based on Final Confidence (Advanced Research Feature)
            if final_confidence < 0.6:
                pos_factor = 0.0 # Extreme doubt, do not trade
                self.logger.info(f"🛡️ Dynamic Sizing: {ticker} confidence {final_confidence:.2f} < 0.60. Position cut to 0.0")
            elif final_confidence < 0.75:
                pos_factor *= 0.5 # Low confidence, half position
            elif final_confidence > 0.9:
                pos_factor = min(2.0, pos_factor * 1.5) # High confidence, boost position
                self.logger.info(f"🔥 Dynamic Sizing: {ticker} confidence {final_confidence:.2f} > 0.90. Position boosted!")

            recommendation = self._build_recommendation(
                ticker, pred_value, current_prices.get(ticker), news_warning,
                final_confidence, var_95, pos_factor, adaptive_params
            )

            self._classify_recommendation(recommendation, pred_value, adaptive_params, recommendations)

    def _extract_prediction_value(self, pred: dict[str, Any]) ->float:
        predictions = pred.get('predictions', pred.get('prediction', 0))
        val = 0.0
        if isinstance(predictions, (list, tuple, np.ndarray)):
            val = float(predictions[-1]) if predictions else 0.0
        elif predictions is not None and hasattr(predictions, 'item'):
            val = float(predictions.item())
        else:
            val = float(predictions)
        if abs(val) > 10:
            return float(np.sign(val) * (np.log1p(abs(val)) / 100.0))
        return normalize_prediction(val)

    def _check_news_warning(self, ticker: str, news_impact_scores: dict[str,
        Any]) ->(str | None):
        if ticker in news_impact_scores and news_impact_scores[ticker].get(
            'score', 0) < -0.3:
            return 'Negative news impact'
        return None

    def _fallback_recommendations(self, predictions: list[dict[str, Any]],
        current_prices: dict[str, float]) ->dict[str, Any]:
        recommendations = {'buy_recommendations': [],
            'sell_recommendations': [], 'risk_warnings': [],
            'fallback_mode': True}
        for pred in predictions:
            ticker = pred.get('ticker')
            pred_value = self._extract_prediction_value(pred)
            if pred_value > 0.01:
                recommendations['buy_recommendations'].append({'ticker':
                    str(ticker), 'predicted_return': pred_value,
                    'current_price': current_prices.get(str(ticker)),
                    'confidence': 0.5, 'reason': 'Positive (Fallback)'})
            elif pred_value < -0.01:
                recommendations['sell_recommendations'].append({'ticker':
                    str(ticker), 'predicted_return': pred_value,
                    'current_price': current_prices.get(str(ticker)),
                    'confidence': 0.5, 'reason': 'Negative (Fallback)'})
        return recommendations

    def _determine_asset_class(self, ticker: str) -> str:
        """Classify ticker into asset class for adaptive parameter selection.

        Classification is rule-based using well-known ticker lists.
        Defaults to 'large_cap' for unrecognised symbols.
        """
        t = ticker.upper()

        # ETFs / indices
        _etfs = {'SPY', 'QQQ', 'IWM', 'DIA', 'VTI', 'VOO', 'GLD', 'SLV',
                 'TLT', 'IEF', 'HYG', 'LQD', 'XLK', 'XLF', 'XLE', 'XLV'}
        if t in _etfs:
            return 'etf'

        # Crypto proxies (COIN, MSTR, MARA, RIOT are traded on equity exchanges)
        _crypto_proxies = {'COIN', 'MSTR', 'MARA', 'RIOT', 'HUT', 'CLSK'}
        if t in _crypto_proxies:
            return 'crypto'

        # Large-cap mega tech / well-known blue chips already in assets.yaml
        _large_caps = {
            'AAPL', 'MSFT', 'GOOGL', 'GOOG', 'AMZN', 'NVDA', 'META',
            'TSLA', 'BRK', 'JPM', 'V', 'MA', 'UNH', 'XOM', 'CVX',
            'BAC', 'WMT', 'KO', 'PEP', 'JNJ', 'PG', 'HD', 'MRK',
            'GS', 'MS', 'TSM',
        }
        if t in _large_caps:
            return 'large_cap'

        # Mid-cap semiconductors / tech in assets.yaml
        _mid_caps = {'AMD', 'INTC', 'QCOM', 'MU', 'AMAT', 'LRCX', 'KLAC'}
        if t in _mid_caps:
            return 'mid_cap'

        # Default — treat unknown tickers conservatively as large_cap
        return 'large_cap'

    def _calculate_robust_confidence(self, ticker: str, target_key: str,
        models_metadata: dict[str, Any], predictions: list[dict[str, Any]]
        ) ->float:
        try:
            champion_data = self._find_champion_model(ticker, target_key,
                models_metadata)
            base_confidence = self._calculate_base_confidence(champion_data)
            error_ratios = self._calculate_error_ratios(champion_data)
            anomaly_penalty = self._calculate_anomaly_penalty(ticker,
                predictions)
            consensus_boost = self._calculate_consensus_boost(predictions)
            confidence = (base_confidence * 0.4 + error_ratios['mae_ratio'] *
                0.3 + error_ratios['rmse_ratio'] * 0.2) * (1 - anomaly_penalty
                ) + consensus_boost
            if self.adaptive_calibrator is not None:
                calibrated_confidence = self.adaptive_calibrator.calibrate(
                    confidence)
                return max(0.01, min(1.0, calibrated_confidence))
            return max(0.01, min(1.0, confidence))
        except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError) as e:
            self.logger.warning(f'Error calculating robust confidence for {ticker}/{target_key}: {e}. Returning default 0.3.')
            return 0.3

    # Classification target type prefixes — used to pick the right sort metric.
    _CLASSIFICATION_TARGET_PREFIXES = (
        'target_up_', 'target_multi_', 'target_intraday_up_',
        'target_hourly_up_', 'target_hourly_volume_spike_',
        'target_hourly_breakout_', 'target_weekly_up_',
    )

    def _is_classification_target(self, target_name: str) -> bool:
        """Return True when target_name is a classification (binary/multiclass) target."""
        return any(target_name.startswith(p) for p in self._CLASSIFICATION_TARGET_PREFIXES)

    def _find_champion_model(self, ticker: str, target_key: str,
        models_metadata: dict[str, Any]) ->(dict[str, Any] | None):
        """Find the best model for ticker+target pair.

        For regression targets (target_return_*, target_rsi_*, etc.) rank by R².
        For classification targets (target_up_*, target_multi_*, etc.) rank by
        balanced_accuracy > accuracy > f1, falling back to 0.0 when absent.
        Using R²=0.0 as the sort key for classifiers caused all candidates to
        tie and the "champion" was effectively chosen by dict insertion order.
        """
        # Determine the bare target name without ticker prefix
        # target_key may be "AAPL_target_up_1d" or just "target_up_1d"
        bare_target = target_key.split('_', 1)[1] if '_' in target_key else target_key
        is_clf = self._is_classification_target(bare_target)

        heavy_models_list = []
        for _context_id, meta in models_metadata.items():
            if meta.get('target') != bare_target:
                continue
            if meta.get('ticker', '') != ticker:
                continue

            test_metrics = meta.get('test_metrics', {})
            if is_clf:
                # Prefer balanced_accuracy (handles imbalanced classes),
                # fall back to accuracy then f1.
                sort_score = (
                    test_metrics.get('balanced_accuracy')
                    or test_metrics.get('accuracy')
                    or test_metrics.get('f1')
                    or 0.0
                )
            else:
                sort_score = test_metrics.get('r2', 0.0)

            heavy_models_list.append({
                'type': meta.get('model_type', ''),
                'r2': test_metrics.get('r2', 0.0),   # kept for _calculate_base_confidence
                'sort_score': float(sort_score),
                'is_classification': is_clf,
                'meta': meta,
            })

        if not heavy_models_list:
            return None

        all_models = sorted(heavy_models_list, key=lambda x: x['sort_score'], reverse=True)
        return all_models[0]

    def _calculate_base_confidence(self, champion_data: (dict[str, Any] | None)
        ) ->float:
        if not champion_data:
            return 0.3

        # For classification targets the meaningful quality metric is accuracy /
        # balanced_accuracy (0..1), not R² which is always 0.0 for classifiers.
        if champion_data.get('is_classification'):
            score = champion_data.get('sort_score', 0.0)
            # Map accuracy [0.5, 1.0] → confidence [0.1, 1.0]
            # A random classifier gives ~0.5, so we anchor there.
            if score <= 0.5:
                return 0.1
            return float(min(1.0, 0.1 + (score - 0.5) * 1.8))

        # Regression: map R² → confidence
        r2 = champion_data.get('r2', 0.0)
        if r2 < -2:
            return 0.1
        elif r2 < 0:
            return float(0.2 + r2 / 2 * 0.2)
        return float(0.3 + r2 * 0.7)

    def _calculate_error_ratios(self, champion_data: (dict[str, Any] | None)
        ) ->dict[str, float]:
        if champion_data is None:
            return {'mae_ratio': 0.5, 'rmse_ratio': 0.5}

        # For classification targets mae/rmse are not meaningful metrics.
        # Return neutral 0.5 so they don't distort the final confidence formula.
        if champion_data.get('is_classification'):
            return {'mae_ratio': 0.5, 'rmse_ratio': 0.5}

        test_metrics = champion_data['meta'].get('test_metrics', {})
        rmse = test_metrics.get('rmse', 1.0)
        mae = test_metrics.get('mae', 1.0)
        return {'mae_ratio': 1.0 / (1.0 + mae * 2), 'rmse_ratio': 1.0 / (
            1.0 + rmse * 2)}

    def _calculate_anomaly_penalty(self, ticker: str, predictions: list[
        dict[str, Any]]) ->float:
        anomaly_score = 0.5
        for pred in predictions:
            if pred.get('ticker') == ticker:
                anomaly_score = pred.get('anomaly_score', 0.5)
                break
        return anomaly_score * 0.2

    def _calculate_consensus_boost(self, predictions: list[dict[str, Any]]
        ) ->float:
        positive_votes = len([p for p in predictions if p.get('predictions',
            0) > 0.001])
        total_models = len(predictions)
        return positive_votes / max(total_models, 1) * 0.1

    def _validate_with_monte_carlo(self, ticker: str) ->tuple[float, float,
        float]:
        if self.risk_metrics is None:
            return 0.5, 0.03, 1.0
        try:
            var_hist = self.risk_metrics.compute_historical_simulation_var(
                ticker, confidence_level=0.95, lookback_days=252)
            var_garch = self.risk_metrics.compute_garch_var(ticker,
                confidence_level=0.95)
            var_cf, _ = self.risk_metrics.compute_cornish_fisher_var(ticker,
                confidence_level=0.95, lookback_days=252)
            var_95 = 0.4 * var_hist + 0.35 * var_garch + 0.25 * var_cf
            stress_result = self.risk_metrics.run_stress_test({ticker: 1.0},
                scenario='market_crash')
            stress_impact = abs(stress_result['portfolio_impact'])
            var_threshold = 0.05
            position_size_factor = 1.0
            if var_95 > var_threshold:
                excess_var = var_95 - var_threshold
                reduction = excess_var / var_threshold * 0.5
                position_size_factor = max(0.1, 1.0 - reduction)
                self.logger.warning(
                    f'⚠️ Elite VaR {var_95:.3f} > {var_threshold:.3f} for {ticker}, factor reduced to {position_size_factor:.1%}'
                    )
            if stress_impact > 0.1:
                stress_reduction = min(stress_impact, 0.5)
                position_size_factor *= 1 - stress_reduction
                if self.logger.isEnabledFor(logging.DEBUG):
                    self.logger.debug(
                        f'⚠️ Stress test shows {stress_impact:.1%} loss for {ticker}, factor reduced to {position_size_factor:.1%}'
                        )
            return 0.5 * position_size_factor, var_95, position_size_factor
        except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError) as e:
            self.logger.warning(
                f'⚠️ Elite risk validation failed for {ticker}: {e}. Using safe defaults.')
            # Graceful degradation: VaR calculation failure for one ticker must
            # not abort recommendations for the whole batch. Return safe defaults
            # (50% confidence, 3% VaR, full position factor).
            return 0.5, 0.03, 1.0

    def _create_consolidated_table(self, buy_recs: list[dict[str, Any]],
        sell_recs: list[dict[str, Any]], models_metadata: dict[str, Any],
        predictions: (list[dict[str, Any]] | None)=None) ->list[dict[str, Any]
        ]:
        consolidated: list[dict[str, Any]] = []
        predictions = predictions or []
        ticker_model_preds: dict[str, list[float]] = {}
        for pred in predictions:
            t = pred.get('ticker')
            if not t:
                continue
            by_model = pred.get('predictions_by_model', {})
            if by_model:
                vals = [float(v) for v in by_model.values() if self.
                    _safe_float(v) is not None]
                ticker_model_preds.setdefault(t, []).extend(vals)
            else:
                try:
                    ticker_model_preds.setdefault(t, []).append(float(pred.
                        get('predictions', 0)))
                except (TypeError, ValueError):
                    self.logger.debug(
                        "Skipping non-numeric prediction for %s: %r",
                        t,
                        pred.get("predictions"),
                        exc_info=True,
                    )

        def build_record(rec: dict[str, Any], action: str, direction: str
            ) ->dict[str, Any]:
            ticker = rec['ticker']
            supporting_models = [m for m in models_metadata.values() if m.
                get('ticker') == ticker]
            values = ticker_model_preds.get(ticker, [])
            count = sum(1 for v in values if v > (0.001 if direction ==
                'buy' else -0.001))
            total_models = len(values) if values else max(len(
                supporting_models), 1)
            consensus = count / total_models
            return {'ticker': ticker, 'action': action, 'priority': 1 if
                action == 'BUY' else 3, 'confidence': rec['confidence'],
                'signal_strength': self._get_signal_strength(rec[
                'confidence']), 'predicted_return': rec['predicted_return'],
                'expected_return_pct':
                f"{rec['predicted_return'] * 100:.2f}%", 'current_price':
                rec['current_price'], 'champion_model': rec.get(
                'champion_model'), 'supporting_models_count': count,
                'total_models': total_models, 'model_consensus': consensus,
                'risk_score': self._calculate_risk_score(rec),
                'news_warning': rec.get('news_warning'), 'var_95': rec.get(
                'var_95', 0.03), 'position_size_factor': rec.get(
                'position_size_factor', 1.0), 'stop_loss_pct': rec.get(
                'stop_loss_pct', 0.05), 'take_profit_pct': rec.get(
                'take_profit_pct', 0.1), 'reason': rec.get('reason'),
                'composite_score': rec['confidence'] * consensus,
                'timestamp': datetime.now().isoformat()}
        for rec in buy_recs:
            consolidated.append(build_record(rec, 'BUY', 'buy'))
        for rec in sell_recs:
            consolidated.append(build_record(rec, 'SELL', 'sell'))
        consolidated.sort(key=lambda x: x['composite_score'], reverse=True)
        return consolidated

    def _safe_float(self, value: Any) ->(float | None):
        try:
            return float(value)
        except (TypeError, ValueError):
            return None
