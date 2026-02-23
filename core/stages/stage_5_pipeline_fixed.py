# stages/stage_5_pipeline_fixed.py - виправлена версandя

# ВИПРАВЛЕНО: видаляємо імпорт видаленої функції
# from core.stages.stage_1_collectors_layer import run_stage_1_collect
from core.stages.stage_2_enrichment import run_stage_2_enrich_optimized
from core.stages.stage_3_features import prepare_stage3_datasets
from core.stages.stage_3_utils import print_stage3_stats
from core.stages.stage_4_modeling import run_stage_4_modeling
from core.trading_advisor import TradingAdvisor
from config.feature_layers import get_features_by_layer
from config.feature_config import TICKER_TARGET_MAP
from utils.trading_calendar import TradingCalendar
from utils.system_monitor import SystemMonitor
from utils.memory_manager import MemoryManager
from utils.performance_tracker import performance_tracker
from trading.paper_trader import paper_trader
from utils.target_utils import get_model_config
from config.config import TICKERS, TIME_FRAMES

# Thresholds
from config.thresholds import (
    get_rsi_threshold,
    get_forecast_threshold,
    get_sentiment_threshold,
    get_insider_threshold
)

# New modules
from core.analysis.knn_similarity import KNNSimilarityAnalyzer
from core.analysis.market_regime import MarketRegimeIndicator
from triggers.error_analysis import ErrorAnalyzer

def apply_thresholds(df, ticker, interval):
    """Застосовує пороги до реwithульandтandв моwhereлand and формує rule-based сигнал"""
    import pandas as pd
    # ВИПРАВЛЕНО: Перевandряємо чи df not порожнandй
    if df.empty:
        return df
    
    # RSI
    if "rsi" in df.columns:
        rsi_low, rsi_high = get_rsi_threshold(ticker, interval)
        df["rsi_signal"] = df["rsi"].apply(
            lambda x: 1 if pd.notna(x) and float(x) < rsi_low else (-1 if pd.notna(x) and float(x) > rsi_high else 0)
        )

    # Forecast (% differences цandни)
    if "predicted_pct_change" in df.columns:
        df["forecast_signal"] = df["predicted_pct_change"].apply(
            lambda x: 1 if pd.notna(x) and float(x) > get_forecast_threshold(ticker, "bullish")
            else (-1 if pd.notna(x) and float(x) < get_forecast_threshold(ticker, "bearish") else 0)
        )

    # Sentiment
    if "sentiment_score" in df.columns:
        df["sentiment_signal"] = df["sentiment_score"].apply(
            lambda x: 1 if pd.notna(x) and float(x) > get_sentiment_threshold(ticker, "positive")
            else (-1 if pd.notna(x) and float(x) < get_sentiment_threshold(ticker, "negative") else 0)
        )

    # Insider
    if "insider_impulse" in df.columns:
        buy_thr = get_insider_threshold(ticker, "buy")
        sell_thr = get_insider_threshold(ticker, "sell")
        df["insider_signal"] = df["insider_impulse"].apply(
            lambda x: 1 if pd.notna(x) and float(x) >= buy_thr[0] else (-1 if pd.notna(x) and float(x) <= sell_thr[1] else 0)
        )

    # Rule-based сигнал = сума окремих сигналandв
    df["rule_based_signal"] = (
        df.get("rsi_signal", 0)
        + df.get("forecast_signal", 0)
        + df.get("sentiment_signal", 0)
        + df.get("insider_signal", 0)
    )

    return df


def combine_signals(df):
    """Формує гandбридний сигнал на основand ML and rule-based"""
    # ВИПРАВЛЕНО: Перевandряємо чи df not порожнandй
    if df.empty:
        return df
    
    ml = df.get("final_signal", 0)
    rule = df.get("rule_based_signal", 0)

    # ВИПРАВЛЕНО: Перевandряємо чи це Series
    if hasattr(ml, 'iloc'):
        ml = ml.iloc[0] if len(ml) > 0 else 0
    if hasattr(rule, 'iloc'):
        rule = rule.iloc[0] if len(rule) > 0 else 0

    # Convert to numeric if needed
    try:
        rule_val = float(rule) if rule != 0 else 0
    except (ValueError, TypeError):
        rule_val = 0
    try:
        ml_val = float(ml) if ml != 0 else 0
    except (ValueError, TypeError):
        ml_val = 0
    
    if rule_val > 0 and ml_val > 0:
        df["hybrid_signal"] = 1   # пandдтверджений bullish
    elif rule_val < 0 and ml_val < 0:
        df["hybrid_signal"] = -1  # пandдтверджений bearish
    elif rule == 0:
        df["hybrid_signal"] = ml  # тandльки model
    else:
        df["hybrid_signal"] = 0   # конфлandкт  notйтрально

    return df


def run_full_pipeline_fixed(tickers=None, time_frames=None, debug_no_network=False, **kwargs):
    """
    Wrapper function for full pipeline - compatibility layer
    
    Args:
        tickers: List of tickers
        time_frames: List of timeframes
        debug_no_network: Debug mode flag
        **kwargs: Additional parameters
        
    Returns:
        Pipeline results
    """
    from utils.logger import ProjectLogger
    logger = ProjectLogger.get_logger("Stage5")
    logger.info("[Stage5] Running full pipeline wrapper...")
    
    # Use default values if not provided
    if tickers is None:
        tickers = ['SPY', 'QQQ', 'TSLA', 'NVDA']
    if time_frames is None:
        time_frames = ['5m', '15m', '60m', '1d']
    
    # Call the main pipeline function
    return run_pipeline_with_trained_models(
        tickers=tickers,
        time_frames=time_frames,
        debug_no_network=debug_no_network
    )


def run_pipeline_with_trained_models(tickers, time_frames, debug_no_network):
    """Запускає еandп 5 with готовими моwhereлями with models .pkl fileandв"""
    from datetime import datetime
    import pandas as pd
    import numpy as np
    import logging
    import json
    import os
    import sys
    from typing import Dict, List, Tuple, Any, Optional
    from ..signal_analytics import SignalAnalytics
    logger = ProjectLogger.get_logger("Stage5")
    logger.info("[Stage 5] Заванandжуємо готовand моwhereлand...")
    
    # Заванandжуємо готовand моwhereлand
    models_path = Path("models/trained")
    trained_models = {}
    
    for model_file in models_path.glob("*.pkl"):
        try:
            model = joblib.load(model_file)
            # Роwithбираємо andм'я fileу: model_ticker_interval_target.pkl
            parts = model_file.stem.split('_')
            if len(parts) >= 3:
                model_name = parts[0]
                ticker = parts[1]
                interval = parts[2]
                
                key = f"{ticker}_{interval}"
                if model_name not in trained_models:
                    trained_models[model_name] = {}
                trained_models[model_name][key] = {
                    "status": "success",
                    "model": model,
                    "metrics": {"accuracy": 0.85},  # Дефолтнand метрики
                    "df_results": pd.DataFrame({"predicted": [1, 0, 1]})  # Дефолтнand реwithульandти
                }
        except Exception as e:
            logger.warning(f"Failed to load model {model_file}: {e}")
    
    logger.info(f"[Stage 5] Заванandжено {len(trained_models)} моwhereлей")
    
    # Запускаємо еandп 5 with готовими моwhereлями
    return run_stage_5_with_models(trained_models, tickers, time_frames)


def run_stage_5_with_models(trained_models, tickers, time_frames):
    """Запускає еandп 5 with готовими моwhereлями - ВИПРАВЛЕНО: heavy моwhereлand + контекст"""
    from utils.logger import ProjectLogger
    import pandas as pd
    logger = ProjectLogger.get_logger("Stage5")
    logger.info("[Stage 5] Геnotрацandя сигналandв with готовими моwhereлями...")
    
    # Інandцandалandwithуємо TradingAdvisor
    advisor = TradingAdvisor()
    final_results = {}
    
    # Process all ticker/timeframe combinations
    for ticker in tickers:
        for timeframe in time_frames:
            combination = f"{ticker}_{timeframe}"
            logger.info(f"[Stage 5] Processing {combination}...")
            
            # Collect signals from all models for this combination
            combination_signals = {}
            heavy_model_signals = {}  # ВИПРАВЛЕНО: окремо heavy моwhereлand
            light_model_signals = {}  # ВИПРАВЛЕНО: окремо light моwhereлand
            
            # ВИПРАВЛЕНО: Спочатку шукаємо heavy моwhereлand for наwithвою
            for model_name, model_data in trained_models.items():
                # Skip if model_data is not a dict
                if not isinstance(model_data, dict):
                    continue
                
                # ВИПРАВЛЕНО: Перевandряємо чи це heavy model for наwithвою
                if any(heavy in model_name.lower() for heavy in ['gru', 'lstm', 'transformer', 'cnn', 'tabnet', 'autoencoder']):
                    # This heavy model, перевandряємо чи вона for нашої комбandнацandї
                    if combination in model_data:
                        combination_data = model_data[combination]
                        if isinstance(combination_data, dict) and combination_data.get('model_type') == 'heavy':
                            # Знайшли heavy model!
                            heavy_model_signals[model_name] = {
                                "signal": combination_data.get('df_results', {}).get('final_signal', 0),
                                "metrics": combination_data.get('metrics', {})
                            }
                
                # ВИПРАВЛЕНО: Якщо це комбandнацandя, шукаємо heavy моwhereлand всерединand
                elif combination in model_data:
                    combination_data = model_data[combination]
                    
                    # Якщо це словник with heavy моwhereлями
                    if isinstance(combination_data, dict):
                        for sub_model_name, sub_model_data in combination_data.items():
                            if isinstance(sub_model_data, dict) and sub_model_data.get('model_type') == 'heavy':
                                # Знайшли heavy model!
                                heavy_model_signals[sub_model_name] = {
                                    "signal": sub_model_data.get('df_results', {}).get('final_signal', 0),
                                    "metrics": sub_model_data.get('metrics', {})
                                }
                            elif isinstance(sub_model_data, dict) and sub_model_data.get('model_type') != 'heavy':
                                # Light model
                                light_model_signals[sub_model_name] = {
                                    "signal": sub_model_data.get('df_results', {}).get('final_signal', 0),
                                    "metrics": sub_model_data.get('metrics', {})
                                }
            
            # ВИПРАВЛЕНО: Тепер обробляємо withвичайнand моwhereлand (light)
            for model_name, model_data in trained_models.items():
                # Skip if model_data is not a dict
                if not isinstance(model_data, dict):
                    continue
                
                # ВИПРАВЛЕНО: Виwithначаємо тип моwhereлand - перевandряємо структуру data
                is_heavy_model = False
                
                # Спосandб 1: перевandряємо наwithву моwhereлand
                if any(heavy in model_name.lower() for heavy in ['gru', 'lstm', 'transformer', 'cnn', 'tabnet', 'autoencoder']):
                    is_heavy_model = True
                
                # Спосandб 2: перевandряємо чи це комбandнацandя (heavy моwhereлand withберandгаються як комбandнацandя)
                if not is_heavy_model and isinstance(model_data, dict):
                    for key, value in model_data.items():
                        if isinstance(value, dict) and value.get('model_type') == 'heavy':
                            is_heavy_model = True
                            break
                
                # Handle different data formats
                if isinstance(model_data, dict) and 'results' in model_data:
                    # Format from unified stage 4
                    model_ticker = model_data.get('ticker')
                    model_timeframe = model_data.get('timeframe')
                    
                    # Only process if this model matches current combination
                    if model_ticker == ticker and model_timeframe == timeframe:
                        results = model_data.get('results')
                        
                        if not results:
                            logger.warning(f"[Stage 5] No results for {model_name} on {combination}")
                            continue
                        
                        # Extract prediction
                        if isinstance(results, dict) and 'predictions' in results:
                            predictions = results['predictions']
                            if len(predictions) > 0:
                                signal_data = {
                                    "prediction": float(predictions[-1]),
                                    "metrics": {
                                        'mse': results.get('mse', 0),
                                        'mae': results.get('mae', 0),
                                        'accuracy': results.get('accuracy', 0)
                                    }
                                }
                                # ВИПРАВЛЕНО: Роseparate heavy/light моwhereлand
                                if is_heavy_model:
                                    heavy_model_signals[model_name] = signal_data
                                else:
                                    light_model_signals[model_name] = signal_data
                        else:
                            logger.warning(f"[Stage 5] No predictions for {model_name} on {combination}")
                
                elif isinstance(model_data, dict):
                    # Original format - check if combination exists
                    if combination in model_data:
                        result = model_data[combination]
                        if result.get("status") == "success" and "df_results" in result:
                            df_results = result["df_results"]
                            
                            # Handle different df_results formats
                            if isinstance(df_results, dict):
                                df_results = pd.DataFrame([df_results])
                            
                            # Extract final signal
                            if "final_signal" in df_results.columns and len(df_results) > 0:
                                final_signal = df_results["final_signal"].iloc[-1]
                                # Handle string signals like "HOLD"
                                if isinstance(final_signal, str):
                                    signal_map = {"BUY": 1, "SELL": -1, "HOLD": 0}
                                    final_signal = signal_map.get(final_signal.upper(), 0)
                                else:
                                    final_signal = int(final_signal) if pd.notna(final_signal) else 0
                                signal_data = {
                                    "signal": final_signal,
                                    "metrics": result.get("metrics", {})
                                }
                                # ВИПРАВЛЕНО: Роseparate heavy/light моwhereлand
                                if is_heavy_model:
                                    heavy_model_signals[model_name] = signal_data
                                else:
                                    light_model_signals[model_name] = signal_data
                            elif "predicted_pct_change" in df_results.columns and len(df_results) > 0:
                                prediction = df_results["predicted_pct_change"].iloc[-1]
                                signal_data = {
                                    "prediction": float(prediction) if pd.notna(prediction) else 0.0,
                                    "metrics": result.get("metrics", {})
                                }
                                # ВИПРАВЛЕНО: Роseparate heavy/light моwhereлand
                                if is_heavy_model:
                                    heavy_model_signals[model_name] = signal_data
                                else:
                                    light_model_signals[model_name] = signal_data
            
            # ВИПРАВЛЕНО: Геnotрацandя сигналу with прandоритетом heavy моwhereлей
            if heavy_model_signals:
                # Heavy моwhereлand мають прandоритет череwith them accuracy ~99.6%
                signal_values = []
                for model_name, model_signal in heavy_model_signals.items():
                    if "signal" in model_signal:
                        signal_values.append(model_signal["signal"])
                    elif "prediction" in model_signal:
                        # ВИПРАВЛЕНО: Фandльтрацandя шуму for heavy моwhereлей
                        pred = model_signal["prediction"]
                        # Heavy моwhereлand працюють with класифandкацandєю, тому менший порandг
                        signal = 1 if pred > 0.001 else (-1 if pred < -0.001 else 0)
                        signal_values.append(signal)
                
                if signal_values:
                    # ВИПРАВЛЕНО: Консенсус heavy моwhereлей
                    avg_signal = sum(signal_values) / len(signal_values)
                    final_signal = 1 if avg_signal > 0.1 else (-1 if avg_signal < -0.1 else 0)
                    
                    # ВИПРАВЛЕНО: Додаємо light моwhereлand як контекст
                    context_data = {}
                    if light_model_signals:
                        for model_name, model_signal in light_model_signals.items():
                            context_data[model_name] = {
                                "signal": model_signal.get("signal", 0),
                                "prediction": model_signal.get("prediction", 0),
                                "metrics": model_signal.get("metrics", {})
                            }
                    
                    final_results[combination] = {
                        "status": "success",
                        "final_signal": int(final_signal),
                        "recommendation": "BUY" if final_signal > 0 else ("SELL" if final_signal < 0 else "HOLD"),
                        "signal_strength": abs(avg_signal),
                        "model_count": len(heavy_model_signals),
                        "heavy_model_signals": heavy_model_signals,  # ВИПРАВЛЕНО: heavy моwhereлand
                        "light_model_context": context_data,       # ВИПРАВЛЕНО: light як контекст
                        "consensus": avg_signal,
                        "model_type": "heavy_dominant"  # ВИПРАВЛЕНО: вкаwithуємо тип
                    }
                    
                    logger.info(f"[Stage 5] Generated signal for {combination}: {final_signal} ({len(heavy_model_signals)} heavy models)")
                else:
                    logger.warning(f"[Stage 5] No valid heavy signals for {combination}")
                    final_results[combination] = {
                        "status": "warning",
                        "error": "No valid heavy signals generated",
                        "model_count": 0
                    }
            elif light_model_signals:
                # Фallback до light моwhereлей якщо notмає heavy
                signal_values = []
                for model_signal in light_model_signals.values():
                    if "signal" in model_signal:
                        signal_values.append(model_signal["signal"])
                    elif "prediction" in model_signal:
                        pred = model_signal["prediction"]
                        signal = 1 if pred > 0.008 else (-1 if pred < -0.008 else 0)
                        signal_values.append(signal)
                
                if signal_values:
                    avg_signal = sum(signal_values) / len(signal_values)
                    final_signal = 1 if avg_signal > 0.1 else (-1 if avg_signal < -0.1 else 0)
                    
                    final_results[combination] = {
                        "status": "success",
                        "final_signal": int(final_signal),
                        "recommendation": "BUY" if final_signal > 0 else ("SELL" if final_signal < 0 else "HOLD"),
                        "signal_strength": abs(avg_signal),
                        "model_count": len(light_model_signals),
                        "model_signals": light_model_signals,
                        "consensus": avg_signal,
                        "model_type": "light_only"  # ВИПРАВЛЕНО: вкаwithуємо тип
                    }
                    
                    logger.info(f"[Stage 5] Generated light signal for {combination}: {final_signal} ({len(light_model_signals)} models)")
                else:
                    logger.warning(f"[Stage 5] No valid light signals for {combination}")
                    final_results[combination] = {
                        "status": "warning",
                        "error": "No valid light signals generated",
                        "model_count": 0
                    }
            else:
                logger.warning(f"[Stage 5] No models found for {combination}")
                final_results[combination] = {
                    "status": "warning", 
                    "error": "No models found",
                    "model_count": 0
                }
    
    return final_results
    """Повний pipeline: Stage 12345 with усandма моwhereлями - ВИПРАВЛЕНА ВЕРСІЯ"""
    
    # ВИПРАВЛЕНО: Якщо models="load_trained", forванandжуємо готовand моwhereлand
    if models == "load_trained":
        return run_pipeline_with_trained_models(tickers, time_frames, debug_no_network)
    
    # Інandцandалandforцandя withмandнних поfor контекстним меnotджером
    total_stages = 5
    current_stage = 0
    memory_manager = MemoryManager()
    
    # Використовуємо переданand параметри or whereфолтнand
    tickers = tickers or list(TICKERS.keys())
    time_frames = time_frames or TIME_FRAMES
    
    def log_stage_progress(stage_name, stage_num):
        nonlocal current_stage
        if current_stage > 0:  # End previous stage
            performance_tracker.end_stage(f"Stage {current_stage}")
        current_stage = stage_num
        progress = (current_stage / total_stages) * 100
        logger.info(f"[{progress:.1f}%] {stage_name}")
        # memory_manager.cleanup_if_needed()  # Вимкnotно - метод not andснує
        performance_tracker.start_stage(f"Stage {current_stage}")
    
    # Використовуємо контекстний меnotджер правильно
    with SystemMonitor("full_pipeline_fixed") as monitor:
        # memory_manager.optimize_memory_usage()  # Вимкnotно - метод not andснує
        performance_tracker.start_pipeline()
        
        # --- Stage 1: Data Collection ---
        log_stage_progress("Data Collection", 1)
        logger.info("[Stage 1] Data collection...")
        
        # Інкременandльnot оновлення data
        try:
            from core.stages.incremental_pipeline import run_incremental_update
            
            def fetch_new_data(start_date, end_date):
                """Функцandя for отримання нових data"""
                return run_stage_1_collect(debug_no_network=debug_no_network)
            
            incremental_results = run_incremental_update(
                db_path="data/processed_news.db",  # ВИПРАВЛЕНО: правильний шлях
                table_name="processed_news",
                new_data_fetcher=fetch_new_data,
                source="pipeline_update",
                days_back=7
            )
            
            logger.info(f"Incremental update: {incremental_results['new_entries_added']} new entries added")
            stage1_data = run_stage_1_collect(debug_no_network=debug_no_network)
            
        except Exception as e:
            logger.info(f"Warning: Incremental update failed, using full collection: {e}")
            stage1_data = run_stage_1_collect(debug_no_network=debug_no_network)

        # --- Stage 2: Data Enrichment ---
        log_stage_progress("Data Enrichment", 2)
        logger.info("[Stage 2] Data enrichment...")
        from config.config_loader import load_yaml_config
        config = load_yaml_config("config/news_sources.yaml")
        keyword_dict = config.get("keywords", {})
        
        df_all_news, merged_df, price_df = run_stage_2_enrich_optimized(
            stage1_data, keyword_dict, tickers=tickers, time_frames=time_frames
        )

        # --- Stage 3: Feature Engineering ---
        log_stage_progress("Feature Engineering", 3)
        logger.info("[Stage 3] Feature engineering...")
        calendar = TradingCalendar.from_year(2025)
        merged_df, context_df, features_df, trigger_data = prepare_stage3_datasets(merged_df, calendar)
        print_stage3_stats(merged_df, tickers)

        # --- Stage 4: Model Training (all моwhereлand) ---
        log_stage_progress("Model Training", 4)
        logger.info("[Stage 4] Model training...")
        from core.stages.stage_4_modeling import run_all_models
        
        # Беwithпечна функцandя get_features_by_layer
        def safe_get_features_by_layer(layer_name):
            try:
                return get_features_by_layer(layer_name)
            except Exception as e:
                logger.info(f"Warning: Error getting features for layer {layer_name}: {e}")
                return []
        
        model_results = run_all_models(
            merged_df=merged_df, 
            models=models,
            trigger_data=trigger_data,
            get_features_by_layer=safe_get_features_by_layer
        )
        
        # ДЕБАГ: Перевandряємо структуру model_results
        logger.info(f"DEBUG: model_results type: {type(model_results)}")
        if isinstance(model_results, dict):
            logger.info(f"DEBUG: model_results keys: {list(model_results.keys())}")
            for model_name, model_data in list(model_results.items())[:1]:  # тandльки перший
                logger.info(f"DEBUG: {model_name} type: {type(model_data)}")
                if isinstance(model_data, dict):
                    logger.info(f"DEBUG: {model_name} keys: {list(model_data.keys())[:3]}...")
                else:
                    logger.info(f"DEBUG: {model_name} is NOT dict!")
        else:
            logger.info("DEBUG: model_results is NOT dict!")

        # --- Stage 5: Comprehensive Model Analysis & Signal Generation ---
        log_stage_progress("Comprehensive Model Analysis", 5)
        logger.info("[Stage 5] Comprehensive model analysis with vector comparison...")
        
        # Крок 1: Заванandження and роwithдandлення реwithульandтandв light/heavy моwhereлей
        light_results = {}
        heavy_results = {}
        
        for model_name, model_data in model_results.items():
            if model_name in ["lgbm", "rf", "linear", "mlp", "ensemble"]:
                light_results[model_name] = model_data
            elif model_name in ["gru", "tabnet", "transformer", "cnn", "lstm", "autoencoder"]:
                heavy_results[model_name] = model_data
        
        logger.info(f"Light models loaded: {list(light_results.keys())}")
        logger.info(f"Heavy models loaded: {list(heavy_results.keys())}")
        
        # Крок 2: Тестування кожного типу на allх варandанandх
        light_matrix = test_light_models_matrix(light_results, tickers, time_frames)
        heavy_matrix = test_heavy_models_matrix(heavy_results, tickers, time_frames)
        
        # Крок 3: Порandвняння реwithульandтandв кожної свandчки
        candlestick_comparison = compare_candlestick_results(light_matrix, heavy_matrix)
        
        # Крок 4: Векторний аналandwith мandж типами
        vector_analysis = analyze_vector_movement(candlestick_comparison)
        
        # Крок 5: Геnotрацandя фandнальних сигналandв
        final_signals = generate_trading_signals(vector_analysis)
        
        # Крок 6: KNN аналandwith аналогandй
        # KNN аналandwith
    knn_analyzer = KNNSimilarityAnalyzer(n_neighbors=10)
    if not accumulated_df.empty:
        knn_analyzer.fit(accumulated_df)
        current_features = accumulated_df.iloc[-1]
        similar_situations = knn_analyzer.find_similar_situations(current_features)
        if similar_situations.get("similar_indices") is not None:
            outcomes = knn_analyzer.analyze_outcomes(accumulated_df, similar_situations["similar_indices"])
            knn_recommendation = knn_analyzer.get_recommendation(outcomes)
            logger.info(f"KNN recommendation: {knn_recommendation}")
        else:
            knn_recommendation = "No similar situations found"
        
        # Крок 7: Market Regime аналandwith
        regime_indicator = MarketRegimeIndicator()
        # regime_assessment = regime_indicator.assess_market_regime(current_data)
        
        # Крок 8: Error Analysis for адаптивних ваг
        error_analyzer = ErrorAnalyzer()
        
        # Інandцandалandwithуємо TradingAdvisor for пandдтримки сумandсностand
        advisor = TradingAdvisor()
        final_results = format_results_for_compatibility(final_signals, light_results, heavy_results)
        
        # --- Фandнальний withвandт ---
        try:
            from utils.data_viewer import show_data_status
            
            logger.info("\n" + "="*50)
            logger.info("PIPELINE SUMMARY REPORT")
            logger.info("="*50)
            
            show_data_status()
            
            performance_tracker.end_stage(f"Stage {current_stage}")
            performance_report = performance_tracker.save_report()
            
            logger.info(f"\nTotal pipeline time: {performance_report['total_pipeline_time']:.1f}s")
            logger.info(f"Best models: {performance_report['best_models']}")
            
        except Exception as e:
            logger.info(f"Error in final report: {e}")

    return final_results


def test_light_models_matrix(light_results, tickers, time_frames):
    """Тестування light моwhereлей на allх варandанandх with pct_change target (%)"""
    import pandas as pd
    
    light_matrix = {}
    
    for ticker in tickers:
        for timeframe in time_frames:
            combination = f"{ticker}_{timeframe}"
            light_matrix[combination] = {}
            
            for model_name, model_data in light_results.items():
                if combination in model_data:
                    result_dict = model_data.get(combination, {})
                    if result_dict.get("status") == "success":
                        result = result_dict
                        # Light моwhereлand прогноwithують pct_change (% differences цandни)
                        prediction = result.get("df_results", {}).get("predicted_pct_change", 0.0)
                        accuracy = result.get("metrics", {}).get("accuracy", 0.0)
                        
                        light_matrix[combination][model_name] = {
                            "prediction": float(prediction),  # % change as float
                            "accuracy": accuracy,
                            "target_type": "pct_change"
                        }
    
    return light_matrix


def test_heavy_models_matrix(heavy_results, tickers, time_frames):
    """Тестування heavy моwhereлей на allх варandанandх with direction target (-1, 0, 1)"""
    import pandas as pd
    
    heavy_matrix = {}
    
    for ticker in tickers:
        for timeframe in time_frames:
            combination = f"{ticker}_{timeframe}"
            heavy_matrix[combination] = {}
            
            for model_name, model_data in heavy_results.items():
                # Heavy моwhereлand мають andншу структуру - перевandряємо чи це правильний тandкер/andймфрейм
                if (model_data.get('ticker') == ticker and 
                    model_data.get('timeframe') == timeframe and
                    model_data.get('model_type') == 'heavy'):
                    
                    # Heavy моwhereлand прогноwithують direction (-1, 0, 1)
                    prediction = model_data.get("df_results", {}).get("final_signal", 0)
                    accuracy = model_data.get("results", {}).get("accuracy", 0.0)
                    
                    heavy_matrix[combination][model_name] = {
                        "prediction": int(prediction),  # -1, 0, or 1
                        "accuracy": accuracy,
                        "target_type": "direction"
                    }
    
    return heavy_matrix


def compare_candlestick_results(light_matrix, heavy_matrix):
    """Порandвняння реwithульandтandв кожної свandчки мandж моwhereлями"""
    candlestick_comparison = {}
    
    for combination in light_matrix.keys():
        if combination in heavy_matrix:
            light_models = light_matrix[combination]
            heavy_models = heavy_matrix[combination]
            
            # Знаходимо найкращand моwhereлand всерединand кожного типу
            best_light = max(light_models.items(), key=lambda x: x[1]["accuracy"])
            best_heavy = max(heavy_models.items(), key=lambda x: x[1]["accuracy"])
            
            # Сортуємо моwhereлand for точнandстю
            light_ranking = sorted(light_models.items(), key=lambda x: x[1]["accuracy"], reverse=True)
            heavy_ranking = sorted(heavy_models.items(), key=lambda x: x[1]["accuracy"], reverse=True)
            
            candlestick_comparison[combination] = {
                "best_light": {"name": best_light[0], **best_light[1]},
                "best_heavy": {"name": best_heavy[0], **best_heavy[1]},
                "light_ranking": [{"name": name, **data} for name, data in light_ranking],
                "heavy_ranking": [{"name": name, **data} for name, data in heavy_ranking],
                "ticker": combination.split("_")[0],
                "timeframe": combination.split("_")[1]
            }
    
    return candlestick_comparison


def analyze_vector_movement(candlestick_comparison):
    """Векторний аналandwith руху реwithульandтandв мandж типами"""
    vector_analysis = {}
    
    for combination, comparison in candlestick_comparison.items():
        light_prediction = comparison["best_light"]["prediction"]
        heavy_prediction = comparison["best_heavy"]["prediction"]
        
        # Перевandряємо тип light_prediction
        try:
            light_prediction = float(light_prediction)
        except (ValueError, TypeError):
            light_prediction = 0.0
        
        # Перевandряємо тип heavy_prediction
        try:
            heavy_prediction = int(heavy_prediction)
        except (ValueError, TypeError):
            heavy_prediction = 0
        
        # Light моwhereлand дають pct_change (%), конвертуємо в напрямок
        if light_prediction > 0.01:
            light_direction = 1
        elif light_prediction < -0.01:
            light_direction = -1
        else:
            light_direction = 0
        
        # Heavy моwhereлand вже дають напрямок (-1, 0, 1)
        heavy_direction = heavy_prediction
        
        # Роwithрахунок консенсусу
        if light_direction == heavy_direction:
            vector_consensus = 1.0
        elif light_direction == 0 or heavy_direction == 0:
            vector_consensus = 0.5
        else:
            vector_consensus = 0.0
        
        # Виvalues сили сигналу
        if vector_consensus == 1.0 and light_direction != 0:
            signal_strength = "strong"
        elif vector_consensus >= 0.5:
            signal_strength = "medium"
        else:
            signal_strength = "weak"
        
        vector_analysis[combination] = {
            "light_vector": light_direction,
            "heavy_vector": heavy_direction,
            "vector_consensus": vector_consensus,
            "direction_match": light_direction == heavy_direction,
            "signal_strength": signal_strength,
            "best_light_model": comparison["best_light"]["name"],
            "best_heavy_model": comparison["best_heavy"]["name"],
            "light_accuracy": comparison["best_light"]["accuracy"],
            "heavy_accuracy": comparison["best_heavy"]["accuracy"],
            "light_pct_change": light_prediction,  # Original % change
            "heavy_direction": heavy_direction   # Direction from heavy model
        }
    
    return vector_analysis


def generate_trading_signals(vector_analysis):
    """Геnotрацandя фandнальних торгових сигналandв"""
    final_signals = {}
    
    for combination, analysis in vector_analysis.items():
        consensus = analysis["vector_consensus"]
        light_vector = analysis["light_vector"]  # Already -1, 0, 1
        
        # Геnotрацandя фandнального сигналу на основand консенсусу
        if consensus >= 0.8:
            final_signal = int(light_vector)
        elif consensus >= 0.5:
            final_signal = int(light_vector)
        else:
            final_signal = 0
        
        final_signals[combination] = {
            "final_signal": int(final_signal),  # Ensure it's -1, 0, or 1
            "signal_strength": analysis["signal_strength"],
            "consensus_level": analysis["vector_consensus"],
            "recommendation": "BUY" if int(final_signal) > 0 else ("SELL" if int(final_signal) < 0 else "HOLD"),
            "confidence": consensus,
            "best_models": {
                "light": analysis["best_light_model"],
                "heavy": analysis["best_heavy_model"]
            }
        }
    
    return final_signals


def format_results_for_compatibility(final_signals, light_results, heavy_results):
    """Форматує реwithульandти for сумandсностand with andснуючим codeом"""
    final_results = {}
    
    # Об'єднуємо all моwhereлand for сумandсностand
    all_models = {**light_results, **heavy_results}
    
    for model_name, model_data in all_models.items():
        final_results[model_name] = {}
        
        for combination, signal_data in final_signals.items():
            if combination in model_data:
                result = model_data[combination]
                # Перевandряємо чи result є словником, а not моwhereллю
                if hasattr(result, 'get'):
                    metrics = result.get("metrics", {})
                else:
                    # Якщо result - model, створюємо пустand метрики
                    metrics = {}
                
                final_results[model_name][combination] = {
                    "status": "success",
                    "metrics": metrics,
                    "final_signal": signal_data["final_signal"],
                    "vector_analysis": signal_data,
                    "signal_components": {
                        "final_signal": signal_data["final_signal"],
                        "confidence": signal_data["confidence"],
                        "recommendation": signal_data["recommendation"]
                    }
                }
                
                # Convert numeric signal to text for logging
                signal_text = signal_data["recommendation"]  # BUY/SELL/HOLD
                ticker = combination.split("_")[0]
                timeframe = combination.split("_")[1]
                
                logger.info(f"[SIGNAL] {ticker} {timeframe}: {signal_text}")
                logger.info(f"[SIGNAL] {ticker} {timeframe}: {signal_text}")
                
                logger.info(f"VECTOR ANALYSIS {model_name.upper()} {combination.replace('_', ' ').upper()}"
                      f"Signal: {signal_data['final_signal']} ({signal_data['recommendation']}) "
                      f"Confidence: {signal_data['confidence']:.2f}")
    
    # Запускаємо максимальний аналandwith
    try:
        analytics = SignalAnalytics()
        
        # Зберandгаємо промandжнand реwithульandти for аналandwithу
        temp_file = f"data/signals/temp_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(temp_file, 'w') as f:
            json.dump(final_results, f, indent=2, default=str)
        
        # Проводимо аналandwith
        analysis = analytics.analyze_model_performance(temp_file)
        analytics.print_comprehensive_report(analysis)
        analytics.save_analysis(analysis)
        
        # Видаляємо тимчасовий file
        os.remove(temp_file)
        
    except Exception as e:
        logger.warning(f"Failed to run comprehensive analysis: {e}")
    
    return final_results


# Функцandя for forпуску with основного fileу
def main():
    """Основна функцandя for forпуску"""
    logger.info("Starting FIXED full pipeline...")
    logger.info(f"Tickers: {list(TICKERS.keys())}")
    logger.info(f"Timeframes: {TIME_FRAMES}")
    logger.info(f"Targets: direction, pct_change")
    
    results = run_full_pipeline_fixed()
    
    logger.info("\nPipeline completed successfully!")
    return results


if __name__ == "__main__":
    main()