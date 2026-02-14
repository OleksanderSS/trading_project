# utils/target_utils.py - УНІФІКОВАНА СИСТЕМА ТАРАГЕТІВ

from config.feature_config import TARGETS, TICKER_TARGET_MAP
from utils.logger import ProjectLogger
import os
import pandas as pd
import numpy as np
from typing import Dict, List
from pathlib import Path
import logging


# Setup logging
logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

logger = ProjectLogger.get_logger("TradingProjectLogger")

# Оптимальна конфandгурацandя моwhereлей with правильними andргеandми
MODEL_CONFIG = {
    # Light моwhereлand - all використовують pct_change for регресandї
    "lgbm": {"target": "pct_change", "task": "regression", "data_type": "tabular"},
    "rf": {"target": "pct_change", "task": "regression", "data_type": "tabular"},
    "linear": {"target": "pct_change", "task": "regression", "data_type": "tabular"},
    "mlp": {"target": "pct_change", "task": "regression", "data_type": "tabular"},
    
    # Heavy моwhereлand - використовують direction for класифandкацandї
    "gru": {"target": "direction", "task": "classification", "data_type": "sequence", "window_size": 30},
    "tabnet": {"target": "direction", "task": "classification", "data_type": "tabular"},
    "transformer": {"target": "direction", "task": "classification", "data_type": "sequence", "window_size": 30},
    "cnn": {"target": "direction", "task": "classification", "data_type": "sequence", "window_size": 30},
    "lstm": {"target": "direction", "task": "classification", "data_type": "sequence", "window_size": 30},
    "autoencoder": {"target": "direction", "task": "classification", "data_type": "tabular"},
    
    # Додатковand моwhereлand
    "xgb": {"target": "pct_change", "task": "regression", "data_type": "tabular"},
    "catboost": {"target": "pct_change", "task": "regression", "data_type": "tabular"},
    "svm_reg": {"target": "pct_change", "task": "regression", "data_type": "tabular"},
    "svm": {"target": "direction", "task": "classification", "data_type": "tabular"},
    "knn": {"target": "direction", "task": "classification", "data_type": "tabular"},
    "ensemble": {"target": "pct_change", "task": "regression", "data_type": "tabular"},
}

# РОЗШИРЕНІ КОНФІГУРАЦІЇ МОДЕЛЕЙ З НОВИМИ ТАРАГЕТАМИ
ENHANCED_MODEL_CONFIG = {
    # Light моделі - всі інтегровані
    "lgbm": {
        "primary_targets": ["target_volatility_5d", "target_trend_5d", "target_risk_5d"],
        "secondary_targets": ["target_volatility_20d", "target_momentum_10d", "target_behavior_5d"],
        "task": "regression",
        "integration_status": "full"
    },
    "rf": {
        "primary_targets": ["target_volatility_5d", "target_trend_5d", "target_risk_5d"],
        "secondary_targets": ["target_volatility_20d", "target_momentum_10d", "target_behavior_5d"],
        "task": "regression",
        "integration_status": "full"
    },
    "linear": {
        "primary_targets": ["target_mean_reversion", "target_risk_adjusted"],
        "secondary_targets": ["target_volatility_5d", "target_momentum_shift"],
        "task": "regression",
        "integration_status": "full",
        "data_type": "tabular"
    },
    "mlp": {
        "primary_targets": ["target_volatility_5d", "target_risk_adjusted", "target_momentum_shift", "target_mean_reversion"],
        "secondary_targets": ["target_price_acceleration", "target_breakout_probability"],
        "task": "regression",
        "integration_status": "full",
        "data_type": "tabular"
    },
    
    # Heavy моwhereлand with роwithширеними andргеandми
    "gru": {
        "primary_targets": ["target_volatility_20d", "target_max_drawdown", "target_trend_strength", "target_macro_sensitivity"],
        "secondary_targets": ["target_sharpe_ratio", "target_cycle_phase"],
        "task": "classification",
        "data_type": "sequence",
        "window_size": 30
    },
    "lstm": {
        "primary_targets": ["target_volatility_20d", "target_max_drawdown", "target_trend_strength", "target_macro_sensitivity"],
        "secondary_targets": ["target_sharpe_ratio", "target_cycle_phase"],
        "task": "classification",
        "data_type": "sequence",
        "window_size": 30
    },
    "transformer": {
        "primary_targets": ["target_volatility_20d", "target_max_drawdown", "target_trend_strength", "target_macro_sensitivity"],
        "secondary_targets": ["target_sharpe_ratio", "target_cycle_phase", "target_support_resistance"],
        "task": "classification",
        "data_type": "sequence",
        "window_size": 30
    },
    "cnn": {
        "primary_targets": ["target_volatility_20d", "target_trend_strength", "target_momentum_shift"],
        "secondary_targets": ["target_breakout_probability", "target_volume_anomaly"],
        "task": "classification",
        "data_type": "sequence",
        "window_size": 30
    },
    "tabnet": {
        "primary_targets": ["target_volatility_20d", "target_max_drawdown", "target_trend_strength"],
        "secondary_targets": ["target_sharpe_ratio", "target_mean_reversion"],
        "task": "classification",
        "data_type": "tabular"
    },
    "autoencoder": {
        "primary_targets": ["target_volatility_20d", "target_trend_strength", "target_mean_reversion"],
        "secondary_targets": ["target_cycle_phase", "target_support_resistance"],
        "task": "classification",
        "integration_status": "full"
    },
    "lgbm_bayesian": {
        "primary_targets": ["target_volatility_5d", "target_trend_5d", "target_risk_5d"],
        "secondary_targets": ["target_volatility_20d", "target_momentum_10d", "target_behavior_5d"],
        "task": "regression",
        "integration_status": "bayesian_optimized"
    },
    "dean_ensemble": {
        "primary_targets": ["target_direction_5d", "target_risk_5d", "target_behavior_5d"],
        "secondary_targets": ["target_direction_20d", "target_risk_20d", "target_behavior_20d"],
        "task": "reinforcement_learning",
        "integration_status": "rl_integrated"
    },
    "sentiment": {
        "primary_targets": ["target_direction_5d", "target_behavior_5d"],
        "secondary_targets": ["target_direction_20d", "target_behavior_20d"],
        "task": "sentiment_analysis",
        "integration_status": "sentiment_integrated"
    }
}

# КОНФІГУРАЦІЇ РОЗШИРЕНИХ ТАРАГЕТІВ
ENHANCED_TARGETS_CONFIG = {
    # Волатильнand andргети
    'volatility': {
        'targets': [
            'target_volatility_5d',
            'target_volatility_20d', 
            'target_atr_expansion',
            'target_volatility_regime'
        ],
        'description': 'Прогноwith волатильностand and її режимandв',
        'best_for': ['gru', 'lstm', 'transformer', 'lgbm'],
        'task_type': 'regression'
    },
    
    # Риwithиковand andргети
    'risk': {
        'targets': [
            'target_max_drawdown',
            'target_var_95',
            'target_sharpe_ratio',
            'target_risk_adjusted'
        ],
        'description': 'Управлandння риwithиком and риwithик-коригованand метрики',
        'best_for': ['gru', 'lstm', 'transformer', 'tabnet'],
        'task_type': 'regression'
    },
    
    # Трендовand andргети
    'trend': {
        'targets': [
            'target_trend_strength',
            'target_trend_duration',
            'target_momentum_shift',
            'target_cycle_phase'
        ],
        'description': 'Аналandwith трендandв and циклandв',
        'best_for': ['gru', 'lstm', 'transformer', 'cnn'],
        'task_type': 'classification'
    },
    
    # Поведandнковand andргети
    'behavioral': {
        'targets': [
            'target_volume_anomaly',
            'target_price_acceleration',
            'target_sentiment_momentum',
            'target_market_participation'
        ],
        'description': 'Поведandнковand аномалandї and моментум',
        'best_for': ['cnn', 'lgbm', 'rf'],
        'task_type': 'regression'
    },
    
    # Структурнand andргети
    'structural': {
        'targets': [
            'target_support_resistance',
            'target_range_bound',
            'target_breakout_probability',
            'target_mean_reversion'
        ],
        'description': 'Структурнand характеристики ринку',
        'best_for': ['lgbm', 'rf', 'linear', 'tabnet'],
        'task_type': 'classification'
    },
    
    # Макро-кореляцandйнand andргети
    'macro': {
        'targets': [
            'target_macro_sensitivity',
            'target_rate_impact',
            'target_inflation_hedge',
            'target_sector_rotation'
        ],
        'description': 'Вплив макроекономandчних факторandв',
        'best_for': ['gru', 'lstm', 'transformer'],
        'task_type': 'regression'
    }
}

# БАЗОВІ ТАРАГЕТИ (for сумandсностand)
BASE_TARGETS = {
    "heavy": {
        "column": "target_heavy",
        "type": "regression",
        "description": "Absolute price change - for heavy models (GRU, TabNet, Transformer)"
    },
    "light": {
        "column": "target_light", 
        "type": "regression",
        "description": "Percentage price change - for light models (LGBM, RF, Linear, MLP)"
    },
    "direction": {
        "column": "target_direction",
        "type": "classification", 
        "description": "Binary direction (0/1) - for classification models"
    }
}

def get_model_config(model_name: str, ticker: str, interval: str = None) -> dict:
    name = model_name.lower()
    cfg = MODEL_CONFIG.get(name)

    if cfg is None:
        logger.error(f"[target_utils] Конфandгурацandя not withнайwhereна")
        return None

def get_optimal_models_for_task(task_type: str) -> list:
    """Отримати список оптимальних моwhereлей for типу forдачand"""
    models = [name for name, cfg in MODEL_CONFIG.items() if cfg["task"] == task_type]
    logger.info(f"[target_utils] Моwhereлand for {task_type}: {models}")
    return models

def get_model_target_type(model_file: str) -> str:
    """Виwithначити тип цandлand моwhereлand with наwithви fileу"""
    filename = os.path.basename(model_file).lower()
    
    if "direction" in filename or "dir" in filename or "clf" in filename:
        return "direction"
    elif "pct" in filename or "reg" in filename or "change" in filename:
        return "pct_change"
    elif "target" in filename:
        # Аналandwithуємо структуру наwithви
        parts = filename.replace('.pkl', '').split('_')
        if len(parts) >= 4:
            target_part = parts[2]  # model_ticker_timeframe_target
            if "direction" in target_part:
                return "direction"
            elif "pct" in target_part:
                return "pct_change"
    
    return "unknown"

def generate_model_filename(model_name: str, ticker: str, timeframe: str, target_type: str, version: int = 1) -> str:
    """Згеnotрувати andм'я fileу моwhereлand with версandонуванням"""
    return f"{model_name}_{ticker}_{timeframe}_{target_type}_v{version}.pkl"

def get_model_version(model_file: str) -> int:
    """Отримати версandю моwhereлand with наwithви fileу"""
    filename = os.path.basename(model_file)
    if "_v" in filename:
        try:
            version_part = filename.split("_v")[-1].replace(".pkl", "")
            return int(version_part)
        except ValueError:
            pass
    return 1  # Default version

def is_legacy_model(model_file: str) -> bool:
    """Check чи це legacy model"""
    return "legacy" in model_file or get_model_target_type(model_file) == "unknown"

def get_models_by_type(models_dir: str = "models/trained") -> dict:
    """Отримати моwhereлand, withгрупованand for типом"""
    models = {"light": {}, "heavy": {}, "legacy": {}}
    
    for model_file in Path(models_dir).glob("*.pkl"):
        target_type = get_model_target_type(model_file)
        model_name = model_file.stem.split('_')[0].lower()
        
        if model_name in ["lgbm", "rf", "linear", "mlp", "xgb", "catboost", "svm_reg"]:
            models["light"][str(model_file)] = {"target_type": target_type, "model_name": model_name}
        elif model_name in ["gru", "tabnet", "transformer", "cnn", "lstm", "autoencoder", "svm", "knn"]:
            models["heavy"][str(model_file)] = {"target_type": target_type, "model_name": model_name}
        else:
            models["legacy"][str(model_file)] = {"target_type": target_type, "model_name": model_name}
    
    return models

def create_volatility_targets(df: pd.DataFrame) -> pd.DataFrame:
    """Створює волатильнand andргети"""
    logger.info("[UP] Створення волатильних andргетandв...")
    
    for ticker in df['ticker'].unique():
        if pd.isna(ticker):
            continue
            
        ticker_mask = df['ticker'] == ticker
        ticker_data = df[ticker_mask].copy()
        
        if 'close' not in ticker_data.columns or len(ticker_data) < 30:
            continue
        
        # Calculating волатильнandсть
        returns = ticker_data['close'].pct_change()
        
        # 5-whereнна волатильнandсть
        vol_5d = returns.rolling(5).std()
        df.loc[ticker_mask, f'target_volatility_5d_{ticker}'] = vol_5d.shift(-5)
        
        # 20-whereнна волатильнandсть
        vol_20d = returns.rolling(20).std()
        df.loc[ticker_mask, f'target_volatility_20d_{ticker}'] = vol_20d.shift(-20)
        
        # Роwithширення ATR
        if 'atr' in ticker_data.columns:
            atr_expansion = ticker_data['atr'].pct_change()
            df.loc[ticker_mask, f'target_atr_expansion_{ticker}'] = atr_expansion.shift(-5)
        
        # Режим волатильностand (класифandкацandя)
        vol_regime = pd.cut(vol_20d, bins=3, labels=['low', 'medium', 'high'])
        df.loc[ticker_mask, f'target_volatility_regime_{ticker}'] = vol_regime.shift(-5)
    
    return df

def create_risk_targets(df: pd.DataFrame) -> pd.DataFrame:
    """Створює риwithиковand andргети"""
    logger.info("[WARN] Створення риwithикових andргетandв...")
    
    for ticker in df['ticker'].unique():
        if pd.isna(ticker):
            continue
            
        ticker_mask = df['ticker'] == ticker
        ticker_data = df[ticker_mask].copy()
        
        if 'close' not in ticker_data.columns or len(ticker_data) < 50:
            continue
        
        returns = ticker_data['close'].pct_change()
        cumulative = (1 + returns).cumprod()
        
        # Максимальна просадка
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_dd = drawdown.rolling(20).min()
        df.loc[ticker_mask, f'target_max_drawdown_{ticker}'] = max_dd.shift(-20)
        
        # Value at Risk 95%
        var_95 = returns.rolling(20).quantile(0.05)
        df.loc[ticker_mask, f'target_var_95_{ticker}'] = var_95.shift(-5)
        
        # Sharpe ratio
        rolling_sharpe = returns.rolling(20).mean() / returns.rolling(20).std()
        df.loc[ticker_mask, f'target_sharpe_ratio_{ticker}'] = rolling_sharpe.shift(-20)
        
        # Риwithик-коригованand поверnotння
        risk_adjusted = returns / returns.rolling(20).std()
        df.loc[ticker_mask, f'target_risk_adjusted_{ticker}'] = risk_adjusted.shift(-5)
    
    return df

def create_trend_targets(df: pd.DataFrame) -> pd.DataFrame:
    """Створює трендовand andргети"""
    logger.info("[DATA] Створення трендових andргетandв...")
    
    for ticker in df['ticker'].unique():
        if pd.isna(ticker):
            continue
            
        ticker_mask = df['ticker'] == ticker
        ticker_data = df[ticker_mask].copy()
        
        if 'close' not in ticker_data.columns or len(ticker_data) < 50:
            continue
        
        # Сила тренду (череwith RSI)
        delta = ticker_data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        df.loc[ticker_mask, f'target_trend_strength_{ticker}'] = rsi.shift(-5)
        
        # Тривалandсть тренду
        sma_20 = ticker_data['close'].rolling(20).mean()
        trend_direction = np.where(ticker_data['close'] > sma_20, 1, -1)
        trend_duration = pd.Series(trend_direction).groupby((trend_direction != pd.Series(trend_direction).shift()).cumsum()).cumcount()
        df.loc[ticker_mask, f'target_trend_duration_{ticker}'] = trend_duration.shift(-5)
        
        # Змandна моментуму
        momentum = ticker_data['close'].pct_change(10)
        momentum_shift = momentum.diff()
        df.loc[ticker_mask, f'target_momentum_shift_{ticker}'] = momentum_shift.shift(-5)
        
        # Фаfor циклу (спрощена)
        price_position = (ticker_data['close'] - ticker_data['close'].rolling(50).min()) / \
                      (ticker_data['close'].rolling(50).max() - ticker_data['close'].rolling(50).min())
        cycle_phase = pd.cut(price_position, bins=4, labels=['accumulation', 'markup', 'distribution', 'markdown'])
        df.loc[ticker_mask, f'target_cycle_phase_{ticker}'] = cycle_phase.shift(-5)
    
    return df

def create_behavioral_targets(df: pd.DataFrame) -> pd.DataFrame:
    """Створює поведandнковand andргети"""
    logger.info("[BRAIN] Створення поведandнкових andргетandв...")
    
    for ticker in df['ticker'].unique():
        if pd.isna(ticker):
            continue
            
        ticker_mask = df['ticker'] == ticker
        ticker_data = df[ticker_mask].copy()
        
        if 'close' not in ticker_data.columns or len(ticker_data) < 20:
            continue
        
        # Аномалandя обсягandв
        if 'volume' in ticker_data.columns:
            volume_mean = ticker_data['volume'].rolling(20).mean()
            volume_std = ticker_data['volume'].rolling(20).std()
            volume_zscore = (ticker_data['volume'] - volume_mean) / volume_std
            df.loc[ticker_mask, f'target_volume_anomaly_{ticker}'] = volume_zscore.shift(-1)
        
        # Прискорення цandни
        price_change = ticker_data['close'].pct_change()
        price_acceleration = price_change.diff()
        df.loc[ticker_mask, f'target_price_acceleration_{ticker}'] = price_acceleration.shift(-3)
        
        # Моментум сентименту
        if 'sentiment_score' in ticker_data.columns:
            sentiment_momentum = ticker_data['sentiment_score'].diff()
            df.loc[ticker_mask, f'target_sentiment_momentum_{ticker}'] = sentiment_momentum.shift(-2)
        
        # Участь у ринку (активнandсть)
        market_activity = (price_change.abs() * ticker_data['close']).rolling(5).sum() if 'volume' in ticker_data.columns else price_change.abs().rolling(5).sum()
        df.loc[ticker_mask, f'target_market_participation_{ticker}'] = market_activity.shift(-5)
    
    return df

def create_structural_targets(df: pd.DataFrame) -> pd.DataFrame:
    """Створює структурнand andргети"""
    logger.info(" Створення структурних andргетandв...")
    
    for ticker in df['ticker'].unique():
        if pd.isna(ticker):
            continue
            
        ticker_mask = df['ticker'] == ticker
        ticker_data = df[ticker_mask].copy()
        
        if 'close' not in ticker_data.columns or len(ticker_data) < 50:
            continue
        
        # Рandвнand пandдтримки/опору
        highs = ticker_data['close'].rolling(20).max()
        lows = ticker_data['close'].rolling(20).min()
        support_resistance = (ticker_data['close'] - lows) / (highs - lows)
        df.loc[ticker_mask, f'target_support_resistance_{ticker}'] = support_resistance.shift(-5)
        
        # Торговля в каналand
        channel_width = (highs - lows) / ticker_data['close']
        df.loc[ticker_mask, f'target_range_bound_{ticker}'] = channel_width.shift(-5)
        
        # Ймовandрнandсть пробиття
        price_position = (ticker_data['close'] - lows) / (highs - lows)
        breakout_prob = pd.Series(np.where((price_position > 0.9) | (price_position < 0.1), 1, 0), index=ticker_data.index)
        df.loc[ticker_mask, f'target_breakout_probability_{ticker}'] = breakout_prob.shift(-3)
        
        # Поверnotння до середнього
        sma_20 = ticker_data['close'].rolling(20).mean()
        mean_reversion = (sma_20 - ticker_data['close']) / ticker_data['close']
        df.loc[ticker_mask, f'target_mean_reversion_{ticker}'] = mean_reversion.shift(-5)
    
    return df

def create_macro_targets(df: pd.DataFrame) -> pd.DataFrame:
    """Створює макро-кореляцandйнand andргети"""
    logger.info(" Створення макро-andргетandв...")
    
    # Чутливandсть до макро (кореляцandя with VIX)
    if 'vix' in df.columns:
        for ticker in df['ticker'].unique():
            if pd.isna(ticker):
                continue
                
            ticker_mask = df['ticker'] == ticker
            ticker_data = df[ticker_mask].copy()
            
            if 'close' in ticker_data.columns:
                returns = ticker_data['close'].pct_change()
                vix_corr = returns.rolling(20).corr(df['vix'])
                df.loc[ticker_mask, f'target_macro_sensitivity_{ticker}'] = vix_corr.shift(-5)
    
    # Вплив сandвок (кореляцandя with FEDFUNDS)
    if 'fedfunds' in df.columns:
        for ticker in df['ticker'].unique():
            if pd.isna(ticker):
                continue
                
            ticker_mask = df['ticker'] == ticker
            ticker_data = df[ticker_mask].copy()
            
            if 'close' in ticker_data.columns:
                returns = ticker_data['close'].pct_change()
                rate_corr = returns.rolling(20).corr(df['fedfunds'])
                df.loc[ticker_mask, f'target_rate_impact_{ticker}'] = rate_corr.shift(-10)
    
    return df

def create_all_enhanced_targets(df: pd.DataFrame) -> pd.DataFrame:
    """Створює all роwithширенand andргети"""
    logger.info("[TARGET] Створення роwithширених andргетandв...")
    
    # Створюємо all типи andргетandв
    df = create_volatility_targets(df)
    df = create_risk_targets(df)
    df = create_trend_targets(df)
    df = create_behavioral_targets(df)
    df = create_structural_targets(df)
    df = create_macro_targets(df)
    
    # Логуємо сandтистику
    target_cols = [col for col in df.columns if col.startswith('target_')]
    logger.info(f"[OK] Створено {len(target_cols)} роwithширених andргетandв")
    
    # Групуємо for типами
    target_types = {}
    for col in target_cols:
        for target_type, config in ENHANCED_TARGETS_CONFIG.items():
            if any(target in col for target in config['targets']):
                if target_type not in target_types:
                    target_types[target_type] = []
                target_types[target_type].append(col)
    
    logger.info("[DATA] Сandтистика andргетandв for типами:")
    for target_type, cols in target_types.items():
        logger.info(f"  {target_type}: {len(cols)} andргетandв")
    
    return df

def get_enhanced_model_config(model_name: str) -> dict:
    """Отримати роwithширену конфandгурацandю моwhereлand"""
    return ENHANCED_MODEL_CONFIG.get(model_name.lower(), {})

def get_optimal_targets_for_model(model_name: str) -> List[str]:
    """Отримати оптимальнand andргети for моwhereлand"""
    config = get_enhanced_model_config(model_name)
    if config:
        return config.get('primary_targets', [])
    return []

def get_secondary_targets_for_model(model_name: str) -> List[str]:
    """Отримати вториннand andргети for моwhereлand"""
    config = get_enhanced_model_config(model_name)
    if config:
        return config.get('secondary_targets', [])
    return []

def get_target_recommendations(df: pd.DataFrame) -> Dict:
    """Отримати рекомендацandї по andргеandх for конкретного даandсету"""
    recommendations = {
        'available_targets': [],
        'recommended_for_models': {},
        'target_quality': {},
        'usage_tips': {}
    }
    
    # Аналandwithуємо наявнand данand
    has_volume = 'volume' in df.columns
    has_sentiment = 'sentiment_score' in df.columns
    has_macro = any(col in df.columns for col in ['vix', 'fedfunds', 'gdp'])
    
    # Рекомендацandї for моwhereлей
    if has_volume and has_sentiment and has_macro:
        recommendations['recommended_for_models']['heavy'] = [
            'target_volatility_20d',
            'target_max_drawdown', 
            'target_trend_strength',
            'target_macro_sensitivity'
        ]
        recommendations['recommended_for_models']['light'] = [
            'target_volatility_5d',
            'target_risk_adjusted',
            'target_momentum_shift',
            'target_mean_reversion'
        ]
    else:
        # Баwithовand рекомендацandї
        recommendations['recommended_for_models']['heavy'] = [
            'target_volatility_20d',
            'target_trend_strength',
            'target_mean_reversion'
        ]
        recommendations['recommended_for_models']['light'] = [
            'target_volatility_5d',
            'target_momentum_shift'
        ]
    
    return recommendations

def get_all_target_types() -> Dict:
    """Отримати all типи andргетandв"""
    return ENHANCED_TARGETS_CONFIG

def validate_targets(df: pd.DataFrame) -> Dict:
    """Валandдацandя andргетandв в даandсетand"""
    validation_results = {
        'total_targets': 0,
        'valid_targets': 0,
        'missing_targets': [],
        'target_types_found': {},
        'quality_score': 0
    }
    
    target_cols = [col for col in df.columns if col.startswith('target_')]
    validation_results['total_targets'] = len(target_cols)
    
    # Перевandряємо якandсть andргетandв
    for col in target_cols:
        null_pct = df[col].isnull().sum() / len(df) * 100
        if null_pct < 50:  # Менше 50% пропущених withначень
            validation_results['valid_targets'] += 1
        else:
            validation_results['missing_targets'].append(col)
        
        # Групуємо for типами
        for target_type, config in ENHANCED_TARGETS_CONFIG.items():
            if any(target in col for target in config['targets']):
                if target_type not in validation_results['target_types_found']:
                    validation_results['target_types_found'][target_type] = []
                validation_results['target_types_found'][target_type].append(col)
    
    # Calculating якandсть
    if validation_results['total_targets'] > 0:
        validation_results['quality_score'] = validation_results['valid_targets'] / validation_results['total_targets']
    
    return validation_results