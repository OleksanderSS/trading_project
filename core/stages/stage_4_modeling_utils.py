# core/stages/stage_4_modeling_utils.py

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from utils.logger import ProjectLogger
from datetime import datetime, timedelta
import warnings

logger = ProjectLogger.get_logger("Stage4Utils")


def detect_target_leakage(df: pd.DataFrame, target_col: str, macro_cols: list = None) -> dict:
    """
    Перевandряє наявнandсть Target Leakage (витоку data)
    
    Args:
        df: DataFrame with фandчами and andргетом
        target_col: Наwithва цandльової колонки
        macro_cols: Список макро-колонок for перевandрки
        
    Returns:
        dict: Реwithульandти перевandрки на leakage
    """
    logger.info(f"[Stage4Utils] [SEARCH] Detecting target leakage for {target_col}...")
    
    leakage_results = {
        'has_leakage': False,
        'issues': [],
        'warnings': [],
        'correlations': {}
    }
    
    # 1. Перевandрка часової послandдовностand
    if 'published_at' in df.columns or 'date' in df.columns:
        date_col = 'published_at' if 'published_at' in df.columns else 'date'
        df_sorted = df.sort_values(date_col)
        
        # Перевandряємо чи майбутнand данand потрапили в фandчand
        for col in df.columns:
            if col != target_col and col != date_col:
                # Шукаємо майбутнand andндикатори в наwithвах
                future_indicators = ['next_', 'future_', 'target_', 'tomorrow_', 'next_day_']
                
                for indicator in future_indicators:
                    if col.startswith(indicator) and not col == target_col:
                        leakage_results['has_leakage'] = True
                        leakage_results['issues'].append(f"Future indicator found: {col}")
                        logger.warning(f"[Stage4Utils] [WARN] Future indicator detected: {col}")
    
    # 2. Перевandрка макро-data на витandк
    if macro_cols:
        for macro_col in macro_cols:
            if macro_col in df.columns and target_col in df.columns:
                # Calculating кореляцandю
                correlation = df[macro_col].corr(df[target_col])
                leakage_results['correlations'][macro_col] = correlation
                
                # Висока кореляцandя may вкаwithувати на leakage
                if abs(correlation) > 0.9:
                    leakage_results['has_leakage'] = True
                    leakage_results['issues'].append(f"High correlation with {macro_col}: {correlation:.3f}")
                    logger.warning(f"[Stage4Utils] [WARN] High correlation detected: {macro_col} = {correlation:.3f}")
                elif abs(correlation) > 0.7:
                    leakage_results['warnings'].append(f"Moderate correlation with {macro_col}: {correlation:.3f}")
    
    # 3. Перевandрка на andwhereальнand прогноwithи
    if target_col in df.columns:
        target_values = df[target_col].dropna()
        if len(target_values) > 0:
            # Якщо all прогноwithи andwhereально точнand - це пandдоwithрandло
            if target_col.startswith('target_heavy_'):  # Бandнарна класифandкацandя
                accuracy = (target_values == target_values.shift(1)).mean() if len(target_values) > 1 else 0
                if accuracy > 0.95:
                    leakage_results['has_leakage'] = True
                    leakage_results['issues'].append(f"Suspiciously high accuracy: {accuracy:.3f}")
                    logger.warning(f"[Stage4Utils] [WARN] Suspicious accuracy: {accuracy:.3f}")
            
            # Перевandрка на аномально ниwithьку волатильнandсть andргету
            target_std = target_values.std()
            if target_std < 0.01:  # Дуже ниwithька волатильнandсть
                leakage_results['warnings'].append(f"Very low target volatility: {target_std:.6f}")
    
    # 4. Перевandрка часових withон
    if 'published_at' in df.columns and target_col in df.columns:
        df['published_at'] = pd.to_datetime(df['published_at'], errors='coerce')
        
        # Перевandряємо чи andргет not використовує данand with майбутнього
        for idx, row in df.iterrows():
            if pd.notna(row['published_at']):
                # Якщо є майбутнand цandни в фandчах for тandєї ж дати
                future_price_cols = [col for col in df.columns if 'next_' in col and 'close' in col]
                for col in future_price_cols:
                    if pd.notna(row[col]) and col != target_col:
                        # Перевandряємо чи майбутня цandна not with тandєї ж дати
                        if 'date' in col.lower() or 'time' in col.lower():
                            leakage_results['warnings'].append(f"Future price in features: {col}")
    
    # Логуємо реwithульandти
    if leakage_results['has_leakage']:
        logger.error(f"[Stage4Utils] [ERROR] TARGET LEAKAGE DETECTED!")
        for issue in leakage_results['issues']:
            logger.error(f"[Stage4Utils] - {issue}")
    else:
        logger.info(f"[Stage4Utils] [OK] No target leakage detected")
    
    for warning in leakage_results['warnings']:
        logger.warning(f"[Stage4Utils] [WARN] {warning}")
    
    return leakage_results


def compute_enhanced_metrics(y_true, y_pred, y_pred_proba=None, returns=None):
    """
    Обчислює роwithширенand метрики включаючи Sharpe Ratio and Maximum Drawdown
    
    Args:
        y_true: Справжнand values
        y_pred: Прогноwithованand values
        y_pred_proba: Ймовandрностand прогноwithandв (for класифandкацandї)
        returns: Рядок дохandдностей for роwithрахунку фandнансових метрик
        
    Returns:
        dict: Роwithширенand метрики
    """
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_absolute_error, r2_score
    
    metrics = {}
    
    # Баwithовand метрики
    if len(np.unique(y_true)) <= 10:  # Класифandкацandя
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['precision'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        metrics['recall'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        metrics['f1'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        
        if y_pred_proba is not None:
            from sklearn.metrics import roc_auc_score
            try:
                metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba, multi_class='ovr', average='weighted')
            except:
                metrics['roc_auc'] = 0.0
    else:  # Регресandя
        metrics['mae'] = mean_absolute_error(y_true, y_pred)
        metrics['r2'] = r2_score(y_true, y_pred)
    
    # Фandнансовand метрики (якщо є данand про дохandднandсть)
    if returns is not None and len(returns) > 0:
        # Створюємо торговand сигнали на основand прогноwithandв
        if len(np.unique(y_true)) <= 10:  # Класифandкацandя
            # Для бandнарної класифandкацandї: 1 = Buy, -1 = Sell, 0 = Hold
            signals = np.where(y_pred > 0, 1, np.where(y_pred < 0, -1, 0))
        else:  # Регресandя
            # Для регресandї: поwithитивний прогноwith = Buy, notгативний = Sell
            signals = np.where(y_pred > 0, 1, -1)
        
        # Calculating стратегandю дохandдностand
        strategy_returns = returns * signals
        
        # Sharpe Ratio
        if len(strategy_returns) > 1:
            excess_returns = strategy_returns - np.mean(strategy_returns)  # Простий пandдхandд
            sharpe_ratio = np.mean(excess_returns) / np.std(excess_returns) if np.std(excess_returns) > 0 else 0
            metrics['sharpe_ratio'] = sharpe_ratio * np.sqrt(252)  # Annualized
        
        # Maximum Drawdown
        cumulative_returns = np.cumprod(1 + strategy_returns)
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = np.min(drawdown) if len(drawdown) > 0 else 0
        metrics['max_drawdown'] = max_drawdown
        
        # Win Rate
        win_rate = np.mean(strategy_returns > 0) if len(strategy_returns) > 0 else 0
        metrics['win_rate'] = win_rate
        
        # Profit Factor
        gross_profit = np.sum(strategy_returns[strategy_returns > 0])
        gross_loss = np.abs(np.sum(strategy_returns[strategy_returns < 0]))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.inf
        metrics['profit_factor'] = profit_factor
    
    return metrics


def compute_layer_contributions(model, feature_names, feature_to_layer_map, X=None, y=None):
    """
    Обчислює вnotсок шарandв у model.
    - Дерев'янand моwhereлand (RF, LGBM, XGB, CatBoost): .feature_importances_
    - Лandнandйнand моwhereлand (Linear, SVM with лandнandйним ядром): coef_
    - Іншand моwhereлand (MLP, CNN, Transformer, Autoencoder): permutation importance (потрandбнand X, y)
    """

    importances = None

    # --- 1) Дерев'янand моwhereлand ---
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_

    # --- 2) Лandнandйнand моwhereлand ---
    elif hasattr(model, "coef_"):
        coef = model.coef_
        if coef.ndim > 1:  # багатокласова класифandкацandя
            coef = np.mean(np.abs(coef), axis=0)
        importances = np.abs(coef)

    # --- 3) Іншand моwhereлand (MLP, CNN, Transformer, Autoencoder) ---
    elif X is not None and y is not None:
        try:
            result = permutation_importance(model, X, y, n_repeats=5, random_state=42)
            importances = result.importances_mean
        except Exception:
            importances = np.zeros(len(feature_names))

    # --- 4) Якщо нandчого not спрацювало ---
    if importances is None:
        return pd.DataFrame(columns=["importance_sum", "percent"])

    # --- SAFEGUARD: перевandрка довжин ---
    if len(importances) != len(feature_names):
        logger.warning(
            f"[Stage4Utils] [WARN] Несинхроннand довжини: features={len(feature_names)}, importances={len(importances)}")
        return pd.DataFrame(columns=["importance_sum", "percent"])

    # --- 5) Формуємо DataFrame по шарах ---
    df_imp = pd.DataFrame({"feature": feature_names, "importance": importances})
    df_imp["layer"] = df_imp["feature"].map(feature_to_layer_map).fillna("unknown")

    contrib = (df_imp.groupby("layer")["importance"]
               .sum()
               .sort_values(ascending=False))
    total = contrib.sum() or 1.0
    percent = (contrib / total * 100).round(2)

    # --- 6) Логування топфandчей по шарах ---
    for layer in contrib.index:
        top_feats = (df_imp[df_imp["layer"] == layer]
                     .sort_values("importance", ascending=False)
                     .head(5))
        logger.info(f"[Stage4Utils] Layer={layer}, top features: {top_feats['feature'].tolist()}")

    return pd.DataFrame({"importance_sum": contrib, "percent": percent})

def assess_context_and_choose_target(df, ticker, interval, final_features, candidate_targets):
    """
    GEMINI OPTIMIZATION: Чandтка прив'яwithка Heavy/Light моwhereлей до правильних andргетandв
    
    Heavy Models (LSTM/Transformer/GRU): target_heavy_* - бandнарна класифandкацandя (1, -1, 0)
    Light Models (LGBM/XGB/RF): target_light_* - регресandя (% differences)
    """
    t = ticker.lower()
    
    # Виwithначаємо тип моwhereлand яка will useсь
    # This має передаватись withwithовнand череwith model_name параметр
    
    # Якщо є heavy andргети - використовуємо them for Heavy моwhereлей
    heavy_targets = [c for c in candidate_targets if c.startswith(f"target_heavy_{t}_{interval}")]
    if heavy_targets:
        task_type = "classification"
        target_col = heavy_targets[0]
        logger.info(f"[Stage4Utils] Heavy models: using {target_col} (binary classification)")
        return task_type, target_col
    
    # Якщо є light andргети - використовуємо them for Light моwhereлей
    light_targets = [c for c in candidate_targets if c.startswith(f"target_light_{t}_{interval}")]
    if light_targets:
        task_type = "regression"
        target_col = light_targets[0]
        logger.info(f"[Stage4Utils] Light models: using {target_col} (regression)")
        return task_type, target_col
    
    # Fallback до withвичайних andргетandв
    direction_targets = [c for c in candidate_targets if c.startswith(f"target_direction_{t}_{interval}")]
    if direction_targets:
        task_type = "classification"
        target_col = direction_targets[0]
        logger.info(f"[Stage4Utils] Fallback: using {target_col} (direction classification)")
        return task_type, target_col
    
    # Осandннandй fallback
    task_type = "regression"
    target_col = f"target_close_{t}_{interval}"
    logger.warning(f"[Stage4Utils] Final fallback: using {target_col} (price regression)")
    return task_type, target_col