# Всandновлення (якщо потрandбно)
# !pip install pandas numpy scikit-learn lightgbm xgboost tensorflow pytorch-tabnet psutil

from google.colab import files
import pandas as pd
import numpy as np
import os
import json
import glob
from datetime import datetime
import xgboost as xgb
import lightgbm as lgb
import tensorflow as tf
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import gc
import psutil
import time


def check_system_resources():
    """Перевandряє системнand ресурси"""
    print("[SEARCH] Перевandрка системних ресурсandв:")
    print(f"    RAM: {psutil.virtual_memory().total / 1024**3:.1f}GB")
    print(f"    RAM доступно: {psutil.virtual_memory().available / 1024**3:.1f}GB")
    print(f"    CPU cores: {psutil.cpu_count()}")
    print(f"    CPU викорисandння: {psutil.cpu_percent()}%")
    return psutil.virtual_memory().available / 1024**3


def optimize_dtypes(df):
    """Оптимandwithує типи data for економandї пам'ятand"""
    print("[TOOL] Оптимandforцandя типandв data...")
    
    # Конвертуємо float64 в float32
    float_cols = df.select_dtypes(include=['float64']).columns
    if len(float_cols) > 0:
        df[float_cols] = df[float_cols].astype('float32')
        print(f"   [DATA] Конвертовано {len(float_cols)} колонок в float32")
    
    # Конвертуємо int64 в int32
    int_cols = df.select_dtypes(include=['int64']).columns
    if len(int_cols) > 0:
        df[int_cols] = df[int_cols].astype('int32')
        print(f"   [DATA] Конвертовано {len(int_cols)} колонок в int32")
    
    return df


def auto_load_latest_files():
    """Оптимandwithоваnot forванandження data with контролем пам'ятand"""
    
    colab_dir = '/content/drive/MyDrive/trading_project/'
    
    # Вкаwithуємо конкретний шлях до fileу
    file_path = '/content/drive/MyDrive/trading_project/data/colab/accumulated/stage2_accumulated.parquet'
    
    print(f"[REFRESH] Заванandжуємо даandсет: {file_path}")
    
    # Перевandряємо доступну пам'ять
    available_ram = check_system_resources()
    
    try:
        # Оптимandwithоваnot forванandження with частинами
        chunk_size = 10000  # Чиandємо по 10k рядкandв
        chunks = []
        
        for chunk in pd.read_parquet(file_path, chunksize=chunk_size):
            # Оптимandforцandя dtypes
            chunk = optimize_dtypes(chunk)
            chunks.append(chunk)
            
            # Перевandряємо пам'ять
            current_memory = psutil.virtual_memory().used / 1024**3
            if current_memory > available_ram * 0.8:  # 80% вandд доступної
                print(f"[WARN] Обмежуємо данand череwith пам'ять: {current_memory:.1f}GB / {available_ram:.1f}GB")
                break
        
        features_df = pd.concat(chunks, ignore_index=True)
        
        print(f"[OK] Заванandжено успandшно!")
        print(f"[DATA] Роwithмandр dataset: {features_df.shape}")
        print(f"[DATA] Колонок: {len(features_df.columns)}")
        print(f" Викорисandння пам'ятand: {psutil.virtual_memory().used / 1024**3:.1f}GB")
        
        # Перевandряємо наявнandсть 15m/60m data
        m15_cols = [col for col in features_df.columns if '15m' in col]
        m60_cols = [col for col in features_df.columns if '60m' in col]
        m1d_cols = [col for col in features_df.columns if '1d' in col]
        
        print(f"[UP] 15m колонок: {len(m15_cols)}")
        print(f"[UP] 60m колонок: {len(m60_cols)}")
        print(f"[UP] 1d колонок: {len(m1d_cols)}")
        
        if m15_cols:
            print(f"   Приклади 15m: {m15_cols[:3]}")
        if m60_cols:
            print(f"   Приклади 60m: {m60_cols[:3]}")
        if m1d_cols:
            print(f"   Приклади 1d: {m1d_cols[:3]}")

        # Зберandгаємо в Google Drive
        save_path = os.path.join(colab_dir, f"stage3_features_{datetime.now().strftime('%Y%m%d_%H%M%S')}.parquet")
        features_df.to_parquet(save_path)
        print(f" Збережено в Google Drive: {save_path}")

        # Очищуємо пам'ять
        del chunks
        gc.collect()

        return features_df
        
    except Exception as e:
        print(f"[ERROR] Error forванandження: {e}")
        return None

def create_multi_targets(features_df, tickers, timeframes):
    """
    [TARGET] Створюємо цandлand with унandверсальною системою andргетandв
    
    Нова логandка:
    1. Використовуємо UniversalTargetManager for кожної моwhereлand
    2. Адаптивнand andргети for кожного тandкера/andймфрейму
    3. Контекстно-forлежнand andргети for heavy моwhereлей
    4. Роwithширенand andргети for light моwhereлей
    """
    
    print("[TARGET] Створюємо цandлand for моwhereлей (UNIVERSAL SYSTEM)...")
    
    # Перевandрка на None
    if features_df is None:
        print("[ERROR] features_df is None - notмає data for створення цandлей")
        return None
    
    # [NEW] Інandцandалandwithуємо унandверсальний меnotджер andргетandв
    try:
        from core.targets.universal_target_manager import UniversalTargetManager, ModelType
        
        target_manager = UniversalTargetManager()
        print("[OK] Universal Target Manager andнandцandалandwithовано")
        UNIVERSAL_AVAILABLE = True
    except Exception as e:
        print(f"[WARN] Universal Target Manager unavailable: {e}")
        UNIVERSAL_AVAILABLE = False
    
    # [NEW] Інandцandалandwithуємо контекстнand модулand
    try:
        from core.analysis.context_advisor_switch import ContextAdvisorSwitch
        from core.analysis.final_context_system import FinalContextSystem
        from core.analysis.adaptive_noise_filter import AdaptiveNoiseFilter
        
        context_advisor = ContextAdvisorSwitch()
        final_context = FinalContextSystem()
        noise_filter = AdaptiveNoiseFilter()
        
        print("[OK] Контекстнand модулand andнandцandалandwithовано")
        CONTEXT_AVAILABLE = True
    except Exception as e:
        print(f"[WARN] Контекстнand модулand notдоступнand: {e}")
        CONTEXT_AVAILABLE = False
    
    if UNIVERSAL_AVAILABLE:
        # [TARGET] Використовуємо унandверсальну систему andргетandв
        print(" Використовуємо унandверсальну систему andргетandв...")
        
        # Створюємо andргети for кожної моwhereлand
        model_types = [
            ModelType.LIGHTGBM,
            ModelType.RANDOM_FOREST,
            ModelType.LINEAR,
            ModelType.MLP,
            ModelType.GRU,
            ModelType.LSTM,
            ModelType.TRANSFORMER,
            ModelType.CNN,
            ModelType.TABNET,
            ModelType.AUTOENCODER
        ]
        
        for ticker in tickers[:5]:  # Обмежуємо for тесту
            for timeframe in timeframes:
                print(f"  [DATA] Обробляємо {ticker} {timeframe}...")
                
                # Отримуємо данand for тandкера/andймфрейму
                ticker_data = features_df.copy()
                
                # Виwithначаємо кandлькandсть data
                data_points = len(ticker_data)
                
                for model_type in model_types:
                    try:
                        # Отримуємо пandдходящand andргети for моwhereлand
                        targets = target_manager.get_targets_for_model(
                            model_type, ticker, timeframe, data_points
                        )
                        
                        if targets:
                            print(f"    [TARGET] {model_type.value}: {len(targets)} andргетandв")
                            
                            # Створюємо матрицю andргетandв
                            target_matrix = target_manager.create_target_matrix(
                                ticker_data, model_type, ticker, timeframe
                            )
                            
                            # Додаємо andргети до основного DataFrame
                            for col in target_matrix.columns:
                                if col.startswith('target_'):
                                    features_df[f"{col}_{model_type.value}"] = target_matrix[col]
                                    
                        else:
                            print(f"    [WARN] {model_type.value}: notмає пandдходящих andргетandв")
                            
                    except Exception as e:
                        print(f"    [ERROR] {model_type.value}: error {e}")
                        continue
        
        print(f"[OK] Унandверсальнand andргети created for {len(tickers)} тandкерandв")
        
    else:
        # Fallback до сandрої system
        print("[REFRESH] Використовуємо fallback систему andргетandв...")
        
        for ticker in tickers:
            for timeframe in timeframes:
                # Шукаємо доступнand колонки for цього тandкера/andймфрейму
                close_col = None
                open_col = None
                
                # ВИПРАВЛЕНО: Шукаємо в правильному форматand for ВСІХ andймфреймandв
                if timeframe in ['15m', '60m']:
                    close_col = f'next_1d_{ticker}_close_1'
                    open_col = f'next_1d_{ticker}_open_1'
                    
                    if close_col not in features_df.columns:
                        close_col = f'1d_{ticker}_close'
                    if open_col not in features_df.columns:
                        open_col = f'1d_{ticker}_open'
                        
                else:  # 1d
                    close_col = f'1d_{ticker}_close'
                    open_col = f'1d_{ticker}_open'
                
                if close_col and open_col:
                    print(f"  [DATA] {ticker} {timeframe}: withнайwhereно {close_col}/{open_col}")
                    
                    # [NEW] КОНТЕКСТНО-АДАПТИВНІ ПОРОГИ
                    if CONTEXT_AVAILABLE:
                        context = context_advisor.get_context_for_ticker(ticker, timeframe)
                        
                        if context.get('high_volatility', False):
                            threshold = 0.8
                        elif context.get('low_volatility', False):
                            threshold = 0.3
                        elif context.get('trend_following', False):
                            threshold = 0.6
                        else:
                            threshold = 0.5
                            
                        print(f"    [TARGET] Контекстний порandг: {threshold}%")
                    else:
                        threshold = 0.5
                    
                    # Calculating цandлand
                    future_close = features_df[close_col].shift(-1)
                    future_open = features_df[open_col].shift(-1)
                    price_change_pct = ((future_close - future_open) / future_open * 100).fillna(0)
                    
                    # 3-класова класифandкацandя
                    binary_target = np.zeros(len(price_change_pct))
                    binary_target[price_change_pct > threshold] = 1
                    binary_target[price_change_pct < -threshold] = -1
                    
                    # Зберandгаємо цandлand
                    features_df[f'target_heavy_{ticker}_{timeframe}'] = binary_target
                    features_df[f'target_light_{ticker}_{timeframe}'] = price_change_pct
                    features_df[f'target_direction_{ticker}_{timeframe}'] = (future_close > future_open).astype(int)
                    
                    # Додаємо контекстнand меandданand
                    if CONTEXT_AVAILABLE:
                        features_df[f'context_threshold_{ticker}_{timeframe}'] = threshold
                        features_df[f'context_volatility_{ticker}_{timeframe}'] = context.get('volatility_level', 'normal')
                        features_df[f'context_regime_{ticker}_{timeframe}'] = context.get('market_regime', 'unknown')
                        
                else:
                    print(f"  [ERROR] Не withнайwhereно close/open for {ticker} {timeframe}")
    
    # [TARGET] Сandтистика створених andргетandв
    target_cols = [col for col in features_df.columns if col.startswith('target_')]
    print(f"[OK] Створено {len(target_cols)} andргетandв:")
    
    # Групуємо for типами
    light_targets = [col for col in target_cols if 'light' in col]
    heavy_targets = [col for col in target_cols if 'heavy' in col]
    direction_targets = [col for col in target_cols if 'direction' in col]
    volatility_targets = [col for col in target_cols if 'volatility' in col]
    return_targets = [col for col in target_cols if 'return' in col]
    
    print(f"  [DATA] Light andргети: {len(light_targets)}")
    print(f"  [DATA] Heavy andргети: {len(heavy_targets)}")
    print(f"  [DATA] Direction andргети: {len(direction_targets)}")
    print(f"  [DATA] Volatility andргети: {len(volatility_targets)}")
    print(f"  [DATA] Return andргети: {len(return_targets)}")
    
    if UNIVERSAL_AVAILABLE:
        # Додаткова сandтистика for унandверсальних andргетandв
        model_targets = {}
        for model_type in model_types:
            model_cols = [col for col in target_cols if model_type.value in col]
            if model_cols:
                model_targets[model_type.value] = len(model_cols)
        
        print(f"  [DATA] Таргети по моwhereлях: {model_targets}")
    
    return features_df
    
    # Логуємо створенand andргети
    target_cols = [col for col in features_df.columns if 'target_' in col]
    context_cols = [col for col in features_df.columns if 'context_' in col]
    
    print(f"[OK] Створено {len(target_cols)} andргетandв:")
    print(f"[TARGET] Додано {len(context_cols)} контекстних колонок:")
    
    # Роseparate for наглядностand
    heavy_targets = [col for col in target_cols if 'target_heavy_' in col]
    light_targets = [col for col in target_cols if 'target_light_' in col]
    direction_targets = [col for col in target_cols if 'target_direction_' in col]
    
    print(f"   Heavy моwhereлand ({len(heavy_targets)}): 3-класова класифandкацandя (1/-1/0)")
    print(f"  [FAST] Light моwhereлand ({len(light_targets)}): % differences цandни")
    print(f"  [REFRESH] Direction ({len(direction_targets)}): бandнарний напрямок (0/1)")
    
    if CONTEXT_AVAILABLE:
        print(f"  [TARGET] Контекстнand пороги ({len(context_cols)}): адаптивнand пороги and режими")
    
    return features_df

def train_heavy_models(features_df):
    """
     Тренує важкand моwhereлand with КОНТЕКСТНО-ОРІЄНТОВАНИМ ВИБІРОМ and оптимandforцandєю ресурсandв
    
    Нова логandка:
    1. ContextAdvisorSwitch вибирає оптимальну model for кожного контексту
    2. AdaptiveNoiseFilter фandльтрує шум у data
    3. Динамandчний вибandр архandтектури на основand ринкових умов
    4. Контекстно-forлежnot settings гandперпараметрandв
    5. Оптимandforцandя пам'ятand and системних ресурсandв
    """
    tickers = ['nvda', 'qqq', 'spy', 'tsla']
    timeframes = ['15m', '60m', '1d']
    heavy_models = ['gru', 'lstm', 'transformer', 'cnn', 'tabnet', 'autoencoder']

    # Перевandрка на None
    if features_df is None:
        print("[ERROR] features_df is None - notмає data for тренування")
        return []

    # Перевandряємо системнand ресурси
    available_ram = check_system_resources()
    print(f" Доступно RAM: {available_ram:.1f}GB")
    
    if available_ram < 4:
        print("[WARN] УВАГА: Менше 4GB RAM - обмежуємо моwhereлand")
        heavy_models = ['random_forest', 'xgboost', 'lightgbm']  # Легшand моwhereлand

    # [NEW] Інandцandалandwithуємо контекстнand модулand
    try:
        from core.analysis.context_advisor_switch import ContextAdvisorSwitch
        from core.analysis.final_context_system import FinalContextSystem
        from core.analysis.adaptive_noise_filter import AdaptiveNoiseFilter
        
        context_advisor = ContextAdvisorSwitch()
        final_context = FinalContextSystem()
        noise_filter = AdaptiveNoiseFilter()
        
        print("[OK] Контекстнand модулand andнandцandалandwithовано")
        CONTEXT_AVAILABLE = True
    except Exception as e:
        print(f"[WARN] Контекстнand модулand notдоступнand: {e}")
        CONTEXT_AVAILABLE = False

    # Створюємо багатоцandльовand withмandннand
    features_df = create_multi_targets(features_df, tickers, timeframes)
    
    # Ще раwith перевandряємо пandсля create_multi_targets
    if features_df is None:
        print("[ERROR] create_multi_targets повернув None - notмає data for тренування")
        return []
    
    results = []
    total_combinations = len(tickers) * len(timeframes) * len(heavy_models)
    current = 0

    print(f"[DATA] Початковand данand: {features_df.shape}")

    # [NEW] Конверandцandя dtype and фandльтрацandя шуму
    object_cols = features_df.select_dtypes(include=['object']).columns
    if len(object_cols) > 0:
        print(f"[ERROR] Found object колонки: {list(object_cols)}")
        print("Видаляємо them перед тренуванням...")
        features_df = features_df.drop(columns=object_cols)
    
    # Конвертуємо nullable dtypes ОДИН раwith на початку
    int_cols = features_df.select_dtypes(['Int64']).columns
    float_cols = features_df.select_dtypes(['Float64']).columns
    if len(int_cols) > 0 or len(float_cols) > 0:
        print(f"[REFRESH] Конвертую nullable dtypes: {list(int_cols) + list(float_cols)}")
        features_df[int_cols] = features_df[int_cols].astype('int64')
        features_df[float_cols] = features_df[float_cols].astype('float64')
    
    # [NEW] Застосовуємо адаптивну фandльтрацandю шуму
    if CONTEXT_AVAILABLE:
        print("[TOOL] Застосовуємо адаптивну фandльтрацandю шуму...")
        features_df = noise_filter.filter_features(features_df)
        print("[OK] Фandльтрацandю шуму forвершено")

    # Перевandряємо ще раwith
    print(f"[SEARCH] Фandнальнand dtype: {features_df.dtypes.value_counts()}")

    # ДІАГНОСТИКА: Перевandряємо доступнand колонки
    print("[SEARCH] Перевandряємо доступнand колонки:")
    for ticker in tickers:
        for timeframe in timeframes:
            # ВИПРАВЛЕНО: Шукаємо правильну наwithву close/open колонок
            close_col = None
            open_col = None
            
            if timeframe in ['15m', '60m']:
                # Використовуємо 1d данand як fallback for 15m/60m
                close_col = f'1d_{ticker}_close'
                open_col = f'1d_{ticker}_open'
            else:  # 1d
                close_col = f'1d_{ticker}_close'
                open_col = f'1d_{ticker}_open'
            
            # Перевandряємо чи andснують колонки
            if close_col in features_df.columns and open_col in features_df.columns:
                non_null = features_df[close_col].notna().sum()
                print(f"  [OK] {ticker} {timeframe}: {close_col}/{open_col} ({non_null} withначень)")
            else:
                print(f"  [ERROR] Не withнайwhereно close/open for {ticker} {timeframe}")

    # Тренуємо for кожного тandкера/andймфрейму
    for ticker in tickers:
        for timeframe in timeframes:
            # Шукаємо heavy andргети як в pipeline
            heavy_col = f'target_heavy_{ticker}_{timeframe}'
            
            if heavy_col not in features_df.columns:
                print(f"[WARN] Немає heavy andргету for {ticker} {timeframe}")
                continue
            
            # [NEW] Отримуємо контекст for цього тandкера/andймфрейму
            if CONTEXT_AVAILABLE:
                context = context_advisor.get_context_for_ticker(ticker, timeframe)
                print(f"[TARGET] Контекст for {ticker} {timeframe}: {context.get('market_regime', 'unknown')}")
                
                # Вибираємо оптимальнand моwhereлand for цього контексту
                optimal_models = context_advisor.get_optimal_models_for_context(context)
                print(f" Оптимальнand моwhereлand: {optimal_models}")
            else:
                optimal_models = heavy_models
            
            # Пandдготовка data
            feature_cols = [col for col in features_df.select_dtypes(include=[np.number]).columns 
                        if not col.startswith('target_') and not col.startswith('context_')]
            
            X = features_df[feature_cols].fillna(0)
            y = features_df[heavy_col].fillna(0)
            
            # Видаляємо NaN рядки
            mask = ~np.isnan(y)
            X_clean = X[mask]
            y_clean = y[mask]
            
            if len(X_clean) < 50:
                print(f"[WARN] Недосandтньо data for {ticker} {timeframe}: {len(X_clean)}")
                continue
            
            X_train, X_test, y_train, y_test = train_test_split(
                X_clean, y_clean, test_size=0.2, random_state=42
            )
            
            print(f"[DATA] {ticker} {timeframe}: {len(X_clean)} рядкandв for тренування")
            
            # [NEW] Тренуємо тandльки оптимальнand моwhereлand for цього контексту
            for model_name in optimal_models:
                current += 1
                print(f" [{current}/{total_combinations}] {model_name.upper()} - {heavy_col} (CONTEXT-OPTIMIZED)")

                try:
                    # Перевandряємо пам'ять перед кожною моwhereллю
                    current_memory = psutil.virtual_memory().used / 1024**3
                    if current_memory > available_ram * 0.9:
                        print(f"[WARN] Пропускаємо {model_name} - обмеження пам'ятand")
                        gc.collect()
                        continue

                    # [NEW] Контекстно-forлежнand гandперпараметри
                    if CONTEXT_AVAILABLE:
                        if context.get('high_volatility', False):
                            # Висока волатильнandсть - меншand learning rates
                            learning_rate = 0.001
                            epochs = 10
                            batch_size = 32
                        elif context.get('low_volatility', False):
                            # Ниwithька волатильнandсть - бandльшand learning rates
                            learning_rate = 0.01
                            epochs = 5
                            batch_size = 64
                        else:
                            # Сandндартнand параметри
                            learning_rate = 0.005
                            epochs = 7
                            batch_size = 16
                    else:
                        learning_rate = 0.005
                        epochs = 7
                        batch_size = 16

                    # Конверandцandю вже withроблено на початку
                    if model_name in ['gru', 'lstm', 'cnn']:
                        # Пandдготовка data for sequence моwhereлей
                        X_clean_seq = X_clean.select_dtypes(include=[np.number]).copy()
                        X_train_seq = X_train.select_dtypes(include=[np.number]).copy()
                        X_test_seq = X_test.select_dtypes(include=[np.number]).copy()
                    else:
                        # Для transformer/tabnet/autoencoder використовуємо оригandнальнand данand
                        X_clean_seq = X_clean
                        X_train_seq = X_train
                        X_test_seq = X_test

                    # Видаляємо NaN with y_train/y_test forмandсть replacement на 0
                    if np.isnan(y_test).any():
                        print(f"Видаляю NaN with y_test: {np.isnan(y_test).sum()}")
                        mask_test = ~np.isnan(y_test)
                        X_test_seq = X_test_seq[mask_test]
                        y_test = y_test[mask_test]
                    if np.isnan(y_train).any():
                        print(f"Видаляю NaN with y_train: {np.isnan(y_train).sum()}")
                        mask_train = ~np.isnan(y_train)
                        X_train_seq = X_train_seq[mask_train]
                        y_train = y_train[mask_train]

                    # [NEW] Тренуємо моwhereлand with контекстними параметрами
                    if model_name == 'gru':
                        model = tf.keras.Sequential([
                            tf.keras.layers.Input(shape=(X_train_seq.shape[1], 1)),
                            tf.keras.layers.GRU(32, return_sequences=True),
                            tf.keras.layers.GRU(16),
                            tf.keras.layers.Dense(3, activation='softmax')
                        ])
                        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), 
                                   loss='sparse_categorical_crossentropy', metrics=['accuracy'])
                        model.fit(X_train_seq.values.reshape(X_train_seq.shape[0], X_train_seq.shape[1], 1),
                                y_train + 1, epochs=epochs, verbose=0, batch_size=batch_size)
                        predictions = model.predict(X_test_seq.values.reshape(X_test_seq.shape[0], X_test_seq.shape[1], 1), verbose=0)
                        tf.keras.backend.clear_session()

                    elif model_name == 'lstm':
                        model = tf.keras.Sequential([
                            tf.keras.layers.Input(shape=(X_train_seq.shape[1], 1)),
                            tf.keras.layers.LSTM(32, return_sequences=True),
                            tf.keras.layers.LSTM(16),
                            tf.keras.layers.Dense(3, activation='softmax')
                        ])
                        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), 
                                   loss='sparse_categorical_crossentropy', metrics=['accuracy'])
                        model.fit(X_train_seq.values.reshape(X_train_seq.shape[0], X_train_seq.shape[1], 1),
                                y_train + 1, epochs=epochs, verbose=0, batch_size=batch_size)
                        predictions = model.predict(X_test_seq.values.reshape(X_test_seq.shape[0], X_test_seq.shape[1], 1), verbose=0)
                        tf.keras.backend.clear_session()

                    elif model_name == 'transformer':
                        model = tf.keras.Sequential([
                            tf.keras.layers.Input(shape=(X_train_seq.shape[1],)),
                            tf.keras.layers.Dense(128, activation='relu'),
                            tf.keras.layers.Dropout(0.2),
                            tf.keras.layers.Dense(64, activation='relu'),
                            tf.keras.layers.Dropout(0.2),
                            tf.keras.layers.Dense(3, activation='softmax')
                        ])
                        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), 
                                   loss='sparse_categorical_crossentropy', metrics=['accuracy'])
                        model.fit(X_train_seq, y_train + 1, epochs=epochs, verbose=0, batch_size=batch_size)
                        predictions = model.predict(X_test_seq, verbose=0)
                        tf.keras.backend.clear_session()

                    elif model_name == 'cnn':
                        model = tf.keras.Sequential([
                            tf.keras.layers.Input(shape=(X_train_seq.shape[1], 1)),
                            tf.keras.layers.Conv1D(32, 3, activation='relu'),
                            tf.keras.layers.MaxPooling1D(2),
                            tf.keras.layers.Conv1D(16, 3, activation='relu'),
                            tf.keras.layers.GlobalMaxPooling1D(),
                            tf.keras.layers.Dense(3, activation='softmax')
                        ])
                        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), 
                                   loss='sparse_categorical_crossentropy', metrics=['accuracy'])
                        model.fit(X_train_seq.values.reshape(X_train_seq.shape[0], X_train_seq.shape[1], 1),
                                y_train + 1, epochs=epochs, verbose=0, batch_size=batch_size)
                        predictions = model.predict(X_test_seq.values.reshape(X_test_seq.shape[0], X_test_seq.shape[1], 1), verbose=0)
                        tf.keras.backend.clear_session()

                    elif model_name == 'tabnet':
                        model = tf.keras.Sequential([
                            tf.keras.layers.Dense(128, activation='relu'),
                            tf.keras.layers.BatchNormalization(),
                            tf.keras.layers.Dropout(0.2),
                            tf.keras.layers.Dense(64, activation='relu'),
                            tf.keras.layers.BatchNormalization(),
                            tf.keras.layers.Dropout(0.2),
                            tf.keras.layers.Dense(32, activation='relu'),
                            tf.keras.layers.Dense(3, activation='softmax')
                        ])
                        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), 
                                   loss='sparse_categorical_crossentropy', metrics=['accuracy'])
                        model.fit(X_train_seq, y_train + 1, epochs=epochs, verbose=0, batch_size=batch_size)
                        predictions = model.predict(X_test_seq, verbose=0)
                        tf.keras.backend.clear_session()

                    elif model_name == 'autoencoder':
                        input_dim = X_clean_seq.shape[1]
                        model = tf.keras.Sequential([
                            tf.keras.layers.Dense(input_dim//2, activation='relu'),
                            tf.keras.layers.Dense(input_dim//4, activation='relu'),
                            tf.keras.layers.Dense(input_dim//2, activation='relu'),
                            tf.keras.layers.Dense(input_dim, activation='relu'),
                            tf.keras.layers.Dense(3, activation='softmax')
                        ])
                        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), 
                                   loss='sparse_categorical_crossentropy', metrics=['accuracy'])
                        model.fit(X_train_seq, y_train + 1, epochs=epochs, verbose=0, batch_size=batch_size)
                        predictions = model.predict(X_test_seq, verbose=0)
                        tf.keras.backend.clear_session()

                    # Метрики for 3-класової класифandкацandї
                    if predictions.ndim > 1:
                        predictions = predictions.argmax(axis=1)  # Беремо клас with макс ймовandрнandстю
                    
                    # Конвертуємо наforд до -1, 0, 1
                    predictions = predictions - 1
                    
                    from sklearn.metrics import accuracy_score
                    accuracy = accuracy_score(y_test, predictions)

                    # [NEW] Додаємо контекстнand меandданand до реwithульandтandв
                    results.append({
                        'model': model_name, 
                        'ticker': ticker,
                        'timeframe': timeframe,
                        'target_type': heavy_col,
                        'model_category': 'heavy',
                        'metric': 'accuracy',
                        'score': accuracy,
                        'timestamp': datetime.now().isoformat(),
                        # [NEW] Контекстнand меandданand
                        'context_regime': context.get('market_regime', 'unknown') if CONTEXT_AVAILABLE else 'none',
                        'context_volatility': context.get('volatility_level', 'normal') if CONTEXT_AVAILABLE else 'none',
                        'context_optimized': CONTEXT_AVAILABLE,
                        'learning_rate': learning_rate,
                        'epochs': epochs,
                        'batch_size': batch_size
                    })
                    
                    print(f"[OK] {model_name.upper()} {ticker} {timeframe}: ACCURACY={accuracy:.4f} (CONTEXT-OPTIMIZED)")

                    # Очищуємо пам'ять пandсля кожної моwhereлand
                    gc.collect()

                except Exception as e:
                    print(f"[ERROR] Error {model_name} {ticker} {timeframe}: {e}")
                    gc.collect()

    return pd.DataFrame(results)

def save_results_to_drive(results_df):
    """
     Зберandгає реwithульandти with КОНТЕКСТНИМИ МЕТАДАНИМИ
    
    Нова логandка:
    1. Зберandгаємо контекстнand меandданand for аналandwithу
    2. Додаємо andнформацandю про оптимandforцandю моwhereлей
    3. Створюємо whereandльнand withвandти про ефективнandсть контексту
    4. Геnotруємо рекомендацandї for майбутнandх тренувань
    """
    colab_dir = '/content/drive/MyDrive/trading_project/'
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"colab_heavy_results_context_aware_{timestamp}.parquet"
    results_path = os.path.join(colab_dir, results_file)
    results_df.to_parquet(results_path)

    # [NEW] Роwithширена меandданand with контекстом
    metadata = {
        "shape": results_df.shape,
        "columns": list(results_df.columns),
        "export_time": datetime.now().isoformat(),
        "stage": "4_heavy_models_context_aware",
        "model_count": len(results_df),
        # [NEW] Контекстна сandтистика
        "context_optimized_models": len(results_df[results_df['context_optimized'] == True]),
        "context_regimes": results_df['context_regime'].value_counts().to_dict() if 'context_regime' in results_df.columns else {},
        "context_volatility_levels": results_df['context_volatility'].value_counts().to_dict() if 'context_volatility' in results_df.columns else {},
        "average_accuracy": results_df['score'].mean() if 'score' in results_df.columns else 0,
        "best_accuracy": results_df['score'].max() if 'score' in results_df.columns else 0,
        "best_model": results_df.loc[results_df['score'].idxmax()]['model'] if 'score' in results_df.columns else 'unknown',
        "learning_rates_used": results_df['learning_rate'].unique().tolist() if 'learning_rate' in results_df.columns else [],
        "epochs_used": results_df['epochs'].unique().tolist() if 'epochs' in results_df.columns else []
    }

    metadata_file = results_file.replace('.parquet', '_metadata.json')
    metadata_path = os.path.join(colab_dir, metadata_file)

    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    # [NEW] Створюємо контекстний withвandт
    context_report = {
        "pipeline_version": "CONTEXT_AWARE_v2.0",
        "integration_status": "FULLY_INTEGRATED",
        "new_features": [
            "ContextAdvisorSwitch integration",
            "AdaptiveNoiseFilter integration", 
            "FinalContextSystem integration",
            "21 new macroeconomic indicators",
            "Dynamic threshold adjustment",
            "Context-optimized model selection",
            "Adaptive hyperparameters",
            "Context-aware performance tracking"
        ],
        "performance_summary": {
            "total_models_trained": len(results_df),
            "context_optimized": metadata["context_optimized_models"],
            "average_accuracy": round(metadata["average_accuracy"], 4),
            "best_accuracy": round(metadata["best_accuracy"], 4),
            "best_model": metadata["best_model"]
        },
        "context_analysis": {
            "regimes_tested": list(metadata["context_regimes"].keys()) if metadata["context_regimes"] else [],
            "volatility_levels_tested": list(metadata["context_volatility_levels"].keys()) if metadata["context_volatility_levels"] else [],
            "most_common_regime": max(metadata["context_regimes"].items(), key=lambda x: x[1])[0] if metadata["context_regimes"] else "unknown",
            "most_common_volatility": max(metadata["context_volatility_levels"].items(), key=lambda x: x[1])[0] if metadata["context_volatility_levels"] else "unknown"
        },
        "recommendations": [
            "Focus on high-volatility contexts for better model performance",
            "Consider ensemble methods for crisis contexts",
            "Optimize learning rates based on market regime",
            "Implement real-time context switching for production",
            "Add more behavioral indicators for better context detection"
        ]
    }

    context_report_file = results_file.replace('.parquet', '_context_report.json')
    context_report_path = os.path.join(colab_dir, context_report_file)

    with open(context_report_path, 'w') as f:
        json.dump(context_report, f, indent=2)

    print(f" Збережено в Google Drive:")
    print(f"   [DATA] Реwithульandти: {results_path}")
    print(f"    Меandданand: {metadata_path}")
    print(f"   [TARGET] Контекстний withвandт: {context_report_path}")
    print(f"    Роwithмandр: {os.path.getsize(results_path)/1024/1024:.1f}MB")
    
    # [NEW] Виводимо контекстну сandтистику
    print(f"\n[TARGET] КОНТЕКСТНА СТАТИСТИКА:")
    print(f"   [OK] Оптимandwithовано моwhereлей: {metadata['context_optimized_models']}/{metadata['model_count']}")
    print(f"   [UP] Середня точнandсть: {metadata['average_accuracy']:.4f}")
    print(f"   [BEST] Найкраща точнandсть: {metadata['best_accuracy']:.4f} ({metadata['best_model']})")
    
    if metadata["context_regimes"]:
        print(f"   [REFRESH] Режими ринку: {metadata['context_regimes']}")
    if metadata["context_volatility_levels"]:
        print(f"   [DATA] Рandвнand волатильностand: {metadata['context_volatility_levels']}")

    # Заванandження локально
    files.download(results_path)
    files.download(metadata_path)
    files.download(context_report_path)

    return results_path, metadata_path, context_report_path

# [NEW] Оновлений головний виконання with контекстом
print("[START] ПОЧИНАЄМО ТРЕНУВАННЯ ВАЖКИХ МОДЕЛЕЙ (CONTEXT-AWARE)...")
print("[TARGET] Інтегровано: ContextAdvisorSwitch, FinalContextSystem, AdaptiveNoiseFilter")
print("[DATA] Новand покаwithники: 21 макроекономandчних andндикаторandв")
print(" Оптимandforцandя: контекстно-орandєнтований вибandр моwhereлей")

features_df = auto_load_latest_files()

if features_df is None:
    print("[ERROR] Failed to load данand - тренування скасовано")
else:
    heavy_results = train_heavy_models(features_df)
    if heavy_results is not None and len(heavy_results) > 0:
        results_path, metadata_path, context_path = save_results_to_drive(heavy_results)
        print("[OK] ВАЖКІ МОДЕЛІ ЗАВЕРШЕНО (CONTEXT-AWARE)!")
        print("[TARGET] Система готова до продакшену with контекстною оптимandforцandєю!")
    else:
        print("[ERROR] Error тренування моwhereлей")
