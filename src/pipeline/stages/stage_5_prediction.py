"""
Stage 5: Prediction Generation with Stacked Ensembles and Contextual Adjustments

Uses champion models and stacked ensembles to generate forecasts, 
incorporating real-time market regime adjustments and historical performance.
"""

import os
import json
from typing import Optional, Any, Dict, List
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

from src.pipeline.stages.base_stage import BaseStage
from src.config.unified_config_manager import UnifiedConfigManager
from src.core.logging.logger import ProjectLogger
from src.meta_learning.memory.diary_engine import DiaryEngine
from src.ensembling.stacked_ensemble import StackedEnsemble
from src.analytics.context.prediction_adjuster import PredictionAdjuster
from src.models.model_selector.smart_selector import SmartModelSelector
from src.analytics.analyzers.knn_similarity_finder import KnnSimilarityFinder
from src.features.utils.datetime_utils import ensure_datetime_column, normalize_metadata_columns
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

class PredictionStage(BaseStage):
    """
    Stage responsible for generating model predictions using an ensemble approach,
    calculating confidence scores, and adjusting forecasts based on market context.
    """
    def __init__(self, config_manager: UnifiedConfigManager, error_handler, **kwargs):
        super().__init__(config_manager, error_handler, **kwargs)
        self.logger = ProjectLogger.get_logger("PredictionStage")
        self.prediction_config = self.config_manager.get_config('prediction', {})
        
        # Use centralized path getter method
        self.models_path = self.config_manager.get_models_path()
        
        self.diary = DiaryEngine()
        self.adjuster = PredictionAdjuster(config=self.config_manager.get('analysis.prediction_adjustment', {}))
        self.ensemble_factory = StackedEnsemble()
        self.context_selector = SmartModelSelector()
        self.knn_similarity = KnnSimilarityFinder(config={'n_neighbors': 5})

    async def run(self, **kwargs) -> Dict[str, Any]:
        """
        Generates adjusted predictions for tickers processed in earlier stages.

        Args:
            **kwargs: Dictionary containing 'features_data' and 'models_metadata'.

        Returns:
            Dict[str, Any]: Updated pipeline data with 'prediction_results'.
        """
        features_df = kwargs.get('features_data')
        # ✅ FIX: Читаємо models_metadata з kwargs (не з brain)
        models_meta = kwargs.get('models_metadata', {})
        market_regime = kwargs.get('market_regime', 'neutral')
        
        # ✅ NEW: Якщо models_metadata не передана, спробуємо завантажити з диска
        if not models_meta:
            self.logger.warning("⚠️ models_metadata не знайдена в kwargs. Спроба завантажити з диска...")
            models_meta = self._load_models_metadata_from_disk(kwargs)
            if models_meta:
                self.logger.info(f"✅ Завантажено {len(models_meta)} моделей з диска")
            else:
                self.logger.warning("⚠️ Не вдалося завантажити models_metadata з диска")

        self.logger.info(f"📊 DEBUG Stage 5: features_df type: {type(features_df)}")
        self.logger.info(f"📊 DEBUG Stage 5: features_df is None: {features_df is None}")
        if features_df is not None:
            self.logger.info(f"📊 DEBUG Stage 5: features_df shape: {features_df.shape}")
            self.logger.info(f"📊 DEBUG Stage 5: features_df empty: {features_df.empty}")
        self.logger.info(f"📊 DEBUG Stage 5: models_meta type: {type(models_meta)}")
        self.logger.info(f"📊 DEBUG Stage 5: models_meta count: {len(models_meta)}")
        self.logger.info(f"📊 DEBUG Stage 5: models_meta keys: {list(models_meta.keys())[:5] if models_meta else 'empty'}")

        if features_df is None or features_df.empty or not models_meta:
            self.logger.warning("Required features or model metadata not found. Skipping Stage 5.")
            self.logger.warning(f"  - features_df is None: {features_df is None}")
            self.logger.warning(f"  - features_df empty: {features_df.empty if features_df is not None else 'N/A'}")
            self.logger.warning(f"  - models_meta empty: {not models_meta}")
            return {}

        # ✅ CRITICAL FIX: Normalize datetime columns at stage entry
        if isinstance(features_df, pd.DataFrame):
            features_df = normalize_metadata_columns(features_df)
            self.logger.info(f"✅ Normalized features_df at stage entry")

        # ✅ FIX: Перевіряємо, чи моделі доступні локально
        # Моделі можуть бути:
        # 1. Локальні: model_path містить локальний шлях (data\colab\accumulated\...)
        # 2. З Colab: model_path містить /content/drive/... або порожній
        has_local_models = False
        for context_id, meta in models_meta.items():
            model_path = meta.get('model_path', '')
            # Якщо model_path порожній або містить /content/drive, це Colab модель
            # Якщо model_path містить локальний шлях (data\ або data/), це локальна модель
            if model_path and '/content/drive' not in model_path and ('data\\' in model_path or 'data/' in model_path):
                # Це локальна модель
                has_local_models = True
                self.logger.debug(f"✅ Знайдена локальна модель: {context_id} -> {model_path}")
                break
        
        if not has_local_models:
            self.logger.warning("⚠️ Всі моделі з Colab (не доступні локально).")
            self.logger.warning(f"   Перевірені моделі:")
            for context_id, meta in list(models_meta.items())[:5]:
                model_path = meta.get('model_path', '')
                model_type = meta.get('model_type', '')
                self.logger.warning(f"   - {context_id}: model_path='{model_path}', model_type='{model_type}'")
            
            # ✅ КРИТИЧНО: Замість пропускання, спробуємо завантажити моделі з локальної папки
            # Якщо model_path не встановлено, спробуємо знайти моделі в batch_dir/models/
            self.logger.info("🔍 Спроба завантажити моделі з локальної папки...")
            
            # Витягуємо batch_dir з першої моделі
            batch_dir = None
            for context_id, meta in models_meta.items():
                model_path = meta.get('model_path', '')
                if model_path:
                    # Витягуємо batch_dir з model_path
                    parts = model_path.replace('/', '\\').split('\\')
                    if 'models' in parts:
                        models_idx = parts.index('models')
                        if models_idx > 0:
                            batch_name = parts[models_idx - 1]
                            base_dir = Path(self.config_manager.get('system.accumulation.output_dir', 'data/colab/accumulated'))
                            batch_dir = base_dir / batch_name
                            self.logger.info(f"✅ Витягнено batch_dir: {batch_dir}")
                            break
            
            if batch_dir and batch_dir.exists():
                # Оновлюємо model_path для всіх моделей
                models_dir = batch_dir / 'models'
                if models_dir.exists():
                    self.logger.info(f"✅ Знайдена папка моделей: {models_dir}")
                    for context_id, meta in models_meta.items():
                        ticker = meta.get('ticker', '')
                        target = meta.get('target', '')
                        model_type = meta.get('model_type', '')
                        
                        # Конструюємо очікуване ім'я файлу моделі
                        model_filename = f"{model_type}_{ticker}_{target}.pt"
                        model_path = models_dir / model_filename
                        
                        if model_path.exists():
                            meta['model_path'] = str(model_path)
                            self.logger.info(f"✅ Оновлено model_path для {context_id}: {model_path}")
                            has_local_models = True
                        else:
                            self.logger.warning(f"⚠️ Модель не знайдена: {model_path}")
                else:
                    self.logger.warning(f"⚠️ Папка моделей не знайдена: {models_dir}")
            else:
                self.logger.warning(f"⚠️ Не вдалося витягти batch_dir з model_path")
            
            if not has_local_models:
                self.logger.error("❌ Не вдалося знайти жодної локальної моделі. Пропускаємо Stage 5.")
                return {}

        prediction_results = {}
        self.logger.info(f"Generating ensemble predictions for {len(models_meta)} contexts...")

        for context_id, meta in models_meta.items():
            try:
                ticker = meta.get('ticker')
                target_col = meta.get('target', '')
                model_type = meta.get('model_type', '')
                
                self.logger.info(f"🔍 Processing context: {context_id}")
                self.logger.info(f"   ticker={ticker}, target={target_col}, model_type={model_type}")
                
                # Filter features for this specific ticker
                ticker_df = features_df[features_df['ticker'] == ticker].tail(50) # Use recent window
                if ticker_df.empty:
                    self.logger.warning(f"⚠️ No data for ticker {ticker}")
                    continue

                # ✅ FIX: Очищуємо дані перед передачею в моделі
                # Видаляємо non-numeric колонки та конвертуємо в float
                ticker_df_clean = ticker_df.copy()
                
                # Видаляємо metadata колонки
                metadata_cols = ['ticker', 'datetime', 'date', 'interval', 'timeframe', 'hash', 'symbol']
                ticker_df_clean = ticker_df_clean.drop(columns=[c for c in metadata_cols if c in ticker_df_clean.columns], errors='ignore')
                
                # Конвертуємо всі колонки в float
                for col in ticker_df_clean.columns:
                    try:
                        ticker_df_clean[col] = pd.to_numeric(ticker_df_clean[col], errors='coerce')
                    except:
                        ticker_df_clean = ticker_df_clean.drop(columns=[col], errors='ignore')
                
                # Заповнюємо NaN нулями
                ticker_df_clean = ticker_df_clean.fillna(0)
                
                # Замінюємо inf на 0
                ticker_df_clean = ticker_df_clean.replace([np.inf, -np.inf], 0)
                
                # Перевіряємо що всі дані числові
                if ticker_df_clean.empty or ticker_df_clean.dtypes.apply(lambda x: x.kind not in 'biufc').any():
                    self.logger.warning(f"⚠️ Дані для {ticker} містять non-numeric колонки, пропускаємо")
                    continue
                
                # ✅ NEW: Завантажуємо вибрані фічи для цієї моделі
                selected_features = meta.get('selected_features', [])
                self.logger.info(f"🔍 DEBUG Stage 5: context_id={context_id}")
                self.logger.info(f"🔍 DEBUG Stage 5: meta keys={list(meta.keys())}")
                self.logger.info(f"🔍 DEBUG Stage 5: selected_features з metadata: {len(selected_features)} фіч")
                if selected_features:
                    self.logger.info(f"🔍 DEBUG Stage 5: перші 5 фіч: {selected_features[:5]}")
                self.logger.info(f"🔍 DEBUG Stage 5: ticker_df_clean shape ДО фільтрування: {ticker_df_clean.shape}")
                self.logger.info(f"🔍 DEBUG Stage 5: ticker_df_clean columns (перші 5): {list(ticker_df_clean.columns)[:5]}")
                
                # ✅ Фільтруємо дані до вибраних фіч
                filtered_features_list = []  # Track which features we're using
                if selected_features:
                    available_features = [f for f in selected_features if f in ticker_df_clean.columns]
                    self.logger.info(f"🔍 DEBUG Stage 5: available_features={len(available_features)} (з {len(selected_features)} вибраних)")
                    
                    if available_features:
                        ticker_df_clean = ticker_df_clean[available_features]
                        filtered_features_list = available_features
                        self.logger.info(f"✅ Використовуємо {len(available_features)} фіч для {model_type}")
                        self.logger.info(f"🔍 DEBUG Stage 5: ticker_df_clean shape ПІСЛЯ фільтрування: {ticker_df_clean.shape}")
                    else:
                        self.logger.warning(f"⚠️ Жодна з вибраних фіч не знайдена для {model_type}")
                        self.logger.warning(f"   Вибрані фічи: {selected_features[:5]}...")
                        self.logger.warning(f"   Доступні колонки: {list(ticker_df_clean.columns)[:5]}...")
                        continue
                else:
                    self.logger.warning(f"⚠️ Не знайдено вибраних фіч для {model_type}, використовуємо всі {ticker_df_clean.shape[1]} колонок")
                    filtered_features_list = ticker_df_clean.columns.tolist()

                # 1. Load All Available Models for this Context (Ensemble)
                # ✅ FIX: Передаємо models_meta для витягування batch_dir
                models = self._load_available_models(context_id, models_meta)
                if not models:
                    self.logger.warning(f"⚠️ Не знайдено моделей для {context_id}, пропускаємо")
                    continue
                
                # ✅ Завантажуємо scaler для денормалізації (якщо є)
                # ВАЖЛИВО: Scaler для TARGET (1 колонка) зберігається окремо в batch_dir
                # НЕ беремо scaler з моделі, бо там scaler для FEATURES (37 колонок)!
                target_scaler = None
                
                # Спробуємо завантажити scaler з batch_dir
                if target_scaler is None:
                    # ✅ FIX: Scaler для TARGET зберігається окремо в batch_dir
                    # Витягуємо batch_dir з model_path
                    if context_id in models_meta:
                        model_path_str = models_meta[context_id].get('model_path', '')
                        if model_path_str:
                            model_path_str = model_path_str.replace('/', '\\')
                            parts = model_path_str.split('\\')
                            if 'models' in parts:
                                models_idx = parts.index('models')
                                if models_idx > 0:
                                    batch_name = parts[models_idx - 1]
                                    base_dir = Path(self.config_manager.get('system.accumulation.output_dir', 'data/colab/accumulated'))
                                    batch_dir = base_dir / batch_name
                                    
                                    # ✅ Спробуємо завантажити scaler для TARGET
                                    # Формат: scaler_AMD_target_return_1d.pkl
                                    scaler_path = batch_dir / f"scaler_{ticker}_{target_col}.pkl"
                                    if scaler_path.exists():
                                        import joblib
                                        target_scaler = joblib.load(scaler_path)
                                        
                                        # ✅ DEBUG: Перевіряємо, чи це правильний scaler
                                        if hasattr(target_scaler, 'scale_'):
                                            if target_scaler.scale_.shape[0] == 1:
                                                self.logger.info(f"✅ Завантажено ПРАВИЛЬНИЙ target scaler з {scaler_path} (shape: {target_scaler.scale_.shape})")
                                            else:
                                                self.logger.error(f"❌ НЕПРАВИЛЬНИЙ scaler! Має {target_scaler.scale_.shape[0]} features замість 1")
                                                target_scaler = None
                                    else:
                                        # ✅ Альтернатива: шукаємо будь-який scaler у batch_dir
                                        scaler_files = list(batch_dir.glob("scaler_*.pkl"))
                                        if scaler_files:
                                            # Беремо перший знайдений scaler
                                            scaler_path = scaler_files[0]
                                            import joblib
                                            target_scaler = joblib.load(scaler_path)
                                            
                                            # ✅ DEBUG: Перевіряємо, чи це правильний scaler
                                            if hasattr(target_scaler, 'scale_'):
                                                if target_scaler.scale_.shape[0] == 1:
                                                    self.logger.warning(f"⚠️ Точний scaler не знайдено, використовуємо {scaler_path.name} (shape: {target_scaler.scale_.shape})")
                                                else:
                                                    self.logger.error(f"❌ НЕПРАВИЛЬНИЙ scaler! Має {target_scaler.scale_.shape[0]} features замість 1")
                                                    target_scaler = None
                
                if target_scaler is None:
                    self.logger.warning(f"⚠️ Target scaler не знайдено для {context_id} - prediction залишиться нормалізованим")
                    self.logger.info(f"   💡 Порада: Переконайтеся, що моделі були тренувані в Colab з новим кодом (scaler_*.pkl файли)")

                # 2. Contextual Model Selection з KNN Similarity
                models_list = list(models.keys())
                target_type = meta.get('target_type', 'classification')
                
                # ✅ Спробуємо знайти схожі історичні ситуації через KNN
                best_model_name = None
                knn_confidence = 0.0
                
                try:
                    # Підготуємо дані для KNN: останні N рядків як target
                    target_features = ticker_df_clean.tail(5)  # Останні 5 рядків
                    
                    # Історичні дані з diary (якщо є)
                    # ✅ FIX: DiaryEngine не має get_all_performance(), використовуємо get_history_by_agent()
                    # Для KNN нам потрібна історія всіх агентів, тому пропускаємо цей крок
                    historical_performance = pd.DataFrame()  # Поки що порожній DataFrame
                    
                    if historical_performance is not None and not historical_performance.empty:
                        # Витягуємо features з історичних даних
                        # (припускаємо що diary зберігає context features)
                        historical_features = historical_performance.get('features', pd.DataFrame())
                        
                        if not historical_features.empty and len(historical_features.columns) == len(target_features.columns):
                            # Запускаємо KNN аналіз
                            knn_result = self.knn_similarity.analyze({
                                'historical_features': historical_features,
                                'target_features': target_features
                            })
                            
                            if 'similarities' in knn_result and knn_result['similarities']:
                                # Витягуємо найкращу модель з схожих ситуацій
                                similarities = knn_result['similarities']
                                
                                # Беремо останній рядок (найсвіжіший)
                                last_target_id = target_features.index[-1]
                                if last_target_id in similarities:
                                    similar_cases = similarities[last_target_id]
                                    
                                    if similar_cases:
                                        # Знаходимо яка модель найчастіше була успішною
                                        model_votes = {}
                                        for case in similar_cases[:3]:  # Top 3 схожі
                                            case_id = case['id']
                                            similarity_score = case['similarity_score']
                                            
                                            # Витягуємо модель з historical_performance
                                            case_model = historical_performance[historical_performance.index == case_id].get('model_name')
                                            if case_model is not None and not case_model.empty:
                                                model_name = case_model.iloc[0]
                                                if model_name in models_list:
                                                    model_votes[model_name] = model_votes.get(model_name, 0) + similarity_score
                                        
                                        if model_votes:
                                            best_model_name = max(model_votes, key=model_votes.get)
                                            knn_confidence = model_votes[best_model_name] / sum(model_votes.values())
                                            self.logger.info(f"🎯 KNN вибрав '{best_model_name}' з confidence {knn_confidence:.2f}")
                except Exception as e:
                    self.logger.warning(f"⚠️ KNN similarity failed: {e}, falling back to SmartModelSelector")
                
                # Fallback до SmartModelSelector якщо KNN не спрацював
                if best_model_name is None or best_model_name not in models_list:
                    best_model_name, _ = self.context_selector.select_best_model(
                        df=ticker_df_clean, 
                        ticker=ticker, 
                        target_type=target_type, 
                        available_models=models_list
                    )
                    self.logger.info(f"Contextual Selector chose '{best_model_name}' for {ticker} in {market_regime} regime.")
                else:
                    self.logger.info(f"KNN Similarity chose '{best_model_name}' for {ticker} (confidence: {knn_confidence:.2f})")

                # 3. Generate Ensemble Prediction
                # ✅ Розраховуємо anomaly score правильно (Z-score + Isolation Forest + LOF)
                anomaly_score = self._calculate_anomaly_score(ticker_df_clean)
                
                if len(models) > 1:
                    model_preds = {}
                    for m_name, m_inst in models.items():
                        # ✅ FIX: Використовуємо filtered_features_list замість feature_names_in_
                        # Це гарантує що моделі отримають правильну кількість фіч
                        feature_cols = filtered_features_list if filtered_features_list else ticker_df_clean.columns.tolist()
                        m_X = ticker_df_clean[feature_cols] if all(c in ticker_df_clean.columns for c in feature_cols) else ticker_df_clean
                        self.logger.debug(f"   {m_name}: X shape={m_X.shape}, features={len(feature_cols)}")
                        
                        # ✅ Autoencoder використовуємо для anomaly detection, а не prediction
                        if 'autoencoder' in m_name.lower():
                            # Autoencoder не використовується для prediction
                            # Його anomaly detection вже враховано у _calculate_anomaly_score
                            self.logger.debug(f"   ⏭️ Пропускаємо autoencoder для prediction (використовується тільки для anomaly detection)")
                            continue
                        
                        model_preds[m_name] = m_inst.predict(m_X)
                    
                    if not model_preds:
                        self.logger.warning(f"⚠️ Немає моделей для prediction (тільки autoencoder), пропускаємо {context_id}")
                        continue
                    
                    preds_df = pd.DataFrame(model_preds)
                    
                    ensemble_result = self.ensemble_factory.predict(
                        X=preds_df,
                        context_params={"ticker": ticker, "regime": market_regime}
                    )
                    raw_prediction = ensemble_result.final_signal
                    model_contributions = ensemble_result.active_weights
                else:
                    # Тільки одна модель
                    selected_model = models.get(best_model_name, list(models.values())[0])
                    
                    # ✅ Autoencoder - не використовується для prediction
                    if 'autoencoder' in best_model_name.lower():
                        self.logger.warning(f"⚠️ Autoencoder не підходить для regression prediction, пропускаємо {context_id}")
                        continue
                    
                    # ✅ FIX: Використовуємо filtered_features_list замість feature_names_in_
                    feature_cols = filtered_features_list if filtered_features_list else ticker_df_clean.columns.tolist()
                    X = ticker_df_clean[feature_cols] if all(c in ticker_df_clean.columns for c in feature_cols) else ticker_df_clean
                    self.logger.debug(f"   {best_model_name}: X shape={X.shape}, features={len(feature_cols)}")
                    raw_prediction = selected_model.predict(X)
                    # ✅ FIX: Передаємо реальний прогноз, а не вагу моделі
                    pred_value = raw_prediction[-1] if isinstance(raw_prediction, np.ndarray) else raw_prediction
                    model_contributions = {best_model_name: pred_value}

                # 4. Contextual Prediction Adjustment (Market Regime Awareness)
                # ✅ FIX: PredictionAdjuster.analyze() не .adjust()
                adjustment_result = self.adjuster.analyze(
                    data={
                        'predictions': {best_model_name: raw_prediction[-1] if isinstance(raw_prediction, np.ndarray) else raw_prediction},
                        'market_regime': market_regime,
                        'ticker': ticker
                    }
                )
                adjusted_prediction = adjustment_result.get('enhanced_predictions', {}).get(best_model_name, raw_prediction)
                
                # ✅ КРИТИЧНО: Денормалізація prediction назад до реальних значень
                if target_scaler is not None:
                    try:
                        self.logger.debug(f"   Денормалізація: prediction ДО = {adjusted_prediction}")
                        
                        # ✅ DEBUG: Перевіряємо scaler
                        self.logger.debug(f"   Scaler type: {type(target_scaler)}")
                        if hasattr(target_scaler, 'scale_'):
                            self.logger.debug(f"   Scaler.scale_ shape: {target_scaler.scale_.shape}")
                            self.logger.debug(f"   Scaler.mean_: {target_scaler.mean_}")
                        
                        # Конвертуємо prediction в правильний формат для scaler
                        # ВАЖЛИВО: scaler очікує shape (n_samples, 1) для target!
                        if isinstance(adjusted_prediction, np.ndarray):
                            if adjusted_prediction.ndim == 1:
                                pred_to_denorm = adjusted_prediction[-1:].reshape(-1, 1)
                            else:
                                pred_to_denorm = adjusted_prediction.reshape(-1, 1)
                        else:
                            pred_to_denorm = np.array([[adjusted_prediction]])
                        
                        self.logger.debug(f"   Prediction shape перед денормалізацією: {pred_to_denorm.shape}")
                        
                        # Перевіряємо розмір перед денормалізацією
                        if pred_to_denorm.shape[1] != 1:
                            self.logger.warning(f"⚠️ Неправильний розмір prediction: {pred_to_denorm.shape}, очікується (n, 1)")
                            # Якщо неправильний розмір, беремо тільки першу колонку
                            pred_to_denorm = pred_to_denorm[:, :1]
                            self.logger.debug(f"   Prediction shape після корекції: {pred_to_denorm.shape}")
                        
                        # ✅ КРИТИЧНО: Перевіряємо, чи scaler має правильну кількість features
                        if hasattr(target_scaler, 'scale_') and target_scaler.scale_.shape[0] != 1:
                            self.logger.error(f"❌ КРИТИЧНА ПОМИЛКА: Scaler має {target_scaler.scale_.shape[0]} features, очікується 1!")
                            self.logger.error(f"   Це означає, що завантажено scaler для FEATURES, а не для TARGET!")
                            self.logger.error(f"   Prediction залишається нормалізованим")
                            raise ValueError(f"Scaler має неправильну кількість features: {target_scaler.scale_.shape[0]} замість 1")
                        
                        # Денормалізуємо
                        denormalized = target_scaler.inverse_transform(pred_to_denorm)
                        adjusted_prediction = float(denormalized.flatten()[-1])
                        
                        self.logger.info(f"✅ Денормалізовано prediction: {adjusted_prediction:.6f}")
                    except Exception as e:
                        self.logger.warning(f"⚠️ Не вдалося денормалізувати prediction: {e}")
                        self.logger.debug(f"   Scaler shape: {target_scaler.scale_.shape if hasattr(target_scaler, 'scale_') else 'unknown'}")
                        self.logger.debug(f"   Prediction shape: {pred_to_denorm.shape if 'pred_to_denorm' in locals() else 'unknown'}")
                else:
                    self.logger.warning(f"⚠️ Target scaler не знайдено - prediction залишається нормалізованим!")

                # 5. Calculate Final Confidence
                confidence_info = self._calculate_ensemble_confidence(
                    models=models, 
                    X=ticker_df, 
                    prediction=adjusted_prediction, 
                    context_id=context_id
                )
                
                # ✅ Модифікуємо confidence на основі anomaly score
                final_confidence = confidence_info.get('score', 0.5) * anomaly_score
                if anomaly_score < 0.8:
                    self.logger.warning(f"⚠️ Низький anomaly score ({anomaly_score:.2f}) - можлива аномалія в даних!")

                prediction_results[context_id] = {
                    'ticker': ticker,
                    'predictions': adjusted_prediction,
                    'raw_forecast': raw_prediction,
                    'predictions_by_model': model_contributions,
                    'selected_primary_model': best_model_name,
                    'confidence': final_confidence,
                    'anomaly_score': anomaly_score,  # ✅ Додаємо anomaly score
                    'last_price': ticker_df['close'].iloc[-1] if 'close' in ticker_df.columns else (
                        ticker_df[f'{ticker}_1d_close'].iloc[-1] if f'{ticker}_1d_close' in ticker_df.columns else None
                    ),
                    'timestamp': ticker_df.index[-1] if isinstance(ticker_df.index, pd.DatetimeIndex) else None
                }
                
                # Handle both scalar and array predictions
                if isinstance(adjusted_prediction, (np.ndarray, list, pd.Series)) and len(adjusted_prediction) > 0:
                    pred_value = adjusted_prediction[-1]
                else:
                    pred_value = float(adjusted_prediction)
                
                self.logger.info(f"Ensemble forecast for {ticker}: {pred_value:.4f} | Conf: {confidence_info.get('score'):.2%}")

            except Exception as e:
                self.logger.error(f"Prediction failed for context {context_id}: {e}", exc_info=True)

        # ✅ Конвертуємо predictions в list для Stage 6
        predictions_list = list(prediction_results.values())
        
        # ✅ Збираємо current_prices для Stage 6
        current_prices = {}
        for context_id, pred_data in prediction_results.items():
            ticker = pred_data.get('ticker')
            last_price = pred_data.get('last_price')
            if ticker and last_price:
                current_prices[ticker] = last_price
        
        # ✅ FIX: Рахуємо light та heavy моделі
        light_models_count = 0
        heavy_models_count = 0
        
        for context_id, meta in models_meta.items():
            model_category = meta.get('model_category', meta.get('type', 'unknown'))
            if model_category == 'light':
                light_models_count += 1
            elif model_category == 'heavy' or model_category == 'colab':
                heavy_models_count += 1
        
        self.logger.info(f"✅ Stage 5 complete: {len(predictions_list)} predictions, {len(current_prices)} prices")
        self.logger.info(f"📊 Models: {light_models_count} light, {heavy_models_count} heavy, {len(models_meta)} total")
        
        # ✅ NEW: Збереження результатів Stage 5 на диск для гнучкого запуску
        self._save_stage_5_results(
            predictions_list=predictions_list,
            current_prices=current_prices,
            prediction_results=prediction_results,
            models_meta=models_meta,
            kwargs=kwargs
        )
        
        return {
            'predictions': predictions_list,
            'current_prices': current_prices,
            'prediction_results': prediction_results,  # Зберігаємо оригінальний формат для аналізу
            'models_metadata': models_meta,  # ✅ CRITICAL: Передаємо models_metadata для Stage 6
            'light_models_count': light_models_count,  # ✅ NEW
            'heavy_models_count': heavy_models_count,  # ✅ NEW
            'total_models': len(models_meta)  # ✅ NEW
        }

    def _create_pytorch_model(self, model_type: str, input_size: int):
        """
        Створює PyTorch модель за типом.
        
        ✅ ВАЖЛИВО: Архітектури повинні точно збігатися з Colab моделями!
        Dropout не має параметрів, тому його можна додавати/видаляти без впливу на state_dict.
        
        Args:
            model_type: Тип моделі (mlp, lstm, gru, cnn, transformer, tabnet, autoencoder)
            input_size: Розмір вхідних даних
            
        Returns:
            Екземпляр моделі
        """
        import torch
        import torch.nn as nn
        
        # ✅ ВАЖЛИВО: Архітектури для light models (catboost, lightgbm, xgboost, random_forest, linear, svm, knn)
        # Всі light models мають однакову архітектуру: 3 Linear layers (0, 3, 6)
        # Структура: Linear(input→128) + ReLU + Dropout + Linear(128→64) + ReLU + Dropout + Linear(64→1)
        # Індекси: 0 (Linear), 1 (ReLU), 2 (Dropout), 3 (Linear), 4 (ReLU), 5 (Dropout), 6 (Linear)
        light_models = ['catboost', 'lightgbm', 'xgboost', 'random_forest', 'linear', 'svm', 'knn']
        
        if model_type in light_models or model_type == 'tabnet':
            # Light models та TabNet: 3 Linear layers (indices 0, 3, 6)
            return nn.Sequential(
                nn.Linear(input_size, 128),      # 0: weight, bias
                nn.ReLU(),                        # 1
                nn.Dropout(0.5),                  # 2 (no params)
                nn.Linear(128, 64),               # 3: weight, bias
                nn.ReLU(),                        # 4
                nn.Dropout(0.5),                  # 5 (no params)
                nn.Linear(64, 1)                  # 6: weight, bias
            )
        elif model_type == 'mlp':
            # MLP: 4 Linear layers (indices 0, 3, 6, 8)
            return nn.Sequential(
                nn.Linear(input_size, 128),      # 0: weight, bias
                nn.ReLU(),                        # 1
                nn.Dropout(0.5),                  # 2 (no params)
                nn.Linear(128, 64),               # 3: weight, bias
                nn.ReLU(),                        # 4
                nn.Dropout(0.5),                  # 5 (no params)
                nn.Linear(64, 32),                # 6: weight, bias
                nn.ReLU(),                        # 7
                nn.Linear(32, 1)                  # 8: weight, bias
            )
        elif model_type == 'lstm':
            # ✅ LSTM: 2 шари з 64 hidden units
            class LSTMModel(nn.Module):
                def __init__(self, input_sz):
                    super().__init__()
                    self.lstm = nn.LSTM(input_sz, 64, 2, batch_first=True)
                    self.fc = nn.Linear(64, 1)
                def forward(self, x):
                    out, _ = self.lstm(x.unsqueeze(1))
                    return self.fc(out[:, -1, :])
            return LSTMModel(input_size)
        elif model_type == 'gru':
            # ✅ GRU: 2 шари з 64 hidden units
            class GRUModel(nn.Module):
                def __init__(self, input_sz):
                    super().__init__()
                    self.gru = nn.GRU(input_sz, 64, 2, batch_first=True)
                    self.fc = nn.Linear(64, 1)
                def forward(self, x):
                    out, _ = self.gru(x.unsqueeze(1))
                    return self.fc(out[:, -1, :])
            return GRUModel(input_size)
        elif model_type == 'cnn':
            # ✅ CNN: Conv1d(1->32->64) + FC
            class CNNModel(nn.Module):
                def __init__(self, input_sz):
                    super().__init__()
                    self.conv1 = nn.Conv1d(1, 32, kernel_size=3, padding=1)
                    self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
                    self.pool = nn.AdaptiveAvgPool1d(1)
                    self.fc = nn.Linear(64, 1)
                def forward(self, x):
                    x = x.unsqueeze(1)
                    x = torch.relu(self.conv1(x))
                    x = torch.relu(self.conv2(x))
                    return self.fc(self.pool(x).squeeze(-1))
            return CNNModel(input_size)
        elif model_type == 'transformer':
            # ✅ Transformer: embedding + 2 encoder layers
            class TransformerModel(nn.Module):
                def __init__(self, input_sz):
                    super().__init__()
                    self.embedding = nn.Linear(input_sz, 64)
                    encoder_layer = nn.TransformerEncoderLayer(64, 4, dim_feedforward=128, batch_first=True)
                    self.transformer = nn.TransformerEncoder(encoder_layer, 2)
                    self.fc = nn.Linear(64, 1)
                def forward(self, x):
                    x = self.embedding(x.unsqueeze(1))
                    x = self.transformer(x)
                    return self.fc(x[:, -1, :])
            return TransformerModel(input_size)
        elif model_type == 'autoencoder':
            # ✅ Autoencoder: encoder (2 layers) + decoder (2 layers) БЕЗ Dropout
            # Colab структура: encoder має Linear(47→64) + ReLU + Linear(64→32)
            class AutoencoderModel(nn.Module):
                def __init__(self, input_sz):
                    super().__init__()
                    # Encoder: input_sz -> 64 -> 32
                    self.encoder = nn.Sequential(
                        nn.Linear(input_sz, 64),      # 0: weight, bias
                        nn.ReLU(),                     # 1
                        nn.Linear(64, 32)              # 2: weight, bias
                    )
                    # Decoder: 32 -> 16 -> 1
                    self.decoder = nn.Sequential(
                        nn.Linear(32, 16),             # 0: weight, bias
                        nn.ReLU(),                     # 1
                        nn.Linear(16, 1)               # 2: weight, bias
                    )
                def forward(self, x):
                    encoded = self.encoder(x)
                    decoded = self.decoder(encoded)
                    return decoded
            return AutoencoderModel(input_size)
        else:
            # Fallback
            return nn.Sequential(
                nn.Linear(input_size, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 1)
            )

    def _wrap_pytorch_model(self, model, model_type: str, scaler=None):
        """
        Обгортає PyTorch модель щоб мати .predict() метод.
        
        Args:
            model: PyTorch модель
            model_type: Тип моделі
            scaler: Target scaler для денормалізації (опціонально)
            
        Returns:
            Обгорнута модель з .predict() методом
        """
        import torch
        
        class PyTorchPredictor:
            def __init__(self, pytorch_model, model_type, scaler=None):
                self.model = pytorch_model
                self.model_type = model_type
                self.scaler = scaler  # ✅ Зберігаємо scaler
                self.model.eval()
                
            def predict(self, X):
                """Генерує передбачення для X"""
                if isinstance(X, pd.DataFrame):
                    X = X.values
                
                # ✅ КРИТИЧНО: Нормалізуємо features перед prediction
                if self.scaler is not None:
                    X_normalized = self.scaler.transform(X)
                else:
                    X_normalized = X
                
                X_tensor = torch.FloatTensor(X_normalized)
                with torch.no_grad():
                    output = self.model(X_tensor)
                
                # Повертаємо як numpy array
                if isinstance(output, torch.Tensor):
                    return output.cpu().numpy().flatten()
                return output
        
        return PyTorchPredictor(model, model_type, scaler)

    def _save_stage_5_results(self, predictions_list: List[Dict], current_prices: Dict, prediction_results: Dict, models_meta: Dict, kwargs: Dict) -> None:
        """
        ✅ NEW: Збереження результатів Stage 5 на диск для гнучкого запуску.
        
        Зберігає stage_5_results.json у batch_dir для подальшого використання в Stage 6 та 7.
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
                stage_5_results = {
                    'timestamp': datetime.now().isoformat(),
                    'batch_name': batch_name,
                    'predictions': predictions_list,
                    'current_prices': current_prices,
                    'prediction_results': prediction_results,
                    'models_metadata': models_meta,
                    'light_models_count': sum(1 for m in models_meta.values() if m.get('model_category') == 'light'),
                    'heavy_models_count': sum(1 for m in models_meta.values() if m.get('model_category') in ['heavy', 'colab']),
                    'total_models': len(models_meta),
                    'total_predictions': len(predictions_list)
                }
                
                # Зберігаємо на диск
                results_file = batch_dir / "stage_5_results.json"
                with open(results_file, 'w') as f:
                    json.dump(stage_5_results, f, indent=2, default=str)
                
                self.logger.info(f"✅ Результати Stage 5 збережені: {results_file.name}")
        except Exception as e:
            self.logger.warning(f"⚠️ Помилка збереження результатів Stage 5: {e}")

    def _load_models_metadata_from_disk(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """
        ✅ NEW: Завантажує models_metadata з диска, якщо вона не передана через kwargs.
        
        Шукає:
        1. light_models_results_*.json (легкі моделі з етапу 4)
        2. colab_results_summary.json (важкі моделі з Colab)
        
        Returns:
            Dict з об'єднаними метаданими легких та важких моделей
        """
        import json
        from pathlib import Path
        
        models_metadata = {}
        
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
            
            # 1. Завантажуємо легкі моделі
            light_results_files = list(batch_dir.glob("light_models_results_*.json"))
            if light_results_files:
                latest_light = max(light_results_files, key=lambda p: p.stat().st_mtime)
                try:
                    with open(latest_light, 'r') as f:
                        light_results = json.load(f)
                        light_meta = light_results.get('models_metadata', {})
                        models_metadata.update(light_meta)
                        self.logger.info(f"✅ Завантажено {len(light_meta)} легких моделей з {latest_light.name}")
                except Exception as e:
                    self.logger.warning(f"⚠️ Помилка завантаження легких моделей: {e}")
            
            # 2. Завантажуємо важкі моделі з Colab
            colab_summary_file = batch_dir / "colab_results_summary.json"
            if colab_summary_file.exists():
                try:
                    with open(colab_summary_file, 'r') as f:
                        colab_results = json.load(f)
                        
                        # Витягуємо models_metadata з colab_results
                        if 'models_metadata' in colab_results:
                            heavy_meta = colab_results['models_metadata']
                            models_metadata.update(heavy_meta)
                            self.logger.info(f"✅ Завантажено {len(heavy_meta)} важких моделей з {colab_summary_file.name}")
                        else:
                            # Fallback: витягуємо з ticker_results
                            ticker_results = colab_results.get('ticker_results', {})
                            for ticker, ticker_data in ticker_results.items():
                                timeframes = ticker_data.get('timeframes', {})
                                for tf, tf_data in timeframes.items():
                                    results = tf_data.get('results', {})
                                    for target, target_data in results.items():
                                        models = target_data.get('models', {})
                                        for model_type, model_data in models.items():
                                            context_key = f"{ticker}_{target}_{model_type}"
                                            models_metadata[context_key] = {
                                                'ticker': ticker,
                                                'target': target,
                                                'winner': model_type,
                                                'model_type': model_type,
                                                'model_category': 'heavy',
                                                'metrics': model_data.get('metrics', {}),
                                                'selected_features': model_data.get('selected_features', [])
                                            }
                            self.logger.info(f"✅ Витягнено {len(models_metadata)} моделей з colab_results_summary.json")
                except Exception as e:
                    self.logger.warning(f"⚠️ Помилка завантаження важких моделей: {e}")
        
        return models_metadata

    def _load_available_models(self, context_id: str, models_meta: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Завантажує всі доступні моделі (Light/Heavy) для контексту.
        
        Гнучка логіка:
        1. Тестовий режим (test_ticker/test_target): шукає в models/ підпапці
        2. Звичайний режим: шукає в кореневій папці батча
        
        Підтримує:
        - .joblib файли (легкі моделі)
        - .pt файли (важкі моделі PyTorch) - як state_dict так і full models
        - .pkl файли (старий формат)
        
        Args:
            context_id: Ідентифікатор контексту (ticker_target_model)
            models_meta: Метадані моделей (передається з kwargs)
        """
        import torch
        
        loaded_models = {}
        
        # ✅ FIX: Визначаємо batch_dir з models_metadata (якщо є model_path)
        batch_dir = None
        models_meta = models_meta or {}
        
        self.logger.debug(f"🔍 _load_available_models: context_id={context_id}, models_meta keys={list(models_meta.keys())[:5]}")
        
        # Спробуємо витягти batch_dir з model_path
        if context_id in models_meta:
            model_path_str = models_meta[context_id].get('model_path', '')
            self.logger.debug(f"🔍 Знайдено model_path для {context_id}: {model_path_str}")
            if model_path_str:
                # ✅ FIX: Нормалізуємо шлях (замінюємо / на \ для Windows)
                model_path_str = model_path_str.replace('/', '\\')
                # Витягуємо batch_name з шляху (наприклад: ...\\test_ticker_amd_target_return_1d_ep5_iter5\\models\\...)
                parts = model_path_str.split('\\')
                if 'models' in parts:
                    models_idx = parts.index('models')
                    if models_idx > 0:
                        batch_name = parts[models_idx - 1]
                        base_dir = Path(self.config_manager.get('system.accumulation.output_dir', 'data/colab/accumulated'))
                        batch_dir = base_dir / batch_name
                        self.logger.info(f"📂 Витягнуто batch_dir з model_path: {batch_dir}")
        else:
            self.logger.warning(f"⚠️ context_id '{context_id}' не знайдено в models_meta")
        
        # Якщо не вдалося витягти, використовуємо дефолтний шлях
        if batch_dir is None:
            batch_dir = Path(self.config_manager.get('system.accumulation.output_dir', 'data/colab/accumulated'))
            self.logger.warning(f"⚠️ Не вдалося витягти batch_dir з model_path, використовуємо дефолтний: {batch_dir}")
        
        # Визначаємо режим з runtime_params.json
        runtime_params_path = batch_dir / "runtime_params.json"
        is_test_mode = False
        
        if runtime_params_path.exists():
            try:
                with open(runtime_params_path, 'r') as f:
                    runtime_params = json.load(f)
                    test_mode = runtime_params.get('test_mode', {})
                    is_test_mode = test_mode.get('enabled', False) and (
                        test_mode.get('test_ticker') or test_mode.get('test_target')
                    )
                    self.logger.info(f"📋 runtime_params: test_mode.enabled={is_test_mode}")
            except Exception as e:
                self.logger.warning(f"⚠️ Не вдалося прочитати runtime_params.json: {e}")
        else:
            self.logger.warning(f"⚠️ runtime_params.json не знайдено: {runtime_params_path}")
        
        # Визначаємо шлях до моделей
        if is_test_mode:
            # Тестовий режим: models/ підпапка
            models_search_paths = [
                batch_dir / 'models',
                self.models_path / 'models'
            ]
            self.logger.info(f"🧪 Тестовий режим: шукаємо моделі в {batch_dir / 'models'}")
        else:
            # Звичайний режим: коренева папка
            models_search_paths = [
                batch_dir,
                self.models_path
            ]
            self.logger.info(f"📦 Звичайний режим: шукаємо моделі в корені")
        
        # Патерни для пошуку
        # ✅ FIX: Розширені патерни для пошуку моделей
        # Витягуємо ticker, target, model_name з context_id
        # Формат context_id: AMD_target_return_1d_mlp (ticker_target_model)
        # Формат файлів: mlp_AMD_target_return_1d.pt (model_ticker_target)
        parts = context_id.split('_')
        if len(parts) >= 4:
            ticker = parts[0]
            target = '_'.join(parts[1:-1])  # target_return_1d
            model_name = parts[-1]  # mlp
            
            patterns = [
                # ✅ ПРАВИЛЬНИЙ ФОРМАТ: model_ticker_target
                # PyTorch моделі (важкі)
                f"{model_name}_{ticker}_{target}.pt",  # mlp_AMD_target_return_1d.pt
                f"{model_name}_{ticker}_*.pt",  # mlp_AMD_*.pt
                f"*{model_name}*.pt",  # *mlp*.pt
                # Joblib моделі (легкі)
                f"{model_name}_{ticker}_{target}.joblib",  # catboost_AMD_target_return_1d.joblib
                f"{model_name}_{ticker}_*.joblib",  # catboost_AMD_*.joblib
                f"*{model_name}*.joblib",  # *catboost*.joblib
                # Старі формати (для зворотної сумісності)
                f"CHAMP_{ticker}_{target}_{model_name}*.joblib",
                f"MODEL_{ticker}_{target}_{model_name}*.joblib",
                f"*{ticker}_{target}_{model_name}*.pkl"
            ]
        else:
            # Фоллбек до старих патернів
            patterns = [
                f"CHAMP_{context_id}*.joblib",
                f"MODEL_{context_id}*.joblib",
                f"*{context_id}*.pt",
                f"*{context_id}*.pkl"
            ]
        
        self.logger.debug(f"🔍 Патерни пошуку: {patterns}")
        
        for search_path in models_search_paths:
            if not search_path.exists():
                continue
                
            for pattern in patterns:
                for path in search_path.glob(pattern):
                    try:
                        model_name = path.stem.replace(f"_{context_id}", "")
                        
                        # Завантажуємо залежно від формату
                        if path.suffix == '.joblib':
                            loaded_models[model_name] = joblib.load(path)
                            self.logger.info(f"✅ Завантажено .joblib: {model_name}")
                        elif path.suffix == '.pt':
                            # PyTorch моделі потребують спеціального завантаження
                            try:
                                # Спробуємо з weights_only=False (для сумісності з PyTorch 2.6+)
                                loaded_obj = torch.load(path, map_location='cpu', weights_only=False)
                                
                                # ✅ FIX: Перевіряємо чи це state_dict (dict) чи full model
                                if isinstance(loaded_obj, dict):
                                    # Це може бути wrapper dict з метаданими або raw state_dict
                                    self.logger.info(f"🔧 Знайдено dict для {model_name}, аналізуємо формат...")
                                    
                                    # Перевіряємо чи це wrapper dict з метаданими
                                    if 'model_state_dict' in loaded_obj and 'input_size' in loaded_obj:
                                        # Wrapper dict формат (з Colab)
                                        self.logger.debug(f"   ✅ Wrapper dict формат (з метаданими)")
                                        state_dict = loaded_obj['model_state_dict']
                                        input_size = loaded_obj['input_size']
                                        saved_model_type = loaded_obj.get('model_type', model_name)
                                        self.logger.debug(f"   Model type: {saved_model_type}, Input size: {input_size}")
                                    else:
                                        # Raw state_dict
                                        self.logger.debug(f"   ✅ Raw state_dict формат")
                                        state_dict = loaded_obj
                                        
                                        # Витягуємо розмір вхідних даних з state_dict
                                        # Перший шар має форму (output_size, input_size)
                                        first_layer_key = None
                                        for key in state_dict.keys():
                                            if 'weight' in key and '0' in key:
                                                first_layer_key = key
                                                break
                                        
                                        if first_layer_key:
                                            input_size = state_dict[first_layer_key].shape[1]
                                            self.logger.debug(f"   Витягнуто input_size={input_size} з {first_layer_key}")
                                        else:
                                            # Фоллбек: спробуємо знайти будь-який weight
                                            for key, val in state_dict.items():
                                                if 'weight' in key and len(val.shape) >= 2:
                                                    input_size = val.shape[1]
                                                    self.logger.debug(f"   Витягнуто input_size={input_size} з {key}")
                                                    break
                                            else:
                                                input_size = 47  # Default для AMD test case
                                                self.logger.warning(f"   ⚠️ Не вдалося витягти input_size, використовуємо default={input_size}")
                                    
                                    # ✅ ВАЖЛИВО: Визначаємо saved_model_type для wrapper dict
                                    saved_model_type = model_name  # Default
                                    if 'model_state_dict' in loaded_obj and 'input_size' in loaded_obj:
                                        saved_model_type = loaded_obj.get('model_type', model_name)
                                    
                                    # ✅ КРИТИЧНО: Перевіряємо, чи це tree-based модель
                                    # Tree-based моделі (catboost, lightgbm, xgboost, random_forest, linear, svm, knn)
                                    # не мають PyTorch state dict, тому пропускаємо їх
                                    # ✅ ВАЖЛИВО: TabNet це PyTorch модель, НЕ tree-based!
                                    tree_based_models = ['catboost', 'lightgbm', 'xgboost', 'random_forest', 'linear', 'svm', 'knn']
                                    if saved_model_type in tree_based_models:
                                        self.logger.warning(f"⚠️ Tree-based модель {saved_model_type} не підтримується в Stage 5 (тренується локально)")
                                        self.logger.warning(f"   Пропускаємо {model_name}")
                                        continue
                                    
                                    # Реконструюємо модель
                                    pytorch_model = self._create_pytorch_model(saved_model_type, input_size)
                                    pytorch_model.load_state_dict(state_dict)
                                    
                                    # ✅ Витягуємо scaler якщо є
                                    scaler = loaded_obj.get('scaler') if isinstance(loaded_obj, dict) else None
                                    if scaler:
                                        self.logger.info(f"   ✅ Знайдено scaler для {model_name}")
                                    
                                    # Обгортаємо щоб мати .predict() метод
                                    loaded_models[model_name] = self._wrap_pytorch_model(pytorch_model, model_name, scaler)
                                    self.logger.info(f"✅ Реконструйовано та завантажено .pt: {model_name}")
                                else:
                                    # Це full model - просто обгортаємо
                                    loaded_models[model_name] = self._wrap_pytorch_model(loaded_obj, model_name)
                                    self.logger.info(f"✅ Завантажено .pt (full model): {model_name}")
                            except Exception as e:
                                self.logger.warning(f"⚠️ Помилка завантаження {path.name}: {e}")
                        elif path.suffix == '.pkl':
                            import pickle
                            with open(path, 'rb') as f:
                                loaded_models[model_name] = pickle.load(f)
                            self.logger.info(f"✅ Завантажено .pkl: {model_name}")
                    except Exception as e:
                        self.logger.warning(f"⚠️ Failed to load model from {path}: {e}")
        
        if not loaded_models:
            self.logger.warning(f"⚠️ Не знайдено жодної моделі для {context_id}")
        else:
            self.logger.info(f"🎯 Завантажено {len(loaded_models)} моделей для {context_id}")
        
        return loaded_models

    def _calculate_anomaly_score(self, X: pd.DataFrame, historical_data: Optional[pd.DataFrame] = None) -> float:
        """
        Розраховує anomaly score на основі:
        1. Z-score від історичного середнього
        2. Isolation Forest
        3. Local Outlier Factor (LOF)
        
        Returns:
            float: Anomaly score від 0 (нормально) до 1 (аномалія)
        """
        try:
            if X is None or len(X) == 0:
                return 0.5  # Невизначено
            
            # Отримуємо останній рядок даних
            current_row = X.iloc[-1:].values.flatten()
            
            # Якщо немає історичних даних, використовуємо поточні дані
            if historical_data is None or len(historical_data) < 2:
                historical_data = X
            
            historical_values = historical_data.values.flatten()
            
            scores = []
            
            # 1. Z-score метод
            try:
                mean = np.mean(historical_values)
                std = np.std(historical_values)
                
                if std > 0:
                    z_scores = np.abs((current_row - mean) / std)
                    # Якщо z-score > 3, це аномалія
                    z_anomaly = np.mean(z_scores > 3.0)
                    scores.append(z_anomaly)
                else:
                    scores.append(0.0)
            except:
                scores.append(0.0)
            
            # 2. Isolation Forest
            try:
                if len(historical_values) > 10:
                    iso_forest = IsolationForest(contamination=0.1, random_state=42)
                    iso_forest.fit(historical_values.reshape(-1, 1))
                    iso_pred = iso_forest.predict(current_row.reshape(-1, 1))
                    # -1 = аномалія, 1 = нормально
                    iso_anomaly = 1.0 if iso_pred[0] == -1 else 0.0
                    scores.append(iso_anomaly)
                else:
                    scores.append(0.0)
            except:
                scores.append(0.0)
            
            # 3. Local Outlier Factor (LOF)
            try:
                if len(historical_values) > 20:
                    lof = LocalOutlierFactor(n_neighbors=min(20, len(historical_values) - 1))
                    lof.fit(historical_values.reshape(-1, 1))
                    lof_pred = lof.predict(current_row.reshape(-1, 1))
                    # -1 = аномалія, 1 = нормально
                    lof_anomaly = 1.0 if lof_pred[0] == -1 else 0.0
                    scores.append(lof_anomaly)
                else:
                    scores.append(0.0)
            except:
                scores.append(0.0)
            
            # Комбінуємо оцінки
            if scores:
                anomaly_score = np.mean(scores)
            else:
                anomaly_score = 0.5
            
            return np.clip(anomaly_score, 0, 1)
        
        except Exception as e:
            self.logger.warning(f"⚠️ Помилка при розрахунку anomaly score: {e}")
            return 0.5

    def _calculate_anomaly_score(self, X: pd.DataFrame) -> float:
        """
        Розраховує anomaly score на основі:
        1. Z-score від історичного середнього
        2. Isolation Forest
        3. Local Outlier Factor (LOF)
        
        Повертає значення від 0 (нормально) до 1 (аномалія)
        """
        try:
            from sklearn.ensemble import IsolationForest
            from sklearn.neighbors import LocalOutlierFactor
            import numpy as np
            
            if X.empty or len(X) < 2:
                return 0.5  # Невизначено
            
            # Беремо останній рядок як поточні дані
            current_data = X.iloc[-1:].values
            historical_data = X.iloc[:-1].values if len(X) > 1 else X.values
            
            scores = []
            
            # 1. Z-score (0-1)
            try:
                if len(historical_data) > 1:
                    mean = np.mean(historical_data, axis=0)
                    std = np.std(historical_data, axis=0)
                    z_scores = np.abs((current_data - mean) / (std + 1e-6))
                    z_score_anomaly = np.mean(z_scores)
                    z_score_anomaly = np.clip(z_score_anomaly / 3.0, 0, 1)  # Нормалізуємо до [0, 1]
                    scores.append(z_score_anomaly)
                    self.logger.debug(f"   Z-score anomaly: {z_score_anomaly:.3f}")
                else:
                    scores.append(0.5)
            except Exception as e:
                self.logger.warning(f"   Z-score calculation failed: {e}")
                scores.append(0.5)
            
            # 2. Isolation Forest (0-1)
            try:
                if len(historical_data) > 5:
                    iso_forest = IsolationForest(contamination=0.1, random_state=42)
                    iso_forest.fit(historical_data)
                    iso_pred = iso_forest.predict(current_data)
                    iso_anomaly = 1.0 if iso_pred[0] == -1 else 0.0
                    scores.append(iso_anomaly)
                    self.logger.debug(f"   Isolation Forest anomaly: {iso_anomaly:.3f}")
                else:
                    scores.append(0.5)
            except Exception as e:
                self.logger.warning(f"   Isolation Forest failed: {e}")
                scores.append(0.5)
            
            # 3. Local Outlier Factor (0-1)
            try:
                if len(historical_data) > 5:
                    lof = LocalOutlierFactor(n_neighbors=min(20, len(historical_data)-1))
                    lof.fit(historical_data)
                    lof_pred = lof.predict(current_data)
                    lof_anomaly = 1.0 if lof_pred[0] == -1 else 0.0
                    scores.append(lof_anomaly)
                    self.logger.debug(f"   LOF anomaly: {lof_anomaly:.3f}")
                else:
                    scores.append(0.5)
            except Exception as e:
                self.logger.warning(f"   LOF failed: {e}")
                scores.append(0.5)
            
            # Комбінуємо з вагами
            final_anomaly = (scores[0] * 0.5 +  # Z-score: 50%
                           scores[1] * 0.3 +    # Isolation Forest: 30%
                           scores[2] * 0.2)     # LOF: 20%
            
            final_anomaly = np.clip(final_anomaly, 0, 1)
            self.logger.info(f"   📊 Final anomaly score: {final_anomaly:.3f} (z={scores[0]:.2f}, iso={scores[1]:.2f}, lof={scores[2]:.2f})")
            
            return final_anomaly
            
        except Exception as e:
            self.logger.warning(f"   ⚠️ Anomaly score calculation failed: {e}")
            return 0.5  # Невизначено
        """
        Calculates confidence based on:
        1. Model consensus (how many models agree)
        2. Model accuracy (historical performance)
        3. Market volatility (context)
        4. Prediction dispersion (variance)
        """
        scores = []
        
        # 1. Model Consensus (0-1)
        if len(models) > 1:
            try:
                preds = []
                for m in models.values():
                    try:
                        pred = m.predict(X)
                        if isinstance(pred, np.ndarray):
                            preds.append(pred[-1] if len(pred) > 0 else 0)
                        else:
                            preds.append(float(pred))
                    except:
                        continue
                
                if len(preds) > 1:
                    # Consensus: how many models agree on direction
                    mean_pred = np.mean(preds)
                    agreement = np.mean([1 if (p > 0) == (mean_pred > 0) else 0 for p in preds])
                    scores.append(agreement)
                else:
                    scores.append(0.5)
            except:
                scores.append(0.5)
        else:
            scores.append(0.5)
        
        # 2. Model Accuracy (0-1)
        try:
            perf = self.diary.get_recent_performance(context=context_id, window=30)
            accuracy = perf.get('accuracy', 0.5)
            scores.append(np.clip(accuracy, 0, 1))
        except:
            scores.append(0.5)
        
        # 3. Market Volatility Factor (0-1)
        try:
            if len(X) > 1:
                returns = np.diff(X.iloc[:, 0].values) / (X.iloc[:-1, 0].values + 1e-6)
                volatility = np.std(returns)
                # Lower volatility = higher confidence
                vol_factor = 1.0 / (1.0 + volatility * 10)
                scores.append(np.clip(vol_factor, 0, 1))
            else:
                scores.append(0.5)
        except:
            scores.append(0.5)
        
        # 4. Prediction Dispersion (0-1)
        try:
            if len(models) > 1:
                preds = []
                for m in models.values():
                    try:
                        pred = m.predict(X)
                        if isinstance(pred, np.ndarray):
                            preds.append(pred[-1] if len(pred) > 0 else 0)
                        else:
                            preds.append(float(pred))
                    except:
                        continue
                
                if len(preds) > 1:
                    variance = np.var(preds)
                    # Lower variance = higher confidence
                    dispersion = 1.0 / (1.0 + variance)
                    scores.append(np.clip(dispersion, 0, 1))
                else:
                    scores.append(0.5)
            else:
                scores.append(0.5)
        except:
            scores.append(0.5)
        
        # Combine scores with weights
        final_score = (scores[0] * 0.3 +  # Consensus: 30%
                      scores[1] * 0.3 +   # Accuracy: 30%
                      scores[2] * 0.2 +   # Volatility: 20%
                      scores[3] * 0.2)    # Dispersion: 20%
        
        return {'score': np.clip(final_score, 0, 1)}