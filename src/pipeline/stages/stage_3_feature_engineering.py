# src/pipeline/stages/stage_3_feature_engineering.py

import logging
import os
from typing import Optional, Any, Dict, List, Tuple
import pandas as pd
import numpy as np
from datetime import datetime
import psutil
from pathlib import Path

from src.pipeline.stages.base_stage import BaseStage
from src.config.unified_config_manager import UnifiedConfigManager, get_current_config
from src.core.error_handling.error_handler import ErrorHandler
from src.features.feature_orchestrator import FeatureOrchestrator
from src.features.selection.smart_selector import SmartFeatureSelector
from src.targets.target_orchestrator import TargetOrchestrator
from src.utils.trading_calendar import TradingCalendar
from src.core.logging.logger import ProjectLogger
from src.features.utils.datetime_utils import ensure_datetime_column, normalize_metadata_columns, deduplicate_on_metadata, ensure_datetime_sorted

# Advanced financial and context modules
from src.analytics.calculators.fama_french_factors import FamaFrenchFactors
from src.analytics.analyzers.hedge_fund_analyzer import HedgeFundAnalyzer
from src.analytics.analyzers.news_impact_analyzer import NewsImpactAnalyzer
from src.meta_learning.awareness.context_engine import ContextAwarenessEngine
from src.analytics.detectors.critical_signal_detector import CriticalSignalDetector
from src.analytics.analyzers.knn_similarity_finder import KnnSimilarityFinder
from src.analytics.analyzers.causal_event_finder import CausalEngine

logger = ProjectLogger.get_logger("FeatureEngineeringStage")

class FeatureEngineeringStage(BaseStage):
    """
    Stage 3: Advanced Feature Engineering Hub.
    Uses FeatureOrchestrator for modular enrichment and TargetOrchestrator for unified labeling.
    Leverages SmartFeatureSelector for final model feature selection.
    """
    def __init__(self, config_manager: UnifiedConfigManager, error_handler: ErrorHandler, **kwargs):
        super().__init__(config_manager, error_handler, **kwargs)
        self.feature_config = self.config_manager.get_config('features', default={})
        self.calendar = TradingCalendar()
        
        # Initialize dynamic Feature Orchestrator
        self.orchestrator = FeatureOrchestrator.create_from_config(config_manager)
        
        # Initialize TargetOrchestrator with the list of targets
        targets_list = self.config_manager.get('targets').as_dict() if hasattr(self.config_manager.get('targets'), 'as_dict') else self.config_manager.get('targets')
        self.target_orchestrator = TargetOrchestrator(targets_list=targets_list)

        self.selector = SmartFeatureSelector()

        self.output_dir = Path('data/processed/features')
        self.reports_dir = Path('reports/features')
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self.master_features_path = self.output_dir / "enriched_features.parquet"

    async def run(self, **kwargs) -> Dict[str, Any]:
        """
        Runs the streamlined feature engineering pipeline:
        1. Enrichment via FeatureOrchestrator.
        2. Target Generation.
        3. Final Feature Selection for the model.
        """
        cleaned_data = kwargs.get('cleaned_data')
        if not cleaned_data:
            logger.warning("Missing cleaned data for feature generation.")
            return {}

        logger.info(f"Starting Feature Engineering Pipeline. System RAM: {psutil.virtual_memory().percent}%")
        
        # Stage 2 повертає 'prices' або 'market_data'
        market_data_raw = cleaned_data.get('prices') or cleaned_data.get('market_data')
        
        if market_data_raw is None:
            # Спробуємо дістати безпосередньо з kwargs (fallback)
            market_data_raw = kwargs.get('market_data')

        logger.info(f"Market data type received: {type(market_data_raw)}")
        
        if market_data_raw is None:
            logger.error("Market data must be present.")
            return {"status": "failed", "reason": "invalid_market_data"}

        if isinstance(market_data_raw, pd.DataFrame):
            market_data_raw = {'1d': market_data_raw}
        elif not isinstance(market_data_raw, dict):
            logger.error("Market data must be a dictionary of timeframes for Event-Centric mode.")
            return {"status": "failed", "reason": "invalid_market_data"}

        try:
            # ✅ Зчитуємо runtime_params
            import json
            runtime_params = {}
            config_manager = get_current_config()
            params_path = config_manager.get_runtime_params_path()
            if params_path.exists():
                with open(params_path, 'r') as f:
                    runtime_params = json.load(f)
            
            test_mode = runtime_params.get('test_mode', {})
            test_ticker = test_mode.get('test_ticker')
            test_target = test_mode.get('test_target')

            enriched_prices = {}
            
            # Step 1: Enrich all timeframes with indicators and generate targets for 1d
            for tf, df_temp in market_data_raw.items():
                if isinstance(df_temp, dict) and 'data' in df_temp:
                    df_temp = df_temp['data']
                    
                if not isinstance(df_temp, pd.DataFrame):
                    logger.warning(f"Skipping {tf} because data is not a DataFrame. Got: {type(df_temp)}")
                    continue
                
                # Попереджаємо збої через втрату колонки 'interval' у кеші
                actual_tf = '1d' if tf == 'mixed' else tf
                    
                logger.info(f"Enriching time-series for timeframe {actual_tf}...")
                
                # Фільтруємо тікер, якщо ми в тестовому режимі
                if test_ticker and 'ticker' in df_temp.columns:
                    df_temp = df_temp[df_temp['ticker'] == test_ticker]

                if df_temp.empty:
                    logger.warning(f"Market data is empty for tf {tf} after filtering.")
                    continue

                # Add missing columns
                if 'ticker' not in df_temp.columns:
                    ticker_value = test_ticker if test_ticker else 'UNKNOWN'
                    df_temp['ticker'] = ticker_value
                if 'interval' not in df_temp.columns:
                    df_temp['interval'] = actual_tf
                
                df_enriched_tf = self.orchestrator.run(df_temp, **cleaned_data)
                
                # ✅ ПЕРЕВІРКА: Чи додано context_fingerprint
                if 'context_fingerprint' not in df_enriched_tf.columns:
                    logger.warning(f"⚠️ context_fingerprint відсутній після orchestrator.run() для {actual_tf}!")
                    logger.warning(f"⚠️ Форсуємо ContextMapEnricher...")
                    
                    # Форсуємо запуск ContextMapEnricher
                    from src.features.enrichers.context_map_enricher import ContextMapEnricher
                    context_enricher = ContextMapEnricher()
                    df_enriched_tf = context_enricher.enrich(df_enriched_tf)
                    
                    if 'context_fingerprint' in df_enriched_tf.columns:
                        logger.info(f"✅ context_fingerprint додано! Унікальних: {df_enriched_tf['context_fingerprint'].nunique()}")
                    else:
                        logger.error(f"❌ Не вдалося додати context_fingerprint навіть після форсування!")
                else:
                    logger.info(f"✅ context_fingerprint присутній: {df_enriched_tf['context_fingerprint'].nunique()} унікальних")
                
                if actual_tf == '1d': 
                    logger.info("Generating targets for 1d timeframe...")
                    df_enriched_tf = self._generate_targets(df_enriched_tf)
                
                enriched_prices[actual_tf] = df_enriched_tf
            
            if not enriched_prices:
                logger.error("No valid enriched price data generated across any timeframes.")
                return {"status": "failed", "reason": "no_enriched_prices"}

            # Step 2: Generate Event-Centric Dataset using Builder
            logger.info(f"Generating Event-Centric Dataset using NewsEventDatasetBuilder...")
            from src.features.builders.news_event_dataset_builder import NewsEventDatasetBuilder
            news_builder = NewsEventDatasetBuilder(self.calendar, runtime_params)
            
            news_df = cleaned_data.get('news')
            macro_data_raw = cleaned_data.get('macro_data', pd.DataFrame())
            
            # ✅ FIX: Перетворюємо FRED дані з long format в wide format
            if not macro_data_raw.empty and 'series_id' in macro_data_raw.columns:
                logger.info(f"📊 Перетворення macro_data з long format в wide format...")
                logger.info(f"   Raw macro shape: {macro_data_raw.shape}")
                
                # Визначаємо колонку з датою
                date_col = 'date' if 'date' in macro_data_raw.columns else 'datetime'
                
                # Pivot: date × series_id → columns
                try:
                    macro_data = macro_data_raw.pivot_table(
                        index=date_col,
                        columns='series_id',
                        values='value',
                        aggfunc='last'  # Беремо останнє значення якщо є дублікати
                    )
                    # Flatten column names
                    macro_data.columns = [f"{col}" for col in macro_data.columns]
                    macro_data = macro_data.reset_index()
                    logger.info(f"   ✅ Pivoted macro shape: {macro_data.shape}")
                    logger.info(f"   ✅ Macro columns: {macro_data.columns.tolist()[:10]}...")
                except Exception as e:
                    self.handle_stage_error(e, context="MacroDataPivot", severity="warning")
                    logger.warning(f"   ⚠️ Failed to pivot macro_data: {e}. Using empty DataFrame.")
                    macro_data = pd.DataFrame()
            else:
                macro_data = macro_data_raw
            
            # ✅ ДІАГНОСТИКА: Детальне логування
            logger.info(f"📰 cleaned_data keys: {cleaned_data.keys()}")
            logger.info(f"📰 news_df type: {type(news_df)}")
            logger.info(f"📰 news_df is None: {news_df is None}")
            if news_df is not None:
                logger.info(f"📰 news_df shape: {news_df.shape if isinstance(news_df, pd.DataFrame) else 'NOT A DATAFRAME'}")
                logger.info(f"📰 news_df empty: {news_df.empty if isinstance(news_df, pd.DataFrame) else 'N/A'}")
                if isinstance(news_df, pd.DataFrame) and not news_df.empty:
                    logger.info(f"📰 news_df columns: {news_df.columns.tolist()[:10]}")
                    logger.info(f"📰 news_df sample:\n{news_df.head(2)}")
            
            # ✅ FALLBACK: Якщо немає новин, створюємо синтетичні події на основі цін
            if news_df is None or (isinstance(news_df, pd.DataFrame) and news_df.empty):
                logger.warning("⚠️ No news data available. Creating synthetic events from price data...")
                logger.warning("⚠️ This is a fallback mode - results will be less accurate without real news.")
                
                # Створюємо синтетичні події на основі значних рухів цін
                price_1d = enriched_prices.get('1d')
                if price_1d is None or price_1d.empty:
                    logger.error("❌ No price data available for synthetic events. Cannot proceed.")
                    return {"status": "failed", "reason": "no_data"}
                
                # Генеруємо синтетичні події на основі волатильності
                synthetic_events = []
                for ticker in price_1d['ticker'].unique():
                    ticker_data = price_1d[price_1d['ticker'] == ticker].copy()
                    
                    # Розраховуємо денні зміни
                    ticker_data['price_change'] = ticker_data['close'].pct_change()
                    
                    # Створюємо події для значних рухів (>2% зміна)
                    significant_moves = ticker_data[abs(ticker_data['price_change']) > 0.02].copy()
                    
                    # Визначаємо назву колонки з датою/часом (може бути в індексі або колонці)
                    if 'timestamp' in significant_moves.columns:
                        datetime_col = 'timestamp'
                    elif 'datetime' in significant_moves.columns:
                        datetime_col = 'datetime'
                    else:
                        # Якщо дата в індексі, reset_index щоб отримати доступ
                        significant_moves = significant_moves.reset_index()
                        if 'timestamp' in significant_moves.columns:
                            datetime_col = 'timestamp'
                        elif 'datetime' in significant_moves.columns:
                            datetime_col = 'datetime'
                        else:
                            datetime_col = significant_moves.columns[0]  # Перша колонка після reset_index
                    
                    for idx, row in significant_moves.iterrows():
                        event = {
                            'datetime': row[datetime_col],
                            'ticker': ticker,
                            'title': f"Significant price movement: {row['price_change']*100:.2f}%",
                            'description': f"{ticker} moved {row['price_change']*100:.2f}% on {row[datetime_col]}",
                            'sentiment': 1.0 if row['price_change'] > 0 else -1.0,
                            'source': 'synthetic_price_event'
                        }
                        synthetic_events.append(event)
                
                if synthetic_events:
                    news_df = pd.DataFrame(synthetic_events)
                    logger.info(f"✅ Created {len(news_df)} synthetic events from price movements")
                else:
                    logger.error("❌ Could not create synthetic events. Cannot proceed.")
                    return {"status": "failed", "reason": "no_events"}
                
            tickers_to_process = [test_ticker] if test_ticker else list(enriched_prices.get('1d', pd.DataFrame()).get('ticker', pd.Series()).unique())
            
            logger.info(f"📊 Preparing to build event-centric dataset:")
            logger.info(f"   news_df: {news_df.shape if news_df is not None else 'None'}")
            logger.info(f"   tickers_to_process: {tickers_to_process}")
            logger.info(f"   enriched_prices keys: {list(enriched_prices.keys()) if enriched_prices else 'None'}")
            if enriched_prices:
                for tf, df in enriched_prices.items():
                    logger.info(f"   enriched_prices['{tf}']: {df.shape if df is not None else 'None'}")
            logger.info(f"   macro_data: {macro_data.shape if macro_data is not None else 'None'}")
            
            df_with_targets = news_builder.build_dataset(
                news_df=news_df,
                price_data=enriched_prices,
                macro_data=macro_data,
                tickers=tickers_to_process
            )
            
            logger.info(f"📊 news_builder.build_dataset() returned: type={type(df_with_targets)}, shape={df_with_targets.shape if df_with_targets is not None else 'None'}")
            
            if df_with_targets is None or df_with_targets.empty:
                logger.error("❌ Event-Centric Dataset Builder returned empty DataFrame.")
                logger.error(f"   news_df shape: {news_df.shape if news_df is not None else 'None'}")
                logger.error(f"   tickers_to_process: {tickers_to_process}")
                logger.error(f"   enriched_prices keys: {enriched_prices.keys() if enriched_prices else 'None'}")
                return {"status": "failed", "reason": "empty_event_dataset"}
                
            logger.info(f"✅ Event-centric dataset generated: shape={df_with_targets.shape}")
            logger.info(f"✅ Columns: {df_with_targets.columns.tolist()[:20]}")
            
            # ✅ RUN ENRICHERS ON EVENT-CENTRIC DATASET
            # This ensures enrichers like news_impact can work with news columns
            logger.info(f"🔄 Running enrichers on event-centric dataset...")
            logger.info(f"   Before enrichers: shape={df_with_targets.shape}, columns={len(df_with_targets.columns)}")
            try:
                df_with_targets = self.orchestrator.run(df_with_targets, **cleaned_data)
                logger.info(f"✅ Enrichers completed. Shape: {df_with_targets.shape}")
                logger.info(f"✅ Columns after enrichment: {df_with_targets.columns.tolist()[:30]}")
                
                if df_with_targets.empty:
                    logger.error("❌ КРИТИЧНА ПОМИЛКА: df_with_targets став порожнім після enrichers!")
                    logger.error(f"   Перевірте логи enrichers вище для деталей")
                    return {"status": "failed", "reason": "empty_after_enrichment"}
            except Exception as e:
                logger.error(f"❌ Enricher execution failed: {e}", exc_info=True)
                logger.warning(f"⚠️ Continuing without enrichment.")
                # Continue without enrichment rather than failing
            
            # Перевірка наявності таргетів
            target_cols = [c for c in df_with_targets.columns if c.lower().startswith('target_')]
            logger.info(f"✅ Found {len(target_cols)} target columns: {target_cols}")
            
            if not target_cols:
                logger.warning(f"❌ ТАРГЕТИ НЕ ЗНАЙДЕНІ у DataFrame після генерації! Перевірте TargetOrchestrator та targets.yaml. Колонки: {df_with_targets.columns.tolist()[:10]}...")
                return {"status": "failed", "reason": "no_targets_generated"}
            
            logger.info(f"Final selection for {test_ticker or 'all'}: {len(df_with_targets.columns)} columns.")
            logger.info(f"DataFrame shape before groupby: {df_with_targets.shape}")
            logger.info(f"Unique tickers: {df_with_targets['ticker'].unique() if 'ticker' in df_with_targets.columns else 'NO TICKER COLUMN'}")
            
            if self.master_features_path.exists() and not test_mode.get('enabled'):
                logger.info(f"Loading existing master features from {self.master_features_path}")
                try:
                    master_features_df = pd.read_parquet(self.master_features_path)
                    df_with_targets = pd.concat([master_features_df, df_with_targets], ignore_index=True)
                    df_with_targets.drop_duplicates(subset=['ticker', 'datetime'], keep='last', inplace=True)
                except Exception as e:
                    logger.warning(f"Could not load master features: {e}. Starting fresh.")

            processed_tickers = []
            version = datetime.now().strftime("%Y%m%d_%H%M")
            
            # ✅ КРИТИЧНИЙ FIX: Перевірка наявності ticker колонки
            if 'ticker' not in df_with_targets.columns:
                logger.error("❌ КРИТИЧНА ПОМИЛКА: Колонка 'ticker' відсутня в df_with_targets!")
                logger.error(f"Доступні колонки: {df_with_targets.columns.tolist()[:20]}")
                return {
                    'enriched_data': pd.DataFrame(),
                    'feature_version': version,
                    'status': 'no_ticker_column'
                }

            for ticker, group in df_with_targets.groupby('ticker'):
                logger.info(f"Processing final feature selection for Ticker: {ticker}, Group shape: {group.shape}")
                
                target_cols = [c for c in group.columns if c.startswith('target_')]
                logger.info(f"Found {len(target_cols)} target columns: {target_cols}")
                
                if not target_cols:
                    logger.warning(f"No targets found for {ticker}, skipping selection.")
                    continue
                
                # Використовуємо test_target як primary, якщо він є
                primary_target = test_target if test_target in target_cols else target_cols[0]
                context_id = f"{ticker}_{primary_target}"
                logger.info(f"Using primary target: {primary_target}")
                
                df_task = group.dropna(subset=[primary_target]).fillna(0)
                logger.info(f"After dropna on {primary_target}: {df_task.shape}")
                
                if df_task.empty:
                    logger.warning(f"No valid target rows for {ticker} / {primary_target}, skipping.")
                    continue
                
                exclude_metadata = ['datetime', 'ticker'] + target_cols
                feature_pool = [c for c in df_task.columns if c not in exclude_metadata]
                
                X = df_task[feature_pool]
                y = df_task[primary_target]
                
                try:
                    selected_features = self.selector.select(X, y, context_id=context_id)
                except Exception as e:
                    logger.warning(f"Feature selection failed for {ticker}: {e}. Using all features.")
                    selected_features = feature_pool
                
                final_cols = selected_features + target_cols + ['ticker', 'context_fingerprint', 'datetime']
                final_cols_exist = [c for c in final_cols if c in df_task.columns]
                
                # ✅ КРИТИЧНИЙ FIX: Перевірка наявності datetime та ticker
                # Спочатку шукаємо datetime, якщо немає - використовуємо published_at
                datetime_col = None
                if 'datetime' in df_task.columns:
                    datetime_col = 'datetime'
                elif 'published_at' in df_task.columns:
                    datetime_col = 'published_at'
                    logger.info(f"✅ Використовуємо 'published_at' замість 'datetime'")
                elif df_task.index.name == 'datetime' or isinstance(df_task.index, pd.DatetimeIndex):
                    logger.info(f"✅ Знайдено datetime в індексі, додаємо як колонку")
                    df_task = df_task.reset_index()
                    if 'datetime' in df_task.columns:
                        datetime_col = 'datetime'
                    elif 'published_at' in df_task.columns:
                        datetime_col = 'published_at'
                
                if datetime_col is None:
                    logger.error(f"❌ КРИТИЧНА ПОМИЛКА: 'datetime' або 'published_at' відсутні в df_task для {ticker}!")
                    logger.error(f"   Доступні колонки: {df_task.columns.tolist()[:30]}")
                    logger.error(f"   final_cols: {final_cols[:10]}")
                    logger.error(f"   final_cols_exist: {final_cols_exist[:10]}")
                    logger.error(f"❌ datetime не знайдено ні в колонках, ні в індексі. Пропускаємо {ticker}")
                    continue
                
                # Якщо використовуємо published_at, додаємо її замість datetime
                if datetime_col == 'published_at' and 'datetime' in final_cols:
                    final_cols = [datetime_col if c == 'datetime' else c for c in final_cols]
                    final_cols_exist = [c for c in final_cols if c in df_task.columns]
                    logger.info(f"✅ Замінено 'datetime' на 'published_at' в final_cols")
                
                if 'ticker' not in final_cols_exist:
                    logger.warning(f"⚠️ 'ticker' відсутня в df_task, але це нормально якщо вона буде додана пізніше")

                final_df = df_task[final_cols_exist]
                processed_tickers.append(final_df)

            if processed_tickers:
                master_features_df = pd.concat(processed_tickers)
                
                # ✅ КРИТИЧНИЙ FIX: Видалити дублікати колонок перед обробкою
                if master_features_df.columns.duplicated().any():
                    duplicated_cols = master_features_df.columns[master_features_df.columns.duplicated()].tolist()
                    logger.warning(f"⚠️ Знайдено {len(duplicated_cols)} дублікатів колонок: {duplicated_cols[:10]}")
                    logger.warning(f"   Видаляємо дублікати (залишаємо перше входження)")
                    master_features_df = master_features_df.loc[:, ~master_features_df.columns.duplicated()]
                
                # ✅ Видалити дублікати за datetime/published_at та ticker (якщо вони є)
                logger.info(f"Before dedup: {len(master_features_df)} rows")
                logger.info(f"Columns in master_features_df: {master_features_df.columns.tolist()[:30]}")
                
                dedup_cols = []
                # Шукаємо datetime або published_at
                if 'datetime' in master_features_df.columns:
                    dedup_cols.append('datetime')
                elif 'published_at' in master_features_df.columns:
                    dedup_cols.append('published_at')
                    logger.info(f"✅ Використовуємо 'published_at' для dedup замість 'datetime'")
                
                if 'ticker' in master_features_df.columns:
                    dedup_cols.append('ticker')
                
                if dedup_cols:
                    master_features_df = master_features_df.drop_duplicates(
                        subset=dedup_cols, 
                        keep='first'
                    ).reset_index(drop=True)
                    logger.info(f"After dedup by {dedup_cols}: {len(master_features_df)} rows")
                else:
                    logger.warning(f"⚠️ Не можу видалити дублікати - немає datetime/published_at або ticker колонок")
                    master_features_df = master_features_df.reset_index(drop=True)
                
                # ✅ CRITICAL FIX: Use utility to ensure proper datetime handling
                try:
                    master_features_df = normalize_metadata_columns(master_features_df)
                    logger.info(f"✅ Normalized datetime/ticker columns (final stage 3 check)")
                except Exception as e:
                    logger.warning(f"⚠️ Metadata normalization failed: {e}")
                
                # ✅ Сортувати за datetime/published_at та ticker (якщо вони є)
                sort_cols = []
                if 'datetime' in master_features_df.columns:
                    sort_cols.append('datetime')
                elif 'published_at' in master_features_df.columns:
                    sort_cols.append('published_at')
                    logger.info(f"✅ Використовуємо 'published_at' для сортування замість 'datetime'")
                
                if 'ticker' in master_features_df.columns:
                    sort_cols.append('ticker')
                
                if sort_cols:
                    master_features_df = master_features_df.sort_values(sort_cols).reset_index(drop=True)
                    logger.info(f"✅ Sorted by {sort_cols}")
                else:
                    logger.warning(f"⚠️ Не можу сортувати - немає datetime/published_at або ticker колонок")
                
                # Зберігаємо тільки якщо ми НЕ в тестовому режимі
                if not test_mode.get('enabled'):
                    master_features_df.to_parquet(self.master_features_path)
                    logger.info(f"Feature Engineering complete. Master file saved: {self.master_features_path}")
                else:
                    logger.info(f"Feature Engineering complete. Test mode: skipping master file save.")
                
                logger.info(f"Returning enriched_data with {len(master_features_df)} rows and {len(master_features_df.columns)} columns")
                logger.info(f"Columns: {master_features_df.columns.tolist()[:20]}")
                
                result = {
                    'enriched_data': master_features_df,
                    'feature_version': version
                }
                return result
            else:
                logger.warning("No tickers were processed - returning empty enriched_data")
                return {
                    'enriched_data': pd.DataFrame(),
                    'feature_version': version,
                    'status': 'no_data_processed'
                }

        except Exception as e:
            logger.error(f"Critical error in FeatureEngineeringStage: {e}", exc_info=True)
            # Повертаємо порожній DataFrame замість raise, щоб pipeline міг продовжити
            return {
                'enriched_data': pd.DataFrame(),
                'status': 'failed',
                'error': str(e)
            }

    def _generate_targets(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generates targets using the configured TargetOrchestrator."""
        logger.info(f"_generate_targets called with df shape: {df.shape}")
        logger.info(f"Columns before target generation: {df.columns.tolist()[:20]}")
        
        # ✅ Перевірка наявності ticker
        if 'ticker' not in df.columns:
            logger.error("❌ CRITICAL: 'ticker' column missing before target generation!")
            logger.error(f"Available columns: {df.columns.tolist()}")
            raise ValueError("Missing 'ticker' column before target generation")
        
        result = self.target_orchestrator.generate_targets(df)
        logger.info(f"_generate_targets result shape: {result.shape}")
        logger.info(f"Columns after target generation: {result.columns.tolist()[:20]}")
        return result
