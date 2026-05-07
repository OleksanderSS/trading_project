# src/pipeline/stages/stage_3_feature_engineering.py

import os
from typing import Optional, Any, Dict, List, Tuple
import pandas as pd
import numpy as np
from datetime import datetime
import psutil
from pathlib import Path

from src.pipeline.stages.base_stage import BaseStage
from src.pipeline.stages.stage_3_improvements import (
    validate_and_align_features_targets,
    calculate_data_quality_metrics,
    log_data_quality_report
)
from src.config.unified_config_manager import UnifiedConfigManager, get_current_config
from src.core.error_handling.error_handler import ErrorHandler
from src.features.feature_orchestrator import FeatureOrchestrator
from src.features.selection.smart_selector import SmartFeatureSelector
from src.features.selection.enhanced_smart_selector import get_enhanced_smart_selector
from src.features.news_dataset_builder import NewsContextDatasetBuilder
from src.features.news_clusterer import cluster_news_simple
from src.features.validation.feature_leakage_guard import get_leakage_guard
from src.targets.target_orchestrator import TargetOrchestrator
from src.utils.trading_calendar import TradingCalendar
from src.core.logging.logger import ProjectLogger
from src.features.utils.datetime_utils import ensure_datetime_column, normalize_metadata_columns, deduplicate_on_metadata, ensure_datetime_sorted
from src.data.validation.event_dataset_validator import EventDatasetValidator

# ✅ NEW: Temporal safety guards
from src.pipeline.guards.timeframe_alignment_guard import get_timeframe_alignment_guard
from src.pipeline.guards.safe_feature_combiner import get_safe_feature_combiner
from src.pipeline.guards.temporal_target_guard import get_temporal_target_guard
from src.pipeline.guards.temporal_leakage_guard import get_temporal_leakage_guard
from src.pipeline.guards.macro_release_timing_guard import get_macro_release_timing_guard

# ✅ NEW: Enhanced monitoring components
from src.monitoring.feature_drift_monitor import get_feature_drift_monitor
from src.monitoring.data_freshness_monitor import get_data_freshness_monitor
from src.features.analysis.regime_importance_tracker import get_regime_importance_tracker

# Advanced financial and context modules
from src.analytics.calculators.fama_french_factors import FamaFrenchFactors
from src.analytics.analyzers.hedge_fund_analyzer import HedgeFundAnalyzer
from src.analytics.analyzers.news_impact_analyzer import NewsImpactAnalyzer
from src.meta_learning.awareness.context_engine import ContextAwarenessEngine
from src.analytics.detectors.critical_signal_detector import CriticalSignalDetector
from src.analytics.analyzers.knn_similarity_finder import KnnSimilarityFinder
from src.features.feature_cache import get_feature_cache, get_cache_stats

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
        
        # Initialize NewsContextDatasetBuilder for news-based features
        self.news_builder = NewsContextDatasetBuilder(config_manager)
        
        # Initialize TargetOrchestrator with the list of targets
        targets_list = self.config_manager.get('targets').as_dict() if hasattr(self.config_manager.get('targets'), 'as_dict') else self.config_manager.get('targets')
        self.target_orchestrator = TargetOrchestrator(targets_list=targets_list)

        # ✅ ENHANCED: Use Enhanced Smart Feature Selector with full monitoring integration
        self.selector = get_enhanced_smart_selector(config_manager)
        self.event_dataset_validator = EventDatasetValidator()

        self.output_dir = Path('data/processed/features')
        self.reports_dir = Path('reports/features')
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self.master_features_path = self.output_dir / "enriched_features.parquet"
        
        # ✅ Phase 3 Optimization: Initialize feature cache (60-80% speedup)
        cache_dir = self.config_manager.get('performance.feature_cache_dir', 'data/cache/features')
        self.feature_cache = get_feature_cache(cache_dir=cache_dir)
        logger.info("✅ Feature cache enabled (disk-based, parquet compression)")
        
        # ✅ NEW: Initialize temporal safety guards (will be configured in run method based on mode)
        self.timeframe_guard = None
        self.safe_combiner = None
        self.temporal_target_guard = None
        self.temporal_leakage_guard = None
        self.macro_guard = None
        
        # ✅ NEW: Initialize enhanced monitoring components
        self.drift_monitor = get_feature_drift_monitor()
        self.freshness_monitor = get_data_freshness_monitor()
        self.regime_tracker = get_regime_importance_tracker()
        logger.info("✅ Enhanced monitoring components initialized")

    def _validate_and_prepare_market_data(self, **kwargs) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Validate and prepare market data for processing."""
        cleaned_data = kwargs.get('cleaned_data')
        if not cleaned_data:
            logger.warning("Missing cleaned data for feature generation.")
            return {}, {}

        logger.info(f"Starting Feature Engineering Pipeline. System RAM: {psutil.virtual_memory().percent}%")
        
        # Stage 2 returns 'prices' or 'market_data'
        market_data_raw = cleaned_data.get('prices') or cleaned_data.get('market_data')
        
        if market_data_raw is None:
            # Try to get directly from kwargs (fallback)
            market_data_raw = kwargs.get('market_data')

        logger.info(f"Market data type received: {type(market_data_raw)}")
        
        if market_data_raw is None:
            logger.error("Market data must be present.")
            return {}, {"status": "failed", "reason": "invalid_market_data"}

        if isinstance(market_data_raw, pd.DataFrame):
            market_data_raw = {'1d': market_data_raw}
        elif not isinstance(market_data_raw, dict):
            logger.error("Market data must be a dictionary of timeframes for Event-Centric mode.")
            return {}, {"status": "failed", "reason": "invalid_market_data"}
        
        return cleaned_data, market_data_raw

    def _load_runtime_params(self, batch_name: Optional[str] = None) -> Dict[str, Any]:
        """Load runtime parameters from file."""
        import json
        runtime_params = {}
        config_manager = get_current_config()
        params_path = config_manager.get_runtime_params_path(batch_name=batch_name)
        if params_path.exists():
            try:
                with open(params_path, 'r') as f:
                    content = f.read()
                    runtime_params = json.loads(content)
            except Exception as e:
                logger.warning(f"Could not load runtime_params.json: {e}")
        return runtime_params
    
    def _initialize_guards_based_on_mode(self, mode: str = "full"):
        """Initialize temporal safety guards based on pipeline mode.
        
        Args:
            mode: Pipeline mode - "prepare" for data accumulation, "full" for live trading
        """
        from src.pipeline.guards.timeframe_alignment_guard import get_timeframe_alignment_guard
        from src.pipeline.guards.safe_feature_combiner import get_safe_feature_combiner
        from src.pipeline.guards.temporal_target_guard import get_temporal_target_guard
        from src.pipeline.guards.temporal_leakage_guard import get_temporal_leakage_guard
        from src.pipeline.guards.macro_release_timing_guard import get_macro_release_timing_guard
        
        # Use strict mode for live trading, non-strict for data accumulation
        strict_mode = mode != "prepare"
        
        self.timeframe_guard = get_timeframe_alignment_guard(strict_mode=strict_mode)
        self.safe_combiner = get_safe_feature_combiner(self.timeframe_guard)
        self.temporal_target_guard = get_temporal_target_guard()
        self.temporal_leakage_guard = get_temporal_leakage_guard()
        self.macro_guard = get_macro_release_timing_guard()
        
        logger.info(f"✅ Temporal safety guards initialized (mode: {mode}, strict: {strict_mode})")

    def _process_single_timeframe(self, tf: str, df_temp: pd.DataFrame, cleaned_data: Dict[str, Any], 
                                test_ticker: Optional[str]) -> Optional[pd.DataFrame]:
        """Process a single timeframe for feature enrichment."""
        if isinstance(df_temp, dict) and 'data' in df_temp:
            df_temp = df_temp['data']
            
        if not isinstance(df_temp, pd.DataFrame):
            logger.warning(f"Skipping {tf} because data is not a DataFrame. Got: {type(df_temp)}")
            return None
        
        # Prevent failures due to loss of 'interval' column in cache
        actual_tf = '1d' if tf == 'mixed' else tf
            
        logger.info(f"Enriching time-series for timeframe {actual_tf}...")
        
        df_temp = self._filter_test_ticker(df_temp, test_ticker)
        
        if df_temp.empty:
            logger.warning(f"Market data is empty for tf {tf} after filtering.")
            return None
        
        # ✅ FIX: Ensure datetime is a column BEFORE any processing
        df_temp = self._ensure_datetime_column(df_temp, actual_tf)
        
        # ✅ FIX: Remove any NaT values BEFORE enrichment
        if 'datetime' in df_temp.columns and df_temp['datetime'].isna().any():
            nat_count = df_temp['datetime'].isna().sum()
            logger.warning(f"⚠️ Found {nat_count} NaT values in {actual_tf} datetime column BEFORE enrichment")
            
            # Drop rows with NaT datetime (can't be enriched properly)
            df_temp = df_temp.dropna(subset=['datetime'])
            logger.info(f"✅ Dropped {nat_count} rows with NaT datetime, remaining: {len(df_temp)} rows")
            
            if df_temp.empty:
                logger.error(f"❌ All rows had NaT datetime for {actual_tf}")
                return None
        
        df_temp = self._add_missing_columns(df_temp, actual_tf)
        
        # Enrich with cache
        df_enriched = self._enrich_with_cache(df_temp, actual_tf, cleaned_data)
        
        # ✅ CRITICAL FIX: Ensure the enriched DataFrame has datetime column
        if df_enriched is not None and not df_enriched.empty:
            # ✅ DEBUG: Log datetime status before _ensure_enriched_datetime
            has_datetime_before = 'datetime' in df_enriched.columns
            logger.debug(f"🔍 Before _ensure_enriched_datetime for {actual_tf}: datetime={'✅' if has_datetime_before else '❌'}")
            
            df_enriched = self._ensure_enriched_datetime(df_enriched, df_temp, actual_tf)
            
            # ✅ DEBUG: Log datetime status after _ensure_enriched_datetime
            has_datetime_after = 'datetime' in df_enriched.columns
            logger.debug(f"🔍 After _ensure_enriched_datetime for {actual_tf}: datetime={'✅' if has_datetime_after else '❌'}")
            
            if not has_datetime_after:
                logger.error(f"❌ CRITICAL: datetime column lost for {actual_tf}!")
                logger.error(f"   Columns: {df_enriched.columns.tolist()[:20]}")
        
        return df_enriched
    
    def _ensure_datetime_column(self, df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        """Ensure datetime column exists and is valid."""
        if 'datetime' not in df.columns:
            if isinstance(df.index, pd.DatetimeIndex):
                df = df.reset_index()
                if 'index' in df.columns:
                    df = df.rename(columns={'index': 'datetime'})
                logger.info(f"✅ Converted DatetimeIndex to datetime column for {timeframe}")
            elif 'timestamp' in df.columns:
                df['datetime'] = pd.to_datetime(df['timestamp'])
                logger.info(f"✅ Converted timestamp to datetime column for {timeframe}")
            elif 'date' in df.columns:
                df['datetime'] = pd.to_datetime(df['date'])
                logger.info(f"✅ Converted date to datetime column for {timeframe}")
            else:
                logger.error(f"❌ No datetime source found for {timeframe}")
                logger.error(f"   Available columns: {df.columns.tolist()}")
                raise ValueError(f"No datetime column found for {timeframe}")
        
        # Ensure datetime is timezone-naive to avoid comparison issues
        if hasattr(df['datetime'].dt, 'tz') and df['datetime'].dt.tz is not None:
            df['datetime'] = df['datetime'].dt.tz_localize(None)
        
        return df
    
    def _ensure_enriched_datetime(self, df_enriched: pd.DataFrame, df_temp: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        """Ensure enriched DataFrame has valid datetime column."""
        if 'datetime' not in df_enriched.columns:
            logger.warning(f"⚠️ Enriched DataFrame for {timeframe} missing datetime column. Adding from original.")
            
            # Try to get datetime from original df_temp
            if 'datetime' in df_temp.columns:
                # Check if lengths match
                if len(df_enriched) == len(df_temp):
                    df_enriched['datetime'] = df_temp['datetime'].values
                    logger.info(f"✅ Copied datetime from original data for {timeframe}")
                else:
                    logger.error(f"❌ Length mismatch: enriched={len(df_enriched)}, original={len(df_temp)}")
                    logger.error(f"   Cannot copy datetime - creating synthetic datetime")
                    df_enriched['datetime'] = pd.date_range(start='2024-01-01', periods=len(df_enriched), freq='D')
            else:
                logger.error(f"❌ Original data has no datetime column for {timeframe}")
                df_enriched['datetime'] = pd.date_range(start='2024-01-01', periods=len(df_enriched), freq='D')
        else:
            # Ensure existing datetime column is timezone-naive
            if hasattr(df_enriched['datetime'].dt, 'tz') and df_enriched['datetime'].dt.tz is not None:
                df_enriched['datetime'] = df_enriched['datetime'].dt.tz_localize(None)
        
        # ✅ FIX: Check for NaT values in enriched datetime
        if df_enriched['datetime'].isna().any():
            nat_count = df_enriched['datetime'].isna().sum()
            logger.error(f"❌ Enriched DataFrame for {timeframe} has {nat_count} NaT values in datetime column")
            logger.error(f"   This should not happen - dropping rows with NaT datetime")
            df_enriched = df_enriched.dropna(subset=['datetime'])
            logger.info(f"✅ Dropped {nat_count} rows with NaT datetime, remaining: {len(df_enriched)} rows")
        
        return df_enriched

    def _filter_test_ticker(self, df_temp: pd.DataFrame, test_ticker: Optional[str]) -> pd.DataFrame:
        """Filter dataframe for test ticker."""
        if test_ticker and 'ticker' in df_temp.columns:
            return df_temp[df_temp['ticker'] == test_ticker]
        return df_temp
    
    def _add_missing_columns(self, df_temp: pd.DataFrame, actual_tf: str) -> pd.DataFrame:
        """Add missing columns to dataframe."""
        if 'ticker' not in df_temp.columns:
            df_temp['ticker'] = 'UNKNOWN'
        if 'interval' not in df_temp.columns:
            df_temp['interval'] = actual_tf
        return df_temp

    def _enrich_with_cache(self, df_temp: pd.DataFrame, actual_tf: str, 
                          cleaned_data: Dict[str, Any]) -> pd.DataFrame:
        """Enrich dataframe with feature caching."""
        # ✅ Phase 3 Optimization: Feature caching (60-80% speedup for repeated enrichments)
        import hashlib
        import json
        
        # Generate cache key from ticker, timeframe, and enricher config
        ticker_for_cache = df_temp['ticker'].iloc[0] if not df_temp.empty else 'UNKNOWN'
        
        # Hash enricher configuration for cache invalidation
        enricher_config = self.orchestrator.get_config_hash() if hasattr(self.orchestrator, 'get_config_hash') else str(self.feature_config)
        config_hash = hashlib.sha256(json.dumps(enricher_config, sort_keys=True).encode()).hexdigest()
        
        # Use ticker + timeframe + config as cache identifier
        cache_date_key = f"{ticker_for_cache}_{actual_tf}_{config_hash[:8]}"
        
        # Try to get from cache first
        df_enriched_tf = self.feature_cache.get_features(ticker_for_cache, cache_date_key, config_hash)
        
        if df_enriched_tf is not None:
            logger.info(f"🚀 Feature cache hit for {ticker_for_cache} {actual_tf} ({len(df_enriched_tf)} rows)")
            # Check if context_fingerprint is missing in cached data
            df_enriched_tf = self._validate_context_fingerprint(df_enriched_tf)
            
            # Update cache with context_fingerprint if it was added
            if 'context_fingerprint' in df_enriched_tf.columns:
                self.feature_cache.save_features(ticker_for_cache, cache_date_key, config_hash, df_enriched_tf)
                logger.debug(f"💾 Updated cache with context_fingerprint for {ticker_for_cache} {actual_tf}")
        else:
            logger.info(f"🔄 Computing features for {ticker_for_cache} {actual_tf}...")
            df_enriched_tf = self.orchestrator.run(df_temp, **cleaned_data)
            
            # ✅ CRITICAL FIX: Ensure datetime is a column after enrichment
            if df_enriched_tf is not None and not df_enriched_tf.empty:
                if isinstance(df_enriched_tf.index, pd.DatetimeIndex) and 'datetime' not in df_enriched_tf.columns:
                    df_enriched_tf = df_enriched_tf.reset_index()
                    if 'index' in df_enriched_tf.columns:
                        df_enriched_tf = df_enriched_tf.rename(columns={'index': 'datetime'})
                    logger.debug(f"✅ Converted DatetimeIndex to datetime column after enrichment for {actual_tf}")
                
                # Save to cache for future use
                self.feature_cache.save_features(ticker_for_cache, cache_date_key, config_hash, df_enriched_tf)
                logger.debug(f"💾 Cached enriched features for {ticker_for_cache} {actual_tf}")
        
        return self._validate_context_fingerprint(df_enriched_tf)

    def _validate_context_fingerprint(self, df_enriched_tf: pd.DataFrame) -> pd.DataFrame:
        """Validate context_fingerprint - skip forcing in prepare mode since it's computed in Colab."""
        # ✅ In prepare mode, context_fingerprint is computed in Colab, not locally
        if 'context_fingerprint' not in df_enriched_tf.columns:
            logger.info("ℹ️ context_fingerprint missing - will be computed in Colab (prepare mode)")
            logger.info("⚡ Skipping ContextMapEnricher - computed in Colab for better performance")
        else:
            logger.info("✅ context_fingerprint present: {} unique".format(df_enriched_tf['context_fingerprint'].nunique()))
        
        return df_enriched_tf

    def _run_feature_leakage_gate(self, features_df: pd.DataFrame, targets_df: pd.DataFrame) -> Dict[str, Any]:
        """Run the Stage 3 leakage gate before training artifacts leave this stage."""
        if features_df.empty or targets_df.empty:
            return {"status": "skipped", "total_issues": 0}

        target_cols = [col for col in targets_df.columns if col.startswith("target_")]
        if not target_cols:
            return {"status": "skipped", "total_issues": 0}

        feature_part = features_df.reset_index(drop=True).copy()
        target_part = targets_df.reset_index(drop=True).copy()
        audit_df = pd.concat([feature_part, target_part[target_cols]], axis=1)

        tickers = ["all"]
        if "ticker" in audit_df.columns:
            tickers = audit_df["ticker"].dropna().astype(str).unique().tolist() or ["all"]

        guard = get_leakage_guard()
        reports = []
        for ticker in tickers:
            ticker_df = audit_df
            if ticker != "all" and "ticker" in audit_df.columns:
                ticker_df = audit_df[audit_df["ticker"].astype(str) == ticker]
            if ticker_df.empty:
                continue
            report = guard.check(ticker_df, target_cols=target_cols, ticker=ticker)
            reports.append(report.to_dict())

        total_issues = sum(item.get("total_issues", 0) for item in reports)
        status = "blocked" if any(item.get("status") == "blocked" for item in reports) else (
            "warning" if any(item.get("status") == "warning" for item in reports) else "clean"
        )
        return {"status": status, "total_issues": total_issues, "reports": reports}

    async def run(self, **kwargs) -> Dict[str, Any]:
        """
        Runs the streamlined feature engineering pipeline:
        1. Enrichment via FeatureOrchestrator.
        2. Target Generation.
        3. Final Feature Selection for the model.
        """
        cleaned_data, market_data_raw = self._validate_and_prepare_market_data(**kwargs)
        
        if not market_data_raw:
            return {"status": "failed", "reason": "invalid_market_data"}

        try:
            # ✅ Отримуємо batch_name з kwargs
            batch_name = kwargs.get('batch_name', 'main_database')
            
            # ✅ Читаємо runtime_params з batch-specific директорії
            runtime_params = self._load_runtime_params(batch_name)
            mode = runtime_params.get('mode', 'full')
            test_mode = runtime_params.get('test_mode', {})
            test_ticker = test_mode.get('test_ticker') or runtime_params.get('test_ticker')
            
            # ✅ Initialize guards based on pipeline mode
            if self.timeframe_guard is None:
                self._initialize_guards_based_on_mode(mode)

            enriched_prices = {}
            all_targets = {}  # ✅ Окремий dict для таргетів
            
            # Step 1: Enrich all timeframes with indicators and generate targets
            for tf, df_temp in market_data_raw.items():
                df_enriched_tf = self._process_single_timeframe(tf, df_temp, cleaned_data, test_ticker)
                if df_enriched_tf is not None:
                    # ✅ CRITICAL: Verify datetime column exists before any validation
                    if 'datetime' not in df_enriched_tf.columns:
                        logger.error(f"❌ Enriched DataFrame for {tf} missing datetime column after processing!")
                        logger.error(f"   Available columns: {df_enriched_tf.columns.tolist()[:20]}")
                        continue
                    
                    # ✅ Зберігаємо ТІЛЬКИ ФІЧІ (без таргетів)
                    enriched_prices[tf if tf != 'mixed' else '1d'] = df_enriched_tf
                    
                    # ✅ Генеруємо таргети ОКРЕМО з часовою валідацією
                    logger.info(f"Generating safe targets for {tf} timeframe...")
                    current_time = pd.Timestamp.now()
                    
                    # ✅ NEW: Use TemporalTargetGuard for safe target generation
                    targets_df = self.temporal_target_guard.generate_targets_safe(
                        df_enriched_tf, tf, current_time
                    )
                    
                    # ✅ NEW: Validate alignment between features and targets
                    # This now happens AFTER datetime is ensured to exist
                    df_enriched_tf, targets_df = validate_and_align_features_targets(
                        df_enriched_tf, targets_df, tf
                    )
                    
                    if df_enriched_tf.empty or targets_df.empty:
                        logger.error(f"❌ Alignment validation failed for {tf}. Skipping.")
                        continue
                    
                    # Update enriched_prices with aligned features
                    enriched_prices[tf if tf != 'mixed' else '1d'] = df_enriched_tf
                    all_targets[tf if tf != 'mixed' else '1d'] = targets_df
            
            if not enriched_prices:
                logger.error("No valid enriched price data generated across any timeframes.")
                return {"status": "failed", "reason": "no_enriched_prices"}
            
            # ✅ In prepare mode, skip complex validations and just return data
            if mode == "prepare":
                logger.info("📦 Prepare mode: Skipping temporal validations, returning raw data")
                
                # Prepare features DataFrame (concat all timeframes)
                features_dfs = []
                for tf, df in enriched_prices.items():
                    # Ensure datetime column exists
                    if 'datetime' not in df.columns:
                        if isinstance(df.index, pd.DatetimeIndex):
                            df = df.reset_index()
                            if 'index' in df.columns:
                                df = df.rename(columns={'index': 'datetime'})
                    features_dfs.append(df)
                
                features_df = pd.concat(features_dfs, ignore_index=True) if features_dfs else pd.DataFrame()
                logger.info(f"✅ Features combined: {features_df.shape}")
                
                # ✅ NEW: Calculate and log data quality metrics
                quality_metrics = calculate_data_quality_metrics(
                    enriched_prices, 
                    all_targets, 
                    len(self.orchestrator.enrichers) if hasattr(self.orchestrator, 'enrichers') else 0
                )
                log_data_quality_report(quality_metrics)
                
                # Prepare targets DataFrame (concat all timeframes)
                targets_dfs = []
                for tf, df in all_targets.items():
                    # Ensure datetime column exists
                    if 'datetime' not in df.columns:
                        if isinstance(df.index, pd.DatetimeIndex):
                            df = df.reset_index()
                            if 'index' in df.columns:
                                df = df.rename(columns={'index': 'datetime'})
                    
                    targets_dfs.append(df)
                    target_cols = [col for col in df.columns if col.startswith('target_')]
                    logger.info(f"✅ Extracted {len(target_cols)} targets from {tf}: {len(df)} rows")
                
                targets_df = pd.concat(targets_dfs, ignore_index=True) if targets_dfs else pd.DataFrame()
                logger.info(f"✅ Total targets: {targets_df.shape} from {len(targets_dfs)} timeframes")
                
                # Add required fields for EnrichedDataSchema
                feature_columns = [col for col in features_df.columns if col not in ['datetime', 'ticker', 'interval']]
                
                return {
                    "status": "success",
                    "enriched_prices": enriched_prices,
                    "all_targets": all_targets,
                    "combined_features": features_df,  # Simple concat for prepare mode
                    "selected_features": feature_columns,
                    "feature_importance": {},
                    "features_metadata": {
                        "total_features": len(feature_columns),
                        "timeframes": list(enriched_prices.keys()),
                        "rows": len(features_df),
                        "mode": "prepare"
                    },
                    "models_metadata": {
                        "feature_models": {
                            "feature_orchestrator": "FeatureOrchestrator",
                            "target_orchestrator": "TargetOrchestrator"
                        },
                        "version": "1.0",
                        "timestamp": pd.Timestamp.now().isoformat()
                    }
                }
            
            # ✅ Full mode: Run all validations
            # ✅ NEW: Step 1.5: Validate temporal leakage and combine features safely
            current_time = pd.Timestamp.now()
            
            # Validate temporal leakage for each timeframe
            temporal_validation_results = {}
            for tf, df in enriched_prices.items():
                logger.info(f"🔍 Checking temporal leakage for {tf} timeframe...")
                validation_result = self.temporal_leakage_guard.validate_rolling_windows(
                    df, current_time, tf
                )
                temporal_validation_results[tf] = validation_result
                
                if validation_result['status'] == 'invalid':
                    logger.error(f"❌ Temporal leakage detected in {tf}: {validation_result['issues']}")
                else:
                    logger.info(f"✅ {tf} timeframe passed temporal validation")
            
            # Safely combine features from multiple timeframes
            logger.info("🔗 Safely combining multi-timeframe features...")
            combined_features, combination_result = self.safe_combiner.combine_features_safe(
                enriched_prices, current_time
            )
            
            if combination_result['status'] == 'failed':
                logger.error(f"❌ Feature combination failed: {combination_result['issues']}")
                return {
                    "status": "failed", 
                    "reason": "feature_combination_failed",
                    "issues": combination_result['issues']
                }
            
            logger.info(f"✅ Combined features: {combined_features.shape}")
            
            # ✅ Step 2: Generate news-based dataset (if news available)
            news_df = cleaned_data.get('news')
            news_features_df = None
            
            if news_df is not None and not news_df.empty:
                logger.info(f"📰 Generating news-based dataset from {len(news_df)} news articles...")
                
                # Cluster similar news to reduce data volume (70-84% reduction)
                try:
                    news_clustered = cluster_news_simple(
                        news_df,
                        similarity_threshold=0.85,
                        text_column='title'
                    )
                    logger.info(f"✅ Clustered {len(news_df)} → {len(news_clustered)} news ({(1-len(news_clustered)/len(news_df))*100:.1f}% reduction)")
                except Exception as e:
                    logger.warning(f"News clustering failed: {e}. Using all news.")
                    news_clustered = news_df
                
                # Build news context dataset
                try:
                    # ✅ Створюємо prices_dict без таргетів для NewsDatasetBuilder
                    # ✅ FIX: Ensure datetime is a column, not index
                    prices_dict_for_news = {}
                    for tf, df in enriched_prices.items():
                        # Виключаємо колонки таргетів
                        feature_cols = [col for col in df.columns if not col.startswith('target_')]
                        df_for_news = df[feature_cols].copy()
                        
                        # ✅ CRITICAL: Ensure datetime is a column BEFORE any operations
                        if isinstance(df_for_news.index, pd.DatetimeIndex):
                            df_for_news = df_for_news.reset_index()
                            if 'index' in df_for_news.columns:
                                df_for_news = df_for_news.rename(columns={'index': 'datetime'})
                            logger.info(f"✅ Converted DatetimeIndex to datetime column for news dataset ({tf})")
                        
                        # ✅ Now check if datetime column exists
                        if 'datetime' not in df_for_news.columns:
                            if 'timestamp' in df_for_news.columns:
                                df_for_news['datetime'] = pd.to_datetime(df_for_news['timestamp'])
                                logger.info(f"✅ Converted timestamp to datetime column for news dataset ({tf})")
                            elif 'date' in df_for_news.columns:
                                df_for_news['datetime'] = pd.to_datetime(df_for_news['date'])
                                logger.info(f"✅ Converted date to datetime column for news dataset ({tf})")
                            else:
                                logger.warning(f"⚠️ No datetime column found for {tf} in news dataset preparation")
                                logger.warning(f"   Available columns: {df_for_news.columns.tolist()}")
                                # Skip this timeframe for news
                                continue
                        
                        # ✅ Verify datetime column is valid
                        if 'datetime' in df_for_news.columns:
                            logger.info(f"✅ {tf}: datetime column verified ({len(df_for_news)} rows)")
                            prices_dict_for_news[tf] = df_for_news
                        else:
                            logger.warning(f"⚠️ Skipping {tf} for news dataset - no datetime column")
                    
                    news_features_df = self.news_builder.build_dataset(
                        news_df=news_clustered,
                        prices_dict=prices_dict_for_news,
                        macro_df=cleaned_data.get('macro_data'),
                        market_sentiment_df=cleaned_data.get('market_sentiment')
                    )
                    
                    if news_features_df is not None and not news_features_df.empty:
                        logger.info(f"✅ News dataset built: {len(news_features_df)} rows, {len(news_features_df.columns)} columns")
                        
                        # Save news dataset
                        news_output_path = self.output_dir / "news_features.parquet"
                        self.news_builder.save_dataset(news_features_df, news_output_path)
                    else:
                        logger.warning("News dataset is empty")
                        
                except Exception as e:
                    logger.error(f"Failed to build news dataset: {e}", exc_info=True)
                    news_features_df = None
            else:
                logger.info("ℹ️ No news data available, skipping news-based dataset generation")
            
            # ✅ Return enriched data for pipeline
            logger.info(f"✅ Feature engineering completed for {len(enriched_prices)} timeframes")
            
            # ✅ Prepare features DataFrame (concat all timeframes)
            features_dfs = []
            for tf, df in enriched_prices.items():
                # Ensure datetime column exists
                if 'datetime' not in df.columns:
                    if isinstance(df.index, pd.DatetimeIndex):
                        df = df.reset_index()
                        if 'index' in df.columns:
                            df = df.rename(columns={'index': 'datetime'})
                    else:
                        logger.warning(f"⚠️ Timeframe {tf} has no datetime column or DatetimeIndex")
                features_dfs.append(df)
            
            features_df = pd.concat(features_dfs, ignore_index=True) if features_dfs else pd.DataFrame()
            logger.info(f"✅ Features combined: {features_df.shape}")
            
            # ✅ Prepare targets DataFrame (concat all timeframes)
            targets_dfs = []
            for tf, df in all_targets.items():
                # Ensure datetime column exists
                if 'datetime' not in df.columns:
                    if isinstance(df.index, pd.DatetimeIndex):
                        df = df.reset_index()
                        if 'index' in df.columns:
                            df = df.rename(columns={'index': 'datetime'})
                
                targets_dfs.append(df)
                target_cols = [col for col in df.columns if col.startswith('target_')]
                logger.info(f"✅ Extracted {len(target_cols)} targets from {tf}: {len(df)} rows")
            
            targets_df = pd.concat(targets_dfs, ignore_index=True) if targets_dfs else pd.DataFrame()
            logger.info(f"✅ Total targets: {targets_df.shape} from {len(targets_dfs)} timeframes")
            leakage_report = self._run_feature_leakage_gate(features_df, targets_df)
            
            # Add required fields for EnrichedDataSchema
            feature_columns = [col for col in features_df.columns if col not in ['datetime', 'ticker', 'interval']]
            
            return {
                "status": "success",
                "enriched_prices": enriched_prices,
                "all_targets": all_targets,
                "combined_features": combined_features,
                "temporal_validation": temporal_validation_results,
                "feature_combination": combination_result,
                "leakage_check": leakage_report,
                "selected_features": feature_columns,  # Required by schema
                "feature_importance": {},  # Required by schema (empty for prepare mode)
                "features_metadata": {
                    "total_features": len(feature_columns),
                    "timeframes": list(enriched_prices.keys()),
                    "rows": len(features_df),
                    "news_rows": len(news_features_df) if news_features_df is not None else 0,
                    "news_columns": len(news_features_df.columns) if news_features_df is not None else 0,
                    "leakage_status": leakage_report.get("status"),
                    "leakage_issues": leakage_report.get("total_issues"),
                },
                "models_metadata": {  # Required by pipeline validation
                    "feature_models": {
                        "feature_orchestrator": "FeatureOrchestrator",
                        "target_orchestrator": "TargetOrchestrator",
                        "context_map_enricher": "ContextMapEnricher",
                        "news_dataset_builder": "NewsContextDatasetBuilder"  # ✅ Додано
                    },
                    "version": "1.0",
                    "timestamp": pd.Timestamp.now().isoformat()
                }
            }
        
        except Exception as e:
            logger.error(f"Error in feature engineering stage: {e}")
            return {"status": "failed", "error": str(e)}
    
    def _prepare_macro_data(self, macro_data_raw: pd.DataFrame) -> pd.DataFrame:
        """Prepare macro data by converting from long to wide format if needed."""
        if macro_data_raw.empty or 'series_id' not in macro_data_raw.columns:
            return macro_data_raw
        
        logger.info("📊 Converting macro_data from long format...")
        
        # Determine date column
        date_col = self._get_date_column(macro_data_raw)
        
        # Pivot: date × series_id → columns
        macro_data = self._pivot_macro_data(macro_data_raw, date_col)
        
        # Flatten column names
        macro_data.columns = [str(col) for col in macro_data.columns]
        macro_data = macro_data.reset_index()
        logger.info(f"   ✅ Pivoted macro shape: {macro_data.shape}")
        logger.info(f"   ✅ Macro columns: {macro_data.columns.tolist()[:10]}...")
        return macro_data
    
    def _get_date_column(self, df: pd.DataFrame) -> Optional[str]:
        """Helper to find date column."""
        if 'date' in df.columns: return 'date'
        if 'datetime' in df.columns: return 'datetime'
        return None
    
    def _pivot_macro_data(self, macro_data_raw: pd.DataFrame, date_col: str) -> pd.DataFrame:
        """Pivot macro data from long to wide format."""
        try:
            macro_data = macro_data_raw.pivot_table(
                index=date_col,
                columns='series_id',
                values='value',
                aggfunc='last'
            )
            return macro_data
        except Exception as e:
            self.handle_stage_error(e, context="MacroDataPivot", severity="warning")
            logger.warning(f"   ⚠️ Failed to pivot macro_data: {e}. Using empty DataFrame.", e)
            return pd.DataFrame()

    def _create_synthetic_events(self, enriched_prices: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Create synthetic events from price data when no news is available."""
        logger.warning("⚠️ No news data available. Creating synthetic events from price data...")
        logger.warning("⚠️ This is a fallback mode - results will be less accurate without real news.")
        
        # Create synthetic events based on significant price movements
        price_1d = enriched_prices.get('1d')
        if price_1d is None or price_1d.empty:
            logger.error("❌ No price data available for synthetic events. Cannot proceed.")
            return pd.DataFrame()
        
        # Generate synthetic events based on volatility
        synthetic_events = []
        for ticker in price_1d['ticker'].unique():
            ticker_data = self._process_ticker_data(price_1d, ticker)
            if ticker_data is None:
                continue
                
            significant_moves = self._get_significant_moves(ticker_data)
            ticker_events = self._create_events_for_moves(ticker, significant_moves)
            synthetic_events.extend(ticker_events)
        
        return pd.DataFrame(synthetic_events)
    
    def _process_ticker_data(self, price_1d: pd.DataFrame, ticker: str) -> Optional[pd.DataFrame]:
        """Process ticker data for synthetic events."""
        ticker_data = price_1d[price_1d['ticker'] == ticker].copy()
        
        # Calculate daily changes
        ticker_data['price_change'] = ticker_data['close'].pct_change()
        
        return ticker_data
    
    def _get_significant_moves(self, ticker_data: pd.DataFrame) -> pd.DataFrame:
        """Get significant price movements."""
        return ticker_data[abs(ticker_data['price_change']) > 0.02].copy()
    
    def _get_datetime_column(self, df: pd.DataFrame) -> Optional[str]:
        """Get datetime column name."""
        if 'datetime' in df.columns:
            return 'datetime'
        if 'timestamp' in df.columns:
            return 'timestamp'
        return None
    
    def _create_events_for_moves(self, ticker: str, significant_moves: pd.DataFrame) -> list:
        """Create events for significant price movements."""
        events = []
        datetime_col = self._get_datetime_column(significant_moves)
        
        if datetime_col is None:
            return events
        
        for idx, row in significant_moves.iterrows():
            event = {
                'datetime': row[datetime_col],
                'ticker': ticker,
                'title': "Significant price movement: {:.2f}%".format(row['price_change']*100),
                'description': "{} moved {:.2f}% on {}".format(ticker, row['price_change']*100, row[datetime_col]),
                'sentiment': 1.0 if row['price_change'] > 0 else -1.0,
                'source': 'synthetic_price_event'
            }
            events.append(event)
        
        return events

    def _generate_targets(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Generates targets using the configured TargetOrchestrator."""
        if 'ticker' not in df.columns:
            raise ValueError("Missing 'ticker' column before target generation")
        
        # ✅ FIX: Pass news to TargetOrchestrator for post_news targets
        return self.target_orchestrator.generate_targets(df, **kwargs)
