import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Any, Optional
from .base import BaseEnricher
from src.core.logging.logger import ProjectLogger

logger = ProjectLogger.get_logger("ContextMapEnricher")

class ContextMapEnricher(BaseEnricher):
    """
    Generates a 'Context Fingerprint' (Market State) based on signal changes.
    Loads noise filter thresholds from external YAML config.
    """
    
    @property
    def name(self) -> str:
        return "context_map"
    
    @property
    def priority(self) -> int:
        return 80

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        
        # ✅ ЗАВАНТАЖУЄМО NOISE FILTER THRESHOLDS З КОНФІГУ
        self.noise_filter_thresholds = {}
        self.temporal_features = set()
        self.default_dynamic_threshold = 0.005
        self.noise_sensitivity = 1.5
        
        # Спробуємо завантажити з noise_filter_config.yaml
        import yaml
        from pathlib import Path
        
        config_path = Path(__file__).parent.parent.parent / "config" / "noise_filter_config.yaml"
        
        try:
            if config_path.exists():
                with open(config_path, 'r', encoding='utf-8') as f:
                    noise_config = yaml.safe_load(f)
                    self.noise_filter_thresholds = noise_config.get('noise_filter_thresholds', {})
                    self.temporal_features = set(noise_config.get('temporal_features', []))
                    self.default_dynamic_threshold = noise_config.get('default_dynamic_threshold', 0.005)
                    self.noise_sensitivity = noise_config.get('noise_sensitivity', 1.5)
                    logger.info(f"✅ Loaded {len(self.noise_filter_thresholds)} noise thresholds from {config_path}")
            else:
                logger.warning(f"⚠️ Noise filter config not found: {config_path}. Using defaults.")
                self._load_defaults()
        except Exception as e:
            logger.error(f"❌ Failed to load noise config from {config_path}: {e}. Using defaults.")
            self._load_defaults()

        logger.info(f"ContextMapEnricher initialized with {len(self.noise_filter_thresholds)} noise thresholds")
        logger.info(f"Temporal features (not compared): {len(self.temporal_features)} features")
    
    def _load_defaults(self):
        """Завантажує дефолтні пороги якщо конфіг не знайдено."""
        self.noise_filter_thresholds = {
            'VIX': 0.02, '10Y_yield': 0.001, 'DXY': 0.003, 'SPY': 0.005,
            'RSI': 0.05, 'MACD': 0.01, 'BB_width': 0.02, 'ATR': 0.05,
            'volume': 0.1, 'close': 0.005, 'open': 0.005, 'high': 0.005, 'low': 0.005,
        }
        self.temporal_features = {
            'hour', 'day_of_week', 'day_of_month', 'day_of_year',
            'week_of_year', 'month_of_year', 'quarter', 'is_weekend'
        }
        logger.info("Loaded default noise thresholds")

    def enrich(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Generates a contextual fingerprint."""
        if df.empty:
            return df

        res_df = df.copy()
        
        # ✅ ВИКОРИСТОВУЄМО ВСІ ЧИСЛОВІ ПОКАЗНИКИ (без вибору sub_features)
        context_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Виключаємо таргети та службові колонки
        context_columns = [c for c in context_columns if not c.startswith('target_') 
                          and c not in ['hash', 'interval']]
        
        if not context_columns:
            logger.warning("No numeric columns found for context map. Skipping.")
            return df

        logger.info(f"Generating context map from {len(context_columns)} indicators")

        state_cols = []
        temporal_cols = []
        
        for col in context_columns:
            state_col_name = f"state_{col}"
            if col not in res_df.columns:
                logger.debug(f"Column '{col}' not found. Skipping.")
                continue

            # ✅ ЧАСОВІ ПОКАЗНИКИ - просто нормалізуємо (НЕ порівнюємо)
            if col in self.temporal_features:
                res_df[state_col_name] = res_df[col]
                temporal_cols.append(state_col_name)
                continue

            # ✅ ЧИСЛОВІ ПОКАЗНИКИ - порівнюємо з попереднім значенням
            threshold = self._get_threshold(res_df, col)
            prev_val = res_df[col].shift(1)
            change = (res_df[col] - prev_val) / prev_val.replace(0, np.nan)
            change = change.fillna(0)

            # Три стани: -1 (падіння), 0 (без змін), 1 (зростання)
            res_df[state_col_name] = np.where(change > threshold, 1,
                                        np.where(change < -threshold, -1, 0))
            state_cols.append(state_col_name)

        # Generate fingerprint and stability score
        all_state_cols = state_cols + temporal_cols
        
        if all_state_cols:
            # Fingerprint: об'єднуємо всі стани через '|'
            res_df['context_fingerprint'] = res_df[all_state_cols].astype(str).agg('|'.join, axis=1)
            
            # Stability: скільки показників БЕЗ ЗМІН (тільки для числових, не часових)
            if state_cols:
                zero_counts = (res_df[state_cols] == 0).sum(axis=1)
                res_df['context_stability'] = zero_counts / len(state_cols)
            else:
                res_df['context_stability'] = 1.0
            
            # ✅ ПРОЗОРІ ЛОГИ (Статистика станів ринку)
            if len(res_df) > 0:
                last_idx = res_df.index[-1]
                if state_cols:
                    latest_row = res_df[state_cols].iloc[-1]
                    up_count = (latest_row == 1).sum()
                    down_count = (latest_row == -1).sum()
                    flat_count = (latest_row == 0).sum()
                    
                    logger.info(f"📊 Market State at {last_idx}: UP={up_count}, DOWN={down_count}, FLAT={flat_count}")
                    logger.info(f"📊 Temporal features: {len(temporal_cols)}")
                    logger.info(f"📜 Fingerprint sample: {res_df['context_fingerprint'].iloc[-1][:100]}...")
                
            logger.info(f"✅ Context map: {len(state_cols)} numeric + {len(temporal_cols)} temporal = {len(all_state_cols)} total states")
        else:
            logger.warning("No state columns were processed for the context map.")

        return res_df

    def _get_threshold(self, df: pd.DataFrame, col: str) -> float:
        """
        Визначає поріг шуму для показника.
        
        1. Використовує noise_filter_thresholds якщо є
        2. Шукає часткове співпадіння (наприклад 'AMD_close' → 'close')
        3. Інакше розраховує динамічний поріг на основі IQR
        """
        # Пряме співпадіння
        if col in self.noise_filter_thresholds:
            return self.noise_filter_thresholds[col]
        
        # Часткове співпадіння (наприклад 'AMD_close' містить 'close')
        for key, threshold in self.noise_filter_thresholds.items():
            if key in col:
                return threshold
        
        # Динамічний поріг на основі волатильності (IQR)
        changes = df[col].diff().abs().dropna()
        if not changes.empty and len(changes) > 10:
            q1, q3 = changes.quantile(0.25), changes.quantile(0.75)
            iqr = q3 - q1
            if iqr > 0:
                dynamic_threshold = max(iqr * self.noise_sensitivity, 1e-7)
                logger.debug(f"Dynamic threshold for {col}: {dynamic_threshold:.6f} (IQR={iqr:.6f})")
                return dynamic_threshold
        
        # Fallback
        return self.default_dynamic_threshold
