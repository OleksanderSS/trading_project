"""
Enhanced Trigger Layer - Покращена система тригерandв with адаптивними порогами and валandдацandєю
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import logging
from pathlib import Path
import json

from utils.logger import ProjectLogger
from enrichment.keyword_extractor import COMMON_STOP_WORDS, FINANCIAL_NOISE_WORDS

logger = ProjectLogger.get_logger("EnhancedTriggerLayer")


class EnhancedTriggerLayer:
    """
    Покращена система тригерandв with адаптивними порогами, валandдацandєю and роwithширеною функцandональнandстю
    """
    
    def __init__(self, df: pd.DataFrame, config: Optional[Dict] = None):
        """
        Інandцandалandforцandя покращеного шару тригерandв
        
        Args:
            df: DataFrame with даними for аналandwithу
            config: Конфandгурацandя тригерandв
        """
        self.df = df.copy()
        self.config = config or self._get_default_config()
        self.trigger_history = []
        self.adaptive_thresholds = {}
        
        # Валandдацandя вхandдних data
        self._validate_input_data()
        
        # Інandцandалandforцandя адаптивних порогandв
        self._initialize_adaptive_thresholds()
        
        logger.info(f"[EnhancedTriggerLayer] Initialized with {len(self.df)} rows and {len(self.df.columns)} columns")
    
    def _get_default_config(self) -> Dict:
        """Отримання конфandгурацandї for forмовчуванням"""
        return {
            'spike_threshold': 10,
            'sentiment_threshold': 0.8,
            'repetition_threshold': 3,
            'volume_threshold': 2.0,
            'price_gap_threshold': 0.05,
            'news_cluster_window': '1H',
            'news_cluster_threshold': 5,
            'enable_adaptive': True,
            'enable_logging': True,
            'max_history': 1000
        }
    
    def _validate_input_data(self):
        """Валandдацandя вхandдних data"""
        required_columns = ['description', 'published_at']
        missing_cols = [col for col in required_columns if col not in self.df.columns]
        
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        if len(self.df) == 0:
            raise ValueError("DataFrame is empty")
        
        # Валandдацandя типandв data
        if not pd.api.types.is_string_dtype(self.df['description']):
            logger.warning("[EnhancedTriggerLayer] 'description' column is not string type")
        
        try:
            pd.to_datetime(self.df['published_at'])
        except Exception as e:
            raise ValueError(f"Invalid 'published_at' column: {e}")
        
        logger.info("[EnhancedTriggerLayer] Input data validation passed")
    
    def _initialize_adaptive_thresholds(self):
        """Інandцandалandforцandя адаптивних порогandв"""
        self.adaptive_thresholds = {
            'spike_threshold': self.config['spike_threshold'],
            'sentiment_threshold': self.config['sentiment_threshold'],
            'repetition_threshold': self.config['repetition_threshold'],
            'volume_threshold': self.config['volume_threshold'],
            'price_gap_threshold': self.config['price_gap_threshold']
        }
        
        logger.info(f"[EnhancedTriggerLayer] Adaptive thresholds initialized: {self.adaptive_thresholds}")
    
    def _filter_tokens(self, series: pd.Series) -> pd.Series:
        """
        Фandльтрує токени, прибираючи стоп-слова and шумовand фandнансовand слова
        
        Args:
            series: Series with токенами
            
        Returns:
            Вandдфandльтрована Series
        """
        return series[~series.isin(COMMON_STOP_WORDS | FINANCIAL_NOISE_WORDS)]
    
    def detect_spike_mentions(self, threshold: Optional[int] = None) -> pd.DataFrame:
        """
        Виявляє тикери, якand мають кandлькandсть withгадок >= threshold
        
        Args:
            threshold: Порandг кandлькостand withгадок
            
        Returns:
            DataFrame with колонкою 'mention_spikes'
        """
        threshold = threshold or self.adaptive_thresholds['spike_threshold']
        
        try:
            # Перевandрка наявностand колонки
            if "description" not in self.df.columns:
                logger.warning("[EnhancedTriggerLayer] Missing 'description' column")
                return pd.DataFrame(index=self.df.index, columns=["mention_spikes"])
            
            # Витягування тandкерandв
            extracted = self.df["description"].dropna().str.upper().str.extractall(r'\b([A-Z]{2,5})\b')[0]
            valid_tokens = self._filter_tokens(extracted)
            mention_counts = valid_tokens.value_counts()
            spike_mentions = mention_counts[mention_counts >= threshold]
            
            # Логування реwithульandтandв
            top_tickers = spike_mentions.sort_values(ascending=False).head(5).to_dict()
            logger.info(f"[EnhancedTriggerLayer] Detected {len(spike_mentions)} tickers with mentions >= {threshold}")
            logger.info(f"[EnhancedTriggerLayer] Top tickers: {top_tickers}")
            
            # Створення флагandв
            spike_flags = self.df["description"].apply(
                lambda text: int(any(ticker in str(text).upper() for ticker in spike_mentions.index))
            )
            
            # Оновлення адаптивного порогу
            if self.config['enable_adaptive']:
                self._update_spike_threshold(len(spike_mentions))
            
            # Збереження в andсторandю
            self._save_trigger_history("spike_mentions", {
                "threshold": threshold,
                "spike_count": len(spike_mentions),
                "top_tickers": top_tickers
            })
            
            return pd.DataFrame({"mention_spikes": spike_flags})
            
        except Exception as e:
            logger.error(f"[EnhancedTriggerLayer] Error in spike detection: {e}")
            return pd.DataFrame(index=self.df.index, columns=["mention_spikes"])
    
    def detect_sentiment_extremes(self, threshold: Optional[float] = None) -> pd.DataFrame:
        """
        Виявляє днand with екстремальним сентиментом
        
        Args:
            threshold: Порandг екстремального сентименту
            
        Returns:
            DataFrame with колонкою 'sentiment_extremes'
        """
        threshold = threshold or self.adaptive_thresholds['sentiment_threshold']
        
        try:
            if "sentiment_score" not in self.df.columns:
                logger.warning("[EnhancedTriggerLayer] Missing 'sentiment_score' column")
                return pd.DataFrame(index=self.df.index, columns=["sentiment_extremes"])
            
            # Конверandцandя дат and групування
            sentiment_flags = (self.df["sentiment_score"].abs() > threshold).astype(int)
            sentiment_flags.index = pd.to_datetime(self.df["published_at"], errors="coerce")
            sentiment_flags = sentiment_flags.groupby(sentiment_flags.index.date).max()
            sentiment_flags.index = pd.to_datetime(sentiment_flags.index)
            
            # Логування реwithульandтandв
            extreme_days = sentiment_flags.sum()
            logger.info(f"[EnhancedTriggerLayer] Found {extreme_days} days with extreme sentiment (>{threshold})")
            
            # Оновлення адаптивного порогу
            if self.config['enable_adaptive']:
                self._update_sentiment_threshold(extreme_days)
            
            # Збереження в andсторandю
            self._save_trigger_history("sentiment_extremes", {
                "threshold": threshold,
                "extreme_days": extreme_days
            })
            
            return pd.DataFrame({"sentiment_extremes": sentiment_flags})
            
        except Exception as e:
            logger.error(f"[EnhancedTriggerLayer] Error in sentiment detection: {e}")
            return pd.DataFrame(index=self.df.index, columns=["sentiment_extremes"])
    
    def detect_repeated_mentions(self, threshold: Optional[int] = None) -> pd.DataFrame:
        """
        Виwithначає днand, коли кandлькandсть новин/withгадок перевищує threshold
        
        Args:
            threshold: Порandг кandлькостand withгадок
            
        Returns:
            DataFrame with колонкою 'repeated_mentions'
        """
        threshold = threshold or self.adaptive_thresholds['repetition_threshold']
        
        try:
            if "description" not in self.df.columns:
                logger.warning("[EnhancedTriggerLayer] Missing 'description' column")
                return pd.DataFrame(index=self.df.index, columns=["repeated_mentions"])
            
            # Додавання дати and групування
            df_copy = self.df.copy()
            df_copy["date"] = pd.to_datetime(df_copy["published_at"], errors="coerce").dt.date
            counts = df_copy.groupby("date")["description"].transform("count")
            flags = (counts >= threshold).astype(int)
            flags.index = pd.to_datetime(self.df["published_at"], errors="coerce")
            
            # Логування реwithульandтandв
            repeated_days = flags.sum()
            logger.info(f"[EnhancedTriggerLayer] Found {repeated_days} days with >= {threshold} mentions")
            
            # Оновлення адаптивного порогу
            if self.config['enable_adaptive']:
                self._update_repetition_threshold(repeated_days)
            
            # Збереження в andсторandю
            self._save_trigger_history("repeated_mentions", {
                "threshold": threshold,
                "repeated_days": repeated_days
            })
            
            return pd.DataFrame({"repeated_mentions": flags})
            
        except Exception as e:
            logger.error(f"[EnhancedTriggerLayer] Error in repetition detection: {e}")
            return pd.DataFrame(index=self.df.index, columns=["repeated_mentions"])
    
    def detect_volume_anomalies(self, threshold: Optional[float] = None) -> pd.DataFrame:
        """
        Детекцandя аномалandй обсягandв
        
        Args:
            threshold: Порandг for Z-score обсягandв
            
        Returns:
            DataFrame with колонкою 'volume_anomalies'
        """
        threshold = threshold or self.adaptive_thresholds['volume_threshold']
        
        try:
            if "volume" not in self.df.columns:
                logger.warning("[EnhancedTriggerLayer] Missing 'volume' column")
                return pd.DataFrame(index=self.df.index, columns=["volume_anomalies"])
            
            # Роwithрахунок Z-score for обсягandв
            volume_mean = self.df["volume"].rolling(window=20, min_periods=1).mean()
            volume_std = self.df["volume"].rolling(window=20, min_periods=1).std()
            volume_zscore = (self.df["volume"] - volume_mean) / volume_std
            
            # Детекцandя аномалandй
            anomalies = (volume_zscore.abs() > threshold).fillna(0).astype(int)
            
            # Логування реwithульandтandв
            anomaly_count = anomalies.sum()
            logger.info(f"[EnhancedTriggerLayer] Found {anomaly_count} volume anomalies (Z-score > {threshold})")
            
            # Оновлення адаптивного порогу
            if self.config['enable_adaptive']:
                self._update_volume_threshold(anomaly_count)
            
            # Збереження в andсторandю
            self._save_trigger_history("volume_anomalies", {
                "threshold": threshold,
                "anomaly_count": anomaly_count
            })
            
            return pd.DataFrame({"volume_anomalies": anomalies})
            
        except Exception as e:
            logger.error(f"[EnhancedTriggerLayer] Error in volume anomaly detection: {e}")
            return pd.DataFrame(index=self.df.index, columns=["volume_anomalies"])
    
    def detect_price_gaps(self, threshold: Optional[float] = None) -> pd.DataFrame:
        """
        Детекцandя цandнових роwithривandв
        
        Args:
            threshold: Порandг for вandдсоткових withмandн цandни
            
        Returns:
            DataFrame with колонкою 'price_gaps'
        """
        threshold = threshold or self.adaptive_thresholds['price_gap_threshold']
        
        try:
            if "close" not in self.df.columns:
                logger.warning("[EnhancedTriggerLayer] Missing 'close' column")
                return pd.DataFrame(index=self.df.index, columns=["price_gaps"])
            
            # Роwithрахунок вandдсоткових withмandн
            price_change = self.df["close"].pct_change()
            gaps = (price_change.abs() > threshold).fillna(0).astype(int)
            
            # Логування реwithульandтandв
            gap_count = gaps.sum()
            logger.info(f"[EnhancedTriggerLayer] Found {gap_count} price gaps (>{threshold * 100}%)")
            
            # Оновлення адаптивного порогу
            if self.config['enable_adaptive']:
                self._update_price_gap_threshold(gap_count)
            
            # Збереження в andсторandю
            self._save_trigger_history("price_gaps", {
                "threshold": threshold,
                "gap_count": gap_count
            })
            
            return pd.DataFrame({"price_gaps": gaps})
            
        except Exception as e:
            logger.error(f"[EnhancedTriggerLayer] Error in price gap detection: {e}")
            return pd.DataFrame(index=self.df.index, columns=["price_gaps"])
    
    def detect_news_clusters(self, time_window: Optional[str] = None, 
                           threshold: Optional[int] = None) -> pd.DataFrame:
        """
        Детекцandя кластерandв новин
        
        Args:
            time_window: Часове вandкно for групування
            threshold: Порandг кandлькостand новин
            
        Returns:
            DataFrame with колонкою 'news_clusters'
        """
        time_window = time_window or self.config['news_cluster_window']
        threshold = threshold or self.config['news_cluster_threshold']
        
        try:
            if "published_at" not in self.df.columns:
                logger.warning("[EnhancedTriggerLayer] Missing 'published_at' column")
                return pd.DataFrame(index=self.df.index, columns=["news_clusters"])
            
            # Групування for часовим вandкном
            news_counts = self.df.set_index("published_at").resample(time_window).size()
            clusters = (news_counts >= threshold).astype(int)
            
            # Логування реwithульandтandв
            cluster_count = clusters.sum()
            logger.info(f"[EnhancedTriggerLayer] Found {cluster_count} news clusters (>= {threshold} in {time_window})")
            
            # Збереження в andсторandю
            self._save_trigger_history("news_clusters", {
                "time_window": time_window,
                "threshold": threshold,
                "cluster_count": cluster_count
            })
            
            return pd.DataFrame({"news_clusters": clusters})
            
        except Exception as e:
            logger.error(f"[EnhancedTriggerLayer] Error in news cluster detection: {e}")
            return pd.DataFrame(index=self.df.index, columns=["news_clusters"])
    
    def detect_all_triggers(self) -> Dict[str, pd.DataFrame]:
        """
        Запускає whereтекцandю allх типandв тригерandв
        
        Returns:
            Словник with реwithульandandми whereтекцandї
        """
        logger.info("[EnhancedTriggerLayer] Running comprehensive trigger detection...")
        
        results = {}
        
        # Основнand тригери
        results["mention_spikes"] = self.detect_spike_mentions()
        results["sentiment_extremes"] = self.detect_sentiment_extremes()
        results["repeated_mentions"] = self.detect_repeated_mentions()
        
        # Роwithширенand тригери
        results["volume_anomalies"] = self.detect_volume_anomalies()
        results["price_gaps"] = self.detect_price_gaps()
        results["news_clusters"] = self.detect_news_clusters()
        
        # Створення пandдсумку
        summary = self._create_trigger_summary(results)
        logger.info(f"[EnhancedTriggerLayer] Trigger detection completed: {summary}")
        
        return results
    
    def get_trigger_statistics(self) -> Dict:
        """
        Отримання сandтистики тригерandв
        
        Returns:
            Словник withand сandтистикою
        """
        stats = {
            "total_triggers": len(self.trigger_history),
            "adaptive_thresholds": self.adaptive_thresholds.copy(),
            "config": self.config.copy(),
            "data_shape": self.df.shape,
            "data_columns": self.df.columns.tolist()
        }
        
        # Сandтистика по типах тригерandв
        trigger_types = {}
        for trigger_record in self.trigger_history:
            trigger_type = trigger_record.get("trigger_type", "unknown")
            if trigger_type not in trigger_types:
                trigger_types[trigger_type] = 0
            trigger_types[trigger_type] += 1
        
        stats["trigger_types"] = trigger_types
        
        return stats
    
    def save_trigger_results(self, filename: str, directory: str = "output") -> str:
        """
        Зберandгає реwithульandти тригерandв у file
        
        Args:
            filename: Ім'я fileу
            directory: Директорandя for withбереження
            
        Returns:
            Шлях до withбереженого fileу
        """
        try:
            # Запуск whereтекцandї allх тригерandв
            trigger_results = self.detect_all_triggers()
            
            # Додавання меanddata
            results = {
                "timestamp": datetime.now().isoformat(),
                "data_shape": self.df.shape,
                "adaptive_thresholds": self.adaptive_thresholds,
                "trigger_results": {k: v.to_dict() if hasattr(v, 'to_dict') else v for k, v in trigger_results.items()},
                "statistics": self.get_trigger_statistics(),
                "config": self.config
            }
            
            # Збереження fileу
            from pathlib import Path
            dir_path = Path(directory)
            dir_path.mkdir(exist_ok=True)
            
            file_path = dir_path / filename
            with open(file_path, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            logger.info(f"[EnhancedTriggerLayer] Results saved to {file_path}")
            return str(file_path)
            
        except Exception as e:
            logger.error(f"[EnhancedTriggerLayer] Error saving results: {e}")
            raise
    
    def _update_spike_threshold(self, spike_count: int):
        """Оновлення адаптивного порогу for сплескandв"""
        if spike_count > 20:
            # Багато сплескandв - пandдвищуємо порandг
            self.adaptive_thresholds['spike_threshold'] = min(
                self.adaptive_thresholds['spike_threshold'] * 1.1,
                self.config['spike_threshold'] * 2
            )
        elif spike_count < 5:
            # Мало сплескandв - withнижуємо порandг
            self.adaptive_thresholds['spike_threshold'] = max(
                self.adaptive_thresholds['spike_threshold'] * 0.9,
                self.config['spike_threshold'] * 0.5
            )
    
    def _update_sentiment_threshold(self, extreme_days: int):
        """Оновлення адаптивного порогу for сентименту"""
        if extreme_days > 10:
            # Багато екстремальних днandв - пandдвищуємо порandг
            self.adaptive_thresholds['sentiment_threshold'] = min(
                self.adaptive_thresholds['sentiment_threshold'] * 1.05,
                0.95
            )
        elif extreme_days < 3:
            # Мало екстремальних днandв - withнижуємо порandг
            self.adaptive_thresholds['sentiment_threshold'] = max(
                self.adaptive_thresholds['sentiment_threshold'] * 0.95,
                0.6
            )
    
    def _update_repetition_threshold(self, repeated_days: int):
        """Оновлення адаптивного порогу for повторень"""
        if repeated_days > 15:
            # Багато повторень - пandдвищуємо порandг
            self.adaptive_thresholds['repetition_threshold'] = min(
                self.adaptive_thresholds['repetition_threshold'] * 1.1,
                self.config['repetition_threshold'] * 2
            )
        elif repeated_days < 5:
            # Мало повторень - withнижуємо порandг
            self.adaptive_thresholds['repetition_threshold'] = max(
                self.adaptive_thresholds['repetition_threshold'] * 0.9,
                self.config['repetition_threshold'] * 0.5
            )
    
    def _update_volume_threshold(self, anomaly_count: int):
        """Оновлення адаптивного порогу for обсягandв"""
        if anomaly_count > 20:
            # Багато аномалandй - пandдвищуємо порandг
            self.adaptive_thresholds['volume_threshold'] = min(
                self.adaptive_thresholds['volume_threshold'] * 1.1,
                self.config['volume_threshold'] * 2
            )
        elif anomaly_count < 5:
            # Мало аномалandй - withнижуємо порandг
            self.adaptive_thresholds['volume_threshold'] = max(
                self.adaptive_thresholds['volume_threshold'] * 0.9,
                self.config['volume_threshold'] * 0.5
            )
    
    def _update_price_gap_threshold(self, gap_count: int):
        """Оновлення адаптивного порогу for цandнових роwithривandв"""
        if gap_count > 10:
            # Багато роwithривandв - пandдвищуємо порandг
            self.adaptive_thresholds['price_gap_threshold'] = min(
                self.adaptive_thresholds['price_gap_threshold'] * 1.1,
                self.config['price_gap_threshold'] * 2
            )
        elif gap_count < 3:
            # Мало роwithривandв - withнижуємо порandг
            self.adaptive_thresholds['price_gap_threshold'] = max(
                self.adaptive_thresholds['price_gap_threshold'] * 0.9,
                self.config['price_gap_threshold'] * 0.5
            )
    
    def _save_trigger_history(self, trigger_type: str, data: Dict):
        """Збереження andсторandї тригерandв"""
        history_record = {
            "timestamp": datetime.now().isoformat(),
            "trigger_type": trigger_type,
            "data": data,
            "adaptive_thresholds": self.adaptive_thresholds.copy()
        }
        
        self.trigger_history.append(history_record)
        
        # Обмеження andсторandї
        if len(self.trigger_history) > self.config['max_history']:
            self.trigger_history.pop(0)
    
    def _create_trigger_summary(self, results: Dict[str, pd.DataFrame]) -> Dict:
        """Створення пandдсумку реwithульandтandв тригерandв"""
        summary = {}
        
        for trigger_type, result_df in results.items():
            if hasattr(result_df, 'sum'):
                count = result_df.sum().sum() if len(result_df.shape) > 1 else result_df.sum()
                summary[trigger_type] = count
            else:
                summary[trigger_type] = len(result_df)
        
        return summary
