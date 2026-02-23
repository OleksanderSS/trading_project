# pipeline/pipeline_news_aggregator.py

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
from utils.features import FeatureEngineer
from utils.features_utils import FeatureUtils
from utils.logger import ProjectLogger

from collectors.news_collector import NewsCollector
from collectors.gdelt_collector import GDELTCollector
from collectors.newsapi_collector import NewsAPICollector
from collectors.hf_collector import HFCollector
from modules import TrendParser, ContextParser, SignalAnalyzer, ReactionEvaluator


logger = ProjectLogger.get_logger("PipelineNewsAggregator")


class PipelineNewsAggregator:
    """
    Агрегатор новин for ML-моwhereлand по andймфреймах.
    Пandдтримує NewsCollector, GDELT, NewsAPI, Hugging Face.
    """

    def __init__(
        self,
        news_collectors: Optional[List] = None,
        feature_engineer: Optional[FeatureEngineer] = None
    ):
        self.news_collectors = news_collectors if news_collectors else []
        self.feature_engineer = feature_engineer or FeatureEngineer()
        self.feature_utils = FeatureUtils(short_window=14, long_window=200)

    def build(self, forecast_date: pd.Timestamp, max_features: int = 20) -> Dict[str, pd.DataFrame]:
        start_date = forecast_date - pd.DateOffset(years=1)
        df_list = []

        # --- Збandр новин with усandх колекторandв ---
        for collector in self.news_collectors:
            try:
                if isinstance(collector, NewsCollector):
                    df_news = collector.fetch(start_date=start_date, end_date=forecast_date)
                elif isinstance(collector, GDELTCollector):
                    df_news = collector.fetch_gkg(start_date=start_date, end_date=forecast_date)
                elif isinstance(collector, NewsAPICollector):
                    df_news = collector.fetch(query="default_query", start_date=start_date, end_date=forecast_date)
                elif isinstance(collector, HFCollector):
                    df_news = collector.fetch(model_name="default_model", query="default_query")
                else:
                    continue

                if not df_news.empty:
                    df_list.append(df_news)
            except Exception as e:
                logger.error(f"[PipelineNewsAggregator] Error fetching from {collector}: {e}")

        if not df_list:
            logger.warning("[PipelineNewsAggregator] Новин for рandк not withнайwhereно")
            return {}

        # --- Merging новин ---
        df_news_all = pd.concat(df_list, ignore_index=True)
        if 'datetime' in df_news_all.columns:
            df_news_all.rename(columns={'datetime': 'published_at'}, inplace=True)
        elif 'published_at' not in df_news_all.columns:
            df_news_all['published_at'] = pd.to_datetime('now')

        # --- Рandчний фон ---
        df_year = self._aggregate(df_news_all, 'D')
        forecast_date = pd.to_datetime(forecast_date)
        df_year.index = pd.to_datetime(df_year.index)
        df_year = df_year[df_year.index <= forecast_date]
        df_year = self._sanitize_features(df_year)
        df_year_feat = self.feature_utils.transform(df_year, key='year')
        df_year_feat = self._select_features(df_year_feat, max_features)

        # --- Мandсячний фон ---
        df_month_all = self._process_period(df_news_all, df_year_feat, freq='M',
                                            tf_key_prefix='month', max_features=max_features)

        # --- Тижnotвий фон ---
        df_week_all = self._process_period(df_news_all, df_month_all, freq='W',
                                           tf_key_prefix='week', extra_resample=['H', '15T'], max_features=max_features)

        # --- Денний фон ---
        df_day_all = self._process_period(df_news_all, df_week_all, freq='D',
                                          tf_key_prefix='day', extra_resample=['H', '15T'], max_features=max_features)

        # Логування реwithульandтandв
        logger.info(
            f"[PipelineNewsAggregator] [OK] Year={df_year_feat.shape}, "
            f"Month={df_month_all.shape}, Week={df_week_all.shape}, Day={df_day_all.shape}"
        )

        # Фandнальна валandдацandя
        for k, v in {"year": df_year_feat, "month": df_month_all, "week": df_week_all, "day": df_day_all}.items():
            if v.empty:
                logger.warning(f"[PipelineNewsAggregator] [WARN] {k} DataFrame порожнandй")

        return {
            'year': df_year_feat,
            'month': df_month_all,
            'week': df_week_all,
            'day': df_day_all
        }

    def analyze_layers(self, price_df: pd.DataFrame, news_df: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        Аналandwith багатошарової структури: тренд  контекст  сигнал  реакцandя.
        Поверandє словник with реwithульandandми кожного модуля.
        """
        results: Dict[str, Any] = {}
        try:
            trend = TrendParser().classify(price_df)
            context = ContextParser().classify(price_df)
            signal = SignalAnalyzer().analyze(price_df, news_df)
            reaction = ReactionEvaluator().evaluate(price_df, news_df)
            results = {
                "trend": trend,
                "context": context,
                "signal": signal,
                "reaction": reaction,
            }
            logger.info(
                f"[PipelineNewsAggregator]  Layers: trend={trend.label}, context={context.label}, "
                f"signal={signal.category} ({signal.trade}), reaction={reaction.label}"
            )
        except Exception as e:
            logger.error(f"[PipelineNewsAggregator] Error in analyze_layers: {e}")
        return results

    # -------------------- Внутрandшнand методи --------------------
    def _aggregate(self, df: pd.DataFrame, time_unit: str = 'D') -> pd.DataFrame:
        df_agg = df.copy().set_index('published_at').sort_index()

        # [PROTECT] Перевandрка ключових колонок
        required_cols = ["sentiment", "value", "description"]
        for col in required_cols:
            if col not in df_agg.columns:
                df_agg[col] = 0.0

        agg = df_agg.resample(time_unit).agg({
            'sentiment': 'mean',
            'value': 'mean',
            'description': 'count'
        }).rename(columns={'description': 'news_count'})
        return agg

    def _process_period(
        self,
        df_news: pd.DataFrame,
        base_df: pd.DataFrame,
        freq: str,
        tf_key_prefix: str,
        extra_resample: list = None,
        max_features: int = 20
    ) -> pd.DataFrame:
        if extra_resample is None:
            extra_resample = []

        df_list = []
        for period_start, period_group in df_news.groupby(pd.Grouper(key='published_at', freq=freq)):
            period_group['published_at'] = pd.to_datetime(period_group['published_at'])
            base_df.index = pd.to_datetime(base_df.index)
            period_group = period_group[period_group['published_at'] <= base_df.index.max()]
            if period_group.empty:
                continue

            df_day = self._aggregate(period_group, 'D')
            df_day = self._sanitize_features(df_day)

            df_resampled_dict = {}
            for r in extra_resample:
                df_resampled_dict[r] = self._resample_with_background(df_day, base_df, freq=r)

            df_feat_list = []
            for r, df_r in df_resampled_dict.items():
                df_feat = self.feature_utils.transform(df_r, key=f"{tf_key_prefix}_{r}_{period_start.date()}")
                df_feat = self._select_features(df_feat, max_features)
                df_feat_list.append(df_feat)

            if not df_feat_list:
                df_feat_list.append(
                    self._select_features(
                        self.feature_utils.transform(df_day, key=f"{tf_key_prefix}_{period_start.date()}"),
                        max_features
                    )
                )

            df_combined = pd.concat(df_feat_list, axis=1)
            df_list.append(df_combined)

        return pd.concat(df_list).sort_index() if df_list else pd.DataFrame()

    def _resample_with_background(self, df: pd.DataFrame, base: pd.DataFrame, freq: str) -> pd.DataFrame:
        if df.empty:
            return df.copy()
        df_resampled = df.resample(freq).agg({
            'sentiment': 'mean',
            'value': 'mean',
            'news_count': 'sum'
        }).ffill()
        df_combined = self.feature_engineer._merge_with_base(df_resampled, base, tf=f"bg_{freq}", ffill_limit=3)
        return df_combined

    # -------------------- Гнучка логandка фandч --------------------
    def _sanitize_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df = df.fillna(0.0)
        df.replace([np.inf, -np.inf], 0.0, inplace=True)
        return df

    def _select_features(self, df: pd.DataFrame, max_features: int = 20) -> pd.DataFrame:
        if df.empty:
            return df
        k = min(max_features, df.shape[1])
        if k >= df.shape[1]:
            return df
        return df.iloc[:, :k]