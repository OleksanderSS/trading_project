# triggers/triggers_layer.py

import pandas as pd
from utils.logger import ProjectLogger
from enrichment.keyword_extractor import COMMON_STOP_WORDS, FINANCIAL_NOISE_WORDS


logger = ProjectLogger.get_logger("TradingProjectLogger")


class TriggerLayer:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

    def _filter_tokens(self, series: pd.Series) -> pd.Series:
        """Фandльтрує токени, прибираючи стоп-слова and шумовand фandнансовand слова."""
        return series[~series.isin(COMMON_STOP_WORDS | FINANCIAL_NOISE_WORDS)]

    def detect_spike_mentions(self, threshold: int = 10) -> pd.DataFrame:
        """
        Виявляє тикери, якand мають кandлькandсть withгадок >= threshold.
        Поверandє DataFrame with колонкою 'mention_spikes'.
        """
        if "description" not in self.df.columns:
            logger.warning("[TriggerLayer] Вandдсутня колонка 'description'  spike_mentions пропущено")
            return pd.DataFrame(index=self.df.index, columns=["mention_spikes"])

        extracted = self.df["description"].dropna().str.upper().str.extractall(r'\b([A-Z]{2,5})\b')[0]
        valid_tokens = self._filter_tokens(extracted)
        mention_counts = valid_tokens.value_counts()
        spike_mentions = mention_counts[mention_counts >= threshold]

        top_tickers = spike_mentions.sort_values(ascending=False).head(5).to_dict()
        logger.info(f"[TriggerLayer] Виявлено {len(spike_mentions)} тикерandв with mentions  {threshold}. "
                    f"Топтикери: {top_tickers}")

        self.df["mention_spikes"] = self.df["description"].apply(
            lambda text: int(any(ticker in str(text).upper() for ticker in spike_mentions.index))
        )
        return self.df[["mention_spikes"]]

    def detect_sentiment_extremes(self, threshold: float = 0.8) -> pd.DataFrame:
        """
        Виявляє днand with екстремальним сентиментом (>|threshold|).
        Поверandє DataFrame with колонкою 'sentiment_extremes'.
        """
        if "sentiment_score" not in self.df.columns:
            logger.warning("[TriggerLayer] Вandдсутня колонка 'sentiment_score'  sentiment_extremes пропущено")
            return pd.DataFrame(index=self.df.index, columns=["sentiment_extremes"])

        sentiment_flags = (self.df["sentiment_score"].abs() > threshold).astype(int)
        sentiment_flags.index = pd.to_datetime(self.df["published_at"], errors="coerce")
        sentiment_flags = sentiment_flags.groupby(sentiment_flags.index.date).max()
        sentiment_flags.index = pd.to_datetime(sentiment_flags.index)

        logger.info(f"[TriggerLayer] sentiment_extremes: {sentiment_flags.sum()} днandв with екстремальним сентиментом (>{threshold})")
        return pd.DataFrame({"sentiment_extremes": sentiment_flags})

    def detect_repeated_mentions(self, threshold: int = 3) -> pd.DataFrame:
        """
        Виwithначає днand, коли кandлькandсть новин/withгадок перевищує threshold.
        Поверandє DataFrame with колонкою 'repeated_mentions'.
        """
        if "description" not in self.df.columns:
            logger.warning("[TriggerLayer] Вandдсутня колонка 'description'  repeated_mentions пропущено")
            return pd.DataFrame(index=self.df.index, columns=["repeated_mentions"])

        self.df["date"] = pd.to_datetime(self.df["published_at"], errors="coerce").dt.date
        counts = self.df.groupby("date")["description"].transform("count")
        flags = (counts >= threshold).astype(int)
        flags.index = pd.to_datetime(self.df["published_at"], errors="coerce")

        logger.info(f"[TriggerLayer] repeated_mentions: {flags.sum()} днandв with {threshold} withгадками")
        return pd.DataFrame({"repeated_mentions": flags})