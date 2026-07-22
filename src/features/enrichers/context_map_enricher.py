import hashlib
from typing import Any

import numpy as np
import pandas as pd

from src.core.logging.logger import ProjectLogger

from .base import BaseEnricher

logger = ProjectLogger.get_logger("ContextMapEnricher")

class ContextMapEnricher(BaseEnricher):
    """
    Generates a 'Context Fingerprint' (Market State) and 'Pattern Sequence'.

    🎯 ПАТЕРНИ ТА ПОВНИЙ КОНТЕКСТ:
    1. Інтегрує Macro Score, Market Phase та Sentiment у фінгерпрінт.
    2. Реалізує логіку k-NN патернів (пошук схожих послідовностей станів).
    3. Розраховує стабільність та швидкість зміни ринкового режиму.
    """

    @property
    def name(self) -> str:
        return "context_map"

    @property
    def priority(self) -> int:
        return 80

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__()
        self.config = config or {}

        # ✅ LOAD NOISE FILTER THRESHOLDS
        self.noise_filter_thresholds = {}
        self.temporal_features = set()
        self.default_dynamic_threshold = 0.005
        self.noise_sensitivity = 0.5

        # Конфігурація
        self.champion_ticker = self.config.get('champion_ticker', 'SPY')
        self.velocity_window = self.config.get('velocity_window', 10)
        self.pattern_length = self.config.get('pattern_length', 5) # Довжина послідовності для k-NN логіки

        # Ознаки вищого порядку для включення у фінгерпрінт
        self.higher_order_features = [
            'market_phase', 'macro_composite_score', 'sentiment_momentum',
            'yield_curve_slope', 'MOMENTUM_ZSCORE'
        ]

        # Attempt to load noise config
        from pathlib import Path

        import yaml
        config_path = Path(__file__).parent.parent.parent / "config" / "noise_filter_config.yaml"
        try:
            if config_path.exists():
                with open(config_path, encoding='utf-8') as f:
                    noise_config = yaml.safe_load(f)
                    self.noise_filter_thresholds = noise_config.get('noise_filter_thresholds', {})
                    self.temporal_features = set(noise_config.get('temporal_features', []))
                    self.default_dynamic_threshold = noise_config.get('default_dynamic_threshold', 0.005)
                    self.noise_sensitivity = noise_config.get('noise_sensitivity', 0.5)
        except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError) as e:
             logger.exception(f"Failed to load noise filter config: {e}")
             self._load_defaults()

        logger.info(f"ContextMapEnricher initialized. Pattern Length: {self.pattern_length}")

    def _load_defaults(self):
        self.noise_filter_thresholds = {'VIX': 0.02, 'SPY': 0.005, 'close': 0.005}
        self.temporal_features = {'hour', 'day_of_week', 'is_weekend'}

    def _enrich_impl(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Генерує розширений фінгерпрінт та ідентифікатори патернів."""
        if df.empty:
            return df

        res_df = df.copy()

        # 1. Отримуємо стан Чемпіона
        champion_state = self._get_champion_state(res_df)

        # 2. Визначаємо колонки для аналізу
        context_columns = self._get_context_columns(df)
        if not context_columns:
            return df

        # 3. Обробляємо базові стани
        state_cols, temporal_cols = self._process_context_columns(res_df, context_columns)

        # 4. Додаємо стан Чемпіона
        if champion_state is not None:
            res_df['state_champion'] = champion_state
            state_cols.append('state_champion')

        # 5. Додаємо ознаки вищого порядку (Phase, Macro тощо)
        ho_cols = self._integrate_higher_order_features(res_df, state_cols)
        state_cols.extend(ho_cols)

        # 6. Генеруємо фінальний фінгерпрінт
        if state_cols or temporal_cols:
            self._generate_context_features(res_df, state_cols, temporal_cols)

            # 7. ЛОГІКА ПАТЕРНІВ (k-NN style sequence encoding)
            self._generate_pattern_sequences(res_df)

            self._calculate_context_velocity(res_df)
            self._log_context_statistics(res_df, state_cols, temporal_cols)

        return res_df

    def _get_champion_state(self, df: pd.DataFrame) -> pd.Series | None:
        """Визначає режим Чемпіона."""
        if 'ticker' not in df.columns or self.champion_ticker not in df['ticker'].values:
            return None
        champ_data = df[df['ticker'] == self.champion_ticker].copy()
        if champ_data.empty:
            return None
        champ_close = champ_data['close']
        champ_sma = champ_close.rolling(20, min_periods=1).mean()
        state = np.where(champ_close > champ_sma, 1, -1)
        # Create Series with original index and reindex to df.index
        champ_state = pd.Series(state, index=champ_data.index)
        aligned_state = champ_state.reindex(df.index).ffill()
        return aligned_state.where(aligned_state.notna(), 0).astype(int)

    def _get_context_columns(self, df: pd.DataFrame) -> list[str]:
        """Отримує числові колонки, ігноруючи таргети та вже створені стани."""
        context_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        exclude = ['hash', 'interval', 'state_champion',
            'context_pattern_id', 'context_pattern_seq']
        # audit-ignore: ARCHITECTURAL_USAGE
        return [c for c in context_columns if not c.startswith('target_') and not c.startswith('state_') and c not in exclude]

    def _integrate_higher_order_features(self, df: pd.DataFrame, existing_states: list[str]) -> list[str]:
        """Інтегрує аналітичні ознаки (Phase, Macro) у фінгерпрінт."""
        added_cols = []
        existing_state_set = set(existing_states)
        for feat in self.higher_order_features:
            if feat in df.columns:
                state_name = f"state_{feat}"
                if state_name in existing_state_set:
                    continue
                # Для категоріальних (як market_phase) беремо як є, для числових - адаптивний поріг
                if feat == 'market_phase':
                    phase_state = pd.to_numeric(df[feat], errors='coerce')
                    df[state_name] = phase_state.where(phase_state.notna(), 0).astype(int)
                else:
                    self._process_numeric_column(df, feat, state_name, [])
                added_cols.append(state_name)
                existing_state_set.add(state_name)
        return added_cols

    def _process_context_columns(self, res_df: pd.DataFrame, context_columns: list[str]) -> tuple:
        """Перетворює сирі дані у дискретні стани (-1, 0, 1)."""
        state_cols, temporal_cols = [], []
        for col in context_columns:
            if col not in res_df.columns:
                continue
            state_col_name = f"state_{col}"
            if col in self.temporal_features:
                res_df[state_col_name] = res_df[col]
                temporal_cols.append(state_col_name)
            else:
                self._process_numeric_column(res_df, col, state_col_name, state_cols)
        return state_cols, temporal_cols

    def _process_numeric_column(self, res_df: pd.DataFrame, col: str, state_col_name: str, state_cols: list[str]):
        """Адаптивний фільтр шуму."""
        returns = res_df[col].pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
        rolling_std = returns.rolling(window=20, min_periods=2).std()
        threshold = (rolling_std * self.noise_sensitivity).clip(lower=1e-6)
        state = pd.Series(0, index=res_df.index, dtype=int)
        valid = returns.notna() & threshold.notna()
        state.loc[valid] = np.where(
            returns.loc[valid] > threshold.loc[valid],
            1,
            np.where(returns.loc[valid] < -threshold.loc[valid], -1, 0),
        )
        res_df[state_col_name] = state
        if state_col_name not in state_cols:
            state_cols.append(state_col_name)

    def _generate_context_features(self, res_df: pd.DataFrame, state_cols: list[str], temporal_cols: list[str]):
        """Створює фінгерпрінт як конкатенацію станів."""
        all_state_cols = sorted(set(state_cols + temporal_cols))
        res_df['context_fingerprint'] = res_df[all_state_cols].astype(str).agg('|'.join, axis=1)
        if state_cols:
            res_df['context_stability'] = (res_df[state_cols] == 0).sum(axis=1) / len(state_cols)

    def _generate_pattern_sequences(self, df: pd.DataFrame):
        """
        🎯 SEQUENCE ENCODING (k-NN logic):
        Створює ідентифікатор патерна на основі послідовності фінгерпрінтів.
        Це дозволяє моделі розрізняти "початок тренду", "кульмінацію" тощо.
        """

        # Створюємо ковзне вікно послідовності фінгерпрінтів
        # Використовуємо str.cat для ефективного об'єднання серій
        sequences = df['context_fingerprint'].astype(str)
        for i in range(1, self.pattern_length):
            shifted = df.groupby('ticker')['context_fingerprint'].shift(i).fillna("START").astype(str)
            sequences = sequences.str.cat(shifted, sep=">>")

        # Хешуємо отриману послідовність для стиснення розмірності
        # apply(lambda...) — це вузьке місце, але `hashlib` важко векторизувати.
        # Можна спробувати перетворити на список і хешувати в циклі, але залишимо apply,
        # бо він вже є досить оптимізованим для серій.
        # Keep the raw sequence for distance-based pattern matching. The hash is
        # useful as a compact ID, but KNN needs the original state sequence.
        df['context_pattern_seq'] = sequences
        df['context_pattern_id'] = sequences.apply(lambda x: hashlib.sha256(x.encode()).hexdigest()[:8])

    def _calculate_context_velocity(self, res_df: pd.DataFrame):
        """Розраховує швидкість зміни режимів."""
        fingerprint_changed = (res_df['context_fingerprint'] != res_df.groupby('ticker')['context_fingerprint'].shift(1)).astype(int)
        res_df['context_velocity'] = fingerprint_changed.rolling(window=self.velocity_window, min_periods=1).mean()
        res_df['context_anxiety_index'] = (res_df['context_velocity'] > 0.6).astype(int)

    def _log_context_statistics(self, res_df: pd.DataFrame, state_cols: list[str], temporal_cols: list[str]):
        logger.info(f"✅ Context Patterns Generated. Features integrated: {len(self.higher_order_features)}")

    def _get_threshold(self, df: pd.DataFrame, col: str) -> float:
        return self.noise_filter_thresholds.get(col, self.default_dynamic_threshold)
