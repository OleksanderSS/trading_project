import hashlib
from typing import Any

import numpy as np
import pandas as pd

from src.core.logging.logger import ProjectLogger
from src.features.context_schema import record_schema

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

        # Which state columns compose the fingerprint. Empty means all of
        # them, which is what this did unconditionally and why the fingerprint
        # was useless: joining 185 columns left 99.9% of values unique on the
        # 2026-08-14 export -- 7 repeated out of 12,170 on 15m, 3 out of
        # 11,350 on daily. Nothing downstream can match a pattern that occurs
        # once, so context_velocity was 1.0 on every bar and
        # context_anxiety_index the constant 1.
        #
        # Measured on those same rows, non-calendar state columns give:
        #
        #     width 6  -> 215 repeated groups, median size 23
        #     width 8  -> 841 repeated groups, median size  5
        #     width 12 -> 1,611 groups, median size 3
        #     width 185 ->   7 groups, median size 1
        #
        # Width is not the only thing that matters: ranked by entropy the top
        # non-calendar columns are EMA_20, SMA_20, BB_Middle, SMA_10, EMA_10 --
        # eight ways of asking "is price above its moving average", which
        # carries about one bit between them. A useful list spans trend,
        # volatility, volume and momentum instead. Ranked WITH the calendar in
        # it, day_of_year wins outright, and a fingerprint keyed on the date
        # matches nothing but coincidence.
        self.fingerprint_columns = list(self.config.get('fingerprint_columns') or [])

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
        """Перетворює сирі дані у дискретні стани (-1, 0, 1).

        Calendar features are NOT re-emitted here. The branch that handled
        them did `res_df[state_col_name] = res_df[col]` — a verbatim copy, no
        discretisation — so `state_hour_15m` held exactly what `hour_15m`
        held. Measured on the 2026-08-15 export, 24 such duplicates existed
        (8 calendar names across 3 timeframes), each entering the feature pool
        as a second, independently selectable copy of a column already in it.

        They were not harmless. Ranked by entropy, calendar columns beat every
        market column outright — day_of_year scores H=5.06 against 1.58 for
        the best price state — so they won selection on having many distinct
        values rather than on saying anything about the market. In this run's
        importances, state_day_of_month_1d, state_hour_15m and
        ctx_60m_hour_cos_60m were among the most frequently chosen context
        columns. The same pull had already been found distorting the context
        fingerprint, where a date-keyed print matches nothing but coincidence.

        The information is not lost: `hour_15m`, `day_of_week_15m` and the
        rest remain in the frame, once each, for any model that wants them.
        What goes is the duplicate wearing a `state_` prefix, which claimed to
        be a discretised market state and was not.

        test_pipeline_control_train_validation_experiment already asserted
        `state_day_of_month_15m` must not reach selection; it no longer needs
        a downstream filter to arrange that.
        """
        state_cols, temporal_cols = [], []
        skipped = []
        for col in context_columns:
            if col not in res_df.columns:
                continue
            if col in self.temporal_features:
                skipped.append(col)
                continue
            self._process_numeric_column(res_df, col, f"state_{col}", state_cols)
        if skipped:
            logger.info(
                "Calendar features left un-duplicated (%d): %s. The raw "
                "columns stay in the frame; only the identical `state_` copies "
                "are gone.", len(skipped), sorted(skipped),
            )
        return state_cols, temporal_cols

    def _process_numeric_column(self, res_df: pd.DataFrame, col: str, state_col_name: str, state_cols: list[str]):
        """Адаптивний фільтр шуму."""
        if 'ticker' in res_df.columns:
            returns = res_df.groupby('ticker')[col].pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
            rolling_std = returns.groupby(res_df['ticker']).rolling(window=20, min_periods=2).std().reset_index(level=0, drop=True)
        else:
            returns = res_df[col].pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
            rolling_std = returns.rolling(window=20, min_periods=2).std()
        threshold = (rolling_std * self.noise_sensitivity).clip(lower=1e-6)

        # Positional numpy arrays, not .loc[boolean_mask]: res_df's index
        # can carry duplicate labels by this point in the pipeline (other
        # enrichers reset per-ticker indices during their own merge_asof
        # steps), and .loc[] with a duplicate-labeled index misaligns or
        # raises instead of a plain positional match. returns/threshold
        # are already positionally aligned with res_df (groupby-transform
        # preserves row order and length), so operate on .to_numpy().
        returns_vals = returns.to_numpy()
        threshold_vals = threshold.to_numpy()
        valid = ~np.isnan(returns_vals) & ~np.isnan(threshold_vals)
        state_vals = np.zeros(len(res_df), dtype=int)
        state_vals[valid] = np.where(
            returns_vals[valid] > threshold_vals[valid],
            1,
            np.where(returns_vals[valid] < -threshold_vals[valid], -1, 0),
        )
        res_df[state_col_name] = state_vals
        if state_col_name not in state_cols:
            state_cols.append(state_col_name)

    def _fingerprint_drivers(self, res_df: pd.DataFrame, state_cols: list[str],
                             temporal_cols: list[str]) -> list[str]:
        """The columns the fingerprint is built from.

        Configured names are honoured in the order given -- position i of a
        fingerprint IS driver i, so the order is part of the schema and must
        not be re-sorted. Names absent from the frame are reported rather than
        skipped in silence: a typo in the list would otherwise narrow the
        fingerprint invisibly, which is the failure mode this whole change
        exists to end.

        With no list configured the behaviour is unchanged -- every state
        column, sorted -- because narrowing it by guesswork would silently
        redefine every fingerprint ever written. The warning says what that
        costs.
        """
        if not self.fingerprint_columns:
            every = sorted(set(state_cols + temporal_cols))
            logger.warning(
                "Context fingerprint uses all %d state columns. Measured on "
                "the 2026-08-14 export that leaves 99.9%% of fingerprints "
                "unique (7 repeat out of 12,170), so context_velocity is 1.0 "
                "on every bar and context_anxiety_index the constant 1. Set "
                "`fingerprint_columns` to a short list spanning trend, "
                "volatility, volume and momentum.", len(every),
            )
            return every

        available = set(res_df.columns)
        chosen = [c for c in self.fingerprint_columns if c in available]
        missing = [c for c in self.fingerprint_columns if c not in available]
        if missing:
            logger.error(
                "Configured fingerprint columns are absent from the frame and "
                "the fingerprint is narrower than intended: %s", missing,
            )
        if not chosen:
            logger.error(
                "None of the configured fingerprint columns exist; falling "
                "back to every state column."
            )
            return sorted(set(state_cols + temporal_cols))
        logger.info("Context fingerprint built from %d drivers: %s",
                    len(chosen), chosen)
        return chosen

    def _generate_context_features(self, res_df: pd.DataFrame, state_cols: list[str], temporal_cols: list[str]):
        """Створює фінгерпрінт як конкатенацію станів."""
        all_state_cols = self._fingerprint_drivers(res_df, state_cols, temporal_cols)
        # Position i of every fingerprint below IS all_state_cols[i]. Nothing
        # recorded that until now, so any analysis of which driver hurts could
        # only name positions -- and the ordering silently shifts whenever the
        # feature set changes, which re-points every fingerprint ever written.
        # Registering it makes fingerprints decodable and, more importantly,
        # makes a stale decoding detectable.
        recorded = record_schema(all_state_cols)
        if recorded:
            logger.info(
                "Context driver schema %s recorded (%d drivers).",
                recorded, len(all_state_cols),
            )
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
        """Розраховує швидкість зміни режимів.

        Two defects, and the second makes the column a constant.

        The rolling mean ran over the whole frame while the change flag was
        computed per ticker, so each ticker's first `velocity_window` bars
        averaged in the tail of whichever ticker preceded it in row order.

        And the flag itself: `context_fingerprint` joins all 185 state
        columns, which on the 2026-08-14 export left 99.9% of fingerprints
        unique — 7 values repeat out of 12,170 on 15m, 3 out of 11,350 on
        daily. A fingerprint that never repeats changes on every bar, so
        velocity is 1.0 everywhere (measured mean 0.9994) and anxiety, being
        `velocity > 0.6`, is the constant 1. Neither can inform anything.

        Measured on the same rows, a fingerprint of the 8 most informative
        NON-calendar state columns yields 841 repeated groups with a median
        size of 5; of 6 columns, 215 groups with a median of 23. The width is
        the whole story, so it is configurable via `fingerprint_columns` and
        no longer silently "all of them".
        """
        changed = (
            res_df['context_fingerprint']
            != res_df.groupby('ticker')['context_fingerprint'].shift(1)
        ).astype(int)
        res_df['context_velocity'] = (
            changed.groupby(res_df['ticker'])
                   .rolling(window=self.velocity_window, min_periods=1)
                   .mean()
                   .reset_index(level=0, drop=True)
        )
        res_df['context_anxiety_index'] = (res_df['context_velocity'] > 0.6).astype(int)

        # A RANK, so downstream thresholds cannot rot.
        #
        # Velocity is the share of recent bars whose fingerprint changed, and
        # its scale is set by how often the fingerprint repeats -- which is a
        # property of `fingerprint_columns`, not of the market. Widen that list
        # and every absolute threshold downstream silently re-points.
        #
        # It already happened. Stage 6 blocks buys when velocity exceeds 0.85,
        # a number chosen when fingerprints were nearly unique and velocity sat
        # at 1.0 everywhere. Measured on the 2026-08-20 batch it is exceeded on
        # 64% of 15m bars, 65% of 60m and 82% of daily -- a rule written for a
        # rare emergency describing four bars in five.
        #
        # The percentile is EXPANDING and computed in time order per ticker, so
        # the value at a bar uses only that ticker's own past. Both halves of
        # that matter: an expanding statistic computed over the frame's row
        # order rather than over time put 2026 sentiment into a 1996 bar
        # earlier today (see AdvancedAnalyticsEnricher). `groupby.transform`
        # walks row order, and by this point in the chain the frame has been
        # reordered several times.
        if 'datetime' in res_df.columns:
            order = res_df.sort_values(['ticker', 'datetime']).index
            v = res_df.loc[order, 'context_velocity']
            res_df['context_velocity_rank'] = (
                v.groupby(res_df.loc[order, 'ticker'])
                 .transform(lambda s: s.expanding(min_periods=20).rank(pct=True))
                 .reindex(res_df.index)
            )
        else:
            logger.warning(
                "context_velocity_rank skipped: no datetime column, and an "
                "expanding rank over row order would not be causal."
            )
            res_df['context_velocity_rank'] = np.nan

    def _log_context_statistics(self, res_df: pd.DataFrame, state_cols: list[str], temporal_cols: list[str]):
        logger.info(f"✅ Context Patterns Generated. Features integrated: {len(self.higher_order_features)}")

    def _get_threshold(self, df: pd.DataFrame, col: str) -> float:
        return self.noise_filter_thresholds.get(col, self.default_dynamic_threshold)
