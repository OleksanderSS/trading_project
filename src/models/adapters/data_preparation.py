import logging
from typing import Any

# src/models/adapters/data_preparation.py - Уніфікована підготовка даних для ML моделей
import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler

from src.config.feature_budget import get_preselection_ceiling
from src.core.exceptions import DataProcessingError
from src.core.logging.logger import ProjectLogger
from src.pipeline.modeling_context import is_pooled
from src.pipeline.target_column_utils import is_identity_column, is_target_like_column

logger = ProjectLogger.get_logger("DataPreparationAdapter")


def _surviving_feature_names(imputer: SimpleImputer, fit_columns: list[str]) -> list[str]:
    """Which columns the imputer kept.

    A column with no observed value in the training rows has no median to
    store, so SimpleImputer drops it and records NaN in `statistics_`.
    `get_feature_names_out` is the supported answer; reading `statistics_`
    is the fallback for an imputer pickled by an older sklearn.
    """
    try:
        return [str(name) for name in imputer.get_feature_names_out(fit_columns)]
    except (AttributeError, ValueError, TypeError) as exc:
        logger.debug(
            "get_feature_names_out unavailable (%s: %s); recovering the "
            "surviving columns from statistics_.", type(exc).__name__, exc,
        )
        statistics = getattr(imputer, "statistics_", None)
        if statistics is None or len(statistics) != len(fit_columns):
            return list(fit_columns)
        return [
            column for column, statistic in zip(fit_columns, statistics)
            if not np.isnan(statistic)
        ]

class _BlockImputer:
    """A median imputer fitted in column blocks, behaving like one imputer.

    `SimpleImputer(strategy='median')` sorts the WHOLE matrix through a
    masked array. On the pooled context of 2026-08-30 that asked for a
    `(91416, 1353)` int64 index -- 944 MiB for one sort -- and killed the
    stage. A median is per column, so fitting block by block gives byte-identical
    statistics with the peak bounded by the block width.

    It has to quack like the original, because the fitted object is both
    inspected (`_surviving_feature_names`) and kept for later transforms of
    prediction data.
    """

    def __init__(self, block: int = 200) -> None:
        self.block = block
        self.parts: list[tuple[list[str], SimpleImputer]] = []

    def fit_transform(self, frame: pd.DataFrame) -> np.ndarray:
        pieces = []
        for start in range(0, frame.shape[1], self.block):
            stop = min(start + self.block, frame.shape[1])
            columns = list(frame.columns[start:stop])
            part = SimpleImputer(strategy='median')
            pieces.append(part.fit_transform(frame[columns].to_numpy()))
            self.parts.append((columns, part))
        return np.hstack(pieces) if pieces else np.empty((len(frame), 0))

    def _block_slices(self) -> list[tuple[int, int]]:
        """Fit-time column spans, derived rather than stored.

        The first version of this kept a `self.slices` list filled in
        `__init__`. Unpickling does not call `__init__`, so every champion
        already on disk came back without the attribute and Stage 5 refused
        all seven contexts with `'_BlockImputer' object has no attribute
        'slices'`. New state on a class whose instances are pickled is a
        migration; state derived from state that already exists is not.
        """
        spans, start = [], 0
        for columns, _part in self.parts:
            spans.append((start, start + len(columns)))
            start += len(columns)
        return spans

    def transform(self, frame: Any) -> np.ndarray:
        """Impute a frame OR an array already in fit-time column order.

        Training hands this a DataFrame; Stage 5 hands it the numpy array it
        built by reindexing to the fit-time columns, and `frame[columns]` with
        a list of names is an IndexError on an array. That asymmetry survived
        because Stage 5 had never once run: the class was written on
        2026-08-31 to stop a MemoryError in training (#158), the training path
        was tested, and the prediction path had no caller to fail. It failed
        the first time stages 5-7 executed, on all seven contexts (#213).
        """
        if not self.parts:
            return np.empty((len(frame), 0))
        if hasattr(frame, 'columns'):
            pieces = [part.transform(frame[columns].to_numpy())
                      for columns, part in self.parts]
        else:
            values = np.asarray(frame)
            spans = self._block_slices()
            width = spans[-1][1]
            if values.shape[1] != width:
                raise ValueError(
                    f"_BlockImputer was fitted on {width} columns and received "
                    f"{values.shape[1]}. The caller must reindex to the "
                    f"fit-time columns before transforming; a positional "
                    f"transform on a different width would impute the wrong "
                    f"column with the wrong median and raise nothing."
                )
            pieces = [part.transform(values[:, start:stop])
                      for (start, stop), (_columns, part)
                      in zip(spans, self.parts)]
        return np.hstack(pieces) if pieces else np.empty((len(frame), 0))

    def get_feature_names_out(self, _input_features=None) -> list[str]:
        """Names that survived, in order. A block drops its own all-NaN columns."""
        names: list[str] = []
        for columns, part in self.parts:
            names.extend(_surviving_feature_names(part, columns))
        return names

    @property
    def statistics_(self) -> np.ndarray:
        stats = [getattr(part, 'statistics_', np.array([]))
                 for _columns, part in self.parts]
        return np.concatenate(stats) if stats else np.array([])


def _target_correlation_ranking(
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    *,
    block: int = 32,
) -> pd.Series:
    """|Pearson r| between each column and the target, after median filling.

    The same statistic `BaseTrainer._select_features_for_model` uses to spend
    a model's budget -- computed here, on the raw training columns, so the
    ranking exists BEFORE anything the width of the frame is materialised.

    Median filling is what the imputer will do anyway, so ranking on the
    filled column is ranking on the column the model will actually see.
    Standardisation afterwards is affine and positive, which leaves Pearson
    correlation unchanged, so the two rankings agree except where they are
    numerically indistinguishable -- which is what the ceiling's margin is
    for (see `get_preselection_ceiling`).

    Column blocks, never the whole matrix: a block of 32 columns over 490,799
    rows is 125 MiB, and the sort inside `nanmedian` copies one more. The
    same computation across all 474 columns at once is what killed the stage.

    Returns one score per column, in column order, with NaN for a column that
    has no median (all-NaN in training) or no variance -- both of which carry
    no information and must sort last rather than randomly.
    """
    y = np.asarray(y_train, dtype=float).ravel()
    observed = np.isfinite(y)
    y_centred = y[observed] - y[observed].mean() if observed.any() else np.array([])
    y_norm = float(np.sqrt(y_centred @ y_centred)) if y_centred.size else 0.0

    scores = np.full(x_train.shape[1], np.nan, dtype=float)
    if y_norm <= 0.0 or len(y) != len(x_train):
        return pd.Series(scores, index=x_train.columns)

    for start in range(0, x_train.shape[1], block):
        stop = min(start + block, x_train.shape[1])
        values = x_train.iloc[:, start:stop].to_numpy(dtype=float, copy=True)
        missing = np.isnan(values)
        if missing.any():
            # `nanmedian` partitions the block in place; `np.ma.median`
            # inside SimpleImputer argsorts it and allocates an int64 index
            # of the same shape, which is the 749 MiB that killed the run.
            # Infinities are already NaN by here -- the caller replaces them
            # in every column that has any -- so no extra pass for them.
            medians = np.nanmedian(values, axis=0)
            # An all-NaN column keeps its NaN median; the fill leaves it NaN,
            # the correlation comes out NaN, and it sorts last. That is the
            # same fate the imputer gives it, one step earlier.
            rows, cols = np.nonzero(missing)
            values[rows, cols] = medians[cols]
        block_values = values[observed]
        block_values -= block_values.mean(axis=0)
        covariance = block_values.T @ y_centred
        norms = np.sqrt(np.einsum('ij,ij->j', block_values, block_values))
        with np.errstate(divide='ignore', invalid='ignore'):
            scores[start:stop] = np.abs(covariance / (norms * y_norm))
        del values, block_values

    return pd.Series(scores, index=x_train.columns)


def _preselect_features(
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    ceiling: int,
) -> list[str] | None:
    """The `ceiling` best-ranked columns, kept in the frame's own order.

    Order is preserved because the downstream budget sorts with a STABLE sort
    (`kind="mergesort"`), so equal correlations there are broken by column
    order. Re-ordering the survivors here would break the tie differently and
    hand a model a different -- equally good, but different -- feature set.

    Returns None when there is nothing to gain, so the caller can pass the
    frame through untouched rather than paying for a copy.
    """
    if x_train.shape[1] <= ceiling:
        return None
    ranking = _target_correlation_ranking(x_train, y_train)
    if ranking.notna().sum() == 0:
        # Nothing could be ranked -- a degenerate target, or no usable column.
        # Refusing to guess is better than keeping the first `ceiling`
        # columns, which would silently make alphabetical order the selector.
        logger.warning(
            "Feature pre-screen ranked no column against the target; all %d "
            "columns are carried forward.", x_train.shape[1],
        )
        return None
    keep = set(
        ranking.fillna(-1.0)
        .sort_values(ascending=False, kind="mergesort")
        .index[:ceiling]
    )
    return [column for column in x_train.columns if column in keep]


def prepare_data_for_models(
    df: pd.DataFrame,
    ticker: str,
    timeframe: str,
    target_cols: list[str],
    seq_len: int = 10,
    val_size: float = 0.1,
    test_size: float = 0.2,
    gap_size: int = 5,  # ✅ ELITE FIX: Gap to prevent leakage between sets
    scale_target: bool = False
) -> dict[str, Any] | None:
    """
    Уніфікований ML адаптер з Purged Validation.
    Додає буферні зони між вибірками для чесного тестування на часових рядах.
    """
    try:
        logger.info(f"prepare_data_for_models called: ticker={ticker}, timeframe={timeframe}, target_cols={target_cols}, df shape={df.shape}")
        if not target_cols:
            logger.error("target_cols є обов'язковим параметром.")
            return None

        # 1. Фільтрація
        filtered_df = filter_data_by_ticker_timeframe(df, ticker, timeframe)
        if filtered_df.empty:
            logger.warning(f"Немає даних для {ticker} {timeframe}")
            return None

        # 2. Перевірка наявності таргетів
        for col in target_cols:
            if col not in filtered_df.columns:
                logger.error(f"Колонка таргета '{col}' не знайдена.")
                return None
        # Conditional, like every other whole-frame operation in this
        # function: `dropna` builds a new 1.29 GiB frame on the pooled daily
        # context whether or not a single row is missing.
        observed = filtered_df[target_cols].notna().all(axis=1)
        if not bool(observed.all()):
            filtered_df = filtered_df.loc[observed]
        datetime_col = next(
            (
                column
                for column in ("datetime", "timestamp", "date")
                if column in filtered_df.columns
            ),
            None,
        )
        if datetime_col:
            model_datetime = pd.to_datetime(
                filtered_df[datetime_col],
                errors="coerce",
                utc=True,
            )
            # Four whole-frame copies in one chain, and each is 1.29 GiB on
            # the pooled daily context (704,724 rows x 245 float64):
            # `assign` opens with `self.copy(deep=None)`, `dropna` builds a
            # new frame, `sort_values` another, `set_index` another. That
            # chain killed the modelling stage on 2026-08-31 twenty-five
            # seconds into the daily frame -- the THIRD MemoryError of the
            # day in this one function, after the median imputer and the
            # categorical copy.
            #
            # Each step is now conditional and the index is attached rather
            # than materialised through a column:
            #  - rows with no timestamp are dropped only if any exist;
            #  - the sort happens only if the frame is not already ordered,
            #    which it is, coming from a time-sorted pipeline;
            #  - the index is assigned onto a SHALLOW copy, which rebinds
            #    labels without touching a single value.
            usable = model_datetime.notna()
            if not bool(usable.all()):
                filtered_df = filtered_df.loc[usable]
                model_datetime = model_datetime.loc[usable]

            stamps = model_datetime.to_numpy()
            if not pd.Index(stamps).is_monotonic_increasing:
                order = np.argsort(stamps, kind="stable")
                filtered_df = filtered_df.iloc[order]
                stamps = stamps[order]
                logger.info(
                    "Sorted %s %s by time (the frame did not arrive ordered).",
                    ticker, timeframe,
                )

            filtered_df = filtered_df.copy(deep=False)
            filtered_df.index = pd.DatetimeIndex(stamps, name="model_datetime")
        if filtered_df.empty:
            logger.warning(
                "No rows with observed targets for %s %s %s",
                ticker,
                timeframe,
                target_cols,
            )
            return None

        # 3. Обробка категоріальних фіч (включаючи нові патерни)
        df_processed, categorical_info = handle_categorical_features(filtered_df, target_cols)

        # 4. Feature selection
        feature_cols = [c for c in df_processed.select_dtypes(include=[np.number]).columns
                        if not is_target_like_column(c)
                        and c not in ['datetime', 'date']
                        and not is_identity_column(c)]
        feature_cols = [
            column
            for column in feature_cols
            if df_processed[column].notna().any()
        ]

        if len(feature_cols) < 1:
            logger.error("Відсутні ознаки для моделювання.")
            return None

        # Replace infinities only in the columns that have any.
        #
        # `frame[cols].replace(...)` copies the whole frame twice: once to
        # select, once to replace. On the pooled daily context that is
        # (245, 704_210) in float64 -- 1.29 GiB for the second copy alone, and
        # it killed the stage after four targets. Most columns hold no
        # infinity at all, so touching every one of them buys nothing.
        #
        # Fourth instance of this shape in two days: a whole-frame operation
        # where a conditional one would do (see the ticker filter in
        # execute_continue_mode, filter_data_by_ticker_timeframe, and the
        # median imputer).
        X = df_processed[feature_cols]
        infinite = [
            column for column in feature_cols
            if X[column].dtype.kind == "f" and np.isinf(X[column].to_numpy()).any()
        ]
        if infinite:
            X = X.copy()
            for column in infinite:
                values = X[column].to_numpy()
                X[column] = np.where(np.isinf(values), np.nan, values)
            logger.info(
                "Replaced infinities in %d of %d feature column(s).",
                len(infinite), len(feature_cols),
            )
        y = df_processed[target_cols]

        # 5. PURGED SPLIT (чесне ділення з розривами)
        total_len = len(X)
        test_start = int(total_len * (1 - test_size))
        val_start = int(test_start * (1 - val_size / (1 - test_size)))

        # The purge gap is configured in BARS but applied with `.iloc`, so it
        # only spans the intended amount of time when a frame holds one row per
        # timestamp. A pooled frame holds one row per ticker per timestamp, so
        # a 24-bar gap across 22 tickers would purge a single bar -- leaving
        # the model's training window all but touching its holdout, which is
        # the exact leak the gap exists to prevent.
        #
        # Scaled by the frame's own rows-per-timestamp rather than by a ticker
        # count, so it needs to know nothing about pooling: a per-ticker frame
        # measures ~1.0 and comes out unchanged.
        rows_per_bar = 1.0
        if X.index.name == "model_datetime":
            distinct_bars = X.index.nunique()
            if distinct_bars:
                rows_per_bar = max(1.0, len(X) / distinct_bars)
        if rows_per_bar > 1.0:
            scaled = int(round(gap_size * rows_per_bar))
            logger.info(
                "Purge gap scaled %d -> %d bars: %.1f rows per timestamp "
                "(pooled frame).", gap_size, scaled, rows_per_bar,
            )
            gap_size = scaled

        # Адаптивний gap_size на основі волатильності
        adaptive_gap_size = gap_size
        vix_col = next((col for col in X.columns if 'vix' in col.lower()), None)
        if vix_col is not None:
            recent_vix = X[vix_col].iloc[-20:].mean()
            if recent_vix > 30:
                adaptive_gap_size = gap_size * 2
            elif recent_vix > 20:
                adaptive_gap_size = int(gap_size * 1.5)
            logger.info(f"Adaptive gap_size calculated based on VIX: {adaptive_gap_size} (Base: {gap_size})")

        # Визначаємо індекси з урахуванням розривів (gap)
        train_end = val_start - adaptive_gap_size
        val_end = test_start - adaptive_gap_size

        # Which series each row belongs to. A pooled frame interleaves tickers
        # at the same timestamp, so "the value h rows ago" is a different
        # company, not the same one h bars earlier -- which is what the
        # persistence opponent in the promotion gate is supposed to mean.
        # Sliced positionally alongside X, because the index (a timestamp)
        # repeats across tickers and cannot align them.
        group_column = next(
            (
                column
                for column in ("ticker", "symbol", "asset")
                if column in df_processed.columns
            ),
            None,
        )
        groups = df_processed[group_column] if group_column else None

        if train_end <= 0 or val_end <= val_start:
             logger.warning("Занадто малий датасет для Purged Validation. Використовуємо стандартне ділення.")
             x_train, x_val, x_test = X.iloc[:val_start], X.iloc[val_start:test_start], X.iloc[test_start:]
             y_train, y_val, y_test = y.iloc[:val_start], y.iloc[val_start:test_start], y.iloc[test_start:]
             holdout_groups = groups.iloc[test_start:] if groups is not None else None
        else:
             x_train, y_train = X.iloc[:train_end], y.iloc[:train_end]
             x_val, y_val = X.iloc[val_start:val_end], y.iloc[val_start:val_end]
             x_test, y_test = X.iloc[test_start:], y.iloc[test_start:]
             holdout_groups = groups.iloc[test_start:] if groups is not None else None
             logger.info(f"✅ Purged Split: Train={len(x_train)}, Val={len(x_val)}, Test={len(x_test)} (Gap={adaptive_gap_size})")

        # 5b. ВІДБІР ДО ЗАПОВНЕННЯ, а не після нього.
        #
        # Кожна модель витрачає бюджет у 5-35 ознак (models.yaml,
        # per_model.*.max_features), але заповнювались і масштабувались усі.
        # Виміряно 31.08 на об'єднаному денному контексті: 474 колонки на
        # 490 799 рядків, з яких найбільший бюджет бере 35. Решту 439 стадія
        # протягувала крізь медіанний імпутер і померла там з MemoryError
        # після восьми годин прогону — 749 МіБ на один індекс сортування.
        #
        # Ранжування тут — ТОЙ САМИЙ статистик, який далі вирішує бюджет
        # (|кореляція Пірсона| з ціллю на тренувальних рядках), тож стеля
        # вдвічі більша за найбільший бюджет не міняє, на чому вчиться
        # модель: перші 35 із перших 70 — це ті самі перші 35.
        #
        # Пропускається, коли цілей більше однієї: тоді і сам бюджет далі
        # не ранжує (довжини не збігаються), тож ранжувати тут — вигадувати
        # порядок, якого нижче ніхто не використає.
        if len(target_cols) == 1:
            ceiling = get_preselection_ceiling()
            kept = _preselect_features(x_train, y_train.to_numpy(), ceiling)
            if kept is not None:
                logger.info(
                    "Feature pre-screen: %d -> %d columns before imputation "
                    "(ceiling %d, largest model budget is smaller still).",
                    len(feature_cols), len(kept), ceiling,
                )
                feature_cols = kept
                x_train = x_train[kept]
                x_val = x_val[kept]
                x_test = x_test[kept]
        else:
            logger.info(
                "Feature pre-screen skipped: %d targets in one call, so the "
                "per-model budget will not rank either.", len(target_cols),
            )

        # 6. ML Трансформації
        #
        # Медіана рахується ПО КОЛОНКАХ, тож імпутер можна фітити блоками —
        # значення виходять ті самі, а пам'ять обмежена шириною блоку.
        #
        # Чому це знадобилось: `SimpleImputer(strategy='median')` усередині
        # працює через masked array і сортує ВСЮ матрицю. На об'єднаному
        # контексті 30.08 це запросило масив `(91416, 1353)` в int64 —
        # 944 МіБ під один індекс сортування — і вбило стадію.
        #
        # Блоки лише знімають стелю пам'яті, не міняючи жодного числа. Самої
        # стелі виявилось замало: 31.08 та сама медіана впала вже на блоці
        # (490799, 200). Тому вище стоїть крок 5b — колонок сюди доходить
        # стільки, скільки бюджети моделей взагалі можуть витратити.
        scaler = StandardScaler()
        imputer = _BlockImputer()

        x_train_imputed = imputer.fit_transform(x_train)
        x_val_imputed = imputer.transform(x_val)
        x_test_imputed = imputer.transform(x_test)

        x_train_scaled_arr = scaler.fit_transform(x_train_imputed)
        x_val_scaled_arr = scaler.transform(x_val_imputed)
        x_test_scaled_arr = scaler.transform(x_test_imputed)

        # SimpleImputer silently drops a column whose training rows are all
        # NaN -- it has no median to store -- so the imputed matrix can be
        # narrower than `feature_cols`, and rebuilding a frame with the
        # original names blows up:
        #
        #   Критична помилка підготовки даних:
        #   Shape of passed values is (335, 388), indices imply ...
        #
        # which aborted training after 34 champions on the 2026-08-14 batch.
        # The same defect was fixed on the prediction path in
        # DataPreparationService._surviving_columns; this is the training
        # side of it.
        surviving_cols = _surviving_feature_names(imputer, feature_cols)
        if len(surviving_cols) != len(feature_cols):
            logger.info(
                "Imputer dropped %d of %d features with no observed value in "
                "training (%s...); the frame is rebuilt on what survived.",
                len(feature_cols) - len(surviving_cols), len(feature_cols),
                ", ".join(sorted(set(feature_cols) - set(surviving_cols))[:3]),
            )
            feature_cols = surviving_cols

        x_train_scaled_df = pd.DataFrame(x_train_scaled_arr, columns=feature_cols, index=x_train.index)
        x_val_scaled_df = pd.DataFrame(x_val_scaled_arr, columns=feature_cols, index=x_val.index)
        x_test_scaled_df = pd.DataFrame(x_test_scaled_arr, columns=feature_cols, index=x_test.index)

        target_scaler = None
        if scale_target:
            target_scaler = StandardScaler()
            y_train_processed = target_scaler.fit_transform(y_train)
            y_val_processed = target_scaler.transform(y_val)
            y_test_processed = target_scaler.transform(y_test)
        else:
            y_train_processed, y_val_processed, y_test_processed = y_train.values, y_val.values, y_test.values

        light_data = {
            'X_train': x_train_scaled_df, 'X_val': x_val_scaled_df, 'X_test': x_test_scaled_df,
            'y_train': y_train_processed, 'y_val': y_val_processed, 'y_test': y_test_processed,
            'imputer': imputer, 'scaler': scaler, 'target_scaler': target_scaler,
            'feature_names': feature_cols, 'categorical_info': categorical_info,
            # For the promotion gate's naive opponents, which have to lag
            # within a series rather than across the pooled row order.
            'holdout_groups': (
                None if holdout_groups is None
                else holdout_groups.to_numpy()
            ),
        }

        heavy_data = prepare_sequence_data_optimized(
            x_train_scaled_arr, x_val_scaled_arr, x_test_scaled_arr,
            y_train_processed, y_val_processed, y_test_processed,
            seq_len
        )

        return {
            'ticker': ticker, 'timeframe': timeframe, 'target_cols': target_cols,
            'light_models': light_data, 'heavy_models': heavy_data,
            'metadata': {'feature_count': len(feature_cols), 'samples': total_len, 'purged_validation': True}
        }
    except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError) as e:
        logger.error(f"Критична помилка підготовки даних: {e}", exc_info=True)
        raise DataProcessingError(f"Критична помилка підготовки даних: {e}") from e

def handle_categorical_features(df: pd.DataFrame, exclude_cols: list[str]) -> tuple[pd.DataFrame, dict]:
    """Кодує категоріальні колонки."""
    # Decide WHAT needs doing before copying anything.
    #
    # `df.copy()` here was a deep copy of the whole frame. On the pooled
    # daily context that is 704,210 rows by 245 float64 columns, and pandas
    # consolidates blocks while copying, so it asked for 1.29 GiB twice and
    # killed the run of 2026-08-31 for the second time that day -- at a
    # different line from the first.
    #
    # The function only ever touches categorical columns: it drops some and
    # one-hots others. Of 245 columns in that frame, the number it can touch
    # is a handful, and usually none at all. Fifth instance of this shape in
    # three days (the ticker filter at 6.81 GiB, filter_data_by_ticker_
    # timeframe at 679 MiB, the median imputer at 944 MiB, the infinity
    # replacement at 1.29 GiB, and now this).
    #
    # Identity and context columns are excluded here, not only at feature
    # selection: label-encoding them first would turn a row hash into a
    # plausible integer that survives every downstream numeric check.
    cat_cols = [c for c in df.select_dtypes(include=['object', 'category']).columns
                if c not in exclude_cols
                and not is_identity_column(c)
                and 'ticker' not in c.lower() and 'timeframe' not in c.lower()]
    if not cat_cols:
        return df, {}

    # Shallow: `drop` and `concat` below rebind columns rather than write
    # into them, so the caller's values are never touched.
    df_out = df.copy(deep=False)

    info = {}
    # Decided first, applied once. `pd.concat([df_out, dummies], axis=1)`
    # inside the loop rebuilt the WHOLE frame per categorical column, so five
    # of them meant five reconstructions of 704,210 x 245 float64 -- 1.29 GiB
    # apiece. One concat at the end costs one.
    dropped: list[str] = []
    encoded: list[pd.DataFrame] = []
    for col in cat_cols:
        nunique = df[col].nunique()
        if nunique < 2:
            dropped.append(col)
            continue
        if nunique <= 5:
            encoded.append(pd.get_dummies(df[col], prefix=col, drop_first=True))
            dropped.append(col)
            info[col] = 'one-hot'
        else:
            # A label-encoded column cannot exist at prediction time.
            #
            # The encoder is built here and discarded here. Training then
            # sees MARKET_REGIME_1d as integers while the prediction path
            # reads the raw frame, where it is still 'TRENDING_UP' -- and
            # pd.to_numeric turns that into NaN, dropping every candidate
            # row. Measured on the 2026-08-10 run: 101 contexts blocked, and
            # every single one names a MARKET_REGIME_* column as the feature
            # null in all 50 rows.
            #
            # Nothing is lost by dropping them. All five columns that reach
            # this branch are MARKET_REGIME variants, and each already has a
            # numeric counterpart in the frame (MARKET_REGIME_ENCODED_*,
            # produced by the enricher and persisted like any other feature).
            #
            # To bring label encoding back, persist the mapping alongside the
            # imputer and scaler in light_data and apply it in
            # DataPreparationService. Until then a feature that cannot be
            # reproduced is not a feature.
            dropped.append(col)
            info[col] = 'dropped_unpersisted_encoding'

    df_out = df_out.drop(columns=dropped) if dropped else df_out
    if encoded:
        df_out = pd.concat([df_out, *encoded], axis=1)
    return df_out, info

def log_data_distribution(df: pd.DataFrame):
    """Логує статистичні показники розподілу ознак."""
    if df.empty:
        return
    stats = []
    for col in df.columns[:5]:
        vals = df[col].dropna()
        if len(vals) > 0:
            stats.append(f"{col}(S:{skew(vals):.2f},K:{kurtosis(vals):.2f})")
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(f"Feature distribution: {', '.join(stats)}")

def prepare_sequence_data_optimized(x_tr, x_va, x_te, y_tr, y_va, y_te, seq_len) -> dict[str, Any]:
    """Створення 3D вікон для Neural Networks за допомогою numpy strides."""
    def strided_window(x, y, window):
        if len(x) <= window:
            return np.array([]), np.array([])
        shape = (x.shape[0] - window, window, x.shape[1])
        strides = (x.strides[0], x.strides[0], x.strides[1])
        x_win = np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides)
        y_win = y[window:]
        return x_win, y_win

    x_train_s, y_train_s = strided_window(x_tr, y_tr, seq_len)
    x_val_s, y_val_s = strided_window(x_va, y_va, seq_len)
    x_test_s, y_test_s = strided_window(x_te, y_te, seq_len)

    return {
        'X_train': x_train_s, 'X_val': x_val_s, 'X_test': x_test_s,
        'y_train': y_train_s, 'y_val': y_val_s, 'y_test': y_test_s,
        'seq_len': seq_len, 'n_features': x_tr.shape[1]
    }

def filter_data_by_ticker_timeframe(df: pd.DataFrame, ticker: str, timeframe: str) -> pd.DataFrame:
    """Фільтрація вхідного набору даних.

    `POOLED_TICKER` keeps every ticker and narrows by timeframe only. The
    timeframe filter still applies because features carry a timeframe suffix,
    so mixing 15m and 1d rows would leave most columns empty on most rows.
    """
    t_cols = [c for c in df.columns if 'ticker' in c.lower() or 'symbol' in c.lower()]
    tf_cols = [c for c in df.columns if 'timeframe' in c.lower() or 'interval' in c.lower()]

    def _select(mask) -> pd.DataFrame:
        """Filter, unless the filter would keep everything.

        A boolean mask that selects every row still makes pandas build a full
        copy. Pooled contexts arrive already narrowed to one timeframe, so the
        timeframe mask is all-True and the copy buys nothing: measured
        2026-08-30 on the pooled 15m frame, the stage died here with

            MemoryError: Unable to allocate 679. MiB for an array with shape
            (1118, 159149) and data type float32

        having already held about 9 GiB of slices. Second instance of this
        exact shape in two days -- the ticker filter in
        `pipeline_executor.execute_continue_mode` was the first, and asked
        for 6.81 GiB before it was made conditional.
        """
        if bool(mask.all()):
            return df
        return df[mask]

    # `== POOLED_TICKER` was exact and case-sensitive; the sentinel travels
    # through file names, JSON keys and context ids and gets re-cased on the
    # way, and a miss here falls silently into the per-ticker branch.
    if is_pooled(ticker):
        return _select(df[tf_cols[0]] == timeframe) if tf_cols else df
    if t_cols and tf_cols:
        return _select((df[t_cols[0]] == ticker) & (df[tf_cols[0]] == timeframe))
    return df

def validate_data_shapes(data: dict[str, Any]) -> bool:
    """Перевірка розмірностей вихідних даних."""
    if not data:
        return False
    for m_type in ['light_models', 'heavy_models']:
        d = data.get(m_type, {})
        if not d:
            continue
        for subset in ['train', 'val', 'test']:
            x, y = d.get(f'X_{subset}'), d.get(f'y_{subset}')
            if x is not None and y is not None and len(x) != len(y):
                logger.warning(f"{m_type} {subset}: X/y length mismatch")
                return False
    return True
