import logging
from typing import Any

# src/models/adapters/data_preparation.py - Уніфікована підготовка даних для ML моделей
import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler

from src.core.exceptions import DataProcessingError
from src.core.logging.logger import ProjectLogger
from src.pipeline.modeling_context import POOLED_TICKER
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
        filtered_df = filtered_df.dropna(subset=target_cols)
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
            filtered_df = (
                filtered_df.assign(_model_datetime=model_datetime)
                .dropna(subset=["_model_datetime"])
                .sort_values("_model_datetime", kind="mergesort")
                .set_index("_model_datetime", drop=True)
            )
            filtered_df.index.name = "model_datetime"
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

        X = df_processed[feature_cols].replace([np.inf, -np.inf], np.nan)
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

        if train_end <= 0 or val_end <= val_start:
             logger.warning("Занадто малий датасет для Purged Validation. Використовуємо стандартне ділення.")
             x_train, x_val, x_test = X.iloc[:val_start], X.iloc[val_start:test_start], X.iloc[test_start:]
             y_train, y_val, y_test = y.iloc[:val_start], y.iloc[val_start:test_start], y.iloc[test_start:]
        else:
             x_train, y_train = X.iloc[:train_end], y.iloc[:train_end]
             x_val, y_val = X.iloc[val_start:val_end], y.iloc[val_start:val_end]
             x_test, y_test = X.iloc[test_start:], y.iloc[test_start:]
             logger.info(f"✅ Purged Split: Train={len(x_train)}, Val={len(x_val)}, Test={len(x_test)} (Gap={adaptive_gap_size})")

        # 6. ML Трансформації
        imputer = SimpleImputer(strategy='median')
        scaler = StandardScaler()

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
            'feature_names': feature_cols, 'categorical_info': categorical_info
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
    df_out = df.copy()
    # Identity and context columns are excluded here, not only at feature
    # selection: label-encoding them first would turn a row hash into a
    # plausible integer that survives every downstream numeric check.
    cat_cols = [c for c in df_out.select_dtypes(include=['object', 'category']).columns
                if c not in exclude_cols
                and not is_identity_column(c)
                and 'ticker' not in c.lower() and 'timeframe' not in c.lower()]

    info = {}
    for col in cat_cols:
        nunique = df_out[col].nunique()
        if nunique < 2:
            df_out.drop(columns=[col], inplace=True)
            continue
        if nunique <= 5:
            dummies = pd.get_dummies(df_out[col], prefix=col, drop_first=True)
            df_out = pd.concat([df_out, dummies], axis=1).drop(columns=[col])
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
            df_out.drop(columns=[col], inplace=True)
            info[col] = 'dropped_unpersisted_encoding'
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
    if ticker == POOLED_TICKER:
        return df[df[tf_cols[0]] == timeframe] if tf_cols else df
    if t_cols and tf_cols:
        return df[(df[t_cols[0]] == ticker) & (df[tf_cols[0]] == timeframe)]
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
