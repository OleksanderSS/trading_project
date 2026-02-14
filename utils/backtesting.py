# utils.backtesting.py

import os
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List
from sklearn.preprocessing import MinMaxScaler
from utils.features_utils import FeatureUtils
from utils.trading_signals import generate_signals
from utils.visualization import create_charts
from utils.logger import ProjectLogger
from utils.metrics import evaluate_models_ensemble
from utils.trading_calendar import TradingCalendar


logger = ProjectLogger.get_logger("TradingProjectLogger")

class Backtester:
    def __init__(
        self,
        models: Dict[str, Any],
        scaler_X: Optional[MinMaxScaler] = None,
        scaler_y: Optional[MinMaxScaler] = None,
        features_list: Optional[List[str]] = None,
        time_steps: int = 10,
        enable_signals: bool = True,
        max_plot_rows: int = 5000,
        debug: bool = False,
        ensemble_weights: Optional[Dict[str, float]] = None,
        rolling_window: int = 5,
        feature_layers: Optional[List[str]] = None,
        calendar: Optional[TradingCalendar] = None
    ):
        self.models = models or {}
        self.scaler_X = scaler_X or MinMaxScaler()
        self.scaler_y = scaler_y or MinMaxScaler()
        self.features_list = features_list or []
        self.time_steps = time_steps
        self.enable_signals = enable_signals
        self.max_plot_rows = max_plot_rows
        self.debug = debug
        self.fe = FeatureUtils(short_window=3, long_window=200)
        self.ensemble_weights = ensemble_weights
        self.rolling_window = rolling_window
        self.feature_layers = feature_layers or []
        self.calendar = calendar
        self.last_features = None  # кеш for df_features

    def run_backtest(
        self,
        df: pd.DataFrame,
        target_column: str,
        test_ratio: float = 0.2,
        interval: str = "",
        output_path: str = "data/backtests",
        avg_sentiment: Optional[dict] = None
    ) -> Dict[str, Any]:

        if df.empty or target_column not in df.columns:
            logger.warning(f"[Backtester] Порожнandй DataFrame or вandдсутнandй target_column '{target_column}'")
            return {"metrics": {}, "predictions": {}, "signals": {}}

        if df[target_column].dropna().empty:
            logger.warning(f"[Backtester] Target column '{target_column}' мandстить лише NaN  бектест пропущено")
            return {"metrics": {}, "predictions": {}, "signals": {}}

        if self.calendar and "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
            df["is_trading_day"] = df["date"].apply(self.calendar.is_trading_day)
            df = df[df["is_trading_day"]]
            if df.empty:
                logger.warning(f"[Backtester] Всand днand були notторговими  бектест пропущено")
                return {"metrics": {}, "predictions": {}, "signals": {}}

        os.makedirs(output_path, exist_ok=True)

        df_features = self.fe.transform(df, key=interval)
        self.last_features = df_features.copy()

        feature_cols = [c for c in (self.features_list or df_features.select_dtypes(include=np.number).columns)
                        if c in df_features.columns]
        if not feature_cols:
            feature_cols = df_features.select_dtypes(include=np.number).columns.tolist()
            logger.warning("[Backtester] Використовую all числовand колонки як фandчand")
        elif self.features_list and not feature_cols:
            logger.warning("[Backtester] Жодна with fordata оwithнак not withнайwhereна в data")

        logger.info(f"[Backtester] Викорисandнand фandчand: {feature_cols}")
        if self.feature_layers:
            logger.info(f"[Backtester] Шари фandчей: {self.feature_layers}")

        X = np.nan_to_num(df_features[feature_cols].values, nan=0.0, posinf=0.0, neginf=0.0)
        y = np.nan_to_num(df_features[target_column].values.reshape(-1, 1), nan=0.0, posinf=0.0, neginf=0.0)

        n_test = max(int(len(X) * test_ratio), 1)
        if len(X) <= n_test + self.time_steps:
            n_test = max(1, len(X) - self.time_steps)
            logger.warning(f"[Backtester] Тестова вибandрка скорочена до n_test={n_test}")

        try:
            self.scaler_X.fit(X[:-n_test])
            self.scaler_y.fit(y[:-n_test])
            X_test_scaled = self.scaler_X.transform(X[-n_test:])
            y_test_scaled = self.scaler_y.transform(y[-n_test:])
        except Exception as e:
            logger.exception(f"[Backtester] Error трансформацandї тестових data: {e}")
            X_test_scaled = np.nan_to_num(X[-n_test:])
            y_test_scaled = np.nan_to_num(y[-n_test:])

        if np.std(y_test_scaled) == 0:
            logger.warning("[Backtester] y_test_scaled має нульову дисперсandю  метрики можуть бути notкоректнand")

        try:
            results = evaluate_models_ensemble(
                models=self.models,
                X_data_scaled=X_test_scaled,
                y_true_scaled=y_test_scaled,
                y_true_original=y[-n_test:],
                target_scaler=self.scaler_y,
                interval=interval,
                output_path=output_path
            )
        except Exception:
            logger.exception("[Backtester] Error оцandнки моwhereлей, поверandю пустand реwithульandти")
            results = {"metrics": {}, "predictions": {}}

        plot_rows = min(self.max_plot_rows, n_test)
        preds = results.get("predictions", {})
        if not isinstance(preds, dict):
            preds = {}
        df_plot = pd.DataFrame({k: np.nan_to_num(v[-plot_rows:], nan=0.0) for k, v in preds.items()},
                               index=df_features.index[-plot_rows:])
        df_plot[target_column] = y[-plot_rows:]

        try:
            create_charts(df_plot, ticker_symbol=interval, interval=interval, output_path=output_path)
        except Exception:
            logger.exception("[Backtester] Error вandwithуалandforцandї графandкandв")

        signals = {"forecast_signal": "HOLD", "rsi_signal": "HOLD", "sentiment_signal": "HOLD", "final_signal": "HOLD"}
        if self.enable_signals:
            try:
                for k, v in signals.items():
                    df_plot[k] = v if isinstance(v, list) else [v] * len(df_plot)
                signals = generate_signals(
                    df_features,
                    avg_sentiment=avg_sentiment,
                    models_dict=self.models,
                    debug=self.debug,
                    ticker=interval
                )
            except Exception:
                logger.exception("[Backtester] Error геnotрацandї сигналandв")

        return {
            "metrics": results.get("metrics", {}),
            "predictions": results.get("predictions", {}),
            "signals": signals,
            "df_plot": df_plot,
            "features_used": feature_cols,
            "feature_layers": self.feature_layers,
            "charts_path": os.path.join(output_path, f"{interval}_charts.png")
        }