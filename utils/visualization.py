# utils/visualization.py

import os
from typing import Dict, Optional, List
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from utils.logger import ProjectLogger
from config.visualization_config import VISUALIZATION_DEFAULTS

try:
    import seaborn as sns

    _HAS_SEABORN = True
except ImportError:
    _HAS_SEABORN = False

logger = ProjectLogger.get_logger("TradingProjectLogger")


# --------------------------
# CSV helper
# --------------------------
def load_csv_safe(file_path: str) -> pd.DataFrame:
    try:
        return pd.read_csv(file_path, encoding='utf-8')
    except UnicodeDecodeError:
        logger.warning(f"[Plot] Unicode error reading {file_path}, trying latin1...")
        df = pd.read_csv(file_path, encoding='latin1')
        df.to_csv(file_path, encoding='utf-8', index=False)
        return df


# --------------------------
# Helpers
# --------------------------
def _ensure_output_dir(path: str) -> str:
    if not path:
        return os.getcwd()
    dirpath = os.path.dirname(path)
    if not dirpath:
        dirpath = os.getcwd()
    os.makedirs(dirpath, exist_ok=True)
    return dirpath


def _safe_savefig(fig: plt.Figure, out_path: str, show_plot: bool = False) -> str:
    _ensure_output_dir(out_path)
    fig.savefig(out_path, bbox_inches="tight")
    if not show_plot:
        plt.close(fig)
    else:
        plt.show()
    logger.info(f"[Visualization] [OK] Збережено графandк: {out_path}")
    return os.path.abspath(out_path)


def _prepare_df(df: pd.DataFrame, price_col: str = "close") -> (pd.DataFrame, str):
    if df is None or df.empty:
        raise ValueError("DataFrame порожнandй")
    df = df.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ['_'.join([str(i) for i in col if i]) for col in df.columns]
    if not isinstance(df.index, pd.DatetimeIndex):
        date_cols = [c for c in df.columns if 'date' in c.lower()]
        if date_cols:
            df[date_cols[0]] = pd.to_datetime(df[date_cols[0]], errors='coerce')
            df = df.set_index(date_cols[0]).sort_index()
        else:
            df.index = pd.RangeIndex(len(df))
    price_candidates = [price_col, price_col.lower(), price_col.upper(), price_col.capitalize()]
    price_col_found = next((c for c in price_candidates if c in df.columns), None)
    if not price_col_found:
        num_cols = df.select_dtypes(include=[np.number]).columns
        if len(num_cols) == 0:
            raise ValueError("Немає числових колонок for побудови графandка")
        price_col_found = num_cols[0]
    return df, price_col_found


def fill_missing_data(df: pd.DataFrame, numeric_cols: Optional[List[str]] = None) -> pd.DataFrame:
    df = df.copy()
    if numeric_cols is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for col in numeric_cols:
        if col in df.columns:
            df[col] = df[col].ffill().bfill()
    return df


# --------------------------
# Technical indicators plot
# --------------------------
def plot_technical_indicators(
        df: pd.DataFrame,
        price_col: str = "close",
        output_path: Optional[str] = None,
        ticker_symbol: str = "SPY",
        interval: str = "1d",
        show_plot: bool = False
) -> Optional[str]:
    df, price_col_found = _prepare_df(df, price_col)
    df = fill_missing_data(df)
    indicators = [c for c in df.select_dtypes(include=[np.number]).columns if c != price_col_found]

    n_subplots = 1 + len(indicators)
    fig, axes = plt.subplots(n_subplots, 1, figsize=(15, 4 * n_subplots), sharex=True)
    if n_subplots == 1:
        axes = [axes]

    ax_price = axes[0]
    ax_price.plot(df.index, df[price_col_found], label="Price", color="blue", linewidth=1.5)
    ax_price.set_title(f"Цandна closing {ticker_symbol} ({interval})")
    ax_price.legend()
    ax_price.grid(True)

    # --- Фаfor ринку як фон ---
    if "market_phase" in df.columns:
        phases = df["market_phase"].fillna("unknown")
        colors_map = VISUALIZATION_DEFAULTS["colors_map"]
        for i in range(1, len(df)):
            phase = phases.iloc[i]
            color = colors_map.get(phase, colors_map["unknown"])
            ax_price.axvspan(df.index[i - 1], df.index[i], facecolor=color, alpha=0.2)

    # --- Тренд як маркер ---
    if "trend_alignment" in df.columns:
        trend = df["trend_alignment"].fillna("")
        trend_colors = VISUALIZATION_DEFAULTS["trend_colors"]
        for i in range(len(df)):
            if trend.iloc[i] == "aligned":
                ax_price.plot(df.index[i], df[price_col_found].iloc[i], marker="o",
                              color=trend_colors["aligned"], markersize=5)
            elif trend.iloc[i] == "divergent":
                ax_price.plot(df.index[i], df[price_col_found].iloc[i], marker="x",
                              color=trend_colors["divergent"], markersize=5)

    # --- Макро-бandас як текст ---
    if "macro_bias" in df.columns:
        bias = df["macro_bias"].fillna("")
        for i in range(0, len(df), max(1, len(df) // 10)):
            ax_price.annotate(str(bias.iloc[i]), xy=(df.index[i], df[price_col_found].iloc[i]),
                              xytext=(0, 10), textcoords="offset points",
                              fontsize=VISUALIZATION_DEFAULTS["macro_bias_fontsize"],
                              color=VISUALIZATION_DEFAULTS["macro_bias_color"])

    colors = plt.cm.tab20.colors
    for i, col in enumerate(indicators, start=1):
        color = colors[i % len(colors)]
        label = col.upper() if isinstance(col, str) else str(col)
        axes[i].plot(df.index, df[col], label=label, color=color, linewidth=1.2)
        axes[i].set_title(label)
        axes[i].legend()
        axes[i].grid(True)

    if output_path:
        filename = os.path.join(output_path, f"technical_indicators_{ticker_symbol}_{interval}.png")
        return _safe_savefig(fig, filename, show_plot)

    if show_plot:
        plt.show()
    else:
        plt.close(fig)
    return None


# --------------------------
# Create charts
# --------------------------
def create_charts(
        df: pd.DataFrame,
        ticker_symbol: str,
        interval: str,
        output_path: str,
        kde: bool = False,
        show_plot: bool = False
) -> Dict[str, Optional[str]]:
    results: Dict[str, Optional[str]] = {}
    if df is None or df.empty:
        raise ValueError(f"create_charts отримав порожнandй df for {ticker_symbol}")

    df = fill_missing_data(df)

    sentiment_col = "sentiment_score"
    if sentiment_col in df.columns and not df[sentiment_col].isnull().all():
        fig = plt.figure(figsize=VISUALIZATION_DEFAULTS["figsize"])
        if _HAS_SEABORN:
            sns.histplot(df[sentiment_col].dropna(), kde=kde, bins=VISUALIZATION_DEFAULTS["bins"])
        else:
            plt.hist(df[sentiment_col].dropna(), bins=VISUALIZATION_DEFAULTS["bins"])
        plt.title(f"Роwithподandл сентименту for {ticker_symbol} ({interval})")
        plt.xlabel("Оцandнка сентименту")
        plt.ylabel("Частоand")
        plt.grid(True)
        path = os.path.join(output_path, f"sentiment_distribution_{ticker_symbol}_{interval}.png")
        results["sentiment"] = _safe_savefig(fig, path, show_plot)

    ti_path = plot_technical_indicators(
        df, price_col="close",
        output_path=output_path,
        ticker_symbol=ticker_symbol,
        interval=interval,
        show_plot=show_plot
    )
    results["technical_indicators"] = ti_path

    return results


# --------------------------
# Plot model performance
# --------------------------
def plot_model_performance(
        y_test: np.ndarray,
        predictions: dict,
        test_index: Optional[pd.Index] = None,
        output_path: str = "charts",
        interval: str = "1d",
        show_plot: bool = False
) -> Optional[str]:
    if y_test is None or len(y_test) == 0:
        logger.warning("[PlotModel] Порожнandй y_test")
        return None

