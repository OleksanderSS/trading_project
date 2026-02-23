# experiments/compare_layers.py

import pandas as pd
from datetime import datetime, timedelta
import argparse
import itertools
from core.pipeline_helpers import run_full_pipeline
from utils.metrics import extract_core_metrics
from utils.logger import ProjectLogger
from config.config import TICKERS, TIME_FRAMES
from config.feature_layers import FEATURE_LAYERS

logger = ProjectLogger.get_logger("CompareLayers")

#  Геnotруємо all комбandнацandї шарandв (до 3 одночасно)
all_layers = list(FEATURE_LAYERS.keys())
layer_sets = []
for r in range(1, 4):  # комбandнацandї по 1, 2, 3 шари
    for combo in itertools.combinations(all_layers, r):
        layer_sets.append(list(combo))

#  Аргументи командного рядка for гнучких дат
parser = argparse.ArgumentParser()
parser.add_argument("--days", type=int, default=365, help="Скandльки днandв наforд брати")
args = parser.parse_args()

today = datetime.utcnow()
start_date = today - timedelta(days=args.days)
end_date = today

results_summary = []

#  Проганяємо all тикери, andймфрейми and комбandнацandї шарandв
for ticker in TICKERS.keys():
    for tf in TIME_FRAMES:
        for layers in layer_sets:
            logger.info(f"[SEARCH] Тестую {ticker} на {tf} with шарами: {layers}")
            try:
                signals, features_dict, avg_sentiment, metrics_summary, news_count, trigger_signals = run_full_pipeline(
                    trader=None,
                    tickers=[ticker],
                    time_frames=[tf],
                    models_dict=None,
                    thresholds=None,
                    window=3,
                    use_cache=True,
                    force_refresh=False,
                    preferred_base_tf="1d",
                    gdelt_cache_path=None,
                    simulate=False,
                    feature_layers=layers,
                    start_date=start_date,
                    end_date=end_date
                )

                metrics = metrics_summary.get(ticker, {}).get(tf, {})
                core = extract_core_metrics(metrics)

                results_summary.append({
                    "ticker": ticker,
                    "time_frame": tf,
                    "layers": " + ".join(layers),
                    "MAE": core.get("mae"),
                    "RMSE": core.get("rmse"),
                    "R2": core.get("r2"),
                    "Sharpe": core.get("sharpe")
                })

            except Exception as e:
                logger.error(f"[ERROR] Error for {ticker} {tf} {layers}: {e}")
                results_summary.append({
                    "ticker": ticker,
                    "time_frame": tf,
                    "layers": " + ".join(layers),
                    "MAE": None,
                    "RMSE": None,
                    "R2": None,
                    "Sharpe": None
                })

#  Виводимо andблицю реwithульandтandв
df_results = pd.DataFrame(results_summary)
print("\n[DATA] Порandвняння шарandв фandчей:\n")
print(df_results.to_markdown(index=False))