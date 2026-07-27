import pandas as pd
import numpy as np
from pathlib import Path
from typing import Any, Callable
from src.main.modes.base import BaseMode
from src.core.logging.logger import ProjectLogger
from src.analytics.arena.arena_battle import TradingModelArena
import joblib

class HistoricalEventReplayMode(BaseMode):
    """
    Mode for replaying historical events (crises, drops, surges) across different tickers and timeframes.
    Analyzes model performance on these specific isolated events to find meta-patterns.
    """
    def __init__(self, mode_config: dict[str, Any], config_manager):
        super().__init__(config_manager)
        self.mode_config = mode_config
        self.global_config = config_manager.merged_config
        import logging
        self.logger = logging.getLogger("HistoricalReplay")
        self.arena = TradingModelArena()
        
    def _find_events(self, df: pd.DataFrame, event_type: str, threshold: float = -0.05, window: int = 5) -> list[int]:
        """
        Finds indices in the dataframe where a specific event occurred.
        """
        events = []
        if 'close' not in df.columns:
            return events
            
        if event_type == 'sharp_drop':
            # Find rolling returns over 'window' bars that are worse than 'threshold'
            rolling_returns = df['close'].pct_change(window)
            # Find local minimums to avoid clustering the same event
            for i in range(window, len(df)):
                if rolling_returns.iloc[i] < threshold:
                    # Check if it's the worst in the local neighborhood to avoid duplicates
                    local_min = rolling_returns.iloc[max(0, i-window):min(len(df), i+window)].min()
                    if rolling_returns.iloc[i] == local_min and (not events or i - events[-1] > window):
                        events.append(i)
        elif event_type == 'surge':
            rolling_returns = df['close'].pct_change(window)
            for i in range(window, len(df)):
                if rolling_returns.iloc[i] > threshold:
                    local_max = rolling_returns.iloc[max(0, i-window):min(len(df), i+window)].max()
                    if rolling_returns.iloc[i] == local_max and (not events or i - events[-1] > window):
                        events.append(i)
                        
        return events

    def run(self, *args, **kwargs) -> Any:
        self.logger.info("--- Starting HISTORICAL EVENT REPLAY Mode ---")
        
        # 1. Load historical features
        features_path = Path("d:/trading_project/data/processed/features/features.parquet")
        if not features_path.exists():
            self.logger.error(f"Features file not found at {features_path}")
            return
            
        self.logger.info("Loading historical features...")
        df = pd.read_parquet(features_path)
        
        # 2. Prepare model loading
        trained_models_dir = Path(self.global_config.get("paths", {}).get("models", "d:/trading_project/data/trained_models"))
        all_model_paths = list(trained_models_dir.glob("model_*_target_*.joblib"))
        
        if not all_model_paths:
            self.logger.error("No models found to test!")
            return
            
        self.logger.info(f"Identified {len(all_model_paths)} candidate models.")
        
        # 3. Parameters
        event_type = self.mode_config.get('event_type', 'sharp_drop')
        threshold = self.mode_config.get('threshold', -0.05)
        tickers = list(set([m.stem.split('_')[1] for m in all_model_paths if len(m.stem.split('_')) > 1]))
        self.logger.info(f"Running replay on all {len(tickers)} available tickers.")
        timeframes = ['15m', '60m']
        context_bars_before = self.mode_config.get('context_bars_before', 20)
        context_bars_after = self.mode_config.get('context_bars_after', 10)
        
        # We will collect meta-analysis results
        results = []
        
        # Filter dataframe for fast execution
        df = df[df['ticker'].isin(tickers) & df['interval'].isin(timeframes)]
        
        # 4. Group by ticker and interval
        for (ticker, interval), group in df.groupby(['ticker', 'interval']):
            group = group.sort_values('datetime').reset_index(drop=True)
            
            # Find events
            event_indices = self._find_events(group, event_type, threshold)
            self.logger.info(f"Found {len(event_indices)} '{event_type}' events for {ticker} on {interval} timeframe.")
            
            if not event_indices:
                continue
            
            # Lazy load models for this ticker/interval
            active_models = {}
            for mpath in all_model_paths:
                mname = mpath.stem
                if f"_{ticker}_" in mname and "knn" not in mname and "svm" not in mname:
                    try:
                        active_models[mname] = joblib.load(mpath)
                    except Exception as e:
                        self.logger.warning(f"Failed to load model '{mname}' from {mpath}: {e}")
                        
            if not active_models:
                continue
            
            # Batch processing for extreme speedup
            valid_event_indices = [idx for idx in event_indices if idx >= context_bars_before and idx + context_bars_after < len(group)]
            if not valid_event_indices:
                continue
                
            batch_features = group.iloc[valid_event_indices].copy()
            
            for model_name, model in active_models.items():
                try:
                    # 1. Align features
                    expected_cols = None
                    if hasattr(model, 'feature_names_in_'):
                        expected_cols = model.feature_names_in_
                    elif hasattr(model, 'feature_cols'):
                        expected_cols = model.feature_cols
                    elif hasattr(model, 'model'):
                        if hasattr(model.model, 'feature_names_in_'):
                            expected_cols = model.model.feature_names_in_
                        elif hasattr(model.model, 'feature_names_'):
                            expected_cols = model.model.feature_names_
                        elif hasattr(model.model, 'feature_name_'):
                            expected_cols = model.model.feature_name_()
                    
                    if expected_cols is not None:
                        missing_cols = [c for c in expected_cols if c not in batch_features.columns]
                        if missing_cols:
                            df_missing = pd.DataFrame(0.0, index=batch_features.index, columns=missing_cols)
                            temp_features = pd.concat([batch_features, df_missing], axis=1)
                        else:
                            temp_features = batch_features
                            
                        numeric_features = temp_features[expected_cols].copy()
                    else:
                        numeric_features = batch_features.copy()
                        
                    # 2. Coerce to numeric
                    for col in numeric_features.columns:
                        if not pd.api.types.is_numeric_dtype(numeric_features[col]):
                            numeric_features[col] = pd.to_numeric(numeric_features[col], errors='coerce').fillna(0.0)
                            
                    # 3. Predict ALL events at once
                    preds = model.predict(numeric_features)
                    
                    probs = None
                    if hasattr(model, 'predict_proba'):
                        try:
                            probs = model.predict_proba(numeric_features)
                        except:
                            pass
                    
                    # 4. Record results for each event
                    for i, idx in enumerate(valid_event_indices):
                        event_row = group.iloc[idx]
                        event_date = event_row['datetime']
                        pre_event_rsi = event_row.get('rsi_14', 50)
                        pre_event_vol = event_row.get('atr_14', 0)
                        
                        # New context slices
                        trend_state = 1 if event_row.get('sma_20', 0) > event_row.get('sma_50', 0) else 0
                        drop_severity = event_row.get('return_1d', 0) # Assuming return_1d captures the drop
                        
                        # Find actual return in the next context_bars_after
                        future_window = group.iloc[idx : idx + context_bars_after]
                        entry_price = event_row['close']
                        exit_price = future_window.iloc[-1]['close']
                        actual_return = (exit_price - entry_price) / entry_price
                        
                        pred = preds[i]
                        direction_correct = (pred * actual_return) > 0
                        
                        prob_val = 1.0
                        if probs is not None:
                            # Typically predict_proba returns [prob_class_0, prob_class_1]
                            # Or if it's regression, it won't have predict_proba. 
                            prob_val = float(np.max(probs[i])) if len(probs.shape) > 1 else float(probs[i])
                        
                        results.append({
                            'ticker': ticker,
                            'interval': interval,
                            'event_date': event_date,
                            'model': model_name,
                            'pre_event_rsi': pre_event_rsi,
                            'pre_event_vol': pre_event_vol,
                            'trend_state': trend_state,
                            'drop_severity': drop_severity,
                            'actual_return': actual_return,
                            'prediction': pred,
                            'predicted_probability': prob_val,
                            'direction_correct': direction_correct
                        })
                except Exception as e:
                    self.logger.warning(f"Prediction failed for model '{model_name}' ({ticker}/{interval}): {e}", exc_info=True)

        # 5. Meta-Analysis
        if results:
            results_df = pd.DataFrame(results)
            self.logger.info(f"Successfully ran {len(results_df)} predictions across all events.")
            
            # Save results for deep pattern mining
            out_path = Path('d:/trading_project/data/processed/meta_analysis_results.parquet')
            results_df.to_parquet(out_path, index=False)
            self.logger.info(f"Saved detailed results to {out_path} for deep pattern mining.")
            
            # Group by model and interval to see where models perform best
            analysis = results_df.groupby(['model', 'interval'])['direction_correct'].mean().reset_index()
            self.logger.info("--- Meta-Analysis: Accuracy by Model and Timeframe ---")
            for _, row in analysis.iterrows():
                self.logger.info(f"Model: {row['model']}, Timeframe: {row['interval']}, Accuracy: {row['direction_correct']:.2%}")
                
            # Pattern mining: Does high RSI before a crash mean models predict better?
            high_rsi_results = results_df[results_df['pre_event_rsi'] > 60]
            if not high_rsi_results.empty:
                high_rsi_acc = high_rsi_results['direction_correct'].mean()
                self.logger.info(f"Pattern found: When RSI > 60 before {event_type}, average model accuracy is {high_rsi_acc:.2%}")
                
            low_rsi_results = results_df[results_df['pre_event_rsi'] < 40]
            if not low_rsi_results.empty:
                low_rsi_acc = low_rsi_results['direction_correct'].mean()
                self.logger.info(f"Pattern found: When RSI < 40 before {event_type}, average model accuracy is {low_rsi_acc:.2%}")
        else:
            self.logger.warning("No successful predictions were made. Models likely failed due to mismatched feature columns.")
            
        self.logger.info("--- HISTORICAL EVENT REPLAY Completed ---")
        return results
