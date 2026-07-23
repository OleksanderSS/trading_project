import json
import logging
from pathlib import Path

import pandas as pd
import numpy as np

from src.config.unified_config_manager import UnifiedConfigManager
from src.core.logging.logger import ProjectLogger

logger = ProjectLogger.get_logger(__name__)

class MetaPatternMiner:
    def __init__(self):
        self.config = UnifiedConfigManager()
        self.data_dir = Path(self.config.get('paths.processed_data', 'data/processed'))
        self.results_path = self.data_dir / 'meta_analysis_results.parquet'
        self.rules_path = self.data_dir / 'routing_rules.json'

    def run(self):
        if not self.results_path.exists():
            logger.error(f"Results file not found: {self.results_path}")
            return

        logger.info(f"Loading results from {self.results_path}")
        df = pd.read_parquet(self.results_path)
        logger.info(f"Loaded {len(df)} prediction records.")

        # 1. Parse model names to extract target and algo
        def extract_target(mname):
            try:
                if '_target_' in mname:
                    parts = mname.split('_target_')[1].split('_')
                    return '_'.join(parts[:-1])
                return 'unknown'
            except:
                return 'unknown'

        df['target_type'] = df['model'].apply(extract_target)

        # 2. Extract temporal context
        df['event_date'] = pd.to_datetime(df['event_date'])
        df['hour'] = df['event_date'].dt.hour
        df['day_of_week'] = df['event_date'].dt.dayofweek

        # 3. Classify Assets
        def classify_asset(ticker):
            if 'USD' in ticker:
                return 'crypto'
            elif ticker in ['AAPL', 'MSFT', 'NVDA', 'GOOGL', 'META', 'AMZN']:
                return 'tech'
            elif ticker in ['TSLA', 'F', 'GM']:
                return 'auto'
            else:
                return 'other'
                
        df['asset_class'] = df['ticker'].apply(classify_asset)

        # 4. Bin continuous variables for context
        try:
            df['rsi_bin'] = pd.qcut(df['pre_event_rsi'], q=5, labels=['VeryLow', 'Low', 'Medium', 'High', 'VeryHigh'], duplicates='drop')
        except Exception as e:
            logger.warning(f"Failed to bin rsi, using fallback: {e}")
            df['rsi_bin'] = 'Medium'
            
        try:
            df['vol_bin'] = pd.qcut(df['pre_event_vol'], q=5, labels=['VeryLow', 'Low', 'Medium', 'High', 'VeryHigh'], duplicates='drop')
        except Exception as e:
            logger.warning(f"Failed to bin vol, using fallback: {e}")
            df['vol_bin'] = 'Medium'
            
        try:
            df['severity_bin'] = pd.qcut(df['drop_severity'], q=3, labels=['Severe', 'Moderate', 'Mild'], duplicates='drop')
        except Exception as e:
            logger.warning(f"Failed to bin severity, using fallback: {e}")
            df['severity_bin'] = 'Moderate'

        rules = {
            'sharp_drop': {
                'description': 'Rules applied during sharp drop events',
                'target_modifiers': {},
                'rsi_modifiers': {},
                'volatility_modifiers': {},
                'asset_class_modifiers': {},
                'hour_modifiers': {},
                'day_modifiers': {},
                'trend_modifiers': {},
                'probability_calibration': {}
            }
        }

        def build_modifiers(group_col, target_dict, threshold_high=0.65, threshold_low=0.40):
            acc_dict = df.groupby(group_col, observed=False)['direction_correct'].mean().to_dict()
            for key, acc in acc_dict.items():
                if acc > threshold_high:
                    target_dict[str(key)] = 1.2
                elif acc < threshold_low:
                    target_dict[str(key)] = 0.8
                else:
                    target_dict[str(key)] = 1.0

        # Build Modifiers
        target_acc = df.groupby('target_type')['direction_correct'].mean().to_dict()
        for tgt, acc in target_acc.items():
            if acc > 0.65:
                rules['sharp_drop']['target_modifiers'][tgt] = 1.5
            elif acc < 0.35:
                rules['sharp_drop']['target_modifiers'][tgt] = 0.0
            else:
                rules['sharp_drop']['target_modifiers'][tgt] = 1.0
                
        build_modifiers('rsi_bin', rules['sharp_drop']['rsi_modifiers'])
        build_modifiers('vol_bin', rules['sharp_drop']['volatility_modifiers'])
        build_modifiers('asset_class', rules['sharp_drop']['asset_class_modifiers'])
        build_modifiers('hour', rules['sharp_drop']['hour_modifiers'])
        build_modifiers('day_of_week', rules['sharp_drop']['day_modifiers'])
        build_modifiers('trend_state', rules['sharp_drop']['trend_modifiers'])

        # Probability Calibration
        if 'predicted_probability' in df.columns:
            try:
                df['prob_bin'] = pd.cut(df['predicted_probability'], bins=[0, 0.6, 0.7, 0.8, 0.9, 1.0], labels=['<60%', '60-70%', '70-80%', '80-90%', '>90%'])
                prob_acc = df.groupby('prob_bin', observed=False)['direction_correct'].mean().to_dict()
                for pbin, acc in prob_acc.items():
                    if not pd.isna(acc):
                        rules['sharp_drop']['probability_calibration'][str(pbin)] = float(acc)
            except Exception as e:
                logger.warning(f"Could not calibrate probabilities: {e}")

        # Save rules
        with open(self.rules_path, 'w', encoding='utf-8') as f:
            json.dump(rules, f, indent=4)
        
        logger.info(f"Saved dynamic routing rules to {self.rules_path}")

if __name__ == "__main__":
    miner = MetaPatternMiner()
    miner.run()
