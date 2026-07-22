import json
import logging
from pathlib import Path
import pandas as pd
import numpy as np
from src.config.unified_config_manager import UnifiedConfigManager

logger = logging.getLogger("PanicCorrelationAnalyzer")

class PanicCorrelationAnalyzer:
    """
    Analyzes asset correlations specifically during 'sharp_drop' or crisis periods
    to identify true safe havens when correlations typically trend towards 1.0.
    """
    def __init__(self):
        self.config = UnifiedConfigManager()
        self.data_dir = Path(self.config.get('paths.processed_data', 'data/processed'))
        self.output_path = self.data_dir / 'safe_havens.json'
        
    def run_analysis(self) -> dict:
        logger.info("Starting Panic Correlation Analysis...")
        
        # Load accumulated features from all tickers
        colab_data_dir = Path("data/colab/accumulated/main_database")
        
        if not colab_data_dir.exists():
            logger.warning(f"No data found at {colab_data_dir}")
            return {}
            
        all_returns = {}
        
        # Find the latest cleaned data file
        cleaned_files = list(colab_data_dir.glob("main_database_stage2_cleaned_data_*.parquet"))
        if not cleaned_files:
            logger.warning("No cleaned data files found.")
            return {}
            
        latest_file = sorted(cleaned_files)[-1]
        logger.info(f"Loading data from {latest_file}")
        
        try:
            df = pd.read_parquet(latest_file)
            if 'ticker' in df.columns and 'close' in df.columns:
                # Pivot to get tickers as columns and dates as index
                if 'date' in df.columns:
                    df = df.set_index('date')
                
                for ticker in df['ticker'].unique():
                    ticker_df = df[df['ticker'] == ticker]
                    returns = ticker_df['close'].pct_change().dropna()
                    all_returns[ticker] = returns
        except Exception as e:
            logger.error(f"Error processing {latest_file}: {e}")
                
        if not all_returns:
            logger.warning("Could not build returns dictionary.")
            return {}
            
        # Align all returns by index
        combined_returns = pd.DataFrame(all_returns).dropna(how='all')
        
        if 'SPY' not in combined_returns.columns:
            logger.warning("SPY benchmark not found. Cannot determine safe havens relative to SPY.")
            spy_proxy = combined_returns.columns[0]
            logger.info(f"Using {spy_proxy} as market proxy.")
        else:
            spy_proxy = 'SPY'
            
        # Identify "Panic" days: days where SPY drops more than 2%
        panic_days = combined_returns[combined_returns[spy_proxy] < -0.02]
        logger.info(f"Identified {len(panic_days)} panic days in historical data.")
        
        if len(panic_days) < 5:
            logger.warning("Not enough panic days to compute robust correlation. Need at least 5.")
            return {}
            
        # Calculate correlation matrix ONLY on panic days
        panic_corr = panic_days.corr(method='spearman')
        
        # Identify Safe Havens (correlation with SPY is <= 0 during panics)
        spy_corrs = panic_corr[spy_proxy].drop(spy_proxy)
        safe_havens = spy_corrs[spy_corrs <= 0.1].sort_values()
        
        results = {
            "market_proxy": spy_proxy,
            "panic_days_count": len(panic_days),
            "safe_havens": safe_havens.to_dict(),
            "highly_correlated_hazards": spy_corrs[spy_corrs > 0.8].to_dict()
        }
        
        # Save results
        try:
            self.output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.output_path, 'w') as f:
                json.dump(results, f, indent=4)
            logger.info(f"Saved safe havens to {self.output_path}")
        except Exception as e:
            logger.error(f"Failed to save safe havens: {e}")
            
        return results

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    analyzer = PanicCorrelationAnalyzer()
    res = analyzer.run_analysis()
    print("Results:", json.dumps(res, indent=2))
