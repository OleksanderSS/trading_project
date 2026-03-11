# src/experiments/compare_layers.py

"""
Experiment for comparing feature layer combinations and ensemble methods
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import itertools
from concurrent.futures import ProcessPoolExecutor, as_completed
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Dict, Any, Optional

# Import base experiment class
from devtools.experimentation.base import BaseExperiment

# Updated imports for current project structure
from src.pipeline.pipeline_orchestrator import PipelineOrchestrator
from src.metrics.financial_metrics import calculate_performance_metrics
from src.core.logging.logger import ProjectLogger
from src.config.unified_config_manager import UnifiedConfigManager
from src.meta_learning.experience_diary import ExperienceDiary
from src.ensembling.ensemble import ensemble_forecast

logger = ProjectLogger.get_logger("CompareLayers")

class CompareLayersExperiment(BaseExperiment):
    """Experiment for comparing feature layer combinations and ensemble methods across market regimes"""
    
    def __init__(self):
        super().__init__("CompareLayers")
        self.config_manager = UnifiedConfigManager()
        self.exp_config = self.config_manager.get_config('experiments', {}).get('compare_layers', {})
        self.tickers = self.config_manager.get_config('assets', {}).get('tickers', [])
        self.time_frames = self.config_manager.get_config('assets', {}).get('time_frames', [])
        self.feature_layers_cfg = self.config_manager.get_config('features', {}).get('feature_layers', {})
        
        self.ensemble_methods = ["weighted", "mean", "median"]
        self.market_regimes = ["Bull", "Bear", "Sideways", "All"]
        
        self.all_layers = list(self.feature_layers_cfg.keys())
        self.layer_sets = []
        
        # Generate all combinations (1-3 layers)
        for r in range(1, 4):
            for combo in itertools.combinations(self.all_layers, r):
                self.layer_sets.append(list(combo))
                
        self.diary = ExperienceDiary()
        logger.info(f"CompareLayersExperiment initialized with {len(self.layer_sets)} layer combinations.")

    def get_metrics(self) -> List[str]:
        """Get list of metrics this experiment produces"""
        return ["Total Return", "Sharpe", "Max Drawdown", "Win Rate"]
    
    def run_single_test(self, test_params: dict) -> dict:
        """Run single test case involving specific layers, ensemble method and regime"""
        ticker = test_params['ticker']
        tf = test_params['time_frame']
        layers = test_params['layers']
        method = test_params['ensemble_method']
        regime = test_params['market_regime']
        start_date = test_params['start_date']
        end_date = test_params['end_date']
        
        logger.info(f"[SEARCH] Testing {ticker} | {tf} | Regime: {regime} | Method: {method} | Layers: {layers}")
        
        try:
            orchestrator = PipelineOrchestrator(self.config_manager)
            
            # Run pipeline logic restricted to specific layers
            # Note: We simulate the pipeline's internal logic here for experimentation
            results = orchestrator.run_incremental_pipeline(
                tickers=[ticker],
                start_date=start_date,
                end_date=end_date,
                feature_layers=layers
            )

            # Extract predictions and apply chosen ensemble method
            # Assuming results contain model predictions in a structured way
            # In a real scenario, we'd pull from models_predict
            raw_preds = results.get('predictions', {}).get(ticker, {})
            
            # Simple simulation of ensemble for the experiment
            ensemble_result, stats = ensemble_forecast(
                model_predictions=raw_preds,
                method=method,
                market_regime=regime if regime != "All" else None
            )

            # Calculate performance based on ensemble signals
            # Assuming 'returns' are available in the pipeline results
            actual_returns = results.get('market_data', {}).get(ticker, {}).get('returns', pd.Series())
            
            if len(ensemble_result) > 0 and not actual_returns.empty:
                # Align length
                common_len = min(len(ensemble_result), len(actual_returns))
                strategy_returns = ensemble_result[-common_len:] * actual_returns[-common_len:]
                metrics = calculate_performance_metrics(strategy_returns)
            else:
                metrics = {}

            return {
                "ticker": ticker,
                "time_frame": tf,
                "regime": regime,
                "ensemble_method": method,
                "layers": " + ".join(layers),
                "layer_count": len(layers),
                "Total Return": metrics.get("total_return"),
                "Sharpe": metrics.get("sharpe_ratio"),
                "Max Drawdown": metrics.get("max_drawdown"),
                "Win Rate": metrics.get("win_rate"),
                "success": True,
                "error": None
            }

        except Exception as e:
            logger.error(f"[ERROR] Experiment failed for {ticker} {tf} {layers}: {e}")
            return {
                "ticker": ticker, "time_frame": tf, "regime": regime, 
                "ensemble_method": method, "layers": " + ".join(layers),
                "success": False, "error": str(e)
            }
    
    def run_experiment(self, **kwargs) -> List[dict]:
        """Run the complete experiment across all dimensions"""
        days = self.exp_config.get('days', 365)
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=days)
        
        test_cases = []
        for ticker in self.tickers:
            for tf in self.time_frames:
                for regime in self.market_regimes:
                    for method in self.ensemble_methods:
                        for layers in self.layer_sets:
                            test_cases.append({
                                'ticker': ticker, 'time_frame': tf, 'market_regime': regime,
                                'ensemble_method': method, 'layers': layers,
                                'start_date': start_date, 'end_date': end_date
                            })
        
        logger.info(f"[SEARCH] Generated {len(test_cases)} combinations to test.")
        
        if self.exp_config.get('parallel', True):
            results = self._run_parallel(test_cases, self.exp_config.get('workers', 4))
        else:
            results = [self.run_single_test(case) for case in test_cases]
        
        successful_results = [r for r in results if r.get('success', False)]
        self.results = successful_results
        
        # Save results to ExperienceDiary
        self._save_to_diary(successful_results)
        
        return successful_results
    
    def _run_parallel(self, test_cases: List[dict], max_workers: int) -> List[dict]:
        results = []
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_to_case = {executor.submit(self.run_single_test, case): case for case in test_cases}
            for future in as_completed(future_to_case):
                try:
                    results.append(future.result())
                except Exception as e:
                    logger.error(f"Parallel test failed: {e}")
        return results

    def _save_to_diary(self, results: List[dict]):
        """Identifies best configuration per ticker/tf/regime and saves to Meta-Learning"""
        df = pd.DataFrame(results)
        if df.empty: return

        # Find best combination based on Sharpe
        idx = df.groupby(['ticker', 'time_frame', 'regime'])['Sharpe'].idxmax()
        best_configs = df.loc[idx]

        for _, row in best_configs.iterrows():
            experience = {
                "layers": row['layers'],
                "ensemble_method": row['ensemble_method'],
                "sharpe": row['Sharpe'],
                "total_return": row['Total Return'],
                "timestamp": datetime.now().isoformat()
            }
            self.diary.add_entry(
                ticker=row['ticker'],
                regime=row['regime'],
                event_type="optimization_result",
                data=experience
            )
        logger.info(f"Saved {len(best_configs)} best configurations to ExperienceDiary.")

    def generate_visualizations(self):
        """Generate visualization plots with missing data handling"""
        if not self.results: return
        df = pd.DataFrame(self.results).dropna(subset=['Sharpe'])
        if df.empty: return
        
        plots_dir = self.output_dir / "plots"
        plots_dir.mkdir(exist_ok=True, parents=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Performance by Ensemble Method and Regime
        plt.figure(figsize=(12, 6))
        sns.boxplot(data=df, x='regime', y='Sharpe', hue='ensemble_method')
        plt.title('Sharpe Ratio by Market Regime and Ensemble Method')
        plt.savefig(plots_dir / f"ensemble_regime_perf_{timestamp}.png")
        plt.close()

        # Best layers heatmap (for 'All' regime)
        all_df = df[df['regime'] == 'All']
        if not all_df.empty:
            pivot = all_df.pivot_table(index='ticker', columns='layers', values='Sharpe', aggfunc='max')
            plt.figure(figsize=(14, 8))
            sns.heatmap(pivot, annot=True, cmap='RdYlGn', fmt=".2f")
            plt.title('Best Sharpe per Layer Combination')
            plt.tight_layout()
            plt.savefig(plots_dir / f"layers_heatmap_{timestamp}.png")
            plt.close()

def main():
    experiment = CompareLayersExperiment()
    experiment.run()
    experiment.generate_visualizations()

if __name__ == "__main__":
    main()