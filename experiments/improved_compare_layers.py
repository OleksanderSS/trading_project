# experiments/improved_compare_layers.py

"""
Improved version of compare_layers experiment with better documentation,
parallel processing, and enhanced output capabilities
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import argparse
import itertools
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Import base experiment class
from experiments.experiment_base import BaseExperiment

# Import existing dependencies
from core.pipeline_helpers import run_full_pipeline
from utils.metrics import extract_core_metrics
from utils.logger import ProjectLogger
from config.config import TICKERS, TIME_FRAMES
from config.feature_layers import FEATURE_LAYERS

logger = ProjectLogger.get_logger("ImprovedCompareLayers")

class CompareLayersExperiment(BaseExperiment):
    """Improved experiment for comparing feature layer combinations"""
    
    def __init__(self):
        super().__init__("CompareLayers")
        self.all_layers = list(FEATURE_LAYERS.keys())
        self.layer_sets = []
        
        # Generate all combinations (1-3 layers)
        for r in range(1, 4):
            for combo in itertools.combinations(self.all_layers, r):
                self.layer_sets.append(list(combo))
    
    def get_metrics(self) -> List[str]:
        """Get list of metrics this experiment produces"""
        return ["MAE", "RMSE", "R2", "Sharpe"]
    
    def run_single_test(self, test_params: Dict) -> Dict:
        """Run single test case"""
        ticker = test_params['ticker']
        tf = test_params['time_frame']
        layers = test_params['layers']
        start_date = test_params['start_date']
        end_date = test_params['end_date']
        
        logger.info(f"[SEARCH] Testing {ticker} on {tf} with layers: {layers}")
        
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

            return {
                "ticker": ticker,
                "time_frame": tf,
                "layers": " + ".join(layers),
                "layer_count": len(layers),
                "MAE": core.get("mae"),
                "RMSE": core.get("rmse"),
                "R2": core.get("r2"),
                "Sharpe": core.get("sharpe"),
                "news_count": news_count,
                "success": True,
                "error": None
            }

        except Exception as e:
            logger.error(f"[ERROR] Error for {ticker} {tf} {layers}: {e}")
            return {
                "ticker": ticker,
                "time_frame": tf,
                "layers": " + ".join(layers),
                "layer_count": len(layers),
                "MAE": None,
                "RMSE": None,
                "R2": None,
                "Sharpe": None,
                "news_count": None,
                "success": False,
                "error": str(e)
            }
    
    def run_experiment(self, days: int = 365, parallel: bool = True, max_workers: int = None) -> List[Dict]:
        """Run the complete experiment"""
        
        # Calculate date range
        today = datetime.utcnow()
        start_date = today - timedelta(days=days)
        end_date = today
        
        # Generate all test cases
        test_cases = []
        for ticker in TICKERS.keys():
            for tf in TIME_FRAMES:
                for layers in self.layer_sets:
                    test_cases.append({
                        'ticker': ticker,
                        'time_frame': tf,
                        'layers': layers,
                        'start_date': start_date,
                        'end_date': end_date
                    })
        
        logger.info(f"[SEARCH] Generated {len(test_cases)} test cases")
        
        # Run tests
        if parallel and max_workers > 1:
            logger.info(f"[START] Running in parallel with {max_workers} workers")
            results = self._run_parallel(test_cases, max_workers)
        else:
            logger.info(" Running sequentially")
            results = [self.run_single_test(case) for case in test_cases]
        
        # Filter successful results
        successful_results = [r for r in results if r.get('success', False)]
        logger.info(f"[OK] {len(successful_results)}/{len(results)} tests completed successfully")
        
        return successful_results
    
    def _run_parallel(self, test_cases: List[Dict], max_workers: int) -> List[Dict]:
        """Run tests in parallel"""
        results = []
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_case = {
                executor.submit(self.run_single_test, case): case 
                for case in test_cases
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_case):
                case = future_to_case[future]
                try:
                    result = future.result()
                    results.append(result)
                    
                    # Progress indicator
                    completed = len(results)
                    total = len(test_cases)
                    progress = (completed / total) * 100
                    logger.info(f"[DATA] Progress: {completed}/{total} ({progress:.1f}%)")
                    
                except Exception as e:
                    logger.error(f"[ERROR] Parallel execution error: {e}")
                    results.append({
                        "ticker": case['ticker'],
                        "time_frame": case['time_frame'],
                        "layers": " + ".join(case['layers']),
                        "success": False,
                        "error": str(e)
                    })
        
        return results
    
    def generate_visualizations(self):
        """Generate visualization plots"""
        if not self.results:
            return
        
        df = pd.DataFrame(self.results)
        
        # Create plots directory
        plots_dir = self.output_dir / "plots"
        plots_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 1. Performance by layer count
        plt.figure(figsize=(12, 8))
        layer_performance = df.groupby('layer_count').agg({
            'MAE': 'mean',
            'RMSE': 'mean',
            'R2': 'mean',
            'Sharpe': 'mean'
        }).reset_index()
        
        x = layer_performance['layer_count']
        plt.subplot(2, 2, 1)
        plt.plot(x, layer_performance['MAE'], 'o-')
        plt.title('MAE vs Layer Count')
        plt.xlabel('Number of Layers')
        plt.ylabel('MAE')
        
        plt.subplot(2, 2, 2)
        plt.plot(x, layer_performance['RMSE'], 'o-')
        plt.title('RMSE vs Layer Count')
        plt.xlabel('Number of Layers')
        plt.ylabel('RMSE')
        
        plt.subplot(2, 2, 3)
        plt.plot(x, layer_performance['R2'], 'o-')
        plt.title('R vs Layer Count')
        plt.xlabel('Number of Layers')
        plt.ylabel('R')
        
        plt.subplot(2, 2, 4)
        plt.plot(x, layer_performance['Sharpe'], 'o-')
        plt.title('Sharpe vs Layer Count')
        plt.xlabel('Number of Layers')
        plt.ylabel('Sharpe Ratio')
        
        plt.tight_layout()
        plot_path = plots_dir / f"{self.name}_{timestamp}_performance.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"[DATA] Performance plot saved to {plot_path}")
        
        # 2. Best combinations heatmap
        if len(df) > 0:
            # Create pivot table for best Sharpe ratios
            pivot_data = df[df['success'] == True].copy()
            
            if len(pivot_data) > 0:
                # Get best combination for each ticker/timeframe
                best_combinations = pivot_data.loc[pivot_data.groupby(['ticker', 'time_frame'])['Sharpe'].idxmax()]
                
                # Create heatmap data
                heatmap_data = best_combinations.pivot_table(
                    values='Sharpe', 
                    index='ticker', 
                    columns='time_frame', 
                    aggfunc='mean'
                )
                
                plt.figure(figsize=(10, 6))
                sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='RdYlGn')
                plt.title('Best Sharpe Ratio by Ticker/Timeframe')
                plt.tight_layout()
                
                heatmap_path = plots_dir / f"{self.name}_{timestamp}_heatmap.png"
                plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
                plt.close()
                
                logger.info(f"[DATA] Heatmap saved to {heatmap_path}")
    
    def generate_enhanced_report(self) -> str:
        """Generate enhanced markdown report with analysis"""
        if not self.results:
            return "No results to report"
            
        df = pd.DataFrame(self.results)
        successful_df = df[df['success'] == True]
        
        report = f"# {self.name} Enhanced Report\n\n"
        report += f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        report += f"**Total Tests:** {len(df)}\n"
        report += f"**Successful:** {len(successful_df)} ({len(successful_df)/len(df)*100:.1f}%)\n\n"
        
        # Best results
        report += "## [BEST] Best Results\n\n"
        for metric in self.get_metrics():
            best = self.get_best_result(metric, higher_is_better=(metric != "MAE" and metric != "RMSE"))
            if best:
                report += f"- **Best {metric}:** {best['layers']} ({best[metric]:.4f})\n"
        report += "\n"
        
        # Layer count analysis
        if len(successful_df) > 0:
            layer_analysis = successful_df.groupby('layer_count').agg({
                'MAE': ['mean', 'std'],
                'RMSE': ['mean', 'std'],
                'R2': ['mean', 'std'],
                'Sharpe': ['mean', 'std']
            }).round(4)
            
            report += "## [DATA] Performance by Layer Count\n\n"
            report += layer_analysis.to_markdown()
            report += "\n"
        
        # Detailed results
        report += "##  Detailed Results\n\n"
        report += successful_df.to_markdown(index=False)
        
        return report
    
    def save_enhanced_report(self):
        """Save enhanced markdown report"""
        report = self.generate_enhanced_report()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = self.output_dir / f"{self.name}_{timestamp}_enhanced_report.md"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        logger.info(f" Enhanced report saved to {report_path}")
        return report_path

def main():
    """Main function with improved CLI"""
    parser = argparse.ArgumentParser(description="Compare feature layer combinations")
    parser.add_argument("--days", type=int, default=365, help="Days to look back")
    parser.add_argument("--parallel", action="store_true", help="Run in parallel")
    parser.add_argument("--workers", type=int, default=mp.cpu_count(), help="Number of parallel workers")
    parser.add_argument("--output-dir", default="experiments/results", help="Output directory")
    parser.add_argument("--visualize", action="store_true", help="Generate visualizations")
    
    args = parser.parse_args()
    
    # Create and run experiment
    experiment = CompareLayersExperiment()
    experiment.output_dir = Path(args.output_dir)
    
    # Run experiment
    results = experiment.run_experiment(
        days=args.days,
        parallel=args.parallel,
        max_workers=args.workers
    )
    
    # Generate visualizations
    if args.visualize:
        experiment.generate_visualizations()
    
    # Save enhanced report
    experiment.save_enhanced_report()
    
    # Print summary
    print(f"\n[TARGET] Experiment completed!")
    print(f"[DATA] Results: {len(results)} successful tests")
    print(f" Output: {experiment.output_dir}")

if __name__ == "__main__":
    main()
