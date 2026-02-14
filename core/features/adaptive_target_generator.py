"""
Adaptive Target Generator
Геnotратор адаптивних andргетandв for рandwithних andймфреймandв
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import json
from datetime import datetime

# Додаємо шлях до проекту
current_dir = Path(__file__).parent.parent.parent
import sys
sys.path.append(str(current_dir))

from config.adaptive_targets import AdaptiveTargetsSystem, TimeframeType, TargetConfig
from config.tickers import get_tickers
from utils.performance_tracker import PerformanceTracker

logger = logging.getLogger("AdaptiveTargetGenerator")

class AdaptiveTargetGenerator:
    """Геnotратор адаптивних andргетandв"""
    
    def __init__(self):
        self.target_system = AdaptiveTargetsSystem()
        self.performance_tracker = PerformanceTracker()
        
    def analyze_data_availability(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Аналandwithувати доступнandсть data
        
        Args:
            df: DataFrame with даними
            
        Returns:
            Dict[str, Any]: Реwithульandти аналandwithу
        """
        analysis = {
            "total_records": len(df),
            "date_range": {
                "start": df.index.min() if hasattr(df.index, 'min') else df['date'].min(),
                "end": df.index.max() if hasattr(df.index, 'max') else df['date'].max()
            },
            "tickers": [],
            "timeframes": [],
            "data_quality": {}
        }
        
        # Аналandwithуємо тandкери
        if 'ticker' in df.columns:
            analysis["tickers"] = df['ticker'].unique().tolist()
            analysis["ticker_count"] = len(analysis["tickers"])
        
        # Виwithначаємо andймфрейм
        if hasattr(df.index, 'freq'):
            freq = df.index.freq
            if freq:
                if '15T' in str(freq):
                    analysis["timeframes"] = ["15m"]
                    analysis["timeframe_type"] = TimeframeType.INTRADAY_SHORT
                elif '60T' in str(freq):
                    analysis["timeframes"] = ["60m"]
                    analysis["timeframe_type"] = TimeframeType.INTRADAY_LONG
                elif 'D' in str(freq):
                    analysis["timeframes"] = ["1d"]
                    analysis["timeframe_type"] = TimeframeType.DAILY
        else:
            # Евристика for кandлькandстю data
            total_records = len(df)
            if total_records > 10000:
                analysis["timeframes"] = ["15m"]
                analysis["timeframe_type"] = TimeframeType.INTRADAY_SHORT
            elif total_records > 1000:
                analysis["timeframes"] = ["60m"]
                analysis["timeframe_type"] = TimeframeType.INTRADAY_LONG
            else:
                analysis["timeframes"] = ["1d"]
                analysis["timeframe_type"] = TimeframeType.DAILY
        
        # Аналandwithуємо якandсть data
        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col in df.columns:
                missing_pct = df[col].isna().sum() / len(df) * 100
                analysis["data_quality"][col] = {
                    "missing_percentage": missing_pct,
                    "is_valid": missing_pct < 5.0
                }
        
        return analysis
    
    def generate_targets_for_dataframe(self, df: pd.DataFrame, 
                                       timeframe: TimeframeType = None) -> pd.DataFrame:
        """
        Згеnotрувати andргети for DataFrame
        
        Args:
            df: DataFrame with даними
            timeframe: Таймфрейм (автовиvalues якщо None)
            
        Returns:
            pd.DataFrame: DataFrame with andргеandми
        """
        # Аналandwithуємо данand
        analysis = self.analyze_data_availability(df)
        
        # Виwithначаємо andймфрейм
        if timeframe is None:
            timeframe = analysis.get("timeframe_type", TimeframeType.DAILY)
        
        logger.info(f"Generating targets for {timeframe.value} timeframe")
        logger.info(f"Data points: {analysis['total_records']}")
        
        # Перевandряємо якandсть data
        invalid_columns = [col for col, info in analysis["data_quality"].items() 
                          if not info["is_valid"]]
        if invalid_columns:
            logger.warning(f"Invalid data quality in columns: {invalid_columns}")
        
        # Геnotруємо andргети
        try:
            target_df = self.target_system.generate_target_matrix(df, timeframe)
            
            # Додаємо меandданand
            metadata = {
                "generation_time": datetime.now().isoformat(),
                "timeframe": timeframe.value,
                "data_points": len(df),
                "targets_generated": len([col for col in target_df.columns if col.startswith('target_')]),
                "target_categories": self._get_target_categories(target_df)
            }
            
            logger.info(f"Generated {metadata['targets_generated']} targets")
            logger.info(f"Target categories: {list(metadata['target_categories'].keys())}")
            
            return target_df
            
        except Exception as e:
            logger.error(f"Error generating targets: {e}")
            return df
    
    def _get_target_categories(self, df: pd.DataFrame) -> Dict[str, int]:
        """Отримати категорandї andргетandв"""
        target_columns = [col for col in df.columns if col.startswith('target_')]
        
        categories = {
            "volatility": 0,
            "price_return": 0,
            "trend": 0,
            "risk": 0,
            "behavioral": 0,
            "structural": 0
        }
        
        for col in target_columns:
            if "volatility" in col:
                categories["volatility"] += 1
            elif "return" in col:
                categories["price_return"] += 1
            elif "trend" in col or "direction" in col:
                categories["trend"] += 1
            elif "drawdown" in col or "sharpe" in col or "var" in col:
                categories["risk"] += 1
            elif "volume" in col or "acceleration" in col:
                categories["behavioral"] += 1
            elif "support" in col or "resistance" in col or "reversion" in col:
                categories["structural"] += 1
        
        return {k: v for k, v in categories.items() if v > 0}
    
    def generate_targets_for_tickers(self, tickers: List[str], 
                                   timeframes: List[str] = None) -> Dict[str, pd.DataFrame]:
        """
        Згеnotрувати andргети for списку тandкерandв
        
        Args:
            tickers: Список тandкерandв
            timeframes: Список andймфреймandв
            
        Returns:
            Dict[str, pd.DataFrame]: DataFrame with andргеandми for кожного тandкера
        """
        if timeframes is None:
            timeframes = ["15m", "60m", "1d"]
        
        results = {}
        
        for ticker in tickers:
            logger.info(f"Processing ticker: {ticker}")
            
            for timeframe in timeframes:
                try:
                    # Тут має бути логandка forванandження data for тandкера
                    # Для whereмонстрацandї створюємо тестовand данand
                    df = self._create_sample_dataframe(ticker, timeframe)
                    
                    # Геnotруємо andргети
                    target_df = self.generate_targets_for_dataframe(df)
                    
                    key = f"{ticker}_{timeframe}"
                    results[key] = target_df
                    
                    logger.info(f"Generated targets for {key}: {len(target_df)} records")
                    
                except Exception as e:
                    logger.error(f"Error processing {ticker}_{timeframe}: {e}")
                    continue
        
        return results
    
    def _create_sample_dataframe(self, ticker: str, timeframe: str) -> pd.DataFrame:
        """Create тестовий DataFrame"""
        np.random.seed(42)
        
        if timeframe == "15m":
            periods = 4000  # ~60 днandв
            freq = "15T"
        elif timeframe == "60m":
            periods = 780    # ~60 днandв
            freq = "60T"
        else:  # 1d
            periods = 500    # ~2 роки
            freq = "D"
        
        # Створюємо дати
        dates = pd.date_range(start="2023-01-01", periods=periods, freq=freq)
        
        # Геnotруємо цandни (симуляцandя)
        price = 100 + np.cumsum(np.random.normal(0, 0.01, periods))
        high = price * (1 + np.abs(np.random.normal(0, 0.005, periods)))
        low = price * (1 - np.abs(np.random.normal(0, 0.005, periods)))
        open_price = price + np.random.normal(0, 0.1, periods)
        volume = np.random.randint(1000000, 10000000, periods)
        
        df = pd.DataFrame({
            'date': dates,
            'ticker': ticker,
            'open': open_price,
            'high': high,
            'low': low,
            'close': price,
            'volume': volume
        })
        
        df.set_index('date', inplace=True)
        
        return df
    
    def save_target_results(self, results: Dict[str, pd.DataFrame], 
                          output_dir: str = "data/targets") -> Dict[str, str]:
        """
        Зберегти реwithульandти andргетandв
        
        Args:
            results: Реwithульandти andргетandв
            output_dir: Директорandя for withбереження
            
        Returns:
            Dict[str, str]: Шляхи до withбережених fileandв
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        saved_files = {}
        
        for key, df in results.items():
            filename = f"targets_{key}.parquet"
            filepath = output_path / filename
            
            try:
                df.to_parquet(filepath)
                saved_files[key] = str(filepath)
                logger.info(f"Saved {key}: {len(df)} records to {filepath}")
            except Exception as e:
                logger.error(f"Error saving {key}: {e}")
        
        # Зберandгаємо меandданand
        metadata = {
            "generation_time": datetime.now().isoformat(),
            "total_files": len(saved_files),
            "files": {k: {"path": v, "records": len(results[k])} for k, v in saved_files.items()},
            "target_system_version": "1.0"
        }
        
        metadata_path = output_path / "target_generation_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Metadata saved to {metadata_path}")
        
        return saved_files
    
    def create_target_summary_report(self, results: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """
        Create пandдсумковий withвandт
        
        Args:
            results: Реwithульandти andргетandв
            
        Returns:
            Dict[str, Any]: Звandт
        """
        summary = {
            "generation_summary": {
                "total_datasets": len(results),
                "total_records": sum(len(df) for df in results.values()),
                "target_columns": set(),
                "target_categories": {},
                "timeframes": {},
                "tickers": set()
            },
            "quality_metrics": {},
            "recommendations": []
        }
        
        # Аналandwithуємо кожен даandсет
        for key, df in results.items():
            ticker, timeframe = key.split('_')
            
            summary["generation_summary"]["tickers"].add(ticker)
            summary["generation_summary"]["timeframes"][timeframe] = summary["generation_summary"]["timeframes"].get(timeframe, 0) + 1
            
            # Аналandwithуємо andргети
            target_columns = [col for col in df.columns if col.startswith('target_')]
            summary["generation_summary"]["target_columns"].update(target_columns)
            
            # Аналandwithуємо якandсть
            valid_targets = 0
            for col in target_columns:
                missing_pct = df[col].isna().sum() / len(df) * 100
                if missing_pct < 20:  # Допустимо до 20% пропускandв
                    valid_targets += 1
            
            summary["quality_metrics"][key] = {
                "total_targets": len(target_columns),
                "valid_targets": valid_targets,
                "data_quality": valid_targets / len(target_columns) if target_columns else 0
            }
        
        # Конвертуємо sets в lists
        summary["generation_summary"]["tickers"] = list(summary["generation_summary"]["tickers"])
        summary["generation_summary"]["target_columns"] = list(summary["generation_summary"]["target_columns"])
        
        # Аналandwithуємо категорandї andргетandв
        all_target_columns = summary["generation_summary"]["target_columns"]
        for col in all_target_columns:
            if "volatility" in col:
                summary["generation_summary"]["target_categories"]["volatility"] = summary["generation_summary"]["target_categories"].get("volatility", 0) + 1
            elif "return" in col:
                summary["generation_summary"]["target_categories"]["price_return"] = summary["generation_summary"]["target_categories"].get("price_return", 0) + 1
            elif "trend" in col or "direction" in col:
                summary["generation_summary"]["target_categories"]["trend"] = summary["generation_summary"]["target_categories"].get("trend", 0) + 1
            elif "drawdown" in col or "sharpe" in col or "var" in col:
                summary["generation_summary"]["target_categories"]["risk"] = summary["generation_summary"]["target_categories"].get("risk", 0) + 1
            elif "volume" in col or "acceleration" in col:
                summary["generation_summary"]["target_categories"]["behavioral"] = summary["generation_summary"]["target_categories"].get("behavioral", 0) + 1
            elif "support" in col or "resistance" in col or "reversion" in col:
                summary["generation_summary"]["target_categories"]["structural"] = summary["generation_summary"]["target_categories"].get("structural", 0) + 1
        
        # Геnotруємо рекомендацandї
        avg_quality = np.mean([metrics["data_quality"] for metrics in summary["quality_metrics"].values()])
        
        if avg_quality > 0.8:
            summary["recommendations"].append("High data quality - suitable for model training")
        elif avg_quality > 0.6:
            summary["recommendations"].append("Moderate data quality - consider data cleaning")
        else:
            summary["recommendations"].append("Low data quality - extensive data cleaning required")
        
        if len(summary["generation_summary"]["target_categories"]) > 4:
            summary["recommendations"].append("Good target diversity - suitable for multi-objective models")
        else:
            summary["recommendations"].append("Limited target diversity - consider adding more target types")
        
        return summary

def main():
    """Основна функцandя for тестування"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Adaptive Target Generator')
    parser.add_argument('--tickers', default='core', help='Ticker category or list')
    parser.add_argument('--timeframes', nargs='+', default=['15m', '60m', '1d'], help='Timeframes')
    parser.add_argument('--save', action='store_true', help='Save results')
    parser.add_argument('--output-dir', default='data/targets', help='Output directory')
    
    args = parser.parse_args()
    
    # Налаштування logging
    logging.basicConfig(level=logging.INFO, 
                       format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Отримуємо тandкери
    try:
        from config.tickers import get_tickers
        if args.tickers == 'core':
            tickers = get_tickers('core')
        elif args.tickers == 'all':
            tickers = get_tickers('all')[:5]  # Обмежуємо for тесту
        else:
            tickers = get_tickers(args.tickers)
    except ImportError:
        tickers = ['SPY', 'QQQ', 'NVDA']
    
    # Створюємо геnotратор
    generator = AdaptiveTargetGenerator()
    
    print(f"Generating targets for {len(tickers)} tickers: {tickers}")
    print(f"Timeframes: {args.timeframes}")
    
    # Геnotруємо andргети
    results = generator.generate_targets_for_tickers(tickers, args.timeframes)
    
    # Створюємо withвandт
    summary = generator.create_target_summary_report(results)
    
    print(f"\n=== Target Generation Summary ===")
    print(f"Total datasets: {summary['generation_summary']['total_datasets']}")
    print(f"Total records: {summary['generation_summary']['total_records']}")
    print(f"Unique targets: {len(summary['generation_summary']['target_columns'])}")
    print(f"Target categories: {list(summary['generation_summary']['target_categories'].keys())}")
    print(f"Timeframes: {list(summary['generation_summary']['timeframes'].keys())}")
    print(f"Tickers: {summary['generation_summary']['tickers']}")
    
    print(f"\n=== Quality Metrics ===")
    for key, metrics in summary['quality_metrics'].items():
        print(f"{key}: {metrics['valid_targets']}/{metrics['total_targets']} valid ({metrics['data_quality']:.2%})")
    
    print(f"\n=== Recommendations ===")
    for rec in summary['recommendations']:
        print(f"- {rec}")
    
    # Зберandгаємо реwithульandти
    if args.save:
        saved_files = generator.save_target_results(results, args.output_dir)
        print(f"\n=== Saved Files ===")
        for key, path in saved_files.items():
            print(f"{key}: {path}")
        
        # Зберandгаємо withвandт
        report_path = Path(args.output_dir) / "target_generation_report.json"
        with open(report_path, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"Report saved to: {report_path}")

if __name__ == "__main__":
    main()
