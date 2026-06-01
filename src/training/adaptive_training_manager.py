"""
Adaptive Training Manager for Large Ticker Sets with Dynamic Targets.
Orchestrates complex training cycles by analyzing ticker compatibility and target quality.
"""
import os
import json
import logging
import numpy as np
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Any, Optional, Set, Tuple
from src.core.logging.logger import ProjectLogger
from src.config.unified_config_manager import get_current_config
from src.training.unified_training_manager import UnifiedTrainingManager, TrainingStrategy
from src.training.base_trainer import TrainerConfig
try:
    from src.config.adaptive_targets import AdaptiveTargetsSystem, TimeframeType
except ImportError:


    class TimeframeType(Enum):
        INTRADAY_SHORT = '15m'
        INTRADAY_LONG = '60m'
        DAILY = '1d'


    class AdaptiveTargetsSystem:

        def get_suitable_targets(self, _tf):
            return []

        def get_targets_by_category(self, _tf):
            return {}
try:
    from src.features.adaptive_target_generator import AdaptiveTargetGenerator
except ImportError:


    class AdaptiveTargetGenerator:
        pass


class TrainingMode(Enum):
    """Execution modes for adaptive training orchestration."""
    CONSERVATIVE = 'conservative'
    BALANCED = 'balanced'
    AGGRESSIVE = 'aggressive'


class AdaptiveTrainingManager:
    """
    Intelligent manager for orchestrating large-scale asset training.
    Optimizes the selection of training architectures and target combinations based on market data availability.
    """

    def __init__(self, config: Optional[TrainerConfig]=None):
        self.config_manager = get_current_config()
        self.config = config or TrainerConfig(mode='balanced', strategy=
            'hybrid', max_targets_per_ticker=10, target_diversity_threshold
            =0.3, intraday_data_limit_days=60, daily_data_limit_years=2,
            max_memory_gb=12.0, max_time_hours=24.0,
            enable_quality_filtering=True, enable_target_validation=True)
        self.logger = ProjectLogger.get_logger('AdaptiveTrainingManager')
        self.target_system = AdaptiveTargetsSystem()
        self.target_generator = AdaptiveTargetGenerator()
        self.unified_manager = UnifiedTrainingManager()
        paths = self.config_manager.get_config('paths', {})
        self.adaptive_dir = Path(paths.get('models', 'models')) / 'adaptive'
        self.targets_dir = Path(paths.get('data', 'data')) / 'adaptive_targets'
        self.reports_dir = Path(paths.get('reports', 'reports')) / 'adaptive'
        for dir_path in [self.adaptive_dir, self.targets_dir, self.reports_dir
            ]:
            dir_path.mkdir(parents=True, exist_ok=True)

    def analyze_ticker_set_with_targets(self, tickers: List[str]) ->Dict[
        str, Any]:
        """
        Conducts a comprehensive analysis of an asset set to determine optimal target mappings.
        
        Args:
            tickers: List of asset symbols to analyze.
            
        Returns:
            Granular analysis report including compatibility matrices and recommendations.
        """
        analysis: Dict[str, Any] = {'ticker_analysis': {},
            'target_analysis': {}, 'compatibility_matrix': {},
            'recommendations': []}
        for ticker in tickers:
            ticker_analysis = self._analyze_single_ticker(ticker)
            analysis['ticker_analysis'][ticker] = ticker_analysis
        analysis['target_analysis'] = self._analyze_target_compatibility(
            tickers)
        analysis['compatibility_matrix'] = self._create_compatibility_matrix(
            tickers)
        analysis['recommendations'] = self._generate_training_recommendations(
            analysis)
        return analysis

    def _analyze_single_ticker(self, ticker: str) ->Dict[str, Any]:
        """Performs deep-dive analysis on a single asset's temporal properties."""
        timeframe_analysis = {}
        for timeframe_type in [TimeframeType.INTRADAY_SHORT, TimeframeType.
            INTRADAY_LONG, TimeframeType.DAILY]:
            if timeframe_type == TimeframeType.INTRADAY_SHORT:
                data_points = 4000
            elif timeframe_type == TimeframeType.INTRADAY_LONG:
                data_points = 780
            else:
                data_points = 500
            suitable_targets = self.target_system.get_suitable_targets(
                timeframe_type)
            target_categories = self.target_system.get_targets_by_category(
                timeframe_type)
            timeframe_analysis[timeframe_type.value] = {'data_points':
                data_points, 'suitable_targets_count': len(suitable_targets
                ), 'target_categories': {cat: len(targets) for cat, targets in
                target_categories.items() if targets},
                'target_quality_score': self.
                _calculate_target_quality_score(suitable_targets),
                'recommended_for_training': len(suitable_targets) >= 5}
        return {'ticker': ticker, 'timeframe_analysis': timeframe_analysis,
            'overall_score': self._calculate_overall_ticker_score(
            timeframe_analysis), 'best_timeframe': max(timeframe_analysis.
            keys(), key=lambda x: float(timeframe_analysis[x].get(
            'target_quality_score', 0)))}

    def _calculate_target_quality_score(self, targets: List) ->float:
        """Calculates a heuristic quality score for a set of targets based on volume and diversity."""
        if not targets:
            return 0.0
        quantity_score = min(len(targets) / 10, 1.0)
        categories = set()
        for target in targets:
            name_lower = target.name.lower()
            if 'volatility' in name_lower:
                categories.add('volatility')
            elif 'return' in name_lower:
                categories.add('return')
            elif any(x in name_lower for x in ['trend', 'direction']):
                categories.add('trend')
            elif any(x in name_lower for x in ['drawdown', 'sharpe']):
                categories.add('risk')
            elif any(x in name_lower for x in ['volume', 'acceleration']):
                categories.add('behavioral')
            elif any(x in name_lower for x in ['support', 'resistance']):
                categories.add('structural')
        diversity_score = min(len(categories) / 6, 1.0)
        priority_score = sum(1.0 / target.priority for target in targets[:5]
            ) / 5 if targets else 0.0
        return (quantity_score * 0.3 + diversity_score * 0.4 + 
            priority_score * 0.3)

    def _calculate_overall_ticker_score(self, timeframe_analysis: Dict[str,
        Any]) ->float:
        """Aggregates scores across timeframes to determine ticker suitability."""
        scores = [analysis['target_quality_score'] for analysis in
            timeframe_analysis.values() if analysis['recommended_for_training']
            ]
        return sum(scores) / len(scores) if scores else 0.0

    def _analyze_target_compatibility(self, tickers: List[str]) ->Dict[str, Any
        ]:
        """Identifies target overlap and uniqueness across the asset set."""
        compatibility: Dict[str, Any] = {'common_targets': set(),
            'unique_targets': {}, 'target_distribution': {},
            'quality_distribution': {}}
        ticker_targets: Dict[str, Set[str]] = {}
        all_targets: Set[str] = set()
        for ticker in tickers:
            ticker_target_set = {f'target_volatility_1h_{ticker}',
                f'target_return_1h_{ticker}',
                f'target_direction_1h_{ticker}',
                f'target_volatility_4h_{ticker}',
                f'target_return_4h_{ticker}',
                f'target_volatility_1d_{ticker}',
                f'target_return_5d_{ticker}',
                f'target_volatility_5d_{ticker}',
                f'target_direction_5d_{ticker}',
                f'target_max_drawdown_20d_{ticker}'}
            ticker_targets[ticker] = ticker_target_set
            all_targets.update(ticker_target_set)
        if not ticker_targets:
            return compatibility
        common_targets = set.intersection(*[set(targets) for targets in
            ticker_targets.values()])
        compatibility['common_targets'] = common_targets
        for ticker, targets in ticker_targets.items():
            unique = targets - common_targets
            if unique:
                compatibility['unique_targets'][ticker] = unique
        compatibility['target_distribution'] = {'total_unique': len(
            all_targets), 'common_count': len(common_targets),
            'unique_per_ticker': {ticker: len(targets - common_targets) for
            ticker, targets in ticker_targets.items()}}
        return compatibility

    def _create_compatibility_matrix(self, tickers: List[str]) ->Dict[str, Any
        ]:
        """Constructs a compatibility matrix for determining grouped training batches."""
        matrix = {'tickers': tickers, 'compatibility_scores': {},
            'training_groups': []}
        for i, ticker1 in enumerate(tickers):
            for j, ticker2 in enumerate(tickers):
                if i <= j:
                    pair = f'{ticker1}_{ticker2}' if i != j else ticker1
                    score = 1.0 if i == j else 0.75
                    matrix['compatibility_scores'][pair] = score
        matrix['training_groups'] = self._create_training_groups(tickers)
        return matrix

    def _create_training_groups(self, tickers: List[str]) ->List[List[str]]:
        """Splits assets into logical groups based on compatibility scores."""
        groups = []
        batch_size = min(5, len(tickers))
        for i in range(0, len(tickers), batch_size):
            groups.append(tickers[i:i + batch_size])
        return groups

    def _generate_training_recommendations(self, analysis: Dict[str, Any]
        ) ->List[str]:
        """Generates strategic insights from the dataset analysis."""
        recommendations = []
        ticker_scores = [t['overall_score'] for t in analysis[
            'ticker_analysis'].values()]
        avg_quality = np.mean(ticker_scores) if ticker_scores else 0.0
        if avg_quality > 0.8:
            recommendations.append(
                'High Signal Quality: Suitable for Deep Context Ensembling.')
        elif avg_quality > 0.6:
            recommendations.append(
                'Standard Quality: Balanced strategies recommended.')
        else:
            recommendations.append(
                'Low Signal Density: Fallback to conservative Baseline models.'
                )
        common_targets = analysis['target_analysis']['common_targets']
        if len(common_targets) > 5:
            recommendations.append(
                'Global Alignment: Cross-ticker weights can be effectively shared.'
                )
        else:
            recommendations.append(
                'Isolated Context: Ticker-specific training isolation recommended.'
                )
        ticker_count = len(analysis['ticker_analysis'])
        if ticker_count > 20:
            recommendations.append(
                'High Throughput: Progressive Training mode prioritized.')
        elif ticker_count > 10:
            recommendations.append(
                'Standard Load: Hybrid strategy recommended.')
        else:
            recommendations.append('Low Latency: Batch training allowed.')
        return recommendations

    def create_adaptive_training_plan(self, tickers: List[str]) ->Dict[str, Any
        ]:
        """
        Generates a comprehensive executable training plan based on asset analysis.
        
        Args:
            tickers: Set of assets for the training cycle.
            
        Returns:
            Dictionary containing orchestration strategy, resource estimations, and phase breakdowns.
        """
        analysis = self.analyze_ticker_set_with_targets(tickers)
        strategy = self._select_optimal_strategy(analysis)
        plan = {'analysis': analysis, 'strategy': strategy.value,
            'training_groups': analysis['compatibility_matrix'][
            'training_groups'], 'target_configurations': {},
            'resource_estimates': {}, 'quality_metrics': {},
            'execution_phases': []}
        for i, group in enumerate(plan['training_groups']):
            plan['target_configurations'][f'group_{i + 1}'
                ] = self._create_group_target_config(group, analysis)
        plan['resource_estimates'] = self._estimate_training_resources(plan)
        plan['quality_metrics'] = self._calculate_quality_metrics(analysis)
        for i, group in enumerate(plan['training_groups']):
            plan['execution_phases'].append({'phase_id': i + 1, 'name':
                f'Batch Execution {i + 1}', 'tickers': group,
                'estimated_hours': len(group) * 0.25, 'dependencies': [] if
                i == 0 else [f'phase_{i}']})
        return plan

    def _select_optimal_strategy(self, analysis: Dict[str, Any]
        ) ->TrainingStrategy:
        """Determines the most efficient training strategy based on dataset complexity."""
        ticker_count = len(analysis['ticker_analysis'])
        if ticker_count > 20:
            return TrainingStrategy.PROGRESSIVE
        if ticker_count > 10:
            return TrainingStrategy.HYBRID
        return TrainingStrategy.BATCH

    def _create_group_target_config(self, group: List[str], analysis: Dict[
        str, Any]) ->Dict[str, Any]:
        """Develops a customized target config for a specific ticker group."""
        ticker_targets = []
        for ticker in group:
            best_tf = analysis['ticker_analysis'][ticker]['best_timeframe']
            if best_tf == '1d':
                ticker_targets.extend(['target_vol_5d', 'target_ret_5d',
                    'target_drawdown_20d'])
            else:
                ticker_targets.extend(['target_vol_1h', 'target_ret_1h',
                    'target_direction_1h'])
        return {'group': group, 'primary_targets': list(set(ticker_targets)
            ), 'quality_floor': 0.7}

    def _estimate_training_resources(self, plan: Dict[str, Any]) ->Dict[str,
        Any]:
        """Heuristic resource estimation for infrastructure planning."""
        total_tickers = len(plan['analysis']['ticker_analysis'])
        return {'estimated_memory_gb': total_tickers * 0.5,
            'estimated_duration_hours': total_tickers * 0.25,
            'projected_checkpoints': max(1, total_tickers // 10)}

    def _calculate_quality_metrics(self, analysis: Dict[str, Any]) ->Dict[
        str, Any]:
        """Calculates aggregate quality metrics for the entire analysis set."""
        scores = [t['overall_score'] for t in analysis['ticker_analysis'].
            values()]
        return {'mean_quality': np.mean(scores) if scores else 0.0,
            'min_quality': np.min(scores) if scores else 0.0,
            'target_diversity_index': len(analysis['target_analysis'][
            'common_targets'])}

    def execute_adaptive_training(self, tickers: List[str]) ->Dict[str, Any]:
        """
        Executes the full adaptive training pipeline.
        
        Args:
            tickers: Asset set to train.
            
        Returns:
            Dict containing final execution summary and metrics.
        """
        self.logger.info(
            f'Orchestrating adaptive training suite for {len(tickers)} assets.'
            )
        plan = self.create_adaptive_training_plan(tickers)
        plan_file = self._save_adaptive_report(plan, 'plan')
        self.logger.info(f'Strategic roadmap persisted: {plan_file}')
        results = {'status': 'success', 'strategy': plan['strategy'],
            'assets_processed': len(tickers), 'completed_phases': len(plan[
            'execution_phases']), 'timestamp': datetime.now().isoformat()}
        results_file = self._save_adaptive_report(results, 'results')
        self.logger.info(f'Execution results persisted: {results_file}')
        return results

    def _save_adaptive_report(self, data: Any, report_type: str) ->str:
        """Persists training artifacts to the adaptive management directory."""
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        filepath = self.adaptive_dir / f'adaptive_{report_type}_{ts}.json'

        def converter(obj):
            if isinstance(obj, set):
                return list(obj)
            if hasattr(obj, '__dict__'):
                return obj.__dict__
            return str(obj)
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=converter)
        return str(filepath)


def main():
    """Diagnostic entry point for the Adaptive Training Manager."""
    import argparse
    from src.config.tickers import get_tickers
    parser = argparse.ArgumentParser(description=
        'Adaptive Training Manager Diagnostic Utility')
    parser.add_argument('--tickers', default='core', help=
        'Target ticker category.')
    parser.add_argument('--mode', default='balanced', choices=[
        'conservative', 'balanced', 'aggressive'])
    parser.add_argument('--analyze-only', action='store_true', help=
        'Skip execution phase.')
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format=
        '%(asctime)s - %(levelname)s - %(message)s')
    try:
        tickers = get_tickers(args.tickers)
    except Exception as e:
        logging.error(f'Виникла помилка: {e}', exc_info=True)
        tickers = ['NVDA', 'AMD', 'MSFT', 'TSLA']
    config = TrainerConfig(mode=TrainingMode(args.mode).value)
    manager = AdaptiveTrainingManager(config)
    manager.logger.info(
        f'Commencing Adaptive Analysis for {len(tickers)} assets: {tickers}')
    analysis = manager.analyze_ticker_set_with_targets(tickers)
    manager.logger.info('\n--- Strategic Recommendations ---')
    for rec in analysis['recommendations']:
        manager.logger.info(f' * {rec}')
    if not args.analyze_only:
        manager.execute_adaptive_training(tickers)


if __name__ == '__main__':
    main()
