#!/usr/bin/env python3
"""
Complete Data Pipeline
Unified workflow: Real Data → Synthetic Data → Verification → Ready for Training

Usage:
    python scripts/run_complete_data_pipeline.py --tickers AMD NVDA --days 30
    python scripts/run_complete_data_pipeline.py --mode full
    python scripts/run_complete_data_pipeline.py --mode real-only
    python scripts/run_complete_data_pipeline.py --mode synthetic-only
"""

import sys
import os
import json
import asyncio
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, cast

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config.unified_config_manager import UnifiedConfigManager
from src.core.logging.logger import ProjectLogger

# Import data pipeline components
from scripts.accumulate_real_data import RealDataAccumulator
from scripts.generate_synthetic_data import SyntheticDataGenerator
from scripts.verify_enriched_dataset import EnrichedDatasetVerifier

logger = ProjectLogger.get_logger("CompleteDataPipeline")


class CompleteDataPipeline:
    """
    Unified data pipeline combining:
    1. Real data accumulation (Stages 0-3)
    2. Synthetic data generation (3 types)
    3. Dataset verification
    4. Preparation for training
    """
    
    def __init__(self, config_manager: UnifiedConfigManager):
        self.config_manager = config_manager
        self.real_accumulator = RealDataAccumulator(config_manager)
        self.synthetic_generator = SyntheticDataGenerator(config_manager)
        self.verifier = EnrichedDatasetVerifier(config_manager)
        
        self.results: Dict[str, Any] = {
            'real_data': None,
            'synthetic_data': None,
            'verification': None,
            'summary': None
        }
    
    async def run(
        self,
        mode: str = 'full',
        tickers: Optional[List[str]] = None,
        days_back: int = 30,
        synthetic_types: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Run complete data pipeline
        
        Args:
            mode: Pipeline mode
                - 'full': Real + Synthetic + Verification
                - 'real-only': Only real data accumulation
                - 'synthetic-only': Only synthetic generation
                - 'verify-only': Only verification
            tickers: List of tickers to process
            days_back: Days of historical data
            synthetic_types: Types of synthetic scenarios
        
        Returns:
            Complete pipeline results
        """
        logger.info("=" * 80)
        logger.info("🚀 COMPLETE DATA PIPELINE - Starting")
        logger.info(f"   Mode: {mode}")
        logger.info(f"   Tickers: {tickers or 'from config'}")
        logger.info(f"   Days: {days_back}")
        logger.info("=" * 80)
        
        start_time = datetime.now()
        
        try:
            # Phase 1: Real Data Accumulation
            if mode in ['full', 'real-only']:
                logger.info("\n" + "=" * 80)
                logger.info("📊 PHASE 1: Real Data Accumulation")
                logger.info("=" * 80)
                
                self.results['real_data'] = self.real_accumulator.run(
                    tickers=tickers or [],
                    days_back=days_back
                )
                
                if self.results['real_data'] and self.results['real_data'].get('status') != 'success':
                    logger.error("❌ Real data accumulation failed")
                    return self._generate_error_result("Real data accumulation failed")
                
                logger.info("✅ Phase 1 Complete: Real data accumulated")
            
            # Phase 2: Synthetic Data Generation
            if mode in ['full', 'synthetic-only']:
                logger.info("\n" + "=" * 80)
                logger.info("🎲 PHASE 2: Synthetic Data Generation")
                logger.info("=" * 80)
                
                synthetic_types = synthetic_types or ['typical', 'shock', 'context']
                self.results['synthetic_data'] = self.synthetic_generator.run(
                    scenario_types=synthetic_types
                )
                
                if self.results['synthetic_data'] and self.results['synthetic_data'].get('status') != 'success':
                    logger.error("❌ Synthetic data generation failed")
                    return self._generate_error_result("Synthetic data generation failed")
                
                logger.info("✅ Phase 2 Complete: Synthetic data generated")
            
            # Phase 3: Dataset Verification
            if mode in ['full', 'verify-only']:
                logger.info("\n" + "=" * 80)
                logger.info("🔍 PHASE 3: Dataset Verification")
                logger.info("=" * 80)
                
                self.results['verification'] = self.verifier.run()
                
                if self.results['verification'] and self.results['verification'].get('status') != 'verified':
                    logger.warning("⚠️  Dataset verification found issues")
                else:
                    logger.info("✅ Phase 3 Complete: Dataset verified")
            
            # Phase 4: Generate Summary
            logger.info("\n" + "=" * 80)
            logger.info("📋 PHASE 4: Generating Summary")
            logger.info("=" * 80)
            
            self.results['summary'] = self._generate_summary(start_time)
            
            # Save complete results
            self._save_results()
            
            logger.info("\n" + "=" * 80)
            logger.info("✅ COMPLETE DATA PIPELINE - SUCCESS")
            logger.info("=" * 80)
            
            return self.results
            
        except Exception as e:
            logger.exception(f"❌ Pipeline failed: {e}")
            return self._generate_error_result(str(e))
    
    def _generate_summary(self, start_time: datetime) -> Dict[str, Any]:
        """Generate pipeline summary"""
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        summary = {
            'status': 'success',
            'start_time': start_time.isoformat(),
            'end_time': end_time.isoformat(),
            'duration_seconds': duration,
            'phases_completed': []
        }
        
        # Real data summary
        if self.results['real_data']:
            phases_list = cast(List[str], summary['phases_completed'])
            phases_list.append('real_data')
            summary['real_data_summary'] = {
                'tickers': self.results['real_data'].get('tickers', []),
                'rows_collected': self.results['real_data'].get('collected_rows', 0),
                'rows_enriched': self.results['real_data'].get('enriched_rows', 0),
                'features_count': self.results['real_data'].get('enriched_features', 0)
            }
        
        # Synthetic data summary
        if self.results['synthetic_data']:
            phases_list = cast(List[str], summary['phases_completed'])
            phases_list.append('synthetic_data')
            summary['synthetic_data_summary'] = {
                'typical_scenarios': self.results['synthetic_data'].get('typical_scenarios', 0),
                'shock_scenarios': self.results['synthetic_data'].get('shock_scenarios', 0),
                'context_scenarios': self.results['synthetic_data'].get('context_scenarios', 0),
                'total_scenarios': (
                    self.results['synthetic_data'].get('typical_scenarios', 0) +
                    self.results['synthetic_data'].get('shock_scenarios', 0) +
                    self.results['synthetic_data'].get('context_scenarios', 0)
                )
            }
        
        # Verification summary
        if self.results['verification']:
            phases_list = cast(List[str], summary['phases_completed'])
            phases_list.append('verification')
            if self.results['verification']:
                verification_report = self.results['verification'].get('report', {})
            else:
                verification_report = {}
            summary['verification_summary'] = {
                'tables_found': verification_report.get('tables', {}).get('count', 0),
                'event_series_valid': verification_report.get('event_series_format', {}).get('is_valid_event_series', False),
                'data_integrity_ok': verification_report.get('data_integrity', {}).get('null_percentage', 100) < 10,
                'recommendations': self.results['verification'].get('recommendations', [])
            }
        
        # Overall status
        verification_summary_raw = summary.get('verification_summary') if isinstance(summary, dict) else None
        verification_summary = verification_summary_raw if isinstance(verification_summary_raw, dict) else {}
        phases_completed = cast(list[str], summary.get('phases_completed', []) if isinstance(summary, dict) else [])
        
        summary['ready_for_training'] = (
            len(phases_completed) >= 2 and
            verification_summary.get('event_series_valid', False)
        )
        
        logger.info("\n📊 Pipeline Summary:")
        logger.info(f"   Phases completed: {', '.join(phases_completed)}")
        logger.info(f"  Phases: {', '.join(phases_completed)}")
        logger.info(f"  Ready for training: {summary['ready_for_training']}")
        
        return summary
    
    def _generate_error_result(self, error_message: str) -> Dict[str, Any]:
        """Generate error result"""
        return {
            'status': 'failed',
            'error': error_message,
            'timestamp': datetime.now().isoformat(),
            'results': self.results
        }
    
    def _save_results(self):
        """Save complete pipeline results"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_dir = Path('results/pipeline')
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # Save complete results
        results_file = results_dir / f'complete_pipeline_{timestamp}.json'
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        logger.info(f"✓ Results saved to {results_file}")
        
        # Save summary separately
        if self.results['summary']:
            summary_file = results_dir / f'pipeline_summary_{timestamp}.json'
            with open(summary_file, 'w') as f:
                json.dump(self.results['summary'], f, indent=2, default=str)
            
            logger.info(f"✓ Summary saved to {summary_file}")


async def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Complete data pipeline: Real + Synthetic + Verification'
    )
    parser.add_argument(
        '--mode',
        choices=['full', 'real-only', 'synthetic-only', 'verify-only'],
        default='full',
        help='Pipeline mode'
    )
    parser.add_argument(
        '--tickers',
        nargs='+',
        default=None,
        help='Tickers to process (default: from config)'
    )
    parser.add_argument(
        '--days',
        type=int,
        default=30,
        help='Days of historical data'
    )
    parser.add_argument(
        '--synthetic-types',
        nargs='+',
        choices=['typical', 'shock', 'context'],
        default=['typical', 'shock', 'context'],
        help='Types of synthetic scenarios'
    )
    parser.add_argument(
        '--config-dir',
        default='src/config',
        help='Path to config directory'
    )
    
    args = parser.parse_args()
    
    # Initialize config
    config_manager = UnifiedConfigManager(config_dir=args.config_dir)
    
    # Run pipeline
    pipeline = CompleteDataPipeline(config_manager)
    results = await pipeline.run(
        mode=args.mode,
        tickers=args.tickers,
        days_back=args.days,
        synthetic_types=args.synthetic_types
    )
    
    # Print final results
    logger.info("\n" + "=" * 80)
    logger.info("FINAL RESULTS:")
    logger.info(json.dumps(results.get('summary', results), indent=2, default=str))
    logger.info("=" * 80)
    
    return 0 if results.get('status') == 'success' else 1


if __name__ == '__main__':
    sys.exit(asyncio.run(main()))
