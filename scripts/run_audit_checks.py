#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run comprehensive audit checks on the trading system.

Usage:
    python scripts/run_audit_checks.py
    python scripts/run_audit_checks.py --check data
    python scripts/run_audit_checks.py --check all
"""

import argparse
import sys
from pathlib import Path

# Fix Windows console encoding
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
from src.core.logging.logger import ProjectLogger
from src.data.quality import check_data_freshness, check_temporal_alignment
from src.data.management.data_manager import DataManager
from src.config.unified_config_manager import UnifiedConfigManager

logger = ProjectLogger.get_logger("AuditChecker")


def check_data_layer():
    """Run DATA LAYER audit checks."""
    logger.info("=" * 80)
    logger.info("🔍 DATA LAYER AUDIT")
    logger.info("=" * 80)
    
    results = {
        'passed': 0,
        'warnings': 0,
        'failed': 0,
        'checks': []
    }
    
    try:
        # Initialize config and data manager
        config_manager = UnifiedConfigManager()
        data_manager = DataManager(config_manager)
        
        # Check 1: Data Freshness
        logger.info("\n📊 Check 1: Data Freshness")
        logger.info("-" * 40)
        
        # Try different tables and timestamp columns
        tables_to_check = [
            ('market_data_raw', 'datetime'),
            ('enriched_features', 'hash'),  # hash might be timestamp-based
            ('raw_data', 'hash')
        ]
        
        freshness_checked = False
        for table_name, ts_col in tables_to_check:
            df = data_manager.fetch_data_from_table(table_name)
            
            if df is not None and not df.empty:
                # If hash column, try to decode it or use row count as proxy
                if ts_col == 'hash':
                    logger.warning(f"⚠️ Table '{table_name}' uses hash instead of timestamp")
                    logger.info(f"   Rows: {len(df)}, Columns: {len(df.columns)}")
                    logger.warning("   Cannot check freshness without timestamp column")
                    results['warnings'] += 1
                    freshness_checked = True
                    break
                elif ts_col in df.columns:
                    freshness_result = check_data_freshness(
                        df,
                        timestamp_column=ts_col,
                        warning_threshold_hours=1.0,
                        error_threshold_hours=24.0
                    )
                    
                    results['checks'].append((f'data_freshness_{table_name}', freshness_result))
                    
                    if freshness_result['status'] == 'OK':
                        results['passed'] += 1
                        logger.info(f"✅ PASS: {freshness_result['message']}")
                    elif freshness_result['status'] == 'WARNING':
                        results['warnings'] += 1
                        logger.warning(f"⚠️ WARNING: {freshness_result['message']}")
                    else:
                        results['failed'] += 1
                        logger.error(f"❌ FAIL: {freshness_result['message']}")
                    
                    freshness_checked = True
                    break
        
        if not freshness_checked:
            logger.warning("⚠️ SKIP: No suitable table with timestamp found")
            results['warnings'] += 1
        
        # Check 2: Temporal Alignment
        logger.info("\n📊 Check 2: Temporal Alignment")
        logger.info("-" * 40)
        
        # Load raw data
        raw_df = data_manager.fetch_data_from_table('raw_data')
        
        if raw_df is not None and not raw_df.empty:
            # Check if we have news data
            news_df = None
            if 'news_sentiment' in raw_df.columns or 'news_count' in raw_df.columns:
                logger.info("News features detected in data")
                # For now, just log that we should check this
                logger.warning("⚠️ Manual verification needed: Check news timestamp alignment")
                results['warnings'] += 1
            else:
                logger.info("✅ No news features detected")
                results['passed'] += 1
        else:
            logger.warning("⚠️ SKIP: No raw_data table found")
            results['warnings'] += 1
        
        # Check 3: Database Connection Health
        logger.info("\n📊 Check 3: Database Connection Health")
        logger.info("-" * 40)
        
        try:
            tables = data_manager.get_all_table_names()
            logger.info(f"✅ PASS: Database accessible, {len(tables)} tables found")
            logger.info(f"   Tables: {', '.join(tables)}")
            results['passed'] += 1
            results['checks'].append(('db_connection', {'status': 'OK', 'tables': tables}))
        except Exception as e:
            logger.error(f"❌ FAIL: Database connection error: {e}")
            results['failed'] += 1
            results['checks'].append(('db_connection', {'status': 'ERROR', 'error': str(e)}))
        
    except Exception as e:
        logger.error(f"❌ DATA LAYER AUDIT FAILED: {e}", exc_info=True)
        results['failed'] += 1
    
    return results


def check_feature_layer():
    """Run FEATURE LAYER audit checks."""
    logger.info("\n" + "=" * 80)
    logger.info("🔍 FEATURE LAYER AUDIT")
    logger.info("=" * 80)
    
    results = {
        'passed': 0,
        'warnings': 0,
        'failed': 0,
        'checks': []
    }
    
    try:
        config_manager = UnifiedConfigManager()
        data_manager = DataManager(config_manager)
        
        # Check 1: Feature Drift Detection
        logger.info("\n📊 Check 1: Feature Drift Detection")
        logger.info("-" * 40)
        
        try:
            from src.monitoring.feature_drift_monitor import FeatureDriftMonitor, EVIDENTLY_AVAILABLE
            
            if EVIDENTLY_AVAILABLE:
                logger.info("✅ PASS: Evidently AI available")
                logger.info("   Feature drift monitoring enabled")
                results['passed'] += 1
            else:
                logger.warning("⚠️ WARNING: Evidently AI not installed")
                logger.warning("   Install with: pip install evidently")
                results['warnings'] += 1
        except ImportError:
            logger.warning("⚠️ WARNING: FeatureDriftMonitor not found")
            results['warnings'] += 1
        
        # Check 2: Redundant Features
        logger.info("\n📊 Check 2: Redundant Features")
        logger.info("-" * 40)
        
        enriched_df = data_manager.fetch_data_from_table('enriched_features')
        
        if enriched_df is not None and not enriched_df.empty:
            # Check correlation
            numeric_cols = enriched_df.select_dtypes(include=['number']).columns
            if len(numeric_cols) > 1:
                corr_matrix = enriched_df[numeric_cols].corr().abs()
                
                # Find high correlations (> 0.95)
                high_corr_pairs = []
                for i in range(len(corr_matrix.columns)):
                    for j in range(i+1, len(corr_matrix.columns)):
                        if corr_matrix.iloc[i, j] > 0.95:
                            high_corr_pairs.append((
                                corr_matrix.columns[i],
                                corr_matrix.columns[j],
                                corr_matrix.iloc[i, j]
                            ))
                
                if high_corr_pairs:
                    logger.warning(f"⚠️ WARNING: Found {len(high_corr_pairs)} highly correlated feature pairs (>0.95)")
                    for col1, col2, corr in high_corr_pairs[:5]:  # Show first 5
                        logger.warning(f"   {col1} <-> {col2}: {corr:.3f}")
                    results['warnings'] += 1
                else:
                    logger.info("✅ PASS: No highly correlated features found")
                    results['passed'] += 1
            else:
                logger.warning("⚠️ SKIP: Not enough numeric columns")
                results['warnings'] += 1
        else:
            logger.warning("⚠️ SKIP: No enriched_features table")
            results['warnings'] += 1
        
    except Exception as e:
        logger.error(f"❌ FEATURE LAYER AUDIT FAILED: {e}", exc_info=True)
        results['failed'] += 1
    
    return results


def check_model_layer():
    """Run MODEL LAYER audit checks."""
    logger.info("\n" + "=" * 80)
    logger.info("🔍 MODEL LAYER AUDIT")
    logger.info("=" * 80)
    
    results = {
        'passed': 0,
        'warnings': 0,
        'failed': 0,
        'checks': []
    }
    
    # Check if analysis modules exist
    logger.info("\n📊 Check 1: Analysis Modules")
    logger.info("-" * 40)
    
    modules = [
        'src.models.analysis.baseline_dominance_detector',
        'src.models.analysis.regime_winner_analyzer',
        'src.models.analysis.overfitting_detector'
    ]
    
    for module_name in modules:
        try:
            __import__(module_name)
            logger.info(f"✅ PASS: {module_name} available")
            results['passed'] += 1
        except ImportError as e:
            logger.error(f"❌ FAIL: {module_name} not found: {e}")
            results['failed'] += 1
    
    return results


def check_risk_layer():
    """Run RISK LAYER audit checks."""
    logger.info("\n" + "=" * 80)
    logger.info("🔍 RISK LAYER AUDIT")
    logger.info("=" * 80)
    
    results = {
        'passed': 0,
        'warnings': 0,
        'failed': 0,
        'checks': []
    }
    
    try:
        # Check 1: Kill-Switch Tests
        logger.info("\n📊 Check 1: Kill-Switch Logic Testing")
        logger.info("-" * 40)
        
        try:
            # Run kill-switch tests
            import subprocess
            result = subprocess.run(
                ['python', '-m', 'pytest', 'tests/test_risk_manager.py::TestKillSwitch', '-v'],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode == 0:
                logger.info("✅ PASS: Kill-switch tests passed")
                results['passed'] += 1
            else:
                logger.error("❌ FAIL: Kill-switch tests failed")
                results['failed'] += 1
        except FileNotFoundError:
            logger.warning("⚠️ WARNING: tests/test_risk_manager.py not found")
            results['warnings'] += 1
        except Exception as e:
            logger.error(f"❌ FAIL: Error running tests: {e}")
            results['failed'] += 1
        
        # Check 2: Exposure Limits Tests
        logger.info("\n📊 Check 2: Exposure Limits Verification")
        logger.info("-" * 40)
        
        try:
            result = subprocess.run(
                ['python', '-m', 'pytest', 'tests/test_risk_manager.py::TestExposureLimits', '-v'],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode == 0:
                logger.info("✅ PASS: Exposure limits tests passed")
                results['passed'] += 1
            else:
                logger.error("❌ FAIL: Exposure limits tests failed")
                results['failed'] += 1
        except Exception as e:
            logger.error(f"❌ FAIL: Error running tests: {e}")
            results['failed'] += 1
        
        # Check 3: Volatility Scaling Tests
        logger.info("\n📊 Check 3: Volatility Scaling")
        logger.info("-" * 40)
        
        try:
            result = subprocess.run(
                ['python', '-m', 'pytest', 'tests/test_risk_manager.py::TestVolatilityScaling', '-v'],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode == 0:
                logger.info("✅ PASS: Volatility scaling tests passed")
                results['passed'] += 1
            else:
                logger.error("❌ FAIL: Volatility scaling tests failed")
                results['failed'] += 1
        except Exception as e:
            logger.error(f"❌ FAIL: Error running tests: {e}")
            results['failed'] += 1
        
        # Check 4: RiskManager Module
        logger.info("\n📊 Check 4: RiskManager Module Availability")
        logger.info("-" * 40)
        
        try:
            from src.risk.risk_manager import RiskManager, Position, RiskLevel
            logger.info("✅ PASS: RiskManager module available")
            logger.info(f"   Classes: RiskManager, Position, RiskLevel")
            results['passed'] += 1
        except ImportError as e:
            logger.error(f"❌ FAIL: Cannot import RiskManager: {e}")
            results['failed'] += 1
        
    except Exception as e:
        logger.error(f"❌ RISK LAYER AUDIT FAILED: {e}", exc_info=True)
        results['failed'] += 1
    
    return results


def print_summary(all_results):
    """Print audit summary."""
    logger.info("\n" + "=" * 80)
    logger.info("📊 AUDIT SUMMARY")
    logger.info("=" * 80)
    
    total_passed = sum(r['passed'] for r in all_results.values())
    total_warnings = sum(r['warnings'] for r in all_results.values())
    total_failed = sum(r['failed'] for r in all_results.values())
    total_checks = total_passed + total_warnings + total_failed
    
    logger.info(f"\nTotal Checks: {total_checks}")
    logger.info(f"✅ Passed:    {total_passed} ({total_passed/total_checks*100:.1f}%)")
    logger.info(f"⚠️ Warnings:  {total_warnings} ({total_warnings/total_checks*100:.1f}%)")
    logger.info(f"❌ Failed:    {total_failed} ({total_failed/total_checks*100:.1f}%)")
    
    if total_failed > 0:
        logger.error("\n❌ AUDIT FAILED: Critical issues detected")
        return False
    elif total_warnings > 0:
        logger.warning("\n⚠️ AUDIT PASSED WITH WARNINGS: Review recommended")
        return True
    else:
        logger.info("\n✅ AUDIT PASSED: All checks successful")
        return True


def main():
    parser = argparse.ArgumentParser(description='Run audit checks on trading system')
    parser.add_argument(
        '--check',
        choices=['data', 'feature', 'model', 'risk', 'all'],
        default='all',
        help='Which layer to check'
    )
    args = parser.parse_args()
    
    logger.info("🚀 Starting Trading System Audit")
    logger.info(f"Check scope: {args.check}")
    
    all_results = {}
    
    if args.check in ['data', 'all']:
        all_results['data'] = check_data_layer()
    
    if args.check in ['feature', 'all']:
        all_results['feature'] = check_feature_layer()
    
    if args.check in ['model', 'all']:
        all_results['model'] = check_model_layer()
    
    if args.check in ['risk', 'all']:
        all_results['risk'] = check_risk_layer()
    
    success = print_summary(all_results)
    
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
