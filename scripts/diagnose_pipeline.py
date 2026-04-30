#!/usr/bin/env python3
"""
Quick diagnostic script to check pipeline configuration and data.
"""

import sys
from pathlib import Path
import json

def check_test_mode_configs():
    """Check for test mode configurations."""
    print("\n" + "=" * 80)
    print("🔍 CHECKING TEST MODE CONFIGS")
    print("=" * 80)
    
    issues = []
    
    # Check main_database config.json
    main_db_config = Path("data/colab/accumulated/main_database/config.json")
    if main_db_config.exists():
        print("❌ FOUND: config.json in main_database (should NOT exist for full mode)")
        with open(main_db_config, 'r') as f:
            config = json.load(f)
        print(f"   Content: {json.dumps(config, indent=2)}")
        issues.append("config.json in main_database")
    else:
        print("✅ OK: No config.json in main_database")
    
    # Check outputs runtime_params.json
    outputs_dir = Path("outputs")
    if outputs_dir.exists():
        runtime_params = list(outputs_dir.rglob("runtime_params.json"))
        if runtime_params:
            print(f"\n⚠️ FOUND: {len(runtime_params)} runtime_params.json in outputs:")
            for rp in runtime_params:
                print(f"   - {rp}")
                with open(rp, 'r') as f:
                    params = json.load(f)
                if params.get('test_ticker'):
                    print(f"     test_ticker: {params['test_ticker']}")
                    issues.append(f"runtime_params.json with test_ticker={params['test_ticker']}")
        else:
            print("\n✅ OK: No runtime_params.json in outputs")
    
    return issues


def check_assets_config():
    """Check assets configuration."""
    print("\n" + "=" * 80)
    print("📊 CHECKING ASSETS CONFIG")
    print("=" * 80)
    
    assets_config = Path("src/config/assets.yaml")
    if not assets_config.exists():
        print("❌ ERROR: assets.yaml not found!")
        return ["assets.yaml missing"]
    
    try:
        import yaml
        with open(assets_config, 'r') as f:
            config = yaml.safe_load(f)
        
        active_preset = config['assets']['active_preset']
        tickers = config['assets']['presets'][active_preset]['tickers']
        
        print(f"✅ Active preset: {active_preset}")
        print(f"✅ Tickers ({len(tickers)}): {', '.join(tickers)}")
        
        if len(tickers) < 10:
            print(f"⚠️ WARNING: Only {len(tickers)} tickers (expected 10)")
            return [f"Only {len(tickers)} tickers"]
        
        return []
    except Exception as e:
        print(f"❌ ERROR reading assets.yaml: {e}")
        return [f"assets.yaml error: {e}"]


def check_data_files():
    """Check data files."""
    print("\n" + "=" * 80)
    print("📁 CHECKING DATA FILES")
    print("=" * 80)
    
    issues = []
    
    main_db = Path("data/colab/accumulated/main_database")
    if not main_db.exists():
        print("⚠️ WARNING: main_database directory not found")
        return ["main_database missing"]
    
    # Check features.parquet
    features_file = main_db / "features.parquet"
    if features_file.exists():
        try:
            import pandas as pd
            df = pd.read_parquet(features_file)
            tickers = df['ticker'].unique() if 'ticker' in df.columns else []
            print(f"✅ features.parquet: {len(df)} rows, {len(df.columns)} columns")
            print(f"   Tickers ({len(tickers)}): {', '.join(sorted(tickers))}")
            
            if len(tickers) == 1:
                print(f"❌ ERROR: Only 1 ticker found (expected 10)")
                issues.append(f"Only 1 ticker in features: {tickers[0]}")
            elif len(tickers) < 10:
                print(f"⚠️ WARNING: Only {len(tickers)} tickers (expected 10)")
                issues.append(f"Only {len(tickers)} tickers in features")
        except Exception as e:
            print(f"❌ ERROR reading features.parquet: {e}")
            issues.append(f"features.parquet error: {e}")
    else:
        print("⚠️ WARNING: features.parquet not found")
        issues.append("features.parquet missing")
    
    # Check targets.parquet
    targets_file = main_db / "targets.parquet"
    if targets_file.exists():
        try:
            import pandas as pd
            df = pd.read_parquet(targets_file)
            tickers = df['ticker'].unique() if 'ticker' in df.columns else []
            target_cols = [c for c in df.columns if c.startswith('target_')]
            print(f"✅ targets.parquet: {len(df)} rows, {len(target_cols)} target columns")
            print(f"   Tickers ({len(tickers)}): {', '.join(sorted(tickers))}")
            
            if len(tickers) == 1:
                print(f"❌ ERROR: Only 1 ticker found (expected 10)")
                issues.append(f"Only 1 ticker in targets: {tickers[0]}")
        except Exception as e:
            print(f"❌ ERROR reading targets.parquet: {e}")
            issues.append(f"targets.parquet error: {e}")
    else:
        print("⚠️ WARNING: targets.parquet not found")
        issues.append("targets.parquet missing")
    
    return issues


def check_logs():
    """Check recent logs."""
    print("\n" + "=" * 80)
    print("📋 CHECKING RECENT LOGS")
    print("=" * 80)
    
    logs_dir = Path("logs")
    if not logs_dir.exists():
        print("⚠️ WARNING: logs directory not found")
        return []
    
    # Find most recent pipeline log
    pipeline_logs = sorted(logs_dir.glob("pipeline_*.log"), key=lambda x: x.stat().st_mtime, reverse=True)
    
    if not pipeline_logs:
        print("⚠️ WARNING: No pipeline logs found")
        return []
    
    latest_log = pipeline_logs[0]
    print(f"📄 Latest log: {latest_log.name}")
    
    # Check for key indicators
    with open(latest_log, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
    
    # Check mode
    if "🧪 TEST MODE" in content:
        print("⚠️ Last run was in TEST MODE")
    elif "📊 FULL MODE" in content:
        print("✅ Last run was in FULL MODE")
    else:
        print("❓ Mode unclear from logs")
    
    # Check for errors
    error_count = content.count("ERROR")
    warning_count = content.count("WARNING")
    
    print(f"   Errors: {error_count}")
    print(f"   Warnings: {warning_count}")
    
    # Check for specific issues
    if "KeyError: 'datetime'" in content:
        print("❌ Found: KeyError 'datetime'")
    if "Only 1 ticker" in content or "test_ticker.*AMD" in content:
        print("❌ Found: Test mode active (only AMD)")
    if ">10% NaN values" in content:
        print("⚠️ Found: High NaN values")
    
    return []


def main():
    """Main diagnostic."""
    print("=" * 80)
    print("🔧 PIPELINE DIAGNOSTIC TOOL")
    print("=" * 80)
    
    all_issues = []
    
    # Run checks
    all_issues.extend(check_test_mode_configs())
    all_issues.extend(check_assets_config())
    all_issues.extend(check_data_files())
    all_issues.extend(check_logs())
    
    # Summary
    print("\n" + "=" * 80)
    print("📊 DIAGNOSTIC SUMMARY")
    print("=" * 80)
    
    if not all_issues:
        print("✅ No issues found! Pipeline configuration looks good.")
        print("\n💡 Ready to run:")
        print("   python run_hybrid_pipeline.py --mode prepare")
        return 0
    else:
        print(f"❌ Found {len(all_issues)} issue(s):")
        for i, issue in enumerate(all_issues, 1):
            print(f"   {i}. {issue}")
        
        print("\n💡 Recommended action:")
        print("   python scripts/run_full_pipeline.py prepare")
        print("\n📚 For more help:")
        print("   See TROUBLESHOOTING.md")
        return 1


if __name__ == "__main__":
    sys.exit(main())
