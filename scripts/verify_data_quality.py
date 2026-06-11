#!/usr/bin/env python3
"""
Verify data quality after pipeline run.
Checks for errors, data integrity, and quality metrics.
"""

import sys
from pathlib import Path
import pandas as pd
import duckdb

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.core.logging.logger import ProjectLogger

logger = ProjectLogger.get_logger(__name__)


def check_duckdb_data():
    """Check data quality in DuckDB."""
    db_path = project_root / "data" / "trading_data.duckdb"
    
    if not db_path.exists():
        logger.error(f"Database not found: {db_path}")
        return False
    
    logger.info(f"Checking database: {db_path}")
    
    conn = duckdb.connect(str(db_path))
    
    try:
        # Check tables
        tables = conn.execute("SHOW TABLES").fetchall()
        logger.info(f"Found {len(tables)} tables: {[t[0] for t in tables]}")
        
        # Check enriched_features
        if ('enriched_features',) in tables:
            count = conn.execute("SELECT COUNT(*) FROM enriched_features").fetchone()[0]
            logger.info(f"enriched_features: {count} rows")
            
            # Check for nulls
            cols = conn.execute("DESCRIBE enriched_features").fetchall()
            for col_name, col_type, *_ in cols:
                null_count = conn.execute(f"SELECT COUNT(*) FROM enriched_features WHERE {col_name} IS NULL").fetchone()[0]
                null_pct = (null_count / count * 100) if count > 0 else 0
                if null_pct > 50:
                    logger.warning(f"  Column '{col_name}': {null_pct:.1f}% nulls")
        
        # Check targets
        if ('targets',) in tables:
            count = conn.execute("SELECT COUNT(*) FROM targets").fetchone()[0]
            logger.info(f"targets: {count} rows")
        
        # Check raw_data
        if ('raw_data',) in tables:
            count = conn.execute("SELECT COUNT(*) FROM raw_data").fetchone()[0]
            logger.info(f"raw_data: {count} rows")
        
        return True
        
    except Exception as e:
        logger.error(f"Error checking database: {e}")
        return False
    finally:
        conn.close()


def check_parquet_data():
    """Check data quality in Parquet files."""
    colab_dir = project_root / "data" / "colab" / "accumulated" / "main_database"
    
    if not colab_dir.exists():
        logger.warning(f"Colab directory not found: {colab_dir}")
        return False
    
    logger.info(f"Checking Parquet files in: {colab_dir}")
    
    # Check features
    features_path = colab_dir / "features.parquet"
    if features_path.exists():
        df = pd.read_parquet(features_path)
        logger.info(f"features.parquet: {df.shape[0]} rows, {df.shape[1]} columns")
        
        # Check for high null percentage
        null_pct = (df.isnull().sum() / len(df) * 100)
        high_null_cols = null_pct[null_pct > 50]
        if len(high_null_cols) > 0:
            logger.warning(f"  {len(high_null_cols)} columns with >50% nulls")
            for col in high_null_cols.head(5).index:
                logger.warning(f"    {col}: {null_pct[col]:.1f}% nulls")
    
    # Check targets
    targets_path = colab_dir / "targets.parquet"
    if targets_path.exists():
        df = pd.read_parquet(targets_path)
        logger.info(f"targets.parquet: {df.shape[0]} rows, {df.shape[1]} columns")
    
    return True


def check_log_errors():
    """Check for errors in recent logs."""
    log_dir = project_root / "logs"
    
    if not log_dir.exists():
        logger.warning(f"Log directory not found: {log_dir}")
        return True
    
    # Find most recent log file
    log_files = sorted(log_dir.glob("*.log"), key=lambda p: p.stat().st_mtime, reverse=True)
    
    if not log_files:
        logger.warning("No log files found")
        return True
    
    latest_log = log_files[0]
    logger.info(f"Checking log file: {latest_log.name}")
    
    with open(latest_log, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Count errors
    error_count = content.count(" - ERROR - ")
    warning_count = content.count(" - WARNING - ")
    
    # Count specific errors
    datetime_errors = content.count("Invalid comparison between dtype=datetime")
    news_errors = content.count("Error processing news")
    
    logger.info(f"Log statistics:")
    logger.info(f"  Total ERRORS: {error_count}")
    logger.info(f"  Total WARNINGS: {warning_count}")
    logger.info(f"  Datetime errors: {datetime_errors}")
    logger.info(f"  News processing errors: {news_errors}")
    
    if datetime_errors > 0:
        logger.warning(f"Found {datetime_errors} datetime comparison errors!")
        logger.warning("These errors cause news to be skipped, reducing data quality")
        return False
    
    if news_errors > 100:
        logger.warning(f"Found {news_errors} news processing errors!")
        logger.warning("High error rate may indicate data quality issues")
        return False
    
    return True


def main():
    """Main verification function."""
    logger.info("=" * 80)
    logger.info("DATA QUALITY VERIFICATION")
    logger.info("=" * 80)
    
    results = []
    
    # Check DuckDB
    logger.info("\n1. Checking DuckDB data...")
    results.append(("DuckDB", check_duckdb_data()))
    
    # Check Parquet files
    logger.info("\n2. Checking Parquet files...")
    results.append(("Parquet", check_parquet_data()))
    
    # Check logs for errors
    logger.info("\n3. Checking logs for errors...")
    results.append(("Logs", check_log_errors()))
    
    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("SUMMARY")
    logger.info("=" * 80)
    
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        logger.info(f"  {name}: {status}")
    
    all_passed = all(passed for _, passed in results)
    
    if all_passed:
        logger.info("\nAll checks passed!")
        return 0
    else:
        logger.error("\nSome checks failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main())
