#!/usr/bin/env python3
"""
Comprehensive database audit script.
Analyzes how databases are formed, data quality, and cleanliness.
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DatabaseAudit:
    def __init__(self):
        self.audit_results = {
            "timestamp": datetime.now().isoformat(),
            "databases": {},
            "summary": {}
        }
    
    def audit_colab_database(self):
        """Audit the Colab accumulated database"""
        logger.info("=" * 80)
        logger.info("AUDITING COLAB ACCUMULATED DATABASE")
        logger.info("=" * 80)
        
        db_path = Path("data/colab/accumulated/test_ticker_amd_target_return_1d")
        
        if not db_path.exists():
            logger.error(f"Database path not found: {db_path}")
            return
        
        db_info = {
            "path": str(db_path),
            "files": {},
            "data_quality": {},
            "structure": {},
            "issues": []
        }
        
        # Check metadata
        metadata_file = db_path / "batch_metadata.json"
        if metadata_file.exists():
            with open(metadata_file) as f:
                metadata = json.load(f)
            db_info["metadata"] = metadata
            logger.info(f"✓ Metadata found: {metadata['batch_name']}")
            logger.info(f"  Timestamp: {metadata['timestamp']}")
            logger.info(f"  Tickers: {metadata['tickers']}")
            logger.info(f"  Features shape: {metadata['features_shape']}")
            logger.info(f"  Targets shape: {metadata['targets_shape']}")
        
        # Audit features
        features_file = db_path / "features.parquet"
        if features_file.exists():
            logger.info("\n📊 FEATURES AUDIT")
            df_features = pd.read_parquet(features_file)
            db_info["files"]["features"] = self._audit_dataframe(
                df_features, "features"
            )
        
        # Audit targets
        targets_file = db_path / "targets.parquet"
        if targets_file.exists():
            logger.info("\n🎯 TARGETS AUDIT")
            df_targets = pd.read_parquet(targets_file)
            db_info["files"]["targets"] = self._audit_dataframe(
                df_targets, "targets"
            )
        
        # Check config
        config_file = db_path / "config.json"
        if config_file.exists():
            with open(config_file) as f:
                config = json.load(f)
            db_info["config"] = {
                "enabled_enrichers": config.get("features", {}).get("enabled_enrichers", {}),
                "target_types": list(config.get("targets", {}).keys())
            }
            logger.info(f"\n⚙️  CONFIG AUDIT")
            logger.info(f"  Enabled enrichers: {len(config['features']['enabled_enrichers'])}")
            logger.info(f"  Target types: {len(config['targets'])}")
        
        self.audit_results["databases"]["colab"] = db_info
    
    def _audit_dataframe(self, df, name):
        """Audit a single dataframe"""
        info = {
            "shape": df.shape,
            "columns": list(df.columns),
            "dtypes": df.dtypes.to_dict(),
            "memory_usage_mb": df.memory_usage(deep=True).sum() / 1024**2,
            "quality_metrics": {}
        }
        
        logger.info(f"  Shape: {df.shape}")
        logger.info(f"  Columns: {len(df.columns)}")
        logger.info(f"  Memory: {info['memory_usage_mb']:.2f} MB")
        
        # Quality metrics
        quality = {
            "null_count": df.isnull().sum().to_dict(),
            "null_percentage": (df.isnull().sum() / len(df) * 100).to_dict(),
            "duplicates": df.duplicated().sum(),
            "duplicate_percentage": (df.duplicated().sum() / len(df) * 100),
        }
        
        # Check for NaN/Inf
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            quality["inf_count"] = np.isinf(df[numeric_cols]).sum().to_dict()
            quality["inf_percentage"] = (np.isinf(df[numeric_cols]).sum() / len(df) * 100).to_dict()
        
        info["quality_metrics"] = quality
        
        # Log quality issues
        null_cols = [col for col, count in quality["null_count"].items() if count > 0]
        if null_cols:
            logger.warning(f"  ⚠️  Null values found in: {null_cols}")
            for col in null_cols:
                pct = quality["null_percentage"][col]
                logger.warning(f"     {col}: {quality['null_count'][col]} ({pct:.2f}%)")
        
        if quality["duplicates"] > 0:
            logger.warning(f"  ⚠️  Duplicates: {quality['duplicates']} ({quality['duplicate_percentage']:.2f}%)")
        
        if "inf_count" in quality:
            inf_cols = [col for col, count in quality["inf_count"].items() if count > 0]
            if inf_cols:
                logger.warning(f"  ⚠️  Infinite values found in: {inf_cols}")
        
        # Data type consistency
        logger.info(f"  Data types: {len(set(df.dtypes))}")
        
        return info
    
    def audit_stage_files(self):
        """Audit intermediate stage files"""
        logger.info("\n" + "=" * 80)
        logger.info("AUDITING STAGE FILES")
        logger.info("=" * 80)
        
        accumulated_path = Path("data/colab/accumulated")
        
        # Find stage files
        stage_files = {
            "stage1_raw": list(accumulated_path.glob("*stage1_raw_data*.parquet")),
            "stage2_cleaned": list(accumulated_path.glob("*stage2_cleaned_data*.parquet")),
            "stage2_macro": list(accumulated_path.glob("*stage2_macro*.parquet")),
            "stage2_news": list(accumulated_path.glob("*stage2_news*.parquet")),
            "stage3_enriched": list(accumulated_path.glob("*stage3_enriched*.parquet")),
        }
        
        stage_info = {}
        for stage_name, files in stage_files.items():
            if files:
                logger.info(f"\n{stage_name}: {len(files)} files")
                stage_info[stage_name] = {
                    "count": len(files),
                    "files": [f.name for f in sorted(files)[-3:]]  # Last 3
                }
        
        self.audit_results["stage_files"] = stage_info
    
    def generate_report(self):
        """Generate final audit report"""
        logger.info("\n" + "=" * 80)
        logger.info("AUDIT REPORT SUMMARY")
        logger.info("=" * 80)
        
        # Save report
        report_path = Path("results/database_audit_report.json")
        report_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(report_path, 'w') as f:
            json.dump(self.audit_results, f, indent=2, default=str)
        
        logger.info(f"\n✅ Audit report saved to: {report_path}")
        
        # Print summary
        if "colab" in self.audit_results["databases"]:
            db = self.audit_results["databases"]["colab"]
            logger.info("\n📊 COLAB DATABASE SUMMARY:")
            logger.info(f"  Features: {db['files']['features']['shape']}")
            logger.info(f"  Targets: {db['files']['targets']['shape']}")
            
            # Quality summary
            feat_quality = db['files']['features']['quality_metrics']
            targ_quality = db['files']['targets']['quality_metrics']
            
            logger.info(f"\n  Features Quality:")
            logger.info(f"    Duplicates: {feat_quality['duplicates']}")
            logger.info(f"    Null columns: {len([c for c, v in feat_quality['null_count'].items() if v > 0])}")
            
            logger.info(f"\n  Targets Quality:")
            logger.info(f"    Duplicates: {targ_quality['duplicates']}")
            logger.info(f"    Null columns: {len([c for c, v in targ_quality['null_count'].items() if v > 0])}")

def main():
    audit = DatabaseAudit()
    audit.audit_colab_database()
    audit.audit_stage_files()
    audit.generate_report()

if __name__ == "__main__":
    main()
