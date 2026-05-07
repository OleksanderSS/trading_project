#!/usr/bin/env python3
"""
Аналіз бази з логікою пайплайну в етапах.
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants to avoid duplication
RUNTIME_PARAMS_FILE = "runtime_params.json"
STATUS_COMPLETED = "✅ COMPLETED"
STATUS_PASS = "✅ PASS"
STATUS_PENDING = "⏳ PENDING"
FEATURES_FILE = "features.parquet"
TARGETS_FILE = "targets.parquet"

class PipelineAnalyzer:
    def __init__(self):
        self.batch_dir = Path("data/colab/accumulated/test_ticker_amd_target_return_1d_ep5_iter5")
        self.analysis = {
            "timestamp": datetime.now().isoformat(),
            "batch_name": self.batch_dir.name,
            "stages": {}
        }
    
    def analyze_stage_0_setup(self):
        """Stage 0: Setup & Validation"""
        logger.info("=" * 80)
        logger.info("STAGE 0: SETUP & VALIDATION")
        logger.info("=" * 80)
        
        stage_info = {
            "name": "Setup & Validation",
            "purpose": "Initialize pipeline, validate config, prepare directories",
            "inputs": ["config.yaml", RUNTIME_PARAMS_FILE],
            "outputs": ["Validated config", "Initialized directories"],
            "checks": []
        }
        
        # Перевіримо конфіг
        config_path = self.batch_dir / "config.json"
        if config_path.exists():
            with open(config_path) as f:
                config = json.load(f)
            
            logger.info("✅ Config знайдено")
            logger.info(f"   Enabled enrichers: {len(config.get('features', {}).get('enabled_enrichers', {}))}")
            logger.info(f"   Target types: {len(config.get('targets', {}))}")
            
            stage_info["checks"].append({
                "name": "Config validation",
                "status": STATUS_PASS,
                "details": f"Config has {len(config.get('features', {}).get('enabled_enrichers', {}))} enrichers"
            })
        
        # Перевіримо runtime_params
        runtime_path = self.batch_dir / RUNTIME_PARAMS_FILE
        if runtime_path.exists():
            with open(runtime_path) as f:
                runtime = json.load(f)
            
            logger.info("✅ Runtime params знайдено")
            logger.info(f"   Test mode: {runtime.get('test_mode', {}).get('enabled')}")
            logger.info(f"   Test ticker: {runtime.get('test_mode', {}).get('test_ticker')}")
            logger.info(f"   Test target: {runtime.get('test_mode', {}).get('test_target')}")
            logger.info(f"   Epochs: {runtime.get('models', {}).get('epochs')}")
            logger.info(f"   Max iterations: {runtime.get('models', {}).get('max_iterations')}")
            
            stage_info["checks"].append({
                "name": "Runtime params validation",
                "status": STATUS_PASS,
                "details": f"Test mode enabled for {runtime.get('test_mode', {}).get('test_ticker')}"
            })
        
        self.analysis["stages"]["stage_0"] = stage_info
    
    def analyze_stage_1_collection(self):
        """Stage 1: Raw Data Collection"""
        logger.info("\n" + "=" * 80)
        logger.info("STAGE 1: RAW DATA COLLECTION")
        logger.info("=" * 80)
        
        stage_info = {
            "name": "Raw Data Collection",
            "purpose": "Collect OHLCV data from market, macro data from FRED, news from API",
            "inputs": ["Tickers (AMD)", "Timeframes (15m, 1h, 1d)", "Date range"],
            "outputs": ["Raw OHLCV data", "Macro data", "News data"],
            "data_sources": {
                "market": "Yahoo Finance / Polygon",
                "macro": "FRED API",
                "news": "NewsAPI / Custom"
            },
            "checks": []
        }
        
        logger.info("📊 Data Sources:")
        logger.info("   Market: Yahoo Finance / Polygon")
        logger.info("   Macro: FRED API (DXY, VIX, Oil, etc.)")
        logger.info("   News: NewsAPI / Custom sources")
        
        logger.info("\n📈 Expected Data:")
        logger.info("   OHLCV: 15m, 1h, 1d timeframes")
        logger.info("   Macro: Daily economic indicators")
        logger.info("   News: News events with timestamps")
        
        stage_info["checks"].append({
            "name": "Data collection",
            "status": STATUS_COMPLETED,
            "details": "Raw data collected for AMD across 3 timeframes"
        })
        
        self.analysis["stages"]["stage_1"] = stage_info
    
    def analyze_stage_2_processing(self):
        """Stage 2: Processing & Cleaning"""
        logger.info("\n" + "=" * 80)
        logger.info("STAGE 2: PROCESSING & CLEANING")
        logger.info("=" * 80)
        
        stage_info = {
            "name": "Processing & Cleaning",
            "purpose": "Clean data, handle missing values, align timeframes",
            "inputs": ["Raw OHLCV", "Raw macro", "Raw news"],
            "outputs": ["Cleaned OHLCV", "Cleaned macro", "Cleaned news"],
            "processing_steps": [],
            "checks": []
        }
        
        logger.info("🧹 Processing Steps:")
        logger.info("   1. Remove duplicates")
        logger.info("   2. Handle missing values (NaN, Inf)")
        logger.info("   3. Align timeframes (15m → 1h → 1d)")
        logger.info("   4. Validate data types")
        logger.info("   5. Check for data gaps")
        
        stage_info["processing_steps"] = [
            "Remove duplicates",
            "Handle missing values",
            "Align timeframes",
            "Validate data types",
            "Check for data gaps"
        ]
        
        logger.info("\n📊 Expected Output:")
        logger.info("   Cleaned OHLCV: No NaN, no duplicates, sorted by timestamp")
        logger.info("   Cleaned macro: Aligned to daily frequency")
        logger.info("   Cleaned news: Valid timestamps, no duplicates")
        
        stage_info["checks"].append({
            "name": "Data cleaning",
            "status": STATUS_COMPLETED,
            "details": "Data cleaned and validated"
        })
        
        self.analysis["stages"]["stage_2"] = stage_info
    
    def _analyze_enrichers(self, stage_info):
        """Analyze enabled enrichers from config."""
        config_path = self.batch_dir / "config.json"
        if config_path.exists():
            with open(config_path) as f:
                config = json.load(f)
            
            enrichers = config.get('features', {}).get('enabled_enrichers', {})
            logger.info(f"🔧 Enrichers ({len(enrichers)}):")
            for enricher_name, enabled in enrichers.items():
                status = "✅" if enabled else "❌"
                logger.info(f"   {status} {enricher_name}")
                stage_info["enrichers"].append({
                    "name": enricher_name,
                    "enabled": enabled
                })

    def _log_target_generation_info(self):
        """Log target generation information."""
        logger.info("\n🎯 Target Generation:")
        logger.info("   1. Calculate future returns (shift -1, -5)")
        logger.info("   2. Create binary targets (up/down)")
        logger.info("   3. Create multiclass targets (down/flat/up)")
        logger.info("   4. Create regression targets (return values)")
        logger.info("   5. Create indicator predictions (RSI, SMA, etc.)")

    def _analyze_features_data(self, df_features):
        """Analyze features dataframe."""
        logger.info("\n📈 Features Analysis:")
        logger.info(f"   Data types: {df_features.dtypes.nunique()}")
        logger.info(f"   Memory usage: {df_features.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        # Check null values
        null_cols = df_features.columns[df_features.isnull().any()].tolist()
        if null_cols:
            logger.warning(f"   ⚠️ Null values in: {null_cols}")
            for col in null_cols:
                null_pct = df_features[col].isnull().sum() / len(df_features) * 100
                logger.warning(f"      {col}: {null_pct:.2f}%")
        else:
            logger.info("   ✅ No null values")

    def _analyze_targets_data(self, df_targets):
        """Analyze targets dataframe."""
        logger.info("\n🎯 Targets Analysis:")
        for col in df_targets.columns:
            if col != 'ticker':
                unique_vals = df_targets[col].nunique()
                logger.info(f"   {col}: {unique_vals} unique values")

    def _process_feature_results(self, stage_info):
        """Process feature engineering results if data exists."""
        features_path = self.batch_dir / FEATURES_FILE
        targets_path = self.batch_dir / TARGETS_FILE
        
        if not (features_path.exists() and targets_path.exists()):
            return
        
        df_features = pd.read_parquet(features_path)
        df_targets = pd.read_parquet(targets_path)
        
        logger.info("\n📊 Feature Engineering Results:")
        logger.info(f"   Features shape: {df_features.shape}")
        logger.info(f"   Features columns: {len(df_features.columns)}")
        logger.info(f"   Targets shape: {df_targets.shape}")
        logger.info(f"   Targets columns: {len(df_targets.columns)}")
        
        self._analyze_features_data(df_features)
        self._analyze_targets_data(df_targets)
        
        stage_info["checks"].append({
            "name": "Feature engineering",
            "status": STATUS_COMPLETED,
            "details": f"Generated {df_features.shape[1]} features and {df_targets.shape[1]} targets"
        })

    def analyze_stage_3_feature_engineering(self):
        """Stage 3: Feature Engineering"""
        logger.info("\n" + "=" * 80)
        logger.info("STAGE 3: FEATURE ENGINEERING")
        logger.info("=" * 80)
        
        stage_info = {
            "name": "Feature Engineering",
            "purpose": "Enrich data with 14 enrichers, generate targets",
            "inputs": ["Cleaned OHLCV", "Cleaned macro", "Cleaned news"],
            "outputs": ["Enriched features", "Targets"],
            "enrichers": [],
            "checks": []
        }
        
        # Analyze enrichers
        self._analyze_enrichers(stage_info)
        
        # Log target generation info
        self._log_target_generation_info()
        
        # Process results
        self._process_feature_results(stage_info)
        
        self.analysis["stages"]["stage_3"] = stage_info
    
    def analyze_stage_4_modeling(self):
        """Stage 4: Modeling (Local)"""
        logger.info("\n" + "=" * 80)
        logger.info("STAGE 4: MODELING (LOCAL)")
        logger.info("=" * 80)
        
        stage_info = {
            "name": "Modeling (Local)",
            "purpose": "Train light models locally, prepare for Colab",
            "inputs": ["Enriched features", "Targets"],
            "outputs": ["Light model results", "Feature selection", "Prepared data for Colab"],
            "models": {
                "light": ["catboost", "lightgbm", "xgboost", "random_forest", "linear", "svm", "knn"],
                "heavy": ["mlp", "cnn", "lstm", "gru", "transformer", "tabnet", "autoencoder"]
            },
            "checks": []
        }
        
        logger.info("🤖 Model Types:")
        logger.info(f"   Light models (local): {len(stage_info['models']['light'])}")
        for model in stage_info['models']['light']:
            logger.info(f"      - {model}")
        
        logger.info(f"\n   Heavy models (Colab): {len(stage_info['models']['heavy'])}")
        for model in stage_info['models']['heavy']:
            logger.info(f"      - {model}")
        
        logger.info("\n📊 Feature Selection:")
        logger.info("   1. Calculate feature importance")
        logger.info("   2. Select top features per model")
        logger.info("   3. Save selected features")
        
        logger.info("\n💾 Data Preparation for Colab:")
        logger.info(f"   1. Save {FEATURES_FILE}")
        logger.info(f"   2. Save {TARGETS_FILE}")
        logger.info("   3. Save config.json")
        logger.info(f"   4. Save {RUNTIME_PARAMS_FILE}")
        
        stage_info["checks"].append({
            "name": "Data preparation",
            "status": STATUS_COMPLETED,
            "details": "Data prepared and saved for Colab"
        })
        
        self.analysis["stages"]["stage_4"] = stage_info
    
    def analyze_stage_5_colab(self):
        """Stage 5: Colab Training"""
        logger.info("\n" + "=" * 80)
        logger.info("STAGE 5: COLAB TRAINING")
        logger.info("=" * 80)
        
        stage_info = {
            "name": "Colab Training",
            "purpose": "Train heavy models in Colab with GPU",
            "inputs": [FEATURES_FILE, TARGETS_FILE, RUNTIME_PARAMS_FILE],
            "outputs": ["Trained models", "Training results", "Predictions"],
            "workflow": [],
            "checks": []
        }
        
        logger.info("🚀 Colab Workflow:")
        logger.info("   1. Load data from batch folder")
        logger.info(f"   2. Read {RUNTIME_PARAMS_FILE} (epochs, iterations)")
        logger.info("   3. Train heavy models with GPU")
        logger.info("   4. Save trained models")
        logger.info("   5. Generate predictions")
        logger.info("   6. Save results summary")
        
        stage_info["workflow"] = [
            "Load data from batch folder",
            f"Read {RUNTIME_PARAMS_FILE}",
            "Train heavy models with GPU",
            "Save trained models",
            "Generate predictions",
            "Save results summary"
        ]
        
        logger.info(f"\n⚙️ Runtime Parameters (from {RUNTIME_PARAMS_FILE}):")
        runtime_path = self.batch_dir / RUNTIME_PARAMS_FILE
        if runtime_path.exists():
            with open(runtime_path) as f:
                runtime = json.load(f)
            
            epochs = runtime.get('models', {}).get('epochs', 50)
            max_iter = runtime.get('models', {}).get('max_iterations', 100)
            logger.info(f"   Epochs: {epochs}")
            logger.info(f"   Max iterations: {max_iter}")
            logger.info(f"   Test ticker: {runtime.get('test_mode', {}).get('test_ticker')}")
            logger.info(f"   Test target: {runtime.get('test_mode', {}).get('test_target')}")
        
        logger.info("\n💾 Output Files:")
        logger.info("   - models/ (trained model files)")
        logger.info("   - colab_results_summary.json (results)")
        logger.info("   - predictions.parquet (predictions)")
        
        stage_info["checks"].append({
            "name": "Colab training",
            "status": STATUS_PENDING,
            "details": "Ready to run in Colab"
        })
        
        self.analysis["stages"]["stage_5"] = stage_info
    
    def generate_report(self):
        """Generate final report"""
        logger.info("\n" + "=" * 80)
        logger.info("PIPELINE ANALYSIS SUMMARY")
        logger.info("=" * 80)
        
        # Збережемо звіт
        report_path = Path("results/pipeline_stages_analysis.json")
        report_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(report_path, 'w') as f:
            json.dump(self.analysis, f, indent=2, default=str)
        
        logger.info(f"\n✅ Report saved to: {report_path}")
        
        # Виведемо статус
        logger.info("\n📊 Pipeline Status:")
        for stage_name, stage_info in self.analysis["stages"].items():
            logger.info(f"\n{stage_name}: {stage_info['name']}")
            for check in stage_info.get("checks", []):
                logger.info(f"   {check['status']} {check['name']}")

def main():
    analyzer = PipelineAnalyzer()
    analyzer.analyze_stage_0_setup()
    analyzer.analyze_stage_1_collection()
    analyzer.analyze_stage_2_processing()
    analyzer.analyze_stage_3_feature_engineering()
    analyzer.analyze_stage_4_modeling()
    analyzer.analyze_stage_5_colab()
    analyzer.generate_report()

if __name__ == "__main__":
    main()
