"""
Colab Manager for Hybrid Orchestrator.
Handles all Colab-related operations including batch preparation and result loading.
"""

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from src.core.logging.logger import ProjectLogger

logger = ProjectLogger.get_logger(__name__)

# Constants
FEATURES_FILE = "features.parquet"
TARGETS_FILE = "targets.parquet"
BATCH_METADATA_FILE = "batch_metadata.json"
SELECTED_FEATURES_PATTERN = "selected_features_*.json"


@dataclass
class BatchPreparationConfig:
    """Configuration for preparing a Colab batch to avoid excessive arguments."""
    tickers: list[str]
    timeframes: list[str]
    batch_name: str | None = None
    accumulate: bool = True
    check_feature_selection: bool = True
    force_feature_selection: bool = False

    # Test mode parameters (optional - only for test mode)
    test_ticker: str | None = None
    test_target: str | None = None
    test_model: str | None = None
    epochs: int | None = None
    max_iterations: int | None = None


class ColabManager:
    """Manages Colab-related operations for hybrid pipeline."""

    def __init__(self, output_dir: Path, batch_name: str):
        self.output_dir = output_dir
        self.batch_name = batch_name
        self.logger = ProjectLogger.get_logger(__name__)

    def prepare_colab_batch(self,
                            features_df: pd.DataFrame,
                            targets_df: pd.DataFrame,
                            prices_dict: dict[str, pd.DataFrame],
                            config: BatchPreparationConfig,
                            news_df: pd.DataFrame | None = None,
                            economic_df: pd.DataFrame | None = None) -> dict[str, Any]:
        """Prepare data package for Colab training."""

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Resolve batch name for metadata (but don't use for path!)
        base_name = config.batch_name or self.batch_name
        eff_batch_name = base_name.replace('target_target_', 'target_')

        # ✅ output_dir already includes batch_name! (from OrchestratorConfigManager)
        batch_dir = self.output_dir
        batch_dir.mkdir(parents=True, exist_ok=True)

        # Persistent storage for actual data (Stage 3 persistence)
        # Note: We save data here, and metadata/config in batch_dir
        persistent_dir = Path("data/processed/features")
        persistent_dir.mkdir(parents=True, exist_ok=True)

        features_path = persistent_dir / FEATURES_FILE
        targets_path = persistent_dir / TARGETS_FILE
        news_path = persistent_dir / "news_data.parquet"
        economic_path = persistent_dir / "macro_data.parquet"

        # Accumulate data if files exist and accumulate=True
        if config.accumulate and features_path.exists() and targets_path.exists():
            # Load existing data
            try:
                existing_features = pd.read_parquet(features_path)
                existing_targets = pd.read_parquet(targets_path)

                # Combine with new data
                combined_features = pd.concat([existing_features, features_df], ignore_index=True)
                combined_targets = pd.concat([existing_targets, targets_df], ignore_index=True)

                # Remove duplicates
                combined_features = combined_features.drop_duplicates(
                    subset=self._dedupe_subset(combined_features, "features"),
                    keep='last'
                )
                combined_targets = combined_targets.drop_duplicates(
                    subset=self._dedupe_subset(combined_targets, "targets"),
                    keep='last'
                )

                self.logger.info(f"Accumulated data: {len(existing_features)}→{len(combined_features)} features")
                features_df = combined_features
                targets_df = combined_targets
            except Exception as e:
                self.logger.warning(f"⚠️ Failed to accumulate data: {e}. Saving only new data.")

        # Ensure datetime is preserved as a column before saving
        features_df = self._ensure_datetime_column(features_df)
        targets_df = self._ensure_datetime_column(targets_df)

        # Save main data
        features_df.to_parquet(features_path, index=False)
        targets_df.to_parquet(targets_path, index=False)
        self.logger.info(f"💾 Saved features ({features_df.shape}) and targets ({targets_df.shape}) to {persistent_dir}")

        # Also save to batch_dir for self-contained packaging and continue mode
        features_df.to_parquet(batch_dir / FEATURES_FILE, index=False)
        targets_df.to_parquet(batch_dir / TARGETS_FILE, index=False)
        self.logger.info(f"💾 Saved features and targets to batch directory: {batch_dir}")

        # Save additional data (news and economic)
        if news_df is not None and not news_df.empty:
            news_df = self._ensure_datetime_column(news_df)
            news_df.to_parquet(news_path, index=False)
            news_df.to_parquet(batch_dir / "news_data.parquet", index=False)
            self.logger.info(f"💾 Saved news data: {len(news_df)} rows to both persistent and batch directories")

        if economic_df is not None and not economic_df.empty:
            economic_df = self._ensure_datetime_column(economic_df)
            economic_df.to_parquet(economic_path, index=False)
            economic_df.to_parquet(batch_dir / "economic_data.parquet", index=False)
            self.logger.info(f"💾 Saved economic data: {len(economic_df)} rows to both persistent and batch directories")

        # Create config.json ONLY for test mode
        config_path = None
        if self._is_test_mode(config):
            config_path = self._create_test_config(batch_dir, config, timestamp, eff_batch_name)
        else:
            self.logger.info("📊 Full mode: NOT creating config.json (Colab will use all data)")
            # Clean up old config if exists
            old_config = batch_dir / "config.json"
            if old_config.exists():
                old_config.unlink()

        # Create batch metadata in the BATCH directory
        batch_metadata = {
            'batch_name': eff_batch_name,
            'timestamp': timestamp,
            'tickers': config.tickers,
            'timeframes': config.timeframes,
            # Cast shape tuples → list[int] so BatchManifestSchema and json.dump are happy
            'features_shape': list(features_df.shape),
            'targets_shape': list(targets_df.shape),
            'accumulated': config.accumulate,
            'test_mode': self._is_test_mode(config),
            'files': {
                'features': str(features_path),
                'targets': str(targets_path),
                'config': str(config_path) if config_path else None
            }
        }

        # Save metadata to batch_dir
        metadata_path = batch_dir / BATCH_METADATA_FILE
        with open(metadata_path, 'w') as f:
            json.dump(batch_metadata, f, indent=2)

        # Create a stable batch manifest including a simple data signature and code version
        try:
            import hashlib
            # Build a simple signature from shapes and column names
            feat_cols = ','.join(sorted([str(c) for c in features_df.columns])) if not features_df.empty else ''
            targ_cols = ','.join(sorted([str(c) for c in targets_df.columns])) if not features_df.empty else ''
            sig_src = f"{features_df.shape}-{targets_df.shape}-{feat_cols}-{targ_cols}"
            data_signature = hashlib.md5(sig_src.encode('utf-8'), usedforsecurity=False).hexdigest()

            # Try to detect git commit for code version (optional)
            code_version = None
            try:
                import subprocess
                git_rev = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True)
                code_version = git_rev.stdout.strip()
            except Exception:
                code_version = None

            batch_manifest = {
                'batch_name': eff_batch_name,
                'timestamp': timestamp,
                'data_signature': data_signature,
                'code_version': code_version,
                'files': batch_metadata['files'],
                'features_shape': features_df.shape,
                'targets_shape': targets_df.shape,
            }

            manifest_path = batch_dir / "batch_manifest.json"
            with open(manifest_path, 'w', encoding='utf-8') as mf:
                json.dump(batch_manifest, mf, indent=2, default=str)

            self.logger.info(f"💡 Batch manifest created: {manifest_path}")
        except Exception as e:
            self.logger.warning(f"⚠️ Could not create batch manifest: {e}")

        # Explicit Local-Colab Contract Validation
        try:
            from src.validation.pipeline_schemas import validate_batch_dir
            val_report = validate_batch_dir(batch_dir)
            if val_report["valid"]:
                self.logger.info(f"✨ Explicit local-Colab contract verified for {eff_batch_name}!")
            else:
                self.logger.warning(f"⚠️ Explicit local-Colab contract validation warnings: {val_report['errors']}")
        except Exception as ve:
            self.logger.debug(f"Failed to run contract validation: {ve}")

        # Check feature selection using config parameters
        fs_check = self._check_feature_selection(
            batch_dir,
            features_df,
            config.check_feature_selection,
            config.force_feature_selection
        )

        return {
            'status': 'completed',
            'batch_dir': str(batch_dir),
            'batch_name': eff_batch_name,
            'metadata_path': str(metadata_path),
            'files': batch_metadata['files'],
            'feature_selection_check': fs_check,
            'test_mode': self._is_test_mode(config)
        }

    def _ensure_datetime_column(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure datetime column exists."""
        if 'datetime' not in df.columns and isinstance(df.index, pd.DatetimeIndex):
            df = df.reset_index()
            if 'index' in df.columns:
                df = df.rename(columns={'index': 'datetime'})
        return df

    def _dedupe_subset(self, df: pd.DataFrame, label: str) -> list[str]:
        """Return the strongest available row identity for accumulated market data."""
        subset = ['datetime']
        if 'ticker' in df.columns:
            subset.append('ticker')
        if 'timeframe' in df.columns:
            subset.append('timeframe')
        elif 'interval' in df.columns:
            subset.append('interval')
        return subset

    def _is_test_mode(self, config: BatchPreparationConfig) -> bool:
        """Check if this is test mode based on config parameters."""
        return bool(config.test_ticker or config.test_target or config.test_model)

    def _create_test_config(self, batch_dir: Path, config: BatchPreparationConfig,
                           timestamp: str, batch_name: str) -> Path:
        """Create config.json for test mode."""
        config_data = {
            'test_mode': {
                'enabled': True,
                'test_ticker': config.test_ticker,
                'test_target': config.test_target,
                'test_model': config.test_model,
                'epochs': config.epochs or 5,
                'max_iterations': config.max_iterations or 5
            },
            'batch_name': batch_name,
            'created_at': timestamp
        }

        config_path = batch_dir / "config.json"
        with open(config_path, 'w') as f:
            json.dump(config_data, f, indent=2)

        self.logger.info(f"🧪 Test mode config created: {config.test_ticker} | {config.test_target}")
        return config_path

    def _check_feature_selection(self, batch_dir: Path, features_df: pd.DataFrame,
                                 check_selection: bool, force_selection: bool) -> dict[str, Any]:
        """Check if feature selection is needed."""
        if not check_selection:
            return {'needed': False, 'reason': 'Feature selection check disabled'}

        selected_features_files = list(batch_dir.glob(SELECTED_FEATURES_PATTERN))
        if force_selection or not selected_features_files:
            reason = 'Forced selection' if force_selection else 'No existing selection'
            return {'needed': True, 'reason': reason}

        return {'needed': False, 'reason': 'Existing feature selection found'}

    def load_colab_results(self, batch_name: str) -> dict[str, Any]:
        """Loads training results from Colab."""
        batch_name = batch_name.replace('target_target_', 'target_')

        # Search in the batch directory
        batch_dir = self.output_dir
        if not batch_dir.exists():
            self.logger.error(f"Batch directory not found: {batch_dir}")
            return {'error': f'Batch directory not found: {batch_dir}'}

        results = {}
        # Mapping of filenames to result keys
        files_to_load = {
            SELECTED_FEATURES_PATTERN: 'selected_features',
            'trained_models_metadata.json': 'models_metadata',
            'colab_results.json': 'models_metadata',
            'evaluation_results.json': 'evaluation_results'
        }

        for pattern, key in files_to_load.items():
            if "*" in pattern:
                found_files = list(batch_dir.glob(pattern))
                for file_path in found_files:
                    with open(file_path) as f:
                        data = json.load(f)
                        if key not in results:
                            results[key] = data
                        elif isinstance(results[key], dict) and isinstance(data, dict):
                            results[key].update(data)
            else:
                file_path = batch_dir / pattern
                if file_path.exists():
                    with open(file_path) as f:
                        data = json.load(f)
                        if key == 'models_metadata' and 'models_metadata' in data:
                            results[key] = data['models_metadata']
                        else:
                            results[key] = data

        return results
