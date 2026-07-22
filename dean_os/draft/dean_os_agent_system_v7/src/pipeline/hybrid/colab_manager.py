# audit-ignore: ARCHITECTURAL_USAGE
"""
Colab Manager for Hybrid Orchestrator.
Handles all Colab-related operations including batch preparation and result loading.
"""

import hashlib
import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from src.core.logging.logger import ProjectLogger
from src.features.utils.datetime_utils import ensure_datetime_column
from src.features.validation.feature_leakage_guard import FeatureLeakageGuard
from src.pipeline.target_column_utils import is_direct_target_column, is_target_like_column
from src.pipeline.timeframe_lineage import (
    normalize_timeframe,
    partition_market_frame_by_timeframe,
)

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
                            config: BatchPreparationConfig) -> dict[str, Any]:
        """
        Prepare data package for Colab training.
        High-level orchestrator for batch preparation process.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        eff_batch_name = (config.batch_name or self.batch_name).replace('target_target_', 'target_')

        # 1. Setup batch directory
        batch_dir = self.output_dir
        batch_dir.mkdir(parents=True, exist_ok=True)

        # 2. Save and accumulate data
        features_path, targets_path = self._save_and_accumulate_data(
            features_df, targets_df, batch_dir, config
        )

        # 3. Handle configuration (Test vs Full mode)
        config_path = self._handle_batch_configuration(batch_dir, config, timestamp, eff_batch_name)

        # 4. Create metadata
        final_features = pd.read_parquet(features_path)
        final_targets = pd.read_parquet(targets_path)
        metadata = self._create_batch_metadata(
            eff_batch_name, timestamp, config, final_features, final_targets,
            features_path, targets_path, config_path
        )

        # 5. Save metadata
        metadata_path = batch_dir / BATCH_METADATA_FILE
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)

        # 6. Check feature selection
        fs_check = self._check_feature_selection(
            batch_dir, features_df, config.check_feature_selection, config.force_feature_selection
        )

        return self._assemble_preparation_result(
            batch_dir, eff_batch_name, metadata_path, metadata, fs_check, config, config_path
        )

    def _save_and_accumulate_data(self,
                                features_df: pd.DataFrame,
                                targets_df: pd.DataFrame,
                                batch_dir: Path,
                                config: BatchPreparationConfig) -> tuple[Path, Path]:
        """Handles saving and optional accumulation of features and targets."""
        features_path = batch_dir / FEATURES_FILE
        targets_path = batch_dir / TARGETS_FILE
        features_df = self._validate_batch_frame(
            features_df,
            frame_name="features",
            requested_timeframes=config.timeframes,
        )
        targets_df = self._validate_batch_frame(
            targets_df,
            frame_name="targets",
            requested_timeframes=config.timeframes,
        )

        if config.accumulate and features_path.exists() and targets_path.exists():
            # Load existing
            existing_f = self._validate_batch_frame(
                pd.read_parquet(features_path),
                frame_name="existing features",
                requested_timeframes=config.timeframes,
            )
            existing_t = self._validate_batch_frame(
                pd.read_parquet(targets_path),
                frame_name="existing targets",
                requested_timeframes=config.timeframes,
            )

            # Combine
            combined_f = pd.concat([existing_f, features_df], ignore_index=True)
            combined_t = pd.concat([existing_t, targets_df], ignore_index=True)

            # Deduplicate
            combined_f = self._deduplicate_df(combined_f)
            combined_t = self._deduplicate_df(combined_t)

            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"Accumulated data: {len(existing_f)}→{len(combined_f)} features")

            # ✅ Run leakage guard before saving
            combined_f = self._check_feature_leakage(combined_f, combined_t)

            # Save
            self._save_df_to_parquet(combined_f, features_path)
            self._save_df_to_parquet(combined_t, targets_path)
        else:
            # New batch
            # ✅ Run leakage guard before saving
            features_df = self._check_feature_leakage(features_df, targets_df)
            self._save_df_to_parquet(features_df, features_path)
            self._save_df_to_parquet(targets_df, targets_path)
            logger.info(f"Created new batch: {len(features_df)} rows")

        return features_path, targets_path

    def _deduplicate_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove duplicate rows without collapsing separate timeframes."""
        subset = []
        for column in ("datetime", "ticker", "interval"):
            if column in df.columns:
                subset.append(column)

        if subset:
            return df.drop_duplicates(subset=subset, keep='last')
        return df

    def _validate_batch_frame(
        self,
        frame: pd.DataFrame,
        *,
        frame_name: str,
        requested_timeframes: list[str],
    ) -> pd.DataFrame:
        """Require exact, timezone-aware row identity before Colab save."""
        if not isinstance(frame, pd.DataFrame) or frame.empty:
            raise ValueError(f"Colab {frame_name} frame is empty")
        missing = [
            column
            for column in ("ticker", "datetime", "interval")
            if column not in frame.columns
        ]
        if missing:
            raise ValueError(
                f"Colab {frame_name} frame is missing identity columns: "
                + ", ".join(missing)
            )

        normalized = ensure_datetime_column(
            frame,
            raise_on_missing=True,
        )

        # Ensure UTC timezone if naive to prevent downstream Colab failures
        if normalized.attrs.get("datetime_timezone_status") != "timezone_aware":
            if getattr(normalized["datetime"].dt, "tz", None) is None:
                normalized["datetime"] = normalized["datetime"].dt.tz_localize("UTC")
            else:
                normalized["datetime"] = normalized["datetime"].dt.tz_convert("UTC")
            normalized.attrs["datetime_timezone_status"] = "timezone_aware"
            normalized.attrs["datetime_timezone"] = "UTC"

        if (
            normalized.attrs.get("datetime_timezone_status")
            != "timezone_aware"
        ):
            raise ValueError(
                f"Colab {frame_name} datetime timezone is unresolved"
            )
        if normalized["datetime"].isna().any():
            raise ValueError(
                f"Colab {frame_name} contains invalid datetime values"
            )

        requested = {
            normalize_timeframe(value)
            for value in requested_timeframes
            if normalize_timeframe(value)
        }
        observed = {
            normalize_timeframe(value)
            for value in normalized["interval"].dropna().unique()
            if normalize_timeframe(value)
        }
        if not observed:
            raise ValueError(
                f"Colab {frame_name} has no valid interval values"
            )
        unexpected = sorted(observed - requested) if requested else []
        if unexpected:
            raise ValueError(
                f"Colab {frame_name} contains unrequested timeframes: "
                + ", ".join(unexpected)
            )

        normalized["interval"] = normalized["interval"].map(
            normalize_timeframe
        )
        partition_market_frame_by_timeframe(
            normalized[["ticker", "datetime", "interval"]]
        )
        return normalized

    def _check_feature_leakage(self, features_df: pd.DataFrame, targets_df: pd.DataFrame) -> pd.DataFrame:
        """
        Run FeatureLeakageGuard before saving to Parquet.
        Removes forbidden future-leaking columns. Logs warnings for high-correlation features.
        Returns cleaned features_df.
        """
        try:
            guard = FeatureLeakageGuard(block_on_forbidden=False)  # warn only, don't raise
            target_like_feature_cols = [c for c in features_df.columns if is_target_like_column(c)]
            if target_like_feature_cols:
                logger.warning(
                    "[LeakageGuard] Removing %s target-like feature column(s): %s",
                    len(target_like_feature_cols),
                    target_like_feature_cols[:5],
                )
                features_df = features_df.drop(columns=target_like_feature_cols, errors='ignore')

            # Build combined df with both features and targets for correlation check
            target_cols = [c for c in targets_df.columns if is_direct_target_column(c)]
            combined = pd.concat([features_df, targets_df[target_cols]], axis=1) if target_cols else features_df

            report = guard.check(combined, target_cols=target_cols if target_cols else None)

            if report.has_issues:
                if report.forbidden_cols:
                    logger.warning(
                        f"[LeakageGuard] Removing {len(report.forbidden_cols)} forbidden columns: "
                        f"{report.forbidden_cols[:5]}{'...' if len(report.forbidden_cols) > 5 else ''}"
                    )
                    features_df = features_df.drop(columns=report.forbidden_cols, errors='ignore')

                if report.high_corr_cols:
                    logger.warning(
                        f"[LeakageGuard] {len(report.high_corr_cols)} features with high target "
                        f"correlation. Review: {list(report.high_corr_cols.keys())[:5]}"
                    )
            else:
                logger.debug("[LeakageGuard] No leakage detected.")

        except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError) as e:
            logger.warning(f"[LeakageGuard] Check failed (non-blocking): {e}")

        return features_df

    def _save_df_to_parquet(self, df: pd.DataFrame, path: Path):
        """Saves DataFrame to Parquet, ensuring datetime column is preserved."""
        df_to_save = df.copy()
        if 'datetime' not in df_to_save.columns and isinstance(df_to_save.index, pd.DatetimeIndex):
            df_to_save = df_to_save.reset_index()
            if 'index' in df_to_save.columns:
                df_to_save = df_to_save.rename(columns={'index': 'datetime'})

        df_to_save.to_parquet(path, index=False)

    def _handle_batch_configuration(self, batch_dir: Path, config: BatchPreparationConfig,
                                   timestamp: str, batch_name: str) -> Path | None:
        """Handles config.json creation or removal depending on mode."""
        if self._is_test_mode(config):
            return self._create_test_config(batch_dir, config, timestamp, batch_name)

        # Full mode: cleanup old config
        old_config = batch_dir / "config.json"
        if old_config.exists():
            logger.warning("🗑️ Removing old config.json from previous test run")
            old_config.unlink()

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("📊 Full mode: config.json NOT created (all data will be processed)")
        return None

    def _create_batch_metadata(self, name: str, timestamp: str, config: BatchPreparationConfig,
                              f_df: pd.DataFrame, t_df: pd.DataFrame,
                              f_path: Path, t_path: Path, c_path: Path | None) -> dict[str, Any]:
        """Creates the batch metadata dictionary."""
        return {
            'batch_name': name,
            'timestamp': timestamp,
            'tickers': config.tickers,
            'timeframes': config.timeframes,
            'features_shape': f_df.shape,
            'targets_shape': t_df.shape,
            'accumulated': config.accumulate,
            'test_mode': self._is_test_mode(config),
            'files': {
                'features': str(f_path),
                'targets': str(t_path),
                'config': str(c_path) if c_path else None
            },
            'lineage': {
                'features_sha256': self._sha256(f_path),
                'targets_sha256': self._sha256(t_path),
                'identity_columns': [
                    'ticker',
                    'datetime',
                    'interval',
                ],
                'feature_interval_counts': {
                    str(key): int(value)
                    for key, value in f_df['interval']
                    .value_counts()
                    .sort_index()
                    .items()
                },
                'datetime_timezone': str(f_df['datetime'].dt.tz),
                'datetime_timezone_status': 'timezone_aware',
            },
        }

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

    def _assemble_preparation_result(self, batch_dir: Path, name: str, meta_path: Path,
                                    metadata: dict[str, Any], fs_check: dict[str, Any],
                                    config: BatchPreparationConfig, config_path: Path | None) -> dict[str, Any]:
        """Assembles the final dictionary returned by prepare_colab_batch."""
        result = {
            'batch_dir': str(batch_dir),
            'batch_name': name,
            'metadata_path': str(meta_path),
            'files': metadata['files'],
            'feature_selection_check': fs_check,
            'test_mode': self._is_test_mode(config)
        }

        if config_path:
            result['config_path'] = str(config_path)

        return result

    def load_colab_results(self, batch_name: str) -> dict[str, Any]:
        """Loads training results from Colab."""
        batch_name = batch_name.replace('target_target_', 'target_')
        batch_dir = self._find_batch_directory(batch_name)

        if not batch_dir.exists():
            self.logger.error(f"Batch directory not found: {batch_dir}")
            return {'error': f'Batch directory not found: {batch_dir}'}

        results = {}
        files_to_load = {
            SELECTED_FEATURES_PATTERN: 'selected_features',
            'trained_models_metadata.json': 'models_metadata',
            'colab_results.json': 'models_metadata',
            'evaluation_results.json': 'evaluation_results'
        }

        self._load_files_from_directory(batch_dir, files_to_load, results)
        return results

    def _load_files_from_directory(self, batch_dir: Path, files_to_load: dict[str, str], results: dict[str, Any]) -> None:
        """Helper to load files."""
        for pattern, key in files_to_load.items():
            if "*" in pattern:
                found_files = list(batch_dir.glob(pattern))
                for file_path in found_files:
                    self._load_single_file(file_path, key, results)
            else:
                file_path = batch_dir / pattern
                if file_path.exists():
                    self._load_single_file(file_path, key, results)

    def _load_single_file(self, file_path: Path, key: str, results: dict[str, Any]) -> None:
        """Helper to load a single file."""
        with open(file_path, encoding='utf-8') as f:
            data = json.load(f)
            if key == 'models_metadata' and 'models_metadata' in data:
                results[key] = data['models_metadata']
            elif key in results and isinstance(results[key], dict) and isinstance(data, dict):
                results[key].update(data)
            else:
                results[key] = data

    def _find_batch_directory(self, batch_name: str) -> Path:
        """Find the batch directory by name."""
        if self.output_dir.exists():
            return self.output_dir
        return self.output_dir / batch_name

    def _check_feature_selection(self, batch_dir: Path, features_df: pd.DataFrame,
                                 check_selection: bool, force_selection: bool) -> dict[str, Any]:
        """Check if feature selection is needed."""
        if not check_selection:
            return {'needed': False, 'reason': 'Feature selection check disabled'}

        selected_features_files = list(batch_dir.glob(SELECTED_FEATURES_PATTERN))

        if force_selection or not selected_features_files:
            reason = 'Forced selection' if force_selection else 'No existing selection'
            return {'needed': True, 'reason': reason}

        if len(features_df) < 1000:
            return {'needed': False, 'reason': 'Dataset too small for feature selection'}

        return {'needed': False, 'reason': 'Existing feature selection found'}

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
            json.dump(config_data, f, indent=2, default=str)

        self.logger.info(f"🧪 Test mode config created: {config.test_ticker} | {config.test_target} | epochs={config.epochs}")
        return config_path
