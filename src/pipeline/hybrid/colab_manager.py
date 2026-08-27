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
import pyarrow as pa
import pyarrow.parquet as pq

from src.core.logging.logger import ProjectLogger
from src.features.utils.datetime_utils import ensure_datetime_column
from src.features.validation.feature_leakage_guard import FeatureLeakageGuard
from src.pipeline.target_column_utils import is_direct_target_column, is_target_like_column
from src.pipeline.timeframe_lineage import normalize_timeframe
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

        # 2b. Write one file per timeframe beside the combined one.
        #
        # The combined frame carries every timeframe's columns on every row --
        # 154,069 daily rows holding 1,836 unused ones -- and loading it costs
        # 4.85 GiB of resident memory against 0.27 for the daily slice. The
        # loader prefers the slices when they exist, so producing them here is
        # what makes that path real rather than something someone has to
        # remember to run.
        #
        # Failure is logged and not raised: the combined batch is written and
        # valid by this point, and losing it over an optimisation would be the
        # wrong trade.
        self._write_timeframe_slices(batch_dir)

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

    def _write_timeframe_slices(self, batch_dir: Path) -> None:
        """Produce features_<tf>.parquet beside the combined batch."""
        try:
            from src.pipeline.batch_timeframe_split import split_batch

            report = split_batch(batch_dir)
        except Exception as error:  # noqa: BLE001
            # Named, not swallowed. The combined batch is already written, so
            # this is a lost optimisation rather than a lost run -- but a
            # silent one would leave the loader quietly reading the expensive
            # file forever.
            self.logger.error(
                "Could not write per-timeframe slices (%s: %s). The combined "
                "batch is intact; loading will use it instead.",
                type(error).__name__, error,
            )
            return

        for timeframe, entry in sorted(report.items()):
            self.logger.info(
                "Slice %s: %d rows, %d of %d columns.",
                timeframe, entry["rows"], entry["columns"], entry["of"],
            )

    def _save_frames_by_timeframe(self, features_by_timeframe: dict,
                                  targets_by_timeframe, features_path: Path,
                                  targets_path: Path, config) -> tuple[Path, Path]:
        """Write the per-timeframe frames into the two union files.

        `pipeline_runner` has already written these very paths by the time this
        runs, so "existing" is almost always this run's own output -- the
        situation that turned accumulation into a batch joined to itself on
        2026-08-22. The cheap key check still decides, and when the disk holds
        nothing new the frames are simply written.

        Accumulation across timeframes is done one timeframe at a time, reading
        only that interval's rows back from disk. Loading the whole file to
        merge would rebuild the union this entire change exists to avoid.
        """
        from src.pipeline.parquet_union_writer import write_union

        carried_over = 0
        if config.accumulate and features_path.exists() and targets_path.exists():
            identity = pd.concat(
                [
                    frame[[c for c in ("datetime", "ticker", "interval")
                           if c in frame.columns]]
                    for frame in features_by_timeframe.values()
                    if frame is not None and not frame.empty
                ],
                ignore_index=True,
            )
            carried_over = self._rows_not_already_present(features_path, identity)

        if carried_over:
            logger.info(
                "Accumulating per timeframe: %d row(s) on disk are not in "
                "this batch.", carried_over,
            )
            import pyarrow.parquet as pq
            for timeframe, frame in list(features_by_timeframe.items()):
                try:
                    existing = pq.read_table(
                        features_path,
                        filters=[("interval", "=", timeframe)],
                    ).to_pandas()
                except (OSError, ValueError, KeyError) as error:
                    logger.warning(
                        "Could not read %s rows back for accumulation (%s); "
                        "keeping this run's rows only.", timeframe, error,
                    )
                    continue
                if existing.empty:
                    continue
                merged = pd.concat([existing, frame], ignore_index=True)
                features_by_timeframe[timeframe] = self._deduplicate_df(merged)

        for timeframe, frame in list(features_by_timeframe.items()):
            targets = (targets_by_timeframe or {}).get(timeframe) \
                if isinstance(targets_by_timeframe, dict) else None
            if targets is not None:
                features_by_timeframe[timeframe] = self._check_feature_leakage(
                    frame, targets,
                )

        write_union(features_by_timeframe, features_path)
        if isinstance(targets_by_timeframe, dict):
            write_union(targets_by_timeframe, targets_path)
        logger.info(
            "Batch written by timeframe: %s",
            ", ".join(f"{name} {len(frame):,}"
                      for name, frame in features_by_timeframe.items()),
        )
        return features_path, targets_path

    def _save_and_accumulate_data(self,
                                features_df: pd.DataFrame,
                                targets_df: pd.DataFrame,
                                batch_dir: Path,
                                config: BatchPreparationConfig) -> tuple[Path, Path]:
        """Handles saving and optional accumulation of features and targets."""
        features_path = batch_dir / FEATURES_FILE
        targets_path = batch_dir / TARGETS_FILE
        # Validated per timeframe when stage 3 hands back a mapping, which it
        # now does: the union of timeframes is ~11 GiB at 110 tickers and is
        # never built in memory, only written.
        def validate(frames, name):
            if isinstance(frames, dict):
                return {
                    timeframe: self._validate_batch_frame(
                        frame,
                        frame_name=f"{name} {timeframe}",
                        requested_timeframes=[timeframe],
                    )
                    for timeframe, frame in frames.items()
                }
            return self._validate_batch_frame(
                frames, frame_name=name,
                requested_timeframes=config.timeframes,
            )

        features_df = validate(features_df, "features")
        targets_df = validate(targets_df, "targets")

        if isinstance(features_df, dict):
            return self._save_frames_by_timeframe(
                features_df, targets_df, features_path, targets_path, config,
            )

        # Accumulation is worth doing only when the file on disk actually holds
        # rows this batch does not. It usually holds none, because
        # `pipeline_runner` has ALREADY written this run's output to these very
        # paths by the time this runs -- so "existing" is the new batch, and the
        # concat below was the batch joined to itself.
        #
        # On 2026-08-22 that ended the run: 259,133 rows became 518,266 x 2,238,
        # and `drop_duplicates` -- which takes a fresh copy through boolean
        # indexing -- died with "Unable to allocate 437. MiB", five minutes
        # after the batch had already been safely written. The dedup would then
        # have removed precisely the rows the concat had just added.
        #
        # So the keys are read first. Three columns instead of 2,238 costs a
        # few MiB, and when nothing new is found the whole load-concat-dedup is
        # skipped. This keeps real accumulation working -- a previous batch with
        # other tickers or older bars still merges -- while the ordinary case
        # stops paying 9 GiB to rediscover that it has nothing to do.
        if config.accumulate and features_path.exists() and targets_path.exists():
            carried_over = self._rows_not_already_present(features_path, features_df)
        else:
            carried_over = 0

        if config.accumulate and carried_over:
            logger.info(
                "Accumulating: %d row(s) on disk are not in this batch.",
                carried_over,
            )
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

    @staticmethod
    def _rows_not_already_present(
        existing_path: Path,
        incoming: pd.DataFrame,
    ) -> int:
        """How many rows on disk this batch does not already carry.

        Reads the identity columns ONLY -- three of 2,238 -- so asking the
        question costs a few MiB instead of the 4.6 GiB the whole frame needs.
        The answer decides whether the expensive path is worth entering at all.

        Returns 0 when it cannot tell, which keeps the old behaviour: an
        unreadable or differently-keyed file falls through to the full
        accumulate rather than silently dropping rows.
        """
        keys = [k for k in ("datetime", "ticker", "interval") if k in incoming.columns]
        if not keys:
            return 0
        try:
            available = set(pq.ParquetFile(existing_path).schema_arrow.names)
            if not set(keys).issubset(available):
                return 0
            on_disk = pd.read_parquet(existing_path, columns=keys)
        except (OSError, ValueError, pa.ArrowInvalid):
            # Not readable as parquet, or the schema moved. Say "cannot tell".
            return 0

        # Normalise the datetime keys before comparing them. The file on disk
        # and the frame in memory do not have to agree on timezone: pandas
        # refuses outright with "You are trying to merge on datetime64[ns] and
        # datetime64[ns, UTC] columns", and that killed a two-and-a-half-hour
        # rebuild on 2026-08-23 -- after the batch was already safely written,
        # so the work survived and the run reported failure.
        left = on_disk.copy(deep=False)
        right = incoming[keys].drop_duplicates().copy(deep=False)
        for frame in (left, right):
            for key in keys:
                if pd.api.types.is_datetime64_any_dtype(frame[key]):
                    parsed = pd.to_datetime(frame[key], errors="coerce", utc=True)
                    frame[key] = parsed.dt.tz_convert("UTC").dt.tz_localize(None)

        merged = left.merge(
            right,
            on=keys,
            how="left",
            indicator=True,
        )
        return int((merged["_merge"] == "left_only").sum())

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

        block_on_forbidden=True: a forbidden column is a hard stop, not a
        cleanup-and-continue - ValueError is deliberately NOT caught below,
        so it propagates out of this method (and isn't swallowed further
        up the call chain either) and the batch is never written to
        Parquet. Verified against 7 real production batches (accumulated
        + regenerated, multiple tickers/timeframes) before enabling: all
        currently clean, so this does not block anything that was passing
        before.
        """
        guard = FeatureLeakageGuard(block_on_forbidden=True)
        try:
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

            # Checked one ticker at a time. This used to be a single
            # `pd.concat([features_df, targets_df[target_cols]], axis=1)` over
            # the whole batch, and on the 2026-08-11 rebuild that killed the
            # run outright at the very last step, after 100 minutes of work:
            #   MemoryError: Unable to allocate 801 MiB for an array with
            #   shape (1940, 54099)
            # concat has to consolidate, and the enrichers leave the frame in
            # thousands of single-column blocks (the "highly fragmented"
            # warnings ContextMapEnricher emits), so one contiguous float64
            # block of every feature x every row is demanded at once.
            #
            # Nothing needs that frame to exist. FeatureLeakageGuard already
            # groups by ticker internally and measures leakage per instrument
            # (deliberately -- see _check_correlation_per_ticker: the worst
            # instrument decides, pooling would let a leak hide). Slicing per
            # ticker first is therefore the same computation with peak memory
            # divided by the ticker count, and forbidden columns are unioned
            # so the outcome is identical.
            forbidden: list[str] = []
            high_corr: dict[str, Any] = {}
            if 'ticker' in features_df.columns:
                groups = list(features_df.groupby('ticker', sort=False).groups.items())
            else:
                groups = [('unknown', features_df.index)]

            for ticker_name, row_index in groups:
                feature_slice = features_df.loc[row_index]
                if target_cols:
                    # Sliced by INDEX LABEL, never by position. The concat this
                    # replaces aligned on the index, so reindexing preserves
                    # its semantics exactly; taking .iloc positions here would
                    # silently pair each ticker's features with whatever
                    # targets happened to sit at those offsets -- the same
                    # class of defect that put bars on the wrong dates in the
                    # 2026-08-06 batch.
                    target_slice = targets_df[target_cols].reindex(feature_slice.index)
                    combined = pd.concat([feature_slice, target_slice], axis=1)
                else:
                    combined = feature_slice

                report = guard.check(
                    combined,
                    target_cols=target_cols if target_cols else None,
                    ticker=str(ticker_name),
                )
                for column in report.forbidden_cols or []:
                    if column not in forbidden:
                        forbidden.append(column)
                for column, value in (report.high_corr_cols or {}).items():
                    high_corr.setdefault(column, value)
                del combined

            if forbidden or high_corr:
                if forbidden:
                    logger.warning(
                        f"[LeakageGuard] Removing {len(forbidden)} forbidden columns: "
                        f"{forbidden[:5]}{'...' if len(forbidden) > 5 else ''}"
                    )
                    features_df = features_df.drop(columns=forbidden, errors='ignore')

                if high_corr:
                    logger.warning(
                        f"[LeakageGuard] {len(high_corr)} features with high target "
                        f"correlation. Review: {list(high_corr.keys())[:5]}"
                    )
            else:
                logger.debug("[LeakageGuard] No leakage detected.")

        except (TypeError, AttributeError, KeyError, ZeroDivisionError) as e:
            logger.warning(f"[LeakageGuard] Check failed (non-blocking): {e}")

        return features_df

    def _save_df_to_parquet(self, df: pd.DataFrame, path: Path):
        """Saves DataFrame to Parquet, ensuring datetime column is preserved.

        The copy is made only when the frame actually needs changing. It used
        to be unconditional, and at this batch's width that is an 840 MB
        duplicate of a frame that is about to be written unmodified -- on the
        2026-08-11 rebuild the run had already died one step earlier for want
        of 801 MB, so an avoidable full copy here is a second landmine on the
        same path. `to_parquet(index=False)` does not mutate its input, so
        there is nothing to protect against when no reset is required.
        """
        needs_datetime_from_index = (
            'datetime' not in df.columns
            and isinstance(df.index, pd.DatetimeIndex)
        )
        if not needs_datetime_from_index:
            df.to_parquet(path, index=False)
            return

        df_to_save = df.reset_index()
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

    @staticmethod
    def _delivered_timeframes(f_df: pd.DataFrame) -> set[str]:
        """Timeframes actually present in the exported features."""
        column = next(
            (c for c in ('interval', 'timeframe') if c in f_df.columns), None
        )
        if column is None:
            return set()
        return {
            tf for tf in (
                normalize_timeframe(value) for value in f_df[column].dropna().unique()
            ) if tf
        }

    @staticmethod
    def _missing_timeframes(requested: list[str], delivered: set[str]) -> list[str]:
        """Requested timeframes with no rows, compared on normalised names.

        '1h' and '60m' are the same timeframe under two spellings -- the
        request says 1h, the data says 60m -- so a raw set difference reports
        a phantom gap and hides the real one.
        """
        return sorted(
            original for original in requested
            if normalize_timeframe(original) not in delivered
        )

    def _create_batch_metadata(self, name: str, timestamp: str, config: BatchPreparationConfig,
                              f_df: pd.DataFrame, t_df: pd.DataFrame,
                              f_path: Path, t_path: Path, c_path: Path | None) -> dict[str, Any]:
        """Creates the batch metadata dictionary."""
        delivered = self._delivered_timeframes(f_df)
        requested = [str(tf) for tf in (config.timeframes or [])]
        missing = self._missing_timeframes(requested, delivered)
        if missing:
            # A requested timeframe that produced no rows is a third of the
            # run silently absent. The 2026-08-04 batch recorded
            # timeframes: ['15m', '1d', '1h'] while features.parquet held
            # only 1d and 60m, and targets.parquet carried no 15m target at
            # all -- nothing said so, and every downstream stage reported
            # success on two thirds of the requested scope.
            logger.error(
                "Batch '%s' was asked for timeframe(s) %s and produced NONE. "
                "Delivered: %s. Nothing downstream will mention this again -- "
                "no features, no targets, no champions for them.",
                name, missing, sorted(delivered) or '(none)',
            )
        return {
            'batch_name': name,
            'timestamp': timestamp,
            'tickers': config.tickers,
            'timeframes': requested,
            # Requested and delivered, side by side, because they are not the
            # same question and the metadata answered only the first.
            'timeframes_delivered': sorted(delivered),
            'timeframes_missing': missing,
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
        self._prefer_latest_local_run(batch_dir, results)
        return results

    def _prefer_latest_local_run(self, batch_dir: Path, results: dict[str, Any]) -> None:
        """Local training writes somewhere none of the names above look.

        `--mode light` appends each run to `light_models_results.json` as
        `runs[]`, newest last. Nothing in `files_to_load` reads that name, so
        `--mode continue` fell through to `colab_results.json` -- which is
        written once and then never again.

        On 2026-08-23 that meant the freshest champions, 97 of them from the
        run that had just finished, sat in the directory while continue mode
        would have carried 660 models from 2026-08-08 into stages 5 to 7. Two
        weeks of drift, no error, and a result that reads as current.

        Preference is decided by the run's own timestamp rather than by which
        key was assigned last. Dict ordering deciding which artifact wins is
        the kind of mechanism that breaks silently when someone reorders a
        literal.
        """
        path = batch_dir / "light_models_results.json"
        if not path.exists():
            return
        try:
            with open(path, encoding="utf-8") as handle:
                payload = json.load(handle)
        except (OSError, ValueError) as error:
            self.logger.warning("Could not read %s: %s", path.name, error)
            return

        runs = [run for run in (payload.get("runs") or [])
                if isinstance(run, dict) and run.get("models_metadata")]
        if not runs:
            return
        latest = max(runs, key=lambda run: str(run.get("timestamp") or ""))
        metadata = latest["models_metadata"]

        displaced = len(results.get("models_metadata") or {})
        results["models_metadata"] = metadata
        self.logger.info(
            "Using models_metadata from the latest local run %s (%d models)%s.",
            latest.get("timestamp"), len(metadata),
            f", replacing {displaced} loaded from an older artifact" if displaced else "",
        )

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
                data = data['models_metadata']
            if key in results and isinstance(results[key], dict) and isinstance(data, dict):
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
