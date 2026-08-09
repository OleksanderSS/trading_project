"""
Pipeline execution utilities for hybrid pipeline.
"""
import functools
import hashlib
import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
 
from src.core.logging.logger import ProjectLogger
from src.pipeline.target_column_utils import is_direct_target_column, split_model_features_and_targets

logger = ProjectLogger.get_logger(__name__)

_FINGERPRINT_FILE = 'raw_db_fingerprint.json'
# Tables tracked for change detection (fallback if config unavailable)
_DEFAULT_TRACKED_TABLES = [
    'news_articles', 'google_news', 'rss_news', 'newsapi_articles',
    'sec_filings', 'hugging_face_news',
    'market_data_raw', 'market_data',
    'fred_data', 'economic_calendar',
]

def profile_execution(func):
    """Decorator to log execution time of async functions."""
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = await func(*args, **kwargs)
        end_time = time.perf_counter()
        logger.info(f"⏱️ {func.__name__} took {end_time - start_time:.2f} seconds")
        return result
    return wrapper

FEATURES_FILE = 'features.parquet'
TARGETS_FILE = 'targets.parquet'


class PipelineExecutor:
    """Handles pipeline execution for different modes."""

    # Source trees whose content determines what stages 0-3 actually
    # compute. Hashed together into the cache fingerprint (see
    # _compute_code_fingerprint) so a code change — a bug fix, a leakage
    # fix, a new diagnostic — invalidates a stale features.parquet cache
    # even when the raw DB hasn't grown. Previously the cache was purely
    # data-driven, which meant Stage 3 logic changes were silently skipped
    # by cached (pre-fix) features unless someone manually deleted
    # features.parquet/targets.parquet/the fingerprint file first.
    _CODE_FINGERPRINT_DIRS = (
        'src/pipeline/stages',
        'src/features',
        'src/analytics/calculators',
        # Added 2026-08-02. Both were holes in exactly the guarantee this
        # mechanism exists to provide:
        #
        # src/targets — stages/feature_engineering/targets.py is covered, but
        #   it only delegates: TargetOrchestrator and timeframe_contract, which
        #   decide every value in the cached targets.parquet, live here. A fix
        #   to target horizons or boundary masking was silently skipped by a
        #   stale cache.
        # src/processing — Stage 2 cleaning and filtering produce the frame
        #   Stage 3 enriches, so its logic determines the cached features just
        #   as directly as an enricher does.
        'src/targets',
        'src/processing',
        # Added 2026-08-05, and this one closes a CIRCLE rather than a gap.
        #
        # The cache check gates ALL of stages 0-3, collection included, on
        # whether the database grew. But the database only grows if
        # collection runs. So a broken collector produces no new data, the
        # fingerprint stays put, the cache reports "no new data", collection
        # is skipped -- and a fix to the collector can never take effect,
        # because the thing it fixes is the thing being skipped.
        #
        # Observed exactly that: 7395f88c fixed a delisted ticker discarding
        # every other ticker's download, and the next prepare run finished in
        # 102 seconds without collecting anything, reporting success. The
        # database had not gained a row since 2026-07-30.
        #
        # A collector change alters what data arrives, which alters the
        # features built from it, so it belongs in the fingerprint on the
        # same grounds as everything above.
        'src/data/collectors',
        'src/data/validation',
    )

    #: Individual modules outside those trees that the cached stages import
    #: and that decide what the cached artifact CONTAINS.
    #:
    #: Added 2026-08-09. Only src/pipeline/stages is hashed, so
    #: src/pipeline/target_column_utils.py was invisible -- while
    #: stages/feature_engineering/orchestrator.py and targets.py import it to
    #: decide which columns are features, which are targets and which are
    #: identifiers. 069a4341 changed exactly that (a ctx_ prefix was hiding
    #: three identity columns), and the fingerprint would not have moved: a
    #: rebuild would have been skipped and the old split kept.
    #:
    #: timeframe_lineage.py is here for the same reason -- normalize_timeframe
    #: and is_timeframe_token decide how bars are grouped and which suffix a
    #: column gets.
    #:
    #: A directory would be the tidier rule, but src/pipeline also holds the
    #: orchestrators, which run stages 4-7 and belong out (see
    #: _NON_CACHED_STAGE_DIRS). Naming the files keeps the boundary honest;
    #: the test below pins it against what the stages actually import.
    _FINGERPRINT_FILES = (
        'src/pipeline/target_column_utils.py',
        'src/pipeline/timeframe_lineage.py',
    )

    #: Stage subtrees that CANNOT affect the cached artifact, because the
    #: cache gates stages 0-3 only. Everything under 'src/pipeline/stages'
    #: is hashed except these.
    #:
    #: Added 2026-08-06. Hashing them made every Stage 4/5 fix look like a
    #: features change: bb7faa06 touched stages/modeling and
    #: stages/prediction and moved the fingerprint, so the next prepare would
    #: have re-collected and re-enriched for hours without altering a single
    #: feature value. A cache invalidation that fires when nothing it
    #: protects has changed is the kind people learn to work around, and the
    #: workaround is what this mechanism exists to prevent.
    #:
    #: Stated as an EXCLUSION on purpose. A new stage subdirectory is then
    #: covered by default: the failure mode of over-hashing is wasted time,
    #: and the failure mode of under-hashing is a stale cache serving
    #: features built by code that has since been fixed -- which this project
    #: has already lived through (see the collector note above).
    _STAGES_ROOT = 'src/pipeline/stages'
    _NON_CACHED_STAGE_DIRS = (
        'modeling',      # stage 4
        'prediction',    # stage 5
        'trading',       # stage 6
        'evaluation',    # stage 7
        'monitoring',    # runs alongside, reads results
    )

    #: Config decides WHICH enrichers, analyzers and targets run at all, so a
    #: YAML edit changes the cached output as surely as a code edit. Only .py
    #: was hashed before.
    _CONFIG_FINGERPRINT_DIR = 'src/config'

    #: Written BY the pipeline (ContextRuleGenerator._save_rules_to_yaml).
    #: Hashing a file the run itself produces would move the fingerprint on
    #: every run and turn the cache into a permanent miss.
    _GENERATED_CONFIG_PREFIXES = ('generated_',)

    @staticmethod
    def _is_non_cached_stage(path: Path, project_root: Path) -> bool:
        """True for files under a stage that runs after the cached artifact."""
        stages_root = project_root / PipelineExecutor._STAGES_ROOT
        try:
            relative = path.relative_to(stages_root)
        except ValueError:
            return False
        return relative.parts[0] in PipelineExecutor._NON_CACHED_STAGE_DIRS

    @staticmethod
    def _fingerprint_files(project_root: Path) -> list[Path]:
        """Every file whose content determines what stages 0-3 compute."""
        paths: list[Path] = []
        for rel_dir in PipelineExecutor._CODE_FINGERPRINT_DIRS:
            dir_path = project_root / rel_dir
            if dir_path.exists():
                paths.extend(
                    path for path in dir_path.rglob('*.py')
                    if not PipelineExecutor._is_non_cached_stage(
                        path, project_root
                    )
                )

        for rel_file in PipelineExecutor._FINGERPRINT_FILES:
            file_path = project_root / rel_file
            if file_path.exists():
                paths.append(file_path)

        config_dir = project_root / PipelineExecutor._CONFIG_FINGERPRINT_DIR
        if config_dir.exists():
            paths.extend(
                path for path in config_dir.rglob('*.yaml')
                if not path.name.startswith(
                    PipelineExecutor._GENERATED_CONFIG_PREFIXES
                )
            )
        return paths

    @staticmethod
    def _compute_code_fingerprint() -> str:
        """SHA-256 over every relevant file's path + content, in sorted
        (deterministic) order."""
        project_root = Path(__file__).resolve().parent.parent.parent
        hasher = hashlib.sha256()
        for path in sorted(PipelineExecutor._fingerprint_files(project_root)):
            try:
                hasher.update(str(path.relative_to(project_root)).replace('\\', '/').encode('utf-8'))
                hasher.update(path.read_bytes())
            except OSError as e:
                logger.warning(f'Code fingerprint: could not read {path}: {e}')
        return hasher.hexdigest()

    @staticmethod
    @profile_execution
    async def execute_local_mode(orchestrator, tickers: list, timeframes: list
        ):
        """Execute local pipeline stages only."""
        logger.info('Running local pipeline (stages 0-3)...')
        return await orchestrator.run_local_pipeline(tickers=tickers,
            timeframes=timeframes)

    @staticmethod
    @profile_execution
    async def execute_light_mode(orchestrator, tickers: list, timeframes: list
        ):
        """Execute light models training only."""
        logger.info('Running light models training...')
        return await orchestrator.run_light_models(tickers=tickers)

    @staticmethod
    @profile_execution
    async def execute_prepare_mode(orchestrator, tickers: list, timeframes:
        list, **kwargs):
        """Execute preparation for Colab (stages 0-3 + packaging)."""
        logger.info('Preparing data for Colab training...')

        tracker = PipelineExecutor._enable_lineage_tracking_for_run()

        # Check cache before running pipeline
        cached_data = PipelineExecutor._check_cache_before_run(orchestrator)
        if cached_data is not None:
            features_df, targets_df = cached_data
        else:
            logger.info("🔄 No valid cache found - running pipeline stages 0-3")
            features_df, targets_df = await PipelineExecutor._run_local_pipeline_and_extract_data(
                orchestrator, tickers, timeframes
            )
            # Persist DB fingerprint so next run can detect new data automatically
            fp, table_states = PipelineExecutor._compute_db_fingerprint(orchestrator)
            PipelineExecutor._save_db_fingerprint(orchestrator.config.output_dir, fp, table_states)

        PipelineExecutor._capture_final_features(tracker, features_df)

        result = await orchestrator.prepare_colab_data(tickers=tickers,
            timeframes=timeframes, features_df=features_df,
            targets_df=targets_df, **kwargs)

        PipelineExecutor._mark_model_input(tracker, features_df)

        PipelineExecutor._disable_lineage_tracking()
        return result

    @staticmethod
    def _compute_db_fingerprint(orchestrator) -> tuple[str, dict]:
        """
        Compute a SHA-256 fingerprint of raw DB table states.

        Fingerprint captures COUNT(*) and MAX(date) per tracked table.
        If any table grows (new rows accumulated), fingerprint changes,
        triggering a full stages 0-3 re-run.

        Raw data (news, prices, macro) is a permanent chronicle — it never
        expires. Old news had valid market impact at the time they were
        published and remain valid training samples forever.
        """
        try:
            from src.data.management.data_manager import DataManager
            db_manager = DataManager(orchestrator.config_manager)

            # Prefer tracked_tables from config, fallback to defaults
            cfg = orchestrator.config_manager.get_config('cache') or {}
            tracked = cfg.get('tracked_tables', _DEFAULT_TRACKED_TABLES)
            # Also include any table that exists in DB and looks like raw data
            try:
                all_tables = db_manager.get_all_table_names()
                extra = [
                    t for t in all_tables
                    if t not in tracked and t != 'cache_metadata'
                    and any(kw in t for kw in ('news', 'market', 'fred', 'economic', 'sec', 'rss'))
                ]
                tracked = list(dict.fromkeys(tracked + extra))  # preserve order, deduplicate
            except Exception as e:
                logger.warning(f"Failed to fetch extra cache tables: {e}")

            table_states = {}
            state_parts = []
            for table in tracked:
                try:
                    if not db_manager.table_exists(table):
                        continue
                    quoted = f'"{table.replace(chr(34), "")}"'
                    count = (db_manager.fetch_one(f'SELECT COUNT(*) as c FROM {quoted}') or {}).get('c', 0)

                    schema = db_manager.get_table_schema(table)
                    date_col = next(
                        (c for c in ('published_at', 'published_date', 'created_at', 'timestamp', 'datetime', 'date')
                         if c in schema),
                        None,
                    )
                    max_date = 'no_date'
                    if date_col:
                        quoted_col = f'"{date_col.replace(chr(34), "")}"'
                        res = db_manager.fetch_one(f'SELECT MAX({quoted_col}) as m FROM {quoted}')
                        max_date = str((res or {}).get('m', 'null'))

                    table_states[table] = {'count': count, 'max_date': max_date}
                    state_parts.append(f'{table}:{count}:{max_date}')
                except Exception as e:
                    logger.debug(f'Fingerprint: skipped table {table}: {e}')

            fingerprint = hashlib.sha256('|'.join(state_parts).encode()).hexdigest()
            return fingerprint, table_states
        except Exception as e:
            logger.warning(f'Could not compute DB fingerprint: {e}')
            return 'unavailable', {}

    @staticmethod
    def _save_db_fingerprint(output_dir: Path, fingerprint: str, table_states: dict) -> None:
        """Persist the current DB + code fingerprint alongside features.parquet."""
        meta = {
            'fingerprint': fingerprint,
            'code_fingerprint': PipelineExecutor._compute_code_fingerprint(),
            'generated_at': datetime.now().isoformat(),
            'table_states': table_states,
        }
        try:
            fp_path = output_dir / _FINGERPRINT_FILE
            fp_path.write_text(json.dumps(meta, indent=2, default=str), encoding='utf-8')
            logger.info(f'💾 DB fingerprint saved: {fingerprint[:16]}… ({len(table_states)} tables tracked)')
        except Exception as e:
            logger.warning(f'Could not save DB fingerprint: {e}')

    @staticmethod
    def _check_cache_before_run(orchestrator) -> tuple[pd.DataFrame, pd.DataFrame] | None:
        """
        Data- and code-driven cache check:
        1. features.parquet must exist and be non-empty.
        2. DB fingerprint must match what was used to generate those features.
           → If new rows accumulated in raw tables since last run, re-run stages 0-3.
        3. Code fingerprint (stages 0-3 + feature/analytics source) must
           match too → a logic change (bug fix, leakage fix, new
           diagnostic) invalidates the cache even when the raw DB hasn't
           changed, so cached features are never silently stale relative
           to the code that would now produce them.
        """
        output_dir = orchestrator.config.output_dir
        features_path = output_dir / 'features.parquet'
        targets_path = output_dir / 'targets.parquet'
        fp_path = output_dir / _FINGERPRINT_FILE

        # Step 1: parquet files must exist
        if not features_path.exists() or not targets_path.exists():
            logger.info('Cache: features/targets not found — running full pipeline.')
            return None

        # Step 2: fingerprint file must exist (written after every successful prepare)
        if not fp_path.exists():
            logger.info('Cache: no DB fingerprint on record — running full pipeline to establish baseline.')
            return None

        # Step 3: compare current DB state with saved fingerprint
        try:
            saved_meta = json.loads(fp_path.read_text(encoding='utf-8'))
            saved_fp = saved_meta.get('fingerprint', '')
            saved_code_fp = saved_meta.get('code_fingerprint')
            saved_states = saved_meta.get('table_states', {})
            saved_at = saved_meta.get('generated_at', 'unknown')
        except Exception as e:
            logger.warning(f'Cache: could not read fingerprint file ({e}) — running full pipeline.')
            return None

        # Step 2b: code fingerprint. saved_code_fp is None for
        # fingerprint files written before this check existed — treated as
        # a mismatch (conservative: re-run once to establish a code
        # baseline) rather than silently trusting pre-existing cache files
        # of unknown code provenance.
        current_code_fp = PipelineExecutor._compute_code_fingerprint()
        if saved_code_fp != current_code_fp:
            logger.info(
                'Cache: stages 0-3 source code changed since this cache was generated '
                f'(cached {saved_at}) — re-running stages 0-3.'
            )
            return None

        current_fp, current_states = PipelineExecutor._compute_db_fingerprint(orchestrator)

        if current_fp == 'unavailable':
            # DB unreachable — be conservative, use cache
            logger.warning('Cache: DB fingerprint unavailable — using cached features (conservative).')
        elif current_fp != saved_fp:
            # Find which tables grew
            changed = [
                f'{t}: {saved_states.get(t, {}).get("count", "?")} → {v["count"]}'
                for t, v in current_states.items()
                if str(v.get('count')) != str((saved_states.get(t) or {}).get('count', ''))
            ]
            logger.info(
                f'🔄 New data detected since last prepare ({saved_at}) — re-running stages 0-3.'
                + (f' Changed tables: {changed}' if changed else '')
            )
            return None

        # Step 4: load and validate cached features
        try:
            features_df = pd.read_parquet(features_path)
            targets_df = pd.read_parquet(targets_path)
            if features_df.empty or targets_df.empty:
                logger.warning('Cache: parquet files are empty — running full pipeline.')
                return None
            logger.info(
                f'✅ No new data since last prepare ({saved_at}) — using cached features '
                f'features={features_df.shape}, targets={targets_df.shape}'
            )
            return features_df, targets_df
        except Exception as e:
            logger.warning(f'Cache: error reading parquet files ({e}) — running full pipeline.')
            return None

    @staticmethod
    def _enable_lineage_tracking_for_run():
        """Enable lineage tracking for Colab preparation run."""
        from src.features.feature_orchestrator import enable_lineage_tracking
        return enable_lineage_tracking("diagnostic_reports/feature_lineage_report.json")

    @staticmethod
    def _disable_lineage_tracking():
        """Disable lineage tracking and save report."""
        from src.features.feature_orchestrator import disable_lineage_tracking
        disable_lineage_tracking()

    @staticmethod
    async def _run_local_pipeline_and_extract_data(orchestrator, tickers: list, timeframes: list):
        """Run local pipeline and extract features/targets dataframes."""
        local_results = await orchestrator.run_local_pipeline(tickers=tickers,
            timeframes=timeframes)
        results_data = local_results.get('results', {})
        features_df = results_data.get('features_df', pd.DataFrame())
        targets_df = results_data.get('targets_df', pd.DataFrame())

        # If features/targets are empty — cascade fallbacks:
        # 1. saved_files['features'] from this run (written by feature_processor)
        # 2. saved_files['cleaned_data'] from this run
        # 3. data/processed/features/ — persistent processed storage
        if features_df.empty and targets_df.empty:
            saved_files = local_results.get('saved_files', {})
            features_df, targets_df = PipelineExecutor._load_features_targets_with_fallbacks(
                saved_files
            )

        logger.info(f'Local pipeline complete: features={features_df.shape}, targets={targets_df.shape}')
        return features_df, targets_df

    @staticmethod
    def _load_features_targets_with_fallbacks(saved_files: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load features and targets from saved files with multiple fallbacks.

        Priority:
        1. saved_files['features'] — written by feature_processor in this run
        2. saved_files['cleaned_data'] — stage2 output, split into features/targets
        3. data/processed/features/ — persistent storage from previous runs
        """
        features_df = pd.DataFrame()
        targets_df = pd.DataFrame()

        # --- Fallback 1: explicit features path saved by feature_processor ---
        features_path = saved_files.get('features')
        targets_path = saved_files.get('targets')
        if features_path:
            features_df = PipelineExecutor._try_load_parquet(features_path, "saved features")
        if targets_path:
            targets_df = PipelineExecutor._try_load_parquet(targets_path, "saved targets")

        if not features_df.empty and not targets_df.empty:
            return features_df, targets_df

        # --- Fallback 2: cleaned_data from stage2 (split into features/targets) ---
        cleaned_data_path_str = saved_files.get('cleaned_data')
        if cleaned_data_path_str:
            cleaned_data = PipelineExecutor._try_load_pickle_or_parquet(cleaned_data_path_str, "cleaned_data")
            if cleaned_data is not None:
                # Handle dict (nested DataFrames) or DataFrame
                if isinstance(cleaned_data, dict) and 'prices' in cleaned_data:
                    # Extract 1d prices from dict
                    prices_dict = cleaned_data['prices']
                    if '1d' in prices_dict and isinstance(prices_dict['1d'], pd.DataFrame):
                        cleaned_df = prices_dict['1d']
                    else:
                        # Use first available timeframe
                        for tf, df in prices_dict.items():
                            if isinstance(df, pd.DataFrame) and not df.empty:
                                cleaned_df = df
                                break
                        else:
                            cleaned_df = pd.DataFrame()
                elif isinstance(cleaned_data, pd.DataFrame):
                    cleaned_df = cleaned_data
                else:
                    cleaned_df = pd.DataFrame()

                if not cleaned_df.empty:
                    feature_cols, target_cols, dropped = split_model_features_and_targets(cleaned_df.columns)
                    if dropped:
                        logger.warning(
                            "Dropped %s target-derived column(s) from features: %s",
                            len(dropped), list(dropped)[:5],
                        )
                    if feature_cols:
                        features_df = cleaned_df[feature_cols]
                    if target_cols:
                        targets_df = cleaned_df[target_cols]
                    logger.info(f'Loaded data from cleaned_data: features={features_df.shape}, targets={targets_df.shape}')

        if not features_df.empty and not targets_df.empty:
            return features_df, targets_df

        # --- Fallback 3: persistent data/processed/features/ ---
        processed_features = Path("data/processed/features/features.parquet")
        processed_targets = Path("data/processed/features/targets.parquet")

        if features_df.empty and processed_features.exists():
            loaded = PipelineExecutor._try_load_parquet(str(processed_features), "processed features")
            if not loaded.empty:
                features_df = loaded
                logger.info(f'Fallback to processed features: {features_df.shape}')

        if targets_df.empty and processed_targets.exists():
            loaded = PipelineExecutor._try_load_parquet(str(processed_targets), "processed targets")
            if not loaded.empty:
                targets_df = loaded
                logger.info(f'Fallback to processed targets: {targets_df.shape}')

        return features_df, targets_df

    @staticmethod
    def _try_load_parquet(path_str: str, label: str) -> pd.DataFrame:
        """
        Safely load a parquet file. Returns empty DataFrame on any error
        (including corrupted files that raise pyarrow/OSError exceptions).
        """
        path = Path(path_str)
        if not path.exists():
            return pd.DataFrame()
        try:
            df = pd.read_parquet(path)
            if not df.empty:
                logger.info(f'Loaded {label}: {df.shape} from {path.name}')
            return df
        except Exception as e:  # noqa: BLE001 — intentional broad catch for corrupted files
            logger.warning(f'Failed to load {label} from {path}: {e}')
            return pd.DataFrame()

    @staticmethod
    def _try_load_pickle_or_parquet(path_str: str, label: str) -> Any:
        """
        Try to load as pickle first (for cleaned_data dict), then as parquet.
        Returns loaded data or None on failure.
        """
        import pickle

        path = Path(path_str)
        if not path.exists():
            return None

        # Try pickle first (for cleaned_data dict)
        try:
            with open(path, 'rb') as f:
                data = pickle.load(f)
            logger.info(f'Loaded {label} from {path.name} as pickle')
            return data
        except Exception:
            # Not a pickle file, try parquet
            pass

        # Try parquet
        try:
            df = pd.read_parquet(path)
            if not df.empty:
                logger.info(f'Loaded {label}: {df.shape} from {path.name} as parquet')
            return df
        except Exception as e:  # noqa: BLE001 — intentional broad catch
            logger.warning(f'Failed to load {label} from {path}: {e}')
            return None

    @staticmethod
    def _capture_final_features(tracker, features_df):
        """Captures final features for lineage report."""
        if tracker is not None and not features_df.empty:
            try:
                tracker.capture_step("final_features", features_df)
                tracker.mark_model_input(features_df)
            except (AttributeError, TypeError, RuntimeError, KeyError) as e:
                logger.warning(f"[Lineage] Could not capture step: {e}")

    @staticmethod
    def _mark_model_input(tracker, features_df):
        """Marks final features as model input."""
        if tracker is not None:
            try:
                final_features = features_df
                if final_features.empty:
                    final_features = PipelineExecutor._load_features_from_parquet()

                if not final_features.empty:
                    tracker.capture_step("model_input", final_features)
                    tracker.mark_model_input(final_features)
            except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError) as e:
                logger.debug(f"[Lineage] Could not mark model input: {e}")

    @staticmethod
    def _load_features_from_parquet():
        """Loads features from Parquet if features_df is empty."""
        features_path = Path("data/processed/features/features.parquet")
        if not features_path.exists():
            colab_dirs = sorted(Path("data/colab/accumulated").glob("*/features.parquet"))
            if colab_dirs:
                features_path = colab_dirs[-1]

        if features_path.exists():
            final_features = pd.read_parquet(features_path)
            logger.info(f"[Lineage] Loaded features from Parquet: {final_features.shape}")
            return final_features
        return pd.DataFrame()

    @staticmethod
    @profile_execution
    async def execute_full_mode(orchestrator, tickers: list, timeframes: list):
        """Execute the full hybrid preparation flow and pause for Colab."""
        logger.info('Running full hybrid pipeline...')
        from src.pipeline.hybrid_orchestrator import HybridPipelineRequest
        return await orchestrator.run_full_hybrid_pipeline(
            HybridPipelineRequest(tickers=tickers, timeframes=timeframes,
            accumulate=True))

    @staticmethod
    @profile_execution
    async def execute_continue_mode(orchestrator, args):
        """Execute the continue mode after Colab results are ready."""
        batch_name_sanitized = PipelineExecutor._sanitize(getattr(args, 'batch_name', 'unknown'))
        logger.info(f"Running continue mode for batch: {batch_name_sanitized}...")

        # 1. Contract validation
        val_report = PipelineExecutor._validate_batch_contract(orchestrator)
        if not val_report['valid']:
            return {'status': 'failed', 'reason': 'contract_validation_failed', 'errors': val_report['errors']}

        manifest = val_report['manifest']
        PipelineExecutor._log_manifest_details(manifest)

        # 2. Load and validate data
        continue_data = PipelineExecutor._load_continue_data(orchestrator, args)
        (features_df, targets_df, colab_results, news_data, economic_data) = continue_data

        validation_error = PipelineExecutor._validate_continue_inputs(
            features_df, targets_df, colab_results, getattr(args, 'batch_name', 'unknown')
        )
        if validation_error:
            return validation_error

        # 3. Resolve tickers and run light training
        tickers = PipelineExecutor._resolve_tickers(args, colab_results, features_df)
        
        # Filter DataFrames to only include the resolved tickers
        if tickers and hasattr(features_df, 'empty') and not features_df.empty and 'ticker' in features_df.columns:
            features_df = features_df[features_df['ticker'].isin(tickers)].copy()
        if tickers and hasattr(targets_df, 'empty') and not targets_df.empty and 'ticker' in targets_df.columns:
            targets_df = targets_df[targets_df['ticker'].isin(tickers)].copy()
            
        logger.info(f"Resolved tickers for continue mode: {tickers}")
        logger.info("About to run light training for continue mode...")
        light_results = await PipelineExecutor._run_light_training_for_continue(
            orchestrator, features_df, targets_df, tickers, args
        )
        logger.info(f"Light training results: {light_results}")

        # 4. Run final stages (5-7) after light training
        logger.info("Running final stages (5-7) after light training...")
        final_results = await PipelineExecutor._run_final_stages_for_continue(
            orchestrator,
            features_df,
            targets_df,
            colab_results,
            light_results,
            tickers,
            manifest,
            news_data,
            economic_data,
            args
        )
        logger.info("Final stages completed")

        return {
            'status': 'completed',
            'light_results': light_results,
            'final_results': final_results
        }

    @staticmethod
    async def _run_light_training_for_continue(orchestrator, features_df, targets_df, tickers, args):
        """Run light models training for continue mode."""
        return await orchestrator.run_light_models(
            features_df=features_df,
            targets_df=targets_df,
            tickers=tickers,
            test_ticker=getattr(args, 'test_ticker', None),
            test_target=getattr(args, 'test_target', None),
            batch_name=getattr(args, 'batch_name', None)
        )

    @staticmethod
    async def _run_final_stages_for_continue(orchestrator, features_df, targets_df, colab_results, light_results, tickers, manifest, news_data, economic_data, args):
        """Run final stages for continue mode."""
        logger.info('Running final stages...')
        final_request = {
            'features_df': features_df,
            'targets_df': targets_df,
            'colab_results': colab_results,
            'light_results': light_results,
            'tickers': tickers,
            'timeframes': manifest.get('timeframes', ['15m', '60m', '1d']),
            'batch_name': getattr(args, 'batch_name', None),
            'news_data': news_data,
            'economic_data': economic_data,
            'stages_to_run': getattr(args, 'stages', None),
            'execution_mode': getattr(args, 'execution_mode', 'review_only'),
            'evaluation_notification_authorized': getattr(
                args, 'evaluation_notification_authorized', False
            ),
        }
        return await orchestrator.run_final_stages(final_request)

    @staticmethod
    def _validate_batch_contract(orchestrator) -> dict:
        """Validates the batch directory contract."""
        from src.validation.pipeline_schemas import validate_batch_dir
        return validate_batch_dir(orchestrator.config.output_dir)

    @staticmethod
    def _log_manifest_details(manifest: dict):
        """Logs details from the batch manifest with sanitization."""
        logger.info('✨ Explicit local-Colab contract verified successfully! Manifest details:')

        # CWE-117: Sanitize user-controlled data before logging
        def sanitize(val):
            if val is None: return "None"
            return str(val).replace('\r', '\\r').replace('\n', '\\n')

        logger.info(f"   - Batch Name: {sanitize(manifest.get('batch_name'))}")
        logger.info(f"   - Created At: {sanitize(manifest.get('timestamp'))}")
        logger.info(f"   - Tickers: {sanitize(manifest.get('tickers'))}")
        logger.info(f"   - Timeframes: {sanitize(manifest.get('timeframes'))}")

    @staticmethod
    def _sanitize(val: Any) -> str:
        """Utility to sanitize values for logging to prevent CRLF injection."""
        if val is None:
            return "None"
        return str(val).replace('\r', '\\r').replace('\n', '\\n')

    @staticmethod
    def _load_continue_data(orchestrator, args):
        """
        Orchestrates loading of all data required for continue mode.
        """
        features_df, targets_df, colab_results = PipelineExecutor._load_core_continue_data(orchestrator, args)

        if PipelineExecutor._is_error_result(colab_results):
            return PipelineExecutor._return_error_for_invalid_colab_results(args, colab_results)

        news_data, economic_data = PipelineExecutor._load_extra_continue_data(orchestrator, args)
        return features_df, targets_df, colab_results, news_data, economic_data

    @staticmethod
    def _return_error_for_invalid_colab_results(args, colab_results):
        """Return error tuple when Colab results are invalid."""
        batch_name = PipelineExecutor._sanitize(getattr(args, 'batch_name', 'unknown'))
        logger.error(f"No valid Colab results found for batch: {batch_name}")
        return None, None, colab_results, None, None

    @staticmethod
    def _load_core_continue_data(orchestrator, args):
        """Loads Colab results, features and targets dataframes."""
        batch_dir = orchestrator.config.output_dir
        batch_name = getattr(args, 'batch_name', 'unknown')
        colab_results = orchestrator.load_colab_results(batch_name)

        if PipelineExecutor._is_error_result(colab_results):
            return None, None, colab_results

        features_path = batch_dir / FEATURES_FILE
        targets_path = batch_dir / TARGETS_FILE

        features_df = PipelineExecutor._safe_load_parquet(features_path, "Features")
        targets_df = PipelineExecutor._safe_load_parquet(targets_path, "Targets")

        return features_df, targets_df, colab_results

    @staticmethod
    def _load_extra_continue_data(orchestrator, args):
        """Loads or reconstructs news and economic data."""
        batch_dir = orchestrator.config.output_dir

        news_data = PipelineExecutor._load_news_data_with_fallbacks(batch_dir)
        economic_data = PipelineExecutor._load_economic_data_with_fallbacks(batch_dir)

        if news_data is None or economic_data is None:
            news_data, economic_data = PipelineExecutor._reconstruct_data_from_db(orchestrator, news_data, economic_data)

        return news_data, economic_data

    @staticmethod
    def _load_news_data_with_fallbacks(batch_dir):
        """Load news data with batch directory and persistent fallbacks."""
        news_data = PipelineExecutor._safe_load_parquet(batch_dir / 'news_data.parquet', "News (Batch)", silent=True)
        if news_data is None:
            news_data = PipelineExecutor._safe_load_parquet(Path('data/processed/features/news_data.parquet'), "News (Persistent)", silent=True)
        return news_data

    @staticmethod
    def _load_economic_data_with_fallbacks(batch_dir):
        """Load economic data with batch directory and persistent fallbacks."""
        economic_data = PipelineExecutor._safe_load_parquet(batch_dir / 'economic_data.parquet', "Economic (Batch)", silent=True)
        if economic_data is None:
            economic_data = PipelineExecutor._safe_load_parquet(Path('data/processed/features/macro_data.parquet'), "Macro (Persistent)", silent=True)
            if economic_data is None:
                economic_data = PipelineExecutor._safe_load_parquet(Path('data/processed/features/economic_data.parquet'), "Economic (Persistent)", silent=True)
        return economic_data

    @staticmethod
    def _reconstruct_data_from_db(orchestrator, current_news, current_econ):
        """Reconstructs missing news/economic data from database tables."""
        try:
            from src.data.management.data_manager import DataManager  # noqa: F401
            from src.processing.deduplication_utils import deduplicate_dataframe

            db_manager, collector_configs = PipelineExecutor._initialize_db_reconstruction(orchestrator)
            table_names = db_manager.get_all_table_names()

            news_dfs, macro_dfs = PipelineExecutor._process_tables(db_manager, table_names, collector_configs)

            reconstructed_news = PipelineExecutor._reconstruct_category(current_news, news_dfs, "news", deduplicate_dataframe)
            reconstructed_econ = PipelineExecutor._reconstruct_category(current_econ, macro_dfs, "economic", deduplicate_dataframe)

            return reconstructed_news, reconstructed_econ

        except (pd.errors.EmptyDataError, ValueError, KeyError, ImportError) as ex:
            logger.exception(f"⚠️ Failure reconstructing data from database: {ex}")
            raise

    @staticmethod
    def _initialize_db_reconstruction(orchestrator):
        """Initialize database manager and collector configs for reconstruction."""
        from src.data.management.data_manager import DataManager
        db_manager = DataManager(orchestrator.config_manager)
        collector_configs = orchestrator.config_manager.get_config('collectors', {})
        return db_manager, collector_configs

    @staticmethod
    def _process_tables(db_manager, table_names, collector_configs):
        """Processes all database tables and categorizes them."""
        news_dfs, macro_dfs = [], []
        skipped_tables = {'cache_metadata', 'huggingface_data', 'enriched_features', 'experience_diary', 'market_data'}

        for table_name in table_names:
            if table_name in skipped_tables:
                continue

            df = db_manager.fetch_data_from_table(table_name)
            if df is None or df.empty:
                continue

            data_type = PipelineExecutor._identify_table_data_type(table_name, collector_configs)
            news_dfs, macro_dfs = PipelineExecutor._categorize_dataframe_by_type(df, data_type, news_dfs, macro_dfs)

        return news_dfs, macro_dfs

    @staticmethod
    def _categorize_dataframe_by_type(df, data_type, news_dfs, macro_dfs):
        """Categorize dataframe by data type and append to appropriate list."""
        if data_type == 'news':
            news_dfs.append(df)
        elif data_type == 'macro':
            macro_dfs.append(df)
        return news_dfs, macro_dfs

    @staticmethod
    def _reconstruct_category(current_data, dfs, label, deduplicate_func):
        """Reconstructs a category of data."""
        if current_data is None and dfs:
            reconstructed, _ = deduplicate_func(pd.concat(dfs, ignore_index=True), subset_cols=['timestamp'])
            logger.info(f"✅ Reconstructed {label} data from DB: {reconstructed.shape}")
            return reconstructed
        return current_data

    @staticmethod
    def _identify_table_data_type(table_name: str, collector_configs: dict) -> str:
        """Identifies if a table contains news or macro data."""
        # Check by config
        for config in collector_configs.values():
            if config.get('table_name') == table_name:
                dt = config.get('data_type')
                if dt == 'news':
                    return 'news'
                if dt == 'macro_data':
                    return 'macro'

        # Check by name
        name_lower = table_name.lower()
        if 'fred' in name_lower or 'macro' in name_lower:
            return 'macro'
        return 'unknown'

    @staticmethod
    def _safe_load_parquet(path: Path, label: str, silent: bool = False) -> Any:
        """Safely loads a parquet file, logging success or failure.

        Uses broad Exception catch to handle corrupted files (e.g. pyarrow ArrowInvalid,
        OSError) alongside the usual pandas/type errors.
        """
        if path.exists():
            try:
                df = pd.read_parquet(path)
                if not silent:
                    logger.info(f"Loaded {label}: {df.shape}")
                return df
            except Exception as e:  # noqa: BLE001 — intentional: corrupted parquet raises ArrowInvalid
                label_sanitized = PipelineExecutor._sanitize(label)
                logger.warning(f"Failed to load {label_sanitized} from {path}: {e}")
        elif not silent:
            label_sanitized = PipelineExecutor._sanitize(label)
            logger.error(f"{label_sanitized} file not found: {path}")
        return None

    @staticmethod
    def _is_error_result(result: Any) ->bool:
        """Return True when a loader result represents a missing or failed artifact."""
        if not result:
            return True
        if isinstance(result, dict):
            return bool(result.get('error') or result.get('status') == 'error')
        return False

    @staticmethod
    def _validate_continue_inputs(features_df, targets_df, colab_results,
        batch_name: str):
        """Validate continue-mode inputs before starting local training."""
        bn_sanitized = PipelineExecutor._sanitize(batch_name)

        if PipelineExecutor._is_error_result(colab_results):
            return PipelineExecutor._return_validation_error(bn_sanitized, 'missing_colab_results', 'Colab results are missing or invalid')

        if features_df is None or getattr(features_df, 'empty', True):
            return PipelineExecutor._return_validation_error(bn_sanitized, 'missing_features', 'features.parquet is missing or empty')

        if targets_df is None or getattr(targets_df, 'empty', True):
            return PipelineExecutor._return_validation_error(bn_sanitized, 'missing_targets', 'targets.parquet is missing or empty')

        target_cols = PipelineExecutor._extract_target_columns(targets_df)
        if not target_cols:
            return PipelineExecutor._return_validation_error(batch_name, 'missing_target_columns', 'targets.parquet has no target_* columns')

        return None

    @staticmethod
    def _return_validation_error(batch_name, reason, message):
        """Return validation error dictionary with logging."""
        logger.error(f"Cannot continue batch '{batch_name}': {message}")
        return {'status': 'failed', 'reason': reason}

    @staticmethod
    def _extract_target_columns(targets_df):
        """Extract target columns from targets dataframe."""
        # audit-ignore: ARCHITECTURAL_USAGE
        return [col for col in targets_df.columns if is_direct_target_column(col)]

    @staticmethod
    def _resolve_tickers(args, colab_results, features_df):
        """Resolve tickers for continue mode."""
        tickers = PipelineExecutor._get_tickers_from_args_or_colab(args, colab_results)
        tickers = PipelineExecutor._fallback_to_features_tickers(tickers, features_df)

        tickers_sanitized = PipelineExecutor._sanitize(tickers)
        logger.info(f'Resolved tickers for continue mode: {tickers_sanitized}')
        return tickers

    @staticmethod
    def _get_tickers_from_args_or_colab(args, colab_results):
        """Get tickers from args or colab results."""
        if args.test_ticker:
            return [args.test_ticker]
        return list(colab_results.get('ticker_results', {}).keys())

    @staticmethod
    def _fallback_to_features_tickers(tickers, features_df):
        """Fallback to features dataframe tickers if no tickers found."""
        if tickers or features_df is None:
            return tickers

        if isinstance(features_df.index, pd.MultiIndex):
            return list(features_df.index.get_level_values('ticker').unique())
        elif 'ticker' in features_df.columns:
            return list(features_df['ticker'].unique())
        return tickers

    @staticmethod
    def _merge_results_data(colab_results, light_results):
        """Merge colab results with light models results."""
        merged_results = dict(colab_results)
        if light_results.get('models_metadata'):
            merged_results.setdefault('models_metadata', {}).update(
                light_results['models_metadata'])
        logger.info('Merged Colab results with light models')
        return merged_results

    @staticmethod
    def resolve_tickers_and_timeframes(args, config_manager) -> tuple[list, list]:
        """Resolve tickers and timeframes from args or config."""
        tickers = PipelineExecutor._get_tickers(args, config_manager)
        timeframes = PipelineExecutor._get_timeframes(config_manager)

        tickers_final_sanitized = PipelineExecutor._sanitize(tickers)
        timeframes_sanitized = PipelineExecutor._sanitize(timeframes)
        logger.info(f'Final tickers: {tickers_final_sanitized}')
        logger.info(f'Using timeframes: {timeframes_sanitized}')
        return tickers, timeframes

    @staticmethod
    def _get_tickers(args, config_manager) -> list:
        """Resolves tickers from arguments or config."""
        tickers = PipelineExecutor._get_tickers_from_args_or_config(args, config_manager)
        tickers = PipelineExecutor._apply_test_ticker_if_needed(args, tickers)
        return tickers

    @staticmethod
    def _get_tickers_from_args_or_config(args, config_manager):
        """Get tickers from args or config manager."""
        if args.tickers is not None:
            logger.info(f'Using explicitly provided tickers: {PipelineExecutor._sanitize(args.tickers)}')
            return args.tickers

        assets_config = config_manager.get_config('assets') or {}

        # active_preset decides, when one is set.
        #
        # assets.yaml has declared `active_preset: default_volatile` with an
        # 18-ticker list all along, and nothing read it -- this took every
        # ticker from every sector instead. It made no visible difference
        # while the database held 24 tickers, because Stage 3 can only
        # enrich what was collected.
        #
        # Collection starting to work changed that. The 2026-08-06 run put
        # 112 tickers in the database, so the next prepare enriched 110 and
        # produced a 128,033-row export where the previous one had 15,433.
        # Stage 4 would then train roughly five times the models: the last
        # continue run built 506 in two hours.
        preset_tickers = PipelineExecutor._active_preset_tickers(assets_config)
        if preset_tickers:
            logger.info(
                "Using preset '%s': %d ticker(s). Every sector holds %d; set "
                "active_preset to null to model all of them.",
                assets_config.get('active_preset'), len(preset_tickers),
                len(PipelineExecutor._load_tickers_from_sectors(
                    assets_config.get('sectors', {})
                )),
            )
            return preset_tickers

        sectors = assets_config.get('sectors', {})
        return PipelineExecutor._load_tickers_from_sectors(sectors)

    @staticmethod
    def _active_preset_tickers(assets_config: dict) -> list:
        """Tickers of the preset named by `active_preset`, or [].

        Returns empty -- meaning "no opinion, use every sector" -- when no
        preset is named, the name does not resolve, or the preset lists no
        tickers. An unresolvable name is reported rather than silently
        widening the run to every instrument in the file.
        """
        name = assets_config.get('active_preset')
        if not name:
            return []
        presets = assets_config.get('presets') or {}
        preset = presets.get(name)
        if not isinstance(preset, dict):
            logger.warning(
                "active_preset is '%s' but no such preset exists (known: %s); "
                "falling back to every sector.",
                name, sorted(presets) or 'none',
            )
            return []
        tickers = [str(t) for t in (preset.get('tickers') or [])]
        if not tickers:
            logger.warning(
                "Preset '%s' lists no tickers; falling back to every sector.",
                name,
            )
        return tickers

    @staticmethod
    def _apply_test_ticker_if_needed(args, tickers):
        """Apply test ticker if specified in args."""
        if args.test_ticker:
            logger.info(f'Using test ticker: {PipelineExecutor._sanitize(args.test_ticker)}')
            return [args.test_ticker]
        return tickers

    @staticmethod
    def _load_tickers_from_sectors(sectors: dict) -> list:
        """Loads tickers from sector configuration."""
        all_tickers = PipelineExecutor._collect_tickers_from_sectors(sectors)
        tickers = sorted(all_tickers)
        logger.info(f'Loaded {len(tickers)} unique tickers from {len(sectors)} sectors: {PipelineExecutor._sanitize(tickers)}')
        return tickers

    @staticmethod
    def _collect_tickers_from_sectors(sectors: dict):
        """Collect all tickers from sectors configuration."""
        all_tickers = set()
        for sector_name, sector_config in sectors.items():
            sector_assets = sector_config.get('assets', [])
            all_tickers.update(sector_assets)
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"Sector '{PipelineExecutor._sanitize(sector_name)}': {len(sector_assets)} tickers: {PipelineExecutor._sanitize(sector_assets)}")
        return all_tickers

    @staticmethod
    def _get_timeframes(config_manager) -> list:
        """Resolves timeframes from config."""
        collectors = config_manager.get_config('collectors') or {}
        yf_timeframes = collectors.get('yahoo_finance', {}).get('timeframes', {})
        return list(yf_timeframes.keys()) if yf_timeframes else ['15m', '60m', '1d']

    @staticmethod
    def log_test_mode_info(args) ->None:
        """Log test mode information."""
        if PipelineExecutor._is_test_mode_active(args):
            PipelineExecutor._log_test_mode_details(args)

    @staticmethod
    def _is_test_mode_active(args) ->bool:
        """Check if test mode is active."""
        return any([args.test_ticker, args.test_target, args.test_model])

    @staticmethod
    def _log_test_mode_details(args) ->None:
        """Log test mode details."""
        logger.info('TEST MODE ACTIVATED:')
        if args.test_ticker:
            tt_sanitized = PipelineExecutor._sanitize(args.test_ticker)
            logger.info(f'   Ticker: {tt_sanitized}')
        if args.test_target:
            ttg_sanitized = PipelineExecutor._sanitize(args.test_target)
            logger.info(f'   Target: {ttg_sanitized}')
