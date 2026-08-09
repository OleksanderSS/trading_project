"""
ModelResolver: handles all model-path resolution and loading logic
extracted from PredictionStage to reduce file size.
"""
import json
import logging
import os
from pathlib import Path
from typing import Any, ClassVar, cast

from src.core.logging.logger import ProjectLogger
from src.pipeline.constants import heavy_model_key
from src.pipeline.timeframe_lineage import (
    is_timeframe_token,
    normalize_timeframe,
)


class ModelResolver:
    """Resolves, searches, and loads models from batch directories."""

    def __init__(self, config_manager: Any, model_pool: Any, model_loader: Any
        ):
        self.logger = ProjectLogger.get_logger('ModelResolver')
        self.config_manager = config_manager
        self.model_pool = model_pool
        self.model_loader = model_loader
        self.ACCUMULATION_OUTPUT_DIR_CONFIG = 'system.accumulation.output_dir'
        self.DEFAULT_ACCUMULATION_DIR = 'data/colab/accumulated'

    def check_local_models(self, models_meta: dict[str, Any]) ->bool:
        """Check if models are available locally."""
        for context_id, meta in models_meta.items():
            model_path = meta.get('model_path', '')
            if model_path and '/content/drive' not in model_path and (
                'data\\' in model_path or 'data/' in model_path):
                if self.logger.isEnabledFor(logging.DEBUG):
                    self.logger.debug(
                        f'✅ Found local model: {context_id} -> {model_path}')
                return True
        return False

    def log_model_status(self, models_meta: dict[str, Any]) ->None:
        """Log model status information."""
        self.logger.warning(
            '⚠️ All models are from Colab (not available locally).')
        self.logger.warning('   Checked models:')
        for context_id, meta in list(models_meta.items())[:5]:
            model_path = meta.get('model_path', '')
            model_type = meta.get('model_type', '')
            self.logger.warning(
                f"   - {context_id}: model_path='{model_path}', model_type='{model_type}'"
                )

    def _resolve_batch_from_kwargs(self, base_dir: Path, kwargs: dict[str, Any] | None) -> Path | None:
        """Resolve batch directory from kwargs batch_name."""
        if kwargs:
            batch_name = kwargs.get('batch_name')
            if batch_name:
                batch_dir = base_dir / batch_name
                if batch_dir.exists():
                    self.logger.info(f'✅ Resolved batch_dir from batch_name: {batch_dir}')
                    return cast(Path | None, batch_dir)
        return None

    def _resolve_batch_from_model_paths(self, base_dir: Path, models_meta: dict[str, Any]) -> Path | None:
        """Resolve batch directory from model paths."""
        for _context_id, meta in models_meta.items():
            model_path = meta.get('model_path', '')
            if model_path:
                model_path_str = model_path.replace('/', os.sep)
                parts = Path(model_path_str).parts
                for i, part in enumerate(parts):
                    if part == 'accumulated' and i + 1 < len(parts):
                        batch_dir = base_dir / parts[i + 1]
                        if batch_dir.exists():
                            self.logger.info(f'✅ Resolved batch_dir from model_path: {batch_dir}')
                            return cast(Path | None, batch_dir)
        return None

    def _resolve_most_recent_batch(self, base_dir: Path) -> Path | None:
        """Resolve most recent batch directory from base_dir."""
        if base_dir.exists():
            subdirs = [d for d in base_dir.iterdir() if d.is_dir()]
            if subdirs:
                chosen = max(subdirs, key=lambda p: p.stat().st_mtime)
                self.logger.info(f'✅ Using most recent batch_dir: {chosen}')
                return chosen
        return None

    def resolve_batch_directory(self, models_meta: dict[str, Any], kwargs:
        dict[str, Any] | None=None) ->Path | None:
        """Resolve batch directory from kwargs batch_name or model paths."""
        base_dir = Path(self.config_manager.get(self.
            ACCUMULATION_OUTPUT_DIR_CONFIG, self.DEFAULT_ACCUMULATION_DIR))

        batch_dir = self._resolve_batch_from_kwargs(base_dir, kwargs)
        if batch_dir:
            return batch_dir

        batch_dir = self._resolve_batch_from_model_paths(base_dir, models_meta)
        if batch_dir:
            return batch_dir

        return self._resolve_most_recent_batch(base_dir)

    def _find_available_model_files(self, batch_dir: Path) -> dict[str, Path]:
        """Find all model files in batch directory."""
        model_extensions = {'.keras', '.pkl', '.h5', '.pt', '.joblib'}
        available_files = {}
        for f in batch_dir.iterdir():
            if f.is_file() and f.suffix in model_extensions:
                available_files[f.stem.lower()] = f
        return available_files

    # Known multi-word model type suffixes, ordered longest-first so the most
    # specific match wins (e.g. 'random_forest' before 'forest').
    _KNOWN_MODEL_TYPES: ClassVar[list[str]] = [
        'random_forest', 'neural_network', 'base_neural',
        'lightgbm', 'catboost', 'xgboost', 'tabnet',
        'autoencoder', 'transformer', 'lstm', 'gru', 'cnn',
        'mlp', 'linear', 'ensemble', 'knn', 'svm',
    ]

    def _parse_model_stem(self, stem: str) -> tuple[str, str, str, str] | None:
        """
        Parse a model filename stem into (ticker, timeframe, target, model_type).

        Expected format: model_<TICKER>_<TIMEFRAME>_<TARGET>_<MODEL_TYPE>
        where MODEL_TYPE can be multi-word (e.g. random_forest).

        The timeframe segment is recognised, not counted: it is present when
        the second field spells one of the project's timeframes and absent
        otherwise, in which case it comes back empty. Models trained before
        the heavy branch split by timeframe carry the shorter name and still
        parse -- a file already on disk should not become unreadable because
        the naming convention grew a field.

        Returns None if the stem cannot be parsed.
        """
        # Strip leading 'model_' prefix if present
        s = stem.lower()
        if s.startswith('model_'):
            s = s[len('model_'):]

        # Try to match a known model type suffix
        for mt in self._KNOWN_MODEL_TYPES:
            if s.endswith('_' + mt):
                rest = s[: -(len(mt) + 1)]  # everything before _<model_type>
                # rest = <ticker>[_<timeframe>]_<target>
                parts = rest.split('_', 2)
                if len(parts) == 3 and is_timeframe_token(parts[1]):
                    return parts[0], parts[1], parts[2], mt
                head = rest.split('_', 1)
                if len(head) == 2:
                    return head[0], '', head[1], mt
        return None

    def _match_model_file(self, ticker: str, timeframe: str, target: str,
                          model_type: str,
                          available_files: dict[str, Path]) -> Path | None:
        """Match a model file based on ticker, timeframe, target and model type.

        The timeframe has to take part in the match. Colab trains one heavy
        model per timeframe, so a batch holds model_AAPL_15m_<target>_mlp
        alongside model_AAPL_1d_<target>_mlp -- and a comparison that ignores
        the timeframe returns whichever the directory listed first, feeding
        Stage 5 a model fitted to a different bar size than the rows it is
        about to score.
        """
        expected_ticker = ticker.lower()
        expected_tf = normalize_timeframe(timeframe) or ''
        expected_target = target.lower().replace('-', '_')
        expected_model  = model_type.lower().replace('-', '_')

        # An exact timeframe match wins; an unlabelled file is a fallback,
        # never a rival.
        #
        # This used to accept either in one pass, on the reasoning that a
        # pre-split name "is the only candidate there is". It stopped being
        # the only candidate the moment a run wrote labelled names beside the
        # old ones: data/trained_models holds 4,614 labelled models and 3,536
        # unlabelled ones left from 2026-08-04/05, so for most contexts BOTH
        # match and the winner was whichever the directory listing reached
        # first. The stale ones were fitted to a different batch, and on the
        # heavy side to several timeframes mixed together -- exactly the
        # models this naming was introduced to stop using.
        fallback = None
        for stem, fpath in available_files.items():
            parsed = self._parse_model_stem(stem)
            if parsed is None:
                continue
            file_ticker, file_tf, file_target, file_model_type = parsed
            if (file_ticker != expected_ticker
                    or file_model_type != expected_model
                    or file_target != expected_target):
                continue
            if file_tf and expected_tf:
                if normalize_timeframe(file_tf) == expected_tf:
                    return fpath
                continue
            if fallback is None:
                fallback = fpath
        if fallback is not None:
            return fallback

        # Fallback: loose substring match for flexibility.
        needle = '_'.join(
            part for part in
            (expected_ticker, expected_tf, expected_target, expected_model)
            if part
        )
        for stem, fpath in available_files.items():
            if needle in stem.lower():
                return fpath

        return None

    def update_local_model_paths(self, models_meta: dict[str, Any],
        batch_dir: Path) ->bool:
        """Update model paths to use local files found in batch_dir."""
        available_files = self._find_available_model_files(batch_dir)

        if not available_files:
            self.logger.warning(f'⚠️ No model files found in: {batch_dir}')
            return False

        self.logger.info(f'✅ Found {len(available_files)} model files in {batch_dir}')

        has_local_models = False
        for context_id, meta in models_meta.items():
            ticker = meta.get('ticker', '')
            timeframe = meta.get('timeframe', '')
            target = meta.get('target', '')
            model_type = meta.get('model_type', '')

            if not ticker or not model_type:
                continue

            matched = self._match_model_file(ticker, timeframe, target,
                model_type, available_files)

            if matched:
                meta['model_path'] = str(matched)
                has_local_models = True
                if self.logger.isEnabledFor(logging.DEBUG):
                    self.logger.debug(f'✅ Mapped {context_id} -> {matched.name}')

        mapped = sum(1 for m in models_meta.values() if m.get('model_path'))
        self.logger.info(f'📊 Mapped model paths: {mapped}/{len(models_meta)}')
        return has_local_models

    def load_available_models(self, context_id: str, models_meta: dict[str, Any] | None=None) ->dict[str, Any]:
        """Load all available models for a context."""
        models_meta = models_meta or {}
        direct_result = self._try_load_direct_model(context_id, models_meta)
        if direct_result:
            return direct_result
        batch_dir = self._resolve_batch_dir_from_context(context_id,
            models_meta)
        search_patterns = self._get_model_search_patterns(context_id)
        return self._search_and_load_models(batch_dir, search_patterns,
            context_id, models_meta)

    def load_models_metadata_from_disk(self, kwargs: dict[str, Any]) ->dict[
        str, Any]:
        """Load models_metadata from disk if not provided in pipeline kwargs."""
        models_metadata: dict[str, Any] = {}
        batch_dir = self._resolve_batch_directory_from_kwargs(kwargs)
        if batch_dir:
            self._load_light_models_from_disk(batch_dir, models_metadata)
            self._load_heavy_models_from_disk(batch_dir, models_metadata)
        return models_metadata

    def _try_load_direct_model(self, context_id: str, models_meta: dict[str,
        Any]) ->dict[str, Any] | None:
        if context_id not in models_meta:
            return None
        model_path_str = models_meta[context_id].get('model_path', '')
        if not model_path_str:
            return None
        direct_path = Path(model_path_str.replace('/', os.sep))

        # If path is relative (just a filename), try to resolve it against
        # the batch directory resolved from models_meta
        if not direct_path.is_absolute() and not direct_path.exists():
            batch_dir = self.resolve_batch_directory(models_meta)
            if batch_dir:
                candidate = batch_dir / direct_path.name
                if candidate.exists():
                    direct_path = candidate
                    models_meta[context_id]['model_path'] = str(direct_path)

        if not direct_path.exists():
            return None
        try:
            model_name = direct_path.stem
            model_meta = self._create_model_meta(context_id, models_meta,
                model_name, str(direct_path))
            loaded_model = self.model_pool.get_model(model_name, loader_fn=
                lambda path=str(direct_path), meta=model_meta: self.
                model_loader.load_path(path, meta))
            if loaded_model is not None:
                return {model_name: loaded_model}
        except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError) as e:
            self.logger.error(f'Виникла помилка: {e}', exc_info=True)
            self.logger.warning(
                f'Failed to load model via direct path {direct_path}: {e}')
        return None

    def _resolve_batch_dir_from_context(self, context_id: str, models_meta:
        dict[str, Any]) ->Path:
        if context_id in models_meta:
            model_path_str = models_meta[context_id].get('model_path', '')
            if model_path_str:
                batch_dir = self._extract_batch_dir_from_path(model_path_str)
                if batch_dir:
                    return batch_dir
        return Path(self.config_manager.get(self.
            ACCUMULATION_OUTPUT_DIR_CONFIG, self.DEFAULT_ACCUMULATION_DIR))

    def _extract_batch_dir_from_path(self, model_path_str: str) ->Path | None:
        model_path_str = model_path_str.replace('/', '\\')
        parts = model_path_str.split('\\')
        if 'models' in parts:
            models_idx = parts.index('models')
            if models_idx > 0:
                batch_name = parts[models_idx - 1]
                base_dir = Path(self.config_manager.get(self.
                    ACCUMULATION_OUTPUT_DIR_CONFIG, self.
                    DEFAULT_ACCUMULATION_DIR))
                return base_dir / batch_name
        return None

    def _get_model_search_patterns(self, context_id: str) ->list[str]:
        parts = context_id.split('_')
        if len(parts) >= 4:
            ticker = parts[0]
            target = '_'.join(parts[1:-1])
            model_name = parts[-1]
            return [f'model_{ticker}_{target}*_{model_name}.keras',
                f'model_{ticker}_{target}*_{model_name}.pkl',
                f'model_{ticker}_{target}*_{model_name}.h5',
                f'model_{ticker}_{target}*_{model_name}.pt',
                f'model_{ticker}_{target}*_{model_name}.joblib',
                f'*{ticker}*{target}*{model_name}*.*',
                f'{model_name}_{ticker}_{target}.pt',
                f'CHAMP_{context_id}*.joblib',
                f'MODEL_{context_id}*.joblib', f'*{context_id}*.pt',
                f'*{context_id}*.pkl']
        return [f'*{context_id}*.keras', f'*{context_id}*.pkl',
            f'*{context_id}*.pt', f'*{context_id}*.joblib']

    def _search_and_load_models(self, batch_dir: Path, patterns: list[str],
        context_id: str, models_meta: dict[str, Any]) ->dict[str, Any]:
        loaded_models: dict[str, Any] = {}
        self._read_runtime_params_if_exists(batch_dir)
        search_paths = self._get_models_search_paths(batch_dir)
        for search_path in search_paths:
            if not search_path.exists():
                continue
            self._search_patterns_in_path(search_path, patterns, context_id,
                models_meta, loaded_models)
        return loaded_models

    def _get_models_search_paths(self, batch_dir: Path) ->list[Path]:
        models_root = Path(self.config_manager.get(self.
            ACCUMULATION_OUTPUT_DIR_CONFIG, self.DEFAULT_ACCUMULATION_DIR))
        system_models_path = self.config_manager.get_models_path()
        return [batch_dir / 'models', batch_dir, models_root / 'models',
            models_root, system_models_path]

    def _search_patterns_in_path(self, search_path: Path, patterns: list[
        str], context_id: str, models_meta: dict[str, Any], loaded_models:
        dict[str, Any]) ->None:
        for pattern in patterns:
            for path in search_path.glob(pattern):
                self._try_load_model_from_path(path, context_id,
                    models_meta, loaded_models)

    def _try_load_model_from_path(self, path: Path, context_id: str,
        models_meta: dict[str, Any], loaded_models: dict[str, Any]) ->None:
        try:
            cur_model_name = path.stem.replace(f'_{context_id}', '')
            model_meta = self._create_model_meta(context_id, models_meta,
                cur_model_name, str(path))
            # ModelPool is a single process-wide cache shared across every
            # ticker/context resolved in a run - it must be keyed by
            # something globally unique (path.stem), not cur_model_name,
            # which collapses to a generic constant (e.g. "model") for any
            # file matching the standard "model_{ticker}_{target}_{type}"
            # naming convention. Using cur_model_name here would cache the
            # first ticker's model under that generic key and silently
            # serve it to every other ticker that reaches this path
            # (cache hits never call loader_fn again).
            loaded_model = self.model_pool.get_model(path.stem,
                loader_fn=lambda path=str(path), meta=model_meta: self.
                model_loader.load_path(path, meta))
            if loaded_model is not None:
                loaded_models[cur_model_name] = loaded_model
        except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError) as e:
            self.logger.error(f'Виникла помилка: {e}', exc_info=True)
            self.logger.warning(f'Failed to load model from {path}: {e}')

    def _create_model_meta(self, context_id: str, models_meta: dict[str,
        Any], model_name: str, model_path: str) ->dict[str, Any]:
        return {'model_id': model_name, 'model_path': model_path,
            'model_type': models_meta.get(context_id, {}).get('model_type',
            model_name), 'ticker': models_meta.get(context_id, {}).get(
            'ticker'), 'target': models_meta.get(context_id, {}).get('target'),
            'timeframe': models_meta.get(context_id, {}).get('timeframe')}

    def _read_runtime_params_if_exists(self, batch_dir: Path) ->None:
        runtime_params_path = batch_dir / 'runtime_params.json'
        if runtime_params_path.exists():
            try:
                with open(runtime_params_path) as f:
                    runtime_params = json.load(f)
                    _ = runtime_params.get('test_mode', {}).get('enabled',
                        False)
            except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError) as e:
                self.logger.error(f'Виникла помилка: {e}', exc_info=True)
                self.logger.warning(f'Could not read runtime_params.json: {e}')

    def _resolve_batch_directory_from_kwargs(self, kwargs: dict[str, Any]
        ) ->Path | None:
        batch_name = kwargs.get('batch_name')
        output_dir = Path(self.config_manager.get(
            'system.accumulation.output_dir', 'data/colab/accumulated'))
        if not batch_name:
            batch_dirs = list(output_dir.glob('test_ticker_*'))
            if batch_dirs:
                batch_name = max(batch_dirs, key=lambda p: p.stat().st_mtime
                    ).name
                self.logger.info(f'Found latest batch: {batch_name}')
        if batch_name:
            return output_dir / batch_name
        return None

    def _load_light_models_from_disk(self, batch_dir: Path, models_metadata:
        dict[str, Any]) ->None:
        light_results_files = list(batch_dir.glob(
            'light_models_results_*.json'))
        if light_results_files:
            latest_light = max(light_results_files, key=lambda p: p.stat().
                st_mtime)
            try:
                with open(latest_light) as f:
                    light_results = json.load(f)
                    light_meta = light_results.get('models_metadata', {})
                    models_metadata.update(light_meta)
                    self.logger.info(
                        f'Loaded {len(light_meta)} light models from {latest_light.name}'
                        )
            except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError) as e:
                self.logger.error(f'Виникла помилка: {e}', exc_info=True)
                self.logger.warning(f'Error loading light models: {e}')

    #: What Colab actually writes, and the names earlier revisions used.
    #: colab_clean_cell.py._save_results_summary produces colab_results.json;
    #: NOTHING produces colab_results_summary.json. This resolver looked only
    #: for that second name, and when it was absent did nothing at all --
    #: silently. Every heavy model Colab trained was therefore invisible to
    #: Stage 5, which then ran on light models alone while reporting success.
    #: Kept as an ordered list rather than a single name because
    #: hybrid/results_processor._find_results_file already tries exactly these
    #: three; two readers of one artifact disagreeing about its filename is
    #: how this happened.
    _COLAB_RESULT_FILENAMES = (
        'colab_results_summary.json',
        'colab_results.json',
        'results.json',
    )

    def _load_heavy_models_from_disk(self, batch_dir: Path, models_metadata:
        dict[str, Any]) ->None:
        colab_summary_file = next(
            (
                batch_dir / name
                for name in self._COLAB_RESULT_FILENAMES
                if (batch_dir / name).exists()
            ),
            None,
        )
        if colab_summary_file is None:
            # Said out loud. "Colab produced nothing" and "Colab's output is
            # under a name nobody looked for" are indistinguishable from the
            # outside, and only the second is a defect.
            self.logger.warning(
                "No Colab results file in %s (looked for %s); Stage 5 will "
                "resolve light models only.",
                batch_dir, ', '.join(self._COLAB_RESULT_FILENAMES),
            )
            return
        try:
            with open(colab_summary_file, encoding='utf-8') as f:
                colab_results = json.load(f)
                if 'models_metadata' in colab_results:
                    heavy_meta = colab_results['models_metadata']
                    models_metadata.update(heavy_meta)
                    self.logger.info(
                        f'Loaded {len(heavy_meta)} heavy models from {colab_summary_file.name}'
                        )
                else:
                    self._process_ticker_results_from_colab(colab_results,
                        models_metadata)
        except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError, OSError) as e:
            self.logger.error(f'Виникла помилка: {e}', exc_info=True)
            self.logger.warning(f'Error loading colab models: {e}')

    def _process_ticker_results_from_colab(self, colab_results: dict[str,
        Any], models_metadata: dict[str, Any]) ->None:
        ticker_results = colab_results.get('ticker_results', {})
        for ticker, ticker_data in ticker_results.items():
            timeframes = ticker_data.get('timeframes', {})
            for _tf, tf_data in timeframes.items():
                results = tf_data.get('results', {})
                for target, target_data in results.items():
                    models = target_data.get('models', {})
                    for model_type, model_data in models.items():
                        context_key = heavy_model_key(ticker, _tf, target,
                            model_type)
                        models_metadata[context_key] = {'ticker': ticker,
                            'target': target, 'winner': model_type,
                            'model_type': model_type, 'timeframe': _tf,
                            'model_category':
                            'heavy', 'metrics': model_data.get('metrics', {
                            }), 'selected_features': model_data.get(
                            'selected_features', [])}
