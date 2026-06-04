import json
import logging
from pathlib import Path
from typing import Any

import aiofiles

from src.core.logging.logger import ProjectLogger

logger = ProjectLogger.get_logger('Modeling.IO')


async def load_selected_features_async(stage, config) -> list[str]:
    selected_features = []
    file_candidates = [
        config.batch_dir / f'selected_features_{config.model_type}_{config.ticker}_{config.target_name}.json',
        config.batch_dir / f'selected_features_{config.model_type}_{config.ticker}.json',
        config.batch_dir / f'selected_features_{config.model_type}_{config.target_name}.json',
        config.batch_dir / f'selected_features_{config.model_type}.json'
    ]
    for candidate in file_candidates:
        if candidate.exists():
            selected_features = await try_load_features_file_async(stage, candidate, config.model_type)
            if selected_features:
                break

    if not selected_features:
        glob_candidates = list(config.batch_dir.glob(f'selected_features_{config.model_type}*.json'))
        for candidate in glob_candidates:
            selected_features = await try_load_features_file_async(stage, candidate, config.model_type, is_glob=True)
            if selected_features:
                break

    if not selected_features:
        logger.info(f'ℹ️ No selected features file for {config.model_type}, using all available features as fallback')
        exclude_cols = ['datetime', 'ticker', 'published_at', 'news_id', 'news_title', 'news_sentiment'] + \
                       [c for c in config.x_train.columns if c.startswith('target_')]
        selected_features = [c for c in config.x_train.columns if c not in exclude_cols]

    return selected_features


async def try_load_features_file_async(stage, candidate: Path, model_type: str, is_glob: bool = False) -> list[str]:
    try:
        async with aiofiles.open(candidate, encoding='utf-8') as f:
            content = await f.read()
            feat_data = json.loads(content)
            selected_features = feat_data.get('selected_features', [])
            if selected_features:
                log_msg = f'✅ Loaded {len(selected_features)} features for {model_type} from {candidate.name}'
                if is_glob:
                    log_msg += ' (glob fallback)'
                logger.info(log_msg)
                return selected_features
    except Exception as e:
        logger.error(f'❌ Error loading features from {candidate}: {e}', exc_info=True)
        context = f'LoadFeaturesGlob-{candidate.name}' if is_glob else f'LoadFeatures-{candidate.name}'
        stage.handle_stage_error(e, context=context, severity='warning')
        logger.warning(f'⚠️ Failed to load features from {candidate}: {e}')
    return []


def load_selected_features_sync(stage, config) -> list[str]:
    file_candidates = [
        config.batch_dir / f'selected_features_{config.model_type}_{config.ticker}_{config.target_name}.json',
        config.batch_dir / f'selected_features_{config.model_type}_{config.ticker}.json',
        config.batch_dir / f'selected_features_{config.model_type}_{config.target_name}.json',
        config.batch_dir / f'selected_features_{config.model_type}.json'
    ]
    for candidate in file_candidates:
        if candidate.exists():
            selected_features = try_load_features_file_sync(stage, candidate, config.model_type)
            if selected_features:
                return selected_features

    glob_candidates = list(config.batch_dir.glob(f'selected_features_{config.model_type}*.json'))
    for candidate in glob_candidates:
        selected_features = try_load_features_file_sync(stage, candidate, config.model_type, is_glob=True)
        if selected_features:
            return selected_features

    logger.info(f'ℹ️ No selected features file for {config.model_type}, using all available features as fallback')
    exclude_cols = ['datetime', 'ticker', 'published_at', 'news_id', 'news_title', 'news_sentiment'] + \
                   [c for c in config.x_train.columns if c.startswith('target_')]
    return [c for c in config.x_train.columns if c not in exclude_cols]


def try_load_features_file_sync(stage, candidate: Path, model_type: str, is_glob: bool = False) -> list[str]:
    try:
        with open(candidate, encoding='utf-8') as f:
            feat_data = json.load(f)
            selected_features = feat_data.get('selected_features', [])
            if selected_features:
                log_msg = f'✅ Loaded {len(selected_features)} features for {model_type} from {candidate.name}'
                if is_glob:
                    log_msg += ' (glob fallback)'
                logger.info(log_msg)
                return selected_features
    except Exception as e:
        logger.error(f'❌ Exception loading features from {candidate}: {e}', exc_info=True)
        logger.warning(f'⚠️ Failed to load selected features file {candidate}: {e}')
    return []


def save_light_models_results(stage, ticker: str, target_name: str, light_models: dict[str, Any], batch_dir: Path) -> None:
    results_file = batch_dir / 'light_models_results.json'
    try:
        existing: dict[str, Any] = {}
        if results_file.exists():
            with open(results_file, encoding='utf-8') as f:
                raw = json.load(f)
            if any(k not in ('batch_name', 'created', 'last_updated', 'runs') for k in raw):
                existing = raw

        ticker_data = existing.get(ticker, {})
        target_data = ticker_data.get(target_name, {})

        for _context_key, info in light_models.items():
            model_type = info.get('model_type', 'unknown')
            target_data[model_type] = {
                'model_type': model_type,
                'model_key': info.get('model_key', ''),
                'model_path': info.get('model_path', ''),
                'metrics': info.get('metrics', {}),
                'score': info.get('metrics', {}).get('score', 0.0),
                'accuracy': info.get('metrics', {}).get('accuracy', info.get('metrics', {}).get('score', 0.0)),
                'feature_count': info.get('feature_count', 0),
                'selected_features': info.get('selected_features', [])[:10],
                'market_regime': info.get('market_regime', 'unknown'),
                'timestamp': info.get('timestamp', ''),
                'model_category': 'light'
            }

        ticker_data[target_name] = target_data
        existing[ticker] = ticker_data

        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(existing, f, indent=2, default=str)
        logger.info(f'✅ Saved light model results for {ticker}/{target_name} ({len(target_data)} models) -> {results_file.name}')
    except Exception as e:
        logger.error(f'❌ Failed to save light_models_results.json: {e}', exc_info=True)


def resolve_selected_features_batch_dir(stage) -> Path:
    runtime_params_path = stage.config_manager.get_runtime_params_path()
    accumulated_dir = Path(stage.config_manager.get('system.accumulation.output_dir', 'data/colab/accumulated'))

    batch_name = try_get_batch_name_from_runtime_params(runtime_params_path)
    if not batch_name:
        batch_name = search_nested_runtime_params(accumulated_dir)

    if not batch_name:
        batch_name = 'main_database'
        logger.warning(f'⚠️ No batch_name found in runtime params, defaulting to {batch_name}')

    return accumulated_dir / batch_name


def try_get_batch_name_from_runtime_params(runtime_params_path: Path) -> str | None:
    if not runtime_params_path or not runtime_params_path.exists():
        return None
    try:
        with open(runtime_params_path) as f:
            runtime_params = json.load(f)
            batch_name = runtime_params.get('batch', {}).get('batch_name')
            if batch_name:
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(f'✅ Resolved batch_name from {runtime_params_path}: {batch_name}')
                return batch_name
    except Exception as e:
        logger.error(f'⚠️ Failed to load batch_name from {runtime_params_path}: {e}', exc_info=True)
    return None


def search_nested_runtime_params(accumulated_dir: Path) -> str | None:
    try:
        runtime_files = sorted(accumulated_dir.glob('**/runtime_params.json'),
                              key=lambda p: p.stat().st_mtime, reverse=True)
        for runtime_file in runtime_files:
            batch_name = try_get_batch_name_from_file(runtime_file)
            if batch_name:
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(f'✅ Resolved batch_name from nested runtime_params: {runtime_file} -> {batch_name}')
                return batch_name
    except Exception as e:
        logger.error(f'⚠️ Failed to search nested accumulated runtime_params files: {e}', exc_info=True)
    return None


def try_get_batch_name_from_file(runtime_file: Path) -> str | None:
    try:
        with open(runtime_file) as f:
            runtime_params = json.load(f)
            return runtime_params.get('batch', {}).get('batch_name')
    except Exception as e:
        logger.error(f'Could not read batch_name from runtime file {runtime_file}: {e}', exc_info=True)
    return None
