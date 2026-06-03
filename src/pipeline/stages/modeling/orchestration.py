import json
from typing import Any
import pandas as pd
from src.core.logging.logger import ProjectLogger
from src.models.adapters.data_preparation import prepare_data_for_models
from src.pipeline.stages.modeling import io as modeling_io
from src.pipeline.stages.modeling import training as modeling_training
from src.pipeline.stages.modeling.dataclasses import ChampionInfoConfig, FeatureLoadingConfig, SuccessfulTrainingConfig, TargetProcessingConfig, TrainingDebugInfo
from src.training.constants import DEFAULT_TEST_SIZE
logger = ProjectLogger.get_logger('Modeling.Orchestration')


async def create_ensemble_from_top_models_async(stage, training_results:
    dict[str, Any], ticker: str, target_name: str, top_n: int=3) ->(dict[
    str, Any] | None):
    try:
        ticker_result = training_results.get('tickers_results', {}).get(ticker,
            {})
        if ticker_result.get('status') != 'success':
            logger.warning(
                f'Ticker {ticker} training not successful, skipping ensemble creation'
                )
            return None
        all_metrics = ticker_result.get('metrics', {})
        if not all_metrics:
            logger.warning(
                f'No metrics found for {ticker}_{target_name}, skipping ensemble'
                )
            return None
        sorted_models = sorted(all_metrics.items(), key=lambda x: x[1].get(
            'r2_score', 0) if isinstance(x[1], dict) else 0, reverse=True)[:
            top_n]
        if len(sorted_models) < 2:
            logger.warning(
                f'Only {len(sorted_models)} models available, need at least 2 for ensemble'
                )
            return None
        model_list = []
        for model_name, metrics in sorted_models:
            model_list.append((model_name, metrics))
        ensemble_config = {'model_names': [m[0] for m in model_list],
            'metrics': {m[0]: m[1] for m in model_list}, 'weights': [1.0 /
            len(model_list)] * len(model_list), 'ensemble_type':
            'weighted_average', 'ticker': ticker, 'target_name': target_name}
        logger.info(f'✅ Ensemble created for {ticker}_{target_name}:')
        logger.info(f"   Models: {ensemble_config['model_names']}")
        logger.info(
            f"   Weights: {[f'{w:.3f}' for w in ensemble_config['weights']]}")
        ensemble_path = (stage.models_dir /
            f'ensemble_{ticker}_{target_name}.json')
        with open(ensemble_path, 'w') as f:
            json.dump(ensemble_config, f, indent=2)
        logger.info(f'   Saved to: {ensemble_path}')
        return ensemble_config
    except Exception as e:
        logger.error(
            f'Failed to create ensemble for {ticker}_{target_name}: {e}',
            exc_info=True)
        raise RuntimeError(
            f"Failed to create ensemble for {ticker}_{target_name}"
        ) from e


async def process_ticker(stage, ticker: str, df, champions: dict[str, Any]
    ) ->None:
    try:
        target_cols = [c for c in df.columns if c.startswith('target_')]
        timeframe = df['timeframe'].iloc[-1
            ] if 'timeframe' in df.columns else '1d'
        for target_name in target_cols:
            config = TargetProcessingConfig(ticker=ticker, df=df,
                target_name=target_name, timeframe=timeframe, champions=
                champions)
            await process_target(stage, config)
    except Exception as e:
        logger.error(f'Виникла помилка: {e}', exc_info=True)
        stage.handle_stage_error(e, context=f'Modeling-{ticker}', severity=
            'error')
        raise


async def process_target(stage, config: TargetProcessingConfig) ->None:
    context_key = f'{config.ticker}_{config.target_name}'
    prepared_data = prepare_data_for_models(df=config.df, ticker=config.
        ticker, timeframe=config.timeframe, target_cols=[config.target_name
        ], test_size=stage.modeling_config.get('test_size', DEFAULT_TEST_SIZE))
    if not prepared_data:
        logger.warning(f'Data preparation failed for {context_key}. Skipping.')
        return
    training_results = stage.training_manager.execute_unified_training(tickers
        =[config.ticker], data_context=prepared_data)
    comparison_report = stage.comparison_analyzer.compare_models(
        training_results, market_context=stage.brain.get('market_regime',
        'neutral'))
    ticker_result = training_results.get('tickers_results', {}).get(config.
        ticker, {})
    if ticker_result.get('status') == 'success':
        training_config = SuccessfulTrainingConfig(ticker=config.ticker,
            target_name=config.target_name, timeframe=config.timeframe,
            prepared_data=prepared_data, ticker_result=ticker_result,
            comparison_report=comparison_report, champions=config.champions)
        await process_successful_training(stage, training_config)


async def process_successful_training(stage, config: SuccessfulTrainingConfig
    ) ->None:
    context_key = f'{config.ticker}_{config.target_name}'
    context_fingerprint = stage._get_context_fingerprint(config.ticker_result)
    winner_name = config.ticker_result.get('winner')
    all_metrics = config.ticker_result.get('metrics', {})
    winner_metrics = config.ticker_result.get('winner_metrics', all_metrics
        .get(winner_name, {}))
    batch_dir = modeling_io.resolve_selected_features_batch_dir(stage)
    x_train = config.prepared_data.get('light_models', {}).get('X_train')
    if x_train is None:
        x_train = pd.DataFrame(columns=config.prepared_data.get(
            'light_models', {}).get('feature_names', []))
    feature_config = FeatureLoadingConfig(model_type=winner_name, ticker=
        config.ticker, target_name=config.target_name, batch_dir=batch_dir,
        x_train=x_train)
    selected_features = await modeling_io.load_selected_features_async(stage,
        feature_config)
    debug_info = TrainingDebugInfo(context_key=context_key, winner_name=
        winner_name, winner_metrics=winner_metrics, all_metrics=all_metrics,
        selected_features=selected_features)
    stage._log_training_debug_info(debug_info)
    champion_config = ChampionInfoConfig(ticker=config.ticker, target_name=
        config.target_name, winner_name=winner_name, comparison_report=
        config.comparison_report, context_fingerprint=context_fingerprint,
        market_regime=stage.brain.get('market_regime', 'neutral'),
        winner_metrics=winner_metrics, all_metrics=all_metrics,
        ticker_result=config.ticker_result, selected_features=selected_features
        )
    champion_info = stage._create_champion_info(champion_config)
    config.champions[context_key] = champion_info
    logger.info(
        f'✅ Saved heavy model champion_info with {len(selected_features)} features and metrics'
        )
    light_models_trained = modeling_training.train_light_models_locally(stage,
        config.ticker, config.target_name, config.prepared_data, batch_dir,
        context_fingerprint, stage.brain.get('market_regime', 'neutral'),
        stage.brain.get('volatility_regime', 'normal'), config.timeframe)
    config.champions.update(light_models_trained)
    logger.info(f'✅ Trained {len(light_models_trained)} light models locally')
    stage._log_to_diary(champion_info, config.timeframe)
    for light_info in light_models_trained.values():
        stage._log_to_diary(light_info, config.timeframe)
