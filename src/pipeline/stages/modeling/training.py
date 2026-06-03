from pathlib import Path
from typing import Any

from src.core.logging.logger import ProjectLogger
from src.pipeline.stages.modeling import io as modeling_io
from src.pipeline.stages.modeling import metrics as modeling_metrics
from src.pipeline.stages.modeling.dataclasses import LightModelChampionConfig, SingleModelTrainingConfig

logger = ProjectLogger.get_logger('Modeling.Training')


def train_light_models_locally(
    stage,
    ticker: str,
    target_name: str,
    prepared_data: dict[str, Any],
    batch_dir: Path,
    context_fingerprint: str,
    market_regime: str,
    volatility_regime: str,
    timeframe: str
) -> dict[str, dict[str, Any]]:
    from src.training.light_model_trainer import LightModelTrainer

    light_models: dict[str, Any] = {}
    light_trainer = LightModelTrainer()

    training_data = stage._get_light_model_training_data(prepared_data)
    if not training_data:
        return light_models

    x_train, y_train, x_test, y_test = training_data
    task_type = stage._determine_task_type(target_name)
    light_model_types = stage._get_light_model_types()

    for model_type in light_model_types:
        try:
            training_config = SingleModelTrainingConfig(
                model_type=model_type,
                ticker=ticker,
                target_name=target_name,
                timeframe=timeframe,
                batch_dir=batch_dir,
                x_train=x_train,
                y_train=y_train,
                x_test=x_test,
                y_test=y_test,
                task_type=task_type,
                light_trainer=light_trainer,
                context_fingerprint=context_fingerprint,
                market_regime=market_regime,
                volatility_regime=volatility_regime
            )
            model_result = train_single_light_model(stage, training_config)
            if model_result:
                context_key = f"{ticker}_{target_name}_{model_type}"
                light_models[context_key] = model_result

        except Exception as e:
            logger.error(f"❌ Failed to train {model_type}: {e}", exc_info=True)
            continue

    return light_models


def train_single_light_model(stage, config: SingleModelTrainingConfig):
    # Load selected features
    sync_config = config  # Reuse the existing dataclass fields for I/O
    selected_features = modeling_io.load_selected_features_sync(stage, sync_config)

    if not selected_features:
        logger.warning(f"⚠️ No features available for {config.model_type}, skipping")
        return None

    logger.info(f"🔧 Training {config.model_type} with {len(selected_features)} features...")

    x_train_filtered = config.x_train[selected_features]
    x_test_filtered = config.x_test[selected_features]

    train_df = x_train_filtered.copy()
    train_df[config.target_name] = config.y_train.values

    training_params = {
        'model_type': config.model_type,
        'ticker': config.ticker,
        'timeframe': config.timeframe,
        'target_col': config.target_name,
        'task_type': config.task_type
    }

    result = config.light_trainer.train_light_model(
        features_df=train_df,
        config=training_params
    )

    predictions = config.light_trainer.predict(result['model_key'], x_test_filtered)
    metrics = modeling_metrics.calculate_model_metrics(config.y_test, predictions, config.task_type)

    model_path = stage.models_dir / f"{config.model_type}_{config.ticker}_{config.target_name}.joblib"
    config.light_trainer.save_model_to_disk(result['model_key'], str(model_path))

    champion_config = LightModelChampionConfig(
        ticker=config.ticker,
        target_name=config.target_name,
        model_type=config.model_type,
        model_key=result['model_key'],
        selected_features=selected_features,
        metrics=metrics,
        model_path=model_path,
        context_fingerprint=config.context_fingerprint,
        market_regime=config.market_regime,
        volatility_regime=config.volatility_regime
    )

    champion_info = modeling_metrics.create_light_model_champion_info(champion_config)

    logger.info(f"✅ {config.model_type}: score={metrics['score']:.4f}, features={len(selected_features)}")
    return champion_info
