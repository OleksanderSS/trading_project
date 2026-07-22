from pathlib import Path
from typing import Any

from src.core.logging.logger import ProjectLogger
from src.pipeline.stages.modeling import io as modeling_io
from src.pipeline.stages.modeling import metrics as modeling_metrics
from src.pipeline.stages.modeling import pipeline_control_artifacts
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

        except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError) as e:
            logger.exception(f"❌ Failed to train {model_type}: {e}")
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

    if result.get('status') != 'success' or not result.get('model_key'):
        logger.warning(f"Training did not return a usable model key for {config.model_type}")
        return None

    train_predictions = config.light_trainer.predict(result['model_key'], x_train_filtered)
    predictions = config.light_trainer.predict(result['model_key'], x_test_filtered)
    train_metrics = modeling_metrics.calculate_model_metrics(config.y_train, train_predictions, config.task_type)
    validation_metrics = modeling_metrics.calculate_model_metrics(config.y_test, predictions, config.task_type)
    metrics = {
        **validation_metrics,
        "train_score": train_metrics.get("score"),
        "validation_score": validation_metrics.get("score"),
        "test_score": validation_metrics.get("score"),
        "sample_count": int(len(config.y_train)) + int(len(config.y_test)),
        "train_sample_count": int(len(config.y_train)),
        "validation_sample_count": int(len(config.y_test)),
        "train_metrics": train_metrics,
        "validation_metrics": validation_metrics,
    }

    model_path = stage.models_dir / f"{config.model_type}_{config.ticker}_{config.target_name}.joblib"
    config.light_trainer.save_model_to_disk(result['model_key'], str(model_path))
    model = config.light_trainer.models_in_memory.get(result['model_key'])
    feature_importance = pipeline_control_artifacts.extract_native_feature_importance(model, selected_features)
    stability_analysis = pipeline_control_artifacts.build_feature_distribution_stability_analysis(
        x_train_filtered,
        x_test_filtered,
        selected_features,
    )
    evaluation_window = pipeline_control_artifacts.build_split_evaluation_window(x_test_filtered)
    pipeline_control_paths = _write_pipeline_control_candidates(
        config=config,
        context_key=f"{config.ticker}_{config.target_name}_{config.model_type}",
        train_metrics=train_metrics,
        validation_metrics=validation_metrics,
        feature_importance=feature_importance,
        stability_analysis=stability_analysis,
        evaluation_window=evaluation_window,
    )

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
    if pipeline_control_paths:
        champion_info["pipeline_control_metric_artifacts"] = pipeline_control_paths

    logger.info(f"✅ {config.model_type}: score={metrics['score']:.4f}, features={len(selected_features)}")
    return champion_info


def _write_pipeline_control_candidates(
    *,
    config: SingleModelTrainingConfig,
    context_key: str,
    train_metrics: dict[str, Any],
    validation_metrics: dict[str, Any],
    feature_importance: dict[str, float],
    stability_analysis: dict[str, Any],
    evaluation_window: dict[str, Any] | None,
) -> dict[str, Any]:
    try:
        model_candidate = pipeline_control_artifacts.build_model_evaluation_candidate(
            ticker=config.ticker,
            target_name=config.target_name,
            model_type=config.model_type,
            timeframe=config.timeframe,
            context_fingerprint=config.context_fingerprint,
            market_regime=config.market_regime,
            volatility_regime=config.volatility_regime,
            train_metrics=train_metrics,
            validation_metrics=validation_metrics,
            train_sample_count=len(config.y_train),
            validation_sample_count=len(config.y_test),
            max_drawdown=None,
            evaluation_window=evaluation_window,
        )
        feature_candidate = pipeline_control_artifacts.build_feature_stability_candidate(
            ticker=config.ticker,
            target_name=config.target_name,
            model_type=config.model_type,
            timeframe=config.timeframe,
            context_fingerprint=config.context_fingerprint,
            market_regime=config.market_regime,
            volatility_regime=config.volatility_regime,
            feature_importance=feature_importance,
            stability_analysis=stability_analysis,
        )
        return pipeline_control_artifacts.write_pipeline_control_metric_artifact_candidates(
            batch_dir=config.batch_dir,
            context_key=context_key,
            model_evaluation=model_candidate,
            feature_stability=feature_candidate,
        )
    except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError, OSError) as e:
        logger.warning(f"Could not write pipeline-control metric candidates for {context_key}: {e}")
        return {}
