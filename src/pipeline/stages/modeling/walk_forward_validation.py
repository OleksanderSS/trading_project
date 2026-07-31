from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, balanced_accuracy_score

from src.pipeline.stages.modeling.pipeline_control_artifacts import (
    build_feature_distribution_stability_analysis,
)
from src.pipeline.target_column_utils import is_target_like_column


def _get_target_horizon_rows(target_name: str) -> int:
    """
    Return how far forward `target_name` looks, in rows.

    Delegates to PipelinePolicyManager so this is computed in exactly one
    place. It used to return `abs(shift)` alone, which is wrong for the
    forward-window regression methods: `target_daily_trend_strength_1d` has
    shift -1 and window 20, so it reaches 20 rows ahead while this reported 1
    -- the purge gap at the train/validation boundary was 19 rows too narrow
    and validation labels were partly computed from training rows.

    Falls back to 1 if the target is unknown, so the purge requirement is at
    least 1 row (never 0). Callers that need a strict guarantee should pass an
    explicit purge_rows ≥ horizon.
    """
    try:
        # Lazy import to avoid circular deps at module level
        from src.config.unified_config_manager import get_current_config
        from src.policy import get_policy_manager

        return max(1, get_policy_manager(get_current_config()).target_horizon(target_name))
    except (ValueError, TypeError, AttributeError, KeyError, ImportError) as e:
        # This computes the purge gap that prevents label leakage at the
        # train/validation boundary - falling back to 1 silently would
        # under-purge for any target with a real horizon > 1, so this
        # must be logged rather than swallowed.
        import logging
        logging.getLogger(__name__).warning(
            f"Could not resolve horizon for target '{target_name}', "
            f"falling back to purge=1 rows: {e}"
        )
    return 1


@dataclass(frozen=True)
class WalkForwardValidationConfig:
    min_train_rows: int = 360
    validation_rows: int = 120
    step_rows: int = 120
    purge_rows: int = 5
    max_folds: int = 4
    max_features: int = 40
    random_state: int = 42
    n_estimators: int = 128
    max_depth: int = 8
    min_samples_leaf: int = 5
    # Minimum purge that must be satisfied regardless of target horizon.
    # The actual purge used will be max(purge_rows, target_horizon_rows).
    enforce_horizon_purge: bool = True


class PipelineWalkForwardValidationEvaluator:
    """Evaluate one fixed model with purged expanding train/validation folds."""

    def __init__(
        self,
        config: WalkForwardValidationConfig | None = None,
    ):
        self.config = config or WalkForwardValidationConfig()

    def evaluate(
        self,
        frame: pd.DataFrame,
        *,
        ticker: str,
        timeframe: str,
        target_name: str,
        timeframe_context_report: dict[str, Any] | None = None,
        source_lineage: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        prepared = self._prepare_context_frame(
            frame,
            ticker=ticker,
            timeframe=timeframe,
            target_name=target_name,
        )

        # --- Purge-gap horizon enforcement ---
        # purge_rows must be ≥ abs(target shift) to prevent labels from the
        # tail of the training window "looking into" the validation window.
        # Example: target_weekly_up_1w has shift=-7 → purge must be ≥ 7.
        target_horizon = _get_target_horizon_rows(target_name)
        effective_purge = self.config.purge_rows
        if self.config.enforce_horizon_purge and effective_purge < target_horizon:
            import warnings
            warnings.warn(
                f"[WalkForward] purge_rows={effective_purge} < target horizon "
                f"{target_horizon} for '{target_name}'. "
                f"Automatically raising purge_rows to {target_horizon} to prevent "
                f"label-leakage at train/validation boundary.",
                UserWarning,
                stacklevel=2,
            )
            effective_purge = target_horizon

        # Rebuild config with the corrected purge if needed
        from dataclasses import replace as dc_replace
        effective_config = (
            dc_replace(self.config, purge_rows=effective_purge)
            if effective_purge != self.config.purge_rows
            else self.config
        )

        folds = build_purged_expanding_folds(
            len(prepared),
            config=effective_config,
        )
        if not folds:
            raise ValueError(
                f"Insufficient rows for walk-forward validation: {len(prepared)}."
            )

        first_train = prepared.iloc[folds[0]["train_start"]:folds[0]["train_end"]]
        selected_features, feature_selection = select_initial_train_features(
            first_train,
            target_name=target_name,
            max_features=effective_config.max_features,
            random_state=effective_config.random_state,
        )
        if not selected_features:
            raise ValueError("Initial train fold produced no usable features.")

        fold_results = [
            self._evaluate_fold(
                prepared,
                fold=fold,
                fold_number=fold_number,
                target_name=target_name,
                selected_features=selected_features,
            )
            for fold_number, fold in enumerate(folds, start=1)
        ]
        summary = self._summarize_folds(fold_results, effective_config=effective_config)
        context_fingerprint = _context_fingerprint(
            ticker=ticker,
            timeframe=timeframe,
            target_name=target_name,
            selected_features=selected_features,
            fold_results=fold_results,
            timeframe_context_report=timeframe_context_report,
            source_lineage=source_lineage,
            config=effective_config,
        )
        candidate = {
            "artifact_class": "pipeline_control_walk_forward_validation_candidate",
            "evidence_class": "development_train_validation_only",
            "contract_status": (
                "ready_walk_forward_train_validation_candidate"
                if summary["contract_passed"]
                else "walk_forward_candidate_blocked_by_validation_contract"
            ),
            "ticker": ticker,
            "timeframe": timeframe,
            "target_name": target_name,
            "model_type": "review_only_random_forest_fixed_v1",
            "context_fingerprint": context_fingerprint,
            "selected_features": selected_features,
            "feature_selection": feature_selection,
            "folds": fold_results,
            "metrics": summary,
            "timeframe_context_lineage": _context_lineage_snapshot(
                timeframe_context_report,
                base_timeframe=timeframe,
            ),
            "source_lineage": source_lineage or {},
            "test_contract": {
                "test_rows_loaded": 0,
                "test_metrics_read": False,
                "past_evaluation_rows_loaded": 0,
                "frozen_test_windows_accessed": False,
                "eligible_as_locked_test_evidence": False,
            },
            "promotion_contract": {
                "can_freeze_candidate_for_new_holdout": summary["contract_passed"],
                "can_promote_model": False,
                "can_write_production_config": False,
                "can_trade": False,
            },
            "explicit_non_actions": [
                "No test or past-evaluation partition is loaded.",
                "No hyperparameter or feature variant search is run.",
                "The selected feature set is frozen from the first train fold.",
                "Models exist only in memory for fold evaluation and are not persisted.",
                "No production config, learning memory, recommendation, order, or trade is written.",
            ],
        }
        return candidate

    def _prepare_context_frame(
        self,
        frame: pd.DataFrame,
        *,
        ticker: str,
        timeframe: str,
        target_name: str,
    ) -> pd.DataFrame:
        if target_name not in frame.columns:
            raise ValueError(f"Target column is missing: {target_name}.")
        result = frame.copy()
        if "ticker" not in result.columns:
            raise ValueError("Walk-forward frame is missing ticker.")
        result = result.loc[
            result["ticker"].astype(str).str.upper().eq(ticker.upper())
        ]
        interval_column = next(
            (
                column
                for column in ("interval", "timeframe", "tf")
                if column in result.columns
            ),
            None,
        )
        if interval_column is None:
            raise ValueError("Walk-forward frame is missing interval/timeframe.")
        result = result.loc[
            result[interval_column].astype(str).str.lower().eq(timeframe.lower())
        ]
        datetime_column = next(
            (
                column
                for column in ("datetime", "timestamp", "date")
                if column in result.columns
            ),
            None,
        )
        if datetime_column is None:
            raise ValueError("Walk-forward frame is missing datetime.")
        result["__walk_forward_datetime"] = pd.to_datetime(
            result[datetime_column],
            errors="coerce",
            utc=True,
        ).astype("datetime64[ns, UTC]")
        result[target_name] = pd.to_numeric(result[target_name], errors="coerce")
        result = result.dropna(
            subset=["__walk_forward_datetime", target_name]
        )
        result = (
            result.sort_values("__walk_forward_datetime", kind="mergesort")
            .drop_duplicates("__walk_forward_datetime", keep="last")
            .set_index("__walk_forward_datetime", drop=True)
        )
        target_values = set(result[target_name].astype(int).unique())
        if not target_values.issubset({0, 1}):
            raise ValueError(
                f"Walk-forward v1 supports binary targets only, got {target_values}."
            )
        if result[target_name].nunique() < 2:
            raise ValueError("Walk-forward target contains fewer than two classes.")
        return result

    def _evaluate_fold(
        self,
        frame: pd.DataFrame,
        *,
        fold: dict[str, int],
        fold_number: int,
        target_name: str,
        selected_features: list[str],
    ) -> dict[str, Any]:
        train = frame.iloc[fold["train_start"]:fold["train_end"]].copy()
        validation = frame.iloc[
            fold["validation_start"]:fold["validation_end"]
        ].copy()
        if train[target_name].nunique() < 2:
            raise ValueError(f"Fold {fold_number} train target has one class.")
        if validation[target_name].nunique() < 2:
            raise ValueError(f"Fold {fold_number} validation target has one class.")

        imputer = SimpleImputer(strategy="median")
        train_values = train[selected_features].replace(
            [np.inf, -np.inf],
            np.nan,
        )
        validation_values = validation[selected_features].replace(
            [np.inf, -np.inf],
            np.nan,
        )
        train_matrix = pd.DataFrame(
            imputer.fit_transform(train_values),
            index=train.index,
            columns=selected_features,
        )
        validation_matrix = pd.DataFrame(
            imputer.transform(validation_values),
            index=validation.index,
            columns=selected_features,
        )
        train_target = train[target_name].astype(int)
        validation_target = validation[target_name].astype(int)
        model = RandomForestClassifier(
            n_estimators=self.config.n_estimators,
            max_depth=self.config.max_depth,
            min_samples_leaf=self.config.min_samples_leaf,
            class_weight="balanced_subsample",
            random_state=self.config.random_state,
            n_jobs=1,
        )
        model.fit(train_matrix, train_target)
        train_prediction = pd.Series(
            model.predict(train_matrix),
            index=train.index,
        )
        validation_prediction = pd.Series(
            model.predict(validation_matrix),
            index=validation.index,
        )
        train_metrics = _classification_metrics(
            train_target,
            train_prediction,
        )
        validation_metrics = _classification_metrics(
            validation_target,
            validation_prediction,
        )
        stability = build_feature_distribution_stability_analysis(
            train_matrix,
            validation_matrix,
            selected_features,
        )
        importance = _normalized_importance(
            selected_features,
            model.feature_importances_,
        )
        return {
            "fold": fold_number,
            "train_window": _window_payload(train),
            "purge_window": {
                "start_position": fold["train_end"],
                "end_position": fold["validation_start"],
                "row_count": fold["validation_start"] - fold["train_end"],
            },
            "validation_window": _window_payload(validation),
            "train_metrics": train_metrics,
            "validation_metrics": validation_metrics,
            "train_validation_balanced_accuracy_gap": round(
                train_metrics["balanced_accuracy"]
                - validation_metrics["balanced_accuracy"],
                6,
            ),
            "validation_positive_rate_gap": round(
                abs(
                    validation_metrics["actual_positive_rate"]
                    - validation_metrics["predicted_positive_rate"]
                ),
                6,
            ),
            "feature_stability": stability,
            "feature_importance": importance,
            "temporal_contract": {
                "train_precedes_validation": train.index.max()
                < validation.index.min(),
                "purge_rows": fold["validation_start"] - fold["train_end"],
                "validation_rows": len(validation),
            },
        }

    def _summarize_folds(
        self,
        folds: list[dict[str, Any]],
        effective_config: WalkForwardValidationConfig | None = None,
    ) -> dict[str, Any]:
        # Use effective_config (which has the horizon-adjusted purge_rows) when
        # checking the temporal contract; fall back to self.config for tests
        # that call _summarize_folds directly without the new parameter.
        check_config = effective_config if effective_config is not None else self.config
        validation_balanced = [
            fold["validation_metrics"]["balanced_accuracy"] for fold in folds
        ]
        validation_accuracy = [
            fold["validation_metrics"]["accuracy"] for fold in folds
        ]
        majority_baselines = [
            fold["validation_metrics"]["majority_class_baseline"]
            for fold in folds
        ]
        gaps = [
            fold["train_validation_balanced_accuracy_gap"] for fold in folds
        ]
        stability_scores = [
            float(
                fold["feature_stability"].get(
                    "feature_stability_score",
                    0.0,
                )
            )
            for fold in folds
        ]
        positive_rate_gaps = [
            fold["validation_positive_rate_gap"] for fold in folds
        ]
        validation_above_majority_count = sum(
            accuracy >= majority
            for accuracy, majority in zip(
                validation_accuracy,
                majority_baselines,
                strict=True,
            )
        )
        checks = {
            "minimum_three_folds": len(folds) >= 3,
            "mean_validation_balanced_accuracy_at_least_0_52": _mean(
                validation_balanced
            )
            >= 0.52,
            "mean_train_validation_gap_at_most_0_20": _mean(gaps) <= 0.20,
            "mean_feature_stability_at_least_0_60": _mean(stability_scores)
            >= 0.60,
            "maximum_positive_rate_gap_at_most_0_20": max(
                positive_rate_gaps
            )
            <= 0.20,
            "at_least_half_folds_meet_majority_accuracy": (
                validation_above_majority_count >= math.ceil(len(folds) / 2)
            ),
            "all_temporal_contracts_pass": all(
                fold["temporal_contract"]["train_precedes_validation"]
                and fold["temporal_contract"]["purge_rows"]
                >= check_config.purge_rows
                for fold in folds
            ),
        }
        return {
            "fold_count": len(folds),
            "mean_train_balanced_accuracy": round(
                _mean(
                    [
                        fold["train_metrics"]["balanced_accuracy"]
                        for fold in folds
                    ]
                ),
                6,
            ),
            "mean_validation_balanced_accuracy": round(
                _mean(validation_balanced),
                6,
            ),
            "minimum_validation_balanced_accuracy": round(
                min(validation_balanced),
                6,
            ),
            "mean_validation_accuracy": round(
                _mean(validation_accuracy),
                6,
            ),
            "mean_validation_majority_baseline": round(
                _mean(majority_baselines),
                6,
            ),
            "validation_above_majority_fold_count": (
                validation_above_majority_count
            ),
            "mean_train_validation_gap": round(_mean(gaps), 6),
            "mean_feature_stability_score": round(
                _mean(stability_scores),
                6,
            ),
            "maximum_validation_positive_rate_gap": round(
                max(positive_rate_gaps),
                6,
            ),
            "test_rows_loaded": 0,
            "test_metrics_read": False,
            "past_evaluation_rows_loaded": 0,
            "checks": checks,
            "contract_passed": all(checks.values()),
        }


def build_purged_expanding_folds(
    row_count: int,
    *,
    config: WalkForwardValidationConfig,
) -> list[dict[str, int]]:
    """Build deterministic expanding folds and retain the latest configured folds."""
    minimum_train = max(2, int(config.min_train_rows))
    validation_rows = max(1, int(config.validation_rows))
    step_rows = max(1, int(config.step_rows))
    purge_rows = max(1, int(config.purge_rows))
    candidates: list[dict[str, int]] = []
    validation_start = minimum_train
    while validation_start + validation_rows <= int(row_count):
        train_end = validation_start - purge_rows
        if train_end > 1:
            candidates.append(
                {
                    "train_start": 0,
                    "train_end": train_end,
                    "validation_start": validation_start,
                    "validation_end": validation_start + validation_rows,
                }
            )
        validation_start += step_rows
    max_folds = max(1, int(config.max_folds))
    return candidates[-max_folds:]


def select_initial_train_features(
    train: pd.DataFrame,
    *,
    target_name: str,
    max_features: int,
    random_state: int,
) -> tuple[list[str], dict[str, Any]]:
    """Select and freeze features using only the first expanding train fold."""
    excluded = {
        target_name,
        "open",
        "high",
        "low",
        "close",
        "volume",
    }
    numeric = train.select_dtypes(include=[np.number, "bool"])
    candidates: list[str] = []
    for column in numeric.columns:
        name = str(column)
        if name in excluded or is_target_like_column(name):
            continue
        values = pd.to_numeric(train[name], errors="coerce").replace(
            [np.inf, -np.inf],
            np.nan,
        )
        coverage = float(values.notna().mean())
        variance = float(values.var()) if coverage else 0.0
        if coverage < 0.80 or not math.isfinite(variance) or variance <= 0:
            continue
        candidates.append(name)
    if not candidates:
        return [], {
            "method": "initial_train_only_mutual_information_v1",
            "candidate_count": 0,
            "selected_count": 0,
            "validation_labels_used": False,
            "test_rows_used": 0,
        }

    imputer = SimpleImputer(strategy="median")
    values = train[candidates].replace([np.inf, -np.inf], np.nan)
    matrix = pd.DataFrame(
        imputer.fit_transform(values),
        index=train.index,
        columns=candidates,
    )
    target = train[target_name].astype(int)
    relevance = mutual_info_classif(
        matrix,
        target,
        discrete_features="auto",
        random_state=random_state,
    )
    ranking = sorted(
        zip(candidates, relevance, strict=True),
        key=lambda item: (-float(item[1]), item[0]),
    )
    selected: list[str] = []
    removed_correlated: list[str] = []
    for feature, _score in ranking:
        if selected:
            correlations = matrix[selected].corrwith(matrix[feature]).abs()
            if bool((correlations >= 0.98).any()):
                removed_correlated.append(feature)
                continue
        selected.append(feature)
        if len(selected) >= max(1, int(max_features)):
            break
    return selected, {
        "method": "initial_train_only_mutual_information_v1",
        "candidate_count": len(candidates),
        "selected_count": len(selected),
        "selected_features": selected,
        "removed_correlated_features": sorted(removed_correlated),
        "relevance_scores": {
            feature: round(float(score), 10)
            for feature, score in ranking
            if feature in selected
        },
        "train_window": _window_payload(train),
        "validation_labels_used": False,
        "test_rows_used": 0,
        "feature_set_frozen_across_folds": True,
    }


def _classification_metrics(
    actual: pd.Series,
    prediction: pd.Series,
) -> dict[str, float | int]:
    actual_values = actual.astype(int)
    prediction_values = prediction.astype(int)
    majority = float(actual_values.value_counts(normalize=True).max())
    return {
        "sample_count": int(len(actual_values)),
        "accuracy": round(
            float(accuracy_score(actual_values, prediction_values)),
            6,
        ),
        "balanced_accuracy": round(
            float(
                balanced_accuracy_score(
                    actual_values,
                    prediction_values,
                )
            ),
            6,
        ),
        "majority_class_baseline": round(majority, 6),
        "actual_positive_rate": round(float(actual_values.mean()), 6),
        "predicted_positive_rate": round(
            float(prediction_values.mean()),
            6,
        ),
    }


def _window_payload(frame: pd.DataFrame) -> dict[str, Any]:
    return {
        "start": frame.index[0].isoformat(),
        "end": frame.index[-1].isoformat(),
        "sample_count": int(len(frame)),
    }


def _normalized_importance(
    features: list[str],
    values: Any,
) -> dict[str, float]:
    parsed = np.asarray(values, dtype=float)
    total = float(np.abs(parsed).sum())
    if total <= 0:
        return {}
    normalized = np.abs(parsed) / total
    return dict(
        sorted(
            (
                (feature, round(float(value), 10))
                for feature, value in zip(
                    features,
                    normalized,
                    strict=True,
                )
            ),
            key=lambda item: item[1],
            reverse=True,
        )
    )


def _context_lineage_snapshot(
    report: dict[str, Any] | None,
    *,
    base_timeframe: str,
) -> dict[str, Any]:
    if not isinstance(report, dict):
        return {
            "status": "not_supplied",
            "base_timeframe": base_timeframe,
        }
    base_context = next(
        (
            item
            for item in report.get("base_contexts", [])
            if item.get("base_timeframe") == base_timeframe
        ),
        None,
    )
    return {
        "status": report.get("status"),
        "join_direction": report.get("join_direction"),
        "allow_future_context": report.get("allow_future_context"),
        "summary": report.get("summary"),
        "base_context": base_context,
    }


def _context_fingerprint(
    *,
    ticker: str,
    timeframe: str,
    target_name: str,
    selected_features: list[str],
    fold_results: list[dict[str, Any]],
    timeframe_context_report: dict[str, Any] | None,
    source_lineage: dict[str, Any] | None,
    config: WalkForwardValidationConfig,
) -> str:
    payload = {
        "ticker": ticker,
        "timeframe": timeframe,
        "target_name": target_name,
        "selected_features": selected_features,
        "fold_windows": [
            {
                "train_window": fold["train_window"],
                "purge_window": fold["purge_window"],
                "validation_window": fold["validation_window"],
            }
            for fold in fold_results
        ],
        "timeframe_context_lineage": _context_lineage_snapshot(
            timeframe_context_report,
            base_timeframe=timeframe,
        ),
        "source_lineage": source_lineage or {},
        "config": asdict(config),
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0
