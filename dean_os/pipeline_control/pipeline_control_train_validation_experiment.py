from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.impute import SimpleImputer

from dean_os.artifact_writer import ReviewArtifactWriter
from dean_os.pipeline_control.pipeline_control_bounded_evidence_run import (
    _bound_source_frame,
    _classification_metrics,
    _load_source_frame,
    _prepare_macro_input,
    _prepare_model_frame,
    _run_stage_3_enrichment,
)
from dean_os.schemas import utc_now_iso
from dean_os.utils import json_ready
from src.pipeline.stages.modeling.pipeline_control_artifacts import (
    build_feature_distribution_stability_analysis,
)
from src.pipeline.target_column_utils import is_target_like_column


class PipelineControlTrainValidationExperiment:
    """Run one locked feature-selection experiment without reading test rows."""

    def __init__(
        self,
        output_dir: str | Path = "reports/dean_os/pipeline_control_train_validation_experiment_current",
    ):
        self.output_dir = Path(output_dir)

    async def run(
        self,
        *,
        batch_json: str | Path,
        diagnostic_json: str | Path,
        acknowledge_no_test: bool,
        save: bool = True,
    ) -> dict[str, Any]:
        if not acknowledge_no_test:
            raise ValueError("Explicit no-test acknowledgement is required.")
        batch_path = Path(batch_json)
        diagnostic_path = Path(diagnostic_json)
        batch = json.loads(batch_path.read_text(encoding="utf-8"))
        diagnostic = json.loads(diagnostic_path.read_text(encoding="utf-8"))
        contract = diagnostic.get("experiment_contract", {})
        if contract.get("experiment_id") != "train_only_redundancy_guard_v1":
            raise ValueError("Unsupported or missing diagnostic experiment contract.")
        if contract.get("test_access_allowed") is not False:
            raise ValueError("Experiment contract must explicitly prohibit test access.")

        contexts = []
        for item in batch.get("results", []):
            report_path = item.get("report_json")
            if not report_path:
                continue
            report = json.loads(Path(report_path).read_text(encoding="utf-8"))
            contexts.append(await _run_context_experiment(item, report))
        if not contexts:
            raise ValueError("Batch contains no runnable bounded context reports.")

        criteria = _evaluate_contract(contexts, contract)
        status = (
            "train_validation_candidate_passed_contract"
            if criteria["contract_passed"]
            else "train_validation_candidate_rejected"
        )
        summary = {
            "experiment_status": status,
            "context_count": len(contexts),
            "context_pass_count": criteria["context_pass_count"],
            "baseline_mean_validation_balanced_accuracy": _mean(
                contexts,
                "baseline_validation_balanced_accuracy",
            ),
            "candidate_mean_validation_balanced_accuracy": _mean(
                contexts,
                "candidate_validation_balanced_accuracy",
            ),
            "candidate_mean_train_validation_gap": _mean(
                contexts,
                "candidate_train_validation_gap",
            ),
            "candidate_mean_feature_stability_score": _mean(
                contexts,
                "candidate_feature_stability_score",
            ),
            "test_rows_loaded": 0,
            "test_metrics_read": False,
            "contract_passed": criteria["contract_passed"],
            "can_request_new_holdout": criteria["contract_passed"],
            "can_reuse_frozen_test_windows": False,
            "can_write_production_config": False,
            "can_promote_model": False,
            "can_trade": False,
        }
        payload = {
            "run_id": _run_id("pipeline_control_train_validation_experiment"),
            "created_at": utc_now_iso(),
            "mode": "pipeline_control_train_validation_experiment",
            "inputs": {
                "batch_json": str(batch_path),
                "diagnostic_json": str(diagnostic_path),
                "batch_manifest_fingerprint": batch.get("manifest_fingerprint"),
                "experiment_id": contract.get("experiment_id"),
                "acknowledge_no_test": acknowledge_no_test,
            },
            "summary": summary,
            "contract_evaluation": criteria,
            "contexts": contexts,
            "validation_freeze": {
                "frozen_after_this_run": True,
                "context_windows": [
                    {
                        "context_key": context["context_key"],
                        "validation_window": context["validation_window"],
                    }
                    for context in contexts
                ],
                "rule": (
                    "Do not compare more feature/model variants on these validation windows. "
                    "A new proposal requires rolling train-only folds or newly collected data."
                ),
            },
            "next_step": (
                contract.get("next_holdout_rule")
                if criteria["contract_passed"]
                else (
                    "Reject this candidate. Do not inspect frozen test metrics or run another variant "
                    "on the same validation windows."
                )
            ),
            "explicit_non_actions": [
                "Source rows at or after each frozen test start were excluded before Stage 3.",
                "No test metric, return, drawdown, or profitability value was read.",
                "No hyperparameter search or repeated variant loop ran.",
                "No model artifact was promoted or written to production.",
                "No recommendation, order, paper trade, or live trade was created.",
            ],
        }
        if save:
            payload["saved_paths"] = ReviewArtifactWriter(self.output_dir).write(
                payload=payload,
                markdown=render_train_validation_experiment_markdown(payload),
                run_id=payload["run_id"],
            )
        return json_ready(payload)


def render_train_validation_experiment_markdown(payload: dict[str, Any]) -> str:
    summary = payload.get("summary", {})
    lines = [
        "# Pipeline Control Train/Validation Experiment",
        "",
        f"- Status: `{summary.get('experiment_status')}`",
        f"- Contexts: {summary.get('context_count')}",
        f"- Contexts passing: {summary.get('context_pass_count')}",
        f"- Baseline mean validation balanced accuracy: {summary.get('baseline_mean_validation_balanced_accuracy')}",
        f"- Candidate mean validation balanced accuracy: {summary.get('candidate_mean_validation_balanced_accuracy')}",
        f"- Candidate mean train-validation gap: {summary.get('candidate_mean_train_validation_gap')}",
        f"- Candidate mean feature stability: {summary.get('candidate_mean_feature_stability_score')}",
        f"- Test rows loaded: {summary.get('test_rows_loaded')}",
        f"- Test metrics read: {summary.get('test_metrics_read')}",
        f"- Contract passed: {summary.get('contract_passed')}",
        f"- Can trade: {summary.get('can_trade')}",
        "",
        "## Contexts",
        "",
    ]
    for context in payload.get("contexts", []):
        lines.append(
            f"- `{context.get('context_key')}`: baseline_val_bal="
            f"{context.get('baseline_validation_balanced_accuracy')} candidate_val_bal="
            f"{context.get('candidate_validation_balanced_accuracy')} gap="
            f"{context.get('candidate_train_validation_gap')} stability="
            f"{context.get('candidate_feature_stability_score')} pass="
            f"{context.get('context_passed')}"
        )
    return "\n".join(lines).strip() + "\n"


async def _run_context_experiment(
    item: dict[str, Any],
    report: dict[str, Any],
) -> dict[str, Any]:
    inputs = report.get("inputs", {})
    source_path = Path(inputs["source_path"])
    ticker = str(inputs["ticker"])
    timeframe = str(inputs["timeframe"])
    target_name = str(inputs["target_name"])
    source = _load_source_frame(source_path)
    bounded = _bound_source_frame(
        source,
        ticker=ticker,
        timeframe=timeframe,
        start=inputs.get("start"),
        end=inputs.get("end"),
        max_rows=int(inputs.get("max_rows", 480)),
    )
    test_start = pd.to_datetime(
        report["split_windows"]["test"]["start"],
        utc=True,
    )
    datetime_column = next(
        column for column in ("datetime", "timestamp", "date") if column in bounded.columns
    )
    bounded = bounded.loc[
        pd.to_datetime(bounded[datetime_column], errors="coerce", utc=True) < test_start
    ].copy()
    macro_path = report.get("macro_provenance", {}).get("path")
    macro_frame, _, macro_checks = _prepare_macro_input(
        Path(macro_path) if macro_path else None,
        bounded_frame=bounded,
    )
    if any(check["status"] == "fail" for check in macro_checks):
        raise ValueError(f"Macro input failed for {ticker}/{timeframe}: {macro_checks}")
    enriched = await _run_stage_3_enrichment(
        bounded,
        timeframe=timeframe,
        macro_frame=macro_frame,
    )
    prepared = _prepare_model_frame(enriched, target_name=target_name)
    train = _window_slice(prepared, report["split_windows"]["train"])
    validation = _window_slice(prepared, report["split_windows"]["validation"])
    if train.empty or validation.empty:
        raise ValueError(f"Could not reconstruct train/validation windows for {ticker}/{timeframe}.")

    baseline_features = [
        feature
        for feature in report.get("selected_features", [])
        if feature in train.columns
    ]
    baseline_metrics = _fit_and_score(train, validation, target_name, baseline_features)
    candidate_features, selection = _select_candidate_features(
        train,
        target_name=target_name,
        max_features=24,
    )
    candidate_metrics = _fit_and_score(train, validation, target_name, candidate_features)
    context_passed = bool(
        candidate_metrics["validation"]["balanced_accuracy"]
        >= baseline_metrics["validation"]["balanced_accuracy"] + 0.02
        and candidate_metrics["train"]["balanced_accuracy"]
        - candidate_metrics["validation"]["balanced_accuracy"]
        <= 0.20
        and candidate_metrics["feature_stability"]["feature_stability_score"] >= 0.70
        and abs(
            candidate_metrics["validation"]["actual_positive_rate"]
            - candidate_metrics["validation"]["predicted_positive_rate"]
        )
        <= 0.20
    )
    return {
        "context_key": item.get("context_key"),
        "ticker": ticker,
        "timeframe": timeframe,
        "train_window": _window_payload(train),
        "validation_window": _window_payload(validation),
        "test_exclusion_boundary": test_start.isoformat(),
        "test_rows_loaded": 0,
        "baseline_feature_count": len(baseline_features),
        "candidate_feature_count": len(candidate_features),
        "candidate_features": candidate_features,
        "feature_selection": selection,
        "baseline_train_balanced_accuracy": baseline_metrics["train"]["balanced_accuracy"],
        "baseline_validation_balanced_accuracy": baseline_metrics["validation"][
            "balanced_accuracy"
        ],
        "candidate_train_balanced_accuracy": candidate_metrics["train"][
            "balanced_accuracy"
        ],
        "candidate_validation_balanced_accuracy": candidate_metrics["validation"][
            "balanced_accuracy"
        ],
        "candidate_validation_accuracy": candidate_metrics["validation"]["accuracy"],
        "candidate_validation_majority_baseline": candidate_metrics["validation"][
            "majority_class_baseline"
        ],
        "candidate_train_validation_gap": (
            candidate_metrics["train"]["balanced_accuracy"]
            - candidate_metrics["validation"]["balanced_accuracy"]
        ),
        "candidate_validation_positive_rate_gap": abs(
            candidate_metrics["validation"]["actual_positive_rate"]
            - candidate_metrics["validation"]["predicted_positive_rate"]
        ),
        "candidate_feature_stability_score": candidate_metrics["feature_stability"][
            "feature_stability_score"
        ],
        "candidate_unstable_feature_count": candidate_metrics["feature_stability"][
            "unstable_feature_count"
        ],
        "context_passed": context_passed,
    }


def _window_slice(frame: pd.DataFrame, window: dict[str, Any]) -> pd.DataFrame:
    start = pd.to_datetime(window["start"], utc=True)
    end = pd.to_datetime(window["end"], utc=True)
    return frame.loc[(frame.index >= start) & (frame.index <= end)].copy()


def _select_candidate_features(
    train: pd.DataFrame,
    *,
    target_name: str,
    max_features: int,
) -> tuple[list[str], dict[str, Any]]:
    excluded = {
        target_name,
        "__forward_return",
        "open",
        "high",
        "low",
        "close",
        "volume",
    }
    numeric = train.select_dtypes(include=[np.number, "bool"])
    candidates = []
    excluded_calendar = []
    for column in numeric.columns:
        name = str(column)
        if name in excluded or is_target_like_column(name):
            continue
        canonical = _canonical_feature_name(name)
        if canonical in {"day_of_month", "day_of_year", "week_of_year"}:
            excluded_calendar.append(name)
            continue
        values = pd.to_numeric(train[name], errors="coerce").replace(
            [np.inf, -np.inf],
            np.nan,
        )
        if float(values.notna().mean()) < 0.80:
            continue
        variance = float(values.var())
        if not math.isfinite(variance) or variance <= 0:
            continue
        candidates.append(name)

    representation_groups: dict[str, list[str]] = {}
    for feature in candidates:
        representation_groups.setdefault(_canonical_feature_name(feature), []).append(feature)
    deduplicated = []
    removed_representations = []
    for features in representation_groups.values():
        features.sort(key=_representation_priority)
        deduplicated.append(features[0])
        removed_representations.extend(features[1:])

    imputer = SimpleImputer(strategy="median")
    values = train[deduplicated].replace([np.inf, -np.inf], np.nan)
    matrix = pd.DataFrame(
        imputer.fit_transform(values),
        index=values.index,
        columns=deduplicated,
    )
    target = train[target_name].astype(int)
    relevance = mutual_info_classif(
        matrix,
        target,
        discrete_features="auto",
        random_state=42,
    )
    ranking = sorted(
        zip(deduplicated, relevance, strict=True),
        key=lambda item: (-float(item[1]), item[0]),
    )
    selected = []
    removed_correlated = []
    for feature, _score in ranking:
        if selected:
            correlations = matrix[selected].corrwith(matrix[feature]).abs()
            if bool((correlations >= 0.98).any()):
                removed_correlated.append(feature)
                continue
        selected.append(feature)
        if len(selected) >= max(1, int(max_features)):
            break
    if not selected:
        raise ValueError("Train-only selector produced no candidate features.")
    return selected, {
        "method": "train_only_mutual_information_with_redundancy_guard_v1",
        "candidate_count_before_guards": len(candidates),
        "candidate_count_after_representation_guard": len(deduplicated),
        "selected_count": len(selected),
        "excluded_calendar_features": sorted(excluded_calendar),
        "removed_duplicate_representations": sorted(removed_representations),
        "removed_correlated_features": sorted(removed_correlated),
        "relevance_scores": {
            feature: float(score)
            for feature, score in ranking
            if feature in selected
        },
        "target_rows_used": len(target),
        "validation_labels_used_for_selection": False,
        "test_rows_used": 0,
    }


def _fit_and_score(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    target_name: str,
    features: list[str],
) -> dict[str, Any]:
    if not features:
        raise ValueError("Feature list is empty.")
    imputer = SimpleImputer(strategy="median")
    train_values = train[features].replace([np.inf, -np.inf], np.nan)
    validation_values = validation[features].replace([np.inf, -np.inf], np.nan)
    train_matrix = pd.DataFrame(
        imputer.fit_transform(train_values),
        index=train.index,
        columns=features,
    )
    validation_matrix = pd.DataFrame(
        imputer.transform(validation_values),
        index=validation.index,
        columns=features,
    )
    model = RandomForestClassifier(
        n_estimators=128,
        max_depth=8,
        min_samples_leaf=5,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=1,
    )
    train_target = train[target_name].astype(int)
    validation_target = validation[target_name].astype(int)
    model.fit(train_matrix, train_target)
    train_prediction = pd.Series(model.predict(train_matrix), index=train.index)
    validation_prediction = pd.Series(
        model.predict(validation_matrix),
        index=validation.index,
    )
    return {
        "train": _classification_metrics(train_target, train_prediction),
        "validation": _classification_metrics(validation_target, validation_prediction),
        "feature_stability": build_feature_distribution_stability_analysis(
            train_matrix,
            validation_matrix,
            features,
        ),
    }


def _evaluate_contract(
    contexts: list[dict[str, Any]],
    contract: dict[str, Any],
) -> dict[str, Any]:
    criteria = contract.get("validation_acceptance_criteria", {})
    mean_validation = _mean(contexts, "candidate_validation_balanced_accuracy")
    mean_gap = _mean(contexts, "candidate_train_validation_gap")
    mean_stability = _mean(contexts, "candidate_feature_stability_score")
    context_pass_count = sum(1 for context in contexts if context["context_passed"])
    checks = {
        "minimum_contexts_passing": context_pass_count
        >= int(criteria.get("minimum_contexts_passing", len(contexts))),
        "minimum_mean_validation_balanced_accuracy": mean_validation
        >= float(criteria.get("minimum_mean_validation_balanced_accuracy", 1.0)),
        "maximum_mean_train_validation_gap": mean_gap
        <= float(criteria.get("maximum_mean_train_validation_gap", 0.0)),
        "minimum_mean_feature_stability_score": mean_stability
        >= float(criteria.get("minimum_mean_feature_stability_score", 1.0)),
        "maximum_context_positive_rate_gap": all(
            context["candidate_validation_positive_rate_gap"]
            <= float(criteria.get("maximum_context_positive_rate_gap", 0.0))
            for context in contexts
        ),
    }
    return {
        "checks": checks,
        "context_pass_count": context_pass_count,
        "required_context_pass_count": criteria.get("minimum_contexts_passing"),
        "candidate_mean_validation_balanced_accuracy": mean_validation,
        "candidate_mean_train_validation_gap": mean_gap,
        "candidate_mean_feature_stability_score": mean_stability,
        "contract_passed": all(checks.values()),
    }


def _canonical_feature_name(feature: str) -> str:
    normalized = str(feature)
    for prefix in ("state_", "market_context_"):
        if normalized.startswith(prefix):
            normalized = normalized[len(prefix) :]
    if normalized.endswith("_15m"):
        normalized = normalized[:-4]
    return normalized.lower()


def _representation_priority(feature: str) -> tuple[int, str]:
    if feature.startswith("state_"):
        return (2, feature)
    if feature.startswith("market_context_"):
        return (1, feature)
    return (0, feature)


def _window_payload(frame: pd.DataFrame) -> dict[str, Any]:
    return {
        "start": str(frame.index[0]),
        "end": str(frame.index[-1]),
        "sample_count": len(frame),
    }


def _mean(items: list[dict[str, Any]], key: str) -> float:
    values = [float(item[key]) for item in items]
    return sum(values) / len(values)


def _run_id(prefix: str) -> str:
    return f"{prefix}_{utc_now_iso().replace(':', '').replace('-', '').replace('.', '')}"
