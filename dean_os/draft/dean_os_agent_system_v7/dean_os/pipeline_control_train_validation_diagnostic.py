from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from dean_os.draft.dean_os_agent_system_v7.dean_os.artifact_writer import ReviewArtifactWriter
from dean_os.schemas import utc_now_iso
from dean_os.utils import json_ready


class PipelineControlTrainValidationDiagnostic:
    """Diagnose overfit and feature drift without using frozen test metrics."""

    def __init__(
        self,
        output_dir: str | Path = "reports/dean_os/pipeline_control_train_validation_diagnostic_current",
    ):
        self.output_dir = Path(output_dir)

    def build(
        self,
        *,
        batch_json: str | Path,
        save: bool = True,
    ) -> dict[str, Any]:
        batch_path = Path(batch_json)
        batch = json.loads(batch_path.read_text(encoding="utf-8"))
        contexts = []
        drift_by_feature: dict[str, list[float]] = defaultdict(list)
        feature_occurrence: Counter[str] = Counter()
        frozen_windows = []
        for item in batch.get("results", []):
            report_path = item.get("report_json")
            if not report_path:
                continue
            report = json.loads(Path(report_path).read_text(encoding="utf-8"))
            context = _context_diagnostic(item, report)
            contexts.append(context)
            for feature in report.get("selected_features", []):
                feature_occurrence[str(feature)] += 1
            for feature, metrics in (
                report.get("feature_stability_analysis", {})
                .get("feature_distribution_drift", {})
                .items()
            ):
                score = metrics.get("drift_score")
                if score is not None:
                    drift_by_feature[str(feature)].append(float(score))
            test_window = report.get("split_windows", {}).get("test")
            if test_window:
                frozen_windows.append(
                    {
                        "context_key": context["context_key"],
                        "window": test_window,
                        "metrics_read_for_selection": False,
                    }
                )
        if not contexts:
            raise ValueError("Batch contains no readable bounded context reports.")

        cross_context_drift = _cross_context_drift(drift_by_feature)
        duplicate_representations = _duplicate_feature_representations(
            feature_occurrence,
            context_count=len(contexts),
        )
        baseline = _baseline_summary(contexts)
        experiment_contract = _experiment_contract(
            contexts=contexts,
            baseline=baseline,
            cross_context_drift=cross_context_drift,
            duplicate_representations=duplicate_representations,
            frozen_windows=frozen_windows,
        )
        summary = {
            "diagnostic_status": "train_validation_diagnostic_ready",
            "context_count": len(contexts),
            "overfit_context_count": sum(
                1 for context in contexts if context["overfit_gap"] > 0.20
            ),
            "validation_below_majority_baseline_count": sum(
                1 for context in contexts if context["validation_edge_vs_majority"] < 0
            ),
            "mean_train_balanced_accuracy": baseline["mean_train_balanced_accuracy"],
            "mean_validation_balanced_accuracy": baseline[
                "mean_validation_balanced_accuracy"
            ],
            "mean_train_validation_gap": baseline["mean_train_validation_gap"],
            "mean_feature_stability_score": baseline["mean_feature_stability_score"],
            "high_drift_feature_count": sum(
                1 for item in cross_context_drift if item["mean_drift_score"] >= 0.50
            ),
            "duplicate_representation_group_count": len(duplicate_representations),
            "test_metrics_used_for_selection": False,
            "proposal_ready": True,
            "can_run_experiment_automatically": False,
            "can_use_frozen_test_for_selection": False,
            "can_trade": False,
        }
        payload = {
            "run_id": _run_id("pipeline_control_train_validation_diagnostic"),
            "created_at": utc_now_iso(),
            "mode": "pipeline_control_train_validation_diagnostic",
            "inputs": {
                "batch_json": str(batch_path),
                "batch_manifest_fingerprint": batch.get("manifest_fingerprint"),
            },
            "summary": summary,
            "baseline_train_validation": baseline,
            "context_diagnostics": contexts,
            "cross_context_feature_drift": cross_context_drift,
            "duplicate_feature_representations": duplicate_representations,
            "frozen_test_windows": frozen_windows,
            "experiment_contract": experiment_contract,
            "explicit_non_actions": [
                "Frozen test scores, returns, drawdown, and profitability were not read for proposal selection.",
                "No feature set was changed in a saved production model.",
                "No model was trained, evaluated on test, tuned, or promoted.",
                "No learning-memory/config write, recommendation, order, or trade occurred.",
            ],
        }
        if save:
            payload["saved_paths"] = ReviewArtifactWriter(self.output_dir).write(
                payload=payload,
                markdown=render_train_validation_diagnostic_markdown(payload),
                run_id=payload["run_id"],
            )
        return json_ready(payload)


def render_train_validation_diagnostic_markdown(payload: dict[str, Any]) -> str:
    summary = payload.get("summary", {})
    contract = payload.get("experiment_contract", {})
    lines = [
        "# Pipeline Control Train/Validation Diagnostic",
        "",
        f"- Status: `{summary.get('diagnostic_status')}`",
        f"- Contexts: {summary.get('context_count')}",
        f"- Overfit contexts: {summary.get('overfit_context_count')}",
        f"- Validation below majority baseline: {summary.get('validation_below_majority_baseline_count')}",
        f"- Mean train balanced accuracy: {summary.get('mean_train_balanced_accuracy')}",
        f"- Mean validation balanced accuracy: {summary.get('mean_validation_balanced_accuracy')}",
        f"- Mean train-validation gap: {summary.get('mean_train_validation_gap')}",
        f"- Mean feature stability: {summary.get('mean_feature_stability_score')}",
        f"- Test metrics used for selection: {summary.get('test_metrics_used_for_selection')}",
        f"- Can trade: {summary.get('can_trade')}",
        "",
        "## Contexts",
        "",
    ]
    for context in payload.get("context_diagnostics", []):
        lines.append(
            f"- `{context.get('context_key')}`: train_bal={context.get('train_balanced_accuracy')} "
            f"validation_bal={context.get('validation_balanced_accuracy')} "
            f"gap={context.get('overfit_gap')} stability={context.get('feature_stability_score')}"
        )
    lines.extend(["", "## Highest Cross-Context Drift", ""])
    for item in payload.get("cross_context_feature_drift", [])[:12]:
        lines.append(
            f"- `{item.get('feature')}`: mean={item.get('mean_drift_score')} "
            f"max={item.get('max_drift_score')} contexts={item.get('context_count')}"
        )
    lines.extend(
        [
            "",
            "## Proposal-Only Experiment",
            "",
            f"- ID: `{contract.get('experiment_id')}`",
            f"- Changed plane: `{contract.get('changed_plane')}`",
            f"- Model frozen: {contract.get('model_contract_frozen')}",
            f"- Test access allowed: {contract.get('test_access_allowed')}",
        ]
    )
    lines.extend(f"- {item}" for item in contract.get("feature_policy_changes", []))
    return "\n".join(lines).strip() + "\n"


def _context_diagnostic(item: dict[str, Any], report: dict[str, Any]) -> dict[str, Any]:
    split_metrics = report.get("split_metrics", {})
    train = split_metrics.get("train", {})
    validation = split_metrics.get("validation", {})
    stability = report.get("feature_stability_analysis", {})
    train_balanced = float(train.get("balanced_accuracy", train.get("score", 0.0)))
    validation_balanced = float(
        validation.get("balanced_accuracy", validation.get("score", 0.0))
    )
    validation_accuracy = float(validation.get("accuracy", validation.get("score", 0.0)))
    majority = float(validation.get("majority_class_baseline", 0.0))
    feature_drift = stability.get("feature_distribution_drift", {})
    top_drift = sorted(
        (
            {
                "feature": str(feature),
                "drift_score": float(metrics.get("drift_score", 0.0)),
                "status": metrics.get("status"),
            }
            for feature, metrics in feature_drift.items()
        ),
        key=lambda row: (-row["drift_score"], row["feature"]),
    )
    return {
        "context_key": item.get("context_key"),
        "ticker": item.get("ticker"),
        "timeframe": item.get("timeframe"),
        "train_sample_count": train.get("sample_count"),
        "validation_sample_count": validation.get("sample_count"),
        "train_balanced_accuracy": train_balanced,
        "validation_balanced_accuracy": validation_balanced,
        "validation_accuracy": validation_accuracy,
        "validation_majority_baseline": majority,
        "validation_edge_vs_majority": validation_accuracy - majority,
        "overfit_gap": train_balanced - validation_balanced,
        "validation_actual_positive_rate": validation.get("actual_positive_rate"),
        "validation_predicted_positive_rate": validation.get("predicted_positive_rate"),
        "validation_positive_rate_gap": abs(
            float(validation.get("actual_positive_rate", 0.0))
            - float(validation.get("predicted_positive_rate", 0.0))
        ),
        "feature_stability_score": stability.get("feature_stability_score"),
        "features_with_drift_at_least_0_50": sum(
            1 for row in top_drift if row["drift_score"] >= 0.50
        ),
        "top_feature_drift": top_drift[:10],
        "test_metrics_read_for_selection": False,
    }


def _cross_context_drift(drift_by_feature: dict[str, list[float]]) -> list[dict[str, Any]]:
    rows = []
    for feature, values in drift_by_feature.items():
        rows.append(
            {
                "feature": feature,
                "context_count": len(values),
                "mean_drift_score": sum(values) / len(values),
                "max_drift_score": max(values),
                "contexts_above_0_50": sum(1 for value in values if value >= 0.50),
                "contexts_above_0_75": sum(1 for value in values if value >= 0.75),
            }
        )
    return sorted(rows, key=lambda row: (-row["mean_drift_score"], row["feature"]))


def _duplicate_feature_representations(
    feature_occurrence: Counter[str],
    *,
    context_count: int,
) -> list[dict[str, Any]]:
    groups: dict[str, list[str]] = defaultdict(list)
    for feature in feature_occurrence:
        canonical = _canonical_feature_name(feature)
        groups[canonical].append(feature)
    return [
        {
            "canonical_feature": canonical,
            "representations": sorted(features),
            "occurrence_count": sum(feature_occurrence[feature] for feature in features),
            "appears_in_all_contexts": all(
                feature_occurrence[feature] == context_count for feature in features
            ),
        }
        for canonical, features in sorted(groups.items())
        if len(features) > 1
    ]


def _canonical_feature_name(feature: str) -> str:
    normalized = str(feature)
    for prefix in ("state_", "market_context_"):
        if normalized.startswith(prefix):
            normalized = normalized[len(prefix) :]
    if normalized.endswith("_15m"):
        normalized = normalized[:-4]
    return normalized.lower()


def _baseline_summary(contexts: list[dict[str, Any]]) -> dict[str, Any]:
    count = len(contexts)
    return {
        "context_count": count,
        "mean_train_balanced_accuracy": sum(
            context["train_balanced_accuracy"] for context in contexts
        )
        / count,
        "mean_validation_balanced_accuracy": sum(
            context["validation_balanced_accuracy"] for context in contexts
        )
        / count,
        "mean_train_validation_gap": sum(
            context["overfit_gap"] for context in contexts
        )
        / count,
        "mean_feature_stability_score": sum(
            float(context.get("feature_stability_score") or 0.0)
            for context in contexts
        )
        / count,
        "mean_validation_edge_vs_majority": sum(
            context["validation_edge_vs_majority"] for context in contexts
        )
        / count,
    }


def _experiment_contract(
    *,
    contexts: list[dict[str, Any]],
    baseline: dict[str, Any],
    cross_context_drift: list[dict[str, Any]],
    duplicate_representations: list[dict[str, Any]],
    frozen_windows: list[dict[str, Any]],
) -> dict[str, Any]:
    high_drift = [
        item["feature"]
        for item in cross_context_drift
        if item["mean_drift_score"] >= 0.50
    ]
    return {
        "experiment_id": "train_only_redundancy_guard_v1",
        "contract_status": "proposal_ready_operator_invocation_required",
        "objective": "Reduce cross-context overfit and train/validation feature drift.",
        "changed_plane": "feature_selection_only",
        "model_contract_frozen": True,
        "model_contract": {
            "type": "random_forest",
            "n_estimators": 128,
            "max_depth": 8,
            "min_samples_leaf": 5,
            "class_weight": "balanced_subsample",
            "random_state": 42,
        },
        "feature_policy_changes": [
            "Reduce maximum selected features from 40 to 24.",
            "Score candidate relevance on training rows only; validation labels may evaluate but never select features.",
            "Remove duplicate state_/market_context_ representations of the same base calendar field.",
            "Exclude monotonic calendar counters day_of_month, day_of_year, and week_of_year plus their state variants.",
            "Apply a train-only absolute-correlation guard at 0.98 before ranking.",
            "Keep the model, split gaps, costs, and target contract unchanged.",
        ],
        "observed_high_drift_features": high_drift,
        "observed_duplicate_representation_groups": [
            item["canonical_feature"] for item in duplicate_representations
        ],
        "validation_acceptance_criteria": {
            "minimum_contexts_passing": max(1, len(contexts) - 1),
            "minimum_mean_validation_balanced_accuracy": baseline[
                "mean_validation_balanced_accuracy"
            ]
            + 0.03,
            "maximum_mean_train_validation_gap": 0.20,
            "minimum_mean_feature_stability_score": 0.70,
            "maximum_context_positive_rate_gap": 0.20,
        },
        "test_access_allowed": False,
        "frozen_test_windows": frozen_windows,
        "next_holdout_rule": (
            "Only after the train/validation criteria pass, predeclare one new window strictly "
            "after every frozen test end and evaluate the single locked candidate once."
        ),
        "can_run_automatically": False,
        "can_write_production_config": False,
        "can_promote_model": False,
        "can_trade": False,
    }


def _run_id(prefix: str) -> str:
    return f"{prefix}_{utc_now_iso().replace(':', '').replace('-', '').replace('.', '')}"
