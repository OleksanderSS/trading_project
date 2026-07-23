from __future__ import annotations

import json

from dean_os.pipeline_control.pipeline_control_train_validation_diagnostic import (
    PipelineControlTrainValidationDiagnostic,
)


def test_train_validation_diagnostic_ignores_frozen_test_metrics(tmp_path):
    results = []
    for ticker, train_balanced, validation_balanced in (
        ("AAA", 0.86, 0.52),
        ("BBB", 0.84, 0.55),
    ):
        report_path = tmp_path / f"{ticker}.json"
        report_path.write_text(
            json.dumps(
                {
                    "split_metrics": {
                        "train": {
                            "balanced_accuracy": train_balanced,
                            "sample_count": 200,
                        },
                        "validation": {
                            "balanced_accuracy": validation_balanced,
                            "accuracy": 0.54,
                            "majority_class_baseline": 0.58,
                            "actual_positive_rate": 0.45,
                            "predicted_positive_rate": 0.70,
                            "sample_count": 70,
                        },
                        "test": {
                            "score": 0.999999,
                            "secret_test_marker": "must_not_appear",
                        },
                    },
                    "split_windows": {
                        "test": {
                            "start": "2025-02-01",
                            "end": "2025-02-10",
                            "sample_count": 70,
                        }
                    },
                    "selected_features": [
                        "day_of_month_15m",
                        "state_day_of_month_15m",
                        "obv_15m",
                    ],
                    "feature_stability_analysis": {
                        "feature_stability_score": 0.62,
                        "feature_distribution_drift": {
                            "day_of_month_15m": {"drift_score": 0.82},
                            "state_day_of_month_15m": {"drift_score": 0.82},
                            "obv_15m": {"drift_score": 0.60},
                        },
                    },
                }
            ),
            encoding="utf-8",
        )
        results.append(
            {
                "context_key": f"{ticker}/15m",
                "ticker": ticker,
                "timeframe": "15m",
                "report_json": str(report_path),
            }
        )
    batch_path = tmp_path / "batch.json"
    batch_path.write_text(
        json.dumps(
            {
                "manifest_fingerprint": "manifest-1",
                "results": results,
            }
        ),
        encoding="utf-8",
    )

    payload = PipelineControlTrainValidationDiagnostic(tmp_path / "reports").build(
        batch_json=batch_path,
        save=False,
    )

    assert payload["summary"]["overfit_context_count"] == 2
    assert payload["summary"]["test_metrics_used_for_selection"] is False
    assert payload["cross_context_feature_drift"][0]["mean_drift_score"] == 0.82
    assert payload["duplicate_feature_representations"][0]["canonical_feature"] == "day_of_month"
    assert payload["experiment_contract"]["changed_plane"] == "feature_selection_only"
    assert payload["experiment_contract"]["model_contract_frozen"] is True
    assert payload["experiment_contract"]["test_access_allowed"] is False
    rendered = json.dumps(payload)
    assert "secret_test_marker" not in rendered
    assert "0.999999" not in rendered
    assert payload["summary"]["can_trade"] is False
