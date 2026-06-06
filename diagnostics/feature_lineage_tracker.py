"""
Feature lineage tracker.

Purpose:
- track which component added/modified/dropped columns
- track whether columns survive cleaning/selection and reach X_train/model input
- write feature_lineage_report.json

Usage example:
    from diagnostics.feature_lineage_tracker import FeatureLineageTracker

    tracker = FeatureLineageTracker()
    df1 = tracker.capture_step("raw", df)
    out = enricher.enrich(df1)
    tracker.capture_component_output("TechnicalAnalysisEnricher", before=df1, after=out)
    tracker.capture_step("after_feature_selection", selected_df)
    tracker.mark_model_input(X_train)
    tracker.save("diagnostic_reports/feature_lineage_report.json")
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass
class ColumnStats:
    dtype: str
    nan_ratio: float
    inf_count: int
    unique_count: int


@dataclass
class ComponentLineage:
    component: str
    added_columns: list[str]
    removed_columns: list[str]
    modified_columns: list[str]
    row_count_before: int
    row_count_after: int
    added_column_stats: dict[str, ColumnStats]
    warnings: list[str] = field(default_factory=list)


class FeatureLineageTracker:
    def __init__(self):
        self.steps: dict[str, dict[str, Any]] = {}
        self.components: list[ComponentLineage] = []
        self.model_input_columns: list[str] = []
        self.dropped_reasons: dict[str, str] = {}

    def _column_stats(self, df: pd.DataFrame, col: str) -> ColumnStats:
        s = df[col]
        inf_count = 0
        try:
            inf_count = int((s == float("inf")).sum() + (s == float("-inf")).sum())
        except Exception:
            inf_count = 0
        return ColumnStats(
            dtype=str(s.dtype),
            nan_ratio=float(s.isna().mean()) if len(s) else 0.0,
            inf_count=inf_count,
            unique_count=int(s.nunique(dropna=True)),
        )

    def capture_step(self, name: str, df: pd.DataFrame) -> pd.DataFrame:
        self.steps[name] = {
            "rows": int(len(df)),
            "columns": list(df.columns),
            "column_count": int(len(df.columns)),
            "nan_ratio_by_column": {c: float(df[c].isna().mean()) for c in df.columns[:500]},
        }
        return df

    def capture_component_output(self, component: str, before: pd.DataFrame, after: pd.DataFrame) -> pd.DataFrame:
        before_cols = set(before.columns)
        after_cols = set(after.columns)

        added = sorted(after_cols - before_cols)
        removed = sorted(before_cols - after_cols)
        modified = []
        for col in sorted(before_cols & after_cols):
            try:
                if not before[col].equals(after[col]):
                    modified.append(col)
            except Exception:
                modified.append(col)

        warnings = []
        if len(before) != len(after):
            warnings.append("ROW_COUNT_CHANGED")
        if any(c.startswith("target_") for c in added):
            warnings.append("TARGET_COLUMN_ADDED")
        for col in added:
            if after[col].isna().mean() > 0.5:
                warnings.append(f"HIGH_NAN_RATIO:{col}")
            try:
                if ((after[col] == float("inf")) | (after[col] == float("-inf"))).any():
                    warnings.append(f"INF_VALUES:{col}")
            except Exception:
                pass

        self.components.append(ComponentLineage(
            component=component,
            added_columns=added,
            removed_columns=removed,
            modified_columns=modified,
            row_count_before=int(len(before)),
            row_count_after=int(len(after)),
            added_column_stats={c: self._column_stats(after, c) for c in added},
            warnings=warnings,
        ))
        return after

    def mark_dropped(self, column: str, reason: str) -> None:
        self.dropped_reasons[column] = reason

    def mark_model_input(self, X: pd.DataFrame) -> None:
        self.model_input_columns = list(X.columns)

    def report(self) -> dict[str, Any]:
        model_cols = set(self.model_input_columns)
        component_reports = []
        for comp in self.components:
            added = set(comp.added_columns)
            reached = sorted(added & model_cols)
            dropped = sorted(added - model_cols)
            component_reports.append({
                **asdict(comp),
                "added_column_stats": {k: asdict(v) for k, v in comp.added_column_stats.items()},
                "reached_model_input": reached,
                "not_in_model_input": dropped,
                "dropped_reasons": {c: self.dropped_reasons.get(c, "unknown") for c in dropped},
            })

        return {
            "steps": self.steps,
            "model_input_columns": self.model_input_columns,
            "components": component_reports,
        }

    def save(self, path: str | Path = "diagnostic_reports/feature_lineage_report.json") -> None:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(self.report(), indent=2, ensure_ascii=False), encoding="utf-8")
