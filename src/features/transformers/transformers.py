from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler


@dataclass
class _BaseScalerTransformer:
    columns: list[str] = field(default_factory=list)

    scaler_cls: type[Any] = StandardScaler

    def __post_init__(self) -> None:
        self.scalers = {}

    def fit(self, df: pd.DataFrame) -> _BaseScalerTransformer:
        for col in self._available_columns(df):
            values = df[[col]].dropna()
            if values.empty:
                continue
            scaler = self.scaler_cls()
            scaler.fit(values)
            self.scalers[col] = scaler
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        result = df.copy()
        for col, scaler in self.scalers.items():
            if col not in result.columns:
                continue
            values = result[[col]].dropna()
            if values.empty:
                continue
            result.loc[values.index, col] = scaler.transform(values).ravel()
        return result

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)

    def _available_columns(self, df: pd.DataFrame) -> Iterable[str]:
        if self.columns:
            return [col for col in self.columns if col in df.columns]
        return df.select_dtypes(include="number").columns.tolist()


class StandardScalerTransformer(_BaseScalerTransformer):
    scaler_cls: type[Any] = StandardScaler


class MinMaxScalerTransformer(_BaseScalerTransformer):
    scaler_cls: type[Any] = MinMaxScaler
