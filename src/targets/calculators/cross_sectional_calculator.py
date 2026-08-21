"""Targets measured against the other names at the same instant.

Every target this project has ever produced is ABSOLUTE: "did AAPL rise". That
makes the market's own drift the thing a model has to beat, and the drift is
large -- measured on 30 years of daily bars, equal-weight buy-and-hold returns
+18.06% a year. A model that predicts direction well and captures less than the
drift has produced nothing an investor wants, and the promotion gate could not
see the difference until 2026-08-20.

A cross-sectional target removes the drift from the question. Instead of "will
AAPL rise" it asks "will AAPL beat the average of everything we hold", so beta
stops being the opponent and the model is graded on RANKING.

MEASURED BEFORE BUILDING, which is the rule that has saved this project twice.
Walk-forward over 11 independent folds, identical folds, features and model,
excess over passive holding:

    absolute target (what the pipeline makes)   6/11 folds   +0.00021   t 0.55
    relative target                             9/11         +0.00132   t 2.78

Confirmed afterwards on test years 1999/2001/2003 -- the only segment that took
no part in choosing the variant -- against a criterion fixed in advance:
3/3 folds positive. See docs/preregistration/2026-08-20_cross_sectional_regression.md,
including the follow-up showing that as a PORTFOLIO the advantage is worth
about half a percent a year over levering the same holding. The target is
better. It is not a fortune.

WHY THIS CANNOT GO THROUGH THE NORMAL PATH. TargetOrchestrator computes targets
per ticker group. A cross-sectional value needs every ticker at one timestamp,
and computed per group the cross-sectional mean of one ticker IS that ticker,
so the relative return would be exactly zero on every row -- silently, with no
error and a full column of plausible-looking numbers. REQUIRES_FULL_FRAME makes
the orchestrator hand over the whole frame instead, and a test pins the failure
mode it prevents.
"""
from __future__ import annotations

import logging
from typing import Any, ClassVar

import numpy as np
import pandas as pd

from src.core.logging.logger import ProjectLogger

logger = ProjectLogger.get_logger("CrossSectionalCalculator")


class CrossSectionalCalculator:
    """Relative-to-peers targets. Needs the whole frame, not one ticker."""

    #: The orchestrator must NOT split by ticker before calling this.
    REQUIRES_FULL_FRAME: ClassVar[bool] = True

    SUPPORTED_PARAMS = frozenset({
        "base_col", "shift", "method", "min_names", "adjust_for_costs",
        "transaction_costs", "description", "horizon",
    })

    #: How the peer comparison is expressed.
    METHODS = ("demean", "rank")

    def calculate(self, df: pd.DataFrame, base_col: str = "close",
                  shift: int = -5, method: str = "demean",
                  min_names: int = 5, **kwargs: Any) -> pd.Series:
        """Forward return minus the cross-sectional mean of the same instant.

        `method='rank'` returns the percentile rank of that excess within the
        instant instead, on [0, 1]. Ranking is what the decision actually does
        -- buy the top share at each date -- so a rank target trains the model
        on the quantity it will be judged by, and is immune to a single
        outlier dragging the mean.

        `min_names` refuses instants with too few tickers to have a
        cross-section at all. With three names, "beating the average" is
        mostly noise about which of three moved most, and quietly emitting it
        would put that noise in the training set.
        """
        if base_col not in df.columns:
            raise ValueError(f"Base column '{base_col}' not found.")
        if "datetime" not in df.columns:
            raise ValueError(
                "A cross-sectional target needs a datetime column to group "
                "names by instant; without it there is no cross-section."
            )
        if shift >= 0:
            raise ValueError(
                f"Shift must be negative for a future target. Got {shift}."
            )
        if method not in self.METHODS:
            raise ValueError(
                f"Unknown cross-sectional method '{method}'. "
                f"Expected one of {self.METHODS}."
            )
        if "ticker" not in df.columns:
            raise ValueError(
                "A cross-sectional target needs a ticker column. Received a "
                "frame without one, which usually means the orchestrator split "
                "by ticker before calling -- the failure that makes every "
                "relative value exactly zero."
            )

        future = df.groupby("ticker")[base_col].shift(shift)
        raw = (future - df[base_col]) / df[base_col].replace(0, np.nan)

        by_instant = raw.groupby(df["datetime"])
        names = by_instant.transform("count")
        enough = names >= int(min_names)

        if method == "demean":
            out = raw - by_instant.transform("mean")
        else:
            out = by_instant.rank(pct=True)

        out = out.where(enough)

        # Cost is NOT subtracted here, and that is deliberate. A relative
        # target is a spread: harvesting it means one round trip on the name
        # and, if hedged, another on the market. Which of those applies is a
        # decision the strategy makes, not a property of the label -- and
        # baking cost into a label is what conflated "will it rise" with "will
        # it rise enough" on the absolute targets, holding the event rate at
        # 29.9% where the market's own drift implies about 53%.
        if kwargs.get("adjust_for_costs"):
            logger.warning(
                "adjust_for_costs is ignored for cross-sectional targets: the "
                "cost of harvesting a spread depends on whether it is hedged, "
                "so it belongs to the decision layer rather than to the label."
            )

        dropped = int((~enough & raw.notna()).sum())
        if dropped:
            logger.info(
                "Cross-sectional target: %d rows dropped for having fewer than "
                "%d names at their instant.", dropped, int(min_names),
            )
        logger.info(
            "Cross-sectional '%s' target over %d instants, median %d names; "
            "mean %.6f (a demeaned target should sit near zero by construction).",
            method, int(df["datetime"].nunique()), int(names.median() or 0),
            float(out.mean(skipna=True)) if out.notna().any() else float("nan"),
        )
        return out
