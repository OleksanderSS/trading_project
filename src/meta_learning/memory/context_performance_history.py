"""Per-context, per-model performance, in the shapes its two consumers need.

Two subsystems were each missing the same table and neither built it:

- `ContextualModelSelector` wants a `KnnSimilarityFinder` fitted on historical
  contexts, plus an outcomes frame with one `target_<model>` column per model.
  Nothing constructed either, so `select_models` returned None on the first
  isinstance check and the whole selector was inert.
- `LearningLoopsEngine` wants to know which market states a given model loses
  in, to turn them into rules.

Both are answered by the same query over `experience_diary`, so it lives here
once rather than as a third and fourth copy of "group the diary by context"
(the codebase has already paid for that pattern more than once).

WHAT COUNTS AS PERFORMANCE. The win rate over resolved trades, matching
`ContextualWeightCalculator` exactly. Not average P&L: that is unbounded and
scale-dependent across tickers, and `ContextualModelSelector` compares models
with `max()` across a row, where one large-P&L ticker would decide every
context. Consistency with the weighting path matters more here than squeezing
out a better statistic -- two subsystems disagreeing about what "good in this
context" means is the expensive failure.

WHAT IS EXCLUDED. Only `profitable`/`unprofitable`/`break_even`. Every
training row carries `neutral` or `pending`; counting those would divide real
results by trades that never had an outcome. As of 2026-08-02 that is all
19,305 rows, so everything here correctly returns empty until paper trading
produces the first resolved decision.
"""
from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from src.core.logging.logger import ProjectLogger
from src.meta_learning.memory.contextual_weight_calculator import (
    ContextualWeightCalculator,
)

#: Outcomes representing a trade that actually resolved.
RESOLVED_OUTCOMES = ("profitable", "unprofitable", "break_even")

#: A (context, model) pair needs at least this many resolved trades before its
#: win rate is treated as evidence. One trade is an anecdote, and
#: ContextualWeightCalculator already uses 2 for the same reason.
MIN_TRADES_PER_CONTEXT_MODEL = 2

#: A fitted neighbour search needs more contexts than neighbours requested,
#: or it returns the query point and calls it similarity.
MIN_CONTEXTS_FOR_SIMILARITY = 5


class ContextPerformanceHistory:
    """Reads `experience_diary` into per-context model performance."""

    def __init__(self, data_manager: Any, logger: logging.Logger | None = None):
        self.data_manager = data_manager
        self.logger = logger or ProjectLogger.get_logger(self.__class__.__name__)

    def performance_by_context(
        self, *, min_trades: int = MIN_TRADES_PER_CONTEXT_MODEL
    ) -> pd.DataFrame:
        """Win rate per (context_fingerprint, agent_id), long format.

        Columns: context_fingerprint, agent_id, win_rate, trades.
        Empty when the diary holds no resolved decisions.
        """
        placeholders = ",".join("?" * len(RESOLVED_OUTCOMES))
        query = f"""
        SELECT
            context_fingerprint,
            agent_id,
            COUNT(*) AS trades,
            AVG(CASE WHEN outcome = 'profitable' THEN 1.0 ELSE 0.0 END) AS win_rate
        FROM experience_diary
        WHERE outcome IN ({placeholders})
          AND context_fingerprint IS NOT NULL
          AND agent_id IS NOT NULL
        GROUP BY context_fingerprint, agent_id
        HAVING trades >= ?
        """
        try:
            frame = self.data_manager.con.execute(
                query, [*RESOLVED_OUTCOMES, int(min_trades)]
            ).fetchdf()
        except (ValueError, TypeError, AttributeError, KeyError) as exc:
            self.logger.error(
                "Could not read per-context model performance: %s", exc,
                exc_info=True,
            )
            return pd.DataFrame(
                columns=["context_fingerprint", "agent_id", "trades", "win_rate"]
            )
        return frame

    def similarity_inputs(
        self,
        current_fingerprint: str,
        *,
        n_neighbors: int = 5,
        min_contexts: int = MIN_CONTEXTS_FOR_SIMILARITY,
    ) -> dict[str, Any] | None:
        """A fitted finder plus the current context, ready for select_models.

        Returns None -- not an empty structure -- whenever the inputs cannot
        support a real neighbour search. The caller must fall back rather than
        act on a search over one point.
        """
        vector = ContextualWeightCalculator.fingerprint_to_vec(current_fingerprint)
        if not vector:
            # A SHA-256 or the literal 'normal'. Not decodable into a state
            # vector, and fabricating one would fabricate neighbours.
            return None

        history = self.performance_by_context()
        if history.empty:
            return None

        features, outcomes = self._build_matrices(history, width=len(vector))
        if len(features) < int(min_contexts):
            self.logger.debug(
                "Contextual model selection skipped: %d comparable context(s), "
                "need %d.", len(features), int(min_contexts),
            )
            return None

        from src.analytics.analyzers.knn_similarity_finder import KnnSimilarityFinder

        finder = KnnSimilarityFinder(
            config={"n_neighbors": min(int(n_neighbors), len(features))}
        )
        finder.fit(features, outcomes)

        return {
            "current_context": pd.Series(vector, index=features.columns),
            "similarity_finder": finder,
            "contexts_considered": int(len(features)),
        }

    @staticmethod
    def _build_matrices(
        history: pd.DataFrame, *, width: int
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Fingerprint vectors and their `target_<model>` win rates.

        Only fingerprints that decode to exactly `width` values are kept.
        fingerprint_to_vec drops any token it cannot parse, so a fingerprint
        carrying junk yields a SHORTER vector -- and comparing it against a
        full-width one silently compares driver 12 with driver 13. The same
        rule already guards knn_context_finder.
        """
        wide = history.pivot_table(
            index="context_fingerprint",
            columns="agent_id",
            values="win_rate",
            aggfunc="mean",
        )

        rows: dict[str, list[float]] = {}
        for fingerprint in wide.index:
            vector = ContextualWeightCalculator.fingerprint_to_vec(str(fingerprint))
            if len(vector) == width:
                rows[str(fingerprint)] = vector

        columns = [f"driver_{position}" for position in range(width)]
        features = pd.DataFrame.from_dict(
            rows, orient="index", columns=columns
        )
        # ContextualModelSelector reads finder.historical_outcomes["target_<model>"]
        # positionally against the neighbour indices, so the outcomes frame must
        # be the SAME rows in the SAME order as the features it was fitted on.
        outcomes = wide.loc[features.index].rename(
            columns=lambda name: f"target_{name}"
        )
        # A model with no result in a context must not read as a zero win rate,
        # which would rank it below a model that merely lost sometimes. Left as
        # NaN; _calculate_model_scores means over neighbours and skips them.
        return features, outcomes
