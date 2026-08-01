#!/usr/bin/env python3
"""
Contextual Weight Calculator for Model Performance Analysis

This module handles the calculation of contextual model weights based on historical performance
in specific contexts. It provides methods for exact fingerprint matching, pattern sequence matching,
and KNN-based similarity search for contexts with limited historical data.
"""

import logging
from dataclasses import dataclass

import pandas as pd

from src.core.logging.logger import ProjectLogger


@dataclass
class WeightResult:
    """Result of weight calculation with metadata."""
    weights: dict[str, float]
    method: str  # 'exact', 'pattern_seq', 'knn'
    confidence: float  # 0.0 to 1.0 based on data availability


class ContextualWeightCalculator:
    """
    Calculates model weights based on contextual performance.

    This class encapsulates the logic for determining which models perform best
    in specific market contexts, using exact matching, pattern sequences, and
    KNN-based similarity search.
    """

    def __init__(self, data_manager, logger: logging.Logger | None = None):
        """
        Initialize the weight calculator.

        Args:
            data_manager: DataManager instance for database queries
            logger: Optional logger instance
        """
        self.data_manager = data_manager
        self.logger = logger or ProjectLogger.get_logger(self.__class__.__name__)

    def get_contextual_model_weights(
        self,
        context_fingerprint: str
    ) -> dict[str, float]:
        """
        Returns model weights based on historical performance in the given context.

        Args:
            context_fingerprint: Context fingerprint string

        Returns:
            Dict with model weights (model_name -> weight)
        """
        query = """
        SELECT
            agent_id,
            COUNT(*) as total_decisions,
            AVG(CASE WHEN outcome = 'profitable' THEN 1.0 ELSE 0.0 END)
                as performance_score,
            COALESCE(AVG(profit_loss), 0.0) as avg_pnl
        FROM experience_diary
        WHERE context_fingerprint = ?
          -- Realized outcomes only. Training rows carry no realized result
          -- (outcome is always 'neutral') and used to contribute
          -- model_prediction to this average -- which log_training_event
          -- fills with float(metrics), a TRAINING METRIC, not a prediction.
          -- Averaging an unbounded metric together with a 0/1 win rate is not
          -- a quantity, and on the live table it produced a performance_score
          -- of -13,820 for `linear` (its metrics reach -1,420,512), which
          -- became a NEGATIVE ensemble weight of -0.25: the model's forecast
          -- was subtracted rather than ignored, while the weights still
          -- summed to 1.0 and looked healthy.
          AND outcome IN ('profitable', 'unprofitable', 'break_even')
        GROUP BY agent_id
        HAVING total_decisions >= 2
        ORDER BY performance_score DESC, avg_pnl DESC
        """

        try:
            result_df = self.data_manager.con.execute(
                query, [context_fingerprint]
            ).fetchdf()

            if result_df.empty:
                if self.logger.isEnabledFor(logging.DEBUG):
                    self.logger.debug(
                        f"No historical data for context {context_fingerprint}, "
                        "using equal weights"
                    )
                return {}

            return self._calculate_weights_from_dataframe(result_df)

        except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError) as e:
            self.logger.error(
                f"Error getting contextual model weights: {e}",
                exc_info=True
            )
            raise RuntimeError(
                f"Failed to get contextual model weights for {context_fingerprint}"
            ) from e

    def get_contextual_model_weights_by_pattern_seq(
        self,
        context_pattern_seq: str
    ) -> dict[str, float]:
        """
        Return model weights for an exact rolling context-pattern sequence.

        Args:
            context_pattern_seq: Pattern sequence string

        Returns:
            Dict with model weights
        """
        if not context_pattern_seq:
            return {}

        query = """
        SELECT
            agent_id,
            COUNT(*) as total_decisions,
            AVG(CASE WHEN outcome = 'profitable' THEN 1.0 ELSE 0.0 END)
                as performance_score,
            COALESCE(AVG(profit_loss), 0.0) as avg_pnl
        FROM experience_diary
        WHERE context_pattern_seq = ?
          -- Realized outcomes only; see the note on the fingerprint query.
          AND outcome IN ('profitable', 'unprofitable', 'break_even')
        GROUP BY agent_id
        HAVING total_decisions >= 2
        ORDER BY performance_score DESC, avg_pnl DESC
        """

        try:
            result_df = self.data_manager.con.execute(
                query, [context_pattern_seq]
            ).fetchdf()
            return self._calculate_weights_from_dataframe(result_df)
        except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError) as e:
            self.logger.error(
                f"Error getting pattern-sequence model weights: {e}",
                exc_info=True
            )
            raise RuntimeError(
                f"Failed to get contextual model weights for pattern sequence "
                f"{context_pattern_seq}"
            ) from e

    def _calculate_weights_from_dataframe(
        self,
        result_df: pd.DataFrame
    ) -> dict[str, float]:
        """
        Calculate normalized weights from query results.

        Args:
            result_df: DataFrame with performance data

        Returns:
            Normalized weights dictionary
        """
        if result_df.empty:
            return {}

        weights = {}
        total_score = 0.0

        for _, row in result_df.iterrows():
            agent_id = row['agent_id']
            performance_score = row['performance_score']
            avg_pnl = row['avg_pnl']

            # Combined metric: win_rate * (1 + normalized_pnl)
            # Normalize avg_pnl to range [0, 1]
            normalized_pnl = max(0, min(1, (avg_pnl + 1) / 2))
            # Floor at zero. A negative weight does not mean "ignore this
            # model", it means "subtract its forecast" -- and normalising by
            # the SUM hides that, because the weights still add to 1.0. The
            # inputs are bounded now, so this should never bind; it is here so
            # that if some future column arrives negative, the ensemble
            # degrades to ignoring a model rather than inverting it.
            score = max(0.0, float(performance_score)) * (1 + normalized_pnl)

            weights[agent_id] = score
            total_score += score

        # Normalize weights to sum to 1.0
        if total_score > 0:
            weights = {k: v / total_score for k, v in weights.items()}
        else:
            # Every model scored zero in this context: that is "no evidence",
            # not "all models are worthless". Returning the raw zeros would
            # hand the ensemble a set of zero weights; {} is the documented
            # signal for equal weighting.
            return {}

        if self.logger.isEnabledFor(logging.DEBUG):
            self.logger.debug(f"Calculated weights: {weights}")

        return weights

    @staticmethod
    def fingerprint_to_vec(fingerprint: str) -> list[float]:
        """
        Convert fingerprint string to numeric vector.

        Args:
            fingerprint: Fingerprint string with '|' separators

        Returns:
            List of float values
        """
        vec: list[float] = []
        for token in str(fingerprint).split("|"):
            if token == "":
                continue
            try:
                vec.append(float(token))
            except (TypeError, ValueError):
                continue
        return vec

    @classmethod
    def pattern_sequence_to_vec(cls, context_pattern_seq: str) -> list[float]:
        """
        Convert pattern sequence to flattened vector.

        Args:
            context_pattern_seq: Pattern sequence with '>>' separators

        Returns:
            Flattened vector
        """
        parts = [part for part in str(context_pattern_seq).split(">>") if part]
        width = 0
        parsed_parts: list[list[float]] = []

        for part in parts:
            if part == "START":
                parsed_parts.append([])
                continue
            vec = cls.fingerprint_to_vec(part)
            parsed_parts.append(vec)
            if vec and width == 0:
                width = len(vec)

        if width == 0:
            return []

        flattened: list[float] = []
        for vec in parsed_parts:
            if not vec:
                flattened.extend([0.0] * width)
            elif len(vec) < width:
                flattened.extend(vec + [0.0] * (width - len(vec)))
            else:
                flattened.extend(vec[:width])

        return flattened
