#!/usr/bin/env python3
"""
KNN Context Finder for Similar Context Search

This module handles KNN-based similarity search for finding similar contexts
when exact fingerprint matching doesn't have enough historical data.
"""

import logging

import pandas as pd

from src.core.logging.logger import ProjectLogger
from src.meta_learning.memory.contextual_weight_calculator import ContextualWeightCalculator


class KnnContextFinder:
    """
    Finds similar contexts using KNN similarity search.

    This class provides methods for finding similar fingerprints and pattern sequences
    when exact matching doesn't have sufficient historical data.
    """

    def __init__(
        self,
        data_manager,
        weight_calculator: ContextualWeightCalculator,
        logger: logging.Logger | None = None
    ):
        """
        Initialize the KNN context finder.

        Args:
            data_manager: DataManager instance for database queries
            weight_calculator: ContextualWeightCalculator instance
            logger: Optional logger instance
        """
        self.data_manager = data_manager
        self.weight_calculator = weight_calculator
        self.logger = logger or ProjectLogger.get_logger(self.__class__.__name__)

    def get_knn_contextual_model_weights(
        self,
        context_fingerprint: str,
        *,
        context_pattern_seq: str | None = None,
        n_neighbors: int = 5,
        window: int = 5000,
        min_neighbors: int = 3,
    ) -> dict[str, float]:
        """
        KNN expansion for contextual weights.

        If we don't have enough history for the exact fingerprint, we search for similar
        fingerprints (based on tri-state vector tokens) and average their contextual weights.

        Args:
            context_fingerprint: Target context fingerprint
            context_pattern_seq: Optional pattern sequence for faster matching
            n_neighbors: Number of neighbors to consider
            window: Time window for historical data
            min_neighbors: Minimum neighbors required for meaningful results

        Returns:
            Dict with aggregated model weights
        """
        # Fast path: exact fingerprint has some history
        exact = self.weight_calculator.get_contextual_model_weights(context_fingerprint)
        if exact:
            return exact

        if context_pattern_seq:
            exact_seq = self.weight_calculator.get_contextual_model_weights_by_pattern_seq(
                context_pattern_seq
            )
            if exact_seq:
                return exact_seq

        try:
            if context_pattern_seq:
                pattern_weights = self._get_knn_weights_for_pattern_sequence(
                    context_pattern_seq,
                    n_neighbors=n_neighbors,
                    window=window,
                    min_neighbors=min_neighbors,
                )
                if pattern_weights:
                    return pattern_weights

            # Load recent rows that have a fingerprint and a resolved outcome
            query = """
            SELECT context_fingerprint
            FROM experience_diary
            WHERE context_fingerprint IS NOT NULL
              AND context_fingerprint != ''
              AND outcome NOT IN (?, ?)
            ORDER BY decision_timestamp DESC
            LIMIT ?
            """
            df = self.data_manager.con.execute(
                query,
                ['pending', 'not_applicable', int(window)],
            ).fetchdf()

            if df.empty:
                return {}

            hist_fps = df["context_fingerprint"].astype(str).dropna().unique().tolist()
            if len(hist_fps) < min_neighbors:
                return {}

            # Build numeric fingerprint vectors for KNN
            target_vec = self._fingerprint_to_vec(context_fingerprint)
            if not target_vec:
                return {}

            hist_vecs = [(fp, self._fingerprint_to_vec(fp)) for fp in hist_fps]
            # Keep only same-length vectors for stability
            hist_vecs = [(fp, v) for fp, v in hist_vecs if len(v) == len(target_vec)]
            if len(hist_vecs) < min_neighbors:
                return {}

            # Use KNN similarity finder
            from src.analytics.analyzers.knn_similarity_finder import KnnSimilarityFinder

            cols = [f"fp_{i}" for i in range(len(target_vec))]
            hist_df = pd.DataFrame([v for _, v in hist_vecs], columns=cols)
            hist_df.index = [fp for fp, _ in hist_vecs]
            target_df = pd.DataFrame([target_vec], columns=cols, index=["target"])

            finder = KnnSimilarityFinder(config={"n_neighbors": int(n_neighbors)})
            res = finder.analyze({"historical_features": hist_df, "target_features": target_df})
            sims = res.get("similarities", {}).get("target", [])
            neighbor_fps = [s.get("id") for s in sims if s.get("id")]

            if len(neighbor_fps) < min_neighbors:
                return {}

            # Aggregate neighbor weights
            agg: dict[str, float] = {}
            for fp in neighbor_fps:
                w = self.weight_calculator.get_contextual_model_weights(str(fp))
                for k, v in w.items():
                    agg[k] = agg.get(k, 0.0) + float(v)

            if not agg:
                return {}

            total = sum(agg.values())
            if total > 0:
                agg = {k: v / total for k, v in agg.items()}

            return agg

        except Exception as e:
            self.logger.error(
                f"Error getting KNN contextual model weights: {e}",
                exc_info=True
            )
            raise RuntimeError(
                f"Failed to get KNN contextual model weights for {context_fingerprint}"
            ) from e

    def _get_knn_weights_for_pattern_sequence(
        self,
        context_pattern_seq: str,
        *,
        n_neighbors: int,
        window: int,
        min_neighbors: int,
    ) -> dict[str, float]:
        """
        Get KNN weights for pattern sequence matching.

        Args:
            context_pattern_seq: Target pattern sequence
            n_neighbors: Number of neighbors to consider
            window: Time window for historical data
            min_neighbors: Minimum neighbors required

        Returns:
            Dict with aggregated model weights
        """
        query = """
        SELECT context_pattern_seq
        FROM experience_diary
        WHERE context_pattern_seq IS NOT NULL
          AND context_pattern_seq != ''
          AND outcome NOT IN (?, ?)
        ORDER BY decision_timestamp DESC
        LIMIT ?
        """
        df = self.data_manager.con.execute(
            query,
            ['pending', 'not_applicable', int(window)],
        ).fetchdf()

        if df.empty:
            return {}

        hist_patterns = df["context_pattern_seq"].astype(str).dropna().unique().tolist()
        if len(hist_patterns) < min_neighbors:
            return {}

        target_vec = ContextualWeightCalculator.pattern_sequence_to_vec(context_pattern_seq)
        if not target_vec:
            return {}

        hist_vecs = [
            (pattern, ContextualWeightCalculator.pattern_sequence_to_vec(pattern))
            for pattern in hist_patterns
        ]
        hist_vecs = [
            (pattern, vec) for pattern, vec in hist_vecs
            if len(vec) == len(target_vec)
        ]

        if len(hist_vecs) < min_neighbors:
            return {}

        from src.analytics.analyzers.knn_similarity_finder import KnnSimilarityFinder

        cols = [f"pattern_{i}" for i in range(len(target_vec))]
        hist_df = pd.DataFrame([v for _, v in hist_vecs], columns=cols)
        hist_df.index = [pattern for pattern, _ in hist_vecs]
        target_df = pd.DataFrame([target_vec], columns=cols, index=["target"])

        finder = KnnSimilarityFinder(config={"n_neighbors": int(n_neighbors)})
        res = finder.analyze({"historical_features": hist_df, "target_features": target_df})
        sims = res.get("similarities", {}).get("target", [])
        neighbor_patterns = [s.get("id") for s in sims if s.get("id")]

        if len(neighbor_patterns) < min_neighbors:
            return {}

        agg: dict[str, float] = {}
        for pattern in neighbor_patterns:
            weights = self.weight_calculator.get_contextual_model_weights_by_pattern_seq(str(pattern))
            for model_name, value in weights.items():
                agg[model_name] = agg.get(model_name, 0.0) + float(value)

        if not agg:
            return {}

        total = sum(agg.values())
        return {k: v / total for k, v in agg.items()} if total > 0 else agg

    @staticmethod
    def _fingerprint_to_vec(fingerprint: str) -> list[float]:
        """
        Convert fingerprint string to numeric vector.

        Args:
            fingerprint: Fingerprint string

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
