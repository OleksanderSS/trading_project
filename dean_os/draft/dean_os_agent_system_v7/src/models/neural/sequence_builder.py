"""
Sequence Builder for Neural Network Models

This module provides utilities for building proper 3D sequences from 2D data
for neural network models (LSTM, GRU, CNN, Transformer).

The sequence builder enforces explicit sequence construction instead of
automatic reshaping, ensuring users are aware of the sequence requirements.
"""

import warnings
from typing import Literal

import numpy as np

from src.core.logging.logger import ProjectLogger

logger = ProjectLogger.get_logger("SequenceBuilder")


class SequenceBuilder:
    """
    Builder for creating 3D sequences from 2D data for neural networks.
    
    This class provides explicit methods for building sequences with different
    strategies, making the sequence construction process transparent and intentional.
    
    Supported strategies:
    - 'sliding_window': Create overlapping sequences using a sliding window
    - 'reshape': Simple reshape (adds dimension, creates fake sequences)
    - 'timesteps': Group data into fixed-length timesteps
    """

    def __init__(self, strategy: Literal['sliding_window', 'reshape', 'timesteps'] = 'sliding_window'):
        """
        Initialize the sequence builder.
        
        Args:
            strategy: Strategy for building sequences
                - 'sliding_window': Create overlapping sequences (default)
                - 'reshape': Simple reshape (creates fake sequences, use with caution)
                - 'timesteps': Group data into fixed-length timesteps
        """
        self.strategy = strategy
        self.logger = logger

    def build_sequences(
        self,
        X: np.ndarray,
        window_size: int = 10,
        step_size: int = 1
    ) -> np.ndarray:
        """
        Build 3D sequences from 2D data using the configured strategy.
        
        Args:
            X: 2D array of shape (n_samples, n_features)
            window_size: Size of the sequence window (for sliding_window/timesteps)
            step_size: Step size for sliding window (for sliding_window)
            
        Returns:
            3D array of shape (n_sequences, window_size, n_features)
            
        Raises:
            ValueError: If input is not 2D or strategy is invalid
        """
        if len(X.shape) != 2:
            raise ValueError(f"Expected 2D input, got shape {X.shape}")

        if self.strategy == 'sliding_window':
            return self._build_sliding_window(X, window_size, step_size)
        elif self.strategy == 'reshape':
            warnings.warn(
                "Using 'reshape' strategy creates fake sequences. "
                "Consider using 'sliding_window' or 'timesteps' for proper sequence construction.",
                UserWarning,
                stacklevel=2
            )
            return self._build_reshape(X)
        elif self.strategy == 'timesteps':
            return self._build_timesteps(X, window_size)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

    def _build_sliding_window(
        self,
        X: np.ndarray,
        window_size: int,
        step_size: int
    ) -> np.ndarray:
        """
        Build sequences using sliding window approach.
        
        Args:
            X: 2D array of shape (n_samples, n_features)
            window_size: Size of the sliding window
            step_size: Step size for the window
            
        Returns:
            3D array of shape (n_sequences, window_size, n_features)
        """
        n_samples, n_features = X.shape

        if window_size > n_samples:
            raise ValueError(
                f"Window size ({window_size}) cannot be larger than "
                f"number of samples ({n_samples})"
            )

        sequences = []
        for i in range(0, n_samples - window_size + 1, step_size):
            sequences.append(X[i:i + window_size])

        result = np.array(sequences)
        self.logger.info(
            f"Built {len(sequences)} sequences using sliding window "
            f"(window_size={window_size}, step_size={step_size})"
        )
        return result

    def _build_reshape(self, X: np.ndarray) -> np.ndarray:
        """
        Simple reshape that adds a dimension (creates fake sequences).
        
        WARNING: This creates fake sequences by treating each feature as a timestep.
        This is generally not recommended for time series data.
        
        Args:
            X: 2D array of shape (n_samples, n_features)
            
        Returns:
            3D array of shape (n_samples, n_features, 1)
        """
        result = np.reshape(X, (X.shape[0], X.shape[1], 1))
        self.logger.warning(
            f"Used reshape strategy to create fake sequences: {X.shape} -> {result.shape}"
        )
        return result

    def _build_timesteps(self, X: np.ndarray, window_size: int) -> np.ndarray:
        """
        Group data into fixed-length timesteps (non-overlapping).
        
        Args:
            X: 2D array of shape (n_samples, n_features)
            window_size: Size of each timestep group
            
        Returns:
            3D array of shape (n_sequences, window_size, n_features)
        """
        n_samples, n_features = X.shape

        if n_samples % window_size != 0:
            warnings.warn(
                f"Number of samples ({n_samples}) is not divisible by window_size ({window_size}). "
                f"Truncating {n_samples % window_size} samples.",
                UserWarning,
                stacklevel=2
            )
            n_samples = n_samples - (n_samples % window_size)
            X = X[:n_samples]

        n_sequences = n_samples // window_size
        result = X.reshape(n_sequences, window_size, n_features)

        self.logger.info(
            f"Built {n_sequences} sequences using timesteps (window_size={window_size})"
        )
        return result


def build_sequences(
    X: np.ndarray,
    strategy: Literal['sliding_window', 'reshape', 'timesteps'] = 'sliding_window',
    window_size: int = 10,
    step_size: int = 1
) -> np.ndarray:
    """
    Convenience function to build sequences from 2D data.
    
    Args:
        X: 2D array of shape (n_samples, n_features)
        strategy: Strategy for building sequences
        window_size: Size of the sequence window
        step_size: Step size for sliding window
        
    Returns:
        3D array of shape (n_sequences, window_size, n_features)
        
    Example:
        >>> X = np.random.rand(100, 5)  # 100 samples, 5 features
        >>> sequences = build_sequences(X, strategy='sliding_window', window_size=10)
        >>> print(sequences.shape)  # (91, 10, 5)
    """
    builder = SequenceBuilder(strategy=strategy)
    return builder.build_sequences(X, window_size=window_size, step_size=step_size)
