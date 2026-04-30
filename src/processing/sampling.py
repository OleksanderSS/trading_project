
# src/data/sampling.py

import pandas as pd
from typing import Optional

from src.core.logging.logger import ProjectLogger

logger = ProjectLogger.get_logger("DatasetSampler")

class DatasetSampler:
    """
    Provides methods for sampling datasets, such as balancing classes.
    """

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        logger.info(f"DatasetSampler initialized with random_state={self.random_state}")

    def create_balanced_dataset(
        self, 
        df: pd.DataFrame, 
        target_col: str, 
        max_samples_per_class: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Creates a balanced dataset by downsampling the majority class(es).
        This is useful for training classification models on imbalanced data.

        Args:
            df: The input DataFrame.
            target_col: The name of the column to balance (e.g., a label or a boolean flag).
            max_samples_per_class: The maximum number of samples for any class. 
                                     If None, it defaults to the size of the smallest class.

        Returns:
            A new DataFrame with balanced classes.
        """
        if df.empty or target_col not in df.columns:
            logger.warning(f"DataFrame is empty or target column '{target_col}' not found. Returning original DataFrame.")
            return df

        # Count class distribution
        class_counts = df[target_col].value_counts()
        min_class_size = class_counts.min()
        
        if len(class_counts) < 2:
            logger.warning("The target column has fewer than 2 classes. No balancing needed.")
            return df

        # Determine the sample size
        sample_size = min_class_size
        if max_samples_per_class:
            sample_size = min(sample_size, max_samples_per_class)

        logger.info(f"Balancing dataset based on column '{target_col}'. Original class counts:\n{class_counts}")
        logger.info(f"Target sample size per class: {sample_size}")

        # Sample from each class and concatenate
        balanced_dfs = []
        for class_label in class_counts.index:
            class_df = df[df[target_col] == class_label]
            sampled_df = class_df.sample(n=sample_size, random_state=self.random_state)
            balanced_dfs.append(sampled_df)

        balanced_df = pd.concat(balanced_dfs, ignore_index=True)
        
        # Shuffle the final dataset
        balanced_df = balanced_df.sample(frac=1, random_state=self.random_state).reset_index(drop=True)

        logger.info(f"Balanced dataset created with {len(balanced_df)} total samples.")
        return balanced_df

