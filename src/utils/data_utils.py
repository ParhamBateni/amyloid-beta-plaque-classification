"""
Data loading and processing utilities.
"""

import pandas as pd
from typing import Tuple


def load_data_df(
    data_df_path: str,
    labeled_sample_size: int,
    unlabeled_sample_size: int,
    train_mode: str,
    random_seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load and sample data from CSV file.

    Args:
        data_df_path: Path to the CSV file
        labeled_sample_size: Number of labeled samples to use
        unlabeled_sample_size: Number of unlabeled samples to use
        train_mode: Training mode ('supervised' or other)
        random_seed: Random seed for reproducibility

    Returns:
        Tuple of (labeled_data_df, unlabeled_data_df)
    """
    data_df = pd.read_csv(data_df_path)
    labeled_data_df = data_df[data_df["Label"].notna()]
    labeled_data_df = labeled_data_df.sample(
        n=min(labeled_sample_size, len(labeled_data_df)),
        random_state=random_seed,
        replace=False,
    )
    if train_mode != "supervised":
        unlabeled_data_df = data_df[data_df["Label"].isna()]
        unlabeled_data_df = unlabeled_data_df.sample(
            n=min(unlabeled_sample_size, len(unlabeled_data_df)),
            random_state=random_seed,
            replace=False,
        )
        return labeled_data_df, unlabeled_data_df
    else:
        return labeled_data_df, pd.DataFrame()
