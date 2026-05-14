"""
Utility functions for data loading and preprocessing.
"""

import pandas as pd

from src.config import RAW_DATA_PATH, OUTLIER_FEATURES


def load_and_preprocess_data(filepath=None):
    """
    Load dataset and apply preprocessing (BMI outlier clipping).

    Args:
        filepath: Path to CSV file. Defaults to RAW_DATA_PATH.

    Returns:
        pd.DataFrame: Cleaned dataframe ready for privacy pipeline.
    """
    if filepath is None:
        filepath = RAW_DATA_PATH

    df = pd.read_csv(filepath)

    # Outlier clipping (same as notebook)
    df_clean = df.copy()
    for col in OUTLIER_FEATURES:
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1

        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR

        df_clean[col] = df_clean[col].clip(lower=lower, upper=upper)
        print(f' {col}: clipping ke [{lower:.2f}, {upper:.2f}]')

    print(f'\nShape setelah capping: {df_clean.shape}')

    return df_clean
