"""
Generic preprocessing for any dataset before privacy pipeline.
"""

import numpy as np
import pandas as pd


def preprocess_generic(df, config):
    """
    Generic preprocessing pipeline.

    Steps:
    1. Drop identifier attributes
    2. Handle missing values (auto-strategy per column type)
    3. Remove duplicate rows
    4. Clip outliers (IQR method) for numerical QI attributes

    Args:
        df (pd.DataFrame): Raw input dataframe
        config (dict): Dataset configuration

    Returns:
        pd.DataFrame: Cleaned dataframe ready for privacy pipeline
    """
    df_clean = df.copy()

    qi_attrs = config.get('qi_attributes', [])
    sens_attr = config.get('sensitive_attribute', '')
    id_attrs = config.get('identifier_attributes', [])
    non_sens = config.get('non_sensitive_attributes', [])

    all_target = qi_attrs + ([sens_attr] if sens_attr else []) + non_sens

    # Step 1: Drop identifier attributes
    if id_attrs:
        existing_ids = [c for c in id_attrs if c in df_clean.columns]
        if existing_ids:
            df_clean = df_clean.drop(columns=existing_ids)
            print(f'  Dropped identifiers: {existing_ids}')

    # Step 2: Handle missing values
    for col in df_clean.columns:
        n_null = df_clean[col].isnull().sum()
        if n_null > 0:
            if df_clean[col].dtype in ['float64', 'float32', 'int64', 'int32']:
                fill_val = df_clean[col].median()
                df_clean[col] = df_clean[col].fillna(fill_val)
            else:
                fill_val = df_clean[col].mode().iloc[0] if not df_clean[col].mode().empty else 'Unknown'
                df_clean[col] = df_clean[col].fillna(fill_val)
            print(f'  Filled {n_null} missing values in "{col}" with {fill_val}')

    # Step 3: Remove duplicates
    n_before = len(df_clean)
    df_clean = df_clean.drop_duplicates().reset_index(drop=True)
    n_dupes = n_before - len(df_clean)
    if n_dupes > 0:
        print(f'  Removed {n_dupes} duplicate rows')

    # Step 4: Clip outliers for numerical QI attributes
    for attr in qi_attrs:
        if attr not in df_clean.columns:
            continue
        if df_clean[attr].dtype not in ['float64', 'float32', 'int64', 'int32']:
            continue

        Q1 = df_clean[attr].quantile(0.25)
        Q3 = df_clean[attr].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR

        n_before_clip = len(df_clean)
        df_clean[attr] = df_clean[attr].clip(lower=lower, upper=upper)
        n_clipped = (df_clean[attr] < lower).sum() + (df_clean[attr] > upper).sum()

    print(f'\nPreprocessing complete: {len(df_clean)} records, {len(df_clean.columns)} columns')
    return df_clean
