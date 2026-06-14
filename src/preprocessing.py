import numpy as np
import pandas as pd
from src.utils import ensure_list


def preprocess_generic(df, config):
    df_clean = df.copy()

    qi_attrs = config.get('qi_attributes', [])
    sens_attrs = ensure_list(config.get('sensitive_attribute', []))
    id_attrs = config.get('identifier_attributes', [])
    non_sens = config.get('non_sensitive_attributes', [])

    all_target = qi_attrs + sens_attrs + non_sens

    if id_attrs:
        existing_ids = [c for c in id_attrs if c in df_clean.columns]
        if existing_ids:
            df_clean = df_clean.drop(columns=existing_ids)
            print(f'  Dropped identifiers: {existing_ids}')

    for col in df_clean.columns:
        n_null = df_clean[col].isnull().sum()
        if n_null > 0:
            if n_null == len(df_clean):
                print(f'  Warning: Column "{col}" is all NaN, filling with placeholder')
                df_clean[col] = 0
                continue

            if df_clean[col].dtype in ['float64', 'float32', 'int64', 'int32']:
                fill_val = df_clean[col].median()
                if pd.isna(fill_val):
                    fill_val = 0
                df_clean[col] = df_clean[col].fillna(fill_val)
            else:
                fill_val = df_clean[col].mode().iloc[0] if not df_clean[col].mode().empty else 'Unknown'
                df_clean[col] = df_clean[col].fillna(fill_val)
            print(f'  Filled {n_null} missing values in "{col}" with {fill_val}')

    n_before = len(df_clean)
    df_clean = df_clean.drop_duplicates().reset_index(drop=True)
    n_dupes = n_before - len(df_clean)
    if n_dupes > 0:
        print(f'  Removed {n_dupes} duplicate rows')

    for attr in qi_attrs:
        if attr not in df_clean.columns:
            continue
        if df_clean[attr].dtype not in ['float64', 'float32', 'int64', 'int32']:
            continue

        try:
            Q1 = df_clean[attr].quantile(0.25)
            Q3 = df_clean[attr].quantile(0.75)
            IQR = Q3 - Q1

            if IQR <= 0 or pd.isna(IQR):
                continue

            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR

            df_clean[attr] = df_clean[attr].clip(lower=lower, upper=upper)
        except Exception as e:
            print(f'  Warning: Could not clip outliers for "{attr}": {e}')
            continue

    for col in df_clean.select_dtypes(include=[np.number]).columns:
        if np.isinf(df_clean[col]).any():
            print(f'  Warning: Infinite values found in "{col}", replacing with max/min')
            finite_vals = df_clean[col][np.isfinite(df_clean[col])]
            if len(finite_vals) > 0:
                max_val = finite_vals.max()
                min_val = finite_vals.min()
                df_clean[col] = df_clean[col].clip(lower=min_val, upper=max_val)
            else:
                df_clean[col] = 0

    if all_target:
        existing = [c for c in all_target if c in df_clean.columns]
        missing = [c for c in all_target if c not in df_clean.columns]
        if missing:
            print(f'  Warning: Target columns not found: {missing}')
        df_clean = df_clean[existing]

    return df_clean
