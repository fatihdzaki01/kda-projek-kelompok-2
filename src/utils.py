"""
Utility functions: data type detection, config validation, preprocessing.
"""

import pandas as pd
import numpy as np


def detect_column_type(series):
    """
    Auto-detect column type for generalization hierarchy.
    
    Robust detection for:
    - Dates/timestamps
    - Mixed types
    - Numeric strings
    - Boolean values
    - Special characters

    Returns:
        str: 'numerical_continuous', 'numerical_ordinal',
             'categorical_binary', 'categorical_nominal',
             'datetime', 'text'
    """
    # Handle empty series
    if len(series) == 0 or series.dropna().empty:
        return 'categorical_nominal'
    
    n_unique = series.nunique()
    n_total = len(series.dropna())
    dtype = series.dtype
    
    # Binary (2 unique values)
    if n_unique == 2:
        return 'categorical_binary'
    
    # Check if datetime
    if pd.api.types.is_datetime64_any_dtype(series):
        return 'datetime'
    
    # Try to parse as datetime if string
    if dtype == 'object':
        try:
            pd.to_datetime(series.dropna().head(100), errors='raise')
            return 'datetime'
        except:
            pass
    
    # Object/string types
    if dtype == 'object' or dtype.name == 'category':
        # Try to convert to numeric
        try:
            numeric_series = pd.to_numeric(series.dropna(), errors='coerce')
            non_null_ratio = numeric_series.notna().sum() / len(numeric_series)
            
            # If >80% can be converted to numeric, treat as numeric
            if non_null_ratio > 0.8:
                if numeric_series.apply(lambda x: x == int(x) if pd.notna(x) else True).all():
                    # All integers
                    if n_unique <= 20:
                        return 'numerical_ordinal'
                    else:
                        return 'numerical_continuous'
                else:
                    # Has floats
                    return 'numerical_continuous'
        except:
            pass
        
        # Check if text (long strings)
        avg_len = series.astype(str).str.len().mean()
        if avg_len > 50:  # Average length > 50 chars = likely text/description
            return 'text'
        
        # Default: categorical nominal
        return 'categorical_nominal'
    
    # Boolean
    elif pd.api.types.is_bool_dtype(series):
        return 'categorical_binary'
    
    # Float types
    elif dtype in ['float32', 'float64']:
        # Check if actually integer (e.g., 1.0, 2.0, 3.0)
        if series.dropna().apply(lambda x: x == int(x)).all():
            if n_unique <= 20:
                return 'numerical_ordinal'
            else:
                return 'numerical_continuous'
        else:
            return 'numerical_continuous'
    
    # Integer types
    elif dtype in ['int8', 'int16', 'int32', 'int64', 'uint8', 'uint16', 'uint32', 'uint64']:
        # Ordinal: limited unique values (e.g., ratings 1-5, age groups)
        if n_unique <= 20 or n_unique < n_total * 0.1:
            return 'numerical_ordinal'
        else:
            return 'numerical_continuous'
    
    # Default fallback
    else:
        return 'categorical_nominal'


def validate_config(df, config):
    """
    Validate dataset configuration before pipeline runs.

    Returns:
        tuple: (errors: list, warnings: list)
    """
    errors = []
    warnings = []

    all_cols = df.columns.tolist()

    qi_attrs = config.get('qi_attributes', [])
    sens_attr = config.get('sensitive_attribute', '')
    id_attrs = config.get('identifier_attributes', [])
    non_sens = config.get('non_sensitive_attributes', [])

    all_configured = qi_attrs + [sens_attr] + id_attrs + non_sens
    if sens_attr:
        all_configured = qi_attrs + [sens_attr] + id_attrs + non_sens
    else:
        all_configured = qi_attrs + id_attrs + non_sens

    for attr in all_configured:
        if attr and attr not in all_cols:
            errors.append(f"Attribute '{attr}' not found in dataset columns: {all_cols[:10]}...")

    if not sens_attr:
        errors.append("Sensitive attribute must be specified")
    elif sens_attr in qi_attrs:
        errors.append(f"Sensitive attribute '{sens_attr}' cannot also be a QI attribute")

    qi_set = set(qi_attrs)
    id_set = set(id_attrs)

    overlap = qi_set & id_set
    if overlap:
        errors.append(f"Overlap between QI and Identifier: {overlap}")

    overlap_ns = qi_set & set(non_sens)
    if overlap_ns:
        errors.append(f"Overlap between QI and Non-sensitive: {overlap_ns}")

    if sens_attr in id_set:
        errors.append(f"Sensitive attribute '{sens_attr}' cannot be an Identifier")

    if not qi_attrs:
        errors.append("At least one QI attribute must be specified")

    if len(df) < 100:
        warnings.append(f"Dataset has only {len(df)} records. Minimum recommended: 100")

    return errors, warnings
