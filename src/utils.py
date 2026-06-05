"""
Utility functions: data type detection, config validation, preprocessing.
"""

import pandas as pd
import numpy as np


def detect_column_type(series):
    """
    Auto-detect column type for generalization hierarchy.

    Returns:
        str: 'numerical_continuous', 'numerical_ordinal',
             'categorical_binary', 'categorical_nominal'
    """
    n_unique = series.nunique()
    dtype = series.dtype

    if n_unique == 2:
        return 'categorical_binary'
    elif dtype == 'object' or dtype.name == 'category':
        return 'categorical_nominal'
    elif dtype in ['float32', 'float64']:
        return 'numerical_continuous'
    elif dtype in ['int8', 'int16', 'int32', 'int64', 'uint8', 'uint16', 'uint32', 'uint64']:
        if n_unique <= 20:
            return 'numerical_ordinal'
        else:
            return 'numerical_continuous'
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
