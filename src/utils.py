import pandas as pd
import numpy as np


def ensure_list(value):
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return value
    if value is None:
        return []
    return [value]


def detect_column_type(series):
    if len(series) == 0 or series.dropna().empty:
        return 'categorical_nominal'

    n_unique = series.nunique()
    n_total = len(series.dropna())
    dtype = series.dtype

    if n_unique == 2:
        return 'categorical_binary'

    if pd.api.types.is_datetime64_any_dtype(series):
        return 'datetime'

    if dtype == 'object':
        try:
            pd.to_datetime(series.dropna().head(100), errors='raise')
            return 'datetime'
        except:
            pass

    if dtype == 'object' or dtype.name == 'category':
        try:
            numeric_series = pd.to_numeric(series.dropna(), errors='coerce')
            non_null_ratio = numeric_series.notna().sum() / len(numeric_series)

            if non_null_ratio > 0.8:
                if numeric_series.apply(lambda x: x == int(x) if pd.notna(x) else True).all():
                    if n_unique <= 20:
                        return 'numerical_ordinal'
                    else:
                        return 'numerical_continuous'
                else:
                    return 'numerical_continuous'
        except:
            pass

        avg_len = series.astype(str).str.len().mean()
        if avg_len > 50:
            return 'text'

        return 'categorical_nominal'

    elif pd.api.types.is_bool_dtype(series):
        return 'categorical_binary'

    elif dtype in ['float32', 'float64']:
        if series.dropna().apply(lambda x: x == int(x)).all():
            if n_unique <= 20:
                return 'numerical_ordinal'
            else:
                return 'numerical_continuous'
        else:
            return 'numerical_continuous'

    elif dtype in ['int8', 'int16', 'int32', 'int64', 'uint8', 'uint16', 'uint32', 'uint64']:
        if n_unique <= 20 or n_unique < n_total * 0.1:
            return 'numerical_ordinal'
        else:
            return 'numerical_continuous'

    else:
        return 'categorical_nominal'


def validate_config(df, config):
    errors = []
    warnings = []

    all_cols = df.columns.tolist()

    qi_attrs = config.get('qi_attributes', [])
    sens_attrs = ensure_list(config.get('sensitive_attribute', []))
    id_attrs = config.get('identifier_attributes', [])
    non_sens = config.get('non_sensitive_attributes', [])

    all_configured = qi_attrs + sens_attrs + non_sens
    # identifiers are dropped by preprocessing, skip column existence check

    for attr in all_configured:
        if attr and attr not in all_cols:
            errors.append(f"Attribute '{attr}' not found in dataset columns")

    if not sens_attrs:
        errors.append("At least one sensitive attribute must be specified")

    sens_set = set(sens_attrs)
    qi_set = set(qi_attrs)

    overlap_sens_qi = sens_set & qi_set
    if overlap_sens_qi:
        errors.append(f"Sensitive attribute(s) cannot also be QI attributes: {overlap_sens_qi}")

    id_set = set(id_attrs)

    overlap = qi_set & id_set
    if overlap:
        errors.append(f"Overlap between QI and Identifier: {overlap}")

    overlap_ns = qi_set & set(non_sens)
    if overlap_ns:
        errors.append(f"Overlap between QI and Non-sensitive: {overlap_ns}")

    if sens_set & id_set:
        errors.append(f"Sensitive attribute(s) cannot be Identifiers: {sens_set & id_set}")

    if not qi_attrs:
        errors.append("At least one QI attribute must be specified")

    if len(df) < 100:
        warnings.append(f"Dataset has only {len(df)} records. Minimum recommended: 100")

    return errors, warnings
