DATASET_CONFIG = {
    'file_path': 'data/raw/diabetes__health_indicators.csv',
    'identifier_attributes': [],
    'qi_attributes': ['Age', 'Sex', 'Education', 'Income', 'BMI', 'GenHlth'],
    'sensitive_attribute': ['Diabetes_012'],
    'non_sensitive_attributes': [],
}

PRIVACY_CONFIG = {
    'k_anonymity': 5,
    'epsilon': 1.0,
    'max_level': 3,
    'max_tree_depth': 4,
}

HIERARCHY_CONFIG = {
    'n_bins_level1': 4,
    'n_bins_level2': 2,
    'ordinal_group_size': 4,
    'top_k_frequent': 10,
}

CUSTOM_HIERARCHY = {}
