"""
Configuration & Constants for ACDP Tree Privacy Pipeline.
"""

import os

# Dataset paths
RAW_DATA_PATH = 'data/raw/diabetes__health_indicators.csv'
OUTPUT_DIR = 'results/anonymized_data'

# QI Attributes
QI_ATTRIBUTES = ['Age', 'Sex', 'Education', 'Income', 'BMI', 'GenHlth']
SENSITIVE_ATTRIBUTE = 'Diabetes_012'

# Privacy parameters
K_ANONYMITY = 5
EPSILON = 1.0
MAX_LEVEL = 3

# BMI outlier clipping feature list
OUTLIER_FEATURES = ['BMI']
