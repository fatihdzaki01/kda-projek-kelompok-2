"""
ACDP Tree Privacy Pipeline — CLI entry point.

Usage:
    python main.py                              # Uses config from src/config.py
    python main.py --file data.csv --qi Age Sex --sens Diabetes_012
    python main.py --config custom_config.py
"""

import os
import sys
import argparse
import warnings
warnings.filterwarnings('ignore')

from src.pipeline import run_pipeline
from src.config import DATASET_CONFIG, PRIVACY_CONFIG, HIERARCHY_CONFIG, CUSTOM_HIERARCHY


def main():
    parser = argparse.ArgumentParser(
        description='ACDP Tree Privacy-Preserving Data Anonymization Pipeline'
    )
    parser.add_argument('--config', type=str, default=None,
                        help='Path to custom config file')
    parser.add_argument('--output', type=str, default=None,
                        help='Output directory')
    parser.add_argument('--file', type=str, default=None,
                        help='Dataset CSV file path')
    parser.add_argument('--qi', type=str, nargs='+', default=None,
                        help='Quasi-identifier attribute names')
    parser.add_argument('--sens', type=str, nargs='+', default=None,
                        help='Sensitive attribute name(s)')
    parser.add_argument('--id', dest='identifier', type=str, nargs='+', default=None,
                        help='Identifier attribute names (will be dropped)')
    parser.add_argument('--non-sens', dest='non_sensitive', type=str, nargs='+', default=None,
                        help='Non-sensitive attribute names')
    parser.add_argument('--k', type=int, default=None,
                        help='K-anonymity parameter')
    parser.add_argument('--epsilon', type=float, default=None,
                        help='Differential privacy epsilon')
    parser.add_argument('--max-level', type=int, default=None,
                        help='Maximum generalization level')

    args = parser.parse_args()

    config = dict(DATASET_CONFIG)
    privacy_config = dict(PRIVACY_CONFIG)

    if args.file:
        config['file_path'] = args.file
    if args.qi:
        config['qi_attributes'] = args.qi
    if args.sens:
        config['sensitive_attribute'] = args.sens
    if args.identifier is not None:
        config['identifier_attributes'] = args.identifier
    if args.non_sensitive is not None:
        config['non_sensitive_attributes'] = args.non_sensitive
    if args.k is not None:
        privacy_config['k_anonymity'] = args.k
    if args.epsilon is not None:
        privacy_config['epsilon'] = args.epsilon
    if args.max_level is not None:
        privacy_config['max_level'] = args.max_level

    if args.config:
        import importlib.util
        spec = importlib.util.spec_from_file_location('custom_config', args.config)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        config = getattr(mod, 'DATASET_CONFIG', config)
        privacy_config = getattr(mod, 'PRIVACY_CONFIG', privacy_config)
        hierarchy_config = getattr(mod, 'HIERARCHY_CONFIG', HIERARCHY_CONFIG)
        custom_hierarchy = getattr(mod, 'CUSTOM_HIERARCHY', CUSTOM_HIERARCHY)
    else:
        hierarchy_config = HIERARCHY_CONFIG
        custom_hierarchy = CUSTOM_HIERARCHY

    results = run_pipeline(
        config=config,
        privacy_config=privacy_config,
        hierarchy_config=hierarchy_config,
        custom_hierarchy=custom_hierarchy,
        output_dir=args.output,
    )

    return results


if __name__ == '__main__':
    main()
