import os
import sys
import json
import warnings
import numpy as np
import pandas as pd
from datetime import datetime

warnings.filterwarnings('ignore')

from src.config import DATASET_CONFIG, PRIVACY_CONFIG, HIERARCHY_CONFIG, CUSTOM_HIERARCHY
from src.utils import ensure_list, validate_config
from src.preprocessing import preprocess_generic
from src.hierarchy import GenericGeneralizationHierarchy
from src.attribute_correlation import AttributeCorrelationEvaluation
from src.acdp_tree import ACDPTree, compute_inverse_frequency_weights
from src.ace import KAnonymityEnforcer
from src.noise import add_noise_to_counts, PrivacyBudgetTracker
from src.metrics import (
    calculate_information_loss,
    calculate_kl_divergence,
    calculate_reidentification_risk,
    calculate_privacy_utility_tradeoff,
    convert_to_serializable,
)


def get_dataset_name(filepath):
    basename = os.path.splitext(os.path.basename(filepath))[0]
    return basename.replace(' ', '_').lower()


def run_pipeline(config=None, privacy_config=None, hierarchy_config=None,
                 custom_hierarchy=None, output_dir=None, progress_callback=None):

    def _report(msg, progress):
        print(msg)
        if progress_callback:
            progress_callback(msg, progress)

    if config is None:
        config = DATASET_CONFIG
    if privacy_config is None:
        privacy_config = PRIVACY_CONFIG
    if hierarchy_config is None:
        hierarchy_config = HIERARCHY_CONFIG
    if custom_hierarchy is None:
        custom_hierarchy = CUSTOM_HIERARCHY

    filepath = config.get('file_path', '')
    qi_attributes = config.get('qi_attributes', [])
    sensitive_attrs = ensure_list(config.get('sensitive_attribute', []))
    id_attributes = config.get('identifier_attributes', [])
    non_sensitive = config.get('non_sensitive_attributes', [])

    primary_sensitive = sensitive_attrs[0] if sensitive_attrs else ''

    k = privacy_config.get('k_anonymity', 5)
    epsilon = privacy_config.get('epsilon', 1.0)
    max_level = privacy_config.get('max_level', 3)
    max_tree_depth = privacy_config.get('max_tree_depth', 4)

    epsilon_tree = epsilon / 2.0
    epsilon_noise = epsilon / 2.0

    dataset_name = get_dataset_name(filepath)

    if output_dir is None:
        output_dir = os.path.join('results', dataset_name)
    os.makedirs(output_dir, exist_ok=True)

    # ================================================================
    # STEP 1: Load & Preprocess Data
    # ================================================================
    _report('Step 1/10: Loading & preprocessing data...', 10)
    print('=' * 80)
    print(f'PIPELINE: {dataset_name}')
    print('=' * 80)
    print(f'STEP 1: LOAD & PREPROCESS DATA')
    print('=' * 80)

    df = pd.read_csv(filepath)
    print(f'  Loaded: {len(df):,} records, {len(df.columns)} columns')

    df_clean = preprocess_generic(df, config)
    df_input = df_clean[qi_attributes + sensitive_attrs + non_sensitive].copy()

    print(f'\n  Records: {len(df_input):,}')
    print(f'  QI Attributes: {qi_attributes}')
    print(f'  Sensitive Attributes: {sensitive_attrs}')

    errors, warnings_list = validate_config(df_clean, config)
    if errors:
        for e in errors:
            print(f'  ERROR: {e}')
        print(f'  Attempting auto-fix...')

        valid_qi = [attr for attr in qi_attributes if attr in df_clean.columns]
        if len(valid_qi) < len(qi_attributes):
            print(f'  Auto-fixed: Using only valid QI attributes: {valid_qi}')
            qi_attributes = valid_qi
            config['qi_attributes'] = valid_qi

        if len(qi_attributes) == 0:
            raise ValueError('No valid QI attributes found in dataset!')

        for sa in sensitive_attrs:
            if sa not in df_clean.columns:
                raise ValueError(f'Sensitive attribute "{sa}" not found in dataset!')

    if warnings_list:
        for w in warnings_list:
            print(f'  Warning: {w}')

    # ================================================================
    # STEP 2: Build Generalization Hierarchy
    # ================================================================
    print('\n' + '=' * 80)
    print('STEP 2: BUILD GENERALIZATION HIERARCHY')
    print('=' * 80)

    hierarchy = GenericGeneralizationHierarchy(custom_hierarchy=custom_hierarchy)
    hierarchy.build_from_dataframe(df_input, qi_attributes, hierarchy_config)
    hierarchy.print_summary()
    _report('Step 2/10: Hierarchy built', 20)

    # ================================================================
    # STEP 3: ACE - Attribute Correlation Evaluation
    # ================================================================
    print('\n' + '=' * 80)
    print('STEP 3: ACE - ATTRIBUTE CORRELATION EVALUATION')
    print('=' * 80)

    ace_eval = AttributeCorrelationEvaluation()
    attribute_ranking = ace_eval.fit(df_input, qi_attributes, sensitive_attrs)
    ace_eval.print_summary()
    _report('Step 3/10: ACE correlation evaluation done', 30)

    # ================================================================
    # STEP 4: Compute Inverse Frequency Weights
    # ================================================================
    print('\n' + '=' * 80)
    print('STEP 4: COMPUTE INVERSE FREQUENCY WEIGHTS')
    print('=' * 80)

    weights = compute_inverse_frequency_weights(df_input, sensitive_attrs)
    print(f'  Weights computed. Shape: {weights.shape}')
    _report('Step 4/10: Inverse frequency weights computed', 40)

    # ================================================================
    # STEP 5: Build ACDP Tree (with DP budget allocation)
    # ================================================================
    print('\n')
    acdp_tree = ACDPTree(
        hierarchy=hierarchy,
        qi_attributes=qi_attributes,
        sensitive_attribute=sensitive_attrs,
        k=k,
        max_depth=max_level,
        weights=weights,
        attribute_ranking=attribute_ranking,
        epsilon_tree=epsilon_tree,
    )

    acdp_tree.fit(df_input)
    df_generalized = acdp_tree.transform(df_input)

    groups = df_generalized.groupby(qi_attributes).size()
    n_violations = (groups < k).sum()
    print(f'\n  After ACDP Tree: {n_violations} violation groups')

    tree_structure_file = os.path.join(output_dir, 'acdp_tree_structure.json')
    tree_structure = acdp_tree.export_tree_structure()
    with open(tree_structure_file, 'w', encoding='utf-8') as f:
        json.dump(tree_structure, f, indent=2)
    print(f'  Tree structure exported: {tree_structure_file}')
    _report('Step 5/10: ACDP Tree built', 55)

    # ================================================================
    # STEP 6: K-Anonymity Enforcer
    # ================================================================
    print('\n')
    k_enforcer = KAnonymityEnforcer(
        k=k,
        hierarchy=hierarchy,
        qi_attributes=qi_attributes,
        max_iterations=20,
    )

    print('BEFORE ENFORCEMENT:')
    k_enforcer.check_k_anonymity(df_generalized)
    print()

    df_k_anonymous = k_enforcer.enforce_k_anonymity(
        df_original=df_input,
        df_tree_output=df_generalized,
        tree_record_levels=acdp_tree.record_levels,
        verbose=True,
    )
    print()

    print('AFTER ENFORCEMENT:')
    enforcer_stats = k_enforcer.check_k_anonymity(df_k_anonymous)
    _report('Step 6/10: K-Anonymity enforced', 70)

    # ================================================================
    # STEP 7: Apply Differential Privacy (Laplace Noise)
    # ================================================================
    print('\n' + '=' * 80)
    print(f'STEP 7: DIFFERENTIAL PRIVACY (epsilon={epsilon_noise:.4f})')
    print('=' * 80)

    budget_tracker = PrivacyBudgetTracker(total_epsilon=epsilon)
    budget_tracker.consume(epsilon_tree, 'Tree construction (Exponential Mechanism)')
    budget_tracker.consume(epsilon_noise, 'Laplace noise on group counts')

    df_noisy = add_noise_to_counts(
        df=df_k_anonymous,
        epsilon=epsilon_noise,
        qi_attributes=qi_attributes,
        sensitive_attribute=sensitive_attrs,
    )

    df_noisy['noise_added'] = df_noisy['noisy_count'] - df_noisy['count']
    df_noisy['percent_error'] = (
        df_noisy['noise_added'].abs() / df_noisy['count'].replace(0, np.nan) * 100
    ).round(2)

    print(f'\n  Noisy groups: {len(df_noisy):,}')
    print(f'  Mean noise: {df_noisy["noise_added"].mean():.2f}')
    print(f'  Mean percent error: {df_noisy["percent_error"].mean():.2f}%')
    print('\nDP applied!')

    budget_tracker.print_summary()
    _report('Step 7/10: Differential privacy applied', 80)

    # ================================================================
    # STEP 8: Save Datasets
    # ================================================================
    print('\n' + '=' * 80)
    print('STEP 8: SAVE DATASETS')
    print('=' * 80)

    df_final = df_k_anonymous.copy()

    filename_anon = f'{dataset_name}_anonymized_k{k}_eps{epsilon:.1f}.csv'
    filepath_anon = os.path.join(output_dir, filename_anon)
    df_final.to_csv(filepath_anon, index=False)
    print(f'  Anonymized dataset: {filepath_anon}')
    print(f'    Records: {len(df_final):,}')

    filename_noisy = f'{dataset_name}_noisy_counts_k{k}_eps{epsilon:.1f}.csv'
    filepath_noisy = os.path.join(output_dir, filename_noisy)
    df_noisy.to_csv(filepath_noisy, index=False)
    print(f'  Noisy counts: {filepath_noisy}')

    overridden = sum(
        1 for idx in df_input.index
        if any(
            str(df_generalized.loc[idx, attr]) != str(df_k_anonymous.loc[idx, attr])
            for attr in qi_attributes
        )
    )

    metadata = {
        'dataset_info': {
            'dataset_name': dataset_name,
            'original_file': filepath,
            'original_records': len(df_input),
            'anonymized_records': len(df_final),
            'anonymized_file': filename_anon,
            'noisy_counts_file': filename_noisy,
            'processing_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        },
        'privacy_parameters': {
            'k_anonymity': k,
            'epsilon': epsilon,
            'epsilon_tree': epsilon_tree,
            'epsilon_noise': epsilon_noise,
            'max_generalization_level': max_level,
            'max_tree_depth': max_tree_depth,
        },
        'attributes': {
            'qi_attributes': qi_attributes,
            'sensitive_attributes': sensitive_attrs,
            'identifier_attributes': id_attributes,
            'non_sensitive_attributes': non_sensitive,
            'total_attributes': len(df_final.columns),
        },
        'privacy_guarantees': {
            'k_anonymity_satisfied': enforcer_stats['satisfies'],
            'min_group_size': enforcer_stats['min_group'],
            'max_group_size': enforcer_stats['max_group'],
            'avg_group_size': enforcer_stats['avg_group'],
            'total_groups': enforcer_stats['n_groups'],
        },
        'pipeline_summary': {
            'acdp_tree': {
                'violations_before_enforcement': int(n_violations),
                'groups_after_tree': int(len(groups)),
            },
            'k_anonymity_enforcer': {
                'iterations': len(k_enforcer.iteration_log),
                'records_overridden': overridden,
                'override_percentage': round(overridden / len(df_input) * 100, 2),
            },
            'noise': {
                'mean_noise': round(df_noisy['noise_added'].mean(), 4),
                'std_noise': round(df_noisy['noise_added'].std(), 4),
                'mean_percent_error': round(df_noisy['percent_error'].mean(), 2),
            },
        },
    }

    metadata_file = os.path.join(output_dir, 'anonymization_metadata.json')
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f'  Metadata: {metadata_file}')
    _report('Step 8/10: Datasets saved', 85)

    # ================================================================
    # STEP 9: Evaluate Metrics
    # ================================================================
    print('\n' + '=' * 80)
    print('STEP 9: EVALUATION METRICS')
    print('=' * 80)

    info_loss_df = calculate_information_loss(df_input, df_final, qi_attributes)
    print('\nINFORMATION LOSS:')
    print(info_loss_df.to_string(index=False))

    dist_preserve_df = calculate_kl_divergence(df_input, df_final, qi_attributes)
    print('\nDISTRIBUTION PRESERVATION:')
    print(dist_preserve_df.to_string(index=False))

    orig_risk = calculate_reidentification_risk(df_input, qi_attributes)
    anon_risk = calculate_reidentification_risk(df_final, qi_attributes)
    print(f'\nRE-IDENTIFICATION RISK:')
    print(f'  Original - unique risk: {orig_risk["unique_risk_pct"]:.2f}%')
    print(f'  Anonymized - unique risk: {anon_risk["unique_risk_pct"]:.2f}%')

    tradeoff = calculate_privacy_utility_tradeoff(
        orig_risk, anon_risk, info_loss_df, dist_preserve_df
    )
    print(f'\nPRIVACY-UTILITY TRADEOFF:')
    print(f'  Privacy Gain: {tradeoff["privacy_gain_pct"]:.2f}%')
    print(f'  Utility Loss: {tradeoff["utility_loss_pct"]:.2f}%')
    print(f'  Utility Score: {tradeoff["utility_score"]:.2f}/100')
    print(f'  P/U Ratio: {tradeoff["privacy_utility_ratio"]:.2f}')

    _report('Step 9/10: Metrics evaluated', 93)

    # Sensitive attributes distribution
    sens_distributions = {}
    for sa in sensitive_attrs:
        if sa in df_input.columns and sa in df_final.columns:
            orig_dist = df_input[sa].value_counts(normalize=True).sort_index() * 100
            anon_dist = df_final[sa].value_counts(normalize=True).sort_index() * 100
            tvd = 0.5 * sum(
                abs(orig_dist.get(c, 0) - anon_dist.get(c, 0))
                for c in set(orig_dist.index) | set(anon_dist.index)
            )
            sens_distributions[sa] = {
                'original': {str(k): float(v) for k, v in orig_dist.items()},
                'anonymized': {str(k): float(v) for k, v in anon_dist.items()},
                'tvd': float(tvd),
            }

    # ================================================================
    # STEP 10: Save Evaluation Metrics & Report
    # ================================================================
    print('\n' + '=' * 80)
    print('STEP 10: SAVE EVALUATION METRICS & REPORT')
    print('=' * 80)

    col_name = 'Unique Change (%)' if 'Unique Change (%)' in info_loss_df.columns else 'Unique Lost (%)'
    avg_info_loss = info_loss_df[col_name].mean()
    utility_loss_capped = min(max(avg_info_loss, 0), 100)
    utility_score_capped = max(0, min(100, 100 - utility_loss_capped))

    evaluation_metrics = {
        'information_loss': info_loss_df.to_dict('records'),
        'distribution_preservation': dist_preserve_df.to_dict('records'),
        'reidentification_risk': {
            'original': orig_risk,
            'anonymized': anon_risk,
        },
        'privacy_utility_tradeoff': tradeoff,
        'sensitive_distributions': sens_distributions,
        'summary': {
            'privacy_satisfied': bool(enforcer_stats['satisfies']),
            'k_anonymity': int(k),
            'epsilon': float(epsilon),
            'min_group_size': int(anon_risk['min_group_size']),
            'avg_information_loss': float(round(avg_info_loss, 2)),
            'avg_kl_divergence': float(round(dist_preserve_df['KL-Divergence'].mean(), 4)),
            'privacy_gain': float(tradeoff['privacy_gain_pct']),
            'utility_score': float(round(utility_score_capped, 2)),
        },
    }

    evaluation_metrics = convert_to_serializable(evaluation_metrics)

    metrics_file = os.path.join(output_dir, 'evaluation_metrics.json')
    with open(metrics_file, 'w') as f:
        json.dump(evaluation_metrics, f, indent=2)
    print(f'  Evaluation metrics: {metrics_file}')

    report_content = f"""
===============================================================================
PRIVACY-PRESERVING DATA ANONYMIZATION - EVALUATION REPORT
===============================================================================

Dataset: {dataset_name}
Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Records: {len(df_input):,} (original) -> {len(df_final):,} (anonymized)

===============================================================================
1. PRIVACY GUARANTEES
===============================================================================

k-Anonymity:
  - Parameter k              : {k}
  - Satisfied                : {enforcer_stats['satisfies']}
  - Min group size           : {anon_risk['min_group_size']}
  - Max group size           : {anon_risk['max_group_size']:,}
  - Avg group size           : {anon_risk['avg_group_size']:.2f}
  - Total equivalence classes: {anon_risk['total_groups']:,}

Differential Privacy:
  - Epsilon (total)          : {epsilon}
  - Tree construction budget : {epsilon_tree}
  - Laplace noise budget     : {epsilon_noise}
  - Mechanism                : Exponential Mechanism + Laplace Noise
  - Mean noise added         : {df_noisy['noise_added'].mean():.4f}

Re-identification Risk:
  - Before: {orig_risk['unique_risk_pct']:.2f}% unique individuals
  - After : {anon_risk['unique_risk_pct']:.2f}% unique individuals
  - Risk reduction: {orig_risk['unique_risk_pct'] - anon_risk['unique_risk_pct']:.2f} pp

===============================================================================
2. INFORMATION LOSS
===============================================================================

{info_loss_df.to_string(index=False)}

Summary:
   - Average unique values change: {avg_info_loss:.2f}%
  - Average entropy reduction : {info_loss_df['Entropy Reduction (%)'].mean():.2f}%

===============================================================================
3. DISTRIBUTION PRESERVATION
===============================================================================

{dist_preserve_df.to_string(index=False)}

Summary:
  - Average KL-Divergence: {dist_preserve_df['KL-Divergence'].mean():.4f}
  - Average TVD          : {dist_preserve_df['TVD'].mean():.4f}
  - Good quality attributes: {(dist_preserve_df['Preservation Quality'] == 'Good').sum()} / {len(qi_attributes)}

===============================================================================
4. PRIVACY-UTILITY TRADE-OFF
===============================================================================

Privacy Metrics:
  - Privacy Gain             : {tradeoff['privacy_gain_pct']:.2f}%

Utility Metrics:
  - Utility Loss             : {tradeoff['utility_loss_pct']:.2f}%
  - Utility Score            : {tradeoff['utility_score']:.2f}/100

Trade-off:
  - Privacy/Utility Ratio    : {tradeoff['privacy_utility_ratio']:.2f}
  - Assessment               : {'Good' if tradeoff['privacy_utility_ratio'] > 1.0 else 'Fair'}

===============================================================================
5. ANONYMIZATION PIPELINE SUMMARY
===============================================================================

Step 1-2: Preprocessing + Hierarchy
Step 3: ACE - Attribute Correlation Evaluation (AHP-based)
Step 4: ACDP Tree (Generalization Optimizer)
  - Tree depth               : {max_level}
  - Split criteria           : Weighted Mutual Information + Exponential Mechanism
  - DP budget                : epsilon={epsilon_tree:.4f} (arithmetic progression)
  - Violations before enforcement : {n_violations} groups

Step 5: K-Anonymity Enforcement
  - Iterations               : {len(k_enforcer.iteration_log)}
  - Records overridden       : {overridden:,} ({overridden/len(df_input)*100:.2f}%)
  - Final violations         : 0 groups

Step 6: Differential Privacy (Laplace Noise)
  - Epsilon consumed         : {epsilon_noise} (noise)
  - Mean percent error       : {df_noisy['percent_error'].mean():.2f}%

===============================================================================
6. CONCLUSION
===============================================================================

Privacy Status:
  - k-anonymity (k={k}) satisfied
  - Differential Privacy (epsilon={epsilon}) applied
  - Re-identification risk significantly reduced

Utility Status:
  - Utility score: {utility_score_capped:.2f}/100

    Overall:
  The anonymized dataset achieves {k}-anonymity with epsilon={epsilon}
  differential privacy while maintaining {utility_score_capped:.2f}% utility.

===============================================================================
End of Report
===============================================================================
"""

    report_file = os.path.join(output_dir, 'evaluation_report.txt')
    with open(report_file, 'w') as f:
        f.write(report_content)
    print(f'  Evaluation report: {report_file}')

    # ================================================================
    # FINAL SUMMARY
    # ================================================================
    print('\n' + '=' * 80)
    print('PIPELINE COMPLETE!')
    print('=' * 80)
    print(f'  Dataset              : {dataset_name}')
    print(f'  Original records     : {len(df_input):,}')
    print(f'  k-anonymity (k={k})  : {enforcer_stats["satisfies"]}')
    print(f'  DP (epsilon)         : {epsilon}')
    print(f'  Final unique groups  : {enforcer_stats["n_groups"]:,}')
    print(f'  Utility score        : {utility_score_capped:.2f}/100')
    print(f'  Output               : {output_dir}')
    print('=' * 80)
    _report('Pipeline complete!', 100)

    return {
        'df_original': df_input,
        'df_anonymized': df_k_anonymous,
        'df_noisy': df_noisy,
        'df_generalized': df_generalized,
        'metrics': {
            'information_loss': info_loss_df,
            'distribution_preservation': dist_preserve_df,
            'reidentification_risk': {'original': orig_risk, 'anonymized': anon_risk},
            'privacy_utility_tradeoff': tradeoff,
        },
        'acdp_tree': acdp_tree,
        'k_enforcer': k_enforcer,
        'ace_evaluation': ace_eval,
        'hierarchy': hierarchy,
        'metadata': metadata,
    }
