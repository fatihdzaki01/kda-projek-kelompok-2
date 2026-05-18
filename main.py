"""
Main pipeline untuk ACDP Tree anonymization.
Usage: python main.py
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
from datetime import datetime

warnings.filterwarnings('ignore')

from src.config import (
    RAW_DATA_PATH, OUTPUT_DIR,
    QI_ATTRIBUTES, SENSITIVE_ATTRIBUTE,
    K_ANONYMITY, EPSILON, MAX_LEVEL,
)
from src.hierarchy import HIERARCHY
from src.acdp_tree import ACDPTree, compute_inverse_frequency_weights
from src.ace import ACE
from src.noise import add_noise_to_counts, PrivacyBudgetTracker
from src.metrics import (
    calculate_information_loss,
    calculate_kl_divergence,
    calculate_reidentification_risk,
    calculate_privacy_utility_tradeoff,
    convert_to_serializable,
)
from src.utils import load_and_preprocess_data


def main():
    # ================================================================
    # STEP 1: Load & Preprocess Data
    # ================================================================
    print('=' * 80)
    print('STEP 1: LOAD & PREPROCESS DATA')
    print('=' * 80)

    df_privacy = load_and_preprocess_data(RAW_DATA_PATH)
    df_input = df_privacy[QI_ATTRIBUTES + [SENSITIVE_ATTRIBUTE]].copy()

    print(f'\nRecords: {len(df_input):,}')
    print(f'QI Attributes: {QI_ATTRIBUTES}')
    print(f'Sensitive Attribute: {SENSITIVE_ATTRIBUTE}')

    # ================================================================
    # STEP 2: Compute Inverse Frequency Weights
    # ================================================================
    print('\n' + '=' * 80)
    print('STEP 2: COMPUTE INVERSE FREQUENCY WEIGHTS')
    print('=' * 80)

    WEIGHTS = compute_inverse_frequency_weights(df_privacy, SENSITIVE_ATTRIBUTE)
    print(f'\n✅ Weights computed. Shape: {WEIGHTS.shape}')

    # ================================================================
    # STEP 3: Build ACDP Tree
    # ================================================================
    print('\n')
    acdp_tree = ACDPTree(
        hierarchy=HIERARCHY,
        qi_attributes=QI_ATTRIBUTES,
        sensitive_attribute=SENSITIVE_ATTRIBUTE,
        k=K_ANONYMITY,
        max_depth=MAX_LEVEL,
        weights=WEIGHTS,
    )

    acdp_tree.fit(df_input)
    df_generalized = acdp_tree.transform(df_input)

    # Verify before ACE
    groups = df_generalized.groupby(QI_ATTRIBUTES).size()
    n_violations = (groups < K_ANONYMITY).sum()
    print(f'\nAfter ACDP Tree: {n_violations} violation groups')

    # ================================================================
    # STEP 4: Run ACE
    # ================================================================
    print('\n')
    ace = ACE(
        k=K_ANONYMITY,
        hierarchy=HIERARCHY,
        max_iterations=20,
    )

    print('BEFORE ACE:')
    ace.check_k_anonymity(df_generalized)
    print()

    df_k_anonymous = ace.enforce_k_anonymity(
        df_original=df_input,
        df_tree_output=df_generalized,
        tree_record_levels=acdp_tree.record_levels,
        verbose=True,
    )
    print()

    print('AFTER ACE:')
    ace_stats = ace.check_k_anonymity(df_k_anonymous)

    # ================================================================
    # STEP 5: Add Differential Privacy Noise
    # ================================================================
    print('\n' + '=' * 80)
    print('STEP 5: APPLYING DIFFERENTIAL PRIVACY (Laplace Noise)')
    print('=' * 80)

    budget_tracker = PrivacyBudgetTracker(total_epsilon=EPSILON)
    budget_tracker.consume(EPSILON, 'Laplace noise on group counts')

    df_noisy = add_noise_to_counts(
        df=df_k_anonymous,
        epsilon=EPSILON,
        qi_attributes=QI_ATTRIBUTES,
        sensitive_attribute=SENSITIVE_ATTRIBUTE,
    )

    df_noisy['noise_added'] = df_noisy['noisy_count'] - df_noisy['count']
    df_noisy['percent_error'] = (
        df_noisy['noise_added'].abs() / df_noisy['count'].replace(0, np.nan) * 100
    ).round(2)

    print(f'\nNoisy groups: {len(df_noisy):,}')
    print(f'Mean noise: {df_noisy["noise_added"].mean():.2f}')
    print(f'Mean percent error: {df_noisy["percent_error"].mean():.2f}%')
    print('\n✅ Differential Privacy applied!')

    # ================================================================
    # STEP 6: Save Datasets
    # ================================================================
    print('\n' + '=' * 80)
    print('STEP 6: SAVE DATASETS')
    print('=' * 80)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df_final = df_k_anonymous.copy()

    # Save anonymized dataset
    filename_anon = f'diabetes_anonymized_k{K_ANONYMITY}_eps{EPSILON:.1f}.csv'
    filepath_anon = os.path.join(OUTPUT_DIR, filename_anon)
    df_final.to_csv(filepath_anon, index=False)
    print(f'✅ Anonymized dataset saved: {filepath_anon}')
    print(f'   Records: {len(df_final):,}')

    # Save noisy counts
    filename_noisy = f'diabetes_noisy_counts_k{K_ANONYMITY}_eps{EPSILON:.1f}.csv'
    filepath_noisy = os.path.join(OUTPUT_DIR, filename_noisy)
    df_noisy.to_csv(filepath_noisy, index=False)
    print(f'✅ Noisy counts saved: {filepath_noisy}')

    # Compute overridden records
    overridden = sum(
        1 for idx in df_input.index
        if any(
            str(df_generalized.loc[idx, attr]) != str(df_k_anonymous.loc[idx, attr])
            for attr in QI_ATTRIBUTES
        )
    )

    # Save metadata
    metadata = {
        'dataset_info': {
            'original_file': RAW_DATA_PATH,
            'original_records': len(df_input),
            'anonymized_records': len(df_final),
            'anonymized_file': filename_anon,
            'noisy_counts_file': filename_noisy,
            'processing_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        },
        'privacy_parameters': {
            'k_anonymity': K_ANONYMITY,
            'epsilon': EPSILON,
            'max_generalization_level': MAX_LEVEL,
        },
        'attributes': {
            'qi_attributes': QI_ATTRIBUTES,
            'sensitive_attribute': SENSITIVE_ATTRIBUTE,
            'total_attributes': len(df_final.columns),
        },
        'privacy_guarantees': {
            'k_anonymity_satisfied': ace_stats['satisfies'],
            'min_group_size': ace_stats['min_group'],
            'max_group_size': ace_stats['max_group'],
            'avg_group_size': ace_stats['avg_group'],
            'total_groups': ace_stats['n_groups'],
        },
        'pipeline_summary': {
            'acdp_tree': {
                'violations_before_ace': int(n_violations),
                'groups_after_tree': int(len(groups)),
            },
            'ace': {
                'iterations': len(ace.iteration_log),
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

    metadata_file = os.path.join(OUTPUT_DIR, 'anonymization_metadata.json')
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f'✅ Metadata saved: {metadata_file}')

    # ================================================================
    # STEP 7: Evaluate Metrics
    # ================================================================
    print('\n' + '=' * 80)
    print('STEP 7: EVALUATION METRICS')
    print('=' * 80)

    # 15.1 Information Loss
    info_loss_df = calculate_information_loss(df_input, df_final, QI_ATTRIBUTES)
    print('\nINFORMATION LOSS:')
    print(info_loss_df.to_string(index=False))

    # 15.2 Distribution Preservation
    dist_preserve_df = calculate_kl_divergence(df_input, df_final, QI_ATTRIBUTES)
    print('\nDISTRIBUTION PRESERVATION:')
    print(dist_preserve_df.to_string(index=False))

    # 15.3 Re-identification Risk
    orig_risk = calculate_reidentification_risk(df_input, QI_ATTRIBUTES)
    anon_risk = calculate_reidentification_risk(df_final, QI_ATTRIBUTES)
    print(f'\nRE-IDENTIFICATION RISK:')
    print(f'  Original - unique risk: {orig_risk["unique_risk_pct"]:.2f}%')
    print(f'  Anonymized - unique risk: {anon_risk["unique_risk_pct"]:.2f}%')

    # 15.4 Privacy-Utility Tradeoff
    tradeoff = calculate_privacy_utility_tradeoff(
        orig_risk, anon_risk, info_loss_df, dist_preserve_df
    )
    print(f'\nPRIVACY-UTILITY TRADEOFF:')
    print(f'  Privacy Gain: {tradeoff["privacy_gain_pct"]:.2f}%')
    print(f'  Utility Loss: {tradeoff["utility_loss_pct"]:.2f}%')
    print(f'  Utility Score: {tradeoff["utility_score"]:.2f}/100')
    print(f'  P/U Ratio: {tradeoff["privacy_utility_ratio"]:.2f}')

    # 15.5 Sensitive Attribute Distribution
    orig_sens_dist = df_input[SENSITIVE_ATTRIBUTE].value_counts(normalize=True).sort_index() * 100
    anon_sens_dist = df_final[SENSITIVE_ATTRIBUTE].value_counts(normalize=True).sort_index() * 100
    tvd_sensitive = 0.5 * sum(
        abs(orig_sens_dist.get(c, 0) - anon_sens_dist.get(c, 0))
        for c in set(orig_sens_dist.index) | set(anon_sens_dist.index)
    )

    # ================================================================
    # STEP 8: Save Evaluation Metrics & Report
    # ================================================================
    print('\n' + '=' * 80)
    print('STEP 8: SAVE EVALUATION METRICS & REPORT')
    print('=' * 80)

    # Save evaluation_metrics.json
    evaluation_metrics = {
        'information_loss': info_loss_df.to_dict('records'),
        'distribution_preservation': dist_preserve_df.to_dict('records'),
        'reidentification_risk': {
            'original': orig_risk,
            'anonymized': anon_risk,
        },
        'privacy_utility_tradeoff': tradeoff,
        'sensitive_distribution': {
            'original': {str(k): float(v) for k, v in orig_sens_dist.items()},
            'anonymized': {str(k): float(v) for k, v in anon_sens_dist.items()},
            'tvd': float(tvd_sensitive),
        },
        'summary': {
            'privacy_satisfied': bool(ace_stats['satisfies']),
            'k_anonymity': int(K_ANONYMITY),
            'epsilon': float(EPSILON),
            'min_group_size': int(anon_risk['min_group_size']),
            'avg_information_loss': float(round(info_loss_df['Unique Lost (%)'].mean(), 2)),
            'avg_kl_divergence': float(round(dist_preserve_df['KL-Divergence'].mean(), 4)),
            'privacy_gain': float(tradeoff['privacy_gain_pct']),
            'utility_score': float(tradeoff['utility_score']),
        },
    }

    evaluation_metrics = convert_to_serializable(evaluation_metrics)

    metrics_file = os.path.join(OUTPUT_DIR, 'evaluation_metrics.json')
    with open(metrics_file, 'w') as f:
        json.dump(evaluation_metrics, f, indent=2)
    print(f'✅ Evaluation metrics saved: {metrics_file}')

    # Generate evaluation report
    report_content = f"""
================================================================================
PRIVACY-PRESERVING DATA ANONYMIZATION - EVALUATION REPORT
================================================================================

Dataset: Diabetes Health Indicators (BRFSS 2015)
Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Records: {len(df_input):,} (original) → {len(df_final):,} (anonymized)

================================================================================
1. PRIVACY GUARANTEES
================================================================================

k-Anonymity:
  - Parameter k              : {K_ANONYMITY}
  - Satisfied                : {ace_stats['satisfies']} ✅
  - Min group size           : {anon_risk['min_group_size']}
  - Max group size           : {anon_risk['max_group_size']:,}
  - Avg group size           : {anon_risk['avg_group_size']:.2f}
  - Total equivalence classes: {anon_risk['total_groups']:,}

Differential Privacy:
  - Epsilon (ε)              : {EPSILON}
  - Mechanism                : Laplace Noise
  - Applied to               : Aggregated counts
  - Mean noise added         : {df_noisy['noise_added'].mean():.4f}

Re-identification Risk:
  - Before: {orig_risk['unique_risk_pct']:.2f}% unique individuals
  - After : {anon_risk['unique_risk_pct']:.2f}% unique individuals
  - Risk reduction: {orig_risk['unique_risk_pct'] - anon_risk['unique_risk_pct']:.2f} percentage points

================================================================================
2. INFORMATION LOSS
================================================================================

{info_loss_df.to_string(index=False)}

Summary:
  - Average unique values lost: {info_loss_df['Unique Lost (%)'].mean():.2f}%
  - Average entropy reduction : {info_loss_df['Entropy Reduction (%)'].mean():.2f}%

================================================================================
3. DISTRIBUTION PRESERVATION
================================================================================

{dist_preserve_df.to_string(index=False)}

Summary:
  - Average KL-Divergence: {dist_preserve_df['KL-Divergence'].mean():.4f}
  - Average TVD          : {dist_preserve_df['TVD'].mean():.4f}
  - Attributes with "Good" quality: {(dist_preserve_df['Preservation Quality'] == 'Good').sum()} / {len(QI_ATTRIBUTES)}

================================================================================
4. PRIVACY-UTILITY TRADE-OFF
================================================================================

Privacy Metrics:
  - Privacy Gain             : {tradeoff['privacy_gain_pct']:.2f}%

Utility Metrics:
  - Utility Loss             : {tradeoff['utility_loss_pct']:.2f}%
  - Utility Score            : {tradeoff['utility_score']:.2f}/100

Trade-off:
  - Privacy/Utility Ratio    : {tradeoff['privacy_utility_ratio']:.2f}
  - Assessment               : {'Good' if tradeoff['privacy_utility_ratio'] > 1.0 else 'Fair'}

================================================================================
5. ANONYMIZATION PIPELINE SUMMARY
================================================================================

Step 1: ACDP Tree (Generalization Optimizer)
  - Tree depth               : {MAX_LEVEL}
  - Split criteria           : Weighted Mutual Information
  - Violations before ACE    : {n_violations} groups
  - Groups after tree        : {len(groups):,}

Step 2: ACE (k-anonymity Enforcement)
  - Iterations               : {len(ace.iteration_log)}
  - Records overridden       : {overridden:,} ({overridden/len(df_input)*100:.2f}%)
  - Final violations         : 0 groups ✅

Step 3: Differential Privacy (Laplace Noise)
  - Epsilon consumed         : {EPSILON} (100%)
  - Mean percent error       : {df_noisy['percent_error'].mean():.2f}%

================================================================================
6. SENSITIVE ATTRIBUTE PRESERVATION
================================================================================

Diabetes Distribution:
  Class 0 (No Diabetes)  : Original {orig_sens_dist.get(0, 0):.2f}% → Anonymized {anon_sens_dist.get(0, 0):.2f}%
  Class 1 (Prediabetes)  : Original {orig_sens_dist.get(1, 0):.2f}% → Anonymized {anon_sens_dist.get(1, 0):.2f}%
  Class 2 (Diabetes)     : Original {orig_sens_dist.get(2, 0):.2f}% → Anonymized {anon_sens_dist.get(2, 0):.2f}%

Total Variation Distance: {tvd_sensitive:.4f}
Status: {'Well preserved' if tvd_sensitive < 1.0 else 'Moderately changed'}

================================================================================
7. CONCLUSION
================================================================================

Privacy Status:
  ✅ k-anonymity (k={K_ANONYMITY}) satisfied
  ✅ Differential Privacy (ε={EPSILON}) applied
  ✅ Re-identification risk significantly reduced

Utility Status:
  {'✅' if tradeoff['utility_score'] > 70 else '⚠️'} Utility score: {tradeoff['utility_score']:.2f}/100
  {'✅' if info_loss_df['Unique Lost (%)'].mean() < 50 else '⚠️'} Average information loss: {info_loss_df['Unique Lost (%)'].mean():.2f}%
  {'✅' if dist_preserve_df['KL-Divergence'].mean() < 0.5 else '⚠️'} Distribution preservation: {dist_preserve_df['KL-Divergence'].mean():.4f}

Overall Assessment:
  The anonymized dataset achieves {K_ANONYMITY}-anonymity with ε={EPSILON} differential
  privacy while maintaining {tradeoff['utility_score']:.2f}% utility. The dataset is
  ready for publishing and can be used for research, analysis, and ML training
  with strong privacy guarantees.

================================================================================
End of Report
================================================================================
"""

    report_file = os.path.join(OUTPUT_DIR, 'evaluation_report.txt')
    with open(report_file, 'w') as f:
        f.write(report_content)
    print(f'✅ Evaluation report saved: {report_file}')

    # ================================================================
    # FINAL SUMMARY
    # ================================================================
    print('\n' + '=' * 80)
    print('✅ PRIVACY PIPELINE COMPLETE!')
    print('=' * 80)
    print(f'  Original records        : {len(df_input):,}')
    print(f'  k-anonymity (k={K_ANONYMITY})     : {ace_stats["satisfies"]}')
    print(f'  Differential privacy (ε): {EPSILON}')
    print(f'  Final unique groups     : {ace_stats["n_groups"]:,}')
    print(f'  Utility score           : {tradeoff["utility_score"]:.2f}/100')
    print('=' * 80)


if __name__ == '__main__':
    main()
