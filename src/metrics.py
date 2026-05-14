"""
Evaluation Metrics for Privacy-Preserving Data Anonymization.

Contains information loss, KL-divergence, re-identification risk,
privacy-utility tradeoff, and serialization utilities.
"""

import numpy as np
import pandas as pd
from scipy.stats import entropy


def calculate_information_loss(df_original, df_anonymized, qi_attributes):
    """
    Hitung information loss per attribute.

    Metrics:
    1. Unique values lost
    2. Entropy reduction (generalization intensity)
    """
    results = []

    for attr in qi_attributes:
        orig_unique = df_original[attr].nunique()
        anon_unique = df_anonymized[attr].nunique()

        unique_lost_pct = (1 - anon_unique / orig_unique) * 100

        orig_entropy = -sum(
            (df_original[attr].value_counts(normalize=True) *
             np.log2(df_original[attr].value_counts(normalize=True) + 1e-10))
        )
        anon_entropy = -sum(
            (df_anonymized[attr].value_counts(normalize=True) *
             np.log2(df_anonymized[attr].value_counts(normalize=True) + 1e-10))
        )

        entropy_reduction = (1 - anon_entropy / (orig_entropy + 1e-10)) * 100

        results.append({
            'Attribute': attr,
            'Original Unique': orig_unique,
            'Anonymized Unique': anon_unique,
            'Unique Lost (%)': round(unique_lost_pct, 2),
            'Entropy Reduction (%)': round(entropy_reduction, 2),
        })

    return pd.DataFrame(results)


def calculate_kl_divergence(df_original, df_anonymized, qi_attributes):
    """
    Hitung KL-divergence untuk setiap attribute.
    KL-divergence mengukur seberapa berbeda distribusi anonymized dari original.

    Lower = better (distribusi lebih terjaga)
    """
    results = []

    for attr in qi_attributes:
        orig_dist = df_original[attr].value_counts(normalize=True)
        anon_dist = df_anonymized[attr].value_counts(normalize=True)

        # Convert index to string untuk avoid type error
        orig_dist.index = orig_dist.index.astype(str)
        anon_dist.index = anon_dist.index.astype(str)

        # Align indices
        all_values = sorted(set(orig_dist.index) | set(anon_dist.index))
        orig_aligned = [orig_dist.get(v, 1e-10) for v in all_values]
        anon_aligned = [anon_dist.get(v, 1e-10) for v in all_values]

        # Calculate KL-divergence
        kl_div = entropy(orig_aligned, anon_aligned)

        # Total Variation Distance
        tvd = 0.5 * sum(abs(o - a) for o, a in zip(orig_aligned, anon_aligned))

        results.append({
            'Attribute': attr,
            'KL-Divergence': round(kl_div, 4),
            'TVD': round(tvd, 4),
            'Preservation Quality': 'Good' if kl_div < 0.5 else 'Fair' if kl_div < 1.0 else 'Poor',
        })

    return pd.DataFrame(results)


def calculate_reidentification_risk(df, qi_attributes):
    """
    Hitung re-identification risk.

    Metrics:
    1. Percentage of unique individuals (group size = 1)
    2. Percentage of small groups (group size < 5)
    3. Average group size
    """
    groups = df.groupby(qi_attributes).size()

    unique_individuals = (groups == 1).sum()
    small_groups = (groups < 5).sum()
    total_groups = len(groups)

    unique_risk = unique_individuals / total_groups * 100 if total_groups > 0 else 0
    small_group_risk = small_groups / total_groups * 100 if total_groups > 0 else 0

    return {
        'total_groups': total_groups,
        'unique_individuals': unique_individuals,
        'small_groups': small_groups,
        'unique_risk_pct': round(unique_risk, 2),
        'small_group_risk_pct': round(small_group_risk, 2),
        'avg_group_size': round(groups.mean(), 2),
        'min_group_size': int(groups.min()),
        'max_group_size': int(groups.max()),
    }


def calculate_privacy_utility_tradeoff(
    orig_risk, anon_risk, info_loss_df, dist_preserve_df
):
    """
    Hitung trade-off antara privacy gain dan utility loss.
    """
    # Privacy gain (0-100, higher = better)
    privacy_gain = (
        (orig_risk['unique_risk_pct'] - anon_risk['unique_risk_pct']) /
        (orig_risk['unique_risk_pct'] + 1e-10) * 100
    )

    # Utility loss (0-100, lower = better)
    avg_info_loss = info_loss_df['Unique Lost (%)'].mean()
    avg_kl_div = dist_preserve_df['KL-Divergence'].mean()

    # Combined utility score (0-100, higher = better)
    utility_score = 100 - avg_info_loss

    # Privacy-Utility Ratio
    pu_ratio = privacy_gain / (avg_info_loss + 1e-10)

    return {
        'privacy_gain_pct': round(privacy_gain, 2),
        'utility_loss_pct': round(avg_info_loss, 2),
        'utility_score': round(utility_score, 2),
        'avg_kl_divergence': round(avg_kl_div, 4),
        'privacy_utility_ratio': round(pu_ratio, 2),
    }


def convert_to_serializable(obj):
    """Convert numpy/pandas types to JSON-serializable Python types."""
    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    elif isinstance(obj, pd.Series):
        return convert_to_serializable(obj.to_dict())
    else:
        return obj
