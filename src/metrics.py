"""
Evaluation Metrics for Privacy-Preserving Data Anonymization.

Contains information loss, KL-divergence, re-identification risk,
privacy-utility tradeoff, and serialization utilities.
"""

import numpy as np
import pandas as pd
from scipy.stats import entropy


def calculate_information_loss(df_original, df_anonymized, qi_attributes, use_sampling=True, max_sample_size=100000):
    """
    Hitung information loss per attribute.

    Metrics:
    1. Unique values lost
    2. Entropy reduction (generalization intensity)
    
    Args:
        df_original: Original dataframe
        df_anonymized: Anonymized dataframe
        qi_attributes: List of QI attribute names
        use_sampling: Use sampling for large datasets (default True)
        max_sample_size: Maximum sample size (default 100k)
    """
    results = []
    
    # Optimization: Sample for large datasets (entropy calculation)
    if use_sampling and len(df_original) > max_sample_size:
        print(f"  ⚡ Sampling {max_sample_size:,} rows for information loss calculation")
        df_orig_calc = df_original.sample(n=max_sample_size, random_state=42)
        df_anon_calc = df_anonymized.sample(n=max_sample_size, random_state=42)
    else:
        df_orig_calc = df_original
        df_anon_calc = df_anonymized

    for attr in qi_attributes:
        try:
            # Handle missing attributes
            if attr not in df_original.columns or attr not in df_anonymized.columns:
                continue
            
            # Use full data for unique count (fast operation)
            orig_unique = df_original[attr].nunique()
            anon_unique = df_anonymized[attr].nunique()

            # Handle zero unique values
            if orig_unique == 0:
                unique_lost_pct = 0.0
            else:
                unique_lost_pct = (1 - anon_unique / orig_unique) * 100

            # Use sampled data for entropy (slow operation)
            orig_value_counts = df_orig_calc[attr].value_counts(normalize=True)
            anon_value_counts = df_anon_calc[attr].value_counts(normalize=True)
            
            if len(orig_value_counts) == 0:
                orig_entropy = 0.0
            else:
                orig_entropy = -sum(
                    (orig_value_counts * np.log2(orig_value_counts + 1e-10))
                )
            
            if len(anon_value_counts) == 0:
                anon_entropy = 0.0
            else:
                anon_entropy = -sum(
                    (anon_value_counts * np.log2(anon_value_counts + 1e-10))
                )

            if orig_entropy == 0:
                entropy_reduction = 0.0
            else:
                entropy_reduction = (1 - anon_entropy / orig_entropy) * 100

            results.append({
                'Attribute': attr,
                'Original Unique': orig_unique,
                'Anonymized Unique': anon_unique,
                'Unique Change (%)': round(unique_lost_pct, 2),
                'Entropy Reduction (%)': round(entropy_reduction, 2),
            })
        except Exception as e:
            print(f"  Warning: Failed to calculate information loss for '{attr}': {e}")
            results.append({
                'Attribute': attr,
                'Original Unique': 0,
                'Anonymized Unique': 0,
                'Unique Change (%)': 0.0,
                'Entropy Reduction (%)': 0.0,
            })

    return pd.DataFrame(results)


def calculate_kl_divergence(df_original, df_anonymized, qi_attributes, use_sampling=True, max_sample_size=100000):
    """
    Hitung KL-divergence untuk setiap attribute.
    KL-divergence mengukur seberapa berbeda distribusi anonymized dari original.

    Lower = better (distribusi lebih terjaga)
    
    Args:
        df_original: Original dataframe
        df_anonymized: Anonymized dataframe
        qi_attributes: List of QI attribute names
        use_sampling: Use sampling for large datasets (default True)
        max_sample_size: Maximum sample size (default 100k)
    """
    import numpy as np
    results = []
    
    # Optimization: Sample for large datasets
    if use_sampling and len(df_original) > max_sample_size:
        print(f"  ⚡ Sampling {max_sample_size:,} rows for KL-divergence calculation")
        df_orig_calc = df_original.sample(n=max_sample_size, random_state=42)
        df_anon_calc = df_anonymized.sample(n=max_sample_size, random_state=42)
    else:
        df_orig_calc = df_original
        df_anon_calc = df_anonymized

    for attr in qi_attributes:
        try:
            # Handle missing attributes
            if attr not in df_orig_calc.columns or attr not in df_anon_calc.columns:
                continue
            
            orig_dist = df_orig_calc[attr].value_counts(normalize=True)
            anon_dist = df_anon_calc[attr].value_counts(normalize=True)

            # Handle empty distributions
            if len(orig_dist) == 0 or len(anon_dist) == 0:
                results.append({
                    'Attribute': attr,
                    'KL-Divergence': 0.0,
                    'TVD': 0.0,
                    'Preservation Quality': 'N/A',
                })
                continue

            # Convert index to string untuk avoid type error
            orig_dist.index = orig_dist.index.astype(str)
            anon_dist.index = anon_dist.index.astype(str)

            # Align indices
            all_values = sorted(set(orig_dist.index) | set(anon_dist.index))
            orig_aligned = np.array([orig_dist.get(v, 1e-10) for v in all_values])
            anon_aligned = np.array([anon_dist.get(v, 1e-10) for v in all_values])

            # Normalize to ensure they sum to 1
            orig_aligned = orig_aligned / (orig_aligned.sum() + 1e-10)
            anon_aligned = anon_aligned / (anon_aligned.sum() + 1e-10)

            # Calculate KL-divergence (with safety check)
            kl_div = entropy(orig_aligned, anon_aligned)
            if np.isnan(kl_div) or np.isinf(kl_div):
                kl_div = 0.0

            # Total Variation Distance
            tvd = 0.5 * np.sum(np.abs(orig_aligned - anon_aligned))

            results.append({
                'Attribute': attr,
                'KL-Divergence': round(float(kl_div), 4),
                'TVD': round(float(tvd), 4),
                'Preservation Quality': 'Good' if kl_div < 0.5 else 'Fair' if kl_div < 1.0 else 'Poor',
            })
        except Exception as e:
            print(f"  Warning: Failed to calculate KL-divergence for '{attr}': {e}")
            results.append({
                'Attribute': attr,
                'KL-Divergence': 0.0,
                'TVD': 0.0,
                'Preservation Quality': 'N/A',
            })

    return pd.DataFrame(results)


def calculate_reidentification_risk(df, qi_attributes, use_sampling=True, max_sample_size=100000):
    """
    Hitung re-identification risk.

    Metrics:
    1. Percentage of unique individuals (group size = 1)
    2. Percentage of small groups (group size < 5)
    3. Average group size
    
    Args:
        df: DataFrame to analyze
        qi_attributes: List of QI attribute names
        use_sampling: Use sampling for large datasets (default True)
        max_sample_size: Maximum sample size for risk calculation (default 100k)
    """
    try:
        # Handle empty dataframe
        if len(df) == 0:
            return {
                'total_groups': 0,
                'unique_individuals': 0,
                'small_groups': 0,
                'unique_risk_pct': 0.0,
                'small_group_risk_pct': 0.0,
                'avg_group_size': 0.0,
                'min_group_size': 0,
                'max_group_size': 0,
            }
        
        # Filter QI attributes that exist in dataframe
        valid_qi = [attr for attr in qi_attributes if attr in df.columns]
        if not valid_qi:
            print("  Warning: No valid QI attributes found in dataframe")
            return {
                'total_groups': 0,
                'unique_individuals': 0,
                'small_groups': 0,
                'unique_risk_pct': 0.0,
                'small_group_risk_pct': 0.0,
                'avg_group_size': 0.0,
                'min_group_size': 0,
                'max_group_size': 0,
            }
        
        # Optimization: Sample for large datasets
        df_calc = df
        if use_sampling and len(df) > max_sample_size:
            print(f"  ⚡ Sampling {max_sample_size:,} rows for re-ID risk calculation (from {len(df):,} total)")
            df_calc = df.sample(n=max_sample_size, random_state=42)
        
        groups = df_calc.groupby(valid_qi, observed=True).size()  # observed=True for categorical optimization

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
            'avg_group_size': round(groups.mean(), 2) if len(groups) > 0 else 0.0,
            'min_group_size': int(groups.min()) if len(groups) > 0 else 0,
            'max_group_size': int(groups.max()) if len(groups) > 0 else 0,
        }
    except Exception as e:
        print(f"  Warning: Failed to calculate re-identification risk: {e}")
        return {
            'total_groups': 0,
            'unique_individuals': 0,
            'small_groups': 0,
            'unique_risk_pct': 0.0,
            'small_group_risk_pct': 0.0,
            'avg_group_size': 0.0,
            'min_group_size': 0,
            'max_group_size': 0,
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
    col_name = 'Unique Change (%)' if 'Unique Change (%)' in info_loss_df.columns else 'Unique Lost (%)'
    avg_info_loss = info_loss_df[col_name].mean()
    avg_kl_div = dist_preserve_df['KL-Divergence'].mean()

    # Combined utility score (0-100, higher = better)
    # Cap between 0 and 100 to handle edge cases
    utility_score = 100 - avg_info_loss
    utility_score = max(0, min(100, utility_score))  # Clamp to [0, 100]

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
