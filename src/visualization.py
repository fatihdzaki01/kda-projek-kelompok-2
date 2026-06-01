"""
Visualization functions for ACDP Tree pipeline.

Contains tree visualization (Graphviz) and all plotting functions.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from graphviz import Digraph

from src.config import QI_ATTRIBUTES, SENSITIVE_ATTRIBUTE, K_ANONYMITY, EPSILON


def visualize_acdp_tree(node, max_display_depth=3):
    """
    Visualisasi ACDP Tree menggunakan Graphviz.
    Setiap node menampilkan:
    - Attribute yang di-generalisasi
    - Level generalisasi
    - Jumlah records
    """
    dot = Digraph(comment='ACDP Tree - Generalization Optimizer')
    dot.attr(rankdir='TB')
    dot.attr('node', fontsize='10')

    counter = [0]

    def add_node(n, parent_id=None, edge_label=None):
        if n is None:
            return

        node_id = str(counter[0])
        counter[0] += 1

        n_records = len(n.record_indices)

        if n.is_leaf:
            levels_str = '\n'.join(
                f'{a}:L{l}' for a, l in n.final_levels.items() if l > 0
            ) or 'No generalization'
            label = f'LEAF\n{levels_str}\nn={n_records}'
            fillcolor = '#90EE90'
            shape = 'box'
        else:
            label = (f'Split: {n.attribute}\n'
                     f'→ Level {n.generalization_level}\n'
                     f'n={n_records}')
            fillcolor = '#87CEEB'
            shape = 'ellipse'

        dot.node(node_id, label,
                 shape=shape, style='filled',
                 fillcolor=fillcolor)

        if parent_id is not None:
            dot.edge(parent_id, node_id, label=str(edge_label)[:15])

        if n.depth < max_display_depth:
            for val, child in n.children.items():
                add_node(child, node_id, val)

    add_node(node)

    dot.attr(dpi='200')
    dot.attr(size='20,20')

    return dot


def plot_generalization_summary(acdp_tree):
    """Plot distribusi generalization level per attribute (Section 11.5)."""
    fig, axes = plt.subplots(1, len(QI_ATTRIBUTES), figsize=(18, 4))

    for i, attr in enumerate(QI_ATTRIBUTES):
        levels = [lvl[attr] for lvl in acdp_tree.record_levels.values()]
        level_counts = Counter(levels)

        x = sorted(level_counts.keys())
        y = [level_counts[l] for l in x]

        axes[i].bar(x, y, color='steelblue', edgecolor='white')
        axes[i].set_title(attr, fontweight='bold')
        axes[i].set_xlabel('Level')
        axes[i].set_ylabel('Records')
        axes[i].set_xticks(x)

    plt.suptitle('Distribusi Generalization Level per Attribute (ACDP Tree)',
                 fontweight='bold', fontsize=13)
    plt.tight_layout()
    plt.show()


def plot_ace_iteration_log(ace):
    """Plot ACE iteration log (Section 12.3)."""
    if not ace.iteration_log:
        print('✅ Tidak ada iterasi ACE — k-anonymity sudah terpenuhi dari ACDP Tree!')
        return

    log_df = pd.DataFrame(ace.iteration_log)

    print('ACE ITERATION LOG:')
    print(log_df.to_string(index=False))
    print()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(
        log_df['iteration'],
        log_df['violation_groups'],
        marker='o', linewidth=2,
        markersize=6, color='#e74c3c'
    )
    axes[0].set_xlabel('Iteration', fontweight='bold')
    axes[0].set_ylabel('Violation Groups', fontweight='bold')
    axes[0].set_title('ACE: Violation Reduction', fontweight='bold')
    axes[0].axhline(y=0, color='green', linestyle='--',
                    alpha=0.5, label='Target (0 violations)')
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    attr_counts = log_df['attribute'].value_counts()
    axes[1].bar(
        attr_counts.index,
        attr_counts.values,
        color='steelblue', edgecolor='white'
    )
    axes[1].set_xlabel('Attribute', fontweight='bold')
    axes[1].set_ylabel('Times Generalized', fontweight='bold')
    axes[1].set_title('ACE: Most Generalized Attributes', fontweight='bold')
    axes[1].grid(alpha=0.3, axis='y')

    plt.tight_layout()
    plt.show()


def plot_noise_impact(df_noisy):
    """Plot noise impact visualization (Section 13.4)."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    max_val = max(df_noisy['count'].max(), df_noisy['noisy_count'].max())
    axes[0].scatter(
        df_noisy['count'], df_noisy['noisy_count'],
        alpha=0.4, s=20, color='steelblue'
    )
    axes[0].plot(
        [0, max_val], [0, max_val],
        'r--', alpha=0.6, label='Perfect match'
    )
    axes[0].set_xlabel('Original Count', fontweight='bold')
    axes[0].set_ylabel('Noisy Count', fontweight='bold')
    axes[0].set_title(f'Original vs Noisy Counts (ε={EPSILON})', fontweight='bold')
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].hist(
        df_noisy['noise_added'],
        bins=40, color='coral',
        edgecolor='white', alpha=0.8
    )
    axes[1].axvline(x=0, color='red', linestyle='--',
                    linewidth=2, label='No noise')
    axes[1].axvline(
        x=df_noisy['noise_added'].mean(),
        color='blue', linestyle='--',
        linewidth=1.5,
        label=f'Mean={df_noisy["noise_added"].mean():.2f}'
    )
    axes[1].set_xlabel('Noise Added', fontweight='bold')
    axes[1].set_ylabel('Frequency', fontweight='bold')
    axes[1].set_title('Distribusi Laplace Noise', fontweight='bold')
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    axes[2].hist(
        df_noisy['percent_error'].dropna(),
        bins=40, color='mediumseagreen',
        edgecolor='white', alpha=0.8
    )
    axes[2].axvline(
        x=df_noisy['percent_error'].mean(),
        color='red', linestyle='--',
        linewidth=1.5,
        label=f'Mean={df_noisy["percent_error"].mean():.2f}%'
    )
    axes[2].set_xlabel('Percent Error (%)', fontweight='bold')
    axes[2].set_ylabel('Frequency', fontweight='bold')
    axes[2].set_title('Distribusi Relative Error', fontweight='bold')
    axes[2].legend()
    axes[2].grid(alpha=0.3)

    plt.suptitle(
        f'Differential Privacy: Laplace Noise Impact (ε={EPSILON})',
        fontweight='bold', fontsize=13
    )
    plt.tight_layout()
    plt.show()


def plot_information_loss(info_loss_df):
    """Plot information loss per attribute (Section 15.1)."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].barh(info_loss_df['Attribute'], info_loss_df['Unique Lost (%)'],
                 color='coral', edgecolor='white')
    axes[0].set_xlabel('Unique Values Lost (%)', fontweight='bold')
    axes[0].set_title('Information Loss per Attribute', fontweight='bold')
    axes[0].grid(alpha=0.3, axis='x')

    axes[1].barh(info_loss_df['Attribute'], info_loss_df['Entropy Reduction (%)'],
                 color='steelblue', edgecolor='white')
    axes[1].set_xlabel('Entropy Reduction (%)', fontweight='bold')
    axes[1].set_title('Entropy Reduction per Attribute', fontweight='bold')
    axes[1].grid(alpha=0.3, axis='x')

    plt.tight_layout()
    plt.show()


def plot_distribution_preservation(dist_preserve_df):
    """Plot distribution preservation (Section 15.2)."""
    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(dist_preserve_df))
    width = 0.35

    ax.bar(x - width / 2, dist_preserve_df['KL-Divergence'],
           width, label='KL-Divergence', color='coral', edgecolor='white')
    ax.bar(x + width / 2, dist_preserve_df['TVD'],
           width, label='Total Variation Distance', color='steelblue', edgecolor='white')

    ax.set_xlabel('Attribute', fontweight='bold')
    ax.set_ylabel('Distance', fontweight='bold')
    ax.set_title('Distribution Preservation (Lower = Better)', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(dist_preserve_df['Attribute'])
    ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Good/Fair threshold')
    ax.legend()
    ax.grid(alpha=0.3, axis='y')

    plt.tight_layout()
    plt.show()


def plot_reidentification_risk(df_input, df_final, orig_risk, anon_risk):
    """Plot re-identification risk (Section 15.3)."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    categories = ['Unique\nIndividuals (%)', 'Small Groups\n(< k=5) (%)']
    original_vals = [orig_risk['unique_risk_pct'], orig_risk['small_group_risk_pct']]
    anonymized_vals = [anon_risk['unique_risk_pct'], anon_risk['small_group_risk_pct']]

    x = np.arange(len(categories))
    width = 0.35

    axes[0].bar(x - width / 2, original_vals, width,
                label='Original', color='#e74c3c', edgecolor='white')
    axes[0].bar(x + width / 2, anonymized_vals, width,
                label='Anonymized', color='#2ecc71', edgecolor='white')
    axes[0].set_ylabel('Percentage (%)', fontweight='bold')
    axes[0].set_title('Re-identification Risk (Lower = Better)', fontweight='bold')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(categories)
    axes[0].legend()
    axes[0].grid(alpha=0.3, axis='y')

    group_sizes_orig = df_input.groupby(QI_ATTRIBUTES).size()
    group_sizes_anon = df_final.groupby(QI_ATTRIBUTES).size()

    axes[1].hist([group_sizes_orig, group_sizes_anon],
                 bins=30, label=['Original', 'Anonymized'],
                 color=['#e74c3c', '#2ecc71'], alpha=0.7, edgecolor='white')
    axes[1].axvline(x=K_ANONYMITY, color='blue', linestyle='--',
                    linewidth=2, label=f'k={K_ANONYMITY} threshold')
    axes[1].set_xlabel('Group Size', fontweight='bold')
    axes[1].set_ylabel('Frequency', fontweight='bold')
    axes[1].set_title('Group Size Distribution', fontweight='bold')
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    axes[1].set_yscale('log')

    plt.tight_layout()
    plt.show()


def plot_privacy_utility_tradeoff(tradeoff):
    """Plot privacy-utility tradeoff (Section 15.4)."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    metrics = ['Privacy\nGain', 'Utility\nLoss']
    values = [tradeoff['privacy_gain_pct'], tradeoff['utility_loss_pct']]
    colors = ['#2ecc71', '#e74c3c']

    axes[0].bar(metrics, values, color=colors, edgecolor='white', width=0.6)
    axes[0].set_ylabel('Percentage (%)', fontweight='bold')
    axes[0].set_title('Privacy Gain vs Utility Loss', fontweight='bold')
    axes[0].grid(alpha=0.3, axis='y')

    for i, v in enumerate(values):
        axes[0].text(i, v + 2, f'{v:.1f}%', ha='center',
                     fontweight='bold', fontsize=12)

    axes[1].scatter([tradeoff['utility_loss_pct']],
                    [tradeoff['privacy_gain_pct']],
                    s=300, color='steelblue', edgecolor='white',
                    linewidth=2, zorder=3)
    axes[1].annotate(f'Our Result\n(k={K_ANONYMITY}, ε={EPSILON})',
                     xy=(tradeoff['utility_loss_pct'], tradeoff['privacy_gain_pct']),
                     xytext=(tradeoff['utility_loss_pct'] + 5, tradeoff['privacy_gain_pct'] - 5),
                     fontsize=10, fontweight='bold',
                     arrowprops=dict(arrowstyle='->', color='black', lw=1.5))

    axes[1].axhline(y=50, color='green', linestyle='--',
                    alpha=0.3, label='High privacy gain')
    axes[1].axvline(x=50, color='red', linestyle='--',
                    alpha=0.3, label='High utility loss')
    axes[1].fill_between([0, 50], 50, 100, alpha=0.1, color='green',
                         label='Ideal zone')

    axes[1].set_xlabel('Utility Loss (%)', fontweight='bold')
    axes[1].set_ylabel('Privacy Gain (%)', fontweight='bold')
    axes[1].set_title('Privacy-Utility Trade-off Space', fontweight='bold')
    axes[1].set_xlim(0, 100)
    axes[1].set_ylim(0, 100)
    axes[1].legend(loc='lower right')
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_sensitive_distribution(df_input, df_final, orig_sens_dist, anon_sens_dist):
    """Plot sensitive attribute distribution (Section 15.5)."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    x = np.arange(len(orig_sens_dist))
    width = 0.35

    axes[0].bar(x - width / 2, orig_sens_dist.values, width,
                label='Original', color='#3498db', edgecolor='white')
    axes[0].bar(x + width / 2, anon_sens_dist.values, width,
                label='Anonymized', color='#e74c3c', edgecolor='white')
    axes[0].set_xlabel('Diabetes Class', fontweight='bold')
    axes[0].set_ylabel('Percentage (%)', fontweight='bold')
    axes[0].set_title(f'{SENSITIVE_ATTRIBUTE} Distribution', fontweight='bold')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(['No Diabetes (0)', 'Prediabetes (1)', 'Diabetes (2)'])
    axes[0].legend()
    axes[0].grid(alpha=0.3, axis='y')

    labels = ['No Diabetes', 'Prediabetes', 'Diabetes']
    colors_pie = ['#2ecc71', '#f39c12', '#e74c3c']

    axes[1].pie([orig_sens_dist.get(0, 0), orig_sens_dist.get(1, 0), orig_sens_dist.get(2, 0)],
                labels=labels, autopct='%1.1f%%', colors=colors_pie,
                startangle=90, wedgeprops={'edgecolor': 'white', 'linewidth': 2})
    axes[1].set_title('Anonymized Distribution', fontweight='bold')

    plt.tight_layout()
    plt.show()
