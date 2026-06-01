"""
ACDP Tree (Generalization Optimizer).

Contains weighted mutual information, inverse frequency weights,
ACDPTreeNode, and ACDPTree classes.
"""

import numpy as np
import pandas as pd
from collections import Counter


def weighted_mutual_info(feature, target, weights):
    """
    Hitung Weighted Mutual Information antara feature dan target.
    Digunakan sebagai split criteria di ACDP Tree.

    Args:
        feature  : pd.Series -> nilai fitur (QI attribute)
        target   : pd.Series -> sensitive attribute
        weights  : pd.Series -> bobot per record

    Returns:
        float -> nilai WMI (makin tinggi = makin baik untuk split)
    """
    total_weight = weights.sum()
    if total_weight == 0:
        return 0.0

    def weighted_entropy(t, w):
        """Hitung entropy berbobot"""
        w_total = w.sum()
        if w_total == 0:
            return 0.0
        classes = t.unique()
        entropy = 0.0
        for c in classes:
            mask = (t == c)
            p = w[mask].sum() / w_total
            if p > 0:
                entropy -= p * np.log2(p)
        return entropy

    h_target = weighted_entropy(target, weights)

    h_conditional = 0.0
    for val in feature.unique():
        mask = (feature == val)
        w_subset = weights[mask]
        t_subset = target[mask]
        p_val = w_subset.sum() / total_weight
        h_conditional += p_val * weighted_entropy(t_subset, w_subset)

    wmi = h_target - h_conditional
    return max(0.0, wmi)


def compute_inverse_frequency_weights(df, target_col):
    """
    Hitung bobot per record menggunakan inverse frequency.
    Kelas minoritas dapat bobot lebih tinggi.

    Formula: weight = total / (n_classes x count_per_class)

    Args:
        df         : pd.DataFrame
        target_col : str -> nama kolom target

    Returns:
        pd.Series -> bobot per record (index sama dengan df)
    """
    counts = Counter(df[target_col])
    total = len(df)
    n_cls = len(counts)

    class_weights = {
        cls: total / (n_cls * cnt)
        for cls, cnt in counts.items()
    }

    weights = df[target_col].map(class_weights)

    print('Inverse Frequency Weights:')
    print('=' * 40)
    for cls, w in sorted(class_weights.items()):
        cnt = counts[cls]
        pct = cnt / total * 100
        print(f'  Kelas {int(cls)}: count={cnt:,} ({pct:.1f}%) -> weight={w:.4f}')
    print('=' * 40)

    return weights


def exponential_mechanism_select(scores, epsilon, sensitivity):
    """
    Select an index using the Exponential Mechanism.

    Args:
        scores (list): Score values for each candidate
        epsilon (float): Privacy budget for this selection
        sensitivity (float): Sensitivity of the score function

    Returns:
        int: Selected index
    """
    if epsilon <= 0 or len(scores) == 0:
        return np.random.randint(len(scores))

    scores = np.array(scores)
    scores = scores - scores.max()

    exp_scores = np.exp(epsilon * scores / (2.0 * max(sensitivity, 1e-10)))
    probs = exp_scores / exp_scores.sum()
    probs = np.nan_to_num(probs, nan=1.0 / len(probs))

    return np.random.choice(len(scores), p=probs)


class ACDPTreeNode:
    """
    Representasi satu node di ACDP Tree.

    Decision Node : memilih attribute & level generalisasi terbaik
    Leaf Node     : menyimpan final generalization levels per record
    """

    def __init__(self):
        self.is_leaf = False
        self.attribute = None
        self.generalization_level = None
        self.children = {}
        self.final_levels = None
        self.record_indices = []
        self.depth = 0


class ACDPTree:
    """
    ACDP Tree: Generalization Optimizer

    Fungsi:
    - Decide attribute mana yang di-generalisasi
    - Decide level berapa untuk setiap attribute
    - Output: per-record generalization decision

    Stopping criteria (mana yang tercapai duluan):
    - Max depth tercapai
    - Semua group dalam node sudah >= k
    """

    def __init__(
        self,
        hierarchy,
        qi_attributes,
        sensitive_attribute,
        k=5,
        max_depth=4,
        weights=None,
        attribute_ranking=None,
        epsilon_tree=None,
        max_tree_depth=None,
    ):
        self.hierarchy = hierarchy
        self.qi_attributes = qi_attributes
        self.sensitive_attribute = sensitive_attribute
        self.k = k
        self.max_depth = max_depth if max_depth else (max_tree_depth or 4)
        self.weights = weights
        self.attribute_ranking = attribute_ranking
        self.epsilon_tree = epsilon_tree
        self.root = None
        self.record_levels = {}

    def _compute_epsilon_level(self, depth, h):
        """Compute privacy budget for a given tree level using arithmetic progression.

        Formula (paper): epsilon_level = epsilon / (h+1) + (h/2 - depth) * d
        where d = 2 * epsilon / (h * (h+1))
        """
        if self.epsilon_tree is None or h <= 0:
            return None
        d = 2.0 * self.epsilon_tree / (h * (h + 1))
        eps = self.epsilon_tree / (h + 1) + (h / 2.0 - depth) * d
        return max(eps, 1e-6)

    def _get_sensitivity(self, df):
        """Compute sensitivity for Exponential Mechanism: log2(n_classes)."""
        n_classes = df[self.sensitive_attribute].nunique()
        return np.log2(max(n_classes, 2))

    def _check_k_satisfied(self, df):
        """Cek apakah semua group di df sudah >= k"""
        if len(df) == 0:
            return True
        groups = df.groupby(self.qi_attributes).size()
        return (groups >= self.k).all()

    def _get_current_generalized(self, df_original, current_levels):
        """Apply current_levels ke df_original untuk dapat nilai tergeneralisasi."""
        df_gen = df_original.copy()
        for attr, level in current_levels.items():
            if level > 0:
                df_gen[attr] = df_original[attr].apply(
                    lambda x: self.hierarchy.generalize(attr, x, level)
                )
        return df_gen

    def _select_best_split(self, df_original, current_levels, available_attrs, epsilon_level):
        """
        Pilih attribute & level terbaik untuk split berikutnya
        menggunakan Exponential Mechanism + Weighted Mutual Information.

        Untuk continuous attributes: Exponential Mechanism memilih split point
        dari candidates dengan probabilitas proportional ke exp(eps * score / (2 * delta)).

        Untuk discrete attributes: WMI langsung digunakan sebagai score.
        """
        candidates = []

        w = self.weights.loc[df_original.index] if self.weights is not None \
            else pd.Series(np.ones(len(df_original)), index=df_original.index)

        target = df_original[self.sensitive_attribute]

        for attr in available_attrs:
            current_level = current_levels.get(attr, 0)
            max_level = self.hierarchy.get_max_level(attr)

            for next_level in range(current_level + 1, max_level + 1):
                gen_values = df_original[attr].apply(
                    lambda x: self.hierarchy.generalize(attr, x, next_level)
                )

                wmi = weighted_mutual_info(gen_values, target, w)

                candidates.append({
                    'attribute': attr,
                    'level': next_level,
                    'wmi': wmi,
                })

        if not candidates:
            return None, None, -1

        # Use attribute_ranking as tiebreaker if available
        if self.attribute_ranking:
            rank_order = {attr: i for i, attr in enumerate(self.attribute_ranking)}
            for c in candidates:
                c['rank'] = rank_order.get(c['attribute'], 999)

        if epsilon_level is not None and epsilon_level > 0:
            # Exponential Mechanism selection
            scores = [c['wmi'] for c in candidates]
            sensitivity = self._get_sensitivity(df_original)
            selected_idx = exponential_mechanism_select(scores, epsilon_level, sensitivity)
            best = candidates[selected_idx]
        else:
            # Deterministic: best WMI, tiebreak by ACE ranking
            best = max(candidates, key=lambda c: (c['wmi'], -c.get('rank', 999)))

        return best['attribute'], best['level'], best['wmi']

    def _build(self, df_original, current_levels, available_attrs, depth, tree_height):
        """
        Rekursif build ACDP Tree.
        """
        node = ACDPTreeNode()
        node.depth = depth
        node.record_indices = list(df_original.index)
        node.final_levels = current_levels.copy()

        df_gen = self._get_current_generalized(df_original, current_levels)

        k_satisfied = self._check_k_satisfied(df_gen)
        max_depth_hit = depth >= self.max_depth
        no_attrs_left = len(available_attrs) == 0

        if k_satisfied or max_depth_hit or no_attrs_left:
            node.is_leaf = True
            node.final_levels = current_levels.copy()
            return node

        # Compute privacy budget for this level
        eps_level = self._compute_epsilon_level(depth, tree_height)

        best_attr, best_level, best_wmi = self._select_best_split(
            df_original, current_levels, available_attrs, eps_level
        )

        if best_attr is None:
            node.is_leaf = True
            node.final_levels = current_levels.copy()
            return node

        node.attribute = best_attr
        node.generalization_level = best_level

        new_levels = current_levels.copy()
        new_levels[best_attr] = best_level

        remaining_attrs = [a for a in available_attrs if a != best_attr] \
            if best_level >= self.hierarchy.get_max_level(best_attr) \
            else available_attrs

        gen_values = df_original[best_attr].apply(
            lambda x: self.hierarchy.generalize(best_attr, x, best_level)
        )

        for val in gen_values.unique():
            mask = (gen_values == val)
            subset = df_original[mask]

            if len(subset) == 0:
                continue

            child_node = self._build(
                subset,
                new_levels,
                remaining_attrs,
                depth + 1,
                tree_height,
            )
            node.children[val] = child_node

        return node

    def _compute_tree_height(self, df):
        """Estimate tree height for budget allocation."""
        if self.max_depth:
            return self.max_depth
        return min(len(self.qi_attributes), 4)

    def fit(self, df):
        """
        Build ACDP Tree dari dataset.
        """
        print('=' * 80)
        print('BUILDING ACDP TREE (Generalization Optimizer)')
        print('=' * 80)
        print(f'  Records      : {len(df):,}')
        print(f'  k-anonymity  : {self.k}')
        print(f'  Max depth    : {self.max_depth}')
        print(f'  QI Attributes: {self.qi_attributes}')
        if self.attribute_ranking:
            print(f'  ACE Ranking  : {list(self.attribute_ranking.keys())}')
        if self.epsilon_tree is not None:
            print(f'  DP budget    : epsilon={self.epsilon_tree:.4f} (tree construction)')
        print('=' * 80)

        initial_levels = {attr: 0 for attr in self.qi_attributes}
        tree_height = self._compute_tree_height(df)

        self.root = self._build(
            df_original=df,
            current_levels=initial_levels,
            available_attrs=self.qi_attributes.copy(),
            depth=0,
            tree_height=tree_height,
        )

        self._collect_record_levels(self.root)

        print(f'\nACDP Tree built successfully!')
        print(f'   Total records mapped: {len(self.record_levels):,}')

        return self

    def _collect_record_levels(self, node):
        """Rekursif collect final levels untuk setiap record"""
        if node is None:
            return

        if node.is_leaf:
            for idx in node.record_indices:
                self.record_levels[idx] = node.final_levels.copy()
            return

        for child in node.children.values():
            self._collect_record_levels(child)

        for idx in node.record_indices:
            if idx not in self.record_levels:
                self.record_levels[idx] = node.final_levels.copy()

    def transform(self, df):
        """
        Apply per-record generalization decisions ke dataset.
        """
        df_result = df.copy()

        for attr in self.qi_attributes:
            if attr in df_result.columns:
                df_result[attr] = df_result[attr].astype(object)

        for idx, levels in self.record_levels.items():
            if idx not in df_result.index:
                continue
            for attr, level in levels.items():
                if level > 0 and attr in df_result.columns:
                    original_val = df.loc[idx, attr]
                    df_result.at[idx, attr] = self.hierarchy.generalize(
                        attr, original_val, level
                    )

        return df_result

    def get_generalization_summary(self):
        """
        Summary statistik generalization levels yang diputuskan tree.
        """
        if not self.record_levels:
            print('Tree belum di-fit!')
            return None

        summary_data = []
        for attr in self.qi_attributes:
            levels = [lvl[attr] for lvl in self.record_levels.values()]
            summary_data.append({
                'Attribute': attr,
                'Min Level': min(levels),
                'Max Level': max(levels),
                'Mean Level': round(np.mean(levels), 2),
                'Level 0 (%)': round(levels.count(0) / len(levels) * 100, 1),
                'Level 1 (%)': round(levels.count(1) / len(levels) * 100, 1),
                'Level 2 (%)': round(levels.count(2) / len(levels) * 100, 1),
                'Level 3 (%)': round(levels.count(3) / len(levels) * 100, 1),
            })

        return pd.DataFrame(summary_data)
