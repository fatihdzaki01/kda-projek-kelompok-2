import numpy as np
import pandas as pd
from collections import Counter
from src.utils import ensure_list


def _weighted_entropy(series, weights):
    w_total = weights.sum()
    if w_total == 0:
        return 0.0

    n_unique = series.nunique()
    is_continuous = (
        pd.api.types.is_float_dtype(series) and n_unique > 20
    ) or (n_unique > len(series) * 0.5)

    if is_continuous:
        n_bins = min(20, max(5, len(series) // 100))
        try:
            binned = pd.cut(series, bins=n_bins, duplicates='drop')
            grouped = weights.groupby(binned, sort=False).sum()
        except Exception:
            grouped = weights.groupby(series, sort=False).sum()
    else:
        grouped = weights.groupby(series, sort=False).sum()

    p = grouped.values / w_total
    p = p[p > 0]
    return -np.sum(p * np.log2(p))


def weighted_mutual_info(feature, target, weights):
    total_weight = weights.sum()
    if total_weight == 0:
        return 0.0

    h_target = _weighted_entropy(target, weights)

    h_conditional = 0.0
    for val in feature.unique():
        mask = (feature == val)
        w_subset = weights[mask]
        t_subset = target[mask]
        p_val = w_subset.sum() / total_weight
        if p_val > 0:
            h_conditional += p_val * _weighted_entropy(t_subset, w_subset)

    return max(0.0, h_target - h_conditional)


def compute_inverse_frequency_weights(df, target_col):
    target_cols = ensure_list(target_col)
    if not target_cols:
        print('  Warning: No target columns specified, using uniform weights')
        return pd.Series(np.ones(len(df)), index=df.index)

    primary_target = target_cols[0]

    if len(df) == 0:
        return pd.Series([], dtype=float)

    if primary_target not in df.columns:
        print(f'  Warning: Target column "{primary_target}" not found, using uniform weights')
        return pd.Series(np.ones(len(df)), index=df.index)

    if df[primary_target].isna().all():
        print(f'  Warning: All values in "{primary_target}" are NaN, using uniform weights')
        return pd.Series(np.ones(len(df)), index=df.index)

    counts = Counter(df[primary_target].dropna())
    total = len(df)
    n_cls = len(counts)

    if n_cls == 0:
        return pd.Series(np.ones(len(df)), index=df.index)

    class_weights = {
        cls: total / (n_cls * cnt)
        for cls, cnt in counts.items()
    }

    weights = df[primary_target].map(class_weights)
    weights = weights.fillna(1.0)

    print('Inverse Frequency Weights:')
    print('=' * 40)
    for cls, w in sorted(class_weights.items()):
        cnt = counts[cls]
        pct = cnt / total * 100
        print(f'  Kelas {cls}: count={cnt:,} ({pct:.1f}%) -> weight={w:.4f}')
    print('=' * 40)

    return weights


def exponential_mechanism_select(scores, epsilon, sensitivity):
    if epsilon <= 0 or len(scores) == 0:
        return np.random.randint(len(scores))

    scores = np.array(scores)
    scores = scores - scores.max()

    exp_scores = np.exp(epsilon * scores / (2.0 * max(sensitivity, 1e-10)))
    probs = exp_scores / exp_scores.sum()
    probs = np.nan_to_num(probs, nan=1.0 / len(probs))

    return np.random.choice(len(scores), p=probs)


class ACDPTreeNode:

    def __init__(self):
        self.is_leaf = False
        self.attribute = None
        self.generalization_level = None
        self.children = {}
        self.final_levels = None
        self.record_indices = []
        self.depth = 0


class ACDPTree:

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
        self.sensitive_attributes = ensure_list(sensitive_attribute)
        self.sensitive_attribute = self.sensitive_attributes[0] if self.sensitive_attributes else ''
        self.k = k
        self.max_depth = max_depth if max_depth else (max_tree_depth or 4)
        self.weights = weights
        self.attribute_ranking = attribute_ranking
        self.epsilon_tree = epsilon_tree
        self.root = None
        self.record_levels = {}
        self._gen_cache = {}

    def _get_gen_mapping(self, attr, level):
        if level == 0:
            return {}
        key = (attr, level)
        if key in self._gen_cache:
            return self._gen_cache[key]
        h = self.hierarchy.hierarchies.get(attr, {})
        max_lvl = h.get('max_level', 3)
        if level >= max_lvl:
            self._gen_cache[key] = None
            return None
        mapping = h.get('mapping', {}).get(level, {})
        if not mapping:
            self._gen_cache[key] = None
            return None
        self._gen_cache[key] = mapping
        return mapping

    def _generalize_series(self, series, attr, level):
        mapping = self._get_gen_mapping(attr, level)
        if mapping is None:
            return pd.Series('Any', index=series.index)
        return series.map(mapping).fillna('Any')

    def _compute_epsilon_level(self, depth, h):
        if self.epsilon_tree is None or h <= 0:
            return None
        d = 2.0 * self.epsilon_tree / (h * (h + 1))
        eps = self.epsilon_tree / (h + 1) + (h / 2.0 - depth) * d
        return max(eps, 1e-6)

    def _get_sensitivity(self, df):
        if self.sensitive_attribute and self.sensitive_attribute in df.columns:
            n_classes = df[self.sensitive_attribute].nunique()
        else:
            n_classes = 2
        return np.log2(max(n_classes, 2))

    def _check_k_satisfied(self, df):
        if len(df) == 0:
            return True
        groups = df.groupby(self.qi_attributes).size()
        return (groups >= self.k).all()

    def _get_current_generalized(self, df_original, current_levels):
        df_gen = df_original.copy()
        for attr, level in current_levels.items():
            if level > 0:
                df_gen[attr] = self._generalize_series(df_original[attr], attr, level).astype(object)
        return df_gen

    def _select_best_split(self, df_original, current_levels, available_attrs, epsilon_level):
        candidates = []

        w = self.weights.loc[df_original.index] if self.weights is not None \
            else pd.Series(np.ones(len(df_original)), index=df_original.index)

        target = df_original[self.sensitive_attribute] if self.sensitive_attribute and self.sensitive_attribute in df_original.columns \
            else pd.Series(np.zeros(len(df_original)), index=df_original.index)

        for attr in available_attrs:
            current_level = current_levels.get(attr, 0)
            max_level = self.hierarchy.get_max_level(attr)

            for next_level in range(current_level + 1, max_level + 1):
                gen_values = self._generalize_series(df_original[attr], attr, next_level)

                wmi = weighted_mutual_info(gen_values, target, w)

                candidates.append({
                    'attribute': attr,
                    'level': next_level,
                    'wmi': wmi,
                })

        if not candidates:
            return None, None, -1

        if self.attribute_ranking:
            rank_order = {attr: i for i, attr in enumerate(self.attribute_ranking)}
            for c in candidates:
                c['rank'] = rank_order.get(c['attribute'], 999)

        if epsilon_level is not None and epsilon_level > 0:
            scores = [c['wmi'] for c in candidates]
            sensitivity = self._get_sensitivity(df_original)
            selected_idx = exponential_mechanism_select(scores, epsilon_level, sensitivity)
            best = candidates[selected_idx]
        else:
            best = max(candidates, key=lambda c: (c['wmi'], -c.get('rank', 999)))

        return best['attribute'], best['level'], best['wmi']

    def _build(self, df_original, current_levels, available_attrs, depth, tree_height):
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

        gen_values = self._generalize_series(df_original[best_attr], best_attr, best_level)

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
        if self.max_depth:
            return self.max_depth
        return min(len(self.qi_attributes), 4)

    def fit(self, df):
        print('=' * 80)
        print('BUILDING ACDP TREE (Generalization Optimizer)')
        print('=' * 80)
        print(f'  Records      : {len(df):,}')
        print(f'  k-anonymity  : {self.k}')
        print(f'  Max depth    : {self.max_depth}')
        print(f'  QI Attributes: {self.qi_attributes}')
        print(f'  Sensitive Attr(s): {self.sensitive_attributes}')
        if self.attribute_ranking:
            print(f'  ACE Ranking  : {list(self.attribute_ranking.keys())}')
        if self.epsilon_tree is not None:
            print(f'  DP budget    : epsilon={self.epsilon_tree:.4f} (tree construction)')
        print('=' * 80)

        self._gen_cache = {}

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

    def export_tree_structure(self):
        def node_to_dict(node, depth=0):
            if node is None:
                return None

            node_dict = {
                'depth': depth,
                'is_leaf': node.is_leaf,
                'record_count': len(node.record_indices),
                'attribute': node.attribute,
                'generalization_level': node.generalization_level,
                'final_levels': node.final_levels if node.is_leaf else None,
                'children': []
            }

            if not node.is_leaf and node.children:
                for value, child_node in node.children.items():
                    child_dict = node_to_dict(child_node, depth + 1)
                    if child_dict:
                        child_dict['parent_value'] = str(value)
                        node_dict['children'].append(child_dict)

            return node_dict

        if self.root is None:
            return None

        tree_structure = {
            'metadata': {
                'k_anonymity': self.k,
                'max_depth': self.max_depth,
                'qi_attributes': self.qi_attributes,
                'sensitive_attributes': self.sensitive_attributes,
                'total_records': len(self.record_levels)
            },
            'tree': node_to_dict(self.root)
        }

        return tree_structure
