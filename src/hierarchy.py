"""
Generic Generalization Hierarchy for any dataset.

Auto-builds multi-level generalization hierarchies based on column type.
"""

import numpy as np
import pandas as pd

from src.utils import detect_column_type


class GenericGeneralizationHierarchy:
    """
    Auto-build multi-level generalization hierarchy for all QI attributes.

    Usage:
        hierarchy = GenericGeneralizationHierarchy()
        hierarchy.build_from_dataframe(df, qi_attributes, hierarchy_config)

        val = hierarchy.generalize('Age', 5, level=1)
        max_lvl = hierarchy.get_max_level('Age')
    """

    def __init__(self, custom_hierarchy=None):
        self.hierarchies = {}
        self.column_types = {}
        self.max_levels = {}
        self.custom_hierarchy = custom_hierarchy or {}

    def build_from_dataframe(self, df, qi_attributes, hierarchy_config=None):
        """Auto-detect types and build hierarchy for each QI attribute."""
        if hierarchy_config is None:
            hierarchy_config = {}

        for attr in qi_attributes:
            if attr not in df.columns:
                continue

            if attr in self.custom_hierarchy:
                cust = self.custom_hierarchy[attr]
                self.hierarchies[attr] = cust
                self.column_types[attr] = cust.get('type', 'categorical_nominal')
                self.max_levels[attr] = cust.get('max_level', 3)
                continue

            col_type = detect_column_type(df[attr])
            self.column_types[attr] = col_type

            if col_type == 'numerical_continuous':
                self._build_continuous_hierarchy(df[attr], attr, hierarchy_config)
            elif col_type == 'numerical_ordinal':
                self._build_ordinal_hierarchy(df[attr], attr, hierarchy_config)
            elif col_type == 'categorical_binary':
                self._build_binary_hierarchy(df[attr], attr)
            elif col_type == 'categorical_nominal':
                self._build_nominal_hierarchy(df[attr], attr, hierarchy_config)
            elif col_type == 'datetime':
                self._build_datetime_hierarchy(df[attr], attr, hierarchy_config)
            elif col_type == 'text':
                self._build_text_hierarchy(df[attr], attr, hierarchy_config)
            else:
                # Fallback: treat as nominal
                self._build_nominal_hierarchy(df[attr], attr, hierarchy_config)

    # ------------------------------------------------------------------ #
    # Hierarchy builders
    # ------------------------------------------------------------------ #

    def _build_continuous_hierarchy(self, series, attr, config):
        """Numerical continuous: quantile-based binning."""
        n_bins_1 = config.get('n_bins_level1', 4)
        n_bins_2 = config.get('n_bins_level2', 2)

        values = series.dropna().unique()
        values_sorted = np.sort(values)

        # Level 1 bins
        boundaries_1 = self._quantile_boundaries(series, n_bins_1)
        labels_1 = [f'B{i+1}' for i in range(n_bins_1)]

        # Level 2 bins (broader)
        boundaries_2 = self._quantile_boundaries(series, n_bins_2)
        labels_2 = [f'G{i+1}' for i in range(n_bins_2)]

        mapping_1 = self._apply_bins(values_sorted, boundaries_1, labels_1)
        mapping_2 = self._apply_bins(values_sorted, boundaries_2, labels_2)

        self.hierarchies[attr] = {
            'type': 'numerical_continuous',
            'mapping': {1: mapping_1, 2: mapping_2},
            'boundaries': {1: boundaries_1, 2: boundaries_2},
            'labels': {1: labels_1, 2: labels_2},
            'max_level': 3,
        }
        self.max_levels[attr] = 3

    def _build_ordinal_hierarchy(self, series, attr, config):
        """Numerical ordinal: group consecutive values."""
        group_size = config.get('ordinal_group_size', 4)
        unique_vals = sorted(series.dropna().unique())
        n = len(unique_vals)

        if n <= group_size:
            self._build_binary_hierarchy(series, attr)
            return

        mapping_1 = {}
        for i, val in enumerate(unique_vals):
            g = i // group_size
            mapping_1[val] = f'G{g+1}'

        # Level 2: merge level-1 groups
        l1_groups = sorted(set(mapping_1.values()))
        n_l1 = len(l1_groups)
        l2_group_size = max(1, n_l1 // 2)

        mapping_2 = {}
        for val in unique_vals:
            g1 = mapping_1[val]
            g1_idx = l1_groups.index(g1)
            g2 = g1_idx // l2_group_size
            mapping_2[val] = f'SG{g2+1}'

        self.hierarchies[attr] = {
            'type': 'numerical_ordinal',
            'mapping': {1: mapping_1, 2: mapping_2},
            'max_level': 3,
        }
        self.max_levels[attr] = 3

    def _build_binary_hierarchy(self, series, attr):
        """Categorical binary: level 1 = string labels, level 2 = Any."""
        unique_vals = sorted(series.dropna().unique())

        mapping_1 = {val: str(val) for val in unique_vals}

        self.hierarchies[attr] = {
            'type': 'categorical_binary',
            'mapping': {1: mapping_1},
            'max_level': 2,
        }
        self.max_levels[attr] = 2

    def _build_nominal_hierarchy(self, series, attr, config):
        """Categorical nominal: frequency-based grouping."""
        top_k = config.get('top_k_frequent', 10)
        value_counts = series.value_counts()
        n_unique = len(value_counts)

        # Level 1: top-k frequent kept, rest → 'Other'
        top_values = set(value_counts.head(top_k).index)
        mapping_1 = {}
        for val in series.dropna().unique():
            mapping_1[val] = str(val) if val in top_values else 'Other'

        # Level 2: if many groups, merge further
        l1_distinct = set(mapping_1.values())
        if len(l1_distinct) > 5:
            mapping_2 = {val: ('Frequent' if mapping_1[val] != 'Other' else 'Other')
                         for val in mapping_1}
        else:
            mapping_2 = {val: 'Any' for val in mapping_1}

        self.hierarchies[attr] = {
            'type': 'categorical_nominal',
            'mapping': {1: mapping_1, 2: mapping_2},
            'max_level': 3,
        }
        self.max_levels[attr] = 3

    def _build_datetime_hierarchy(self, series, attr, config):
        """Datetime: Year -> Year-Quarter -> Year-Month."""
        try:
            dt_series = pd.to_datetime(series, errors='coerce')
            unique_dates = dt_series.dropna().unique()
            
            # Level 1: Year-Month (e.g., "2023-01")
            mapping_1 = {}
            for dt in unique_dates:
                dt_obj = pd.Timestamp(dt)
                mapping_1[dt] = f"{dt_obj.year}-{dt_obj.month:02d}"
            
            # Level 2: Year-Quarter (e.g., "2023-Q1")
            mapping_2 = {}
            for dt in unique_dates:
                dt_obj = pd.Timestamp(dt)
                quarter = (dt_obj.month - 1) // 3 + 1
                mapping_2[dt] = f"{dt_obj.year}-Q{quarter}"
            
            # Also map original string values if they exist
            for orig_val in series.dropna().unique():
                if orig_val not in mapping_1:
                    try:
                        dt_obj = pd.Timestamp(orig_val)
                        mapping_1[orig_val] = f"{dt_obj.year}-{dt_obj.month:02d}"
                        quarter = (dt_obj.month - 1) // 3 + 1
                        mapping_2[orig_val] = f"{dt_obj.year}-Q{quarter}"
                    except:
                        mapping_1[orig_val] = "Unknown"
                        mapping_2[orig_val] = "Unknown"
            
            self.hierarchies[attr] = {
                'type': 'datetime',
                'mapping': {1: mapping_1, 2: mapping_2},
                'max_level': 3,
            }
            self.max_levels[attr] = 3
        except Exception as e:
            print(f"  Warning: Failed to build datetime hierarchy for '{attr}': {e}")
            # Fallback to nominal
            self._build_nominal_hierarchy(series, attr, config)

    def _build_text_hierarchy(self, series, attr, config):
        """Text: Length-based or keyword-based generalization."""
        try:
            # Level 1: Short/Medium/Long based on character count
            mapping_1 = {}
            for val in series.dropna().unique():
                text_len = len(str(val))
                if text_len < 20:
                    mapping_1[val] = "Short"
                elif text_len < 100:
                    mapping_1[val] = "Medium"
                else:
                    mapping_1[val] = "Long"
            
            # Level 2: Just "Text"
            mapping_2 = {val: "Text" for val in mapping_1}
            
            self.hierarchies[attr] = {
                'type': 'text',
                'mapping': {1: mapping_1, 2: mapping_2},
                'max_level': 3,
            }
            self.max_levels[attr] = 3
        except Exception as e:
            print(f"  Warning: Failed to build text hierarchy for '{attr}': {e}")
            # Fallback to nominal
            self._build_nominal_hierarchy(series, attr, config)

    # ------------------------------------------------------------------ #
    # Generalize
    # ------------------------------------------------------------------ #

    def generalize(self, attribute, value, level):
        """Generalize a value to the specified level."""
        if level == 0:
            return value

        if attribute not in self.hierarchies:
            raise ValueError(f"Unknown attribute: {attribute}")

        h = self.hierarchies[attribute]
        max_lvl = h['max_level']

        if level >= max_lvl:
            return 'Any'

        mapping = h['mapping'].get(level, {})
        if value in mapping:
            return mapping[value]

        # If value not in mapping (e.g. already generalized),
        # return 'Any' for safety
        return 'Any'

    def get_max_level(self, attribute):
        """Get maximum generalization level for attribute."""
        return self.max_levels.get(attribute, 3)

    def apply_to_dataframe(self, df, levels_dict):
        """Apply generalization to entire dataframe."""
        df_gen = df.copy()
        for attr, level in levels_dict.items():
            if attr in df_gen.columns and level > 0:
                df_gen[attr] = df_gen[attr].apply(
                    lambda x: self.generalize(attr, x, level)
                )
        return df_gen

    def print_summary(self):
        """Print summary of built hierarchies."""
        print('\nGENERALIZATION HIERARCHY SUMMARY')
        print('=' * 60)
        for attr, h in self.hierarchies.items():
            col_type = self.column_types.get(attr, 'unknown')
            max_lvl = self.max_levels.get(attr, 3)
            n_levels = len(h['mapping'])
            print(f'  {attr:15s} | type={col_type:25s} | max_level={max_lvl} | levels={n_levels}')
        print('=' * 60)

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    def _quantile_boundaries(self, series, n_bins):
        """Calculate quantile-based bin boundaries."""
        if n_bins <= 1:
            return []
        quantiles = np.linspace(0, 1, n_bins + 1)[1:-1]
        boundaries = series.dropna().quantile(quantiles).tolist()
        # Deduplicate
        boundaries = sorted(set(boundaries))
        return boundaries

    def _apply_bins(self, values, boundaries, labels):
        """Assign each value to a bin based on boundaries."""
        mapping = {}
        for v in values:
            assigned = False
            for i, b in enumerate(boundaries):
                if v <= b:
                    mapping[v] = labels[i]
                    assigned = True
                    break
            if not assigned:
                mapping[v] = labels[-1]
        return mapping
