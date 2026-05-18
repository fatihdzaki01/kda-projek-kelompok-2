"""
ACE (Anonymity-Conscious Extension).

Safety net after ACDP Tree to enforce k-anonymity.
"""

import pandas as pd

from src.config import K_ANONYMITY, QI_ATTRIBUTES
from src.hierarchy import HIERARCHY


class ACE:
    """
    Anonymity-Conscious Extension (ACE)

    Fungsi:
    - Safety net setelah ACDP Tree
    - Enforce k-anonymity yang belum terpenuhi
    - Boleh override keputusan tree (naikan level generalisasi)
    - Selalu generalize dari ORIGINAL values
    """

    def __init__(self, k=K_ANONYMITY, hierarchy=None, max_iterations=20):
        self.k = k
        self.hierarchy = hierarchy if hierarchy else HIERARCHY
        self.max_iterations = max_iterations
        self.current_levels = {}
        self.iteration_log = []

    def _find_violations(self, df):
        """Cari equivalence class yang violate k-anonymity."""
        groups = df.groupby(QI_ATTRIBUTES).size()
        violations = groups[groups < self.k]
        return violations

    def _get_violation_indices(self, df, violations):
        """Dapat index records yang masuk violation groups."""
        violation_indices = []

        for group_key in violations.index:
            if not isinstance(group_key, tuple):
                group_key = (group_key,)

            mask = pd.Series([True] * len(df), index=df.index)
            for attr, val in zip(QI_ATTRIBUTES, group_key):
                mask = mask & (df[attr].astype(str) == str(val))

            violation_indices.extend(df[mask].index.tolist())

        return violation_indices

    def _select_attribute_to_generalize(self, df_original, violation_indices):
        """
        Pilih attribute paling perlu di-generalisasi.

        Strategy: attribute dengan nunique tertinggi di violation records
        (paling bervariasi = paling butuh digeneralisasi)
        """
        violation_df = df_original.loc[violation_indices]

        best_attr = None
        best_variance = -1

        for attr in QI_ATTRIBUTES:
            can_generalize = any(
                self.current_levels.get(idx, {}).get(attr, 0)
                < self.hierarchy.get_max_level(attr)
                for idx in violation_indices
            )

            if not can_generalize:
                continue

            variance = violation_df[attr].nunique()
            if variance > best_variance:
                best_variance = variance
                best_attr = attr

        return best_attr

    def _apply_generalization(self, df_original, df_current,
                              violation_indices, attribute):
        """
        Naikan level generalisasi violation records pada attribute tertentu.
        Selalu dari ORIGINAL values.
        """
        df_result = df_current.copy()

        # Pastikan kolom bisa menerima string values
        if df_result[attribute].dtype != object:
            df_result[attribute] = df_result[attribute].astype(object)

        for idx in violation_indices:
            current_level = self.current_levels.get(idx, {}).get(attribute, 0)
            max_level = self.hierarchy.get_max_level(attribute)

            if current_level >= max_level:
                continue

            new_level = current_level + 1

            if idx not in self.current_levels:
                self.current_levels[idx] = {}
            self.current_levels[idx][attribute] = new_level

            original_val = df_original.loc[idx, attribute]
            df_result.at[idx, attribute] = self.hierarchy.generalize(
                attribute, original_val, new_level
            )

        return df_result

    def enforce_k_anonymity(self, df_original, df_tree_output,
                            tree_record_levels, verbose=True):
        """
        Main ACE: enforce k-anonymity pada output ACDP Tree.

        Args:
            df_original       : DataFrame original
            df_tree_output    : Output dari ACDP Tree
            tree_record_levels: acdp_tree.record_levels (langsung pakai)
            verbose           : Print progress

        Returns:
            pd.DataFrame: k-anonymous dataset
        """
        for idx, levels in tree_record_levels.items():
            self.current_levels[idx] = levels.copy()

        df_current = df_tree_output.copy()

        if verbose:
            print('=' * 80)
            print(f'ACE: Enforcing {self.k}-Anonymity')
            print('=' * 80)

        for iteration in range(self.max_iterations):
            violations = self._find_violations(df_current)
            n_violations = len(violations)
            n_records_vio = int(violations.sum()) if n_violations > 0 else 0

            if verbose:
                print(f'\nIteration {iteration + 1}:')
                print(f'  Violations : {n_violations} groups, '
                      f'{n_records_vio} records')

            if n_violations == 0:
                if verbose:
                    print(f'\n✅ k-anonymity satisfied after {iteration} iterations!')
                break

            violation_indices = self._get_violation_indices(df_current, violations)

            if not violation_indices:
                break

            attr_to_gen = self._select_attribute_to_generalize(
                df_original, violation_indices
            )

            if attr_to_gen is None:
                if verbose:
                    print('\n⚠️  Semua attribute sudah max level.')
                    print(f'   Remaining violations: {n_violations} groups')
                break

            if verbose:
                print(f'  → Generalizing "{attr_to_gen}" '
                      f'for {len(violation_indices)} records')

            df_current = self._apply_generalization(
                df_original, df_current, violation_indices, attr_to_gen
            )

            self.iteration_log.append({
                'iteration': iteration + 1,
                'attribute': attr_to_gen,
                'violation_groups': n_violations,
                'violation_records': n_records_vio,
            })

        if verbose:
            print('=' * 80)

        return df_current

    def check_k_anonymity(self, df, verbose=True):
        """Cek apakah dataset memenuhi k-anonymity."""
        groups = df.groupby(QI_ATTRIBUTES).size()

        result = {
            'satisfies': bool((groups >= self.k).all()),
            'min_group': int(groups.min()),
            'max_group': int(groups.max()),
            'avg_group': round(float(groups.mean()), 2),
            'n_groups': int(len(groups)),
            'n_violations': int((groups < self.k).sum()),
            'n_records_vio': int(groups[groups < self.k].sum()),
        }

        if verbose:
            print('=' * 80)
            print('K-ANONYMITY CHECK')
            print('=' * 80)
            print(f'k                   : {self.k}')
            print(f'Satisfies k-anon    : {result["satisfies"]}')
            print(f'Total groups        : {result["n_groups"]:,}')
            print(f'Min group size      : {result["min_group"]}')
            print(f'Max group size      : {result["max_group"]}')
            print(f'Avg group size      : {result["avg_group"]}')
            if not result['satisfies']:
                print(f'Violations (groups) : {result["n_violations"]:,}')
                print(f'Violations (records): {result["n_records_vio"]:,}')
            print('=' * 80)

        return result
