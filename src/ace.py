"""
K-Anonymity Enforcer (formerly ACE).

Safety net after ACDP Tree to enforce k-anonymity by iteratively
increasing generalization levels for violation groups.
"""

import pandas as pd


class KAnonymityEnforcer:
    """
    Enforce k-anonymity on ACDP Tree output.

    Fungsi:
    - Safety net setelah ACDP Tree
    - Enforce k-anonymity yang belum terpenuhi
    - Boleh override keputusan tree (naikan level generalisasi)
    - Selalu generalize dari ORIGINAL values
    """

    def __init__(self, k=5, hierarchy=None, qi_attributes=None, max_iterations=20, use_fast_mode=True):
        self.k = k
        self.hierarchy = hierarchy
        self.qi_attributes = qi_attributes or []
        self.max_iterations = max_iterations
        self.use_fast_mode = use_fast_mode  # Enable optimization
        self.current_levels = {}
        self.iteration_log = []
        self._group_cache = {}  # Cache for groupby results

    def _find_violations(self, df):
        """Cari equivalence class yang violate k-anonymity."""
        if self.use_fast_mode:
            # Fast mode: use observed=True for categorical optimization
            groups = df.groupby(self.qi_attributes, observed=True, sort=False).size()
        else:
            groups = df.groupby(self.qi_attributes).size()
        
        violations = groups[groups < self.k]
        return violations

    def _get_violation_indices(self, df, violations):
        """Dapat index records yang masuk violation groups (OPTIMIZED)."""
        if self.use_fast_mode and len(violations) > 0:
            # Fast mode: vectorized approach using merge
            violation_keys = pd.DataFrame(
                [k if isinstance(k, tuple) else (k,) for k in violations.index],
                columns=self.qi_attributes
            )
            
            # Create temporary join key
            df_temp = df[self.qi_attributes].copy()
            df_temp['_idx'] = df.index
            
            # Merge to find violation indices
            merged = df_temp.merge(
                violation_keys,
                on=self.qi_attributes,
                how='inner'
            )
            
            return merged['_idx'].tolist()
        else:
            # Original slow approach (fallback)
            violation_indices = []

            for group_key in violations.index:
                if not isinstance(group_key, tuple):
                    group_key = (group_key,)

                mask = pd.Series([True] * len(df), index=df.index)
                for attr, val in zip(self.qi_attributes, group_key):
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

        for attr in self.qi_attributes:
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
        Selalu dari ORIGINAL values. (OPTIMIZED for vectorization)
        """
        df_result = df_current.copy()

        if df_result[attribute].dtype != object:
            df_result[attribute] = df_result[attribute].astype(object)

        # Batch processing for speed
        if self.use_fast_mode and len(violation_indices) > 100:
            # Vectorized batch update
            updates = []
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
                generalized_val = self.hierarchy.generalize(
                    attribute, original_val, new_level
                )
                updates.append((idx, generalized_val))
            
            # Batch update
            if updates:
                indices, values = zip(*updates)
                df_result.loc[list(indices), attribute] = list(values)
        else:
            # Original approach for small batches
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
        Main enforcement: enforce k-anonymity pada output ACDP Tree.

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
            print(f'K-ANONYMITY ENFORCER: Enforcing {self.k}-Anonymity')
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
                    print(f'\nk-anonymity satisfied after {iteration} iterations!')
                break

            violation_indices = self._get_violation_indices(df_current, violations)

            if not violation_indices:
                break

            attr_to_gen = self._select_attribute_to_generalize(
                df_original, violation_indices
            )

            if attr_to_gen is None:
                if verbose:
                    print('\nSemua attribute sudah max level.')
                    print(f'   Remaining violations: {n_violations} groups')
                break

            if verbose:
                print(f'  -> Generalizing "{attr_to_gen}" '
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

        # Fallback: if violations remain after all iterations,
        # force remaining violation records to max level for ALL attributes
        if n_violations > 0:
            if verbose:
                print(f'\nFallback: forcing remaining {n_records_vio} records to max level')

            for attr in self.qi_attributes:
                max_lvl = self.hierarchy.get_max_level(attr)
                for idx in violation_indices:
                    current_level = self.current_levels.get(idx, {}).get(attr, 0)
                    if current_level < max_lvl:
                        if idx not in self.current_levels:
                            self.current_levels[idx] = {}
                        self.current_levels[idx][attr] = max_lvl
                        original_val = df_original.loc[idx, attr]
                        df_current.at[idx, attr] = self.hierarchy.generalize(
                            attr, original_val, max_lvl
                        )

            # Re-check
            final_violations = self._find_violations(df_current)
            if verbose:
                n_final = len(final_violations)
                print(f'  Remaining violations after fallback: {n_final} groups')
                if n_final == 0:
                    print('  k-anonymity satisfied!')

        if verbose:
            print('=' * 80)

        return df_current

    def check_k_anonymity(self, df, verbose=True):
        """Cek apakah dataset memenuhi k-anonymity."""
        groups = df.groupby(self.qi_attributes).size()

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
