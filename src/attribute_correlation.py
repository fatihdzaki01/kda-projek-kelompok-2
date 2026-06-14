import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from src.utils import ensure_list

RI_TABLE = {1: 0.0, 2: 0.0, 3: 0.58, 4: 0.90, 5: 1.12,
            6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45, 10: 1.49}


class AttributeCorrelationEvaluation:

    def __init__(self):
        self.weights_ = None
        self.ranking_ = None
        self.pairwise_matrix_ = None
        self.consistency_ratio_ = None
        self.nmi_scores_ = None

    def fit(self, df, qi_attributes, sensitive_attribute):
        sens_attrs = ensure_list(sensitive_attribute)

        combined_nmi = {}
        for attr in qi_attributes:
            combined_nmi[attr] = 0.0

        for sens_attr in sens_attrs:
            if sens_attr not in df.columns:
                print(f'  Skipping sensitive attribute "{sens_attr}": not in dataframe')
                continue

            sensitive_values = df[sens_attr].dropna()
            if len(sensitive_values) == 0:
                continue

            n_unique = sensitive_values.nunique()
            is_continuous = (
                pd.api.types.is_float_dtype(sensitive_values) and
                n_unique > 20
            ) or (
                pd.api.types.is_numeric_dtype(sensitive_values) and
                n_unique > len(df) * 0.5
            )

            if is_continuous:
                print(f'  Sensitive "{sens_attr}": continuous (unique: {n_unique}) — mutual_info_regression')
                use_regression = True
                y_data = pd.to_numeric(sensitive_values, errors='coerce').fillna(sensitive_values.mean()).values
            else:
                print(f'  Sensitive "{sens_attr}": discrete (unique: {n_unique}) — mutual_info_classif')
                use_regression = False
                y_data = pd.factorize(sensitive_values)[0]

            for attr in qi_attributes:
                le_data = pd.factorize(df[attr])[0]
                le_data = np.array(le_data).reshape(-1, 1)
                valid_indices = sensitive_values.index
                le_data_aligned = le_data[df.index.isin(valid_indices)]

                if use_regression:
                    mi = mutual_info_regression(
                        le_data_aligned,
                        y_data,
                        random_state=42
                    )[0]
                else:
                    mi = mutual_info_classif(
                        le_data_aligned,
                        y_data,
                        discrete_features=True,
                        random_state=42
                    )[0]

                sens_entropy = self._entropy(y_data)
                nmi = mi / sens_entropy if sens_entropy > 0 else 0.0
                combined_nmi[attr] += max(0.0, nmi)

        n_sens = len(sens_attrs)
        if n_sens > 0:
            for attr in qi_attributes:
                combined_nmi[attr] /= n_sens

        self.nmi_scores_ = combined_nmi

        n = len(qi_attributes)
        matrix = np.ones((n, n))
        attrs = qi_attributes

        for i in range(n):
            for j in range(i + 1, n):
                val_i = combined_nmi[attrs[i]]
                val_j = combined_nmi[attrs[j]]

                if val_j == 0 and val_i == 0:
                    scale = 1
                elif val_j == 0:
                    scale = 3
                elif val_i == 0:
                    scale = 1 / 3
                else:
                    ratio = val_i / val_j
                    scale = self._map_ratio_to_scale(ratio)

                matrix[i][j] = scale
                matrix[j][i] = 1.0 / scale

        self.pairwise_matrix_ = matrix

        geometric_means = np.array([
            np.prod(matrix[i, :]) ** (1.0 / n)
            for i in range(n)
        ])
        weights = geometric_means / geometric_means.sum()
        self.weights_ = {attrs[i]: weights[i] for i in range(n)}

        n_attrs = n
        if n_attrs > 1:
            weighted_sum = matrix @ weights
            lambda_max = np.mean(weighted_sum / weights)
            ci = (lambda_max - n_attrs) / (n_attrs - 1)
            ri = RI_TABLE.get(n_attrs, 1.49)
            cr = ci / ri if ri > 0 else 0.0
            self.consistency_ratio_ = cr

        sorted_attrs = sorted(self.weights_, key=lambda a: self.weights_[a], reverse=True)
        self.ranking_ = {attr: float(self.weights_[attr]) for attr in sorted_attrs}

        return self.ranking_

    def get_ordered_attributes(self):
        if self.ranking_ is None:
            raise ValueError("ACE has not been fitted yet. Call fit() first.")
        return list(self.ranking_.keys())

    def print_summary(self):
        if self.ranking_ is None:
            print("ACE not fitted yet.")
            return

        print('\nATTRIBUTE CORRELATION EVALUATION (ACE)')
        print('=' * 60)
        print(f'{"Rank":<5s} {"Attribute":<20s} {"Weight":<10s} {"NMI":<10s}')
        print('-' * 60)
        for rank, (attr, weight) in enumerate(self.ranking_.items(), 1):
            nmi = self.nmi_scores_.get(attr, 0)
            print(f'{rank:<5d} {attr:<20s} {weight:<10.4f} {nmi:<10.4f}')
        print('-' * 60)
        if self.consistency_ratio_ is not None:
            cr_status = '[OK]' if self.consistency_ratio_ < 0.1 else '[WARN]'
            print(f'Consistency Ratio (CR): {self.consistency_ratio_:.4f} {cr_status}')
        print('=' * 60)

    def _map_ratio_to_scale(self, ratio):
        if ratio < 0.8:
            return 1
        elif ratio < 1.5:
            return 2
        else:
            return 3

    def _entropy(self, labels):
        if len(labels) == 0:
            return 0.0

        if pd.api.types.is_float_dtype(labels) and len(np.unique(labels)) > 20:
            counts, _ = np.histogram(labels, bins=min(50, len(labels) // 10))
            counts = counts[counts > 0]
            probs = counts / counts.sum()
        else:
            _, counts = np.unique(labels, return_counts=True)
            probs = counts / counts.sum()

        return -np.sum(probs * np.log2(probs + 1e-10))
