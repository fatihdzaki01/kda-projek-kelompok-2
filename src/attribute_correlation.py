"""
ACE: Attribute Correlation Evaluation (AHP-based).

Implements the Attribute Correlation Evaluation from the paper
(Zhang & Li, 2022, Scientific Reports).

Uses a modified AHP (Analytic Hierarchy Process) with 3-level importance scale
to compute correlation weights between QI attributes and the sensitive attribute.
"""

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif


# Random Index (RI) values for AHP consistency check (n=1..10)
RI_TABLE = {1: 0.0, 2: 0.0, 3: 0.58, 4: 0.90, 5: 1.12,
            6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45, 10: 1.49}


class AttributeCorrelationEvaluation:
    """
    ACE: Compute correlation ranking between QI attributes and sensitive attribute.

    The ranking determines the order of attribute processing in the ACDP Tree.
    Higher weight = stronger correlation → processed first.

    Usage:
        ace = AttributeCorrelationEvaluation()
        ranking = ace.fit(df, qi_attributes, sensitive_attribute)
        # ranking = {'Age': 0.35, 'BMI': 0.28, 'Sex': 0.12, ...}
    """

    def __init__(self):
        self.weights_ = None
        self.ranking_ = None
        self.pairwise_matrix_ = None
        self.consistency_ratio_ = None
        self.nmi_scores_ = None

    def fit(self, df, qi_attributes, sensitive_attribute):
        """
        Compute ACE weights using AHP.

        Steps:
        1. Compute NMI between each QI and sensitive attribute
        2. Build pairwise comparison matrix (3-level scale)
        3. Compute priority weights (geometric mean method)
        4. Check consistency ratio

        Args:
            df (pd.DataFrame): Input data
            qi_attributes (list): List of QI attribute names
            sensitive_attribute (str): Sensitive attribute name

        Returns:
            dict: {attribute: weight} sorted by weight descending
        """

        # Step 1: Compute NMI
        n_samples = len(df)
        n_classes = df[sensitive_attribute].nunique()

        nmi_scores = {}
        for attr in qi_attributes:
            if df[attr].dtype == 'object' or df[attr].dtype.name == 'category':
                le_data = df[attr].astype('category').cat.codes
            else:
                le_data = df[attr].values

            mi = mutual_info_classif(
                le_data.reshape(-1, 1),
                df[sensitive_attribute].values,
                discrete_features=True,
                random_state=42
            )[0]

            # Normalize by entropy of sensitive attribute for NMI
            sens_entropy = self._entropy(df[sensitive_attribute].values)
            nmi = mi / sens_entropy if sens_entropy > 0 else 0.0
            nmi_scores[attr] = max(0.0, nmi)

        self.nmi_scores_ = nmi_scores

        # Step 2: Build pairwise comparison matrix
        n = len(qi_attributes)
        matrix = np.ones((n, n))
        attrs = qi_attributes

        for i in range(n):
            for j in range(i + 1, n):
                val_i = nmi_scores[attrs[i]]
                val_j = nmi_scores[attrs[j]]

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

        # Step 3: Compute priority weights (geometric mean method)
        geometric_means = np.array([
            np.prod(matrix[i, :]) ** (1.0 / n)
            for i in range(n)
        ])
        weights = geometric_means / geometric_means.sum()
        self.weights_ = {attrs[i]: weights[i] for i in range(n)}

        # Step 4: Check consistency
        n_attrs = n
        if n_attrs > 1:
            weighted_sum = matrix @ weights
            lambda_max = np.mean(weighted_sum / weights)
            ci = (lambda_max - n_attrs) / (n_attrs - 1)
            ri = RI_TABLE.get(n_attrs, 1.49)
            cr = ci / ri if ri > 0 else 0.0
            self.consistency_ratio_ = cr

        # Step 5: Sort by weight descending
        sorted_attrs = sorted(self.weights_, key=lambda a: self.weights_[a], reverse=True)
        self.ranking_ = {attr: float(self.weights_[attr]) for attr in sorted_attrs}

        return self.ranking_

    def get_ordered_attributes(self):
        """Get QI attributes ordered by correlation (highest first)."""
        if self.ranking_ is None:
            raise ValueError("ACE has not been fitted yet. Call fit() first.")
        return list(self.ranking_.keys())

    def print_summary(self):
        """Print ACE results."""
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
            cr_status = '✅' if self.consistency_ratio_ < 0.1 else '⚠️'
            print(f'Consistency Ratio (CR): {self.consistency_ratio_:.4f} {cr_status}')
        print('=' * 60)

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    def _map_ratio_to_scale(self, ratio):
        """Map NMI ratio to 3-level AHP scale."""
        if ratio < 0.8:
            return 1
        elif ratio < 1.5:
            return 2
        else:
            return 3

    def _entropy(self, labels):
        """Compute entropy of a label array."""
        _, counts = np.unique(labels, return_counts=True)
        probs = counts / counts.sum()
        return -np.sum(probs * np.log2(probs + 1e-10))
