"""
Differential Privacy: Laplace Noise and Privacy Budget Tracker.
"""

import numpy as np
import pandas as pd
from datetime import datetime


def add_laplace_noise(value, epsilon, sensitivity=1):
    """
    Tambah Laplace noise ke satu nilai.

    Formula: noise ~ Laplace(0, sensitivity/epsilon)

    Args:
        value       : float → nilai asli
        epsilon     : float → privacy budget (kecil = lebih noise)
        sensitivity : float → max perubahan dari 1 record (default=1)

    Returns:
        float → nilai + noise
    """
    if epsilon <= 0:
        raise ValueError('Epsilon harus > 0')

    scale = sensitivity / epsilon
    noise = np.random.laplace(loc=0, scale=scale)

    return value + noise


def add_noise_to_counts(df, epsilon, qi_attributes, sensitive_attribute=None):
    """
    Tambah Laplace noise ke group counts.

    Args:
        df                 : pd.DataFrame → k-anonymous dataset
        epsilon            : float → privacy budget
        qi_attributes      : list → QI attributes
        sensitive_attribute: str → optional, include distribusi sensitive attr

    Returns:
        pd.DataFrame → group counts dengan noisy_count
    """
    if sensitive_attribute:
        groups = df.groupby(
            qi_attributes + [sensitive_attribute]
        ).size().reset_index(name='count')
    else:
        groups = df.groupby(
            qi_attributes
        ).size().reset_index(name='count')

    groups['noisy_count'] = groups['count'].apply(
        lambda x: max(0, add_laplace_noise(x, epsilon))
    )

    groups['noisy_count'] = groups['noisy_count'].round().astype(int)

    return groups


class PrivacyBudgetTracker:
    """
    Track konsumsi epsilon (privacy budget).
    Implementasi composition theorem DP.
    """

    def __init__(self, total_epsilon):
        self.total_epsilon = total_epsilon
        self.used_epsilon = 0.0
        self.operations = []

    def consume(self, epsilon, operation_name):
        """
        Konsumsi privacy budget.

        Raises:
            ValueError jika budget habis
        """
        if epsilon <= 0:
            raise ValueError('Epsilon harus > 0')

        if self.used_epsilon + epsilon > self.total_epsilon + 1e-9:
            raise ValueError(
                f'Budget habis! '
                f'Diminta: {epsilon}, '
                f'Tersisa: {self.remaining():.4f}'
            )

        self.used_epsilon += epsilon
        self.operations.append({
            'operation': operation_name,
            'epsilon': epsilon,
            'cumulative_epsilon': self.used_epsilon,
            'timestamp': datetime.now().strftime('%H:%M:%S'),
        })

        print(f'[OK] eps={epsilon:.4f} dikonsumsi -> "{operation_name}"')
        print(f'  Sisa budget: eps={self.remaining():.4f}')

    def remaining(self):
        return self.total_epsilon - self.used_epsilon

    def print_summary(self):
        print('=' * 60)
        print('PRIVACY BUDGET SUMMARY')
        print('=' * 60)
        print(f'Total  : ε={self.total_epsilon:.4f}')
        print(f'Used   : ε={self.used_epsilon:.4f} '
              f'({self.used_epsilon / self.total_epsilon * 100:.1f}%)')
        print(f'Remaining: ε={self.remaining():.4f} '
              f'({self.remaining() / self.total_epsilon * 100:.1f}%)')
        print(f'\nOperations ({len(self.operations)}):')
        for i, op in enumerate(self.operations, 1):
            print(f'  {i}. [{op["timestamp"]}] '
                  f'{op["operation"]}: ε={op["epsilon"]:.4f} '
                  f'(cumulative: ε={op["cumulative_epsilon"]:.4f})')
        print('=' * 60)
