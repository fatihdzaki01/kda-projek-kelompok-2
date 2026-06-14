import numpy as np
import pandas as pd
from datetime import datetime
from src.utils import ensure_list


def add_laplace_noise(value, epsilon, sensitivity=1):
    if epsilon <= 0:
        raise ValueError('Epsilon harus > 0')

    scale = sensitivity / epsilon
    noise = np.random.laplace(loc=0, scale=scale)

    return value + noise


def add_noise_to_counts(df, epsilon, qi_attributes, sensitive_attribute=None):
    sens_attrs = ensure_list(sensitive_attribute) if sensitive_attribute else []

    if sens_attrs:
        existing = [a for a in sens_attrs if a in df.columns]
        group_by = qi_attributes + existing if existing else qi_attributes
    else:
        group_by = qi_attributes

    groups = df.groupby(group_by).size().reset_index(name='count')

    groups['noisy_count'] = groups['count'].apply(
        lambda x: max(0, add_laplace_noise(x, epsilon))
    )

    groups['noisy_count'] = groups['noisy_count'].round().astype(int)

    return groups


class PrivacyBudgetTracker:

    def __init__(self, total_epsilon):
        self.total_epsilon = total_epsilon
        self.used_epsilon = 0.0
        self.operations = []

    def consume(self, epsilon, operation_name):
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
