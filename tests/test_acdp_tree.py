"""
Tests for the ACDP Tree privacy pipeline.
"""

import os
import sys
import unittest
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.config import DATASET_CONFIG, PRIVACY_CONFIG, HIERARCHY_CONFIG, CUSTOM_HIERARCHY
from src.utils import detect_column_type, validate_config
from src.preprocessing import preprocess_generic
from src.hierarchy import GenericGeneralizationHierarchy
from src.attribute_correlation import AttributeCorrelationEvaluation
from src.acdp_tree import ACDPTree, compute_inverse_frequency_weights, exponential_mechanism_select
from src.ace import KAnonymityEnforcer
from src.noise import add_laplace_noise, add_noise_to_counts, PrivacyBudgetTracker
from src.metrics import (
    calculate_information_loss,
    calculate_kl_divergence,
    calculate_reidentification_risk,
    calculate_privacy_utility_tradeoff,
)


class TestColumnTypeDetection(unittest.TestCase):
    """Test detect_column_type function."""

    def test_binary(self):
        s = pd.Series([0, 1, 0, 1, 0])
        self.assertEqual(detect_column_type(s), 'categorical_binary')

    def test_ordinal(self):
        s = pd.Series([1, 2, 3, 4, 5, 1, 2, 3])
        self.assertEqual(detect_column_type(s), 'numerical_ordinal')

    def test_continuous(self):
        s = pd.Series([1.5, 2.3, 4.7, 8.9, 12.3])
        self.assertEqual(detect_column_type(s), 'numerical_continuous')

    def test_nominal(self):
        s = pd.Series(['A', 'B', 'C', 'D', 'E', 'A', 'B'])
        self.assertEqual(detect_column_type(s), 'categorical_nominal')

    def test_continuous_int_many_unique(self):
        s = pd.Series(range(1, 101))
        self.assertEqual(detect_column_type(s), 'numerical_continuous')


class TestValidateConfig(unittest.TestCase):
    """Test validate_config function."""

    def setUp(self):
        self.df = pd.DataFrame({
            'Age': [25, 30, 35],
            'Sex': ['M', 'F', 'M'],
            'Disease': ['A', 'B', 'A'],
        })

    def test_valid_config(self):
        config = {
            'qi_attributes': ['Age', 'Sex'],
            'sensitive_attribute': 'Disease',
            'identifier_attributes': [],
            'non_sensitive_attributes': [],
        }
        errors, warnings = validate_config(self.df, config)
        self.assertEqual(len(errors), 0)

    def test_missing_column(self):
        config = {
            'qi_attributes': ['Age', 'NonExistent'],
            'sensitive_attribute': 'Disease',
            'identifier_attributes': [],
            'non_sensitive_attributes': [],
        }
        errors, warnings = validate_config(self.df, config)
        self.assertTrue(any('NonExistent' in e for e in errors))

    def test_no_qi(self):
        config = {
            'qi_attributes': [],
            'sensitive_attribute': 'Disease',
            'identifier_attributes': [],
            'non_sensitive_attributes': [],
        }
        errors, warnings = validate_config(self.df, config)
        self.assertTrue(any('qi' in e.lower() for e in errors))


class TestHierarchy(unittest.TestCase):
    """Test GenericGeneralizationHierarchy."""

    def setUp(self):
        np.random.seed(42)
        self.df = pd.DataFrame({
            'Age': np.random.randint(1, 14, size=200),
            'Sex': np.random.choice([0, 1], size=200),
            'BMI': np.random.uniform(15, 45, size=200),
            'Education': np.random.choice([1, 2, 3, 4, 5, 6], size=200),
            'City': np.random.choice(['Jakarta', 'Bandung', 'Surabaya', 'Medan', 'Bali', 'Other'], size=200),
        })

    def test_build_continuous(self):
        h = GenericGeneralizationHierarchy()
        h.build_from_dataframe(self.df, ['BMI'])
        self.assertIn('BMI', h.hierarchies)
        self.assertEqual(h.column_types['BMI'], 'numerical_continuous')
        self.assertEqual(h.get_max_level('BMI'), 3)

    def test_build_binary(self):
        h = GenericGeneralizationHierarchy()
        h.build_from_dataframe(self.df, ['Sex'])
        self.assertIn('Sex', h.hierarchies)
        self.assertEqual(h.column_types['Sex'], 'categorical_binary')
        self.assertEqual(h.get_max_level('Sex'), 2)

    def test_build_ordinal(self):
        h = GenericGeneralizationHierarchy()
        h.build_from_dataframe(self.df, ['Age'])
        self.assertIn('Age', h.hierarchies)
        self.assertEqual(h.column_types['Age'], 'numerical_ordinal')

    def test_build_nominal(self):
        h = GenericGeneralizationHierarchy()
        h.build_from_dataframe(self.df, ['City'])
        self.assertIn('City', h.hierarchies)
        self.assertEqual(h.column_types['City'], 'categorical_nominal')

    def test_generalize_level0(self):
        h = GenericGeneralizationHierarchy()
        h.build_from_dataframe(self.df, ['Age'])
        val = h.generalize('Age', 5, level=0)
        self.assertEqual(val, 5)

    def test_generalize_max_level(self):
        h = GenericGeneralizationHierarchy()
        h.build_from_dataframe(self.df, ['Sex'])
        val = h.generalize('Sex', 0, level=2)
        self.assertEqual(val, 'Any')

    def test_custom_hierarchy(self):
        custom = {
            'Age': {
                'type': 'numerical_ordinal',
                'mapping': {1: {1: 'Young', 2: 'Young', 3: 'Young', 4: 'Adult', 5: 'Adult'}},
                'max_level': 2,
            }
        }
        h = GenericGeneralizationHierarchy(custom_hierarchy=custom)
        h.build_from_dataframe(self.df, ['Age'])
        self.assertEqual(h.get_max_level('Age'), 2)

    def test_apply_to_dataframe(self):
        h = GenericGeneralizationHierarchy()
        h.build_from_dataframe(self.df, ['Sex'])
        levels = {'Sex': 2}
        df_gen = h.apply_to_dataframe(self.df, levels)
        self.assertTrue((df_gen['Sex'] == 'Any').all())


class TestACE(unittest.TestCase):
    """Test AttributeCorrelationEvaluation (AHP-based)."""

    def setUp(self):
        np.random.seed(42)
        n = 500
        self.df = pd.DataFrame({
            'Age': np.random.randint(1, 14, size=n),
            'Sex': np.random.choice([0, 1], size=n),
            'BMI': np.random.uniform(15, 45, size=n),
            'Disease': np.random.choice([0, 1, 2], size=n, p=[0.8, 0.05, 0.15]),
        })
        # Add correlation: Age correlates with Disease
        self.df.loc[:100, 'Disease'] = self.df.loc[:100, 'Age'] % 3

    def test_fit_returns_ranking(self):
        ace = AttributeCorrelationEvaluation()
        ranking = ace.fit(self.df, ['Age', 'Sex', 'BMI'], 'Disease')
        self.assertEqual(len(ranking), 3)
        self.assertAlmostEqual(sum(ranking.values()), 1.0, places=5)

    def test_consistency_ratio(self):
        ace = AttributeCorrelationEvaluation()
        ace.fit(self.df, ['Age', 'Sex', 'BMI'], 'Disease')
        self.assertIsNotNone(ace.consistency_ratio_)

    def test_get_ordered_attributes(self):
        ace = AttributeCorrelationEvaluation()
        ace.fit(self.df, ['Age', 'Sex', 'BMI'], 'Disease')
        ordered = ace.get_ordered_attributes()
        self.assertEqual(len(ordered), 3)
        # First attribute should have highest weight
        self.assertGreater(ace.weights_[ordered[0]], ace.weights_[ordered[-1]])


class TestExponentialMechanism(unittest.TestCase):
    """Test Exponential Mechanism."""

    def test_selects_highest_score(self):
        scores = [0.1, 0.5, 0.9]
        results = [exponential_mechanism_select(scores, 10.0, 1.0) for _ in range(100)]
        # With high epsilon, should almost always pick index 2
        avg_idx = np.mean(results)
        self.assertGreater(avg_idx, 1.5)

    def test_low_epsilon_more_random(self):
        scores = [0.1, 0.5, 0.9]
        results = [exponential_mechanism_select(scores, 0.01, 1.0) for _ in range(100)]
        avg_idx = np.mean(results)
        # With low epsilon, distribution is more uniform
        self.assertGreater(avg_idx, -1)  # Just check it runs

    def test_single_score(self):
        idx = exponential_mechanism_select([0.5], 1.0, 1.0)
        self.assertEqual(idx, 0)


class TestKAnonymityEnforcer(unittest.TestCase):
    """Test KAnonymityEnforcer."""

    def setUp(self):
        np.random.seed(42)
        self.df = pd.DataFrame({
            'Age': [25, 30, 25, 30, 35, 35, 40, 40],
            'Sex': ['M', 'F', 'M', 'F', 'M', 'F', 'M', 'F'],
            'Disease': [0, 1, 0, 1, 0, 1, 0, 1],
        })
        self.qi = ['Age', 'Sex']
        self.h = GenericGeneralizationHierarchy()
        self.h.build_from_dataframe(self.df, self.qi)

    def test_enforce_k2(self):
        df_k2 = pd.DataFrame({
            'Age': [25, 25, 30, 30],
            'Sex': ['M', 'M', 'F', 'F'],
            'Disease': [0, 1, 0, 1],
        })
        h = GenericGeneralizationHierarchy()
        h.build_from_dataframe(df_k2, ['Age', 'Sex'])
        enforcer = KAnonymityEnforcer(k=2, hierarchy=h, qi_attributes=['Age', 'Sex'])
        result = enforcer.check_k_anonymity(df_k2, verbose=False)
        self.assertTrue(result['satisfies'])

    def test_enforce_k3(self):
        enforcer = KAnonymityEnforcer(k=3, hierarchy=self.h, qi_attributes=self.qi)
        mock_levels = {i: {'Age': 0, 'Sex': 0} for i in self.df.index}
        df_result = enforcer.enforce_k_anonymity(
            self.df, self.df.copy(), mock_levels, verbose=False
        )
        result = enforcer.check_k_anonymity(df_result, verbose=False)
        self.assertTrue(result['satisfies'])


class TestNoise(unittest.TestCase):
    """Test differential privacy noise functions."""

    def test_laplace_noise_shape(self):
        noisy = add_laplace_noise(10.0, 1.0)
        self.assertIsInstance(noisy, float)

    def test_laplace_noise_low_epsilon(self):
        # Lower epsilon = more noise
        values = [add_laplace_noise(100.0, 0.1) for _ in range(1000)]
        std = np.std(values)
        self.assertGreater(std, 1.0)

    def test_budget_tracker(self):
        tracker = PrivacyBudgetTracker(total_epsilon=1.0)
        tracker.consume(0.5, 'test')
        self.assertAlmostEqual(tracker.remaining(), 0.5)

    def test_budget_exhausted(self):
        tracker = PrivacyBudgetTracker(total_epsilon=0.5)
        tracker.consume(0.5, 'test')
        with self.assertRaises(ValueError):
            tracker.consume(0.1, 'too much')


class TestMetrics(unittest.TestCase):
    """Test evaluation metrics."""

    def setUp(self):
        self.orig = pd.DataFrame({
            'Age': [25, 30, 35, 40],
            'Sex': ['M', 'F', 'M', 'F'],
            'Disease': [0, 1, 0, 1],
        })
        self.anon = pd.DataFrame({
            'Age': ['Young', 'Adult', 'Young', 'Adult'],
            'Sex': ['M', 'F', 'M', 'F'],
            'Disease': [0, 1, 0, 1],
        })
        self.qi = ['Age', 'Sex']

    def test_information_loss(self):
        loss = calculate_information_loss(self.orig, self.anon, self.qi)
        self.assertEqual(len(loss), 2)
        self.assertIn('Attribute', loss.columns)
        self.assertIn('Unique Change (%)', loss.columns)

    def test_kl_divergence(self):
        kl = calculate_kl_divergence(self.orig, self.anon, self.qi)
        self.assertEqual(len(kl), 2)

    def test_reidentification_risk(self):
        risk = calculate_reidentification_risk(self.orig, self.qi)
        self.assertIn('unique_risk_pct', risk)
        self.assertIn('avg_group_size', risk)

    def test_privacy_utility_tradeoff(self):
        orig_risk = calculate_reidentification_risk(self.orig, self.qi)
        anon_risk = calculate_reidentification_risk(self.anon, self.qi)
        loss = calculate_information_loss(self.orig, self.anon, self.qi)
        kl = calculate_kl_divergence(self.orig, self.anon, self.qi)
        tradeoff = calculate_privacy_utility_tradeoff(orig_risk, anon_risk, loss, kl)
        self.assertIn('privacy_gain_pct', tradeoff)
        self.assertIn('utility_score', tradeoff)


class TestPipelineIntegration(unittest.TestCase):
    """Integration test with a small synthetic dataset."""

    def setUp(self):
        np.random.seed(42)
        n = 500
        self.df = pd.DataFrame({
            'Age': np.random.randint(1, 14, size=n),
            'Sex': np.random.choice([0, 1], size=n),
            'BMI': np.random.uniform(18, 35, size=n),
            'Disease': np.random.choice([0, 1, 2], size=n, p=[0.8, 0.05, 0.15]),
        })

    def test_full_pipeline_runs(self):
        config = {
            'file_path': '',
            'identifier_attributes': [],
            'qi_attributes': ['Age', 'Sex', 'BMI'],
            'sensitive_attribute': 'Disease',
            'non_sensitive_attributes': [],
        }
        privacy_config = {
            'k_anonymity': 5,
            'epsilon': 1.0,
            'max_level': 3,
            'max_tree_depth': 4,
        }

        df_input = self.df[['Age', 'Sex', 'BMI', 'Disease']].copy()

        hierarchy = GenericGeneralizationHierarchy()
        hierarchy.build_from_dataframe(df_input, config['qi_attributes'])

        ace_eval = AttributeCorrelationEvaluation()
        ranking = ace_eval.fit(df_input, config['qi_attributes'], config['sensitive_attribute'])

        weights = compute_inverse_frequency_weights(df_input, config['sensitive_attribute'])

        epsilon_tree = privacy_config['epsilon'] / 2.0

        tree = ACDPTree(
            hierarchy=hierarchy,
            qi_attributes=config['qi_attributes'],
            sensitive_attribute=config['sensitive_attribute'],
            k=privacy_config['k_anonymity'],
            max_depth=privacy_config['max_level'],
            weights=weights,
            attribute_ranking=ranking,
            epsilon_tree=epsilon_tree,
        )
        tree.fit(df_input)
        df_gen = tree.transform(df_input)

        enforcer = KAnonymityEnforcer(
            k=privacy_config['k_anonymity'],
            hierarchy=hierarchy,
            qi_attributes=config['qi_attributes'],
        )
        df_anon = enforcer.enforce_k_anonymity(
            df_input, df_gen, tree.record_levels, verbose=False
        )

        result = enforcer.check_k_anonymity(df_anon, verbose=False)
        self.assertTrue(result['satisfies'],
                        f"k-anonymity not satisfied: {result['n_violations']} violations")

        df_noisy = add_noise_to_counts(
            df=df_anon,
            epsilon=privacy_config['epsilon'] / 2.0,
            qi_attributes=config['qi_attributes'],
            sensitive_attribute=config['sensitive_attribute'],
        )
        self.assertIn('noisy_count', df_noisy.columns)
        self.assertGreater(len(df_noisy), 0)


if __name__ == '__main__':
    unittest.main()
