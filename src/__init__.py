"""
ACDP Tree Privacy-Preserving Data Publishing Package.
"""

from src.config import (
    DATASET_CONFIG,
    PRIVACY_CONFIG,
    HIERARCHY_CONFIG,
    CUSTOM_HIERARCHY,
)

from src.hierarchy import GenericGeneralizationHierarchy

from src.attribute_correlation import AttributeCorrelationEvaluation

from src.acdp_tree import (
    ACDPTree,
    ACDPTreeNode,
    weighted_mutual_info,
    compute_inverse_frequency_weights,
    exponential_mechanism_select,
)

from src.ace import KAnonymityEnforcer

from src.noise import add_laplace_noise, add_noise_to_counts, PrivacyBudgetTracker

from src.metrics import (
    calculate_information_loss,
    calculate_kl_divergence,
    calculate_reidentification_risk,
    calculate_privacy_utility_tradeoff,
    convert_to_serializable,
)

from src.preprocessing import preprocess_generic
from src.utils import detect_column_type, validate_config
