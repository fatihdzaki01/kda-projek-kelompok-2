import pandas as pd
from src.config import QI_ATTRIBUTES, SENSITIVE_ATTRIBUTE, RAW_DATA_PATH, K_ANONYMITY, MAX_LEVEL
from src.hierarchy import HIERARCHY
from src.acdp_tree import ACDPTree, compute_inverse_frequency_weights
from src.utils import load_and_preprocess_data

print("Loading data...")
df = load_and_preprocess_data(RAW_DATA_PATH)
df_input = df[QI_ATTRIBUTES + [SENSITIVE_ATTRIBUTE]].copy()

print("Computing weights...")
weights = compute_inverse_frequency_weights(df, SENSITIVE_ATTRIBUTE)

# Initialize ACDPTree
acdp_tree = ACDPTree(
    hierarchy=HIERARCHY,
    qi_attributes=QI_ATTRIBUTES,
    sensitive_attribute=SENSITIVE_ATTRIBUTE,
    k=K_ANONYMITY,
    max_depth=MAX_LEVEL,
    weights=weights,
)

# Fit & Transform
acdp_tree.fit(df_input)
df_generalized = acdp_tree.transform(df_input)

print("\n--- Generalization Summary ---")
print(acdp_tree.get_generalization_summary())

# --- Tree PDF Visualization ---
print("\nExporting tree visualization to PDF...")
from src.visualization import visualize_acdp_tree
tree_viz = visualize_acdp_tree(acdp_tree.root, max_display_depth=3)
tree_viz.render('acdp_tree_generalization', format='pdf', cleanup=True)
print("[OK] Tree visualization exported to 'acdp_tree_generalization.pdf'!")

