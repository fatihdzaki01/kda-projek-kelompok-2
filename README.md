# ACDP Tree — Privacy-Preserving Data Anonymization

Implementasi **ACDP-Tree (Attribute Correlation Differential Privacy Tree)** berdasarkan penelitian:

> **"Differential Privacy Medical Data Publishing Method Based on Attribute Correlation"**
> Zhang & Li, Scientific Reports, Nature, 2022.

## Anggota Kelompok

| Nama | NIM |
|---|---|
| Najma Syakira | L0224023 |
| Yoeke Sekti Pertiwi | L0224027 |
| Fatih Dzaki Nabhani | L0224042 |
| Muhammad Darell Hylmi | L0224045 |

## Pipeline

```
Load CSV → Preprocess → ACE (AHP Ranking) → ACDP Tree (Exp. Mech. + Budget)
→ KAnonymityEnforcer → Laplace Noise → Evaluate → Save
```

## Cara Pakai

### Setup

```bash
pip install -r requirements.txt
```

### Run dengan dataset default

```bash
python main.py
```

### Run dengan config custom

```bash
python main.py --config path/to/config.py
```

### Output folder custom

```bash
python main.py --output results/my_experiment/
```

### Run unit tests

```bash
python -m unittest tests.test_acdp_tree -v
```

## Struktur File

```
.
├── main.py                         # Pipeline entry point
├── src/
│   ├── config.py                   # Dataset & privacy configuration
│   ├── preprocessing.py            # Data cleaning (missing, duplicates, outliers)
│   ├── utils.py                    # validate_config(), detect_column_type()
│   ├── hierarchy.py                # GenericGeneralizationHierarchy (auto-build)
│   ├── attribute_correlation.py    # ACE: AHP-based attribute ranking
│   ├── acdp_tree.py                # ACDP Tree + Exponential Mechanism + budget
│   ├── ace.py                      # KAnonymityEnforcer (k-anonymity safety net)
│   ├── noise.py                    # Laplace noise + PrivacyBudgetTracker
│   ├── metrics.py                  # Evaluation metrics (info loss, KL-div, risk)
│   └── visualization.py            # Plotting functions (parameter-based)
├── tests/
│   └── test_acdp_tree.py           # 33 unit tests
├── data/raw/                       # Input datasets
├── results/                        # Output results (per-dataset folder)
└── requirements.txt
```

## Konfigurasi

Edit `src/config.py`:

```python
DATASET_CONFIG = {
    'file_path': 'data/raw/dataset.csv',
    'identifier_attributes': ['Name', 'ID'],    # akan di-drop
    'qi_attributes': ['Age', 'Sex', 'BMI'],      # akan digeneralisasi
    'sensitive_attribute': 'Disease',             # target privacy
    'non_sensitive_attributes': ['Smoker'],       # dibiarkan as-is
}

PRIVACY_CONFIG = {
    'k_anonymity': 5,            # k-anonymity parameter
    'epsilon': 1.0,               # total DP budget (split: ε/2 tree + ε/2 noise)
    'max_level': 3,               # max generalization depth
    'max_tree_depth': 4,          # max tree depth
}

HIERARCHY_CONFIG = {
    'n_bins_level1': 4,           # bins for continuous at level 1
    'n_bins_level2': 2,           # bins for continuous at level 2
    'ordinal_group_size': 4,      # group size for ordinal attributes
    'top_k_frequent': 10,         # top-k categories for nominal
}

CUSTOM_HIERARCHY = {
    # Optional: override auto-detect hierarchy
    # 'Age': {'type': 'numerical_ordinal', 'mapping': {...}, 'max_level': 3}
}
```

## Privacy Budget (ε)

Total ε dibagi dua:

- **ε/2** → Tree construction (Exponential Mechanism, arithmetic progression per level)
- **ε/2** → Laplace noise pada leaf node counts

## Komponen Utama

### ACE (Attribute Correlation Evaluation)
AHP-based pairwise comparison matrix dengan 3-level importance scale. Menghasilkan ranking QI attributes berdasarkan korelasi terhadap sensitive attribute.

### ACDP Tree
Generalization optimizer dengan per-record decision. Menggunakan:
- **Weighted Mutual Information** sebagai split criteria
- **Exponential Mechanism** untuk split point continuous attributes
- **Arithmetic progression** budget allocation per tree level

### KAnonymityEnforcer
Safety net setelah tree: enforce k-anonymity dengan iterative generalization dari original values.

### Differential Privacy
Laplace mechanism dengan sensitivity = 1 (count query).

## Evaluasi Metrics

- **Information Loss**: unique values lost + entropy reduction
- **KL-Divergence**: distribution preservation
- **Re-identification Risk**: unique individuals percentage
- **Privacy-Utility Tradeoff**: privacy gain vs utility loss
- **Sensitive Attribute TVD**: distribution change

## Output

Semua hasil disimpan di `results/{dataset_name}/`:

```
results/{dataset_name}/
├── {dataset}_anonymized_k5_eps1.0.csv
├── {dataset}_noisy_counts_k5_eps1.0.csv
├── anonymization_metadata.json
├── evaluation_metrics.json
└── evaluation_report.txt
```
