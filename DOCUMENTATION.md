# ACDP Tree Implementation

**Privacy-Preserving Data Anonymization Framework**

This repository implements the ACDP-Tree (Attribute Correlation Differential Privacy Tree) algorithm based on research by Zhang & Li (2022), published in Scientific Reports, Nature. The implementation provides a complete pipeline combining k-anonymity, attribute correlation evaluation, and differential privacy for privacy-preserving data publication.

## Research Foundation

**Paper:** "Differential Privacy Medical Data Publishing Method Based on Attribute Correlation"  
**Authors:** Zhang, X., & Li, Y. (2022)  
**Publication:** Scientific Reports, Nature

## Contributors

| Name | Student ID |
|------|------------|
| Najma Syakira | L0224023 |
| Yoeke Sekti Pertiwi | L0224027 |
| Fatih Dzaki Nabhani | L0224042 |
| Muhammad Darell Hylmi | L0224045 |

---

## Table of Contents

1. [Getting Started](#getting-started)
2. [Architecture](#architecture)
3. [Methodology](#methodology)
4. [Configuration](#configuration)
5. [Interactive Dashboard](#interactive-dashboard)
6. [Evaluation Metrics](#evaluation-metrics)
7. [Advanced Usage](#advanced-usage)
8. [Troubleshooting](#troubleshooting)

---

## Getting Started

### Prerequisites

- Python 3.8 or higher
- pip package manager
- 4GB RAM minimum (8GB recommended for large datasets)

### Installation

Install required dependencies:

```bash
pip install -r requirements.txt
```

**Core Dependencies:**
- Data Processing: `numpy`, `pandas`, `scipy`
- Visualization: `matplotlib`, `seaborn`, `plotly`
- Web Interface: `streamlit`

### Basic Usage

Execute the anonymization pipeline:

```bash
python main.py
```

The pipeline performs seven stages:
1. Data loading and preprocessing
2. Generalization hierarchy construction
3. Attribute Correlation Evaluation (ACE)
4. ACDP Tree construction with differential privacy
5. K-anonymity enforcement
6. Laplace noise addition
7. Evaluation and output generation

Expected runtime: 3-5 minutes (varies with dataset size and parameters).

### Output

Results are saved to `results/{dataset_name}/`:
```
{dataset_name}_anonymized_k{k}_eps{eps}.csv    # Anonymized dataset
{dataset_name}_noisy_counts_k{k}_eps{eps}.csv  # Noisy group counts
anonymization_metadata.json                     # Pipeline metadata
evaluation_metrics.json                         # Quantitative metrics
evaluation_report.txt                           # Human-readable report
```

---

## Architecture

### Project Structure

```
.
├── main.py                         # Pipeline orchestration
├── src/
│   ├── config.py                   # Configuration management
│   ├── preprocessing.py            # Data preprocessing
│   ├── utils.py                    # Utility functions
│   ├── hierarchy.py                # Generalization hierarchy
│   ├── attribute_correlation.py    # ACE implementation
│   ├── acdp_tree.py                # ACDP Tree algorithm
│   ├── ace.py                      # K-anonymity enforcement
│   ├── noise.py                    # Differential privacy
│   ├── metrics.py                  # Evaluation metrics
│   └── visualization.py            # Plotting utilities
├── frontend/dashboard.py           # Web interface
├── tests/test_acdp_tree.py        # Unit tests
├── data/raw/                       # Input datasets
└── results/                        # Output directory
```

### Pipeline Flow

```
Data Input → Preprocessing → Hierarchy Construction → ACE Ranking
→ ACDP Tree (DP Mechanism) → K-Anonymity Enforcement
→ Laplace Noise → Evaluation → Output
```

---

## Methodology

### 1. Attribute Correlation Evaluation (ACE)

ACE employs the Analytic Hierarchy Process (AHP) to rank quasi-identifiers based on correlation with the sensitive attribute. This ranking guides the tree construction to prioritize attributes with stronger relationships to sensitive data.

**Pairwise Comparison Scale:**
- 1: Equal importance
- 3: Slightly more important
- 5: Moderately more important
- 7: Strongly more important
- 9: Extremely more important

**Consistency Check:**  
Consistency Ratio (CR) < 0.1 indicates reliable judgments.

**Algorithm Output:**  
Ranked list of QI attributes by weight, derived from normalized eigenvector of comparison matrix.

### 2. ACDP Tree Construction

The ACDP Tree optimizes generalization through recursive partitioning with differential privacy guarantees.

**Split Criterion:**  
Weighted Mutual Information (WMI) between QI attributes and sensitive attribute:

```
WMI(A, S) = I(A; S) × weight(A)
```

where `I(A; S)` is mutual information and `weight(A)` from ACE ranking.

**Privacy Mechanism:**  
Exponential Mechanism selects split points with probability proportional to:

```
P(split) ∝ exp(ε × WMI(split) / (2Δ))
```

where Δ is sensitivity of the WMI function.

**Budget Allocation:**  
Arithmetic progression distributes ε/2 across tree levels:

```
ε_level_i = (2 × (max_depth - i + 1)) / (max_depth × (max_depth + 1)) × (ε/2)
```

This provides stronger privacy at root levels where decisions affect more records.

### 3. K-Anonymity Enforcement

Post-tree enforcement ensures all equivalence classes satisfy k-anonymity:

1. Identify groups with size < k
2. Apply generalization to violating records (increment level)
3. Iterate until k-anonymity satisfied or max level reached
4. Fallback: force remaining violations to maximum generalization

This two-phase approach balances utility preservation with privacy compliance.

### 4. Differential Privacy Noise

Laplace noise is applied to group counts:

```
noisy_count = true_count + Laplace(0, 1/ε_noise)
```

where ε_noise = ε/2 (remaining budget after tree construction).

**Total Privacy Budget:**  
ε_total = ε_tree + ε_noise, providing (ε_total, 0)-differential privacy.

---

## Configuration

### Basic Configuration

Edit `src/config.py`:

```python
DATASET_CONFIG = {
    'file_path': 'data/raw/your_dataset.csv',
    'identifier_attributes': [],        # Dropped (e.g., ID, Name)
    'qi_attributes': [...],             # Quasi-identifiers for generalization
    'sensitive_attribute': '...',       # Target attribute to protect
    'non_sensitive_attributes': [],     # Preserved as-is
}

PRIVACY_CONFIG = {
    'k_anonymity': 5,         # Minimum group size
    'epsilon': 1.0,           # Total DP budget
    'max_level': 3,           # Max generalization depth
    'max_tree_depth': 4,      # Max tree depth
}

HIERARCHY_CONFIG = {
    'n_bins_level1': 4,       # Bins for continuous (level 1)
    'n_bins_level2': 2,       # Bins for continuous (level 2)
    'ordinal_group_size': 4,  # Grouping for ordinal
    'top_k_frequent': 10,     # Top-k for categorical
}
```

### Custom Configuration File

```bash
python main.py --config custom_config.py
python main.py --output results/experiment_01/
```

### Parameter Guidelines

| Parameter | Effect | Recommended Range |
|-----------|--------|-------------------|
| k | Higher k → more privacy, less utility | 5-10 for balanced |
| ε | Lower ε → more privacy, more noise | 0.5-2.0 for balanced |
| max_level | Higher → more flexibility, more loss | 3-4 optimal |
| max_tree_depth | Deeper → finer partitions, longer runtime | 3-5 optimal |

---

## Interactive Dashboard

### Launching the Dashboard

```bash
streamlit run frontend/dashboard.py
```

Access at: `http://localhost:8501`

### Dashboard Features

The web interface provides eight analytical views:

**1. Run Anonymization** 🔄  
Interactive pipeline execution:
- Upload custom CSV files
- Configure QI/sensitive attributes
- Adjust privacy parameters (k, ε)
- Execute pipeline and view results
- Download anonymized data

**2. Overview**  
Summary metrics and privacy status:
- Privacy gain percentage
- Utility score (0-100)
- K-anonymity compliance
- Equivalence class statistics

**3. Data Comparison**  
Original vs. anonymized data:
- Distribution charts per attribute
- Side-by-side value frequencies
- Data preview tables

**4. Privacy Metrics**  
Re-identification risk analysis:
- Unique individual percentages
- Small group risks
- Group size distributions
- Differential privacy noise impact

**5. Utility Metrics**  
Information loss assessment:
- Unique values lost per attribute
- Entropy reduction
- KL-divergence and TVD
- Distribution preservation quality

**6. Visualizations**  
Interactive charts:
- Sensitive attribute distribution
- Correlation heatmaps
- Metric comparisons

**7. Tree Simulation**  
Privacy mechanism visualization:
- Laplace noise distribution
- Original vs. noisy counts
- Parameter sensitivity analysis

**8. Algorithm Comparison**  
Benchmark against baselines:
- ACDP Tree vs. DPDT
- ACDP Tree vs. IPA
- Utility-privacy tradeoff curves

### Using Custom Datasets

1. Navigate to "Run Anonymization"
2. Upload CSV (UTF-8 encoding, with headers)
3. Select columns:
   - **QI attributes:** Demographic columns (age, gender, location)
   - **Sensitive:** Attribute to protect (diagnosis, salary)
   - **Identifiers:** Direct identifiers to remove (ID, name)
4. Configure privacy parameters
5. Execute pipeline (3-10 minutes depending on size)
6. Review results and navigate to other pages for analysis

**Dataset Requirements:**
- Clean CSV format (UTF-8)
- Minimum 1000 records recommended
- Maximum 1 million records (memory constraints)
- QI attributes: 4-8 columns optimal
- Avoid high-cardinality attributes (e.g., timestamps, free text)

---

## Evaluation Metrics

### Information Loss

**Unique Values Lost:**  
Percentage of distinct values eliminated through generalization.

**Entropy Reduction:**  
Decrease in Shannon entropy: `H(original) - H(anonymized)`

**Interpretation:** Lower values indicate better utility preservation.

### Distribution Preservation

**KL-Divergence:**  
Kullback-Leibler divergence measures distribution shift:

```
KL(P||Q) = Σ P(x) log(P(x)/Q(x))
```

**Total Variation Distance (TVD):**  
Maximum probability mass shift: `0.5 × Σ |P(x) - Q(x)|`

**Quality Thresholds:**
- Good: KL < 0.5, TVD < 0.2
- Fair: 0.5 ≤ KL < 1.0, 0.2 ≤ TVD < 0.4
- Poor: KL ≥ 1.0, TVD ≥ 0.4

### Re-identification Risk

**Unique Individuals:**  
Percentage of equivalence classes with size = 1 (violates k-anonymity).

**Small Group Risk:**  
Percentage of groups with size < k.

**Average Group Size:**  
Mean equivalence class size (higher is better for privacy).

### Privacy-Utility Tradeoff

**Privacy Gain:**  
`(original_risk - anonymized_risk) / original_risk × 100%`

**Utility Loss:**  
Average information loss across QI attributes.

**Utility Score:**  
`100 - utility_loss` (capped at [0, 100])

**Privacy-Utility Ratio:**  
`privacy_gain / utility_loss` (higher indicates better tradeoff)

---

## Advanced Usage

### Custom Hierarchy Definition

Override automatic hierarchy detection:

```python
CUSTOM_HIERARCHY = {
    'Age': {
        'type': 'numerical_ordinal',
        'mapping': {
            0: {0: '18-30', 1: '31-45', 2: '46-60', 3: '61+'},
            1: {0: '18-45', 1: '46+'},
            2: {0: 'All Ages'}
        },
        'max_level': 2
    },
    'Education': {
        'type': 'categorical_ordinal',
        'mapping': {
            0: {0: 'Elementary', 1: 'Middle', 2: 'High', 3: 'Bachelor', 4: 'Graduate'},
            1: {0: 'K-12', 1: 'Higher Ed'},
            2: {0: 'All'}
        },
        'max_level': 2
    }
}
```

### Running Unit Tests

```bash
python -m unittest tests.test_acdp_tree -v
```

Test coverage includes:
- Hierarchy construction
- ACE ranking algorithm
- ACDP Tree split decisions
- K-anonymity enforcement
- Differential privacy mechanisms
- Evaluation metrics calculation

### Batch Processing

Process multiple datasets or parameter configurations:

```python
from main import run_pipeline

configs = [
    {'k_anonymity': 5, 'epsilon': 0.5},
    {'k_anonymity': 5, 'epsilon': 1.0},
    {'k_anonymity': 10, 'epsilon': 1.0},
]

for i, privacy_config in enumerate(configs):
    results = run_pipeline(
        privacy_config=privacy_config,
        output_dir=f'results/experiment_{i:02d}/'
    )
```

---

## Troubleshooting

### Installation Issues

**ModuleNotFoundError**
```bash
pip install -r requirements.txt
```

**Python version compatibility**
```bash
python --version  # Should be 3.8+
```

### Runtime Errors

**FileNotFoundError: dataset not found**
- Verify `file_path` in `config.py` points to existing CSV
- Check file permissions and encoding (UTF-8 required)

**ValueError: KL-divergence calculation failed**
- Update to latest code (numpy array conversion fix applied)
- Check for extreme outliers or empty categories in data

**K-anonymity not satisfied after enforcement**
- Dataset too small for chosen k → reduce k or increase data size
- Too many unique combinations → reduce number of QI attributes
- High-cardinality attributes → consider removing or grouping

**MemoryError during tree construction**
- Dataset exceeds available RAM → sample dataset or increase memory
- Too many QI attributes → reduce to 6-8 attributes
- High max_tree_depth → reduce to 3-4

### Dashboard Issues

**Dashboard fails to load data**
1. Confirm backend has executed successfully: `python main.py`
2. Verify output files exist in `results/{dataset_name}/`
3. Check `src/config.py` matches actual dataset name

**Port already in use**
```bash
streamlit run frontend/dashboard.py --server.port 8502
```

**Results not updating after re-run**
- Refresh browser (F5 or Ctrl+R)
- Clear Streamlit cache: Click menu → "Clear cache"
- Restart Streamlit server

**Tree Simulation page shows error**
- Backend does not export tree structure by default
- Feature requires manual tree export call in `main.py`
- Alternatively, disable this page in navigation

### Performance Optimization

**Pipeline runtime >15 minutes**
- Dataset size >500k rows → consider sampling
- Too many QI attributes (>8) → reduce to 4-6
- High max_tree_depth (>5) → reduce to 3-4
- Parallel processing not implemented → run on faster hardware

**Dashboard slow or freezing**
- Large anonymized dataset → filter or sample for visualization
- Multiple concurrent users → deploy with proper WSGI server
- Memory exhaustion → increase available RAM or reduce data

---

## Citation

If you use this implementation in your research, please cite both the original paper and this implementation:

```bibtex
@article{zhang2022differential,
  title={Differential Privacy Medical Data Publishing Method Based on Attribute Correlation},
  author={Zhang, Xin and Li, Yuan},
  journal={Scientific Reports},
  publisher={Nature},
  year={2022}
}

@misc{acdptree2026,
  title={ACDP Tree Implementation},
  author={Syakira, N. and Pertiwi, Y.S. and Nabhani, F.D. and Hylmi, M.D.},
  year={2026},
  url={https://github.com/your-repo/acdp-tree}
}
```

---

## License

This implementation is provided for educational and research purposes. Please refer to the original paper for algorithmic details and theoretical foundations.

---

**Last Updated:** June 5, 2026
