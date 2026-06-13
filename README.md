# ACDP Tree — Privacy-Preserving Data Anonymization

Implementasi **ACDP-Tree (Attribute Correlation Differential Privacy Tree)** berdasarkan penelitian:

> **"Differential Privacy Medical Data Publishing Method Based on Attribute Correlation"**  
> Zhang & Li, Scientific Reports, Nature, 2022.

---

## 👥 Anggota Kelompok

| Nama | NIM |
|---|---|
| Najma Syakira | L0224023 |
| Yoeke Sekti Pertiwi | L0224027 |
| Fatih Dzaki Nabhani | L0224042 |
| Muhammad Darell Hylmi | L0224045 |

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Backend (Generate Anonymized Data)
```bash
python main.py
```

### 3. Run Frontend (Interactive Dashboard)
```bash
streamlit run frontend/dashboard.py
```

Dashboard opens at: **http://localhost:8501**

---

## 📘 Complete Documentation

**👉 See [`DOCUMENTATION.md`](DOCUMENTATION.md) for:**
- Full setup guide
- Configuration options
- Interactive features (upload CSV, custom parameters)
- Privacy pipeline explanation
- Dashboard pages overview
- Troubleshooting
- Advanced usage

---

## 🔒 Privacy Pipeline

```
Load CSV → Preprocess → ACE (AHP Ranking) 
→ ACDP Tree (Exponential Mechanism + Budget)
→ K-Anonymity Enforcer → Laplace Noise 
→ Evaluate → Save Results
```

**Privacy Guarantees:**
- ✅ K-Anonymity (k=5 default)
- ✅ Differential Privacy (ε=1.0 default)
- ✅ Re-identification risk minimized

---

## 📂 Project Structure

```
.
├── main.py                    # Backend pipeline
├── src/                       # Core algorithms
├── frontend/                  # Streamlit dashboard
├── data/raw/                  # Input datasets
├── results/                   # Output results
├── tests/                     # Unit tests
├── DOCUMENTATION.md           # 📘 Complete docs
└── requirements.txt
```

---

## 🎯 Features

### Backend:
- ✅ Generic pipeline (works with any CSV)
- ✅ ACE attribute ranking
- ✅ ACDP Tree with Exponential Mechanism
- ✅ K-anonymity enforcement
- ✅ Differential Privacy noise
- ✅ Comprehensive evaluation metrics

### Frontend Dashboard:
- ✅ **Interactive Run Anonymization** (upload CSV, configure parameters)
- ✅ Overview (key metrics, privacy status)
- ✅ Data Comparison (original vs anonymized)
- ✅ Privacy Metrics (re-ID risk, DP noise)
- ✅ Utility Metrics (information loss, KL-divergence)
- ✅ Visualizations (interactive charts)
- ✅ Tree Simulation (Laplace noise)
- ✅ Algorithm Comparison

---

## ⚙️ Configuration

Edit `src/config.py`:

```python
DATASET_CONFIG = {
    'file_path': 'data/raw/diabetes__health_indicators.csv',
    'qi_attributes': ['Age', 'Sex', 'Education', 'Income', 'BMI', 'GenHlth'],
    'sensitive_attribute': 'Diabetes_012',
}

PRIVACY_CONFIG = {
    'k_anonymity': 5,      # Minimum group size
    'epsilon': 1.0,        # DP budget
    'max_level': 3,        # Max generalization depth
}
```

Or use **interactive dashboard** to configure without editing files!

---

## 📊 Output

Results saved to `results/{dataset_name}/`:
```
{dataset_name}_anonymized_k5_eps1.0.csv    # Anonymized data
{dataset_name}_noisy_counts_k5_eps1.0.csv  # Noisy counts
anonymization_metadata.json                 # Metadata
evaluation_metrics.json                     # Metrics
evaluation_report.txt                       # Full report
```

---

## 🐛 Troubleshooting

**Backend fails?**
```bash
pip install -r requirements.txt
```

**Dashboard shows "File not found"?**
```bash
# Run backend first to generate data
python main.py
```

**Port 8501 already in use?**
```bash
streamlit run frontend/dashboard.py --server.port 8502
```

**More help:** See [`DOCUMENTATION.md`](DOCUMENTATION.md)

---

## 📖 Citation

```
Zhang, X., & Li, Y. (2022). Differential Privacy Medical Data Publishing 
Method Based on Attribute Correlation. Scientific Reports, Nature.
```

---

**📘 For detailed documentation, see [`DOCUMENTATION.md`](DOCUMENTATION.md)**

**Happy Anonymizing! 🔒✨**

