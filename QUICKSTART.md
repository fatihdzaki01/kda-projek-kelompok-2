# 🚀 Quick Start Guide

Panduan cepat untuk menjalankan proyek ACDP Tree.

## 📦 Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

**Dependencies yang akan diinstall:**
- numpy, pandas, scipy (data processing)
- matplotlib, seaborn (plotting)
- graphviz (tree visualization)
- streamlit, plotly (dashboard interaktif)

---

## 🔧 Step 2: Generate Anonymized Data

Jalankan pipeline ACDP Tree:

```bash
python main.py
```

**Proses yang terjadi:**
1. Load data dari `data/raw/diabetes__health_indicators.csv`
2. Build ACDP Tree (generalization optimizer)
3. Run ACE (k-anonymity enforcement)
4. Add Differential Privacy noise
5. Calculate evaluation metrics
6. Save results ke `results/anonymized_data/`

**Waktu eksekusi:** ~2-5 menit (tergantung spesifikasi komputer)

**Output files:**
- ✅ `diabetes_anonymized_k5_eps1.0.csv` - Dataset anonymized
- ✅ `diabetes_noisy_counts_k5_eps1.0.csv` - Counts dengan noise
- ✅ `anonymization_metadata.json` - Metadata
- ✅ `evaluation_metrics.json` - Metrics
- ✅ `evaluation_report.txt` - Laporan lengkap

---

## 🎨 Step 3: Jalankan Dashboard (Opsional)

Buka dashboard interaktif untuk visualisasi:

```bash
streamlit run frontend/dashboard.py
```

Dashboard akan otomatis terbuka di browser pada **http://localhost:8501**

**Fitur Dashboard:**
- 📊 Overview: Key metrics dan privacy status
- 📈 Data Comparison: Original vs Anonymized
- 🔐 Privacy Metrics: Re-ID risk, DP noise
- 📉 Utility Metrics: Information loss, KL-divergence
- 🎨 Visualizations: Interactive charts

---

## ⚙️ Mengubah Parameter (Opsional)

Edit file `src/config.py`:

```python
# Privacy parameters
K_ANONYMITY = 5      # Ubah ke 3, 5, atau 10
EPSILON = 1.0        # Ubah ke 0.5, 1.0, atau 2.0
MAX_LEVEL = 3        # Max generalization level
```

Setelah edit, jalankan ulang `python main.py`

---

## 🐛 Troubleshooting

### Error: ModuleNotFoundError
```bash
pip install -r requirements.txt
```

### Error: File not found (diabetes CSV)
Pastikan file `data/raw/diabetes__health_indicators.csv` ada.

### Dashboard tidak muncul
1. Pastikan sudah run `python main.py` terlebih dahulu
2. Cek apakah file di `results/anonymized_data/` sudah ada
3. Jalankan ulang `streamlit run frontend/dashboard.py`

### Port 8501 sudah digunakan
```bash
streamlit run frontend/dashboard.py --server.port 8502
```

---

## 📞 Bantuan

Jika ada masalah, cek:
1. Python version >= 3.8
2. Semua dependencies terinstall
3. File CSV original ada di `data/raw/`
4. Sudah run `python main.py` sebelum dashboard

---

**Happy Anonymizing! 🔒**
