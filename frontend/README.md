# ACDP Tree Dashboard - Frontend

Interactive web dashboard untuk visualisasi hasil anonymisasi data menggunakan ACDP Tree.

## 📋 Fitur

### **Halaman Utama:**
- **Overview**: Ringkasan metrics privacy dan utility
- **Data Comparison**: Perbandingan distribusi original vs anonymized
- **Privacy Metrics**: Re-identification risk dan differential privacy noise
- **Utility Metrics**: Information loss dan distribution preservation
- **Visualizations**: Grafik interaktif untuk semua metrics

### **Fitur Baru:** ⭐
- **Tree Simulation**: Visualisasi ACDP Tree structure dengan Laplace noise (Sunburst chart)
- **Algorithm Comparison**: Perbandingan ACDP-Tree vs DPDT vs IPA dengan 4 metrics

## 🚀 Cara Menjalankan

### 1. Install Dependencies

```bash
pip install streamlit plotly
```

### 2. Generate Data (Jika Belum)

Jalankan pipeline ACDP Tree terlebih dahulu:

```bash
python main.py
```

### 3. Jalankan Dashboard

```bash
streamlit run frontend/dashboard.py
```

Dashboard akan otomatis terbuka di browser pada `http://localhost:8501`

## 📊 Navigasi Dashboard

### Overview
- Key metrics: Privacy Gain, Utility Score, Information Loss
- Privacy status: K-anonymity dan Differential Privacy
- Summary statistics

### Data Comparison
- Pilih attribute untuk dibandingkan
- Visualisasi distribusi original vs anonymized
- Preview data (10 rows pertama)

### Privacy Metrics
- Re-identification risk comparison
- Differential privacy noise distribution
- Original vs noisy counts scatter plot

### Utility Metrics
- Information loss per attribute
- Distribution preservation (KL-divergence, TVD)
- Privacy-utility tradeoff space

### Visualizations
- Sensitive attribute distribution
- All attributes overview tables

### Tree Simulation ⭐ NEW
- Interactive sunburst chart untuk visualisasi tree structure
- Slider untuk adjust privacy budget (ε)
- Real count vs Noisy count comparison
- Top 3 attributes berdasarkan information gain

### Algorithm Comparison ⭐ NEW
- Perbandingan ACDP-Tree vs DPDT vs IPA
- 4 metrics: Information Loss, Absolute Error, Data Leakage Probability, Execution Time
- Interactive line charts untuk setiap metric
- Summary cards dengan highlight best algorithm
- Metric descriptions dengan interpretasi

## 🎨 Teknologi

- **Streamlit**: Web framework
- **Plotly**: Interactive charts (bar, scatter, line, sunburst)
- **Pandas**: Data manipulation
- **NumPy**: Numerical operations (Laplace noise)

## 📝 Catatan

- Dashboard ini untuk visualisasi hasil yang sudah ada
- Tree Simulation menggunakan sample max 1200 records untuk performa
- Algorithm Comparison menggunakan simulasi metrics (bisa diganti dengan real calculation)
- Untuk generate data baru dengan parameter berbeda, edit `src/config.py` dan jalankan ulang `python main.py`

## 🆕 Update Log

### Version 2.0 (Latest)
- ✅ Added Tree Simulation page with Sunburst chart
- ✅ Added Algorithm Comparison page (ACDP-Tree vs DPDT vs IPA)
- ✅ Interactive epsilon adjustment for tree simulation
- ✅ 4 comparison metrics with line charts
- ✅ Metric descriptions and interpretations

### Version 1.0
- Initial release with 5 main pages
- Overview, Data Comparison, Privacy Metrics, Utility Metrics, Visualizations
