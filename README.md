# kda-projek-kelompok-2
projek mata kuliah keamanan data dan aplikasinya

anggota : 
1. Najma Syakira (L0224023)
2. Yoeke Sekti Pertiwi (L0224027)
3. Fatih Dzaki Nabhani (L0224042)
4. Muhammad Darell Hylmi (L0224045)

## 📋 Deskripsi Proyek

Implementasi **ACDP Tree (Anonymity-Conscious Decision Tree)** untuk anonymisasi data kesehatan diabetes dengan privacy guarantees:
- **K-Anonymity**: Setiap individu tidak dapat dibedakan dari minimal k-1 individu lain
- **Differential Privacy**: Penambahan noise Laplace untuk melindungi informasi individu

## 🗂️ Struktur Proyek

```
kda-projek-kelompok-2/
├── data/
│   └── raw/
│       └── diabetes__health_indicators.csv    # Dataset original
├── src/
│   ├── acdp_tree.py          # ACDP Tree algorithm
│   ├── ace.py                # ACE (k-anonymity enforcement)
│   ├── config.py             # Konfigurasi parameter
│   ├── hierarchy.py          # Generalization hierarchy
│   ├── metrics.py            # Evaluation metrics
│   ├── noise.py              # Differential privacy
│   ├── utils.py              # Utility functions
│   └── visualization.py      # Plotting functions
├── frontend/
│   ├── dashboard.py          # Streamlit dashboard
│   └── README.md             # Dokumentasi frontend
├── results/
│   └── anonymized_data/      # Output hasil anonymisasi
├── main.py                   # Pipeline utama
├── requirements.txt          # Dependencies
└── README.md                 # Dokumentasi ini
```

## 🚀 Cara Menjalankan

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Jalankan Pipeline ACDP Tree

```bash
python main.py
```

**Output:**
- `results/anonymized_data/diabetes_anonymized_k5_eps1.0.csv` - Dataset anonymized
- `results/anonymized_data/diabetes_noisy_counts_k5_eps1.0.csv` - Counts dengan noise
- `results/anonymized_data/anonymization_metadata.json` - Metadata proses
- `results/anonymized_data/evaluation_metrics.json` - Metrics evaluasi
- `results/anonymized_data/evaluation_report.txt` - Laporan lengkap

### 3. Jalankan Dashboard (Opsional)

```bash
streamlit run frontend/dashboard.py
```

Dashboard akan terbuka di browser pada `http://localhost:8501`

## ⚙️ Konfigurasi Parameter

Edit file `src/config.py` untuk mengubah parameter:

```python
# Privacy parameters
K_ANONYMITY = 5      # Minimal group size
EPSILON = 1.0        # Privacy budget (smaller = more privacy)
MAX_LEVEL = 3        # Max generalization level
```

## 📊 Fitur Dashboard

- **Overview**: Key metrics dan privacy status
- **Data Comparison**: Perbandingan distribusi original vs anonymized
- **Privacy Metrics**: Re-identification risk dan DP noise
- **Utility Metrics**: Information loss dan distribution preservation
- **Visualizations**: Grafik interaktif untuk semua metrics

## 📈 Pipeline ACDP Tree

1. **Load & Preprocess Data** - Baca dan bersihkan data
2. **Compute Weights** - Hitung inverse frequency weights
3. **Build ACDP Tree** - Optimasi generalisasi dengan weighted MI
4. **Run ACE** - Enforce k-anonymity
5. **Add DP Noise** - Tambah Laplace noise
6. **Save Results** - Simpan dataset dan metrics
7. **Evaluate** - Hitung information loss, KL-divergence, dll
8. **Generate Report** - Buat laporan evaluasi

## 🔐 Privacy Guarantees

- ✅ **K-Anonymity (k=5)**: Setiap group minimal 5 records
- ✅ **Differential Privacy (ε=1.0)**: Laplace noise pada counts
- ✅ **Re-identification Risk**: Reduced dari ~X% ke ~Y%

## 📝 Lisensi

Proyek ini dibuat untuk keperluan akademik - Mata Kuliah Keamanan Data dan Aplikasinya
