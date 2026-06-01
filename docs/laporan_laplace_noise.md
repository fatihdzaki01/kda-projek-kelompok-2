# Penerapan Laplace Noise untuk Differential Privacy

## Latar Belakang

Dalam pipeline anonimisasi data Diabetes Health Indicators (BRFSS 2015), Laplace noise diterapkan sebagai mekanisme *differential privacy* pada tahap akhir setelah proses generalisasi (ACDP Tree) dan penegakan k-anonymity (ACE). Tujuannya adalah memberikan jaminan privasi formal secara matematis, sehingga keberadaan atau ketiadaan satu individu dalam dataset tidak dapat diidentifikasi dari output yang dipublikasikan.

## Konsep Laplace Mechanism

Mekanisme Laplace menambahkan noise yang diambil dari distribusi Laplace ke hasil query (dalam kasus ini, jumlah record per equivalence class). Formula:

```
noise ~ Laplace(0, sensitivity / epsilon)
```

- **Sensitivity (Δf)** = 1, karena penambahan/penghapusan satu record hanya mengubah count sebesar 1.
- **Epsilon (ε)** = 1.0, merupakan privacy budget yang mengontrol trade-off antara privasi dan utilitas. Semakin kecil ε, semakin besar noise (privasi lebih kuat), namun utilitas data menurun.

## Implementasi Kode

### Fungsi Utama Laplace Noise (`src/noise.py`)

```python
import numpy as np
import pandas as pd
from datetime import datetime


def add_laplace_noise(value, epsilon, sensitivity=1):
    """
    Tambah Laplace noise ke satu nilai.
    Formula: noise ~ Laplace(0, sensitivity/epsilon)
    """
    if epsilon <= 0:
        raise ValueError('Epsilon harus > 0')

    scale = sensitivity / epsilon
    noise = np.random.laplace(loc=0, scale=scale)
    return value + noise


def add_noise_to_counts(df, epsilon, qi_attributes, sensitive_attribute=None):
    """
    Tambah Laplace noise ke group counts dari dataset k-anonymous.
    """
    if sensitive_attribute:
        groups = df.groupby(
            qi_attributes + [sensitive_attribute]
        ).size().reset_index(name='count')
    else:
        groups = df.groupby(qi_attributes).size().reset_index(name='count')

    groups['noisy_count'] = groups['count'].apply(
        lambda x: max(0, add_laplace_noise(x, epsilon))
    )
    groups['noisy_count'] = groups['noisy_count'].round().astype(int)
    return groups
```

### Privacy Budget Tracker

```python
class PrivacyBudgetTracker:
    """Track konsumsi epsilon (privacy budget) - composition theorem."""

    def __init__(self, total_epsilon):
        self.total_epsilon = total_epsilon
        self.used_epsilon = 0.0
        self.operations = []

    def consume(self, epsilon, operation_name):
        if self.used_epsilon + epsilon > self.total_epsilon + 1e-9:
            raise ValueError(f'Budget habis! Tersisa: {self.remaining():.4f}')
        self.used_epsilon += epsilon
        self.operations.append({
            'operation': operation_name,
            'epsilon': epsilon,
            'cumulative_epsilon': self.used_epsilon,
            'timestamp': datetime.now().strftime('%H:%M:%S'),
        })

    def remaining(self):
        return self.total_epsilon - self.used_epsilon
```

### Pemanggilan dalam Pipeline (`main.py`)

```python
# STEP 5: Add Differential Privacy Noise
budget_tracker = PrivacyBudgetTracker(total_epsilon=EPSILON)  # ε = 1.0
budget_tracker.consume(EPSILON, 'Laplace noise on group counts')

df_noisy = add_noise_to_counts(
    df=df_k_anonymous,
    epsilon=EPSILON,
    qi_attributes=QI_ATTRIBUTES,
    sensitive_attribute=SENSITIVE_ATTRIBUTE,
)
```

## Hasil Evaluasi

Berdasarkan hasil eksperimen dengan parameter **k=5** dan **ε=1.0** pada 253.680 record:

| Metrik | Nilai |
|--------|-------|
| Mean noise added | 0.0252 |
| Mean percent error | 23.17% |
| Epsilon consumed | 1.0 (100%) |
| Re-identification risk (sebelum) | 42.16% unique individuals |
| Re-identification risk (sesudah) | 0.00% unique individuals |
| Utility score | 93.66/100 |

## Analisis

1. **Noise yang ditambahkan relatif kecil** (mean ≈ 0.025) karena distribusi Laplace bersifat simetris di sekitar 0, sehingga rata-rata noise mendekati nol.
2. **Mean percent error 23.17%** menunjukkan bahwa pada level individual group count, terdapat deviasi yang cukup signifikan — ini adalah trade-off yang diharapkan untuk mendapatkan jaminan privasi.
3. **Post-processing**: Noisy count di-clip ke minimum 0 (`max(0, ...)`) dan dibulatkan ke integer, karena count negatif tidak bermakna.
4. **Privacy budget** dikonsumsi seluruhnya (100%) dalam satu operasi, sesuai dengan *sequential composition theorem* — jika ada query tambahan, budget harus dibagi.

## Kesimpulan

Penerapan Laplace noise berhasil memberikan jaminan ε-differential privacy pada aggregated counts, dengan dampak utilitas yang terkontrol (utility score 93.66%). Kombinasi k-anonymity + differential privacy memberikan perlindungan berlapis: k-anonymity melindungi dari serangan linkage pada level record, sementara Laplace noise melindungi dari serangan inferensi pada level statistik agregat.
