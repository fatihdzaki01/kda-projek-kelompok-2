"""
Script untuk analisis dataset dan rekomendasi attribute selection.
Digunakan untuk eksperimen ACDP Tree dengan berbagai dataset.

Dibuat oleh: Tim Kelompok 2 - Keamanan Data
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Import modul yang sudah ada
from src.attribute_correlation import AttributeCorrelationEvaluation


def analyze_dataset(filepath, verbose=True):
    """
    Menganalisis dataset dan memberikan rekomendasi attribute selection.
    
    Args:
        filepath: Path ke file CSV
        verbose: Tampilkan detail analisis atau tidak
    
    Returns:
        Dictionary berisi rekomendasi identifiers, QI, dan sensitive attribute
    """
    df = pd.read_csv(filepath)
    
    if verbose:
        print("=" * 80)
        print(f"ANALISIS DATASET: {filepath}")
        print("=" * 80)
        print(f"Jumlah baris: {len(df):,}")
        print(f"Jumlah kolom: {len(df.columns)}")
        print(f"\nDaftar kolom: {list(df.columns)}")
        print("=" * 80)
    
    # Tahap 1: Identifikasi kolom yang merupakan identifier
    identifiers = []
    candidates = []
    
    if verbose:
        print("\nTAHAP 1: IDENTIFIKASI IDENTIFIER ATTRIBUTES")
        print("-" * 80)
    
    for col in df.columns:
        unique_ratio = df[col].nunique() / len(df)
        
        if unique_ratio > 0.95:
            identifiers.append(col)
            if verbose:
                print(f"[DROP] {col:30s} Uniqueness: {unique_ratio:.3f} -> Identifier")
        else:
            candidates.append(col)
            if verbose:
                status = "[OK]  " if unique_ratio < 0.5 else "[WARN]"
                print(f"{status} {col:30s} Uniqueness: {unique_ratio:.3f}")
    
    if verbose:
        print(f"\nIdentifier yang harus di-drop: {identifiers}")
        print(f"Kandidat attribute: {len(candidates)}")
    
    # Tahap 2: Tentukan sensitive attribute (entropy/variance tertinggi)
    if verbose:
        print("\n" + "=" * 80)
        print("TAHAP 2: REKOMENDASI SENSITIVE ATTRIBUTE")
        print("-" * 80)
    
    sensitive_scores = {}
    
    for col in candidates:
        if df[col].dtype in ['object', 'category']:
            # Kategorikal: hitung entropy
            value_counts = df[col].value_counts(normalize=True)
            entropy = -np.sum(value_counts * np.log2(value_counts + 1e-10))
            sensitive_scores[col] = entropy
            
            if verbose:
                print(f"{col:30s} Entropy: {entropy:.3f}")
        
        elif df[col].dtype in ['int64', 'int32', 'float64', 'float32']:
            # Numerikal: hitung coefficient of variation
            variance = df[col].var()
            mean = df[col].mean()
            cv = variance / (mean + 1e-10)
            sensitive_scores[col] = cv
            
            if verbose:
                print(f"{col:30s} Coef.Var: {cv:.3f}")
    
    # Sort berdasarkan skor
    sorted_sensitive = sorted(sensitive_scores.items(), key=lambda x: x[1], reverse=True)
    recommended_sensitive = sorted_sensitive[0][0] if sorted_sensitive else None
    
    if verbose:
        print(f"\nREKOMENDASI SENSITIVE ATTRIBUTE: {recommended_sensitive}")
        print(f"(Attribute dengan entropy/variance tertinggi)")
    
    # Tahap 3: Gunakan ACE module untuk hitung NMI
    if verbose:
        print("\n" + "=" * 80)
        print("TAHAP 3: HITUNG NMI MENGGUNAKAN ACE MODULE")
        print(f"Referensi: {recommended_sensitive}")
        print("-" * 80)
    
    # Gunakan ACE module yang sudah ada
    qi_candidates = [c for c in candidates if c != recommended_sensitive]
    
    if len(qi_candidates) == 0:
        if verbose:
            print("Warning: Tidak ada kandidat QI!")
        recommended_qi = []
        nmi_scores = {}
    else:
        # Inisialisasi ACE
        ace_eval = AttributeCorrelationEvaluation()
        
        # Fit ACE module
        try:
            ranking = ace_eval.fit(df, qi_candidates, recommended_sensitive)
            
            # Ambil NMI scores
            nmi_scores = ace_eval.nmi_scores_
            
            if verbose:
                print(f"Tipe sensitive: {'CONTINUOUS' if ace_eval.nmi_scores_ else 'DISCRETE'}")
                print()
                
                # Tampilkan hasil
                for attr in qi_candidates:
                    nmi = nmi_scores.get(attr, 0)
                    
                    if nmi > 0.9:
                        status = "[DROP]"
                        note = "(terlalu kuat, redundan)"
                    elif nmi > 0.3:
                        status = "[GOOD]"
                        note = "(korelasi kuat)"
                    elif nmi > 0.1:
                        status = "[WEAK]"
                        note = "(korelasi lemah)"
                    else:
                        status = "[DROP]"
                        note = "(tidak berkorelasi)"
                    
                    print(f"{status} {attr:30s} NMI: {nmi:.3f} {note}")
        
        except Exception as e:
            if verbose:
                print(f"Error saat menjalankan ACE: {e}")
            nmi_scores = {attr: 0.0 for attr in qi_candidates}
            ranking = {}
    
    # Tahap 4: Pilih QI attributes
    if verbose:
        print("\n" + "=" * 80)
        print("TAHAP 4: SELEKSI QI ATTRIBUTES")
        print("-" * 80)
    
    # Filter: 0.1 < NMI < 0.9 (sesuai standar ACE)
    good_qi = [attr for attr, nmi in nmi_scores.items() if 0.1 <= nmi <= 0.9]
    weak_qi = [attr for attr, nmi in nmi_scores.items() if 0.05 <= nmi < 0.1]
    irrelevant = [attr for attr, nmi in nmi_scores.items() if nmi < 0.05]
    redundant = [attr for attr, nmi in nmi_scores.items() if nmi > 0.9]
    
    # Sort by NMI descending (urutan ACE)
    good_qi_sorted = sorted([(attr, nmi_scores[attr]) for attr in good_qi], 
                           key=lambda x: x[1], reverse=True)
    
    if verbose:
        print(f"\nQI BAGUS (0.1 < NMI < 0.9): {len(good_qi)} attributes")
        for attr, nmi in good_qi_sorted[:10]:
            print(f"   {attr:30s} NMI: {nmi:.3f}")
        
        if weak_qi:
            print(f"\nQI LEMAH (0.05 < NMI < 0.1): {len(weak_qi)} attributes")
            for attr in weak_qi[:5]:
                print(f"   {attr:30s} NMI: {nmi_scores[attr]:.3f}")
        
        if redundant:
            print(f"\nREDUNDAN (NMI > 0.9): {len(redundant)} attributes (harus di-drop!)")
            for attr in redundant:
                print(f"   {attr:30s} NMI: {nmi_scores[attr]:.3f}")
        
        if irrelevant:
            print(f"\nTIDAK RELEVAN (NMI < 0.05): {len(irrelevant)} attributes")
    
    # Rekomendasi: ambil top 5 QI dengan NMI terbaik
    recommended_qi = [attr for attr, _ in good_qi_sorted[:5]]
    
    if verbose:
        print("\n" + "=" * 80)
        print("RINGKASAN REKOMENDASI")
        print("=" * 80)
        print(f"\nIdentifier Attributes (harus di-drop):")
        if identifiers:
            for attr in identifiers:
                print(f"   - {attr}")
        else:
            print("   (tidak ada)")
        
        print(f"\nSensitive Attribute (attribute yang dilindungi):")
        print(f"   - {recommended_sensitive}")
        
        print(f"\nQuasi-Identifier Attributes (gunakan 3-5 dari daftar ini):")
        for i, attr in enumerate(recommended_qi, 1):
            print(f"   {i}. {attr:30s} (NMI: {nmi_scores.get(attr, 0):.3f})")
        
        print(f"\nKonfigurasi untuk pipeline:")
        print(f"   qi_attributes = {recommended_qi[:5]}")
        print(f"   sensitive_attribute = '{recommended_sensitive}'")
        print(f"   identifier_attributes = {identifiers}")
        print("=" * 80)
    
    return {
        'identifiers': identifiers,
        'sensitive_attribute': recommended_sensitive,
        'qi_attributes': recommended_qi[:5],  # Top 5
        'all_qi_candidates': good_qi_sorted,
        'weak_qi': weak_qi,
        'irrelevant_attributes': irrelevant,
        'redundant_attributes': redundant,
        'nmi_scores': nmi_scores,
    }


def sample_dataset(filepath, output_path, n_rows=500, random_state=42):
    """
    Sampling dataset untuk membuat versi yang lebih kecil.
    
    Args:
        filepath: Path input CSV
        output_path: Path output CSV
        n_rows: Jumlah baris yang akan di-sample
        random_state: Random seed untuk reproducibility
    """
    df = pd.read_csv(filepath)
    
    if len(df) <= n_rows:
        print(f"Warning: Dataset hanya memiliki {len(df)} baris (diminta {n_rows})")
        print(f"         Menyalin seluruh dataset ke {output_path}")
        df.to_csv(output_path, index=False)
    else:
        df_sample = df.sample(n=n_rows, random_state=random_state)
        df_sample.to_csv(output_path, index=False)
        print(f"Berhasil sampling {n_rows} baris dari {len(df)} baris -> {output_path}")
    
    return output_path



def analyze_folder(folder_path='E:\\D_Files\\_Kuliah_\\Semester_4\\_MataPelajaran\\Keamanan_Data\\_TBP_\\_DatasetRandom_'):
    """
    Menganalisis semua file CSV dalam sebuah folder.
    
    Args:
        folder_path: Path ke folder yang berisi file-file CSV
    """
    import os
    
    print("\n" + "=" * 80)
    print(f"📁 ANALYZING FOLDER: {folder_path}")
    print("=" * 80)
    
    if not os.path.exists(folder_path):
        print(f"❌ Error: Folder not found: {folder_path}")
        return []
    
    # Find all CSV files
    csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
    
    if not csv_files:
        print(f"❌ No CSV files found in {folder_path}")
        return []
    
    print(f"✅ Found {len(csv_files)} CSV files")
    for i, f in enumerate(csv_files, 1):
        print(f"   {i}. {f}")
    
    print("\n" + "=" * 80)
    
    results = {}
    
    for csv_file in csv_files:
        filepath = os.path.join(folder_path, csv_file)
        print(f"\n{'='*80}")
        print(f"📊 ANALYZING: {csv_file}")
        print(f"{'='*80}")
        
        try:
            result = analyze_dataset(filepath, verbose=True)
            results[csv_file] = result
            
            print(f"\n✅ Analysis complete for {csv_file}")
            print("-" * 80)
            
        except Exception as e:
            print(f"\n❌ Error analyzing {csv_file}: {e}")
            results[csv_file] = {'error': str(e)}
    
    # Summary
    print("\n" + "=" * 80)
    print("📋 ANALYSIS SUMMARY")
    print("=" * 80)
    
    for csv_file, result in results.items():
        if 'error' in result:
            print(f"\n❌ {csv_file}: ERROR")
            print(f"   {result['error']}")
        else:
            print(f"\n✅ {csv_file}:")
            print(f"   Rows: {pd.read_csv(os.path.join(folder_path, csv_file)).shape[0]:,}")
            print(f"   Sensitive: {result['sensitive_attribute']}")
            print(f"   QI: {result['qi_attributes']}")
            print(f"   Identifiers: {result['identifiers']}")
    
    print("\n" + "=" * 80)
    
    return results


def quick_analyze(filepath):
    """
    Quick analysis of a single dataset (no verbose output).
    
    Args:
        filepath (str): Path to CSV file
    
    Returns:
        dict: Recommendations
    """
    return analyze_dataset(filepath, verbose=False)


if __name__ == '__main__':
    import sys
    
    print("\n" + "=" * 80)
    print("🧪 ACDP TREE EXPERIMENT TOOL")
    print("=" * 80)
    print("\nUsage:")
    print("  1. Analyze single file: python experiment.py <filepath>")
    print("  2. Analyze folder:      python experiment.py folder")
    print("  3. Sample dataset:      python experiment.py sample <input> <output> <n_rows>")
    print("=" * 80)
    
    if len(sys.argv) == 1:
        # ============================================================
        # 🔧 EDIT DI SINI: Ganti path file CSV yang mau dianalysis
        # ============================================================
        
        # Option A: Analyze single file (uncomment line di bawah)
        # analyze_dataset(r'E:\D_Files\_Kuliah_\Semester_4\_MataPelajaran\Keamanan_Data\_TBP_\_DatasetRandom_\nama_file.csv', verbose=True)
        
        # Option B: Analyze folder (default)
        analyze_folder(r'E:\D_Files\_Kuliah_\Semester_4\_MataPelajaran\Keamanan_Data\_TBP_\_DatasetRandom_')
    
    elif sys.argv[1] == 'folder':
        # Analyze folder
        folder = sys.argv[2] if len(sys.argv) > 2 else r'E:\D_Files\_Kuliah_\Semester_4\_MataPelajaran\Keamanan_Data\_TBP_\_DatasetRandom_'
        analyze_folder(folder)
    
    elif sys.argv[1] == 'sample':
        # Sample dataset
        if len(sys.argv) < 5:
            print("❌ Usage: python experiment.py sample <input.csv> <output.csv> <n_rows>")
        else:
            input_file = sys.argv[2]
            output_file = sys.argv[3]
            n_rows = int(sys.argv[4])
            sample_dataset(input_file, output_file, n_rows)
    
    else:
        # Analyze single file
        filepath = sys.argv[1]
        analyze_dataset(filepath, verbose=True)
