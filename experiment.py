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
    
    # Keyword patterns untuk identifier (harus tepat, tidak overlap dengan sensitive)
    identifier_keywords = [
        'uuid', 'guid', 'ssn', 'passport', 'license', 'card', 'account',
        'employeeid', 'customerid', 'userid', 'patientid',
        'transactionid', 'orderid', 'invoiceid', 'recordid',
        'identifier', 'uniqueid'
    ]
    
    # Patterns yang mengindikasikan ID (harus exact match atau suffix)
    id_patterns = ['_id', 'id_', '_key', 'key_', '_code', 'code_', '_no', 'no_']
    
    if verbose:
        print("\nTAHAP 1: IDENTIFIKASI IDENTIFIER ATTRIBUTES")
        print("-" * 80)
    
    for col in df.columns:
        unique_ratio = df[col].nunique() / len(df)
        col_lower = col.lower().replace(' ', '')
        col_normalized = col_lower.replace('_', '')
        
        # Deteksi identifier berdasarkan:
        # 1. Exact keyword matching (tanpa underscore)
        # 2. ID pattern matching (_id, id_, etc.)
        # 3. Uniqueness > 95% (unique identifier)
        # 4. Uniqueness > 30% DAN contains ID pattern (semi-unique identifier)
        
        is_identifier = False
        reason = ""
        
        # Check exact keyword matching (e.g., 'patientid', 'customerid')
        if any(keyword == col_normalized for keyword in identifier_keywords):
            is_identifier = True
            reason = "-> Keyword match"
        
        # Check ID pattern matching (_id, id_, _key, etc.)
        elif any(pattern in col_lower for pattern in id_patterns):
            is_identifier = True
            reason = "-> ID pattern"
        
        # Check high uniqueness (>95%)
        elif unique_ratio > 0.95:
            is_identifier = True
            reason = "-> 95%+ unique"
        
        # Check medium-high uniqueness (>30%) with ID pattern
        elif unique_ratio > 0.30 and any(pattern in col_lower for pattern in id_patterns):
            is_identifier = True
            reason = f"-> {unique_ratio*100:.1f}% unique + ID pattern"
        
        if is_identifier:
            identifiers.append(col)
            if verbose:
                print(f"[DROP] {col:30s} Uniqueness: {unique_ratio:.3f} {reason}")
        else:
            candidates.append(col)
            if verbose:
                status = "[OK]  " if unique_ratio < 0.5 else "[WARN]"
                print(f"{status} {col:30s} Uniqueness: {unique_ratio:.3f}")
    
    if verbose:
        print(f"\nIdentifier yang harus di-drop: {identifiers}")
        print(f"Kandidat attribute: {len(candidates)}")
    
    # Tahap 2: Tentukan sensitive attribute dengan domain-aware heuristics
    if verbose:
        print("\n" + "=" * 80)
        print("TAHAP 2: REKOMENDASI SENSITIVE ATTRIBUTE")
        print("-" * 80)
    
    # Keyword patterns untuk sensitive attributes (domain-aware)
    sensitive_keywords = {
        'medical': ['diagnosis', 'disease', 'condition', 'symptom', 'treatment', 'medication', 'icd'],
        'financial': ['salary', 'income', 'revenue', 'profit', 'credit', 'balance', 'debt'],
        'personal': ['religion', 'ethnicity', 'race', 'orientation', 'disability', 'health'],
        'location': ['address', 'zip', 'postal', 'street', 'location', 'gps'],
        'demographic': ['age', 'dob', 'birth', 'gender', 'sex', 'marital']
    }
    
    # Priority keywords (if column contains these, boost even more)
    priority_keywords = ['primary', 'main', 'target', 'label', 'class', 'outcome']
    
    sensitive_scores = {}
    domain_boost = {}
    
    for col in candidates:
        col_lower = col.lower().replace('_', '').replace(' ', '')
        
        # Check domain keyword matching
        domain_score = 0
        matched_domain = None
        for domain, keywords in sensitive_keywords.items():
            if any(keyword in col_lower for keyword in keywords):
                domain_score = 10.0  # Boost score untuk keyword match
                matched_domain = domain
                break
        
        # Check priority keyword matching (e.g., "primary")
        priority_score = 0
        if any(priority in col_lower for priority in priority_keywords):
            priority_score = 5.0  # Extra boost for "primary", "main", etc.
        
        domain_boost[col] = (domain_score + priority_score, matched_domain)
        
        # Calculate statistical score
        if df[col].dtype in ['object', 'category']:
            # Kategorikal: hitung entropy
            value_counts = df[col].value_counts(normalize=True)
            entropy = -np.sum(value_counts * np.log2(value_counts + 1e-10))
            stat_score = entropy
            
            if verbose:
                domain_info = f" [{matched_domain.upper()}]" if matched_domain else ""
                priority_info = " [PRIMARY]" if priority_score > 0 else ""
                print(f"{col:30s} Entropy: {entropy:.3f}{domain_info}{priority_info}")
        
        elif df[col].dtype in ['int64', 'int32', 'float64', 'float32']:
            # Numerikal: hitung coefficient of variation
            variance = df[col].var()
            mean = df[col].mean()
            cv = variance / (mean + 1e-10)
            stat_score = min(cv, 10.0)  # Cap at 10
            
            if verbose:
                domain_info = f" [{matched_domain.upper()}]" if matched_domain else ""
                priority_info = " [PRIMARY]" if priority_score > 0 else ""
                print(f"{col:30s} Coef.Var: {cv:.3f}{domain_info}{priority_info}")
        else:
            stat_score = 0
        
        # Final score = domain boost + priority boost + statistical score
        sensitive_scores[col] = domain_score + priority_score + stat_score
    
    # Sort berdasarkan skor (domain-aware priority)
    sorted_sensitive = sorted(sensitive_scores.items(), key=lambda x: x[1], reverse=True)
    recommended_sensitive = sorted_sensitive[0][0] if sorted_sensitive else None
    
    if verbose:
        print(f"\n{'='*80}")
        print(f"REKOMENDASI SENSITIVE ATTRIBUTE: {recommended_sensitive}")
        boost_info = domain_boost.get(recommended_sensitive, (0, None))
        if boost_info[1]:
            print(f"(Domain: {boost_info[1].upper()} - keyword match detected)")
        else:
            print(f"(Based on entropy/variance - no domain keyword found)")
        print(f"{'='*80}")
    
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
    
    # Tahap 4: Pilih QI attributes dengan smart filtering
    if verbose:
        print("\n" + "=" * 80)
        print("TAHAP 4: SELEKSI QI ATTRIBUTES")
        print("-" * 80)
    
    # Domain-aware patterns untuk exclude dari QI
    # (Simplified - no need for complex categorization)
    
    # Deteksi attributes yang harus di-exclude
    excluded_qi = []
    exclusion_reasons = {}
    
    for attr, nmi in nmi_scores.items():
        attr_lower = attr.lower().replace('_', '').replace(' ', '')
        sensitive_lower = recommended_sensitive.lower().replace('_', '').replace(' ', '')
        excluded = False
        reason = []
        
        # Check 1: Redundant (NMI > 0.9)
        if nmi > 0.9:
            excluded = True
            reason.append(f"redundant with {recommended_sensitive} (NMI={nmi:.3f})")
        
        # Check 2: Medical/classification codes (ALWAYS exclude)
        medical_code_keywords = ['icd', 'cpt', 'snomed', 'loinc', 'ndc', 'rxnorm', 'hcpcs']
        if any(keyword in attr_lower for keyword in medical_code_keywords):
            excluded = True
            reason.append("medical/classification codes should not be used as QI")
        
        # Check 3: Other code patterns (conditional exclude based on NMI)
        elif not excluded:
            other_code_keywords = ['code', 'class', 'category', 'score', 'index', 'rating', 'level', 'grade']
            for keyword in other_code_keywords:
                if keyword in attr_lower:
                    # Jika NMI tinggi (>0.4) dan ada token overlap dengan sensitive
                    if nmi > 0.4:
                        attr_tokens = set(attr_lower.replace(keyword, '').split('_'))
                        sens_tokens = set(sensitive_lower.split('_'))
                        
                        if attr_tokens & sens_tokens:
                            excluded = True
                            reason.append(f"derived/classification attribute - likely encoded version of {recommended_sensitive}")
                            break
                    
                    # Atau jika NMI sangat tinggi (>0.6)
                    elif nmi > 0.6:
                        excluded = True
                        reason.append(f"derived attribute - high correlation (NMI={nmi:.3f}) suggests redundancy")
                        break
        
        if excluded:
            excluded_qi.append(attr)
            exclusion_reasons[attr] = ', '.join(reason)
    
    # Filter QI berdasarkan NMI dan exclusions
    good_qi = [attr for attr, nmi in nmi_scores.items() 
               if 0.1 <= nmi <= 0.9 and attr not in excluded_qi]
    weak_qi = [attr for attr, nmi in nmi_scores.items() 
               if 0.05 <= nmi < 0.1 and attr not in excluded_qi]
    irrelevant = [attr for attr, nmi in nmi_scores.items() 
                  if nmi < 0.05 and attr not in excluded_qi]
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
        
        if excluded_qi:
            print(f"\n⚠️  EXCLUDED (domain-aware filtering): {len(excluded_qi)} attributes")
            for attr in excluded_qi:
                reason = exclusion_reasons.get(attr, 'unknown')
                print(f"   {attr:30s} NMI: {nmi_scores[attr]:.3f} -> {reason}")
        
        if redundant and not any(r in excluded_qi for r in redundant):
            print(f"\nREDUNDAN (NMI > 0.9): {len(redundant)} attributes (harus di-drop!)")
            for attr in redundant:
                if attr not in excluded_qi:
                    print(f"   {attr:30s} NMI: {nmi_scores[attr]:.3f}")
        
        if irrelevant:
            print(f"\nTIDAK RELEVAN (NMI < 0.05): {len(irrelevant)} attributes")
    
    # Rekomendasi: ambil top 5 QI dengan NMI terbaik (exclude yang di-filter)
    recommended_qi = [attr for attr, _ in good_qi_sorted[:5]]
    
    # Tambahkan weak_qi jika good_qi kurang dari 3
    if len(recommended_qi) < 3 and weak_qi:
        weak_qi_sorted = sorted([(attr, nmi_scores[attr]) for attr in weak_qi], 
                               key=lambda x: x[1], reverse=True)
        additional_qi = [attr for attr, _ in weak_qi_sorted[:max(3, 5-len(recommended_qi))]]
        recommended_qi.extend(additional_qi)
    
    # Fallback: jika masih kosong, ambil dari irrelevant (last resort)
    if len(recommended_qi) == 0 and irrelevant:
        # Ambil yang NMI-nya paling tinggi dari irrelevant
        irrelevant_sorted = sorted([(attr, nmi_scores[attr]) for attr in irrelevant], 
                                  key=lambda x: x[1], reverse=True)
        fallback_qi = [attr for attr, _ in irrelevant_sorted[:3]]
        recommended_qi.extend(fallback_qi)
    
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
        if recommended_qi:
            for i, attr in enumerate(recommended_qi, 1):
                nmi = nmi_scores.get(attr, 0)
                # Cek apakah ini dari good_qi, weak_qi, atau fallback
                if nmi >= 0.1:
                    status = ""
                elif nmi >= 0.05:
                    status = " [WEAK - consider using]"
                else:
                    status = " [FALLBACK - low correlation]"
                print(f"   {i}. {attr:30s} (NMI: {nmi:.3f}){status}")
        else:
            print("   ⚠️  WARNING: Tidak ada QI yang memenuhi kriteria!")
            print("   Pertimbangkan untuk:")
            print("   - Gunakan attributes dengan NMI lemah (0.05-0.1)")
            print("   - Tambahkan attributes eksternal (tanggal, lokasi, dll.)")
        
        if excluded_qi:
            print(f"\nExcluded Attributes (jangan gunakan sebagai QI):")
            for attr in excluded_qi[:5]:
                reason = exclusion_reasons.get(attr, 'domain filtering')
                print(f"   - {attr:30s} ({reason})")
        
        print(f"\nKonfigurasi untuk pipeline:")
        print(f"   qi_attributes = {recommended_qi}")
        print(f"   sensitive_attribute = '{recommended_sensitive}'")
        print(f"   identifier_attributes = {identifiers}")
        
        if excluded_qi:
            print(f"\n💡 CATATAN:")
            print(f"   Tool ini menggunakan domain-aware filtering untuk exclude:")
            print(f"   - Medical/classification codes (ICD, CPT, SNOMED, dll.)")
            print(f"   - Redundant attributes (NMI > 0.9 dengan sensitive)")
            print(f"   - Derived attributes (scores, indexes, ratings)")
        
        print("=" * 80)
    
    return {
        'identifiers': identifiers,
        'sensitive_attribute': recommended_sensitive,
        'qi_attributes': recommended_qi,
        'all_qi_candidates': good_qi_sorted,
        'weak_qi': weak_qi,
        'irrelevant_attributes': irrelevant,
        'redundant_attributes': redundant,
        'excluded_attributes': excluded_qi,
        'exclusion_reasons': exclusion_reasons,
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
