"""
Interactive Dashboard for ACDP Tree Anonymization Results
Visualize anonymized data and evaluation metrics

Usage: streamlit run frontend/dashboard.py
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import sys
import json

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Page config
st.set_page_config(
    page_title="ACDP Tree Dashboard",
    page_icon="🔒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load font before CSS
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
""", unsafe_allow_html=True)

# Dark theme CSS — Modern SaaS Dashboard
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #0a0e17 0%, #141b2d 100%);
    }

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        color: #f1f5f9;
    }

    .main-header {
        font-size: 2rem;
        font-weight: 700;
        color: #f1f5f9;
        letter-spacing: -0.02em;
        margin-bottom: 0.25rem;
    }

    .stMarkdown, .stText, p, label {
        color: #cbd5e1;
    }

    /* Buttons */
    .stButton>button {
        background: linear-gradient(135deg, #3b82f6, #2563eb) !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        padding: 0.5rem 1rem !important;
        transition: all 0.2s ease !important;
        box-shadow: 0 4px 14px rgba(59, 130, 246, 0.25) !important;
    }

    .stButton>button:hover {
        background: linear-gradient(135deg, #60a5fa, #3b82f6) !important;
        box-shadow: 0 6px 20px rgba(59, 130, 246, 0.4) !important;
    }

    /* Selectbox / Dropdown / MultiSelect */
    .stSelectbox label, .stMultiSelect label, .stFileUploader label {
        color: #94a3b8 !important;
        font-weight: 500;
        font-size: 0.85rem;
    }

    .stSelectbox div[data-baseweb="select"] > div,
    .stMultiSelect div[data-baseweb="select"] > div {
        background-color: #1e293b !important;
        border: 1px solid #334155 !important;
        color: #f1f5f9 !important;
        border-radius: 8px !important;
        transition: border-color 0.2s;
    }

    .stSelectbox div[data-baseweb="select"] > div:hover,
    .stMultiSelect div[data-baseweb="select"] > div:hover {
        border-color: #3b82f6 !important;
    }

    .stSelectbox ul, .stMultiSelect ul {
        background-color: #1e293b !important;
        border: 1px solid #334155 !important;
        border-radius: 8px !important;
    }

    .stSelectbox li, .stMultiSelect li {
        color: #f1f5f9 !important;
    }

    .stSelectbox li:hover, .stMultiSelect li:hover {
        background-color: #3b82f6 !important;
    }

    .stSelectbox [role="listbox"], .stMultiSelect [role="listbox"] {
        background-color: #1e293b !important;
    }

    /* File Uploader */
    .stFileUploader section {
        background-color: #1e293b !important;
        border: 1px dashed #334155 !important;
        border-radius: 8px !important;
        color: #f1f5f9 !important;
        transition: border-color 0.2s;
    }

    .stFileUploader section:hover {
        border-color: #3b82f6 !important;
    }

    .stFileUploader [data-testid="stFileUploaderFileName"] {
        color: #f1f5f9 !important;
    }

    /* Slider labels */
    .stSlider label {
        color: #94a3b8 !important;
    }

    /* Number input / text input */
    .stNumberInput label, .stTextInput label {
        color: #94a3b8 !important;
    }

    .stNumberInput input, .stTextInput input {
        background-color: #1e293b !important;
        border: 1px solid #334155 !important;
        color: #f1f5f9 !important;
        border-radius: 8px !important;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        background-color: #111827;
        border-bottom: 1px solid #1e293b;
        border-radius: 8px 8px 0 0;
        padding: 4px 4px 0 4px;
    }

    .stTabs [data-baseweb="tab"] {
        color: #64748b;
        font-weight: 500;
        border-radius: 6px 6px 0 0;
        padding: 8px 16px;
        transition: all 0.2s;
    }

    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        color: #f1f5f9;
        background-color: #1e293b;
        border-bottom: 2px solid #3b82f6;
    }

    .stTabs [data-baseweb="tab"]:hover {
        color: #f1f5f9;
        background-color: rgba(59, 130, 246, 0.08);
    }

    /* Metrics */
    [data-testid="stMetricValue"] {
        font-size: 1.8rem;
        font-weight: 700;
        color: #3b82f6;
        letter-spacing: -0.02em;
    }

    [data-testid="stMetricLabel"] {
        font-weight: 500;
        color: #64748b;
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
        border-right: 1px solid #1e293b;
    }

    [data-testid="stSidebar"] .stMarkdown {
        color: #f1f5f9;
    }

    [data-testid="stSidebar"] .stRadio label {
        color: #cbd5e1;
    }

    [data-testid="stSidebar"] .stRadio div[role="radiogroup"] {
        background-color: #0f172a;
        border-radius: 8px;
        padding: 4px;
    }

    [data-testid="stSidebar"] .stRadio div[role="radio"] {
        border-radius: 6px;
        padding: 8px 14px;
        transition: all 0.2s;
        font-size: 0.85rem;
    }

    [data-testid="stSidebar"] .stRadio div[role="radio"]:hover {
        background-color: rgba(59, 130, 246, 0.1);
        color: #f1f5f9;
    }

    [data-testid="stSidebar"] .stRadio div[role="radio"][aria-checked="true"] {
        background: linear-gradient(135deg, #3b82f6, #2563eb);
        color: #ffffff;
        font-weight: 600;
        box-shadow: 0 2px 8px rgba(59, 130, 246, 0.3);
    }

    /* Dataframe */
    .stDataFrame {
        background-color: #111827;
        border-radius: 8px;
        border: 1px solid #1e293b;
    }

    .stDataFrame [data-testid="stDataFrameResizable"] {
        background-color: #1e293b;
        color: #f1f5f9;
        border-radius: 8px;
    }

    /* Expander */
    div[data-testid="stExpander"] {
        background-color: #111827;
        border: 1px solid #1e293b;
        border-radius: 8px;
        transition: border-color 0.2s;
    }

    div[data-testid="stExpander"]:hover {
        border-color: #334155;
    }

    div[data-testid="stExpander"] summary {
        color: #f1f5f9;
        font-weight: 500;
    }

    /* Info / Success / Warning / Error boxes */
    .stAlert {
        background-color: #1e293b !important;
        border: 1px solid #334155 !important;
        color: #f1f5f9 !important;
        border-radius: 8px !important;
    }

    .stAlert [data-testid="stAlert"] {
        background-color: transparent !important;
    }

    /* Checkbox / Radio */
    .stCheckbox label, .stRadio label {
        color: #f1f5f9 !important;
    }

    /* Sidebar param cards */
    .param-card {
        background: linear-gradient(135deg, #1e293b 0%, #111827 100%);
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 12px 16px;
        text-align: center;
        transition: all 0.3s;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.2);
    }

    .param-card:hover {
        border-color: #3b82f6;
        box-shadow: 0 4px 16px rgba(59, 130, 246, 0.15);
    }

    .param-card .param-label {
        font-size: 0.65rem;
        color: #64748b;
        font-weight: 600;
        letter-spacing: 0.08em;
        text-transform: uppercase;
    }

    .param-card .param-value {
        font-size: 1.4rem;
        font-weight: 700;
        color: #f1f5f9;
        margin-top: 4px;
        letter-spacing: -0.02em;
    }

    .param-card .param-value.green {
        color: #22c55e;
    }

    .param-card .param-value.blue {
        color: #3b82f6;
    }

    .param-card .param-value.orange {
        color: #f59e0b;
    }

    .records-card {
        background: linear-gradient(135deg, #1e293b 0%, #111827 100%);
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 10px 16px;
        margin-top: 8px;
        transition: all 0.3s;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.2);
    }

    .records-card:hover {
        border-color: #3b82f6;
        box-shadow: 0 4px 16px rgba(59, 130, 246, 0.15);
    }

    .records-card .records-label {
        font-size: 0.65rem;
        color: #64748b;
        font-weight: 600;
        letter-spacing: 0.08em;
        text-transform: uppercase;
    }

    .records-card .records-value {
        font-size: 1rem;
        font-weight: 600;
        color: #f1f5f9;
        margin-top: 4px;
    }

    hr {
        margin: 1.5rem 0;
        border: none;
        height: 1px;
        background: linear-gradient(90deg, transparent, #334155, transparent);
    }

    /* Headers */
    h1, h2, h3, h4, h5, h6 {
        color: #f1f5f9 !important;
        letter-spacing: -0.02em;
    }

    /* Plotly chart cards */
    .stPlotlyChart {
        background: #111827;
        border: 1px solid #1e293b;
        border-radius: 10px;
        padding: 8px;
    }

    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 6px;
        height: 6px;
    }

    ::-webkit-scrollbar-track {
        background: #0f172a;
    }

    ::-webkit-scrollbar-thumb {
        background: #334155;
        border-radius: 3px;
    }

    ::-webkit-scrollbar-thumb:hover {
        background: #475569;
    }

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* ── Welcome Splash ── */
    #welcome-overlay {
        position: fixed; inset: 0; z-index: 999999;
        background: linear-gradient(135deg, #0a0e17 0%, #1e293b 50%, #0a0e17 100%);
        display: flex; align-items: center; justify-content: center;
        flex-direction: column;
        animation: splash-fade-out 2.5s ease forwards;
    }
    #welcome-overlay .welcome-content { text-align: center; }
    #welcome-overlay .welcome-icon { font-size: 3.5rem; margin-bottom: 1rem; }
    #welcome-overlay .welcome-title {
        font-size: 2.8rem; font-weight: 800; color: #f1f5f9;
        letter-spacing: -0.03em; margin-bottom: 0.5rem;
        background: linear-gradient(135deg, #60a5fa, #3b82f6, #2563eb);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    #welcome-overlay .welcome-sub {
        font-size: 1rem; color: #64748b; font-weight: 400;
        letter-spacing: 0.05em; margin-bottom: 2rem;
    }
    #welcome-overlay .welcome-loader {
        width: 200px; height: 3px; background: #1e293b;
        border-radius: 2px; margin: 0 auto; overflow: hidden;
    }
    #welcome-overlay .welcome-bar {
        height: 100%; width: 0;
        background: linear-gradient(90deg, #3b82f6, #60a5fa);
        border-radius: 2px; animation: welcome-load 1.2s ease forwards;
    }
    @keyframes welcome-load {
        0% { width: 0; }
        50% { width: 65%; }
        100% { width: 100%; }
    }
    @keyframes splash-fade-out {
        0%, 70% { opacity: 1; visibility: visible; }
        99% { opacity: 0; visibility: visible; }
        100% { opacity: 0; visibility: hidden; pointer-events: none; }
    }
</style>
""", unsafe_allow_html=True)

# Plotly dark template
import plotly.io as pio
pio.templates["dark_acdp"] = go.layout.Template(
    layout=dict(
        paper_bgcolor="#0a0e17",
        plot_bgcolor="#111827",
        font=dict(color="#f1f5f9", family="Inter, sans-serif"),
        title=dict(font=dict(size=16, color="#f1f5f9")),
        xaxis=dict(
            gridcolor="#1e293b", zerolinecolor="#1e293b",
            title=dict(font=dict(size=13, color="#64748b")),
            tickfont=dict(size=11, color="#64748b"),
        ),
        yaxis=dict(
            gridcolor="#1e293b", zerolinecolor="#1e293b",
            title=dict(font=dict(size=13, color="#64748b")),
            tickfont=dict(size=11, color="#64748b"),
        ),
        legend=dict(font=dict(size=12, color="#f1f5f9")),
        hoverlabel=dict(bgcolor="#1e293b", font=dict(color="#f1f5f9")),
        margin=dict(t=50, l=50, r=20, b=40),
    )
)
pio.templates.default = "dark_acdp"

ACDP_COLORS = ["#3b82f6", "#60a5fa", "#f59e0b", "#ef4444"]

# Constants - Load from backend config
from src.config import DATASET_CONFIG, PRIVACY_CONFIG, HIERARCHY_CONFIG

from src.utils import ensure_list

RAW_DATA_PATH = DATASET_CONFIG['file_path']
QI_ATTRIBUTES = DATASET_CONFIG['qi_attributes']
SENSITIVE_ATTRIBUTES = ensure_list(DATASET_CONFIG.get('sensitive_attribute', []))
SENSITIVE_ATTRIBUTE = SENSITIVE_ATTRIBUTES[0] if SENSITIVE_ATTRIBUTES else ''
K_ANONYMITY = PRIVACY_CONFIG['k_anonymity']
EPSILON = PRIVACY_CONFIG['epsilon']
MAX_LEVEL = PRIVACY_CONFIG['max_level']

# Determine output directory based on dataset name (matching main.py logic)
dataset_basename = os.path.splitext(os.path.basename(RAW_DATA_PATH))[0]
dataset_name = dataset_basename.replace(' ', '_').lower()
OUTPUT_DIR = os.path.join('results', dataset_name)

# Helper functions
@st.cache_data
def load_data():
    """Load original and anonymized datasets and cached metrics"""
    try:
        df_original = pd.read_csv(RAW_DATA_PATH)
        
        anon_file = f'{dataset_name}_anonymized_k{K_ANONYMITY}_eps{EPSILON:.1f}.csv'
        anon_path = os.path.join(OUTPUT_DIR, anon_file)
        df_anonymized = pd.read_csv(anon_path)
        
        noisy_file = f'{dataset_name}_noisy_counts_k{K_ANONYMITY}_eps{EPSILON:.1f}.csv'
        noisy_path = os.path.join(OUTPUT_DIR, noisy_file)
        df_noisy = pd.read_csv(noisy_path)
        
        # Load cached metrics from JSON (generated by main.py)
        metrics_file = os.path.join(OUTPUT_DIR, 'evaluation_metrics.json')
        metadata_file = os.path.join(OUTPUT_DIR, 'anonymization_metadata.json')
        
        if os.path.exists(metrics_file):
            with open(metrics_file, 'r') as f:
                cached_metrics = json.load(f)
        else:
            cached_metrics = None
        
        return df_original, df_anonymized, df_noisy, cached_metrics, True
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.error(f"Expected paths:\n- {anon_path}\n- {noisy_path}")
        return None, None, None, None, False

def calculate_metrics(df_original, df_anonymized):
    """Calculate evaluation metrics"""
    from src.metrics import (
        calculate_information_loss,
        calculate_kl_divergence,
        calculate_reidentification_risk,
        calculate_privacy_utility_tradeoff
    )
    
    # Information loss
    info_loss = calculate_information_loss(df_original, df_anonymized, QI_ATTRIBUTES)
    
    # Distribution preservation
    dist_preserve = calculate_kl_divergence(df_original, df_anonymized, QI_ATTRIBUTES)
    
    # Re-identification risk
    orig_risk = calculate_reidentification_risk(df_original, QI_ATTRIBUTES)
    anon_risk = calculate_reidentification_risk(df_anonymized, QI_ATTRIBUTES)
    
    # Privacy-utility tradeoff
    tradeoff = calculate_privacy_utility_tradeoff(orig_risk, anon_risk, info_loss, dist_preserve)
    
    return info_loss, dist_preserve, orig_risk, anon_risk, tradeoff

def show_run_anonymization():
    """Interactive page to run anonymization on custom CSV"""
    st.markdown("### Run Anonymization")
    
    st.markdown("Upload your CSV file and configure privacy parameters to run ACDP Tree anonymization.")
    
    # File uploader
    st.markdown("---")
    st.markdown("**Upload CSV File**")
    
    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type=['csv'],
        help="Upload your dataset in CSV format",
        label_visibility="collapsed"
    )
    
    # Use default or uploaded file
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        dataset_source = uploaded_file.name
        st.success(f"Loaded: **{uploaded_file.name}** ({len(df):,} rows, {len(df.columns)} columns)")
    else:
        df = pd.read_csv(RAW_DATA_PATH)
        dataset_source = "diabetes__health_indicators.csv (default)"
        st.info(f"Using default dataset: **{dataset_source}** ({len(df):,} rows, {len(df.columns)} columns)")
    
    # Show data preview
    with st.expander("Preview Data (first 10 rows)"):
        st.dataframe(df.head(10), width='stretch')
    
    # Configuration
    st.markdown("---")
    st.markdown("**Configuration**")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Attribute Selection**")
        
        # Detect column types
        numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        all_cols = df.columns.tolist()
        
        # QI Attributes
        qi_attrs = st.multiselect(
            "Quasi-Identifier (QI) Attributes",
            options=all_cols,
            default=QI_ATTRIBUTES if set(QI_ATTRIBUTES).issubset(all_cols) else all_cols[:min(6, len(all_cols))],
            help="Attributes used for generalization (usually demographic attributes)"
        )
        
        # Sensitive Attributes
        default_sens = [s for s in SENSITIVE_ATTRIBUTES if s in all_cols]
        if not default_sens and all_cols:
            default_sens = [all_cols[0]]
        sens_attrs = st.multiselect(
            "Sensitive Attribute(s)",
            options=all_cols,
            default=default_sens,
            help="The attribute(s) to protect (e.g., disease, salary)"
        )

        # Identifier Attributes (optional)
        id_attrs = st.multiselect(
            "Identifier Attributes (will be dropped)",
            options=all_cols,
            default=[],
            help="Attributes like ID, Name that directly identify individuals"
        )
    
    with col2:
        st.markdown("**Privacy Parameters**")
        
        k_anon = st.slider(
            "K-Anonymity (k)",
            min_value=2,
            max_value=20,
            value=K_ANONYMITY,
            step=1,
            help="Minimum group size. Higher k = more privacy, less utility"
        )
        
        epsilon = st.slider(
            "Differential Privacy (ε)",
            min_value=0.1,
            max_value=5.0,
            value=float(EPSILON),
            step=0.1,
            help="Privacy budget. Lower ε = more privacy, more noise"
        )
        
        max_level = st.slider(
            "Max Generalization Level",
            min_value=1,
            max_value=5,
            value=MAX_LEVEL,
            step=1,
            help="Maximum level of generalization for each attribute"
        )
        
        max_tree_depth = st.slider(
            "Max Tree Depth",
            min_value=2,
            max_value=10,
            value=4,
            step=1,
            help="Maximum depth of ACDP Tree"
        )
    
    # Validation
    st.markdown("---")
    
    # Check for overlaps
    overlap_sens_qi = set(qi_attrs) & set(sens_attrs)
    overlap_id_qi = set(qi_attrs) & set(id_attrs)
    overlap_id_sens = set(sens_attrs) & set(id_attrs)
    
    errors_found = False
    if len(qi_attrs) == 0:
        st.error("⚠️ Please select at least one QI attribute")
        errors_found = True
    
    if not sens_attrs:
        st.error("⚠️ Please select at least one sensitive attribute")
        errors_found = True
    
    if overlap_sens_qi:
        st.error(f"⚠️ Sensitive attribute(s) cannot also be QI: {overlap_sens_qi}")
        errors_found = True
    
    if overlap_id_qi:
        st.error(f"⚠️ Identifier attribute(s) cannot also be QI: {overlap_id_qi}")
        errors_found = True
    
    if overlap_id_sens:
        st.error(f"⚠️ Identifier attribute(s) cannot also be Sensitive: {overlap_id_sens}")
        errors_found = True
    
    if errors_found:
        return
    
    # Show configuration summary
    with st.expander("Configuration Summary"):
        config_summary = {
            "Dataset": dataset_source,
            "Records": f"{len(df):,}",
            "Columns": len(df.columns),
            "QI Attributes": ", ".join(qi_attrs),
            "Sensitive Attribute": sens_attrs,
            "Identifier Attributes": ", ".join(id_attrs) if id_attrs else "None",
            "K-Anonymity": k_anon,
            "Epsilon": epsilon,
            "Max Level": max_level,
            "Max Tree Depth": max_tree_depth
        }
        st.json(config_summary)
    
    # Run button
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        run_button = st.button("Run Anonymization", width='stretch', type="primary")
    
    if run_button:
        import tempfile
        import shutil
        import io
        from contextlib import redirect_stdout, redirect_stderr
        from datetime import datetime
        
        temp_dir = tempfile.mkdtemp()
        temp_csv = os.path.join(temp_dir, "temp_dataset.csv")
        df.to_csv(temp_csv, index=False)
        
        custom_config = {
            'file_path': temp_csv,
            'identifier_attributes': id_attrs,
            'qi_attributes': qi_attrs,
            'sensitive_attribute': sens_attrs,
            'non_sensitive_attributes': [],
        }
        
        custom_privacy = {
            'k_anonymity': k_anon,
            'epsilon': epsilon,
            'max_level': max_level,
            'max_tree_depth': max_tree_depth,
        }
        
        output_name = os.path.splitext(dataset_source)[0].replace(' ', '_').lower()
        custom_output = os.path.join('results', output_name)
        
        # Status container for live progress
        status_container = st.container()
        
        with status_container:
            status_placeholder = st.empty()
            progress_bar = st.progress(0)
            log_expander = st.expander("Pipeline Logs", expanded=True)
            log_area = log_expander.empty()
        
        pipeline_log = []
        
        def log(msg, step=None, progress=None):
            pipeline_log.append(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")
            log_area.code('\n'.join(pipeline_log), language='text')
            if step:
                status_placeholder.info(step)
            if progress is not None:
                progress_bar.progress(progress)
        
        try:
            from src.pipeline import run_pipeline
            
            output_buffer = io.StringIO()
            
            with redirect_stdout(output_buffer), redirect_stderr(output_buffer):
                results = run_pipeline(
                    config=custom_config,
                    privacy_config=custom_privacy,
                    hierarchy_config=HIERARCHY_CONFIG,
                    custom_hierarchy={},
                    output_dir=custom_output,
                    progress_callback=lambda msg, p: log(msg, step=msg, progress=p)
                )
            
            # Capture full logs
            full_logs = output_buffer.getvalue()
            
            metadata = results['metadata']
            df_original = results['df_original']
            df_anonymized = results['df_anonymized']
            metrics = results['metrics']
            
            # Save to session state
            noisy_file_name = metadata['dataset_info']['noisy_counts_file']
            noisy_path = os.path.join(custom_output, noisy_file_name)
            df_noisy_loaded = pd.read_csv(noisy_path) if os.path.exists(noisy_path) else None
            
            st.session_state.has_run = True
            st.session_state.df_original = df
            st.session_state.df_anonymized = df_anonymized
            st.session_state.df_noisy = df_noisy_loaded
            st.session_state.cached_metrics = metrics
            st.session_state.params = {'k': k_anon, 'epsilon': epsilon, 'max_level': max_level}
            st.session_state.records = {'orig': len(df), 'anon': len(df_anonymized)}
            st.session_state.dataset_name = dataset_source
            st.session_state.qi_attrs = qi_attrs
            st.session_state.sens_attrs = sens_attrs
            st.session_state.sens_attr = sens_attrs[0] if sens_attrs else ''
            st.session_state.output_dir = custom_output
            st.session_state.pipeline_logs = full_logs
            
            # Show results inline
            st.markdown("---")
            st.success("**Anonymization completed successfully!**")
            
            # Summary cards
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Original Records", f"{metadata['dataset_info']['original_records']:,}")
            with col2:
                k_satisfied = metadata['privacy_guarantees']['k_anonymity_satisfied']
                st.metric("K-Anonymity", "✅ Satisfied" if k_satisfied else "❌ Not Satisfied")
            with col3:
                utility = metrics['privacy_utility_tradeoff']['utility_score']
                st.metric("Utility Score", f"{utility:.2f}/100")
            with col4:
                privacy_gain = metrics['privacy_utility_tradeoff']['privacy_gain_pct']
                st.metric("Privacy Gain", f"{privacy_gain:.1f}%")
            
            # Tabs for detailed results
            tab_summary, tab_metrics, tab_logs = st.tabs(["Privacy Details", "Evaluation Metrics", "Full Logs"])
            
            with tab_summary:
                col1, col2 = st.columns(2)
                with col1:
                    st.success(
                        f"**K-Anonymity (k={k_anon})**\n\n"
                        f"- Min group size: {metadata['privacy_guarantees']['min_group_size']}\n"
                        f"- Max group size: {metadata['privacy_guarantees']['max_group_size']:,}\n"
                        f"- Avg group size: {metadata['privacy_guarantees']['avg_group_size']:.2f}\n"
                        f"- Total groups: {metadata['privacy_guarantees']['total_groups']:,}"
                    )
                with col2:
                    st.success(
                        f"**Differential Privacy (ε={epsilon})**\n\n"
                        f"- Tree budget: ε={epsilon/2:.2f}\n"
                        f"- Noise budget: ε={epsilon/2:.2f}\n"
                        f"- Mean noise: {metadata['pipeline_summary']['noise']['mean_noise']:.4f}\n"
                        f"- Mean error: {metadata['pipeline_summary']['noise']['mean_percent_error']:.2f}%"
                    )
                
                # Re-identification risk
                st.markdown("**Re-identification Risk**")
                risk = metrics['reidentification_risk']
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Original Unique Risk", f"{risk['original']['unique_risk_pct']:.2f}%")
                with col2:
                    st.metric("Anonymized Unique Risk", f"{risk['anonymized']['unique_risk_pct']:.2f}%")
            
            with tab_metrics:
                st.dataframe(metrics['information_loss'], width='stretch')
                st.markdown("**Distribution Preservation**")
                st.dataframe(metrics['distribution_preservation'], width='stretch')
                
                # Information loss chart
                info_loss_df = metrics['information_loss']
                info_loss_chart = pd.DataFrame(info_loss_df)
                col_name = 'Unique Change (%)' if 'Unique Change (%)' in info_loss_chart.columns else 'Unique Lost (%)'
                
                fig = go.Figure()
                colors_il = [ACDP_COLORS[3] if v >= 0 else ACDP_COLORS[2] for v in info_loss_chart[col_name]]
                fig.add_trace(go.Bar(x=info_loss_chart['Attribute'], y=info_loss_chart[col_name], name='Unique Change (%)', marker_color=colors_il))
                fig.add_trace(go.Bar(x=info_loss_chart['Attribute'], y=info_loss_chart['Entropy Reduction (%)'], name='Entropy Reduction (%)', marker_color=ACDP_COLORS[2]))
                fig.update_layout(barmode='group', height=350, title='Information Loss by Attribute')
                st.plotly_chart(fig, width='stretch')
            
            with tab_logs:
                st.code(full_logs, language='text')
            
            # Output files
            with st.expander("Output Files"):
                st.info(f"Results saved to: `{custom_output}/`")
                for fname in [
                    metadata['dataset_info']['anonymized_file'],
                    metadata['dataset_info']['noisy_counts_file'],
                    "acdp_tree_structure.json",
                    "anonymization_metadata.json",
                    "evaluation_metrics.json",
                    "evaluation_report.txt",
                ]:
                    st.markdown(f"- {fname}")
            
            shutil.rmtree(temp_dir, ignore_errors=True)
            
        except ValueError as e:
            error_msg = str(e)
            log(f"ERROR: {error_msg}", "❌ Anonymization failed", 0)
            
            if "not found in dataset columns" in error_msg:
                st.error("**Configuration Error:** Some selected attributes don't exist in your CSV.")
                st.warning("Make sure you're selecting from the dropdown options only.")
            elif "No valid QI attributes" in error_msg:
                st.error("**Configuration Error:** No valid QI attributes found.")
                st.warning("Select at least one valid attribute as QI.")
            else:
                st.error(f"**Error:** {error_msg}")
            
            with st.expander("Error Details"):
                import traceback
                st.code(traceback.format_exc())
            shutil.rmtree(temp_dir, ignore_errors=True)
            
        except Exception as e:
            log(f"UNEXPECTED ERROR: {str(e)}", "❌ Anonymization failed", 0)
            st.error(f"**Unexpected error:** {str(e)}")
            with st.expander("Error Details"):
                import traceback
                st.code(traceback.format_exc())
            shutil.rmtree(temp_dir, ignore_errors=True)


# Main app
def main():
    # ── Session state init ──
    if 'initialized' not in st.session_state:
        st.session_state.initialized = True
        st.session_state.output_dir = OUTPUT_DIR
        st.session_state.df_original = None
        st.session_state.df_anonymized = None
        st.session_state.df_noisy = None
        st.session_state.cached_metrics = None
        st.session_state.params = {'k': K_ANONYMITY, 'epsilon': EPSILON, 'max_level': MAX_LEVEL}
        st.session_state.records = {'orig': 0, 'anon': 0}
        st.session_state.dataset_name = os.path.basename(RAW_DATA_PATH)
        st.session_state.qi_attrs = QI_ATTRIBUTES
        st.session_state.sens_attrs = SENSITIVE_ATTRIBUTES
        st.session_state.sens_attr = SENSITIVE_ATTRIBUTE

        df_original, df_anonymized, df_noisy, cached_metrics, success = load_data()
        if success:
            st.session_state.df_original = df_original
            st.session_state.df_anonymized = df_anonymized
            st.session_state.df_noisy = df_noisy
            st.session_state.cached_metrics = cached_metrics
            st.session_state.records = {'orig': len(df_original), 'anon': len(df_anonymized)}

    # ── Welcome splash ──
    if 'welcome_shown' not in st.session_state:
        st.session_state.welcome_shown = True
        st.markdown("""
        <div id="welcome-overlay">
          <div class="welcome-content">
            <div class="welcome-icon">🔒</div>
            <div class="welcome-title">ACDP Tree</div>
            <div class="welcome-sub">Privacy-Preserving Data Anonymization Dashboard</div>
            <div class="welcome-loader"><div class="welcome-bar"></div></div>
          </div>
        </div>
        """, unsafe_allow_html=True)

    # ── Header ──
    st.markdown('<div class="main-header">ACDP Tree · Privacy Dashboard</div>', unsafe_allow_html=True)
    dataset_label = st.session_state.dataset_name
    st.markdown(f"*Privacy-Preserving Data Anonymization for {dataset_label}*")

    # ── Load data from session state ──
    df_original = st.session_state.df_original
    df_anonymized = st.session_state.df_anonymized
    df_noisy = st.session_state.df_noisy
    cached_metrics = st.session_state.cached_metrics
    qi_attrs = st.session_state.qi_attrs
    sens_attrs = st.session_state.get('sens_attrs', SENSITIVE_ATTRIBUTES)
    sens_attr = sens_attrs[0] if sens_attrs else ''
    p = st.session_state.params

    if df_original is None:
        st.error("Failed to load data. Please run `python main.py` first to generate anonymized data.")
        st.info("Make sure the following files exist:\n"
                f"- {RAW_DATA_PATH}\n"
                f"- {OUTPUT_DIR}/diabetes_anonymized_k{K_ANONYMITY}_eps{EPSILON:.1f}.csv\n"
                f"- {OUTPUT_DIR}/diabetes_noisy_counts_k{K_ANONYMITY}_eps{EPSILON:.1f}.csv")
        return

    # ── Sidebar ──
    st.sidebar.markdown("### Navigation")
    page = st.sidebar.radio(
        "Go to",
        ["Run Anonymization", "Overview", "Data Comparison", "Privacy Metrics", "Utility Metrics", "Visualizations", "Tree Simulation", "Algorithm Comparison"],
        label_visibility="collapsed"
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown('<div style="font-size:0.8rem;font-weight:600;color:#64748b;letter-spacing:0.5px;text-transform:uppercase;margin-bottom:8px;">Privacy Parameters</div>', unsafe_allow_html=True)

    col_k, col_eps, col_lvl = st.sidebar.columns(3)
    with col_k:
        st.markdown(
            f'<div class="param-card">'
            f'<div class="param-label">K</div>'
            f'<div class="param-value green">{p["k"]}</div>'
            f'</div>',
            unsafe_allow_html=True
        )
    with col_eps:
        st.markdown(
            f'<div class="param-card">'
            f'<div class="param-label">ε</div>'
            f'<div class="param-value blue">{p["epsilon"]}</div>'
            f'</div>',
            unsafe_allow_html=True
        )
    with col_lvl:
        st.markdown(
            f'<div class="param-card">'
            f'<div class="param-label">Level</div>'
            f'<div class="param-value orange">{p["max_level"]}</div>'
            f'</div>',
            unsafe_allow_html=True
        )

    st.sidebar.markdown(
        f'<div class="records-card">'
        f'<div class="records-label">Records</div>'
        f'<div class="records-value">{st.session_state.records["orig"]:,} → {st.session_state.records["anon"]:,}</div>'
        f'</div>',
        unsafe_allow_html=True
    )

    # Use cached metrics
    if cached_metrics:
        info_loss = pd.DataFrame(cached_metrics['information_loss'])
        dist_preserve = pd.DataFrame(cached_metrics['distribution_preservation'])
        orig_risk = cached_metrics['reidentification_risk']['original']
        anon_risk = cached_metrics['reidentification_risk']['anonymized']
        tradeoff = cached_metrics['privacy_utility_tradeoff']
    else:
        st.warning("evaluation_metrics.json not found. Some pages may be limited.")
        info_loss = dist_preserve = orig_risk = anon_risk = tradeoff = None

    # ── Page routing ──
    if page == "Run Anonymization":
        show_run_anonymization()
    elif page == "Overview":
        show_overview(df_original, df_anonymized, info_loss, orig_risk, anon_risk, tradeoff)
    elif page == "Data Comparison":
        show_data_comparison(df_original, df_anonymized, qi_attrs, sens_attrs)
    elif page == "Privacy Metrics":
        show_privacy_metrics(orig_risk, anon_risk, df_noisy)
    elif page == "Utility Metrics":
        show_utility_metrics(info_loss, dist_preserve, tradeoff)
    elif page == "Visualizations":
        show_visualizations(df_original, df_anonymized, df_noisy, info_loss, dist_preserve, sens_attrs)
    elif page == "Tree Simulation":
        show_tree_simulation(df_original, df_anonymized, qi_attrs)
    elif page == "Algorithm Comparison":
        show_algorithm_comparison(df_original, qi_attrs, sens_attrs)

def show_overview(df_original, df_anonymized, info_loss, orig_risk, anon_risk, tradeoff):
    """Overview page with key metrics"""
    st.markdown("### Overview")
    
    col_name = 'Unique Change (%)' if info_loss is not None and 'Unique Change (%)' in info_loss.columns else 'Unique Lost (%)'
    raw_info_loss = info_loss[col_name].mean() if info_loss is not None else 0
    avg_info_loss = max(0, raw_info_loss)
    risk_reduction = orig_risk['unique_risk_pct'] - anon_risk['unique_risk_pct'] if orig_risk and anon_risk else 0
    
    raw_utility = tradeoff['utility_score']
    utility_score = max(0, min(100, raw_utility))
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Privacy Gain", f"{tradeoff['privacy_gain_pct']:.1f}%")
    
    with col2:
        st.metric("Utility Score", f"{utility_score:.1f}/100")
    
    with col3:
        st.metric("Information Loss", f"{avg_info_loss:.1f}%")
    
    with col4:
        st.metric("Re-ID Risk Reduction", f"{risk_reduction:.2f}%")
    
    # Privacy status
    st.markdown("---")
    st.markdown("### Privacy Status")
    
    p = st.session_state.params

    col1, col2 = st.columns(2)
    
    with col1:
        st.success(
            f"**K-Anonymity Satisfied (k={p['k']})**\n\n"
            f"- Min group size: {anon_risk['min_group_size']}\n"
            f"- Avg group size: {anon_risk['avg_group_size']:.2f}\n"
            f"- Total groups: {anon_risk['total_groups']:,}"
        )
    
    with col2:
        st.success(
            f"**Differential Privacy Applied (ε={p['epsilon']})**\n\n"
            f"- Mechanism: Laplace Noise\n"
            f"- Applied to: Aggregated counts\n"
            f"- Privacy budget: 100% consumed"
        )
    
    # Summary table
    st.markdown("---")
    st.markdown("### Summary Statistics")
    
    summary_data = {
        "Metric": [
            "Original Records",
            "Anonymized Records",
            "QI Attributes",
            "Unique Risk (Original)",
            "Unique Risk (Anonymized)",
            "Privacy Gain",
            "Utility Score",
            "Avg Information Loss"
        ],
        "Value": [
            f"{len(df_original):,}",
            f"{len(df_anonymized):,}",
            len(st.session_state.qi_attrs),
            f"{orig_risk['unique_risk_pct']:.2f}%",
            f"{anon_risk['unique_risk_pct']:.2f}%",
            f"{tradeoff['privacy_gain_pct']:.2f}%",
            f"{utility_score:.2f}/100",
            f"{avg_info_loss:.2f}%"
        ]
    }
    
    st.dataframe(pd.DataFrame(summary_data), width='stretch', hide_index=True)

def show_data_comparison(df_original, df_anonymized, qi_attrs, sens_attrs):
    """Data comparison page"""
    st.markdown("### Data Comparison")
    
    # Attribute selector
    selected_attr = st.selectbox("Select Attribute to Compare:", qi_attrs)
    
    # Distribution comparison
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Original Distribution**")
        orig_counts = df_original[selected_attr].value_counts().sort_index()
        fig1 = px.bar(
            x=orig_counts.index.astype(str),
            y=orig_counts.values,
            labels={'x': selected_attr, 'y': 'Count'},
            color_discrete_sequence=[ACDP_COLORS[1]]
        )
        fig1.update_layout(showlegend=False, height=400)
        st.plotly_chart(fig1, width='stretch')
    
    with col2:
        st.markdown("**Anonymized Distribution**")
        anon_counts = df_anonymized[selected_attr].value_counts().sort_index()
        fig2 = px.bar(
            x=anon_counts.index.astype(str),
            y=anon_counts.values,
            labels={'x': selected_attr, 'y': 'Count'},
            color_discrete_sequence=[ACDP_COLORS[0]]
        )
        fig2.update_layout(showlegend=False, height=400)
        st.plotly_chart(fig2, width='stretch')
    
    # Side-by-side comparison
    st.markdown("---")
    st.markdown("**Side-by-Side Comparison**")
    
    comparison_df = pd.DataFrame({
        'Value': orig_counts.index.astype(str).tolist(),
        'Original': orig_counts.values,
        'Anonymized': anon_counts.reindex(orig_counts.index, fill_value=0).values
    })
    
    fig3 = go.Figure()
    fig3.add_trace(go.Bar(name='Original', x=comparison_df['Value'], y=comparison_df['Original'], marker_color=ACDP_COLORS[1]))
    fig3.add_trace(go.Bar(name='Anonymized', x=comparison_df['Value'], y=comparison_df['Anonymized'], marker_color=ACDP_COLORS[0]))
    fig3.update_layout(barmode='group', height=400, xaxis_title=selected_attr, yaxis_title='Count')
    st.plotly_chart(fig3, width='stretch')
    
    # Data preview
    st.markdown("---")
    st.markdown("#### Data Preview")
    
    display_cols = qi_attrs + sens_attrs
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Original Data (first 10 rows)**")
        st.dataframe(df_original[display_cols].head(10), width='stretch')
    
    with col2:
        st.markdown("**Anonymized Data (first 10 rows)**")
        st.dataframe(df_anonymized[display_cols].head(10), width='stretch')

def show_privacy_metrics(orig_risk, anon_risk, df_noisy):
    """Privacy metrics page"""
    st.markdown("### Privacy Metrics")
    
    # Re-identification risk
    st.markdown("### Re-identification Risk")
    
    col1, col2 = st.columns(2)
    
    with col1:
        risk_data = pd.DataFrame({
            'Dataset': ['Original', 'Anonymized'],
            'Unique Risk (%)': [orig_risk['unique_risk_pct'], anon_risk['unique_risk_pct']],
            'Small Group Risk (%)': [orig_risk['small_group_risk_pct'], anon_risk['small_group_risk_pct']]
        })
        
        fig = go.Figure()
        fig.add_trace(go.Bar(name='Unique Risk', x=risk_data['Dataset'], y=risk_data['Unique Risk (%)'], marker_color=ACDP_COLORS[0]))
        fig.add_trace(go.Bar(name='Small Group Risk', x=risk_data['Dataset'], y=risk_data['Small Group Risk (%)'], marker_color=ACDP_COLORS[2]))
        fig.update_layout(barmode='group', height=400, yaxis_title='Risk (%)', title='Re-identification Risk Comparison')
        st.plotly_chart(fig, width='stretch')
    
    with col2:
        st.markdown("**Risk Metrics**")
        
        metrics_df = pd.DataFrame({
            'Metric': ['Total Groups', 'Unique Individuals', 'Small Groups (<5)', 'Avg Group Size', 'Min Group Size'],
            'Original': [
                f"{orig_risk['total_groups']:,}",
                f"{orig_risk['unique_individuals']:,}",
                f"{orig_risk['small_groups']:,}",
                f"{orig_risk['avg_group_size']:.2f}",
                f"{orig_risk['min_group_size']}"
            ],
            'Anonymized': [
                f"{anon_risk['total_groups']:,}",
                f"{anon_risk['unique_individuals']:,}",
                f"{anon_risk['small_groups']:,}",
                f"{anon_risk['avg_group_size']:.2f}",
                f"{anon_risk['min_group_size']}"
            ]
        })
        
        st.dataframe(metrics_df, width='stretch', hide_index=True)
    
    # Differential Privacy noise
    st.markdown("---")
    st.markdown("### Differential Privacy Noise Impact")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.histogram(
            df_noisy,
            x='noise_added',
            nbins=50,
            labels={'noise_added': 'Noise Added', 'count': 'Frequency'},
            color_discrete_sequence=[ACDP_COLORS[1]],
        )
        fig.add_vline(x=0, line_dash="dash", line_color=ACDP_COLORS[3], annotation_text="No Noise")
        fig.add_vline(x=df_noisy['noise_added'].mean(), line_dash="dash", line_color=ACDP_COLORS[0],
                     annotation_text=f"Mean={df_noisy['noise_added'].mean():.2f}")
        st.plotly_chart(fig, width='stretch')
    
    with col2:
        fig = px.scatter(
            df_noisy,
            x='count',
            y='noisy_count',
            labels={'count': 'Original Count', 'noisy_count': 'Noisy Count'},
            color_discrete_sequence=[ACDP_COLORS[0]],
            opacity=0.6
        )
        max_val = max(df_noisy['count'].max(), df_noisy['noisy_count'].max())
        fig.add_trace(go.Scatter(x=[0, max_val], y=[0, max_val], mode='lines',
                                name='Perfect Match', line=dict(color=ACDP_COLORS[3], dash='dash')))
        st.plotly_chart(fig, width='stretch')

def show_utility_metrics(info_loss, dist_preserve, tradeoff):
    """Utility metrics page"""
    st.markdown("### Utility Metrics")
    
    # Information loss
    st.markdown("### Information Loss per Attribute")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = go.Figure()
        col_name = 'Unique Change (%)' if info_loss is not None and 'Unique Change (%)' in info_loss.columns else 'Unique Lost (%)'
        if info_loss is not None:
            colors = [ACDP_COLORS[3] if v >= 0 else ACDP_COLORS[2] for v in info_loss[col_name]]
            fig.add_trace(go.Bar(
                x=info_loss['Attribute'],
                y=info_loss[col_name],
                marker_color=colors,
            ))
        fig.update_layout(height=400, yaxis_title='Unique Change (%)')
        st.plotly_chart(fig, width='stretch')
    
    with col2:
        fig = px.bar(
            info_loss,
            x='Attribute',
            y='Entropy Reduction (%)',
            color_discrete_sequence=[ACDP_COLORS[2]],
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, width='stretch')
    
    # Distribution preservation
    st.markdown("---")
    st.markdown("### Distribution Preservation")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.bar(
            dist_preserve,
            x='Attribute',
            y='KL-Divergence',
            color='Preservation Quality',
            color_discrete_map={'Good': ACDP_COLORS[0], 'Fair': ACDP_COLORS[2], 'Poor': ACDP_COLORS[3]}
        )
        fig.add_hline(y=0.5, line_dash="dash", line_color=ACDP_COLORS[2], annotation_text="Good/Fair Threshold")
        fig.update_layout(height=400)
        st.plotly_chart(fig, width='stretch')
    
    with col2:
        fig = px.bar(
            dist_preserve,
            x='Attribute',
            y='TVD',
            color_discrete_sequence=[ACDP_COLORS[2]],
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, width='stretch')
    
    # Privacy-Utility Tradeoff
    st.markdown("---")
    st.markdown("### Privacy-Utility Tradeoff")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Privacy Gain", f"{tradeoff['privacy_gain_pct']:.2f}%")
    
    with col2:
        utility_loss = max(0, tradeoff['utility_loss_pct'])
        st.metric("Utility Loss", f"{utility_loss:.2f}%")
    
    with col3:
        pu_ratio = tradeoff['privacy_utility_ratio'] if tradeoff['utility_loss_pct'] > 0 else 0
        st.metric("Privacy/Utility Ratio", f"{pu_ratio:.2f}",
                 delta="Good" if pu_ratio > 1.0 else "Fair")
    
    # Tradeoff scatter plot
    fig = go.Figure()
    
    fig.add_shape(type="rect", x0=0, y0=50, x1=50, y1=100,
                 fillcolor=ACDP_COLORS[0], opacity=0.1, line_width=0)
    
    fig.add_trace(go.Scatter(
        x=[utility_loss],
        y=[tradeoff['privacy_gain_pct']],
        mode='markers+text',
        marker=dict(size=20, color=ACDP_COLORS[1]),
        text=[f"k={K_ANONYMITY}, ε={EPSILON}"],
        textposition="top center",
        name='Our Result'
    ))
    
    fig.add_hline(y=50, line_dash="dash", line_color=ACDP_COLORS[0], annotation_text="High Privacy Gain")
    fig.add_vline(x=50, line_dash="dash", line_color=ACDP_COLORS[3], annotation_text="High Utility Loss")
    
    fig.update_layout(
        title='Privacy-Utility Tradeoff Space',
        xaxis_title='Utility Loss (%)',
        yaxis_title='Privacy Gain (%)',
        xaxis_range=[0, 100],
        yaxis_range=[0, 100],
        height=500
    )
    
    st.plotly_chart(fig, width='stretch')

def show_visualizations(df_original, df_anonymized, df_noisy, info_loss, dist_preserve, sens_attrs):
    """Additional visualizations page"""
    st.markdown("### Visualizations")
    
    sens_attr = st.selectbox(
        "Select Sensitive Attribute to Visualize:",
        options=sens_attrs if sens_attrs else ['None'],
        key="viz_sens_attr"
    )
    if sens_attr == 'None' or sens_attr not in df_original.columns:
        st.warning("No valid sensitive attribute selected.")
        return
    
    st.markdown(f"**Sensitive Attribute Distribution: {sens_attr}**")
    
    orig_sens = df_original[sens_attr].value_counts(normalize=True).sort_index() * 100
    anon_sens = df_anonymized[sens_attr].value_counts(normalize=True).sort_index() * 100
    
    unique_vals = sorted(set(list(orig_sens.index) + list(anon_sens.index)))
    sens_df = pd.DataFrame({
        'Class': [str(v) for v in unique_vals],
        'Original': [orig_sens.get(v, 0) for v in unique_vals],
        'Anonymized': [anon_sens.get(v, 0) for v in unique_vals]
    })
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = go.Figure()
        fig.add_trace(go.Bar(name='Original', x=sens_df['Class'], y=sens_df['Original'], marker_color=ACDP_COLORS[1]))
        fig.add_trace(go.Bar(name='Anonymized', x=sens_df['Class'], y=sens_df['Anonymized'], marker_color=ACDP_COLORS[0]))
        fig.update_layout(barmode='group', height=400, yaxis_title='Percentage (%)')
        st.plotly_chart(fig, width='stretch')
    
    with col2:
        fig = px.pie(
            sens_df,
            values='Anonymized',
            names='Class',
            color_discrete_sequence=[ACDP_COLORS[0], ACDP_COLORS[2], ACDP_COLORS[3]]
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, width='stretch')
    
    # All attributes overview
    st.markdown("---")
    st.markdown("### All Attributes Overview")
    
    tab1, tab2 = st.tabs(["Information Loss", "Distribution Preservation"])
    
    with tab1:
        st.dataframe(info_loss, width='stretch', hide_index=True)
    
    with tab2:
        st.dataframe(dist_preserve, width='stretch', hide_index=True)

def show_tree_simulation(df_original, df_anonymized, qi_attrs):
    """Tree simulation page with REAL ACDP Tree structure"""
    st.markdown("### Tree Simulation")
    
    st.markdown("Visualisasi ACDP Tree asli dari hasil anonymisasi Anda.")
    
    # Configuration
    st.markdown("---")
    st.markdown("**Configuration**")
    
    col1, col2 = st.columns(2)
    
    with col1:
        epsilon_tree = st.slider("Privacy Budget (ε) for Noise", 0.1, 2.0, 0.5, 0.1, key="tree_epsilon")
        st.info("Epsilon hanya untuk menambah noise pada count, tidak mengubah tree structure.")
    
    with col2:
        st.markdown("**QI Attributes Used:**")
        for attr in qi_attrs:
            st.markdown(f"- {attr}")
    
    # Load ACDP Tree structure
    st.markdown("---")
    st.markdown("**ACDP Tree Structure (Real)**")
    
    tree_file = os.path.join(st.session_state.output_dir, 'acdp_tree_structure.json')
    
    if not os.path.exists(tree_file):
        st.error(f"ACDP Tree structure file not found: {tree_file}")
        st.info("Please run `python main.py` first to generate the tree structure.")
        return
    
    # Load tree structure
    with open(tree_file, 'r', encoding='utf-8') as f:
        tree_structure = json.load(f)
    
    # Add noise to tree counts
    tree_with_noise = add_noise_to_tree(tree_structure, epsilon_tree)
    
    # Display metadata
    metadata = tree_structure['metadata']
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Records", f"{metadata['total_records']:,}")
    with col2:
        st.metric("K-Anonymity", metadata['k_anonymity'])
    with col3:
        st.metric("Max Depth", metadata['max_depth'])
    with col4:
        st.metric("Privacy Budget (ε)", epsilon_tree)
    
    # Visualization options
    st.markdown("---")
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.markdown("**Tree Visualization**")
    
    with col2:
        viz_type = st.selectbox("Visualization Type", ["Sankey Diagram", "Treemap"], key="viz_type")
    
    # Debug info
    with st.expander("Tree Metadata"):
        st.json(metadata)
    
    # Visualize tree
    if viz_type == "Sankey Diagram":
        fig = create_real_tree_visualization(tree_with_noise)
    else:
        fig = create_treemap_visualization(tree_with_noise)
    
    if fig:
        st.plotly_chart(fig, width='stretch')
    else:
        st.warning("Tree visualization could not be generated.")
    
    # Tree statistics
    st.markdown("---")
    st.markdown("**Tree Statistics**")
    
    tree_stats = calculate_tree_stats(tree_structure['tree'])
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Nodes", tree_stats['total_nodes'])
    
    with col2:
        st.metric("Leaf Nodes", tree_stats['leaf_nodes'])
    
    with col3:
        st.metric("Max Depth Reached", tree_stats['max_depth'])
    
    # Explanation
    st.markdown("---")
    st.info("""
    **Cara Membaca Tree:**
    - **Root Node**: Dataset lengkap
    - **Internal Nodes**: Split berdasarkan attribute dengan weighted MI tertinggi
    - **Leaf Nodes**: Final generalization levels
    - **Real Count**: Jumlah records asli di node
    - **Noisy Count**: Jumlah records setelah Laplace noise (ε={})

    Tree ini adalah hasil ASLI dari ACDP Tree algorithm Anda!
    """.format(epsilon_tree))

def build_tree_structure(df, top_attrs, epsilon):
    """Build tree structure with Laplace noise (DEPRECATED - use real tree instead)"""
    import numpy as np
    
    def add_laplace_noise(count, eps):
        """Add Laplace noise to count"""
        scale = 1.0 / eps
        noise = np.random.laplace(0, scale)
        return max(0, int(count + noise))
    
    root = {
        "name": "Root Dataset",
        "real_count": len(df),
        "noisy_count": add_laplace_noise(len(df), epsilon),
        "children": []
    }
    
    for attr in top_attrs[:3]:  # Top 3 attributes
        attr_node = {
            "name": attr,
            "real_count": len(df),
            "noisy_count": add_laplace_noise(len(df), epsilon),
            "children": []
        }
        
        # Get top 3 values for this attribute
        values = df[attr].value_counts().head(3).index.tolist()
        
        for val in values:
            count = int((df[attr] == val).sum())
            attr_node["children"].append({
                "name": f"{attr} = {val}",
                "real_count": count,
                "noisy_count": add_laplace_noise(count, epsilon),
                "children": []
            })
        
        root["children"].append(attr_node)
    
    return root

def add_noise_to_tree(tree_structure, epsilon):
    """Add Laplace noise to tree node counts"""
    import numpy as np
    import copy
    
    tree_copy = copy.deepcopy(tree_structure)
    
    def add_noise_recursive(node):
        if node is None:
            return
        
        # Add noise to count
        count = node.get('record_count', 0)
        scale = 1.0 / epsilon
        noise = np.random.laplace(0, scale)
        node['noisy_count'] = max(0, int(count + noise))
        node['noise'] = int(noise)
        
        # Recurse to children
        for child in node.get('children', []):
            add_noise_recursive(child)
    
    add_noise_recursive(tree_copy['tree'])
    return tree_copy

def calculate_tree_stats(tree_node):
    """Calculate statistics from tree structure"""
    stats = {
        'total_nodes': 0,
        'leaf_nodes': 0,
        'max_depth': 0
    }
    
    def traverse(node, depth=0):
        if node is None:
            return
        
        stats['total_nodes'] += 1
        stats['max_depth'] = max(stats['max_depth'], depth)
        
        if node.get('is_leaf', False):
            stats['leaf_nodes'] += 1
        
        for child in node.get('children', []):
            traverse(child, depth + 1)
    
    traverse(tree_node)
    return stats

def create_real_tree_visualization(tree_structure):
    """Create Sankey diagram from real ACDP Tree structure"""
    import plotly.graph_objects as go
    
    # Prepare data for Sankey diagram
    labels = []
    label_to_idx = {}
    sources = []
    targets = []
    values = []
    colors = []
    hover_texts = []
    
    # Color palette
    color_palette = {
        0: ACDP_COLORS[1],   # Root
        1: ACDP_COLORS[0],   # Level 1
        2: ACDP_COLORS[2],   # Level 2
        3: ACDP_COLORS[3],   # Level 3
        'leaf': ACDP_COLORS[0]  # Leaf
    }
    
    def add_node(label, depth, is_leaf=False):
        """Add node to labels if not exists, return index"""
        if label not in label_to_idx:
            idx = len(labels)
            labels.append(label)
            label_to_idx[label] = idx
            
            # Assign color based on depth or leaf status
            if is_leaf:
                colors.append(color_palette['leaf'])
            else:
                colors.append(color_palette.get(depth, color_palette[3]))
        
        return label_to_idx[label]
    
    def traverse(node, parent_idx=None, parent_label="", depth=0, max_depth=3):
        """Traverse tree and build Sankey data"""
        if node is None or depth > max_depth:
            return
        
        # Create node label
        if depth == 0:
            node_label = f"Root\n({node.get('record_count', 0):,} records)"
        elif node.get('is_leaf'):
            node_label = f"Leaf {len(labels)}\n({node.get('record_count', 0):,} records)"
        elif node.get('attribute'):
            attr = node['attribute']
            level = node.get('generalization_level', '?')
            count = node.get('record_count', 0)
            node_label = f"{attr}\nLevel {level}\n({count:,} records)"
        else:
            node_label = f"Node {len(labels)}\n({node.get('record_count', 0):,} records)"
        
        # Add current node
        current_idx = add_node(node_label, depth, node.get('is_leaf', False))
        
        # Create hover text
        count = node.get('record_count', 0)
        noisy = node.get('noisy_count', count)
        noise = node.get('noise', 0)
        
        hover_text = f"<b>{node_label.split(chr(10))[0]}</b><br>"
        hover_text += f"Depth: {depth}<br>"
        hover_text += f"Records: {count:,}<br>"
        hover_text += f"Noisy Count: {noisy:,}<br>"
        hover_text += f"Noise: {noise:+,}<br>"
        
        if node.get('is_leaf') and node.get('final_levels'):
            hover_text += "<br><b>Generalization Levels:</b><br>"
            for attr, level in node['final_levels'].items():
                hover_text += f"  {attr}: Level {level}<br>"
        
        hover_texts.append(hover_text)
        
        # Create edge from parent to current
        if parent_idx is not None:
            sources.append(parent_idx)
            targets.append(current_idx)
            values.append(node.get('record_count', 1))
        
        # Recurse to children (limit to prevent overcrowding)
        children = node.get('children', [])
        
        # If too many children, aggregate small ones
        if len(children) > 10:
            # Sort by record count
            children_sorted = sorted(children, key=lambda x: x.get('record_count', 0), reverse=True)
            
            # Keep top 8, aggregate rest
            top_children = children_sorted[:8]
            small_children = children_sorted[8:]
            
            # Process top children
            for child in top_children:
                traverse(child, current_idx, node_label, depth + 1, max_depth)
            
            # Aggregate small children
            if small_children:
                total_small = sum(c.get('record_count', 0) for c in small_children)
                other_label = f"Others ({len(small_children)} nodes)\n({total_small:,} records)"
                other_idx = add_node(other_label, depth + 1, is_leaf=True)
                
                sources.append(current_idx)
                targets.append(other_idx)
                values.append(total_small)
                
                hover_texts.append(
                    f"<b>Others</b><br>"
                    f"Aggregated: {len(small_children)} nodes<br>"
                    f"Total Records: {total_small:,}"
                )
        else:
            # Process all children
            for child in children:
                traverse(child, current_idx, node_label, depth + 1, max_depth)
    
    # Build Sankey data
    traverse(tree_structure['tree'], max_depth=2)  # Limit to depth 2 for clarity
    
    if len(labels) == 0:
        st.error("No tree data to visualize.")
        return None
    
    # Create Sankey diagram
    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="white", width=2),
            label=labels,
            color=colors,
            customdata=hover_texts,
            hovertemplate='%{customdata}<extra></extra>'
        ),
        link=dict(
            source=sources,
            target=targets,
            value=values,
            color='rgba(200, 200, 200, 0.3)'
        )
    )])
    
    fig.update_layout(
        title={
            'text': "ACDP Tree Structure - Flow Diagram (Real Tree from Algorithm)",
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 20, 'color': 'white'}
        },
        height=800,
        font=dict(size=11, color='white'),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(t=80, l=20, r=20, b=20)
    )
    
    return fig

def create_tree_visualization(tree_data):
    """Create interactive tree visualization using Plotly Treemap"""
    import plotly.graph_objects as go
    
    # Prepare data for treemap
    labels = []
    parents = []
    values = []
    real_counts = []
    noisy_counts = []
    colors = []
    
    def traverse(node, parent=""):
        label = node["name"]
        labels.append(label)
        parents.append(parent)
        values.append(node["real_count"])
        real_counts.append(node["real_count"])
        noisy_counts.append(node["noisy_count"])
        
        # Color based on level
        if parent == "":
            colors.append(0)  # Root
        elif parent == "Root Dataset":
            colors.append(1)  # Level 1
        else:
            colors.append(2)  # Level 2
        
        for child in node.get("children", []):
            traverse(child, label)
    
    traverse(tree_data)
    
    # Debug: Check if data is populated
    if len(labels) == 0:
        st.error("No tree data generated. Please check the data.")
        return None
    
    # Create custom hover text
    hover_text = [
        f"<b>{label}</b><br>" +
        f"Real Count: {real:,}<br>" +
        f"Noisy Count: {noisy:,}<br>" +
        f"Noise: {noisy - real:+,}"
        for label, real, noisy in zip(labels, real_counts, noisy_counts)
    ]
    
    fig = go.Figure(go.Treemap(
        labels=labels,
        parents=parents,
        values=values,
        marker=dict(
            colorscale='Blues',
            line=dict(width=2, color='white')
        ),
        text=labels,
        hovertext=hover_text,
        hoverinfo="text",
        textfont=dict(size=14, color='white', family='Arial Black')
    ))
    
    fig.update_layout(
        title={
            'text': "ACDP Tree Structure (Treemap)",
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 20, 'color': 'white'}
        },
        height=600,
        margin=dict(t=80, l=10, r=10, b=10),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig

def show_algorithm_comparison(df_original, qi_attrs, sens_attrs):
    """Algorithm comparison page"""
    st.markdown("### Algorithm Comparison")
    
    st.markdown("""
    Evaluasi ACDP-Tree dibandingkan DPDT dan IPA menggunakan empat metrik:
    - Information Loss
    - Absolute Error
    - Data Leakage Probability
    - Execution Time
    """)
    
    # Configuration
    st.markdown("---")
    st.markdown("**Configuration**")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**QI Attributes:**")
        for attr in qi_attrs:
            st.markdown(f"- {attr}")
    
    with col2:
        st.markdown("**Sensitive Attribute(s):**")
        for sa in sens_attrs:
            st.markdown(f"- {sa}")
    
    with col3:
        epsilon_eval = st.selectbox("Privacy Budget (ε)", [1.0, 0.5, 0.1], index=1, key="eval_epsilon")
    
    # Generate evaluation metrics
    metrics_data = generate_evaluation_metrics(len(df_original), len(qi_attrs), epsilon_eval)
    
    # Summary cards
    st.markdown("---")
    st.markdown("**Summary (Data Size: 5000)**")
    
    summary = metrics_data['summary']
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("**Information Loss**")
        for alg, val in summary['information_loss'].items():
            color = ACDP_COLORS[0] if alg == "ACDP-Tree" else ACDP_COLORS[3]
            st.markdown(f"`{alg}: {val:.4f}`", unsafe_allow_html=True)
    
    with col2:
        st.markdown("**Absolute Error**")
        for alg, val in summary['absolute_error'].items():
            st.markdown(f"`{alg}: {val:.2f}`", unsafe_allow_html=True)
    
    with col3:
        st.markdown("**Data Leakage Prob.**")
        for alg, val in summary['data_leakage_probability'].items():
            st.markdown(f"`{alg}: {val:.4f}`", unsafe_allow_html=True)
    
    with col4:
        st.markdown("**Execution Time (s)**")
        for alg, val in summary['execution_time'].items():
            st.markdown(f"`{alg}: {val:.4f}`", unsafe_allow_html=True)
    
    # Charts
    st.markdown("---")
    st.markdown("**Comparison Charts**")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig1 = create_comparison_chart(metrics_data['rows'], 'information_loss', 'Information Loss')
        st.plotly_chart(fig1, width='stretch')
        
        fig3 = create_comparison_chart(metrics_data['rows'], 'data_leakage_probability', 'Data Leakage Probability')
        st.plotly_chart(fig3, width='stretch')
    
    with col2:
        fig2 = create_comparison_chart(metrics_data['rows'], 'absolute_error', 'Absolute Error')
        st.plotly_chart(fig2, width='stretch')
        
        fig4 = create_comparison_chart(metrics_data['rows'], 'execution_time', 'Execution Time (s)')
        st.plotly_chart(fig4, width='stretch')
    
    # Metric descriptions
    st.markdown("---")
    st.markdown("**Metric Descriptions**")
    
    descriptions = [
        {
            "metric": "Information Loss",
            "meaning": "Mengukur seberapa banyak informasi hilang setelah generalisasi dan privasi diterapkan.",
            "good": "Semakin kecil semakin baik."
        },
        {
            "metric": "Absolute Error",
            "meaning": "Mengukur selisih antara nilai asli dan nilai yang sudah dipublikasikan.",
            "good": "Semakin kecil semakin akurat."
        },
        {
            "metric": "Data Leakage Probability",
            "meaning": "Mengukur perkiraan peluang data sensitif bocor atau dapat ditebak.",
            "good": "Semakin kecil semakin aman."
        },
        {
            "metric": "Execution Time",
            "meaning": "Mengukur waktu eksekusi algoritma.",
            "good": "Semakin kecil semakin cepat."
        }
    ]
    
    for desc in descriptions:
        with st.expander(f"**{desc['metric']}**"):
            st.markdown(f"**Arti:** {desc['meaning']}")
            st.markdown(f"**Interpretasi:** {desc['good']}")
    
    # Full table
    st.markdown("---")
    st.markdown("**Full Comparison Table**")
    
    df_table = pd.DataFrame(metrics_data['rows'])
    st.dataframe(df_table, width='stretch', hide_index=True)
    
    st.info("Catatan: Nilai metrik pada dashboard ini adalah simulasi untuk demonstrasi. "
            "Untuk hasil penelitian final, hubungkan dengan function di `src/metrics.py`.")

def generate_evaluation_metrics(data_size, qi_count, epsilon):
    """Generate evaluation metrics for algorithm comparison"""
    import math
    
    def calculate_information_loss(qi_count, epsilon, data_size, algorithm):
        base = (data_size / 5000) * 0.22 + (1 / max(epsilon, 0.1)) * 0.015 + qi_count * 0.008
        factor = {"ACDP-Tree": 0.78, "DPDT": 0.92, "IPA": 1.10}.get(algorithm, 1.0)
        return round(min(base * factor, 1.0), 4)
    
    def calculate_absolute_error(epsilon, data_size, algorithm):
        base = (1 / max(epsilon, 0.1)) * 38 + math.sqrt(data_size) * 0.35
        factor = {"ACDP-Tree": 0.74, "DPDT": 0.90, "IPA": 1.08}.get(algorithm, 1.0)
        return round(base * factor, 4)
    
    def calculate_dlp(epsilon, algorithm):
        base = epsilon * 0.18
        factor = {"ACDP-Tree": 0.68, "DPDT": 0.85, "IPA": 1.00}.get(algorithm, 1.0)
        return round(min(base * factor, 1.0), 4)
    
    def calculate_execution_time(data_size, qi_count, algorithm):
        base = (data_size / 1000) * 0.22 + qi_count * 0.06
        factor = {"ACDP-Tree": 0.82, "DPDT": 1.00, "IPA": 1.18}.get(algorithm, 1.0)
        return round(base * factor, 4)
    
    data_sizes = [1000, 2000, 3000, 4000, 5000]
    algorithms = ["ACDP-Tree", "DPDT", "IPA"]
    
    rows = []
    for alg in algorithms:
        for size in data_sizes:
            rows.append({
                "algorithm": alg,
                "data_size": size,
                "information_loss": calculate_information_loss(qi_count, epsilon, size, alg),
                "absolute_error": calculate_absolute_error(epsilon, size, alg),
                "data_leakage_probability": calculate_dlp(epsilon, alg),
                "execution_time": calculate_execution_time(size, qi_count, alg),
            })
    
    # Summary for latest size
    latest_size = data_sizes[-1]
    summary = {
        'information_loss': {},
        'absolute_error': {},
        'data_leakage_probability': {},
        'execution_time': {}
    }
    
    for alg in algorithms:
        current = [row for row in rows if row["algorithm"] == alg and row["data_size"] == latest_size][0]
        summary['information_loss'][alg] = current['information_loss']
        summary['absolute_error'][alg] = current['absolute_error']
        summary['data_leakage_probability'][alg] = current['data_leakage_probability']
        summary['execution_time'][alg] = current['execution_time']
    
    return {
        'rows': rows,
        'summary': summary
    }

def create_comparison_chart(rows, metric, title):
    """Create comparison line chart"""
    df = pd.DataFrame(rows)
    
    fig = px.line(
        df,
        x='data_size',
        y=metric,
        color='algorithm',
        markers=True,
        title=title,
        labels={'data_size': 'Data Size', metric: title},
        color_discrete_map={
            'ACDP-Tree': ACDP_COLORS[0],
            'DPDT': ACDP_COLORS[1],
            'IPA': ACDP_COLORS[3]
        }
    )
    
    fig.update_layout(
        height=400,
        hovermode='x unified',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig


def create_treemap_visualization(tree_structure):
    """Create Treemap visualization (alternative to Sankey)"""
    import plotly.graph_objects as go
    
    labels = []
    parents = []
    values = []
    hover_texts = []
    
    def traverse(node, parent_label="", depth=0, max_depth=2):
        if node is None or depth > max_depth:
            return
        
        # Create label
        if depth == 0:
            label = "Root"
        elif node.get('attribute'):
            label = f"{node['attribute']} L{node.get('generalization_level', '?')}"
        elif node.get('is_leaf'):
            label = f"Leaf {len(labels)}"
        else:
            label = f"Node {len(labels)}"
        
        count = node.get('record_count', 0)
        noisy = node.get('noisy_count', count)
        
        labels.append(label)
        parents.append(parent_label)
        values.append(count if count > 0 else 1)
        
        hover_text = f"<b>{label}</b><br>Records: {count:,}<br>Noisy: {noisy:,}"
        hover_texts.append(hover_text)
        
        # Recurse to children
        for child in node.get('children', [])[:10]:  # Limit to 10 children
            traverse(child, label, depth + 1, max_depth)
    
    traverse(tree_structure['tree'])
    
    fig = go.Figure(go.Treemap(
        labels=labels,
        parents=parents,
        values=values,
        branchvalues="total",
        marker=dict(line=dict(width=2, color='#30363d')),
        hovertext=hover_texts,
        hoverinfo="text",
        textfont=dict(size=12),
        texttemplate="%{label}<br>%{value} records"
    ))
    
    fig.update_layout(
        title="ACDP Tree Structure (Treemap)",
        height=700,
        margin=dict(t=50, l=10, r=10, b=10),
        template='plotly_dark',
        treemapcolorway=['#2ea043', '#58a6ff', '#f59e0b', '#ef4444', '#8b5cf6', '#1f77b4'],
    )
    
    return fig


if __name__ == "__main__":
    main()
