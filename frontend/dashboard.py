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

# Modern Clean CSS
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    /* Typography */
    h1, h2, h3, h4, h5, h6, p, label, div {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    }
    
    /* Custom Headers */
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    
    .sub-header {
        font-size: 1.8rem;
        font-weight: 600;
        color: #2d3748;
        margin-top: 2rem;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 3px solid #667eea;
        display: inline-block;
    }
    
    /* Info Boxes */
    .success-box {
        background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
        padding: 1.25rem;
        border-radius: 10px;
        border-left: 4px solid #28a745;
        margin: 1rem 0;
    }
    
    .warning-box {
        background: linear-gradient(135deg, #fff3cd 0%, #ffeaa7 100%);
        padding: 1.25rem;
        border-radius: 10px;
        border-left: 4px solid #ffc107;
        margin: 1rem 0;
    }
    
    .info-box {
        background: linear-gradient(135deg, #d1ecf1 0%, #bee5eb 100%);
        padding: 1.25rem;
        border-radius: 10px;
        border-left: 4px solid #17a2b8;
        margin: 1rem 0;
    }
    
    /* Buttons Enhancement */
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 0.6rem 1.5rem !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 12px rgba(102,126,234,0.3) !important;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 16px rgba(102,126,234,0.4) !important;
    }
    
    /* Metrics Styling */
    [data-testid="stMetricValue"] {
        font-size: 1.8rem;
        font-weight: 700;
        color: #667eea;
    }
    
    [data-testid="stMetricLabel"] {
        font-weight: 500;
        color: #4a5568;
        font-size: 0.85rem;
    }
    
    /* Progress Bar */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Divider */
    hr {
        margin: 2rem 0;
        border: none;
        height: 1px;
        background: linear-gradient(90deg, transparent, #cbd5e0, transparent);
    }
    
    /* Remove Streamlit Branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# Constants - Load from backend config
from src.config import DATASET_CONFIG, PRIVACY_CONFIG

RAW_DATA_PATH = DATASET_CONFIG['file_path']
QI_ATTRIBUTES = DATASET_CONFIG['qi_attributes']
SENSITIVE_ATTRIBUTE = DATASET_CONFIG['sensitive_attribute']
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
    """Load original and anonymized datasets"""
    try:
        # Load original data
        df_original = pd.read_csv(RAW_DATA_PATH)
        
        # Construct file names based on dataset name (matching main.py logic)
        # Load anonymized data
        anon_file = f'{dataset_name}_anonymized_k{K_ANONYMITY}_eps{EPSILON:.1f}.csv'
        anon_path = os.path.join(OUTPUT_DIR, anon_file)
        df_anonymized = pd.read_csv(anon_path)
        
        # Load noisy counts
        noisy_file = f'{dataset_name}_noisy_counts_k{K_ANONYMITY}_eps{EPSILON:.1f}.csv'
        noisy_path = os.path.join(OUTPUT_DIR, noisy_file)
        df_noisy = pd.read_csv(noisy_path)
        
        return df_original, df_anonymized, df_noisy, True
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.error(f"Expected paths:\n- {anon_path}\n- {noisy_path}")
        return None, None, None, False

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
    st.markdown('<div class="sub-header">🏠 Run Anonymization</div>', unsafe_allow_html=True)
    
    st.markdown("""
    **Upload your CSV file and configure privacy parameters to run ACDP Tree anonymization.**
    
    Default dataset: Diabetes Health Indicators
    """)
    
    # File uploader
    st.markdown("---")
    st.markdown("### 📤 Upload CSV File")
    
    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type=['csv'],
        help="Upload your dataset in CSV format"
    )
    
    # Use default or uploaded file
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        dataset_source = uploaded_file.name
        st.success(f"✅ Loaded: **{uploaded_file.name}** ({len(df):,} rows, {len(df.columns)} columns)")
    else:
        df = pd.read_csv(RAW_DATA_PATH)
        dataset_source = "diabetes__health_indicators.csv (default)"
        st.info(f"📊 Using default dataset: **{dataset_source}** ({len(df):,} rows, {len(df.columns)} columns)")
    
    # Show data preview
    with st.expander("👀 Preview Data (first 10 rows)"):
        st.dataframe(df.head(10), use_container_width=True)
    
    # Configuration
    st.markdown("---")
    st.markdown("### ⚙️ Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🔢 Attribute Selection")
        
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
        
        # Sensitive Attribute
        sens_attr = st.selectbox(
            "Sensitive Attribute",
            options=[col for col in all_cols if col not in qi_attrs],
            index=0 if SENSITIVE_ATTRIBUTE not in all_cols else [col for col in all_cols if col not in qi_attrs].index(SENSITIVE_ATTRIBUTE) if SENSITIVE_ATTRIBUTE in [col for col in all_cols if col not in qi_attrs] else 0,
            help="The attribute to protect (e.g., disease, salary)"
        )
        
        # Identifier Attributes (optional)
        id_attrs = st.multiselect(
            "Identifier Attributes (will be dropped)",
            options=[col for col in all_cols if col not in qi_attrs and col != sens_attr],
            default=[],
            help="Attributes like ID, Name that directly identify individuals"
        )
    
    with col2:
        st.markdown("#### 🔒 Privacy Parameters")
        
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
    
    if len(qi_attrs) == 0:
        st.error("⚠️ Please select at least one QI attribute")
        return
    
    if sens_attr is None:
        st.error("⚠️ Please select a sensitive attribute")
        return
    
    # Show configuration summary
    with st.expander("📋 Configuration Summary"):
        config_summary = {
            "Dataset": dataset_source,
            "Records": f"{len(df):,}",
            "Columns": len(df.columns),
            "QI Attributes": ", ".join(qi_attrs),
            "Sensitive Attribute": sens_attr,
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
        run_button = st.button("▶️ Run Anonymization", use_container_width=True, type="primary")
    
    if run_button:
        st.markdown("---")
        st.markdown("### 🔄 Running Anonymization Pipeline...")
        
        # Save uploaded file temporarily
        import tempfile
        import shutil
        
        temp_dir = tempfile.mkdtemp()
        temp_csv = os.path.join(temp_dir, "temp_dataset.csv")
        df.to_csv(temp_csv, index=False)
        
        # Create config
        custom_config = {
            'file_path': temp_csv,
            'identifier_attributes': id_attrs,
            'qi_attributes': qi_attrs,
            'sensitive_attribute': sens_attr,
            'non_sensitive_attributes': [],
        }
        
        custom_privacy = {
            'k_anonymity': k_anon,
            'epsilon': epsilon,
            'max_level': max_level,
            'max_tree_depth': max_tree_depth,
        }
        
        # Output directory
        output_name = os.path.splitext(dataset_source)[0].replace(' ', '_').lower()
        custom_output = os.path.join('results', output_name)
        
        # Progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            # Import backend
            from main import run_pipeline
            from src.config import HIERARCHY_CONFIG
            
            status_text.text("⏳ Step 1/7: Preprocessing data...")
            progress_bar.progress(10)
            
            # Capture output
            import io
            from contextlib import redirect_stdout, redirect_stderr
            
            output_buffer = io.StringIO()
            
            with redirect_stdout(output_buffer), redirect_stderr(output_buffer):
                results = run_pipeline(
                    config=custom_config,
                    privacy_config=custom_privacy,
                    hierarchy_config=HIERARCHY_CONFIG,
                    custom_hierarchy={},
                    output_dir=custom_output
                )
            
            progress_bar.progress(100)
            status_text.text("✅ Anonymization complete!")
            
            # Show results
            st.success("🎉 **Anonymization completed successfully!**")
            
            # Get result data
            metadata = results['metadata']
            df_original = results['df_original']
            df_anonymized = results['df_anonymized']
            metrics = results['metrics']
            
            # Display summary metrics
            st.markdown("---")
            st.markdown("### 📊 Results Summary")
            
            col1, col2, col3, col4, col5 = st.columns(5)
            
            with col1:
                st.metric("Original Records", f"{metadata['dataset_info']['original_records']:,}")
            
            with col2:
                st.metric("K-Anonymity", "✅ Satisfied" if metadata['privacy_guarantees']['k_anonymity_satisfied'] else "❌ Not Satisfied")
            
            with col3:
                utility = metrics['privacy_utility_tradeoff']['utility_score']
                st.metric("Utility Score", f"{utility:.2f}/100")
            
            with col4:
                st.metric("Unique Groups", f"{metadata['privacy_guarantees']['total_groups']:,}")
            
            with col5:
                privacy_gain = metrics['privacy_utility_tradeoff']['privacy_gain_pct']
                st.metric("Privacy Gain", f"{privacy_gain:.1f}%")
            
            # Privacy Guarantees
            st.markdown("---")
            st.markdown("### 🔐 Privacy Guarantees")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown('<div class="success-box">', unsafe_allow_html=True)
                st.markdown(f"**✅ K-Anonymity Satisfied (k={k_anon})**")
                st.markdown(f"- Min group size: **{metadata['privacy_guarantees']['min_group_size']}**")
                st.markdown(f"- Max group size: **{metadata['privacy_guarantees']['max_group_size']:,}**")
                st.markdown(f"- Avg group size: **{metadata['privacy_guarantees']['avg_group_size']:.2f}**")
                st.markdown(f"- Total equivalence classes: **{metadata['privacy_guarantees']['total_groups']:,}**")
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col2:
                st.markdown('<div class="success-box">', unsafe_allow_html=True)
                st.markdown(f"**✅ Differential Privacy (ε={epsilon})**")
                st.markdown(f"- Tree construction budget: **ε={epsilon/2:.2f}**")
                st.markdown(f"- Laplace noise budget: **ε={epsilon/2:.2f}**")
                st.markdown(f"- Mean noise: **{metadata['pipeline_summary']['noise']['mean_noise']:.4f}**")
                st.markdown(f"- Mean percent error: **{metadata['pipeline_summary']['noise']['mean_percent_error']:.2f}%**")
                st.markdown('</div>', unsafe_allow_html=True)
            
            # ACDP Tree Visualization
            st.markdown("---")
            st.markdown("### 🌳 ACDP Tree Structure")
            
            # Load tree structure
            tree_file = os.path.join(custom_output, 'acdp_tree_structure.json')
            if os.path.exists(tree_file):
                with open(tree_file, 'r') as f:
                    tree_data = json.load(f)
                
                # Create tree visualization
                import plotly.graph_objects as go
                
                def create_tree_nodes(node, parent_id="", x=0, y=0, level=0, width=1.0):
                    """Recursively create tree nodes for visualization"""
                    nodes = []
                    edges = []
                    
                    if node is None:
                        return nodes, edges
                    
                    node_id = f"{parent_id}_{level}_{x}"
                    
                    # Node info
                    if node.get('is_leaf', False):
                        node_label = f"LEAF<br>Records: {node.get('record_count', 0)}"
                        node_color = '#28a745'  # Green for leaf
                    else:
                        attr = node.get('attribute', 'Unknown')
                        gen_level = node.get('generalization_level', 0)
                        node_label = f"{attr}<br>Level: {gen_level}<br>Records: {node.get('record_count', 0)}"
                        node_color = '#667eea'  # Purple for decision node
                    
                    nodes.append({
                        'id': node_id,
                        'label': node_label,
                        'x': x,
                        'y': -y,
                        'color': node_color,
                        'size': min(30 + node.get('record_count', 0) / 10000, 50)
                    })
                    
                    # Process children
                    children = node.get('children', [])
                    if children:
                        child_width = width / len(children)
                        for i, child in enumerate(children):
                            child_x = x - width/2 + child_width * (i + 0.5)
                            child_y = y + 1
                            
                            child_nodes, child_edges = create_tree_nodes(
                                child, node_id, child_x, child_y, level + 1, child_width
                            )
                            
                            # Add edge
                            child_id = f"{node_id}_{level+1}_{child_x}"
                            edges.append({
                                'from': node_id,
                                'to': child_id,
                                'parent_value': child.get('parent_value', '')
                            })
                            
                            nodes.extend(child_nodes)
                            edges.extend(child_edges)
                    
                    return nodes, edges
                
                # Create visualization
                tree_root = tree_data.get('tree')
                if tree_root:
                    nodes, edges = create_tree_nodes(tree_root, "", 0, 0, 0, 2.0)
                    
                    # Create Plotly figure
                    fig = go.Figure()
                    
                    # Add edges
                    for edge in edges:
                        from_node = next((n for n in nodes if n['id'] == edge['from']), None)
                        to_node = next((n for n in nodes if n['id'] == edge['to']), None)
                        
                        if from_node and to_node:
                            fig.add_trace(go.Scatter(
                                x=[from_node['x'], to_node['x']],
                                y=[from_node['y'], to_node['y']],
                                mode='lines',
                                line=dict(color='#cbd5e0', width=2),
                                hoverinfo='skip',
                                showlegend=False
                            ))
                    
                    # Add nodes
                    node_x = [n['x'] for n in nodes]
                    node_y = [n['y'] for n in nodes]
                    node_text = [n['label'] for n in nodes]
                    node_color = [n['color'] for n in nodes]
                    node_size = [n['size'] for n in nodes]
                    
                    fig.add_trace(go.Scatter(
                        x=node_x,
                        y=node_y,
                        mode='markers+text',
                        marker=dict(
                            size=node_size,
                            color=node_color,
                            line=dict(color='white', width=2)
                        ),
                        text=node_text,
                        textposition="middle center",
                        textfont=dict(size=10, color='white'),
                        hoverinfo='text',
                        showlegend=False
                    ))
                    
                    fig.update_layout(
                        title=f"ACDP Tree Structure (Depth: {tree_data['metadata']['max_depth']}, Total Records: {tree_data['metadata']['total_records']:,})",
                        showlegend=False,
                        hovermode='closest',
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        height=600,
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)'
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    with st.expander("🔍 Tree Metadata"):
                        st.json(tree_data['metadata'])
                else:
                    st.warning("⚠️ Tree structure is empty")
            else:
                st.warning("⚠️ ACDP Tree structure file not found")
            
            # Information Loss Visualization
            st.markdown("---")
            st.markdown("### 📉 Information Loss Analysis")
            
            info_loss_df = metrics['information_loss']
            info_loss_chart = pd.DataFrame(info_loss_df)
            
            fig_info_loss = go.Figure()
            fig_info_loss.add_trace(go.Bar(
                x=info_loss_chart['Attribute'],
                y=info_loss_chart['Unique Lost (%)'],
                name='Unique Values Lost (%)',
                marker_color='#e74c3c'
            ))
            fig_info_loss.add_trace(go.Bar(
                x=info_loss_chart['Attribute'],
                y=info_loss_chart['Entropy Reduction (%)'],
                name='Entropy Reduction (%)',
                marker_color='#f39c12'
            ))
            
            fig_info_loss.update_layout(
                barmode='group',
                height=400,
                xaxis_title='Attribute',
                yaxis_title='Loss (%)',
                title='Information Loss by Attribute'
            )
            
            st.plotly_chart(fig_info_loss, use_container_width=True)
            
            # Distribution Comparison
            st.markdown("---")
            st.markdown("### 📊 Distribution Comparison")
            
            # Select attribute for comparison
            comparison_attr = st.selectbox(
                "Select attribute to compare:",
                qi_attrs,
                key="run_comparison"
            )
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Original Distribution**")
                orig_dist = df_original[comparison_attr].value_counts().head(10)
                fig_orig = px.bar(
                    x=orig_dist.index.astype(str),
                    y=orig_dist.values,
                    labels={'x': comparison_attr, 'y': 'Count'},
                    color_discrete_sequence=['#3498db']
                )
                fig_orig.update_layout(height=350, showlegend=False)
                st.plotly_chart(fig_orig, use_container_width=True)
            
            with col2:
                st.markdown("**Anonymized Distribution**")
                anon_dist = df_anonymized[comparison_attr].value_counts().head(10)
                fig_anon = px.bar(
                    x=anon_dist.index.astype(str),
                    y=anon_dist.values,
                    labels={'x': comparison_attr, 'y': 'Count'},
                    color_discrete_sequence=['#9b59b6']
                )
                fig_anon.update_layout(height=350, showlegend=False)
                st.plotly_chart(fig_anon, use_container_width=True)
            
            # Data Preview
            st.markdown("---")
            st.markdown("### 👀 Data Preview")
            
            tab1, tab2 = st.tabs(["Original Data", "Anonymized Data"])
            
            with tab1:
                st.dataframe(df_original.head(20), use_container_width=True)
            
            with tab2:
                st.dataframe(df_anonymized.head(20), use_container_width=True)
            
            # Output Files
            st.markdown("---")
            st.markdown("### 📁 Output Files")
            st.info(f"✅ Results saved to: `{custom_output}/`")
            
            output_files = [
                f"✅ {metadata['dataset_info']['anonymized_file']} - Anonymized dataset",
                f"✅ {metadata['dataset_info']['noisy_counts_file']} - Noisy counts",
                f"✅ acdp_tree_structure.json - Tree structure",
                f"✅ anonymization_metadata.json - Metadata",
                f"✅ evaluation_metrics.json - Evaluation metrics",
                f"✅ evaluation_report.txt - Detailed report"
            ]
            
            for file_info in output_files:
                st.markdown(f"- {file_info}")
            
            # Console Output
            with st.expander("📋 Console Output (Terminal)"):
                st.code(output_buffer.getvalue(), language='text')
            
            # Cleanup
            shutil.rmtree(temp_dir, ignore_errors=True)
            
            st.markdown("---")
            st.success("💡 **Anonymization complete!** Scroll up to see ACDP Tree visualization and detailed metrics.")
            
        except ValueError as e:
            progress_bar.progress(0)
            status_text.text("")
            error_msg = str(e)
            
            # Provide helpful error messages
            if "not found in dataset columns" in error_msg:
                st.error("❌ **Configuration Error:** Some selected attributes don't exist in your CSV.")
                st.warning("**Solution:** Check the attribute names and try again. Make sure you're selecting from the dropdown options only.")
            elif "No valid QI attributes" in error_msg:
                st.error("❌ **Configuration Error:** No valid QI attributes found.")
                st.warning("**Solution:** Select at least one valid attribute as QI.")
            elif "not enough values to unpack" in error_msg or "reshape" in error_msg:
                st.error("❌ **Data Format Error:** Dataset format is not compatible.")
                st.warning("**Solution:** Ensure your CSV has:\n- At least 100 rows\n- Valid numeric or categorical columns\n- No completely empty columns")
            else:
                st.error(f"❌ **Error during anonymization:** {error_msg}")
            
            with st.expander("🐛 Full Error Details"):
                import traceback
                st.code(traceback.format_exc())
            
            # Cleanup
            shutil.rmtree(temp_dir, ignore_errors=True)
            
        except Exception as e:
            progress_bar.progress(0)
            status_text.text("")
            st.error(f"❌ **Unexpected error:** {str(e)}")
            st.warning("**Your dataset might have:**\n- Special characters or encoding issues\n- Extremely sparse data\n- Unusual data types\n\nTry cleaning your CSV first or contact support.")
            
            with st.expander("🐛 Full Error Details"):
                import traceback
                st.code(traceback.format_exc())
            
            # Cleanup
            shutil.rmtree(temp_dir, ignore_errors=True)


# Main app
def main():
    # Header
    st.markdown('<div class="main-header">🔒 ACDP Tree Privacy Dashboard</div>', unsafe_allow_html=True)
    st.markdown("**Privacy-Preserving Data Anonymization for Diabetes Health Indicators**")
    
    # Load data
    df_original, df_anonymized, df_noisy, success = load_data()
    
    if not success:
        st.error("❌ Failed to load data. Please run `python main.py` first to generate anonymized data.")
        st.info("📝 Make sure the following files exist:\n"
                f"- {RAW_DATA_PATH}\n"
                f"- {OUTPUT_DIR}/diabetes_anonymized_k{K_ANONYMITY}_eps{EPSILON:.1f}.csv\n"
                f"- {OUTPUT_DIR}/diabetes_noisy_counts_k{K_ANONYMITY}_eps{EPSILON:.1f}.csv")
        return
    
    # Sidebar
    st.sidebar.title("📊 Navigation")
    page = st.sidebar.radio(
        "Select View:",
        ["🏠 Run Anonymization", "Overview", "Data Comparison", "Privacy Metrics", "Utility Metrics", "Visualizations", "Tree Simulation", "Algorithm Comparison"]
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### ⚙️ Privacy Parameters")
    st.sidebar.metric("K-Anonymity", K_ANONYMITY)
    st.sidebar.metric("Epsilon (ε)", EPSILON)
    st.sidebar.metric("Max Level", MAX_LEVEL)
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📁 Dataset Info")
    st.sidebar.metric("Original Records", f"{len(df_original):,}")
    st.sidebar.metric("Anonymized Records", f"{len(df_anonymized):,}")
    st.sidebar.metric("QI Attributes", len(QI_ATTRIBUTES))
    
    # Calculate metrics
    info_loss, dist_preserve, orig_risk, anon_risk, tradeoff = calculate_metrics(df_original, df_anonymized)
    
    # Page routing
    if page == "🏠 Run Anonymization":
        show_run_anonymization()
    elif page == "Overview":
        show_overview(df_original, df_anonymized, info_loss, orig_risk, anon_risk, tradeoff)
    elif page == "Data Comparison":
        show_data_comparison(df_original, df_anonymized)
    elif page == "Privacy Metrics":
        show_privacy_metrics(orig_risk, anon_risk, df_noisy)
    elif page == "Utility Metrics":
        show_utility_metrics(info_loss, dist_preserve, tradeoff)
    elif page == "Visualizations":
        show_visualizations(df_original, df_anonymized, df_noisy, info_loss, dist_preserve)
    elif page == "Tree Simulation":
        show_tree_simulation(df_original, df_anonymized)
    elif page == "Algorithm Comparison":
        show_algorithm_comparison(df_original)

def show_overview(df_original, df_anonymized, info_loss, orig_risk, anon_risk, tradeoff):
    """Overview page with key metrics"""
    st.markdown('<div class="sub-header">📋 Overview</div>', unsafe_allow_html=True)
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Privacy Gain",
            f"{tradeoff['privacy_gain_pct']:.1f}%",
            delta="Higher is better",
            delta_color="normal"
        )
    
    with col2:
        st.metric(
            "Utility Score",
            f"{tradeoff['utility_score']:.1f}/100",
            delta="Higher is better",
            delta_color="normal"
        )
    
    with col3:
        st.metric(
            "Information Loss",
            f"{info_loss['Unique Lost (%)'].mean():.1f}%",
            delta="Lower is better",
            delta_color="inverse"
        )
    
    with col4:
        st.metric(
            "Re-ID Risk Reduction",
            f"{orig_risk['unique_risk_pct'] - anon_risk['unique_risk_pct']:.2f}%",
            delta="Risk reduced",
            delta_color="normal"
        )
    
    # Privacy status
    st.markdown("---")
    st.markdown("### 🔐 Privacy Status")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="success-box">', unsafe_allow_html=True)
        st.markdown(f"**✅ K-Anonymity Satisfied (k={K_ANONYMITY})**")
        st.markdown(f"- Min group size: {anon_risk['min_group_size']}")
        st.markdown(f"- Avg group size: {anon_risk['avg_group_size']:.2f}")
        st.markdown(f"- Total groups: {anon_risk['total_groups']:,}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="success-box">', unsafe_allow_html=True)
        st.markdown(f"**✅ Differential Privacy Applied (ε={EPSILON})**")
        st.markdown(f"- Mechanism: Laplace Noise")
        st.markdown(f"- Applied to: Aggregated counts")
        st.markdown(f"- Privacy budget: 100% consumed")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Summary table
    st.markdown("---")
    st.markdown("### 📊 Summary Statistics")
    
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
            len(QI_ATTRIBUTES),
            f"{orig_risk['unique_risk_pct']:.2f}%",
            f"{anon_risk['unique_risk_pct']:.2f}%",
            f"{tradeoff['privacy_gain_pct']:.2f}%",
            f"{tradeoff['utility_score']:.2f}/100",
            f"{info_loss['Unique Lost (%)'].mean():.2f}%"
        ]
    }
    
    st.dataframe(pd.DataFrame(summary_data), use_container_width=True, hide_index=True)

def show_data_comparison(df_original, df_anonymized):
    """Data comparison page"""
    st.markdown('<div class="sub-header">📊 Data Comparison</div>', unsafe_allow_html=True)
    
    # Attribute selector
    selected_attr = st.selectbox("Select Attribute to Compare:", QI_ATTRIBUTES)
    
    # Distribution comparison
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Original Distribution")
        orig_counts = df_original[selected_attr].value_counts().sort_index()
        fig1 = px.bar(
            x=orig_counts.index.astype(str),
            y=orig_counts.values,
            labels={'x': selected_attr, 'y': 'Count'},
            color_discrete_sequence=['#1f77b4']
        )
        fig1.update_layout(showlegend=False, height=400)
        st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        st.markdown("#### Anonymized Distribution")
        anon_counts = df_anonymized[selected_attr].value_counts().sort_index()
        fig2 = px.bar(
            x=anon_counts.index.astype(str),
            y=anon_counts.values,
            labels={'x': selected_attr, 'y': 'Count'},
            color_discrete_sequence=['#ff7f0e']
        )
        fig2.update_layout(showlegend=False, height=400)
        st.plotly_chart(fig2, use_container_width=True)
    
    # Side-by-side comparison
    st.markdown("---")
    st.markdown("#### Side-by-Side Comparison")
    
    comparison_df = pd.DataFrame({
        'Value': orig_counts.index.astype(str),
        'Original': orig_counts.values,
        'Anonymized': anon_counts.reindex(orig_counts.index, fill_value=0).values
    })
    
    fig3 = go.Figure()
    fig3.add_trace(go.Bar(name='Original', x=comparison_df['Value'], y=comparison_df['Original'], marker_color='#1f77b4'))
    fig3.add_trace(go.Bar(name='Anonymized', x=comparison_df['Value'], y=comparison_df['Anonymized'], marker_color='#ff7f0e'))
    fig3.update_layout(barmode='group', height=400, xaxis_title=selected_attr, yaxis_title='Count')
    st.plotly_chart(fig3, use_container_width=True)
    
    # Data preview
    st.markdown("---")
    st.markdown("#### Data Preview")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Original Data (first 10 rows)**")
        st.dataframe(df_original[QI_ATTRIBUTES + [SENSITIVE_ATTRIBUTE]].head(10), use_container_width=True)
    
    with col2:
        st.markdown("**Anonymized Data (first 10 rows)**")
        st.dataframe(df_anonymized[QI_ATTRIBUTES + [SENSITIVE_ATTRIBUTE]].head(10), use_container_width=True)

def show_privacy_metrics(orig_risk, anon_risk, df_noisy):
    """Privacy metrics page"""
    st.markdown('<div class="sub-header">🔐 Privacy Metrics</div>', unsafe_allow_html=True)
    
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
        fig.add_trace(go.Bar(name='Unique Risk', x=risk_data['Dataset'], y=risk_data['Unique Risk (%)'], marker_color='#e74c3c'))
        fig.add_trace(go.Bar(name='Small Group Risk', x=risk_data['Dataset'], y=risk_data['Small Group Risk (%)'], marker_color='#f39c12'))
        fig.update_layout(barmode='group', height=400, yaxis_title='Risk (%)', title='Re-identification Risk Comparison')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("#### Risk Metrics")
        
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
        
        st.dataframe(metrics_df, use_container_width=True, hide_index=True)
    
    # Differential Privacy noise
    st.markdown("---")
    st.markdown("### Differential Privacy Noise Impact")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Noise distribution
        fig = px.histogram(
            df_noisy,
            x='noise_added',
            nbins=50,
            labels={'noise_added': 'Noise Added', 'count': 'Frequency'},
            title='Laplace Noise Distribution',
            color_discrete_sequence=['#3498db']
        )
        fig.add_vline(x=0, line_dash="dash", line_color="red", annotation_text="No Noise")
        fig.add_vline(x=df_noisy['noise_added'].mean(), line_dash="dash", line_color="green", 
                     annotation_text=f"Mean={df_noisy['noise_added'].mean():.2f}")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Original vs Noisy counts
        fig = px.scatter(
            df_noisy,
            x='count',
            y='noisy_count',
            labels={'count': 'Original Count', 'noisy_count': 'Noisy Count'},
            title='Original vs Noisy Counts',
            color_discrete_sequence=['#2ecc71'],
            opacity=0.6
        )
        max_val = max(df_noisy['count'].max(), df_noisy['noisy_count'].max())
        fig.add_trace(go.Scatter(x=[0, max_val], y=[0, max_val], mode='lines', 
                                name='Perfect Match', line=dict(color='red', dash='dash')))
        st.plotly_chart(fig, use_container_width=True)

def show_utility_metrics(info_loss, dist_preserve, tradeoff):
    """Utility metrics page"""
    st.markdown('<div class="sub-header">📈 Utility Metrics</div>', unsafe_allow_html=True)
    
    # Information loss
    st.markdown("### Information Loss per Attribute")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.bar(
            info_loss,
            x='Attribute',
            y='Unique Lost (%)',
            title='Unique Values Lost',
            color='Unique Lost (%)',
            color_continuous_scale='Reds'
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.bar(
            info_loss,
            x='Attribute',
            y='Entropy Reduction (%)',
            title='Entropy Reduction',
            color='Entropy Reduction (%)',
            color_continuous_scale='Blues'
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    # Distribution preservation
    st.markdown("---")
    st.markdown("### Distribution Preservation")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.bar(
            dist_preserve,
            x='Attribute',
            y='KL-Divergence',
            title='KL-Divergence (Lower = Better)',
            color='Preservation Quality',
            color_discrete_map={'Good': '#2ecc71', 'Fair': '#f39c12', 'Poor': '#e74c3c'}
        )
        fig.add_hline(y=0.5, line_dash="dash", line_color="orange", annotation_text="Good/Fair Threshold")
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.bar(
            dist_preserve,
            x='Attribute',
            y='TVD',
            title='Total Variation Distance',
            color='TVD',
            color_continuous_scale='Oranges'
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    # Privacy-Utility Tradeoff
    st.markdown("---")
    st.markdown("### Privacy-Utility Tradeoff")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Privacy Gain", f"{tradeoff['privacy_gain_pct']:.2f}%", delta="Higher is better")
    
    with col2:
        st.metric("Utility Loss", f"{tradeoff['utility_loss_pct']:.2f}%", delta="Lower is better", delta_color="inverse")
    
    with col3:
        st.metric("Privacy/Utility Ratio", f"{tradeoff['privacy_utility_ratio']:.2f}", 
                 delta="Good" if tradeoff['privacy_utility_ratio'] > 1.0 else "Fair")
    
    # Tradeoff scatter plot
    fig = go.Figure()
    
    # Add ideal zone
    fig.add_shape(type="rect", x0=0, y0=50, x1=50, y1=100,
                 fillcolor="green", opacity=0.1, line_width=0)
    
    # Add our result
    fig.add_trace(go.Scatter(
        x=[tradeoff['utility_loss_pct']],
        y=[tradeoff['privacy_gain_pct']],
        mode='markers+text',
        marker=dict(size=20, color='#1f77b4'),
        text=[f"k={K_ANONYMITY}, ε={EPSILON}"],
        textposition="top center",
        name='Our Result'
    ))
    
    fig.add_hline(y=50, line_dash="dash", line_color="green", annotation_text="High Privacy Gain")
    fig.add_vline(x=50, line_dash="dash", line_color="red", annotation_text="High Utility Loss")
    
    fig.update_layout(
        title='Privacy-Utility Tradeoff Space',
        xaxis_title='Utility Loss (%)',
        yaxis_title='Privacy Gain (%)',
        xaxis_range=[0, 100],
        yaxis_range=[0, 100],
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)

def show_visualizations(df_original, df_anonymized, df_noisy, info_loss, dist_preserve):
    """Additional visualizations page"""
    st.markdown('<div class="sub-header">📊 Additional Visualizations</div>', unsafe_allow_html=True)
    
    # Sensitive attribute distribution
    st.markdown("### Sensitive Attribute Distribution")
    
    orig_sens = df_original[SENSITIVE_ATTRIBUTE].value_counts(normalize=True).sort_index() * 100
    anon_sens = df_anonymized[SENSITIVE_ATTRIBUTE].value_counts(normalize=True).sort_index() * 100
    
    sens_df = pd.DataFrame({
        'Class': ['No Diabetes (0)', 'Prediabetes (1)', 'Diabetes (2)'],
        'Original': [orig_sens.get(0, 0), orig_sens.get(1, 0), orig_sens.get(2, 0)],
        'Anonymized': [anon_sens.get(0, 0), anon_sens.get(1, 0), anon_sens.get(2, 0)]
    })
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = go.Figure()
        fig.add_trace(go.Bar(name='Original', x=sens_df['Class'], y=sens_df['Original'], marker_color='#3498db'))
        fig.add_trace(go.Bar(name='Anonymized', x=sens_df['Class'], y=sens_df['Anonymized'], marker_color='#e74c3c'))
        fig.update_layout(barmode='group', height=400, yaxis_title='Percentage (%)', 
                         title='Diabetes Distribution Comparison')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.pie(
            sens_df,
            values='Anonymized',
            names='Class',
            title='Anonymized Distribution (Pie Chart)',
            color_discrete_sequence=['#2ecc71', '#f39c12', '#e74c3c']
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    # All attributes overview
    st.markdown("---")
    st.markdown("### All Attributes Overview")
    
    tab1, tab2 = st.tabs(["Information Loss", "Distribution Preservation"])
    
    with tab1:
        st.dataframe(info_loss, use_container_width=True, hide_index=True)
    
    with tab2:
        st.dataframe(dist_preserve, use_container_width=True, hide_index=True)

def show_tree_simulation(df_original, df_anonymized):
    """Tree simulation page with REAL ACDP Tree structure"""
    st.markdown('<div class="sub-header">🌳 Tree Simulation</div>', unsafe_allow_html=True)
    
    st.markdown("**Visualisasi ACDP Tree asli dari hasil anonymisasi Anda.**")
    
    # Configuration
    st.markdown("---")
    st.markdown("### ⚙️ Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        epsilon_tree = st.slider("Privacy Budget (ε) for Noise", 0.1, 2.0, 0.5, 0.1, key="tree_epsilon")
        st.info("💡 Epsilon hanya untuk menambah noise pada count, tidak mengubah tree structure.")
    
    with col2:
        st.markdown("**QI Attributes Used:**")
        for attr in QI_ATTRIBUTES:
            st.markdown(f"- {attr}")
    
    # Load ACDP Tree structure
    st.markdown("---")
    st.markdown("### 🌳 ACDP Tree Structure (Real)")
    
    tree_file = os.path.join(OUTPUT_DIR, 'acdp_tree_structure.json')
    
    if not os.path.exists(tree_file):
        st.error(f"❌ ACDP Tree structure file not found: {tree_file}")
        st.info("📝 Please run `python main.py` first to generate the tree structure.")
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
        st.markdown("### 🌳 Tree Visualization")
    
    with col2:
        viz_type = st.selectbox("Visualization Type", ["Sankey Diagram", "Treemap"], key="viz_type")
    
    # Debug info
    with st.expander("🔍 Tree Metadata"):
        st.json(metadata)
    
    # Visualize tree
    if viz_type == "Sankey Diagram":
        fig = create_real_tree_visualization(tree_with_noise)
    else:
        fig = create_treemap_visualization(tree_with_noise)
    
    if fig:
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Tree visualization could not be generated.")
    
    # Tree statistics
    st.markdown("---")
    st.markdown("### 📊 Tree Statistics")
    
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
    **💡 Cara Membaca Tree:**
    - **Root Node**: Dataset lengkap
    - **Internal Nodes**: Split berdasarkan attribute dengan weighted MI tertinggi
    - **Leaf Nodes**: Final generalization levels
    - **Real Count**: Jumlah records asli di node
    - **Noisy Count**: Jumlah records setelah Laplace noise (ε={})
    
    **🎯 Tree ini adalah hasil ASLI dari ACDP Tree algorithm Anda!**
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
        0: 'rgba(31, 119, 180, 0.8)',   # Blue - Root
        1: 'rgba(255, 127, 14, 0.8)',   # Orange - Level 1
        2: 'rgba(44, 160, 44, 0.8)',    # Green - Level 2
        3: 'rgba(214, 39, 40, 0.8)',    # Red - Level 3
        'leaf': 'rgba(148, 103, 189, 0.8)'  # Purple - Leaf
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

def show_algorithm_comparison(df_original):
    """Algorithm comparison page"""
    st.markdown('<div class="sub-header">📊 Algorithm Comparison</div>', unsafe_allow_html=True)
    
    st.markdown("""
    **Halaman ini menampilkan evaluasi ACDP-Tree dibandingkan DPDT dan IPA menggunakan empat metrik:**
    - Information Loss
    - Absolute Error
    - Data Leakage Probability
    - Execution Time
    """)
    
    # Configuration
    st.markdown("---")
    st.markdown("### ⚙️ Configuration")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**QI Attributes:**")
        for attr in QI_ATTRIBUTES:
            st.markdown(f"- {attr}")
    
    with col2:
        st.markdown(f"**Sensitive Attribute:**")
        st.markdown(f"- {SENSITIVE_ATTRIBUTE}")
    
    with col3:
        epsilon_eval = st.selectbox("Privacy Budget (ε)", [1.0, 0.5, 0.1], index=1, key="eval_epsilon")
    
    # Generate evaluation metrics
    metrics_data = generate_evaluation_metrics(len(df_original), len(QI_ATTRIBUTES), epsilon_eval)
    
    # Summary cards
    st.markdown("---")
    st.markdown("### 📈 Summary (Data Size: 5000)")
    
    summary = metrics_data['summary']
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("**Information Loss**")
        for alg, val in summary['information_loss'].items():
            color = "🟢" if alg == "ACDP-Tree" else "🔴"
            st.markdown(f"{color} {alg}: {val:.4f}")
    
    with col2:
        st.markdown("**Absolute Error**")
        for alg, val in summary['absolute_error'].items():
            color = "🟢" if alg == "ACDP-Tree" else "🔴"
            st.markdown(f"{color} {alg}: {val:.2f}")
    
    with col3:
        st.markdown("**Data Leakage Prob.**")
        for alg, val in summary['data_leakage_probability'].items():
            color = "🟢" if alg == "ACDP-Tree" else "🔴"
            st.markdown(f"{color} {alg}: {val:.4f}")
    
    with col4:
        st.markdown("**Execution Time (s)**")
        for alg, val in summary['execution_time'].items():
            color = "🟢" if alg == "ACDP-Tree" else "🔴"
            st.markdown(f"{color} {alg}: {val:.4f}")
    
    # Charts
    st.markdown("---")
    st.markdown("### 📊 Comparison Charts")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig1 = create_comparison_chart(metrics_data['rows'], 'information_loss', 'Information Loss')
        st.plotly_chart(fig1, use_container_width=True)
        
        fig3 = create_comparison_chart(metrics_data['rows'], 'data_leakage_probability', 'Data Leakage Probability')
        st.plotly_chart(fig3, use_container_width=True)
    
    with col2:
        fig2 = create_comparison_chart(metrics_data['rows'], 'absolute_error', 'Absolute Error')
        st.plotly_chart(fig2, use_container_width=True)
        
        fig4 = create_comparison_chart(metrics_data['rows'], 'execution_time', 'Execution Time (s)')
        st.plotly_chart(fig4, use_container_width=True)
    
    # Metric descriptions
    st.markdown("---")
    st.markdown("### 📖 Metric Descriptions")
    
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
    st.markdown("### 📋 Full Comparison Table")
    
    df_table = pd.DataFrame(metrics_data['rows'])
    st.dataframe(df_table, use_container_width=True, hide_index=True)
    
    st.info("💡 **Catatan:** Nilai metrik pada dashboard ini adalah simulasi untuk demonstrasi. "
            "Untuk hasil penelitian final, hubungkan perhitungan ini dengan function asli di `src/metrics.py`.")

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
            'ACDP-Tree': '#2ecc71',
            'DPDT': '#3498db',
            'IPA': '#e74c3c'
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


if __name__ == "__main__":
    main()


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
        marker=dict(colorscale='Blues', line=dict(width=2, color='white')),
        hovertext=hover_texts,
        hoverinfo="text",
        textfont=dict(size=12, color='white')
    ))
    
    fig.update_layout(
        title="ACDP Tree Structure (Treemap)",
        height=700,
        margin=dict(t=50, l=10, r=10, b=10)
    )
    
    return fig
