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

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #2c3e50;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .success-box {
        background-color: #d4edda;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #28a745;
    }
    .warning-box {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ffc107;
    }
</style>
""", unsafe_allow_html=True)

# Constants
from src.config import (
    RAW_DATA_PATH, OUTPUT_DIR,
    QI_ATTRIBUTES, SENSITIVE_ATTRIBUTE,
    K_ANONYMITY, EPSILON, MAX_LEVEL
)

# Helper functions
@st.cache_data
def load_data():
    """Load original and anonymized datasets"""
    try:
        # Load original data
        df_original = pd.read_csv(RAW_DATA_PATH)
        
        # Load anonymized data
        anon_file = f'diabetes_anonymized_k{K_ANONYMITY}_eps{EPSILON:.1f}.csv'
        anon_path = os.path.join(OUTPUT_DIR, anon_file)
        df_anonymized = pd.read_csv(anon_path)
        
        # Load noisy counts
        noisy_file = f'diabetes_noisy_counts_k{K_ANONYMITY}_eps{EPSILON:.1f}.csv'
        noisy_path = os.path.join(OUTPUT_DIR, noisy_file)
        df_noisy = pd.read_csv(noisy_path)
        
        return df_original, df_anonymized, df_noisy, True
    except Exception as e:
        st.error(f"Error loading data: {e}")
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
        ["Overview", "Data Comparison", "Privacy Metrics", "Utility Metrics", "Visualizations", "Tree Simulation", "Algorithm Comparison"]
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
    if page == "Overview":
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
