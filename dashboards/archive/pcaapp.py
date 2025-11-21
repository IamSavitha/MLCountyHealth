"""
"County Level : Socio economic vulnerability and metabolic health outcomes"  Project - Interactive PCA Dashboard
Course: DATA-245 Machine Learning
Group 3 : Savitha , Jane , Rishi , Kapil

Streamlit app for exploring PCA results interactively
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Page configuration
st.set_page_config(
    page_title="Food Desert PCA Dashboard",
    page_icon="🍎",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 42px;
        font-weight: bold;
        color: #2E86AB;
        text-align: center;
        padding: 20px;
        background: linear-gradient(90deg, #e3f2fd 0%, #bbdefb 100%);
        border-radius: 10px;
        margin-bottom: 30px;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #2E86AB;
    }
    .insight-box {
        background-color: #fff3cd;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #ffc107;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load all necessary data"""
    df = pd.read_csv('../data/processed/cleaned_health_data.csv')
    pca_results = pd.read_csv('../data/output/pca_results.csv')
    loadings = pd.read_csv('../data/output/pca_loadings.csv', index_col=0)
    return df, pca_results, loadings

@st.cache_resource
def fit_pca_model(df):
    """Fit PCA model"""
    features = [
        '% Adults with Obesity',
        '% Adults with Diabetes',
        'Food Environment Index',
        '% Children in Poverty',
        '% Completed High School',
        'Income Ratio',
        '% Uninsured',
        '% Rural',
        'Food_Access_Barrier_Index',
        'Socioeconomic_Vulnerability_Index',
        'Health_Risk_Score'
    ]
    
    X = df[features].dropna()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    pca = PCA(n_components=11)
    principal_components = pca.fit_transform(X_scaled)
    
    return pca, principal_components, X.index, features

def main():
    # Header
    st.markdown('<div class="main-header">🍎 Food Desert Project: PCA Analysis Dashboard 🍎</div>', 
                unsafe_allow_html=True)
    
    st.markdown("""
    **Course:** DATA-245 Machine Learning | **Group 3**  
    Explore principal components of nutritional vulnerability and metabolic health outcomes
    """)
    
    # Load data
    with st.spinner("Loading data..."):
        df, pca_results, loadings = load_data()
        pca, principal_components, valid_indices, features = fit_pca_model(df)
    
    st.success(f"✓ Loaded {len(df):,} counties | {len(features)} features analyzed")
    
    # Sidebar
    st.sidebar.title("🎛️ Dashboard Controls")
    st.sidebar.markdown("---")
    
    # Navigation
    page = st.sidebar.radio(
        "Navigate to:",
        ["📊 Overview", "🔍 Variance Analysis", "🎯 Biplot Explorer", 
         "🌐 3D Visualization", "📈 Feature Loadings", "🏥 County Explorer"]
    )
    
    st.sidebar.markdown("---")
    
    # Filters
    st.sidebar.subheader("🔎 Filters")
    
    # State filter
    states = ['All States'] + sorted(df['State'].unique().tolist())
    selected_state = st.sidebar.selectbox("Select State:", states)
    
    # Area type filter
    if 'Area_Type' in df.columns:
        area_types = ['All Types'] + sorted(df['Area_Type'].dropna().unique().tolist())
        selected_area = st.sidebar.selectbox("Select Area Type:", area_types)
    else:
        selected_area = 'All Types'
    
    # Apply filters
    filtered_df = df.copy()
    if selected_state != 'All States':
        filtered_df = filtered_df[filtered_df['State'] == selected_state]
    if selected_area != 'All Types' and 'Area_Type' in df.columns:
        filtered_df = filtered_df[filtered_df['Area_Type'] == selected_area]
    
    st.sidebar.markdown(f"**Filtered Counties:** {len(filtered_df):,}")
    
    # Main content based on page selection
    if page == "📊 Overview":
        show_overview(pca, df, principal_components, features)
    
    elif page == "🔍 Variance Analysis":
        show_variance_analysis(pca, features)
    
    elif page == "🎯 Biplot Explorer":
        show_biplot(pca, principal_components, features, df, valid_indices, filtered_df)
    
    elif page == "🌐 3D Visualization":
        show_3d_plot(pca, principal_components, df, valid_indices, filtered_df)
    
    elif page == "📈 Feature Loadings":
        show_loadings(loadings, pca)
    
    elif page == "🏥 County Explorer":
        show_county_explorer(pca_results, filtered_df)

def show_overview(pca, df, principal_components, features):
    """Overview page"""
    st.header("📊 PCA Analysis Overview")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Total Variance (PC1)", f"{pca.explained_variance_ratio_[0]*100:.1f}%")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("PC1 + PC2", f"{sum(pca.explained_variance_ratio_[:2])*100:.1f}%")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("PC1 + PC2 + PC3", f"{sum(pca.explained_variance_ratio_[:3])*100:.1f}%")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col4:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Total Features", len(features))
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Key insights
    st.subheader("🔑 Key Insights")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="insight-box">
        <h4>📌 PC1: Socioeconomic & Health Dimension (55.0%)</h4>
        <ul>
            <li><strong>Strongest loadings:</strong> Socioeconomic Vulnerability, Poverty, Diabetes</li>
            <li><strong>Interpretation:</strong> Communities with high poverty show higher diabetes rates and overall socioeconomic stress</li>
            <li><strong>Policy implication:</strong> Address poverty as a root cause of metabolic health issues</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="insight-box">
        <h4>📌 PC2: Obesity-Specific Dimension (13.5%)</h4>
        <ul>
            <li><strong>Strongest loadings:</strong> Adult Obesity, Health Risk Score</li>
            <li><strong>Interpretation:</strong> Obesity patterns independent of poverty/diabetes</li>
            <li><strong>Policy implication:</strong> Obesity requires targeted interventions beyond economic support</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="insight-box">
    <h4>📌 PC3: Urban-Rural Divide (9.5%)</h4>
    <ul>
        <li><strong>Strongest loading:</strong> % Rural (0.842)</li>
        <li><strong>Interpretation:</strong> Geographic/infrastructure differences in food access</li>
        <li><strong>Policy implication:</strong> Different strategies needed for urban vs rural food deserts</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Summary statistics
    st.subheader("📈 Summary Statistics")
    
    summary_data = {
        'Metric': [
            'Total Counties Analyzed',
            'Total Features',
            'Variance Explained (PC1-PC3)',
            'Cumulative Variance (PC1-PC5)',
            'Average Health Risk Score',
            'Average Food Barrier Index'
        ],
        'Value': [
            f"{len(principal_components):,}",
            len(features),
            f"{sum(pca.explained_variance_ratio_[:3])*100:.1f}%",
            f"{sum(pca.explained_variance_ratio_[:5])*100:.1f}%",
            f"{df['Health_Risk_Score'].mean():.3f}",
            f"{df['Food_Access_Barrier_Index'].mean():.3f}"
        ]
    }
    
    st.table(pd.DataFrame(summary_data))

def show_variance_analysis(pca, features):
    """Variance analysis page"""
    st.header("🔍 Variance Analysis")
    
    # Scree plot
    st.subheader("Scree Plot: Variance Explained")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Individual variance
        fig, ax = plt.subplots(figsize=(10, 6))
        variance = pca.explained_variance_ratio_ * 100
        ax.bar(range(1, len(variance) + 1), variance, color='steelblue', alpha=0.8, edgecolor='black')
        ax.set_xlabel('Principal Component', fontsize=12, fontweight='bold')
        ax.set_ylabel('Variance Explained (%)', fontsize=12, fontweight='bold')
        ax.set_title('Individual Variance per Component', fontsize=14, fontweight='bold')
        ax.grid(alpha=0.3)
        st.pyplot(fig)
    
    with col2:
        # Cumulative variance
        fig, ax = plt.subplots(figsize=(10, 6))
        cum_variance = np.cumsum(variance)
        ax.plot(range(1, len(cum_variance) + 1), cum_variance, 
                marker='o', markersize=8, linewidth=2, color='darkred')
        ax.axhline(y=80, color='green', linestyle='--', linewidth=2, label='80% threshold')
        ax.fill_between(range(1, len(cum_variance) + 1), cum_variance, alpha=0.3, color='darkred')
        ax.set_xlabel('Number of Components', fontsize=12, fontweight='bold')
        ax.set_ylabel('Cumulative Variance (%)', fontsize=12, fontweight='bold')
        ax.set_title('Cumulative Variance Explained', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)
        st.pyplot(fig)
    
    # Variance table
    st.subheader("📊 Detailed Variance Table")
    
    variance_df = pd.DataFrame({
        'Component': [f'PC{i+1}' for i in range(len(pca.explained_variance_ratio_))],
        'Variance (%)': pca.explained_variance_ratio_ * 100,
        'Cumulative (%)': np.cumsum(pca.explained_variance_ratio_) * 100
    })
    
    st.dataframe(variance_df.style.format({'Variance (%)': '{:.2f}', 'Cumulative (%)': '{:.2f}'}))
    
    # Insights
    st.markdown("""
    <div class="insight-box">
    <h4>💡 Interpretation Guide</h4>
    <ul>
        <li><strong>PC1 dominates</strong> with 55% of variance → Strong underlying socioeconomic-health factor</li>
        <li><strong>First 3 PCs capture 78%</strong> → Dimensionality can be reduced from 11 to 3 features</li>
        <li><strong>Elbow at PC3</strong> → Subsequent components add minimal information</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

def show_biplot(pca, principal_components, features, df, valid_indices, filtered_df):
    """Interactive biplot page"""
    st.header("🎯 Interactive Biplot Explorer")
    
    # Component selection
    col1, col2 = st.columns(2)
    with col1:
        pc_x = st.selectbox("X-axis:", [f'PC{i+1}' for i in range(min(5, pca.n_components_))], index=0)
    with col2:
        pc_y = st.selectbox("Y-axis:", [f'PC{i+1}' for i in range(min(5, pca.n_components_))], index=1)
    
    pc_x_idx = int(pc_x[2:]) - 1
    pc_y_idx = int(pc_y[2:]) - 1
    
    # Color by
    color_options = ['Health_Risk_Score', '% Adults with Obesity', '% Adults with Diabetes', 
                    'Food_Access_Barrier_Index', 'Socioeconomic_Vulnerability_Index', 'Area_Type']
    color_by = st.selectbox("Color points by:", 
                           [opt for opt in color_options if opt in df.columns])
    
    # Create interactive biplot with plotly
    df_plot = df.loc[valid_indices].copy()
    df_plot['PC_X'] = principal_components[:, pc_x_idx]
    df_plot['PC_Y'] = principal_components[:, pc_y_idx]
    
    # Filter for display
    if len(filtered_df) < len(df):
        mask = df_plot['FIPS'].isin(filtered_df['FIPS'])
        df_plot_display = df_plot[mask]
    else:
        df_plot_display = df_plot
    
    fig = px.scatter(
        df_plot_display,
        x='PC_X',
        y='PC_Y',
        color=color_by,
        hover_data=['State', 'County', '% Adults with Obesity', '% Adults with Diabetes'],
        title=f'PCA Biplot: {pc_x} vs {pc_y}',
        labels={'PC_X': f'{pc_x} ({pca.explained_variance_ratio_[pc_x_idx]*100:.1f}%)',
                'PC_Y': f'{pc_y} ({pca.explained_variance_ratio_[pc_y_idx]*100:.1f}%)'},
        color_continuous_scale='RdYlGn_r' if color_by != 'Area_Type' else None
    )
    
    # Add feature loadings as arrows
    loadings = pca.components_
    scale = st.slider("Loading vector scale:", 1.0, 10.0, 4.0, 0.5)
    
    if st.checkbox("Show feature loadings", value=True):
        for i, feature in enumerate(features):
            fig.add_annotation(
                ax=0, ay=0,
                axref='x', ayref='y',
                x=loadings[pc_x_idx, i] * scale,
                y=loadings[pc_y_idx, i] * scale,
                xref='x', yref='y',
                showarrow=True,
                arrowhead=2,
                arrowsize=1,
                arrowwidth=2,
                arrowcolor='darkblue',
                opacity=0.7
            )
            fig.add_annotation(
                x=loadings[pc_x_idx, i] * scale * 1.1,
                y=loadings[pc_y_idx, i] * scale * 1.1,
                text=feature,
                showarrow=False,
                font=dict(size=9, color='darkblue'),
                bgcolor='rgba(255, 255, 200, 0.8)',
                bordercolor='darkblue',
                borderwidth=1
            )
    
    fig.update_layout(height=700, showlegend=True)
    st.plotly_chart(fig, use_container_width=True)
    
    # Interpretation
    st.markdown("""
    <div class="insight-box">
    <h4>📖 How to Read This Biplot</h4>
    <ul>
        <li><strong>Points:</strong> Each point is a county, colored by selected metric</li>
        <li><strong>Arrows:</strong> Show how original features contribute to each component</li>
        <li><strong>Arrow direction:</strong> Points in that direction score high on that feature</li>
        <li><strong>Arrow length:</strong> Longer = stronger contribution to the component</li>
        <li><strong>Clusters:</strong> Counties close together have similar characteristics</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

def show_3d_plot(pca, principal_components, df, valid_indices, filtered_df):
    """3D visualization page"""
    st.header("🌐 3D Principal Component Visualization")
    
    # Color by
    color_options = ['Health_Risk_Score', '% Adults with Obesity', '% Adults with Diabetes', 
                    'Food_Access_Barrier_Index', 'Area_Type']
    color_by = st.selectbox("Color points by:", 
                           [opt for opt in color_options if opt in df.columns],
                           key='3d_color')
    
    # Create 3D plot
    df_plot = df.loc[valid_indices].copy()
    df_plot['PC1'] = principal_components[:, 0]
    df_plot['PC2'] = principal_components[:, 1]
    df_plot['PC3'] = principal_components[:, 2]
    
    # Filter
    if len(filtered_df) < len(df):
        mask = df_plot['FIPS'].isin(filtered_df['FIPS'])
        df_plot = df_plot[mask]
    
    fig = px.scatter_3d(
        df_plot,
        x='PC1',
        y='PC2',
        z='PC3',
        color=color_by,
        hover_data=['State', 'County', '% Adults with Obesity', '% Adults with Diabetes'],
        title='3D PCA: First Three Principal Components',
        labels={
            'PC1': f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)',
            'PC2': f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)',
            'PC3': f'PC3 ({pca.explained_variance_ratio_[2]*100:.1f}%)'
        },
        color_continuous_scale='RdYlGn_r' if color_by != 'Area_Type' else None
    )
    
    fig.update_layout(height=700)
    st.plotly_chart(fig, use_container_width=True)
    
    st.info(f"💡 **Tip:** Click and drag to rotate the 3D plot. These 3 components explain {sum(pca.explained_variance_ratio_[:3])*100:.1f}% of total variance!")

def show_loadings(loadings, pca):
    """Feature loadings page"""
    st.header("📈 Feature Loadings Analysis")
    
    st.markdown("""
    Feature loadings show how much each original feature contributes to each principal component.
    Higher absolute values indicate stronger contributions.
    """)
    
    # Component selection
    selected_pc = st.selectbox("Select Component:", 
                              [f'PC{i+1}' for i in range(min(5, pca.n_components_))])
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Loadings bar chart
        loadings_series = loadings[selected_pc].sort_values(key=abs, ascending=False)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        colors = ['green' if x > 0 else 'red' for x in loadings_series]
        ax.barh(range(len(loadings_series)), loadings_series, color=colors, alpha=0.7, edgecolor='black')
        ax.set_yticks(range(len(loadings_series)))
        ax.set_yticklabels(loadings_series.index)
        ax.set_xlabel('Loading Value', fontsize=12, fontweight='bold')
        ax.set_title(f'{selected_pc} Feature Loadings', fontsize=14, fontweight='bold')
        ax.axvline(x=0, color='black', linewidth=1)
        ax.grid(alpha=0.3, axis='x')
        st.pyplot(fig)
    
    with col2:
        st.subheader("Top Contributors")
        top_loadings = loadings[selected_pc].abs().sort_values(ascending=False).head(5)
        for i, (feature, value) in enumerate(top_loadings.items(), 1):
            actual_value = loadings.loc[feature, selected_pc]
            st.metric(f"#{i} {feature}", f"{actual_value:.3f}")
    
    # Full loadings table
    st.subheader("📋 Complete Loadings Table")
    st.dataframe(loadings.style.format('{:.4f}').background_gradient(cmap='RdYlGn', axis=0))
    
    # Download button
    csv = loadings.to_csv()
    st.download_button(
        label="📥 Download Loadings CSV",
        data=csv,
        file_name="pca_loadings.csv",
        mime="text/csv"
    )

def show_county_explorer(pca_results, filtered_df):
    """County explorer page"""
    st.header("🏥 County Explorer")
    
    st.markdown("Search and explore individual counties with their PCA scores")
    
    # Search
    search_term = st.text_input("🔍 Search for a county (name or state):", "")
    
    if search_term:
        mask = (filtered_df['County'].str.contains(search_term, case=False, na=False) | 
                filtered_df['State'].str.contains(search_term, case=False, na=False))
        search_results = filtered_df[mask]
        
        if len(search_results) > 0:
            st.success(f"Found {len(search_results)} matching counties")
            
            # Select county
            county_options = [f"{row['County']}, {row['State']}" for _, row in search_results.iterrows()]
            selected = st.selectbox("Select a county:", county_options)
            
            if selected:
                county_name, state_name = selected.rsplit(', ', 1)
                county_data = search_results[
                    (search_results['County'] == county_name) & 
                    (search_results['State'] == state_name)
                ].iloc[0]
                
                # Display county info
                st.subheader(f"📍 {county_name}, {state_name}")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Obesity Rate", f"{county_data['% Adults with Obesity']:.1f}%")
                with col2:
                    st.metric("Diabetes Rate", f"{county_data['% Adults with Diabetes']:.1f}%")
                with col3:
                    st.metric("Food Environment", f"{county_data['Food Environment Index']:.1f}/10")
                with col4:
                    st.metric("Health Risk Score", f"{county_data['Health_Risk_Score']:.3f}")
                
                # PCA scores
                if 'PC1' in county_data:
                    st.subheader("Principal Component Scores")
                    pc_scores = {
                        'PC1': county_data.get('PC1', 0),
                        'PC2': county_data.get('PC2', 0),
                        'PC3': county_data.get('PC3', 0)
                    }
                    
                    fig, ax = plt.subplots(figsize=(10, 4))
                    ax.bar(pc_scores.keys(), pc_scores.values(), color='steelblue', alpha=0.7, edgecolor='black')
                    ax.axhline(y=0, color='black', linewidth=1)
                    ax.set_ylabel('Score', fontweight='bold')
                    ax.set_title(f'PCA Scores for {county_name}', fontweight='bold')
                    ax.grid(alpha=0.3, axis='y')
                    st.pyplot(fig)
        else:
            st.warning("No counties found matching your search.")
    else:
        st.info("👆 Enter a county name or state to begin exploring")
    
    # Summary statistics
    st.markdown("---")
    st.subheader("📊 Filtered Dataset Summary")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Health Metrics:**")
        st.write(f"- Average Obesity: {filtered_df['% Adults with Obesity'].mean():.1f}%")
        st.write(f"- Average Diabetes: {filtered_df['% Adults with Diabetes'].mean():.1f}%")
        st.write(f"- Average Health Risk: {filtered_df['Health_Risk_Score'].mean():.3f}")
    
    with col2:
        st.write("**Food & Socioeconomic:**")
        st.write(f"- Average Food Index: {filtered_df['Food Environment Index'].mean():.1f}/10")
        st.write(f"- Average Poverty: {filtered_df['% Children in Poverty'].mean():.1f}%")
        st.write(f"- Average Food Barrier: {filtered_df['Food_Access_Barrier_Index'].mean():.3f}")

# Footer
def show_footer():
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 20px;'>
        <p><strong>Project: PCA Analysis Dashboard</strong></p>
        <p>DATA-245 Machine Learning | Group 3</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
    show_footer()