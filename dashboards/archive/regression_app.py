"""
Food Desert Project - Interactive Regression Dashboard
Course: DATA-245 Machine Learning
Group 3: Jane Heng (Regression Lead), Kapil Reddy Sanikommu (Dashboard Lead)

Streamlit app for exploring regression results interactively
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
import pickle

# Page configuration
st.set_page_config(
    page_title="Food Desert Regression Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 42px;
        font-weight: bold;
        color: #1e88e5;
        text-align: center;
        padding: 20px;
        background: linear-gradient(90deg, #e3f2fd 0%, #90caf9 100%);
        border-radius: 10px;
        margin-bottom: 30px;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #1e88e5;
    }
    .insight-box {
        background-color: #e8f5e9;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #4caf50;
        margin: 10px 0;
    }
    .warning-box {
        background-color: #fff3e0;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #ff9800;
        margin: 10px 0;
    }
    .equation-box {
        background-color: #f3e5f5;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #9c27b0;
        margin: 10px 0;
        font-family: 'Courier New', monospace;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load all necessary data"""
    df = pd.read_csv('../data/processed/cleaned_health_data.csv')

    # Load regression results
    obesity_results = pd.read_csv('../data/output/regression_results_%_adults_with_obesity.csv')
    diabetes_results = pd.read_csv('../data/output/regression_results_%_adults_with_diabetes.csv')

    obesity_coef = pd.read_csv('../data/output/regression_coefficients_%_adults_with_obesity.csv')
    diabetes_coef = pd.read_csv('../data/output/regression_coefficients_%_adults_with_diabetes.csv')
    
    return df, obesity_results, diabetes_results, obesity_coef, diabetes_coef

@st.cache_resource
def train_models(df):
    """Train models for interactive predictions"""
    # Prepare features
    feature_names = [
        'Food_Access_Barrier_Index_normalized',
        'Socioeconomic_Vulnerability_Index_normalized',
        '% Completed High School_normalized',
        'Income Ratio_normalized',
        '% Rural_normalized',
        '% Uninsured_normalized',
        'Food Environment Index_normalized'
    ]
    
    X = df[feature_names].dropna()
    y_obesity = df.loc[X.index, '% Adults with Obesity']
    y_diabetes = df.loc[X.index, '% Adults with Diabetes']
    
    # Train models
    models = {}
    
    # Obesity models
    X_train, X_test, y_train, y_test = train_test_split(X, y_obesity, test_size=0.2, random_state=42)
    
    ols_obesity = LinearRegression()
    ols_obesity.fit(X_train, y_train)
    models['obesity_ols'] = ols_obesity
    
    ridge_obesity = Ridge(alpha=1.0)
    ridge_obesity.fit(X_train, y_train)
    models['obesity_ridge'] = ridge_obesity
    
    lasso_obesity = Lasso(alpha=0.001, max_iter=10000)
    lasso_obesity.fit(X_train, y_train)
    models['obesity_lasso'] = lasso_obesity
    
    # Diabetes models
    X_train, X_test, y_train, y_test = train_test_split(X, y_diabetes, test_size=0.2, random_state=42)
    
    ols_diabetes = LinearRegression()
    ols_diabetes.fit(X_train, y_train)
    models['diabetes_ols'] = ols_diabetes
    
    ridge_diabetes = Ridge(alpha=10.0)
    ridge_diabetes.fit(X_train, y_train)
    models['diabetes_ridge'] = ridge_diabetes
    
    lasso_diabetes = Lasso(alpha=0.001, max_iter=10000)
    lasso_diabetes.fit(X_train, y_train)
    models['diabetes_lasso'] = lasso_diabetes
    
    return models, feature_names, X, y_obesity, y_diabetes

def main():
    # Header
    st.markdown('<div class="main-header">📊 Food Desert Project: Regression Analysis Dashboard 📊</div>', 
                unsafe_allow_html=True)
    
    st.markdown("""
    **Course:** DATA-245 Machine Learning | **Group 3**  
    Explore regression models predicting obesity and diabetes from food access and socioeconomic factors
    """)
    
    # Load data
    with st.spinner("Loading data and models..."):
        df, obesity_results, diabetes_results, obesity_coef, diabetes_coef = load_data()
        models, feature_names, X, y_obesity, y_diabetes = train_models(df)
    
    st.success(f"✓ Loaded {len(df):,} counties | Trained 6 regression models")
    
    # Sidebar
    st.sidebar.title("🎛️ Dashboard Controls")
    st.sidebar.markdown("---")
    
    # Navigation
    page = st.sidebar.radio(
        "Navigate to:",
        ["📊 Overview", "🎯 Model Comparison", "📈 Coefficient Analysis", 
         "🔮 Predictions", "🗺️ Geographic Analysis", "🏥 County Explorer"]
    )
    
    st.sidebar.markdown("---")
    
    # Target selection
    st.sidebar.subheader("🎯 Target Variable")
    target = st.sidebar.selectbox(
        "Select target:",
        ["Obesity", "Diabetes", "Both"]
    )
    
    st.sidebar.markdown("---")
    
    # Model selection
    st.sidebar.subheader("🤖 Model Selection")
    model_type = st.sidebar.selectbox(
        "Select model:",
        ["OLS", "Ridge", "Lasso", "Compare All"]
    )
    
    # Main content based on page selection
    if page == "📊 Overview":
        show_overview(obesity_results, diabetes_results, target)
    
    elif page == "🎯 Model Comparison":
        show_model_comparison(obesity_results, diabetes_results, target)
    
    elif page == "📈 Coefficient Analysis":
        show_coefficient_analysis(obesity_coef, diabetes_coef, target, model_type)
    
    elif page == "🔮 Predictions":
        show_predictions(models, feature_names, df, target, model_type)
    
    elif page == "🗺️ Geographic Analysis":
        show_geographic_analysis(df, models, feature_names, target)
    
    elif page == "🏥 County Explorer":
        show_county_explorer(df, models, feature_names)

def show_overview(obesity_results, diabetes_results, target):
    """Overview page"""
    st.header("📊 Regression Analysis Overview")
    
    # Key metrics
    if target in ["Obesity", "Both"]:
        st.subheader("🔴 Obesity Prediction Results")
        
        col1, col2, col3, col4 = st.columns(4)
        
        best_obesity = obesity_results.loc[obesity_results['Test_R2'].idxmax()]
        
        with col1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Best Model", best_obesity['Model'])
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Test R²", f"{best_obesity['Test_R2']:.4f}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col3:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Test RMSE", f"{best_obesity['Test_RMSE']:.4f}%")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col4:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("CV R²", f"{best_obesity['CV_R2_Mean']:.4f}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown("""
        <div class="insight-box">
        <h4>🔍 Obesity Model Insights</h4>
        <ul>
            <li><strong>R² = 0.183:</strong> Model explains ~18% of obesity variance</li>
            <li><strong>RMSE = 3.15%:</strong> Average prediction error is 3.15 percentage points</li>
            <li><strong>Key finding:</strong> Obesity has weaker predictive power from food access alone</li>
            <li><strong>Implication:</strong> Other factors (lifestyle, culture) play significant roles</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    if target in ["Diabetes", "Both"]:
        st.markdown("---")
        st.subheader("🔵 Diabetes Prediction Results")
        
        col1, col2, col3, col4 = st.columns(4)
        
        best_diabetes = diabetes_results.loc[diabetes_results['Test_R2'].idxmax()]
        
        with col1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Best Model", best_diabetes['Model'])
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Test R²", f"{best_diabetes['Test_R2']:.4f}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col3:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Test RMSE", f"{best_diabetes['Test_RMSE']:.4f}%")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col4:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("CV R²", f"{best_diabetes['CV_R2_Mean']:.4f}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown("""
        <div class="insight-box">
        <h4>🔍 Diabetes Model Insights</h4>
        <ul>
            <li><strong>R² = 0.644:</strong> Model explains ~64% of diabetes variance (EXCELLENT!)</li>
            <li><strong>RMSE = 1.08%:</strong> Average prediction error is 1.08 percentage points</li>
            <li><strong>Key finding:</strong> Diabetes strongly correlated with poverty and food access</li>
            <li><strong>Implication:</strong> Socioeconomic interventions can effectively reduce diabetes</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Comparative insights
    st.markdown("---")
    st.subheader("🔬 Key Comparative Findings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="insight-box">
        <h4>✅ Why Diabetes Model Performs Better</h4>
        <ol>
            <li><strong>Stronger poverty correlation:</strong> r = 0.73 (vs 0.40 for obesity)</li>
            <li><strong>More direct link:</strong> Food insecurity → malnutrition → diabetes</li>
            <li><strong>Less cultural variation:</strong> Diabetes etiology more consistent across regions</li>
            <li><strong>Socioeconomic dominance:</strong> Economic factors explain most variance</li>
        </ol>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="warning-box">
        <h4>⚠️ Why Obesity Model Is Weaker</h4>
        <ol>
            <li><strong>Multi-factorial:</strong> Genetics, lifestyle, culture all contribute</li>
            <li><strong>Weaker food link:</strong> Obesity can occur with adequate food access</li>
            <li><strong>Regional variation:</strong> Cultural eating patterns vary widely</li>
            <li><strong>Hidden variables:</strong> Exercise, stress, sleep not captured</li>
        </ol>
        </div>
        """, unsafe_allow_html=True)
    
    # Summary table
    st.markdown("---")
    st.subheader("📋 Model Performance Summary")
    
    summary = pd.DataFrame({
        'Target': ['Obesity', 'Diabetes'],
        'Best Model': [
            obesity_results.loc[obesity_results['Test_R2'].idxmax(), 'Model'],
            diabetes_results.loc[diabetes_results['Test_R2'].idxmax(), 'Model']
        ],
        'Test R²': [
            obesity_results['Test_R2'].max(),
            diabetes_results['Test_R2'].max()
        ],
        'Test RMSE': [
            obesity_results.loc[obesity_results['Test_R2'].idxmax(), 'Test_RMSE'],
            diabetes_results.loc[diabetes_results['Test_R2'].idxmax(), 'Test_RMSE']
        ],
        'CV R² Mean': [
            obesity_results.loc[obesity_results['Test_R2'].idxmax(), 'CV_R2_Mean'],
            diabetes_results.loc[diabetes_results['Test_R2'].idxmax(), 'CV_R2_Mean']
        ],
        'Performance': ['Moderate (18%)', 'Excellent (64%)']
    })
    
    st.dataframe(summary.style.format({
        'Test R²': '{:.4f}',
        'Test RMSE': '{:.4f}',
        'CV R² Mean': '{:.4f}'
    }), use_container_width=True)

def show_model_comparison(obesity_results, diabetes_results, target):
    """Model comparison page"""
    st.header("🎯 Model Comparison")
    
    if target in ["Obesity", "Both"]:
        st.subheader("🔴 Obesity Models")
        
        # Bar chart comparison
        fig = go.Figure()
        
        models = obesity_results['Model'].tolist()
        
        fig.add_trace(go.Bar(
            name='Train R²',
            x=models,
            y=obesity_results['Train_R2'],
            marker_color='steelblue'
        ))
        
        fig.add_trace(go.Bar(
            name='Test R²',
            x=models,
            y=obesity_results['Test_R2'],
            marker_color='coral'
        ))
        
        fig.update_layout(
            title='Obesity Models: R² Comparison',
            xaxis_title='Model',
            yaxis_title='R² Score',
            barmode='group',
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Detailed table
        st.dataframe(obesity_results.style.format({
            'Train_R2': '{:.4f}',
            'Test_R2': '{:.4f}',
            'Train_RMSE': '{:.4f}',
            'Test_RMSE': '{:.4f}',
            'CV_R2_Mean': '{:.4f}',
            'CV_R2_Std': '{:.4f}'
        }), use_container_width=True)
    
    if target in ["Diabetes", "Both"]:
        st.markdown("---")
        st.subheader("🔵 Diabetes Models")
        
        # Bar chart comparison
        fig = go.Figure()
        
        models = diabetes_results['Model'].tolist()
        
        fig.add_trace(go.Bar(
            name='Train R²',
            x=models,
            y=diabetes_results['Train_R2'],
            marker_color='steelblue'
        ))
        
        fig.add_trace(go.Bar(
            name='Test R²',
            x=models,
            y=diabetes_results['Test_R2'],
            marker_color='coral'
        ))
        
        fig.update_layout(
            title='Diabetes Models: R² Comparison',
            xaxis_title='Model',
            yaxis_title='R² Score',
            barmode='group',
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Detailed table
        st.dataframe(diabetes_results.style.format({
            'Train_R2': '{:.4f}',
            'Test_R2': '{:.4f}',
            'Train_RMSE': '{:.4f}',
            'Test_RMSE': '{:.4f}',
            'CV_R2_Mean': '{:.4f}',
            'CV_R2_Std': '{:.4f}'
        }), use_container_width=True)
    
    # Insights
    st.markdown("""
    <div class="insight-box">
    <h4>💡 Model Selection Insights</h4>
    <ul>
        <li><strong>OLS, Ridge, Lasso perform similarly:</strong> Suggests linear relationships are appropriate</li>
        <li><strong>Low overfitting:</strong> Train-test gap < 0.08 indicates good generalization</li>
        <li><strong>Ridge regularization helps diabetes:</strong> Handles multicollinearity in socioeconomic features</li>
        <li><strong>Lasso didn't remove features:</strong> All 7 predictors contribute to predictions</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

def show_coefficient_analysis(obesity_coef, diabetes_coef, target, model_type):
    """Coefficient analysis page"""
    st.header("📈 Coefficient Analysis")
    
    st.markdown("""
    Coefficients show the **change in target** (obesity/diabetes %) for a **1 standard deviation change** in each predictor.
    - **Positive coefficient:** Increases target
    - **Negative coefficient:** Decreases target
    - **Larger absolute value:** Stronger effect
    """)
    
    if target in ["Obesity", "Both"]:
        st.subheader("🔴 Obesity Coefficients (Lasso Model)")
        
        # Remove 'Intercept' column for visualization
        obesity_viz = obesity_coef[obesity_coef['Feature'] != 'Intercept'].copy()
        obesity_viz['Feature'] = obesity_viz['Feature'].str.replace('_', ' ')
        obesity_viz = obesity_viz.sort_values('Coefficient', key=abs, ascending=True)
        
        # Horizontal bar chart
        fig = go.Figure()
        
        colors = ['green' if x > 0 else 'red' for x in obesity_viz['Coefficient']]
        
        fig.add_trace(go.Bar(
            y=obesity_viz['Feature'],
            x=obesity_viz['Coefficient'],
            orientation='h',
            marker_color=colors,
            text=obesity_viz['Coefficient'].round(3),
            textposition='outside'
        ))
        
        fig.update_layout(
            title='Obesity: Feature Importance',
            xaxis_title='Coefficient (% change per 1 SD)',
            yaxis_title='',
            height=500
        )
        
        fig.add_vline(x=0, line_dash="dash", line_color="black")
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Regression equation
        intercept = obesity_coef[obesity_coef['Feature'] == 'Intercept']['Intercept'].values[0]
        
        st.markdown(f"""
        <div class="equation-box">
        <h4>📐 Obesity Prediction Equation:</h4>
        <p><strong>Obesity % = {intercept:.2f}</strong></p>
        <p style="margin-left: 20px;">+ {obesity_viz.iloc[-1]['Coefficient']:.3f} × Food Access Barrier Index</p>
        <p style="margin-left: 20px;">+ {obesity_viz.iloc[-2]['Coefficient']:.3f} × Food Environment Index</p>
        <p style="margin-left: 20px;">{obesity_viz.iloc[-3]['Coefficient']:.3f} × Completed High School</p>
        <p style="margin-left: 20px;">+ other features...</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="insight-box">
        <h4>🔍 Obesity Coefficient Interpretation</h4>
        <ul>
            <li><strong>Food Access Barrier (+2.60):</strong> Counties with worse food access have 2.6% higher obesity</li>
            <li><strong>Food Environment (+1.45):</strong> Surprisingly positive - may indicate confounding</li>
            <li><strong>High School Education (-1.23):</strong> Higher education reduces obesity by 1.23%</li>
            <li><strong>Poverty (-0.52):</strong> Negative sign unexpected - complex relationship</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    if target in ["Diabetes", "Both"]:
        st.markdown("---")
        st.subheader("🔵 Diabetes Coefficients (OLS Model)")
        
        diabetes_viz = diabetes_coef[diabetes_coef['Feature'] != 'Intercept'].copy()
        diabetes_viz['Feature'] = diabetes_viz['Feature'].str.replace('_', ' ')
        diabetes_viz = diabetes_viz.sort_values('Coefficient', key=abs, ascending=True)
        
        # Horizontal bar chart
        fig = go.Figure()
        
        colors = ['green' if x > 0 else 'red' for x in diabetes_viz['Coefficient']]
        
        fig.add_trace(go.Bar(
            y=diabetes_viz['Feature'],
            x=diabetes_viz['Coefficient'],
            orientation='h',
            marker_color=colors,
            text=diabetes_viz['Coefficient'].round(3),
            textposition='outside'
        ))
        
        fig.update_layout(
            title='Diabetes: Feature Importance',
            xaxis_title='Coefficient (% change per 1 SD)',
            yaxis_title='',
            height=500
        )
        
        fig.add_vline(x=0, line_dash="dash", line_color="black")
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Regression equation
        intercept = diabetes_coef[diabetes_coef['Feature'] == 'Intercept']['Intercept'].values[0]
        
        st.markdown(f"""
        <div class="equation-box">
        <h4>📐 Diabetes Prediction Equation:</h4>
        <p><strong>Diabetes % = {intercept:.2f}</strong></p>
        <p style="margin-left: 20px;">+ {diabetes_viz.iloc[-1]['Coefficient']:.3f} × Socioeconomic Vulnerability Index</p>
        <p style="margin-left: 20px;">{diabetes_viz.iloc[-2]['Coefficient']:.3f} × Completed High School</p>
        <p style="margin-left: 20px;">{diabetes_viz.iloc[-3]['Coefficient']:.3f} × Rural %</p>
        <p style="margin-left: 20px;">+ other features...</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="insight-box">
        <h4>🔍 Diabetes Coefficient Interpretation</h4>
        <ul>
            <li><strong>Socioeconomic Vulnerability (+0.92):</strong> DOMINANT factor - poverty increases diabetes by 0.92%</li>
            <li><strong>High School Education (-0.57):</strong> Education reduces diabetes significantly</li>
            <li><strong>Rural (-0.23):</strong> Rural areas have slightly lower diabetes (more physical labor?)</li>
            <li><strong>Food Access Barrier (+0.19):</strong> Weaker than expected - poverty dominates</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

def show_predictions(models, feature_names, df, target, model_type):
    """Interactive predictions page"""
    st.header("🔮 Interactive Predictions")
    
    st.markdown("Adjust the sliders to see how different factors affect obesity and diabetes predictions.")
    
    # Create sliders for each feature
    st.subheader("🎚️ Adjust County Characteristics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        food_barrier = st.slider(
            "Food Access Barrier Index",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.05,
            help="Higher = worse food access"
        )
        
        poverty = st.slider(
            "Socioeconomic Vulnerability",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.05,
            help="Higher = more vulnerable"
        )
        
        education = st.slider(
            "High School Completion Rate (%)",
            min_value=60.0,
            max_value=100.0,
            value=85.0,
            step=1.0
        )
        
        income_ratio = st.slider(
            "Income Inequality Ratio",
            min_value=2.0,
            max_value=10.0,
            value=5.0,
            step=0.5,
            help="80th/20th percentile income"
        )
    
    with col2:
        rural = st.slider(
            "Rural Population (%)",
            min_value=0.0,
            max_value=100.0,
            value=50.0,
            step=5.0
        )
        
        uninsured = st.slider(
            "Uninsured Rate (%)",
            min_value=0.0,
            max_value=30.0,
            value=10.0,
            step=1.0
        )
        
        food_env = st.slider(
            "Food Environment Index",
            min_value=0.0,
            max_value=10.0,
            value=7.0,
            step=0.5,
            help="0=worst, 10=best"
        )
    
    # Normalize inputs (approximate - using dataset stats)
    def normalize_value(value, mean, std):
        return (value - mean) / std
    
    # Get means and stds from data
    feature_stats = {
        'Food_Access_Barrier_Index_normalized': (food_barrier - df['Food_Access_Barrier_Index'].mean()) / df['Food_Access_Barrier_Index'].std(),
        'Socioeconomic_Vulnerability_Index_normalized': (poverty - df['Socioeconomic_Vulnerability_Index'].mean()) / df['Socioeconomic_Vulnerability_Index'].std(),
        '% Completed High School_normalized': (education - df['% Completed High School'].mean()) / df['% Completed High School'].std(),
        'Income Ratio_normalized': (income_ratio - df['Income Ratio'].mean()) / df['Income Ratio'].std(),
        '% Rural_normalized': (rural - df['% Rural'].mean()) / df['% Rural'].std(),
        '% Uninsured_normalized': (uninsured - df['% Uninsured'].mean()) / df['% Uninsured'].std(),
        'Food Environment Index_normalized': (food_env - df['Food Environment Index'].mean()) / df['Food Environment Index'].std()
    }
    
    # Create input array
    X_pred = np.array([[feature_stats[f] for f in feature_names]])
    
    # Make predictions
    st.markdown("---")
    st.subheader("📊 Predictions")
    
    col1, col2, col3 = st.columns(3)
    
    if target in ["Obesity", "Both"]:
        with col1:
            if model_type == "OLS":
                pred = models['obesity_ols'].predict(X_pred)[0]
            elif model_type == "Ridge":
                pred = models['obesity_ridge'].predict(X_pred)[0]
            elif model_type == "Lasso":
                pred = models['obesity_lasso'].predict(X_pred)[0]
            else:
                pred = (models['obesity_ols'].predict(X_pred)[0] + 
                       models['obesity_ridge'].predict(X_pred)[0] + 
                       models['obesity_lasso'].predict(X_pred)[0]) / 3
            
            st.metric("🔴 Predicted Obesity Rate", f"{pred:.2f}%")
            
            if pred < 30:
                st.success("Low obesity risk")
            elif pred < 35:
                st.info("Moderate obesity risk")
            elif pred < 40:
                st.warning("High obesity risk")
            else:
                st.error("Very high obesity risk")
    
    if target in ["Diabetes", "Both"]:
        with col2:
            if model_type == "OLS":
                pred = models['diabetes_ols'].predict(X_pred)[0]
            elif model_type == "Ridge":
                pred = models['diabetes_ridge'].predict(X_pred)[0]
            elif model_type == "Lasso":
                pred = models['diabetes_lasso'].predict(X_pred)[0]
            else:
                pred = (models['diabetes_ols'].predict(X_pred)[0] + 
                       models['diabetes_ridge'].predict(X_pred)[0] + 
                       models['diabetes_lasso'].predict(X_pred)[0]) / 3
            
            st.metric("🔵 Predicted Diabetes Rate", f"{pred:.2f}%")
            
            if pred < 9:
                st.success("Low diabetes risk")
            elif pred < 11:
                st.info("Moderate diabetes risk")
            elif pred < 13:
                st.warning("High diabetes risk")
            else:
                st.error("Very high diabetes risk")
    
    with col3:
        health_risk = (pred / 100) if target == "Diabetes" else ((pred / 100) * 0.6 + 0.4 * 0.11)
        st.metric("❤️ Health Risk Score", f"{health_risk:.3f}")
        
        if health_risk < 0.3:
            st.success("Low overall risk")
        elif health_risk < 0.5:
            st.warning("Moderate overall risk")
        else:
            st.error("High overall risk")

def show_geographic_analysis(df, models, feature_names, target):
    """Geographic analysis page"""
    st.header("🗺️ Geographic Analysis")
    
    st.markdown("Explore predictions across different states and regions.")
    
    # State selection
    states = sorted(df['State'].unique())
    selected_states = st.multiselect(
        "Select states to compare:",
        states,
        default=states[:5]
    )
    
    if selected_states:
        state_df = df[df['State'].isin(selected_states)]
        
        # Calculate predictions
        X_state = state_df[feature_names].dropna()
        
        if target == "Obesity":
            predictions = models['obesity_ridge'].predict(X_state)
            actual = state_df.loc[X_state.index, '% Adults with Obesity']
            metric = 'Obesity Rate (%)'
        else:
            predictions = models['diabetes_ridge'].predict(X_state)
            actual = state_df.loc[X_state.index, '% Adults with Diabetes']
            metric = 'Diabetes Rate (%)'
        
        state_df_pred = state_df.loc[X_state.index].copy()
        state_df_pred['Predicted'] = predictions
        state_df_pred['Actual'] = actual
        state_df_pred['Error'] = state_df_pred['Actual'] - state_df_pred['Predicted']
        
        # Scatter plot: Actual vs Predicted by State
        fig = px.scatter(
            state_df_pred,
            x='Predicted',
            y='Actual',
            color='State',
            hover_data=['County'],
            title=f'Actual vs Predicted {metric} by State',
            labels={'Predicted': f'Predicted {metric}', 'Actual': f'Actual {metric}'}
        )
        
        # Add perfect prediction line
        min_val = min(state_df_pred['Predicted'].min(), state_df_pred['Actual'].min())
        max_val = max(state_df_pred['Predicted'].max(), state_df_pred['Actual'].max())
        fig.add_trace(go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode='lines',
            name='Perfect Prediction',
            line=dict(dash='dash', color='red')
        ))
        
        st.plotly_chart(fig, use_container_width=True)
        
        # State averages
        state_avg = state_df_pred.groupby('State').agg({
            'Actual': 'mean',
            'Predicted': 'mean',
            'Error': 'mean'
        }).reset_index()
        
        st.subheader("📊 State Averages")
        st.dataframe(state_avg.style.format({
            'Actual': '{:.2f}%',
            'Predicted': '{:.2f}%',
            'Error': '{:.2f}%'
        }), use_container_width=True)

def show_county_explorer(df, models, feature_names):
    """County explorer page"""
    st.header("🏥 County Explorer")
    
    st.markdown("Search for counties and see their predictions vs actual values.")
    
    # Search
    search_term = st.text_input("🔍 Search for a county (name or state):", "")
    
    if search_term:
        mask = (df['County'].str.contains(search_term, case=False, na=False) | 
                df['State'].str.contains(search_term, case=False, na=False))
        search_results = df[mask]
        
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
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Actual Obesity", f"{county_data['% Adults with Obesity']:.2f}%")
                    # Get prediction
                    X_county = county_data[feature_names].values.reshape(1, -1)
                    pred_obesity = models['obesity_ridge'].predict(X_county)[0]
                    st.metric("Predicted Obesity", f"{pred_obesity:.2f}%")
                
                with col2:
                    st.metric("Actual Diabetes", f"{county_data['% Adults with Diabetes']:.2f}%")
                    pred_diabetes = models['diabetes_ridge'].predict(X_county)[0]
                    st.metric("Predicted Diabetes", f"{pred_diabetes:.2f}%")
                
                with col3:
                    st.metric("Food Environment", f"{county_data['Food Environment Index']:.2f}/10")
                    st.metric("Poverty Rate", f"{county_data['% Children in Poverty']:.2f}%")
                
                # Feature comparison
                st.subheader("📊 Feature Comparison")
                
                feature_display = {
                    'Food_Access_Barrier_Index': 'Food Access Barrier',
                    'Socioeconomic_Vulnerability_Index': 'Socioeconomic Vulnerability',
                    '% Completed High School': 'HS Completion',
                    'Income Ratio': 'Income Inequality',
                    '% Rural': 'Rural Population',
                    '% Uninsured': 'Uninsured Rate',
                    'Food Environment Index': 'Food Environment'
                }
                
                feature_values = []
                for feat, display_name in feature_display.items():
                    if feat in county_data:
                        feature_values.append({
                            'Feature': display_name,
                            'Value': county_data[feat],
                            'Dataset Average': df[feat].mean()
                        })
                
                feature_df = pd.DataFrame(feature_values)
                feature_df['Difference'] = feature_df['Value'] - feature_df['Dataset Average']
                
                fig = go.Figure()
                
                fig.add_trace(go.Bar(
                    name='County',
                    x=feature_df['Feature'],
                    y=feature_df['Value'],
                    marker_color='steelblue'
                ))
                
                fig.add_trace(go.Bar(
                    name='Average',
                    x=feature_df['Feature'],
                    y=feature_df['Dataset Average'],
                    marker_color='coral'
                ))
                
                fig.update_layout(
                    title=f'{county_name} vs National Average',
                    barmode='group',
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No counties found matching your search.")
    else:
        st.info("👆 Enter a county name or state to begin exploring")

# Footer
def show_footer():
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 20px;'>
        <p><strong> Project: Regression Analysis Dashboard</strong></p>
        <p>DATA-245 Machine Learning | Group 3</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
    show_footer()