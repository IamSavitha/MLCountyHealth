import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, LogisticRegression
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (mean_squared_error, r2_score, accuracy_score, 
                             classification_report, confusion_matrix, roc_curve, 
                             auc, silhouette_score)
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Page config
st.set_page_config(page_title="ML Algorithms and county level health", layout="wide")

# Title
st.title("ML Algorithms and county level health")
st.markdown("### Learn by doing: Experiment with different algorithms and hyperparameters")

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv('../data/processed/cleaned_health_data.csv')
    return df

df = load_data()

# Sidebar for algorithm selection
st.sidebar.header("Algorithm Selection")
algorithm_category = st.sidebar.selectbox(
    "Choose Algorithm Category",
    ["Linear Regression", "Logistic Regression", "SVM", "Decision Tree", 
     "Random Forest", "KNN", "Naive Bayes", "K-Means Clustering", "PCA"]
)

# Display dataset info
with st.expander("Dataset Overview", expanded=False):
    st.write(f"**Shape:** {df.shape[0]} rows × {df.shape[1]} columns")
    st.write("**First few rows:**")
    st.dataframe(df.head())
    st.write("**Numeric columns:**")
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    st.write(numeric_cols)

# Feature selection
st.sidebar.subheader("Feature & Target Selection")
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

# Remove FIPS from features
if 'FIPS' in numeric_cols:
    numeric_cols.remove('FIPS')

# ==================== LINEAR REGRESSION ====================
if algorithm_category == "Linear Regression":
    st.header("Linear Regression Models")
    
    st.markdown("""
    **Linear Regression** models the relationship between features and a continuous target variable.
    
    - **Plain OLS (Ordinary Least Squares)**: Standard linear regression, minimizes sum of squared residuals
    - **Ridge (L2 regularization)**: Adds penalty proportional to square of coefficients (prevents large coefficients)
    - **Lasso (L1 regularization)**: Adds penalty proportional to absolute value of coefficients (can zero out features)
    - **ElasticNet**: Combines both L1 and L2 penalties for balanced regularization
    """)
    
    # Model selection
    model_type = st.selectbox("Select Linear Regression Model", 
                              ["OLS", "Ridge", "Lasso", "ElasticNet"])
    
    # Feature and target selection
    target = st.selectbox("Select Target Variable (continuous)", numeric_cols, 
                         index=numeric_cols.index('Health_Risk_Score') if 'Health_Risk_Score' in numeric_cols else 0)
    available_features = [col for col in numeric_cols if col != target]
    features = st.multiselect("Select Features", available_features, 
                              default=available_features[:5] if len(available_features) >= 5 else available_features)
    
    if features:
        col1, col2 = st.columns(2)
        
        # Hyperparameters based on model type
        if model_type in ["Ridge", "Lasso"]:
            alpha = col1.slider("alpha (regularization strength)", 0.01, 10.0, 1.0, 0.01,
                               help="Higher values = more regularization = simpler models")
        elif model_type == "ElasticNet":
            alpha = col1.slider("alpha (regularization strength)", 0.01, 10.0, 1.0, 0.01)
            l1_ratio = col2.slider("l1_ratio (L1 vs L2 mix)", 0.0, 1.0, 0.5, 0.01,
                                   help="0=Ridge only, 1=Lasso only, 0.5=equal mix")
        
        test_size = col1.slider("Test Set Size", 0.1, 0.5, 0.2, 0.05)
        random_state = col2.number_input("Random State", 0, 100, 42)
        
        if st.button("Train Model", key="train_lr"):
            # Prepare data
            X = df[features].dropna()
            y = df.loc[X.index, target]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train model
            if model_type == "OLS":
                model = LinearRegression()
            elif model_type == "Ridge":
                model = Ridge(alpha=alpha, random_state=random_state)
            elif model_type == "Lasso":
                model = Lasso(alpha=alpha, random_state=random_state)
            else:  # ElasticNet
                model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=random_state)
            
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            
            # Metrics
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test, y_pred)
            
            col1, col2, col3 = st.columns(3)
            col1.metric("MSE", f"{mse:.4f}")
            col2.metric("RMSE", f"{rmse:.4f}")
            col3.metric("R² Score", f"{r2:.4f}")
            
            # Visualizations
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            
            # Actual vs Predicted
            axes[0].scatter(y_test, y_pred, alpha=0.5)
            axes[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
            axes[0].set_xlabel("Actual Values")
            axes[0].set_ylabel("Predicted Values")
            axes[0].set_title("Actual vs Predicted")
            
            # Residuals
            residuals = y_test - y_pred
            axes[1].scatter(y_pred, residuals, alpha=0.5)
            axes[1].axhline(y=0, color='r', linestyle='--')
            axes[1].set_xlabel("Predicted Values")
            axes[1].set_ylabel("Residuals")
            axes[1].set_title("Residual Plot")
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # Feature importance (coefficients)
            st.subheader("Feature Coefficients")
            coef_df = pd.DataFrame({
                'Feature': features,
                'Coefficient': model.coef_
            }).sort_values('Coefficient', key=abs, ascending=False)
            st.dataframe(coef_df, use_container_width=True)

# ==================== LOGISTIC REGRESSION ====================
elif algorithm_category == "Logistic Regression":
    st.header("Logistic Regression (Classification)")
    
    st.markdown("""
    **Logistic Regression** is used for binary or multiclass classification.
    
    - **penalty**: Type of regularization (l1, l2, elasticnet, or none)
    - **C**: Inverse of regularization strength (smaller = stronger regularization)
    - **solver**: Algorithm for optimization (liblinear for small datasets, saga for large/elasticnet)
    - **class_weight**: Adjust weights for imbalanced classes
    """)
    
    # Create binary target
    target = st.selectbox("Select Target Variable", numeric_cols,
                         index=numeric_cols.index('High_Income_Inequality') if 'High_Income_Inequality' in numeric_cols else 0)
    
    available_features = [col for col in numeric_cols if col != target]
    features = st.multiselect("Select Features", available_features,
                              default=available_features[:5] if len(available_features) >= 5 else available_features)
    
    if features:
        col1, col2, col3 = st.columns(3)
        
        penalty = col1.selectbox("penalty", ["l2", "l1", "elasticnet", "none"],
                                help="Regularization type")
        C = col2.slider("C (inverse regularization)", 0.01, 10.0, 1.0, 0.01,
                       help="Smaller C = stronger regularization")
        
        # Solver selection based on penalty
        if penalty == "elasticnet":
            solver = "saga"
            col3.info("solver: saga (required for elasticnet)")
            l1_ratio = col1.slider("l1_ratio", 0.0, 1.0, 0.5, 0.01)
        elif penalty == "l1":
            solver = col3.selectbox("solver", ["liblinear", "saga"])
            l1_ratio = None
        else:
            solver = col3.selectbox("solver", ["liblinear", "saga", "lbfgs", "newton-cg"])
            l1_ratio = None
        
        class_weight = col1.selectbox("class_weight", [None, "balanced"],
                                     help="balanced: adjust weights inversely proportional to class frequencies")
        
        test_size = col2.slider("Test Set Size", 0.1, 0.5, 0.2, 0.05)
        random_state = col3.number_input("Random State", 0, 100, 42)
        
        if st.button("Train Model", key="train_logr"):
            # Prepare data
            X = df[features].dropna()
            y = df.loc[X.index, target]
            
            # Check if binary
            if len(y.unique()) > 10:
                st.warning("Target has many unique values. Converting to binary (median split).")
                y = (y > y.median()).astype(int)
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train model
            model = LogisticRegression(
                penalty=penalty if penalty != "none" else None,
                C=C,
                solver=solver,
                class_weight=class_weight,
                l1_ratio=l1_ratio,
                random_state=random_state,
                max_iter=1000
            )
            
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
            
            # Metrics
            accuracy = accuracy_score(y_test, y_pred)
            
            col1, col2 = st.columns(2)
            col1.metric("Accuracy", f"{accuracy:.4f}")
            
            # Confusion Matrix
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            
            cm = confusion_matrix(y_test, y_pred)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0])
            axes[0].set_xlabel("Predicted")
            axes[0].set_ylabel("Actual")
            axes[0].set_title("Confusion Matrix")
            
            # ROC Curve
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
            roc_auc = auc(fpr, tpr)
            axes[1].plot(fpr, tpr, label=f'ROC (AUC = {roc_auc:.3f})')
            axes[1].plot([0, 1], [0, 1], 'r--')
            axes[1].set_xlabel("False Positive Rate")
            axes[1].set_ylabel("True Positive Rate")
            axes[1].set_title("ROC Curve")
            axes[1].legend()
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # Classification report
            st.subheader("Classification Report")
            report = classification_report(y_test, y_pred, output_dict=True)
            st.dataframe(pd.DataFrame(report).transpose())

# ==================== SVM ====================
elif algorithm_category == "SVM":
    st.header("Support Vector Machine (SVM)")
    
    st.markdown("""
    **SVM** finds the optimal hyperplane that separates classes (SVC) or fits data (SVR).
    
    - **kernel**: Transformation function (linear, rbf=radial basis function, poly, sigmoid)
    - **C**: Penalty parameter (larger C = less regularization = tighter fit to training data)
    - **gamma**: Kernel coefficient for rbf/poly/sigmoid (higher = more complex decision boundary)
    - **probability**: Enable probability estimates (needed for ROC curves, but slower)
    """)
    
    # Task selection
    task = st.selectbox("Select Task", ["Classification (SVC)", "Regression (SVR)"])
    
    target = st.selectbox("Select Target Variable", numeric_cols)
    available_features = [col for col in numeric_cols if col != target]
    features = st.multiselect("Select Features", available_features,
                              default=available_features[:5] if len(available_features) >= 5 else available_features)
    
    if features:
        col1, col2, col3 = st.columns(3)
        
        kernel = col1.selectbox("kernel", ["linear", "rbf", "poly", "sigmoid"])
        C = col2.slider("C (penalty parameter)", 0.01, 10.0, 1.0, 0.01)
        
        if kernel in ["rbf", "poly", "sigmoid"]:
            gamma_option = col3.selectbox("gamma", ["scale", "auto", "custom"])
            if gamma_option == "custom":
                gamma = col1.slider("gamma value", 0.001, 1.0, 0.1, 0.001)
            else:
                gamma = gamma_option
        else:
            gamma = "scale"
            col3.info("gamma not used for linear kernel")
        
        if "Classification" in task:
            probability = col1.checkbox("Enable probability estimates", value=True,
                                       help="Required for ROC curves")
        
        test_size = col2.slider("Test Set Size", 0.1, 0.5, 0.2, 0.05)
        random_state = col3.number_input("Random State", 0, 100, 42)
        
        if st.button("Train Model", key="train_svm"):
            # Prepare data
            X = df[features].dropna()
            y = df.loc[X.index, target]
            
            if "Classification" in task:
                # Convert to binary if needed
                if len(y.unique()) > 10:
                    st.warning("Target has many unique values. Converting to binary (median split).")
                    y = (y > y.median()).astype(int)
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
            
            # Scale features (important for SVM)
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train model
            with st.spinner("Training SVM... This may take a moment."):
                if "Classification" in task:
                    model = SVC(kernel=kernel, C=C, gamma=gamma, probability=probability, random_state=random_state)
                else:
                    model = SVR(kernel=kernel, C=C, gamma=gamma)
                
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
            
            if "Classification" in task:
                # Classification metrics
                accuracy = accuracy_score(y_test, y_pred)
                st.metric("Accuracy", f"{accuracy:.4f}")
                
                # Visualizations
                if probability:
                    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
                    
                    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
                    
                    # Confusion Matrix
                    cm = confusion_matrix(y_test, y_pred)
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0])
                    axes[0].set_xlabel("Predicted")
                    axes[0].set_ylabel("Actual")
                    axes[0].set_title("Confusion Matrix")
                    
                    # ROC Curve
                    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
                    roc_auc = auc(fpr, tpr)
                    axes[1].plot(fpr, tpr, label=f'ROC (AUC = {roc_auc:.3f})')
                    axes[1].plot([0, 1], [0, 1], 'r--')
                    axes[1].set_xlabel("False Positive Rate")
                    axes[1].set_ylabel("True Positive Rate")
                    axes[1].set_title("ROC Curve")
                    axes[1].legend()
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                else:
                    fig, ax = plt.subplots(figsize=(7, 5))
                    cm = confusion_matrix(y_test, y_pred)
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
                    ax.set_xlabel("Predicted")
                    ax.set_ylabel("Actual")
                    ax.set_title("Confusion Matrix")
                    st.pyplot(fig)
                
                # Classification report
                st.subheader("Classification Report")
                report = classification_report(y_test, y_pred, output_dict=True)
                st.dataframe(pd.DataFrame(report).transpose())
                
            else:
                # Regression metrics
                mse = mean_squared_error(y_test, y_pred)
                rmse = np.sqrt(mse)
                r2 = r2_score(y_test, y_pred)
                
                col1, col2, col3 = st.columns(3)
                col1.metric("MSE", f"{mse:.4f}")
                col2.metric("RMSE", f"{rmse:.4f}")
                col3.metric("R² Score", f"{r2:.4f}")
                
                # Visualizations
                fig, axes = plt.subplots(1, 2, figsize=(14, 5))
                
                axes[0].scatter(y_test, y_pred, alpha=0.5)
                axes[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
                axes[0].set_xlabel("Actual Values")
                axes[0].set_ylabel("Predicted Values")
                axes[0].set_title("Actual vs Predicted")
                
                residuals = y_test - y_pred
                axes[1].scatter(y_pred, residuals, alpha=0.5)
                axes[1].axhline(y=0, color='r', linestyle='--')
                axes[1].set_xlabel("Predicted Values")
                axes[1].set_ylabel("Residuals")
                axes[1].set_title("Residual Plot")
                
                plt.tight_layout()
                st.pyplot(fig)

# ==================== DECISION TREE ====================
elif algorithm_category == "Decision Tree":
    st.header("Decision Tree")
    
    st.markdown("""
    **Decision Trees** create a tree-like model of decisions based on feature values.
    
    - **criterion**: How to measure split quality
      - Classification: gini (impurity), entropy (information gain)
      - Regression: squared_error (variance reduction)
    - **max_depth**: Maximum tree depth (None = unlimited, can overfit)
    - **min_samples_split**: Minimum samples required to split a node
    - **min_samples_leaf**: Minimum samples required in a leaf node
    """)
    
    task = st.selectbox("Select Task", ["Classification", "Regression"])
    
    target = st.selectbox("Select Target Variable", numeric_cols)
    available_features = [col for col in numeric_cols if col != target]
    features = st.multiselect("Select Features", available_features,
                              default=available_features[:5] if len(available_features) >= 5 else available_features)
    
    if features:
        col1, col2, col3 = st.columns(3)
        
        if task == "Classification":
            criterion = col1.selectbox("criterion", ["gini", "entropy"])
        else:
            criterion = col1.selectbox("criterion", ["squared_error", "absolute_error"])
        
        max_depth = col2.slider("max_depth", 1, 20, 5,
                               help="Deeper trees = more complex models = risk of overfitting")
        min_samples_split = col3.slider("min_samples_split", 2, 20, 2)
        min_samples_leaf = col1.slider("min_samples_leaf", 1, 20, 1)
        
        test_size = col2.slider("Test Set Size", 0.1, 0.5, 0.2, 0.05)
        random_state = col3.number_input("Random State", 0, 100, 42)
        
        if st.button("Train Model", key="train_dt"):
            # Prepare data
            X = df[features].dropna()
            y = df.loc[X.index, target]
            
            if task == "Classification":
                if len(y.unique()) > 10:
                    st.warning("Target has many unique values. Converting to binary (median split).")
                    y = (y > y.median()).astype(int)
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
            
            # Train model
            if task == "Classification":
                model = DecisionTreeClassifier(
                    criterion=criterion,
                    max_depth=max_depth,
                    min_samples_split=min_samples_split,
                    min_samples_leaf=min_samples_leaf,
                    random_state=random_state
                )
            else:
                model = DecisionTreeRegressor(
                    criterion=criterion,
                    max_depth=max_depth,
                    min_samples_split=min_samples_split,
                    min_samples_leaf=min_samples_leaf,
                    random_state=random_state
                )
            
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            if task == "Classification":
                accuracy = accuracy_score(y_test, y_pred)
                st.metric("Accuracy", f"{accuracy:.4f}")
                
                fig, ax = plt.subplots(figsize=(7, 5))
                cm = confusion_matrix(y_test, y_pred)
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
                ax.set_xlabel("Predicted")
                ax.set_ylabel("Actual")
                ax.set_title("Confusion Matrix")
                st.pyplot(fig)
                
                st.subheader("Classification Report")
                report = classification_report(y_test, y_pred, output_dict=True)
                st.dataframe(pd.DataFrame(report).transpose())
            else:
                mse = mean_squared_error(y_test, y_pred)
                rmse = np.sqrt(mse)
                r2 = r2_score(y_test, y_pred)
                
                col1, col2, col3 = st.columns(3)
                col1.metric("MSE", f"{mse:.4f}")
                col2.metric("RMSE", f"{rmse:.4f}")
                col3.metric("R² Score", f"{r2:.4f}")
                
                fig, axes = plt.subplots(1, 2, figsize=(14, 5))
                
                axes[0].scatter(y_test, y_pred, alpha=0.5)
                axes[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
                axes[0].set_xlabel("Actual Values")
                axes[0].set_ylabel("Predicted Values")
                axes[0].set_title("Actual vs Predicted")
                
                residuals = y_test - y_pred
                axes[1].scatter(y_pred, residuals, alpha=0.5)
                axes[1].axhline(y=0, color='r', linestyle='--')
                axes[1].set_xlabel("Predicted Values")
                axes[1].set_ylabel("Residuals")
                axes[1].set_title("Residual Plot")
                
                plt.tight_layout()
                st.pyplot(fig)
            
            # Feature importance
            st.subheader("Feature Importance")
            importance_df = pd.DataFrame({
                'Feature': features,
                'Importance': model.feature_importances_
            }).sort_values('Importance', ascending=False)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.barh(importance_df['Feature'], importance_df['Importance'])
            ax.set_xlabel("Importance")
            ax.set_title("Feature Importance")
            plt.tight_layout()
            st.pyplot(fig)

# ==================== RANDOM FOREST ====================
elif algorithm_category == "Random Forest":
    st.header("Random Forest")
    
    st.markdown("""
    **Random Forest** builds multiple decision trees and combines their predictions (ensemble method).
    
    - **n_estimators**: Number of trees in the forest (more trees = better but slower)
    - **max_depth**: Maximum depth of each tree (None = unlimited)
    - **max_features**: Number of features to consider for best split
    - **min_samples_leaf**: Minimum samples in leaf nodes
    - **class_weight**: Balance class weights (for classification)
    """)
    
    task = st.selectbox("Select Task", ["Classification", "Regression"])
    
    target = st.selectbox("Select Target Variable", numeric_cols)
    available_features = [col for col in numeric_cols if col != target]
    features = st.multiselect("Select Features", available_features,
                              default=available_features[:5] if len(available_features) >= 5 else available_features)
    
    if features:
        col1, col2, col3 = st.columns(3)
        
        n_estimators = col1.slider("n_estimators (# of trees)", 10, 200, 100, 10)
        max_depth_option = col2.selectbox("max_depth", ["None", "Custom"])
        if max_depth_option == "Custom":
            max_depth = col3.slider("max_depth value", 1, 20, 5)
        else:
            max_depth = None
        
        max_features = col1.selectbox("max_features", ["sqrt", "log2", None],
                                     help="sqrt: good default for classification, None: use all features")
        min_samples_leaf = col2.slider("min_samples_leaf", 1, 10, 1)
        
        if task == "Classification":
            class_weight = col3.selectbox("class_weight", [None, "balanced"])
        
        test_size = col1.slider("Test Set Size", 0.1, 0.5, 0.2, 0.05)
        random_state = col2.number_input("Random State", 0, 100, 42)
        
        if st.button("Train Model", key="train_rf"):
            # Prepare data
            X = df[features].dropna()
            y = df.loc[X.index, target]
            
            if task == "Classification":
                if len(y.unique()) > 10:
                    st.warning("Target has many unique values. Converting to binary (median split).")
                    y = (y > y.median()).astype(int)
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
            
            # Train model
            with st.spinner("Training Random Forest..."):
                if task == "Classification":
                    model = RandomForestClassifier(
                        n_estimators=n_estimators,
                        max_depth=max_depth,
                        max_features=max_features,
                        min_samples_leaf=min_samples_leaf,
                        class_weight=class_weight,
                        random_state=random_state,
                        n_jobs=-1
                    )
                else:
                    model = RandomForestRegressor(
                        n_estimators=n_estimators,
                        max_depth=max_depth,
                        max_features=max_features,
                        min_samples_leaf=min_samples_leaf,
                        random_state=random_state,
                        n_jobs=-1
                    )
                
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
            
            if task == "Classification":
                accuracy = accuracy_score(y_test, y_pred)
                st.metric("Accuracy", f"{accuracy:.4f}")
                
                fig, ax = plt.subplots(figsize=(7, 5))
                cm = confusion_matrix(y_test, y_pred)
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
                ax.set_xlabel("Predicted")
                ax.set_ylabel("Actual")
                ax.set_title("Confusion Matrix")
                st.pyplot(fig)
                
                st.subheader("Classification Report")
                report = classification_report(y_test, y_pred, output_dict=True)
                st.dataframe(pd.DataFrame(report).transpose())
            else:
                mse = mean_squared_error(y_test, y_pred)
                rmse = np.sqrt(mse)
                r2 = r2_score(y_test, y_pred)
                
                col1, col2, col3 = st.columns(3)
                col1.metric("MSE", f"{mse:.4f}")
                col2.metric("RMSE", f"{rmse:.4f}")
                col3.metric("R² Score", f"{r2:.4f}")
                
                fig, axes = plt.subplots(1, 2, figsize=(14, 5))
                
                axes[0].scatter(y_test, y_pred, alpha=0.5)
                axes[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
                axes[0].set_xlabel("Actual Values")
                axes[0].set_ylabel("Predicted Values")
                axes[0].set_title("Actual vs Predicted")
                
                residuals = y_test - y_pred
                axes[1].scatter(y_pred, residuals, alpha=0.5)
                axes[1].axhline(y=0, color='r', linestyle='--')
                axes[1].set_xlabel("Predicted Values")
                axes[1].set_ylabel("Residuals")
                axes[1].set_title("Residual Plot")
                
                plt.tight_layout()
                st.pyplot(fig)
            
            # Feature importance
            st.subheader("Feature Importance")
            importance_df = pd.DataFrame({
                'Feature': features,
                'Importance': model.feature_importances_
            }).sort_values('Importance', ascending=False)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.barh(importance_df['Feature'], importance_df['Importance'])
            ax.set_xlabel("Importance")
            ax.set_title("Feature Importance (Average across all trees)")
            plt.tight_layout()
            st.pyplot(fig)

# ==================== KNN ====================
elif algorithm_category == "KNN":
    st.header("K-Nearest Neighbors (KNN)")
    
    st.markdown("""
    **KNN** classifies/predicts based on the k nearest neighbors in feature space.
    
    - **n_neighbors**: Number of neighbors to consider (k value)
    - **weights**: How to weight neighbors
      - uniform: all neighbors weighted equally
      - distance: closer neighbors have more influence
    - **metric**: Distance calculation method (minkowski with p=2 is Euclidean distance)
    - **p**: Power parameter for Minkowski metric (p=1: Manhattan, p=2: Euclidean)
    """)
    
    task = st.selectbox("Select Task", ["Classification", "Regression"])
    
    target = st.selectbox("Select Target Variable", numeric_cols)
    available_features = [col for col in numeric_cols if col != target]
    features = st.multiselect("Select Features", available_features,
                              default=available_features[:5] if len(available_features) >= 5 else available_features)
    
    if features:
        col1, col2, col3 = st.columns(3)
        
        n_neighbors = col1.slider("n_neighbors (k)", 1, 50, 5,
                                 help="Smaller k = more complex decision boundary")
        weights = col2.selectbox("weights", ["uniform", "distance"])
        metric = col3.selectbox("metric", ["minkowski", "euclidean", "manhattan"])
        
        if metric == "minkowski":
            p = col1.slider("p (power parameter)", 1, 5, 2,
                          help="p=1: Manhattan distance, p=2: Euclidean distance")
        else:
            p = 2
        
        test_size = col2.slider("Test Set Size", 0.1, 0.5, 0.2, 0.05)
        random_state = col3.number_input("Random State", 0, 100, 42)
        
        if st.button("Train Model", key="train_knn"):
            # Prepare data
            X = df[features].dropna()
            y = df.loc[X.index, target]
            
            if task == "Classification":
                if len(y.unique()) > 10:
                    st.warning("Target has many unique values. Converting to binary (median split).")
                    y = (y > y.median()).astype(int)
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
            
            # Scale features (important for distance-based methods)
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train model
            if task == "Classification":
                model = KNeighborsClassifier(
                    n_neighbors=n_neighbors,
                    weights=weights,
                    metric=metric,
                    p=p
                )
            else:
                model = KNeighborsRegressor(
                    n_neighbors=n_neighbors,
                    weights=weights,
                    metric=metric,
                    p=p
                )
            
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            
            if task == "Classification":
                accuracy = accuracy_score(y_test, y_pred)
                st.metric("Accuracy", f"{accuracy:.4f}")
                
                fig, ax = plt.subplots(figsize=(7, 5))
                cm = confusion_matrix(y_test, y_pred)
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
                ax.set_xlabel("Predicted")
                ax.set_ylabel("Actual")
                ax.set_title("Confusion Matrix")
                st.pyplot(fig)
                
                st.subheader("Classification Report")
                report = classification_report(y_test, y_pred, output_dict=True)
                st.dataframe(pd.DataFrame(report).transpose())
            else:
                mse = mean_squared_error(y_test, y_pred)
                rmse = np.sqrt(mse)
                r2 = r2_score(y_test, y_pred)
                
                col1, col2, col3 = st.columns(3)
                col1.metric("MSE", f"{mse:.4f}")
                col2.metric("RMSE", f"{rmse:.4f}")
                col3.metric("R² Score", f"{r2:.4f}")
                
                fig, axes = plt.subplots(1, 2, figsize=(14, 5))
                
                axes[0].scatter(y_test, y_pred, alpha=0.5)
                axes[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
                axes[0].set_xlabel("Actual Values")
                axes[0].set_ylabel("Predicted Values")
                axes[0].set_title("Actual vs Predicted")
                
                residuals = y_test - y_pred
                axes[1].scatter(y_pred, residuals, alpha=0.5)
                axes[1].axhline(y=0, color='r', linestyle='--')
                axes[1].set_xlabel("Predicted Values")
                axes[1].set_ylabel("Residuals")
                axes[1].set_title("Residual Plot")
                
                plt.tight_layout()
                st.pyplot(fig)

# ==================== NAIVE BAYES ====================
elif algorithm_category == "Naive Bayes":
    st.header("Naive Bayes")
    
    st.markdown("""
    **Naive Bayes** applies Bayes' theorem with the "naive" assumption of feature independence.
    
    - **GaussianNB**: Assumes features follow a Gaussian (normal) distribution
      - var_smoothing: Portion of largest variance added to variances for stability
    - **MultinomialNB**: For discrete counts (e.g., word counts in text)
      - alpha: Additive (Laplace) smoothing parameter
      - Requires non-negative features
    """)
    
    nb_type = st.selectbox("Select Naive Bayes Type", ["GaussianNB", "MultinomialNB"])
    
    target = st.selectbox("Select Target Variable", numeric_cols)
    available_features = [col for col in numeric_cols if col != target]
    features = st.multiselect("Select Features", available_features,
                              default=available_features[:5] if len(available_features) >= 5 else available_features)
    
    if features:
        col1, col2, col3 = st.columns(3)
        
        if nb_type == "GaussianNB":
            var_smoothing = col1.slider("var_smoothing", 1e-9, 1e-5, 1e-9, 1e-10,
                                       format="%.2e",
                                       help="Stability parameter for variance")
        else:
            alpha = col1.slider("alpha (smoothing)", 0.01, 10.0, 1.0, 0.01,
                               help="Laplace smoothing parameter")
            st.info("⚠️ MultinomialNB requires non-negative features. Negative values will be clipped to 0.")
        
        test_size = col2.slider("Test Set Size", 0.1, 0.5, 0.2, 0.05)
        random_state = col3.number_input("Random State", 0, 100, 42)
        
        if st.button("Train Model", key="train_nb"):
            # Prepare data
            X = df[features].dropna()
            y = df.loc[X.index, target]
            
            if len(y.unique()) > 10:
                st.warning("Target has many unique values. Converting to binary (median split).")
                y = (y > y.median()).astype(int)
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
            
            # For MultinomialNB, ensure non-negative features
            if nb_type == "MultinomialNB":
                X_train = np.clip(X_train, 0, None)
                X_test = np.clip(X_test, 0, None)
            
            # Train model
            if nb_type == "GaussianNB":
                model = GaussianNB(var_smoothing=var_smoothing)
            else:
                model = MultinomialNB(alpha=alpha)
            
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            # Metrics
            accuracy = accuracy_score(y_test, y_pred)
            st.metric("Accuracy", f"{accuracy:.4f}")
            
            # Visualizations
            fig, ax = plt.subplots(figsize=(7, 5))
            cm = confusion_matrix(y_test, y_pred)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            ax.set_title("Confusion Matrix")
            st.pyplot(fig)
            
            # Classification report
            st.subheader("Classification Report")
            report = classification_report(y_test, y_pred, output_dict=True)
            st.dataframe(pd.DataFrame(report).transpose())

# ==================== K-MEANS ====================
elif algorithm_category == "K-Means Clustering":
    st.header("K-Means Clustering")
    
    st.markdown("""
    **K-Means** partitions data into k clusters by minimizing within-cluster variance.
    
    - **n_clusters**: Number of clusters (k)
    - **init**: Method for initialization
      - k-means++: Smart initialization (default, usually better)
      - random: Random initialization
    - **n_init**: Number of times to run with different initializations
    - **max_iter**: Maximum number of iterations for convergence
    """)
    
    features = st.multiselect("Select Features for Clustering", numeric_cols,
                              default=numeric_cols[:5] if len(numeric_cols) >= 5 else numeric_cols)
    
    if features:
        col1, col2, col3 = st.columns(3)
        
        n_clusters = col1.slider("n_clusters (k)", 2, 10, 3)
        init = col2.selectbox("init", ["k-means++", "random"])
        n_init = col3.slider("n_init", 1, 20, 10,
                            help="More runs = better result but slower")
        max_iter = col1.slider("max_iter", 100, 1000, 300, 50)
        random_state = col2.number_input("Random State", 0, 100, 42)
        
        if st.button("Perform Clustering", key="train_kmeans"):
            # Prepare data
            X = df[features].dropna()
            
            # Scale features (important for K-Means)
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Train model
            with st.spinner("Running K-Means..."):
                model = KMeans(
                    n_clusters=n_clusters,
                    init=init,
                    n_init=n_init,
                    max_iter=max_iter,
                    random_state=random_state
                )
                
                clusters = model.fit_predict(X_scaled)
            
            # Metrics
            inertia = model.inertia_
            silhouette = silhouette_score(X_scaled, clusters)
            
            col1, col2 = st.columns(2)
            col1.metric("Inertia (within-cluster sum of squares)", f"{inertia:.2f}",
                       help="Lower is better, but decreases with more clusters")
            col2.metric("Silhouette Score", f"{silhouette:.4f}",
                       help="Ranges from -1 to 1. Higher is better. >0.5 is good.")
            
            # Add cluster labels to data
            X_with_clusters = X.copy()
            X_with_clusters['Cluster'] = clusters
            
            # Visualizations
            if len(features) >= 2:
                fig, axes = plt.subplots(1, 2, figsize=(14, 5))
                
                # Scatter plot of first two features
                scatter = axes[0].scatter(X[features[0]], X[features[1]], 
                                        c=clusters, cmap='viridis', alpha=0.6)
                axes[0].scatter(model.cluster_centers_[:, 0] * scaler.scale_[0] + scaler.mean_[0],
                              model.cluster_centers_[:, 1] * scaler.scale_[1] + scaler.mean_[1],
                              c='red', marker='X', s=200, edgecolors='black', label='Centroids')
                axes[0].set_xlabel(features[0])
                axes[0].set_ylabel(features[1])
                axes[0].set_title("Cluster Visualization")
                axes[0].legend()
                plt.colorbar(scatter, ax=axes[0], label='Cluster')
                
                # Cluster sizes
                cluster_sizes = pd.Series(clusters).value_counts().sort_index()
                axes[1].bar(cluster_sizes.index, cluster_sizes.values, color='skyblue')
                axes[1].set_xlabel("Cluster")
                axes[1].set_ylabel("Number of Points")
                axes[1].set_title("Cluster Sizes")
                axes[1].set_xticks(range(n_clusters))
                
                plt.tight_layout()
                st.pyplot(fig)
            
            # Cluster statistics
            st.subheader("Cluster Statistics")
            cluster_stats = X_with_clusters.groupby('Cluster')[features].mean()
            st.dataframe(cluster_stats)
            
            # Elbow plot suggestion
            st.info("💡 **Tip**: To find the optimal number of clusters, try running K-Means with different values of k and plot the inertia (elbow method) or silhouette score.")

# ==================== PCA ====================
elif algorithm_category == "PCA":
    st.header("Principal Component Analysis (PCA)")
    
    st.markdown("""
    **PCA** reduces dimensionality by finding principal components (directions of maximum variance).
    
    - **n_components**: Number of components to keep
      - Integer: Keep exactly that many components
      - Float (0-1): Keep enough components to explain that fraction of variance
      - Example: 0.95 means keep enough components to explain 95% of variance
    """)
    
    features = st.multiselect("Select Features for PCA", numeric_cols,
                              default=numeric_cols[:10] if len(numeric_cols) >= 10 else numeric_cols)
    
    if features:
        col1, col2 = st.columns(2)
        
        n_components_type = col1.selectbox("n_components type", ["Integer (exact number)", "Float (variance threshold)"])
        
        if n_components_type == "Integer (exact number)":
            n_components = col2.slider("n_components", 1, min(len(features), 10), 
                                      min(2, len(features)))
        else:
            n_components = col2.slider("n_components (variance to explain)", 0.5, 0.99, 0.95, 0.01)
        
        random_state = col1.number_input("Random State", 0, 100, 42)
        
        if st.button("Perform PCA", key="train_pca"):
            # Prepare data
            X = df[features].dropna()
            
            # Scale features (important for PCA)
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Perform PCA
            pca = PCA(n_components=n_components, random_state=random_state)
            X_pca = pca.fit_transform(X_scaled)
            
            # Results
            st.subheader("PCA Results")
            
            col1, col2 = st.columns(2)
            col1.metric("Components Kept", pca.n_components_)
            col2.metric("Total Variance Explained", f"{pca.explained_variance_ratio_.sum():.2%}")
            
            # Explained variance
            st.subheader("Explained Variance by Component")
            variance_df = pd.DataFrame({
                'Component': [f'PC{i+1}' for i in range(pca.n_components_)],
                'Variance Explained': pca.explained_variance_ratio_,
                'Cumulative Variance': np.cumsum(pca.explained_variance_ratio_)
            })
            st.dataframe(variance_df)
            
            # Visualizations
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            
            # Scree plot
            axes[0].bar(range(1, pca.n_components_ + 1), pca.explained_variance_ratio_, 
                       color='skyblue', alpha=0.7, label='Individual')
            axes[0].plot(range(1, pca.n_components_ + 1), np.cumsum(pca.explained_variance_ratio_),
                        'ro-', linewidth=2, label='Cumulative')
            axes[0].set_xlabel('Principal Component')
            axes[0].set_ylabel('Variance Explained')
            axes[0].set_title('Scree Plot')
            axes[0].legend()
            axes[0].set_xticks(range(1, pca.n_components_ + 1))
            
            # First two components scatter
            if pca.n_components_ >= 2:
                axes[1].scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.5)
                axes[1].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
                axes[1].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
                axes[1].set_title('First Two Principal Components')
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # Component loadings (contributions of original features)
            st.subheader("Component Loadings")
            st.write("Shows how much each original feature contributes to each principal component")
            
            loadings = pd.DataFrame(
                pca.components_.T,
                columns=[f'PC{i+1}' for i in range(pca.n_components_)],
                index=features
            )
            
            # Heatmap
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(loadings, cmap='coolwarm', center=0, annot=False, ax=ax)
            ax.set_title('PCA Component Loadings Heatmap')
            plt.tight_layout()
            st.pyplot(fig)
            
            # Top contributors
            st.subheader("Top Feature Contributors per Component")
            for i in range(min(3, pca.n_components_)):
                st.write(f"**PC{i+1}:**")
                top_features = loadings[f'PC{i+1}'].abs().sort_values(ascending=False).head(5)
                st.write(top_features)

# Footer with tips
st.markdown("---")
st.markdown("""
### Learning Tips

**Understanding Hyperparameters:**
- Start with default values, then experiment
- **Regularization** (alpha, C): Controls model complexity vs. overfitting
- **Tree depth**: Deeper = more complex = risk of overfitting
- **Number of estimators**: More = better but slower
- **Distance metrics**: Affect how similarity is measured

**Model Selection:**
- **Linear models**: Fast, interpretable, good for linear relationships
- **Tree-based**: Handle non-linear relationships, no scaling needed
- **SVM**: Powerful but can be slow on large datasets
- **KNN**: Simple but slow on large datasets, sensitive to scaling
- **Naive Bayes**: Fast, works well with limited data
- **K-Means**: Unsupervised, requires choosing k
- **PCA**: Reduces dimensions, helps visualization

**Best Practices:**
- Always split data into train/test sets
- Scale features for distance-based methods (SVM, KNN, PCA)
- Check for class imbalance in classification
- Use cross-validation for more robust evaluation
- Start simple, then add complexity if needed
""")