"""
Train and Save All ML Models for County Health Analysis

This script trains all models used in the project and saves them to the models/ directory.
Models include:
- Regression models (Ridge, Linear)
- Classification models (Logistic Regression, SVM, Random Forest, Extra Trees)
- 3-Class Health Prediction models (Random Forest, SVM)
- Clustering models (K-Means)
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.cluster import KMeans

# Define paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_PATH = PROJECT_ROOT / 'data' / 'processed' / 'cleaned_health_data.csv'
RAW_DATA_PATH = PROJECT_ROOT / 'data' / 'raw' / '2025CountyHealthRankingsDatav3.xlsx'
MODELS_DIR = PROJECT_ROOT / 'models'

# Create models directory if it doesn't exist
MODELS_DIR.mkdir(exist_ok=True)

print(f"Loading data from {DATA_PATH}...")

# Load cleaned data for binary classification and regression
df = pd.read_csv(DATA_PATH)

# Convert Primary Care Physicians Ratio from string to float
if df['Primary Care Physicians Ratio'].dtype == 'object':
    df['Primary Care Physicians Ratio'] = df['Primary Care Physicians Ratio'].str.split(':').str[0].astype(float)

# Define features for different tasks
REGRESSION_FEATURES = [
    'Food_Access_Barrier_Index',
    'Socioeconomic_Vulnerability_Index',
    '% Completed High School',
    'Income Ratio',
    '% Uninsured',
    '% Rural',
    'Primary Care Physicians Ratio'
]

CLASSIFICATION_FEATURES = [
    'Food_Access_Barrier_Index',
    'Socioeconomic_Vulnerability_Index',
    '% Completed High School',
    'Income Ratio',
    '% Uninsured',
    '% Rural',
    'Primary Care Physicians Ratio',
    '% Excessive Drinking'
]

print("\n" + "="*80)
print("TRAINING REGRESSION MODELS")
print("="*80)

# Prepare regression data
X_reg = df[REGRESSION_FEATURES].copy()
y_obesity = df['% Adults with Obesity'].copy()
y_diabetes = df['% Adults with Diabetes'].copy()

# Drop rows with NaN values
valid_indices = X_reg.dropna().index.intersection(y_obesity.dropna().index).intersection(y_diabetes.dropna().index)
X_reg = X_reg.loc[valid_indices]
y_obesity = y_obesity.loc[valid_indices]
y_diabetes = y_diabetes.loc[valid_indices]

print(f"Number of valid samples for regression: {len(X_reg)}")

# Split data
X_train_reg, X_test_reg, y_train_obesity, y_test_obesity = train_test_split(
    X_reg, y_obesity, test_size=0.2, random_state=42
)
_, _, y_train_diabetes, y_test_diabetes = train_test_split(
    X_reg, y_diabetes, test_size=0.2, random_state=42
)

# Scale features for regression
scaler_reg = StandardScaler()
X_train_reg_scaled = scaler_reg.fit_transform(X_train_reg)
X_test_reg_scaled = scaler_reg.transform(X_test_reg)

# Train Linear Regression for Obesity
print("\n1. Training Linear Regression (Obesity)...")
lr_obesity = LinearRegression()
lr_obesity.fit(X_train_reg_scaled, y_train_obesity)
joblib.dump(lr_obesity, MODELS_DIR / 'linear_regression_obesity.pkl')
print(f"   Saved to: {MODELS_DIR / 'linear_regression_obesity.pkl'}")

# Train Ridge Regression for Obesity
print("2. Training Ridge Regression (Obesity)...")
ridge_obesity = Ridge(alpha=1.0, random_state=42)
ridge_obesity.fit(X_train_reg_scaled, y_train_obesity)
joblib.dump(ridge_obesity, MODELS_DIR / 'ridge_regression_obesity.pkl')
print(f"   Saved to: {MODELS_DIR / 'ridge_regression_obesity.pkl'}")

# Train Linear Regression for Diabetes
print("3. Training Linear Regression (Diabetes)...")
lr_diabetes = LinearRegression()
lr_diabetes.fit(X_train_reg_scaled, y_train_diabetes)
joblib.dump(lr_diabetes, MODELS_DIR / 'linear_regression_diabetes.pkl')
print(f"   Saved to: {MODELS_DIR / 'linear_regression_diabetes.pkl'}")

# Train Ridge Regression for Diabetes
print("4. Training Ridge Regression (Diabetes)...")
ridge_diabetes = Ridge(alpha=1.0, random_state=42)
ridge_diabetes.fit(X_train_reg_scaled, y_train_diabetes)
joblib.dump(ridge_diabetes, MODELS_DIR / 'ridge_regression_diabetes.pkl')
print(f"   Saved to: {MODELS_DIR / 'ridge_regression_diabetes.pkl'}")

# Save regression scaler
joblib.dump(scaler_reg, MODELS_DIR / 'scaler_regression.pkl')
print(f"5. Saved regression scaler to: {MODELS_DIR / 'scaler_regression.pkl'}")

print("\n" + "="*80)
print("TRAINING BINARY CLASSIFICATION MODELS (Income Inequality)")
print("="*80)

# Prepare classification data
X_clf = df[CLASSIFICATION_FEATURES].copy()
y_clf = df['High_Income_Inequality'].copy()

# Drop rows with NaN values
valid_indices_clf = X_clf.dropna().index.intersection(y_clf.dropna().index)
X_clf = X_clf.loc[valid_indices_clf]
y_clf = y_clf.loc[valid_indices_clf]

print(f"Number of valid samples for classification: {len(X_clf)}")

# Split data
X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(
    X_clf, y_clf, test_size=0.2, random_state=42, stratify=y_clf
)

# Scale features
scaler_clf = StandardScaler()
X_train_clf_scaled = scaler_clf.fit_transform(X_train_clf)
X_test_clf_scaled = scaler_clf.transform(X_test_clf)

# Train Logistic Regression
print("\n1. Training Logistic Regression...")
log_reg = LogisticRegression(max_iter=1000, random_state=42)
log_reg.fit(X_train_clf_scaled, y_train_clf)
joblib.dump(log_reg, MODELS_DIR / 'logistic_regression.pkl')
print(f"   Saved to: {MODELS_DIR / 'logistic_regression.pkl'}")

# Train SVM
print("2. Training SVM (RBF kernel)...")
svm_clf = SVC(kernel='rbf', probability=True, random_state=42)
svm_clf.fit(X_train_clf_scaled, y_train_clf)
joblib.dump(svm_clf, MODELS_DIR / 'svm_binary.pkl')
print(f"   Saved to: {MODELS_DIR / 'svm_binary.pkl'}")

# Train Random Forest
print("3. Training Random Forest...")
rf_clf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
rf_clf.fit(X_train_clf_scaled, y_train_clf)
joblib.dump(rf_clf, MODELS_DIR / 'random_forest_binary.pkl')
print(f"   Saved to: {MODELS_DIR / 'random_forest_binary.pkl'}")

# Train Extra Trees
print("4. Training Extra Trees...")
et_clf = ExtraTreesClassifier(n_estimators=100, max_depth=5, random_state=42)
et_clf.fit(X_train_clf_scaled, y_train_clf)
joblib.dump(et_clf, MODELS_DIR / 'extra_trees_binary.pkl')
print(f"   Saved to: {MODELS_DIR / 'extra_trees_binary.pkl'}")

# Save classification scaler
joblib.dump(scaler_clf, MODELS_DIR / 'scaler_classification.pkl')
print(f"5. Saved classification scaler to: {MODELS_DIR / 'scaler_classification.pkl'}")

print("\n" + "="*80)
print("TRAINING 3-CLASS HEALTH PREDICTION MODELS")
print("="*80)

# Load raw data for 3-class prediction
print(f"\nLoading raw data from {RAW_DATA_PATH}...")
raw_df = pd.read_excel(RAW_DATA_PATH, sheet_name='Select Measure Data', header=1)

# Preprocess exactly like notebook
null_count = raw_df.isnull().sum()
null_cols_gt_700 = list(null_count[null_count > 700].index)
new_df = raw_df.drop(axis=1, columns=null_cols_gt_700)
new_df = new_df.drop(columns=['FIPS', 'State', 'County'], errors='ignore')
cols_to_drop = [x for x in new_df.columns if '95% CI' in x or 'National Z-Score' in x]
new_df = new_df.drop(columns=cols_to_drop)
new_df = new_df.drop(columns=['Presence of Water Violation'], errors='ignore')
final_df = new_df.dropna()

# Create 3-class target
final_df['Health_Class'] = pd.qcut(
    final_df['% Fair or Poor Health'],
    q=3,
    labels=['Good Health', 'Fair Health', 'Poor Health']
)

# Get numeric features
numeric_features = final_df.select_dtypes(include=['float64', 'int64']).columns.tolist()
if '% Fair or Poor Health' in numeric_features:
    numeric_features.remove('% Fair or Poor Health')

X_health = final_df[numeric_features]
y_health = final_df['Health_Class']

# Split data
X_train_health, X_test_health, y_train_health, y_test_health = train_test_split(
    X_health, y_health, test_size=0.2, random_state=42, stratify=y_health
)

# Scale features
scaler_health = StandardScaler()
X_train_health_scaled = scaler_health.fit_transform(X_train_health)
X_test_health_scaled = scaler_health.transform(X_test_health)

# Train Random Forest for 3-class
print("\n1. Training Random Forest (3-class)...")
rf_health = RandomForestClassifier(n_estimators=200, max_depth=20, random_state=42)
rf_health.fit(X_train_health_scaled, y_train_health)
joblib.dump(rf_health, MODELS_DIR / 'random_forest_3class.pkl')
print(f"   Saved to: {MODELS_DIR / 'random_forest_3class.pkl'}")

# Train SVM for 3-class
print("2. Training SVM (3-class)...")
svm_health = SVC(kernel='rbf', C=10, gamma='scale', random_state=42, probability=True)
svm_health.fit(X_train_health_scaled, y_train_health)
joblib.dump(svm_health, MODELS_DIR / 'svm_3class.pkl')
print(f"   Saved to: {MODELS_DIR / 'svm_3class.pkl'}")

# Save 3-class scaler and feature names
joblib.dump(scaler_health, MODELS_DIR / 'scaler_3class.pkl')
joblib.dump(numeric_features, MODELS_DIR / 'feature_names_3class.pkl')
print(f"3. Saved 3-class scaler to: {MODELS_DIR / 'scaler_3class.pkl'}")
print(f"4. Saved feature names to: {MODELS_DIR / 'feature_names_3class.pkl'}")

print("\n" + "="*80)
print("TRAINING CLUSTERING MODELS")
print("="*80)

# Prepare clustering data
CLUSTERING_FEATURES = [
    '% Adults with Obesity',
    '% Adults with Diabetes',
    'Food Environment Index',
    'Income Ratio',
    '% Children in Poverty'
]

X_cluster = df[CLUSTERING_FEATURES].copy()

# Drop rows with NaN values
X_cluster = X_cluster.dropna()

print(f"Number of valid samples for clustering: {len(X_cluster)}")

# Scale features
scaler_cluster = StandardScaler()
X_cluster_scaled = scaler_cluster.fit_transform(X_cluster)

# Train K-Means (k=5)
print("\n1. Training K-Means (k=5)...")
kmeans = KMeans(n_clusters=5, random_state=42)
kmeans.fit(X_cluster_scaled)
joblib.dump(kmeans, MODELS_DIR / 'kmeans_5clusters.pkl')
print(f"   Saved to: {MODELS_DIR / 'kmeans_5clusters.pkl'}")

# Save clustering scaler
joblib.dump(scaler_cluster, MODELS_DIR / 'scaler_clustering.pkl')
print(f"2. Saved clustering scaler to: {MODELS_DIR / 'scaler_clustering.pkl'}")

print("\n" + "="*80)
print("MODEL TRAINING COMPLETE")
print("="*80)
print(f"\nAll models have been saved to: {MODELS_DIR}")
print("\nSummary of saved models:")
print("  Regression Models:")
print("    - linear_regression_obesity.pkl")
print("    - ridge_regression_obesity.pkl")
print("    - linear_regression_diabetes.pkl")
print("    - ridge_regression_diabetes.pkl")
print("    - scaler_regression.pkl")
print("\n  Binary Classification Models:")
print("    - logistic_regression.pkl")
print("    - svm_binary.pkl")
print("    - random_forest_binary.pkl")
print("    - extra_trees_binary.pkl")
print("    - scaler_classification.pkl")
print("\n  3-Class Health Prediction Models:")
print("    - random_forest_3class.pkl")
print("    - svm_3class.pkl")
print("    - scaler_3class.pkl")
print("    - feature_names_3class.pkl")
print("\n  Clustering Models:")
print("    - kmeans_5clusters.pkl")
print("    - scaler_clustering.pkl")
print("="*80)
