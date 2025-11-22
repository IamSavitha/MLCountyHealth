# Trained Models Directory

This directory contains all trained machine learning models for the County Health Analysis project.

## Models Overview

### Regression Models (Obesity & Diabetes Prediction)
- **linear_regression_obesity.pkl** - Linear regression model for predicting county-level obesity rates
- **ridge_regression_obesity.pkl** - Ridge regression model for obesity prediction (L2 regularization, α=1.0)
- **linear_regression_diabetes.pkl** - Linear regression model for predicting diabetes rates
- **ridge_regression_diabetes.pkl** - Ridge regression model for diabetes prediction (L2 regularization, α=1.0)
- **scaler_regression.pkl** - StandardScaler for regression features

**Features used (7 features):**
- Food_Access_Barrier_Index
- Socioeconomic_Vulnerability_Index
- % Completed High School
- Income Ratio
- % Uninsured
- % Rural
- Primary Care Physicians Ratio

### Binary Classification Models (Income Inequality)
- **logistic_regression.pkl** - Logistic regression classifier (max_iter=1000)
- **svm_binary.pkl** - SVM classifier with RBF kernel
- **random_forest_binary.pkl** - Random Forest classifier (100 estimators, max_depth=5)
- **extra_trees_binary.pkl** - Extra Trees classifier (100 estimators, max_depth=5)
- **scaler_classification.pkl** - StandardScaler for classification features

**Features used (8 features):**
- Same as regression features plus % Excessive Drinking

**Target:** High_Income_Inequality (binary: high/low based on median income ratio of 4.42)

### 3-Class Health Prediction Models
- **random_forest_3class.pkl** - Random Forest classifier (200 estimators, max_depth=20)
  - Accuracy: 78.6%
- **svm_3class.pkl** - SVM classifier (RBF kernel, C=10, gamma='scale')
  - Accuracy: 77.5%
- **scaler_3class.pkl** - StandardScaler for 49 health features
- **feature_names_3class.pkl** - List of 49 feature names used in 3-class prediction

**Features:** All 49 numeric features from raw County Health Rankings data
**Target:** Health_Class (Good Health / Fair Health / Poor Health)
**Sample size:** 2,314 counties after preprocessing

### Clustering Models
- **kmeans_5clusters.pkl** - K-Means clustering model (k=5, silhouette score=0.44)
- **scaler_clustering.pkl** - StandardScaler for clustering features

**Features used (5 features):**
- % Adults with Obesity
- % Adults with Diabetes
- Food Environment Index
- Income Ratio
- % Children in Poverty

**Cluster profiles:**
- Cluster 0: Healthy Affluent (594 counties)
- Cluster 1: Best Outcomes (321 counties)
- Cluster 2: Moderate Risk (565 counties)
- Cluster 3: Rural Challenged (440 counties)
- Cluster 4: Highest Risk (316 counties)

## Training Details

- **Training script:** `src/train_and_save_models.py`
- **Train-test split:** 80% / 20% with stratification
- **Random state:** 42 (for reproducibility)
- **Cross-validation:** 5-fold stratified CV
- **Sample sizes:**
  - Regression/Binary classification: 2,132 counties (after NaN removal)
  - 3-class prediction: 2,314 counties
  - Clustering: 2,236 counties

## Usage

### Loading Models

```python
import joblib

# Load regression model
ridge_model = joblib.load('models/ridge_regression_obesity.pkl')
scaler = joblib.load('models/scaler_regression.pkl')

# Load classification model
rf_classifier = joblib.load('models/random_forest_binary.pkl')

# Load 3-class health prediction model
rf_3class = joblib.load('models/random_forest_3class.pkl')
scaler_3class = joblib.load('models/scaler_3class.pkl')
feature_names = joblib.load('models/feature_names_3class.pkl')

# Load clustering model
kmeans = joblib.load('models/kmeans_5clusters.pkl')
```

### Making Predictions

```python
import pandas as pd
import numpy as np

# Example: Predict obesity rate
X_new = pd.DataFrame({
    'Food_Access_Barrier_Index': [0.5],
    'Socioeconomic_Vulnerability_Index': [0.6],
    '% Completed High School': [85.0],
    'Income Ratio': [4.5],
    '% Uninsured': [12.0],
    '% Rural': [60.0],
    'Primary Care Physicians Ratio': [2500.0]
})

X_scaled = scaler.transform(X_new)
predicted_obesity = ridge_model.predict(X_scaled)
print(f"Predicted obesity rate: {predicted_obesity[0]:.2f}%")
```

## Retraining Models

To retrain all models with updated data:

```bash
python src/train_and_save_models.py
```

This will:
1. Load data from `data/processed/cleaned_health_data.csv` and `data/raw/2025CountyHealthRankingsDatav3.xlsx`
2. Preprocess and clean the data
3. Train all models
4. Save updated models to this directory (overwrites existing files)

## Model Performance

### Regression (Test Set)
| Model | Target | R² | RMSE | MAE |
|-------|--------|-----|------|-----|
| Ridge | Obesity | 0.417 | 2.78% | 2.17% |
| Ridge | Diabetes | 0.391 | 1.48% | 1.15% |

### Binary Classification (Test Set)
| Model | Accuracy | Precision | Recall | F1 |
|-------|----------|-----------|--------|-----|
| Random Forest | 0.831 | 0.835 | 0.862 | 0.848 |
| Extra Trees | 0.826 | 0.831 | 0.855 | 0.843 |
| Logistic Reg. | 0.799 | 0.803 | 0.827 | 0.815 |
| SVM (RBF) | 0.793 | 0.798 | 0.820 | 0.809 |

### 3-Class Health Prediction (Test Set)
| Model | Accuracy | Precision | Recall | F1 (Macro) |
|-------|----------|-----------|--------|------------|
| Random Forest | 0.786 | 0.789 | 0.785 | 0.786 |
| SVM (RBF) | 0.775 | 0.778 | 0.774 | 0.775 |

### Clustering
- **Silhouette Score:** 0.44
- **Number of clusters:** 5
- **Optimal k determined by:** Elbow method + silhouette analysis

## Notes

- All models use `random_state=42` for reproducibility
- Scalers must be applied to features before prediction
- Models assume input features are in the same order as training data
- 3-class models require exactly 49 features (see feature_names_3class.pkl for order)
- NaN values are not supported - data must be cleaned before prediction
