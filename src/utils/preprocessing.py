"""
Preprocessing Utilities
Handles data type inference, transformation pipelines, and preprocessing
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from typing import Tuple, List, Dict, Optional
import streamlit as st


def infer_column_types(df: pd.DataFrame) -> Dict[str, List[str]]:
    """
    Automatically infer numeric vs categorical columns
    
    Args:
        df: Input dataframe
        
    Returns:
        Dictionary with 'numeric' and 'categorical' column lists
    """
    numeric_cols = []
    categorical_cols = []
    
    for col in df.columns:
        if df[col].dtype in ['int64', 'float64']:
            # Check if it's actually categorical (few unique values)
            n_unique = df[col].nunique()
            if n_unique <= 10 and df[col].dtype == 'int64':
                categorical_cols.append(col)
            else:
                numeric_cols.append(col)
        else:
            categorical_cols.append(col)
    
    return {'numeric': numeric_cols, 'categorical': categorical_cols}


def detect_task_type(target: pd.Series) -> str:
    """
    Detect if problem is classification or regression based on target
    
    Args:
        target: Target variable series
        
    Returns:
        'classification' or 'regression'
    """
    # Check if numeric
    if target.dtype in ['int64', 'float64']:
        n_unique = target.nunique()
        
        # Binary classification (0/1)
        if n_unique == 2 and set(target.dropna().unique()).issubset({0, 1}):
            return 'classification'
        
        # Multi-class classification (few unique values)
        elif n_unique <= 20:
            return 'classification'
        
        # Regression (many unique values)
        else:
            return 'regression'
    else:
        # Categorical target
        return 'classification'


def get_preprocessing_pipeline(
    numeric_features: List[str],
    categorical_features: List[str],
    imputation: str = 'mean',
    scaling: str = 'standard',
    encoding: str = 'onehot'
) -> ColumnTransformer:
    """
    Create a preprocessing pipeline with specified strategies
    
    Args:
        numeric_features: List of numeric column names
        categorical_features: List of categorical column names
        imputation: Strategy for missing values ('mean', 'median', 'most_frequent')
        scaling: Scaling method ('standard', 'minmax', 'none')
        encoding: Encoding method ('onehot', 'label') - Note: label only for ordinal
        
    Returns:
        ColumnTransformer pipeline
    """
    transformers = []
    
    # Numeric pipeline
    if len(numeric_features) > 0:
        numeric_steps = [
            ('imputer', SimpleImputer(strategy=imputation))
        ]
        
        if scaling == 'standard':
            numeric_steps.append(('scaler', StandardScaler()))
        elif scaling == 'minmax':
            numeric_steps.append(('scaler', MinMaxScaler()))
        # 'none' - no scaling
        
        numeric_pipeline = Pipeline(numeric_steps)
        transformers.append(('numeric', numeric_pipeline, numeric_features))
    
    # Categorical pipeline
    if len(categorical_features) > 0:
        categorical_steps = [
            ('imputer', SimpleImputer(strategy='most_frequent'))
        ]
        
        if encoding == 'onehot':
            categorical_steps.append((
                'encoder',
                OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')
            ))
        
        categorical_pipeline = Pipeline(categorical_steps)
        transformers.append(('categorical', categorical_pipeline, categorical_features))
    
    # Create column transformer
    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder='drop'  # Drop columns not specified
    )
    
    return preprocessor


def check_data_quality(df: pd.DataFrame) -> Dict:
    """
    Perform data quality checks and return diagnostics
    
    Args:
        df: Input dataframe
        
    Returns:
        Dictionary with quality metrics
    """
    quality_report = {
        'total_rows': len(df),
        'total_cols': len(df.columns),
        'missing_values': df.isnull().sum().sum(),
        'missing_pct': (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100,
        'duplicate_rows': df.duplicated().sum(),
        'constant_columns': [],
        'high_cardinality_columns': []
    }
    
    # Check for constant columns
    for col in df.columns:
        if df[col].nunique() <= 1:
            quality_report['constant_columns'].append(col)
    
    # Check for high cardinality categorical columns
    for col in df.select_dtypes(include=['object']).columns:
        if df[col].nunique() > 50:
            quality_report['high_cardinality_columns'].append(col)
    
    return quality_report


def handle_missing_values(df: pd.DataFrame, strategy: str = 'mean') -> pd.DataFrame:
    """
    Handle missing values in dataframe (for display purposes)
    
    Args:
        df: Input dataframe
        strategy: Imputation strategy
        
    Returns:
        Dataframe with missing values handled
    """
    df_cleaned = df.copy()
    
    # Numeric columns
    numeric_cols = df_cleaned.select_dtypes(include=['int64', 'float64']).columns
    for col in numeric_cols:
        if df_cleaned[col].isnull().any():
            if strategy == 'mean':
                df_cleaned[col].fillna(df_cleaned[col].mean(), inplace=True)
            elif strategy == 'median':
                df_cleaned[col].fillna(df_cleaned[col].median(), inplace=True)
            elif strategy == 'most_frequent':
                df_cleaned[col].fillna(df_cleaned[col].mode()[0], inplace=True)
    
    # Categorical columns
    categorical_cols = df_cleaned.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if df_cleaned[col].isnull().any():
            df_cleaned[col].fillna(df_cleaned[col].mode()[0], inplace=True)
    
    return df_cleaned


def get_feature_names_from_transformer(
    preprocessor: ColumnTransformer,
    numeric_features: List[str],
    categorical_features: List[str]
) -> List[str]:
    """
    Extract feature names after transformation
    
    Args:
        preprocessor: Fitted ColumnTransformer
        numeric_features: Original numeric feature names
        categorical_features: Original categorical feature names
        
    Returns:
        List of feature names after transformation
    """
    feature_names = []
    
    # Numeric features (unchanged names)
    feature_names.extend(numeric_features)
    
    # Categorical features (get from OneHotEncoder)
    if len(categorical_features) > 0:
        try:
            # Get the encoder from the pipeline
            cat_pipeline = preprocessor.named_transformers_['categorical']
            if 'encoder' in cat_pipeline.named_steps:
                encoder = cat_pipeline.named_steps['encoder']
                cat_feature_names = encoder.get_feature_names_out(categorical_features)
                feature_names.extend(cat_feature_names)
            else:
                feature_names.extend(categorical_features)
        except:
            feature_names.extend(categorical_features)
    
    return feature_names


def create_synthetic_target(
    df: pd.DataFrame,
    column: str,
    method: str = 'median_split'
) -> pd.Series:
    """
    Create a synthetic binary target for teaching purposes
    
    Args:
        df: Input dataframe
        column: Column to base target on
        method: Method to create target ('median_split', 'quantile_split')
        
    Returns:
        Binary target series
    """
    if method == 'median_split':
        threshold = df[column].median()
        target = (df[column] > threshold).astype(int)
    elif method == 'quantile_split':
        threshold = df[column].quantile(0.75)
        target = (df[column] > threshold).astype(int)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return target


def check_target_leakage(
    feature_columns: List[str],
    target_column: str,
    df: pd.DataFrame
) -> List[str]:
    """
    Check for potential target leakage
    
    Args:
        feature_columns: List of feature column names
        target_column: Target column name
        df: Input dataframe
        
    Returns:
        List of potentially leaky columns
    """
    leaky_columns = []
    
    # Check for perfect correlations
    if target_column in df.select_dtypes(include=['int64', 'float64']).columns:
        for col in feature_columns:
            if col in df.select_dtypes(include=['int64', 'float64']).columns:
                corr = abs(df[col].corr(df[target_column]))
                if corr > 0.95:
                    leaky_columns.append(col)
    
    return leaky_columns


def balance_classes_smote(X: np.ndarray, y: np.ndarray, random_state: int = 42):
    """
    Apply SMOTE for class imbalance (requires imbalanced-learn)
    
    Args:
        X: Feature matrix
        y: Target vector
        random_state: Random seed
        
    Returns:
        Balanced X and y
    """
    try:
        from imblearn.over_sampling import SMOTE
        smote = SMOTE(random_state=random_state)
        X_resampled, y_resampled = smote.fit_resample(X, y)
        return X_resampled, y_resampled
    except ImportError:
        st.warning("imbalanced-learn not installed. Skipping SMOTE.")
        return X, y


@st.cache_data
def load_default_dataset() -> pd.DataFrame:
    """
    Load the default sample dataset
    
    Returns:
        Default dataframe
    """
    try:
        df = pd.read_csv('/mnt/user-data/uploads/cleaned_health_data.csv')
        return df
    except FileNotFoundError:
        st.error("Default dataset not found. Please upload your own data.")
        return None


def get_data_summary(df: pd.DataFrame) -> Dict:
    """
    Generate comprehensive data summary
    
    Args:
        df: Input dataframe
        
    Returns:
        Dictionary with summary statistics
    """
    summary = {
        'n_rows': len(df),
        'n_cols': len(df.columns),
        'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024**2,
        'numeric_cols': len(df.select_dtypes(include=['int64', 'float64']).columns),
        'categorical_cols': len(df.select_dtypes(include=['object']).columns),
        'missing_cells': df.isnull().sum().sum(),
        'missing_pct': (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100,
        'duplicate_rows': df.duplicated().sum()
    }
    
    return summary