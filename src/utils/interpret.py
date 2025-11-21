"""
Interpretation Utilities
Model interpretability and explanation functions
"""

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from typing import List, Dict, Optional
import streamlit as st


def get_linear_coefficients(
    model,
    feature_names: List[str],
    scaled: bool = True
) -> pd.DataFrame:
    """
    Extract and format linear model coefficients
    
    Args:
        model: Fitted linear model
        feature_names: List of feature names
        scaled: Whether features were scaled
        
    Returns:
        DataFrame with coefficients
    """
    if hasattr(model, 'coef_'):
        coef = model.coef_
        
        # Handle multi-class logistic regression
        if len(coef.shape) > 1:
            # Multi-class: create separate columns for each class
            coef_df = pd.DataFrame(
                coef.T,
                columns=[f'Class_{i}' for i in range(coef.shape[0])],
                index=feature_names
            )
        else:
            coef_df = pd.DataFrame({
                'Feature': feature_names,
                'Coefficient': coef,
                'Abs_Coefficient': np.abs(coef)
            }).sort_values('Abs_Coefficient', ascending=False)
        
        return coef_df
    else:
        st.error("Model does not have coefficients attribute.")
        return None


def get_logistic_odds_ratios(
    model,
    feature_names: List[str]
) -> pd.DataFrame:
    """
    Calculate odds ratios for logistic regression
    
    Args:
        model: Fitted logistic regression model
        feature_names: List of feature names
        
    Returns:
        DataFrame with odds ratios
    """
    if hasattr(model, 'coef_'):
        coef = model.coef_
        
        # For binary classification
        if len(coef.shape) == 1 or coef.shape[0] == 1:
            if len(coef.shape) > 1:
                coef = coef[0]
            
            odds_ratios = np.exp(coef)
            
            df = pd.DataFrame({
                'Feature': feature_names,
                'Coefficient': coef,
                'Odds_Ratio': odds_ratios,
                'Percent_Change': (odds_ratios - 1) * 100
            }).sort_values('Coefficient', key=abs, ascending=False)
            
            return df
        else:
            st.warning("Odds ratios calculated for binary classification only.")
            return None
    else:
        return None


def get_tree_feature_importance(
    model,
    feature_names: List[str]
) -> pd.DataFrame:
    """
    Extract feature importance from tree-based models
    
    Args:
        model: Fitted tree-based model
        feature_names: List of feature names
        
    Returns:
        DataFrame with feature importances
    """
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        
        df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances,
            'Importance_Pct': importances * 100
        }).sort_values('Importance', ascending=False)
        
        return df
    else:
        st.error("Model does not have feature_importances_ attribute.")
        return None


def compute_permutation_importance(
    model,
    X,
    y,
    feature_names: List[str],
    n_repeats: int = 10,
    random_state: int = 42,
    scoring: Optional[str] = None
) -> pd.DataFrame:
    """
    Calculate permutation importance for any model
    
    Args:
        model: Fitted model
        X: Feature matrix
        y: Target vector
        feature_names: List of feature names
        n_repeats: Number of permutations
        random_state: Random seed
        scoring: Scoring metric
        
    Returns:
        DataFrame with permutation importances
    """
    result = permutation_importance(
        model, X, y,
        n_repeats=n_repeats,
        random_state=random_state,
        scoring=scoring,
        n_jobs=-1
    )
    
    df = pd.DataFrame({
        'Feature': feature_names,
        'Importance_Mean': result.importances_mean,
        'Importance_Std': result.importances_std
    }).sort_values('Importance_Mean', ascending=False)
    
    return df


def get_pca_loadings(
    pca_model,
    feature_names: List[str],
    n_components: int = None
) -> pd.DataFrame:
    """
    Extract PCA component loadings
    
    Args:
        pca_model: Fitted PCA model
        feature_names: Original feature names
        n_components: Number of components to show
        
    Returns:
        DataFrame with loadings
    """
    if n_components is None:
        n_components = pca_model.n_components_
    
    loadings = pca_model.components_[:n_components]
    
    # Create DataFrame
    columns = [f'PC{i+1}' for i in range(n_components)]
    df = pd.DataFrame(
        loadings.T,
        columns=columns,
        index=feature_names
    )
    
    return df


def get_svm_support_vectors_info(model) -> Dict:
    """
    Extract information about SVM support vectors
    
    Args:
        model: Fitted SVM model
        
    Returns:
        Dictionary with support vector info
    """
    if hasattr(model, 'support_vectors_'):
        info = {
            'n_support_vectors': len(model.support_vectors_),
            'n_support_per_class': model.n_support_ if hasattr(model, 'n_support_') else None,
            'support_vector_pct': (len(model.support_vectors_) / model.support_vectors_.shape[0]) * 100
        }
        return info
    else:
        return None


def interpret_decision_tree_rules(
    model,
    feature_names: List[str],
    max_depth: int = 3
) -> List[str]:
    """
    Extract human-readable rules from decision tree
    
    Args:
        model: Fitted decision tree
        feature_names: List of feature names
        max_depth: Maximum depth to show
        
    Returns:
        List of rule strings
    """
    from sklearn.tree import export_text
    
    rules = export_text(
        model,
        feature_names=feature_names,
        max_depth=max_depth
    )
    
    return rules.split('\n')


def get_naive_bayes_priors(model) -> Dict:
    """
    Extract class priors from Naive Bayes model
    
    Args:
        model: Fitted Naive Bayes model
        
    Returns:
        Dictionary with prior probabilities
    """
    if hasattr(model, 'class_prior_'):
        priors = model.class_prior_
        classes = model.classes_
        
        return {
            'classes': classes.tolist(),
            'priors': priors.tolist(),
            'prior_dict': dict(zip(classes, priors))
        }
    else:
        return None


def get_kmeans_cluster_profiles(
    X: np.ndarray,
    labels: np.ndarray,
    feature_names: List[str]
) -> pd.DataFrame:
    """
    Create cluster profile summaries
    
    Args:
        X: Feature matrix
        labels: Cluster labels
        feature_names: Feature names
        
    Returns:
        DataFrame with cluster statistics
    """
    df = pd.DataFrame(X, columns=feature_names)
    df['Cluster'] = labels
    
    # Calculate means per cluster
    cluster_profiles = df.groupby('Cluster').mean()
    
    # Add cluster sizes
    cluster_profiles['Size'] = df.groupby('Cluster').size()
    
    # Add percentage
    cluster_profiles['Percentage'] = (cluster_profiles['Size'] / len(df)) * 100
    
    return cluster_profiles


def explain_bias_variance(
    train_score: float,
    val_score: float,
    test_score: float,
    task_type: str = 'classification'
) -> str:
    """
    Provide bias-variance interpretation based on scores
    
    Args:
        train_score: Training set score
        val_score: Validation set score
        test_score: Test set score
        task_type: 'classification' or 'regression'
        
    Returns:
        Interpretation string
    """
    gap = train_score - val_score
    
    # Determine thresholds based on task
    if task_type == 'classification':
        high_threshold = 0.8
        gap_threshold = 0.1
    else:  # regression
        high_threshold = 0.7
        gap_threshold = 0.15
    
    interpretation = []
    
    # Check for underfitting
    if train_score < high_threshold and val_score < high_threshold:
        interpretation.append("⚠️ **Underfitting (High Bias)**: Both training and validation scores are low.")
        interpretation.append("Try: Increase model complexity, add features, reduce regularization.")
    
    # Check for overfitting
    elif gap > gap_threshold:
        interpretation.append("⚠️ **Overfitting (High Variance)**: Large gap between training and validation scores.")
        interpretation.append("Try: Reduce model complexity, add regularization, get more data, use cross-validation.")
    
    # Good fit
    elif abs(val_score - test_score) < 0.05:
        interpretation.append("✅ **Good Fit**: Training, validation, and test scores are similar.")
        interpretation.append("Model generalizes well to unseen data.")
    
    else:
        interpretation.append("ℹ️ **Monitor Performance**: Scores suggest reasonable fit, but keep monitoring.")
    
    return '\n\n'.join(interpretation)


def format_interpretation_text(
    model_name: str,
    metrics: Dict[str, float],
    task_type: str
) -> str:
    """
    Generate comprehensive interpretation text
    
    Args:
        model_name: Name of the model
        metrics: Dictionary of metrics
        task_type: 'classification' or 'regression'
        
    Returns:
        Formatted interpretation string
    """
    text = [f"## {model_name} Interpretation\n"]
    
    if task_type == 'classification':
        if 'accuracy' in metrics:
            acc = metrics['accuracy']
            text.append(f"**Accuracy**: {acc:.3f} - {acc*100:.1f}% of predictions are correct.")
        
        if 'f1' in metrics:
            f1 = metrics['f1']
            if f1 > 0.8:
                text.append(f"**F1 Score**: {f1:.3f} - Excellent balance between precision and recall.")
            elif f1 > 0.6:
                text.append(f"**F1 Score**: {f1:.3f} - Good balance, but room for improvement.")
            else:
                text.append(f"**F1 Score**: {f1:.3f} - Consider addressing class imbalance or improving model.")
        
        if 'roc_auc' in metrics:
            auc = metrics['roc_auc']
            if auc > 0.9:
                text.append(f"**ROC-AUC**: {auc:.3f} - Excellent discrimination between classes.")
            elif auc > 0.7:
                text.append(f"**ROC-AUC**: {auc:.3f} - Good discrimination ability.")
            else:
                text.append(f"**ROC-AUC**: {auc:.3f} - Moderate discrimination; consider feature engineering.")
    
    else:  # regression
        if 'r2' in metrics:
            r2 = metrics['r2']
            text.append(f"**R² Score**: {r2:.3f} - Model explains {r2*100:.1f}% of variance in target.")
        
        if 'rmse' in metrics:
            rmse = metrics['rmse']
            text.append(f"**RMSE**: {rmse:.2f} - Average prediction error magnitude.")
        
        if 'mape' in metrics and not np.isnan(metrics['mape']):
            mape = metrics['mape']
            text.append(f"**MAPE**: {mape:.2f}% - Average percentage error.")
    
    return '\n\n'.join(text)