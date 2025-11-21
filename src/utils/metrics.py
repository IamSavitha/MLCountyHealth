"""
Metrics Utilities
Comprehensive evaluation metrics for classification and regression
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    # Classification
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, log_loss,
    confusion_matrix, classification_report,
    roc_curve, precision_recall_curve,
    
    # Regression
    r2_score, mean_absolute_error, mean_squared_error,
    mean_absolute_percentage_error
)
from typing import Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_pred_proba: Optional[np.ndarray] = None,
    average: str = 'weighted'
) -> Dict[str, float]:
    """
    Compute comprehensive classification metrics
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_pred_proba: Predicted probabilities (optional, for ROC-AUC)
        average: Averaging strategy for multi-class ('micro', 'macro', 'weighted')
        
    Returns:
        Dictionary of metrics
    """
    metrics = {}
    
    # Basic metrics
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['precision'] = precision_score(y_true, y_pred, average=average, zero_division=0)
    metrics['recall'] = recall_score(y_true, y_pred, average=average, zero_division=0)
    metrics['f1'] = f1_score(y_true, y_pred, average=average, zero_division=0)
    
    # Probability-based metrics
    if y_pred_proba is not None:
        try:
            # Handle binary vs multi-class
            n_classes = len(np.unique(y_true))
            
            if n_classes == 2:
                # Binary classification
                metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba[:, 1])
                metrics['pr_auc'] = average_precision_score(y_true, y_pred_proba[:, 1])
                metrics['log_loss'] = log_loss(y_true, y_pred_proba)
            else:
                # Multi-class classification
                metrics['roc_auc'] = roc_auc_score(
                    y_true, y_pred_proba,
                    multi_class='ovr', average=average
                )
                metrics['log_loss'] = log_loss(y_true, y_pred_proba)
        except Exception as e:
            pass
    
    return metrics


def compute_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> Dict[str, float]:
    """
    Compute comprehensive regression metrics
    
    Args:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        Dictionary of metrics
    """
    metrics = {}
    
    metrics['r2'] = r2_score(y_true, y_pred)
    metrics['mae'] = mean_absolute_error(y_true, y_pred)
    metrics['rmse'] = np.sqrt(mean_squared_error(y_true, y_pred))
    metrics['mse'] = mean_squared_error(y_true, y_pred)
    
    # MAPE (handle zero values)
    try:
        metrics['mape'] = mean_absolute_percentage_error(y_true, y_pred) * 100
    except:
        # Manual calculation avoiding division by zero
        mask = y_true != 0
        if mask.sum() > 0:
            metrics['mape'] = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
        else:
            metrics['mape'] = np.nan
    
    return metrics


def compute_metrics_by_split(
    y_true_train: np.ndarray,
    y_pred_train: np.ndarray,
    y_true_val: np.ndarray,
    y_pred_val: np.ndarray,
    y_true_test: np.ndarray,
    y_pred_test: np.ndarray,
    task_type: str,
    y_pred_proba_train: Optional[np.ndarray] = None,
    y_pred_proba_val: Optional[np.ndarray] = None,
    y_pred_proba_test: Optional[np.ndarray] = None
) -> pd.DataFrame:
    """
    Compute metrics across train/val/test splits
    
    Args:
        y_true_*: True labels for each split
        y_pred_*: Predictions for each split
        task_type: 'classification' or 'regression'
        y_pred_proba_*: Predicted probabilities (for classification)
        
    Returns:
        DataFrame with metrics for each split
    """
    if task_type == 'classification':
        train_metrics = compute_classification_metrics(
            y_true_train, y_pred_train, y_pred_proba_train
        )
        val_metrics = compute_classification_metrics(
            y_true_val, y_pred_val, y_pred_proba_val
        )
        test_metrics = compute_classification_metrics(
            y_true_test, y_pred_test, y_pred_proba_test
        )
    else:
        train_metrics = compute_regression_metrics(y_true_train, y_pred_train)
        val_metrics = compute_regression_metrics(y_true_val, y_pred_val)
        test_metrics = compute_regression_metrics(y_true_test, y_pred_test)
    
    # Create DataFrame
    df_metrics = pd.DataFrame({
        'Train': train_metrics,
        'Validation': val_metrics,
        'Test': test_metrics
    }).T
    
    return df_metrics


def get_confusion_matrix_data(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: Optional[list] = None
) -> Tuple[np.ndarray, list]:
    """
    Get confusion matrix and labels
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        labels: Optional label names
        
    Returns:
        Confusion matrix and label names
    """
    cm = confusion_matrix(y_true, y_pred)
    
    if labels is None:
        labels = [f"Class {i}" for i in range(cm.shape[0])]
    
    return cm, labels


def get_classification_report_df(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_names: Optional[list] = None
) -> pd.DataFrame:
    """
    Get classification report as DataFrame
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        target_names: Optional class names
        
    Returns:
        Classification report as DataFrame
    """
    report_dict = classification_report(
        y_true, y_pred,
        target_names=target_names,
        output_dict=True,
        zero_division=0
    )
    
    df_report = pd.DataFrame(report_dict).T
    return df_report


def get_roc_curve_data(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    pos_label: int = 1
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Get ROC curve data for binary classification
    
    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities for positive class
        pos_label: Positive class label
        
    Returns:
        FPR, TPR, thresholds
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba, pos_label=pos_label)
    return fpr, tpr, thresholds


def get_pr_curve_data(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    pos_label: int = 1
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Get Precision-Recall curve data
    
    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities for positive class
        pos_label: Positive class label
        
    Returns:
        Precision, Recall, thresholds
    """
    precision, recall, thresholds = precision_recall_curve(
        y_true, y_pred_proba, pos_label=pos_label
    )
    return precision, recall, thresholds


def compute_residuals(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> Dict[str, np.ndarray]:
    """
    Compute various residual metrics for regression
    
    Args:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        Dictionary with residual metrics
    """
    residuals = y_true - y_pred
    
    return {
        'residuals': residuals,
        'abs_residuals': np.abs(residuals),
        'squared_residuals': residuals ** 2,
        'standardized_residuals': residuals / np.std(residuals),
        'percent_error': (residuals / y_true) * 100
    }


def compute_cross_val_metrics(
    cv_results: Dict,
    task_type: str
) -> pd.DataFrame:
    """
    Summarize cross-validation results
    
    Args:
        cv_results: Dict from cross_validate
        task_type: 'classification' or 'regression'
        
    Returns:
        DataFrame with CV metrics
    """
    metrics_df = pd.DataFrame()
    
    for key, values in cv_results.items():
        if key.startswith('test_'):
            metric_name = key.replace('test_', '')
            metrics_df[metric_name] = values
    
    # Add summary statistics
    summary = pd.DataFrame({
        'mean': metrics_df.mean(),
        'std': metrics_df.std(),
        'min': metrics_df.min(),
        'max': metrics_df.max()
    })
    
    return summary


def format_metric_display(
    metric_name: str,
    value: float
) -> str:
    """
    Format metric for display
    
    Args:
        metric_name: Name of metric
        value: Metric value
        
    Returns:
        Formatted string
    """
    if metric_name in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'pr_auc', 'r2']:
        return f"{value:.4f}"
    elif metric_name in ['mae', 'rmse', 'mse']:
        return f"{value:.2f}"
    elif metric_name in ['mape']:
        return f"{value:.2f}%"
    elif metric_name in ['log_loss']:
        return f"{value:.4f}"
    else:
        return f"{value:.4f}"


def get_metric_interpretation(
    metric_name: str,
    value: float
) -> str:
    """
    Provide interpretation guidance for metrics
    
    Args:
        metric_name: Name of metric
        value: Metric value
        
    Returns:
        Interpretation string
    """
    interpretations = {
        'accuracy': "Proportion of correct predictions",
        'precision': "Of predicted positives, how many were actually positive",
        'recall': "Of actual positives, how many were predicted correctly",
        'f1': "Harmonic mean of precision and recall",
        'roc_auc': "Area under ROC curve (1.0 = perfect, 0.5 = random)",
        'pr_auc': "Area under Precision-Recall curve",
        'log_loss': "Lower is better. Measures probability estimate quality",
        'r2': "Proportion of variance explained (1.0 = perfect, 0 = baseline)",
        'mae': "Average absolute error between predictions and truth",
        'rmse': "Root mean squared error (penalizes large errors)",
        'mse': "Mean squared error",
        'mape': "Mean absolute percentage error"
    }
    
    return interpretations.get(metric_name, "")


def compare_models_metrics(
    models_dict: Dict[str, Dict[str, float]]
) -> pd.DataFrame:
    """
    Create comparison DataFrame for multiple models
    
    Args:
        models_dict: Dictionary of {model_name: {metric: value}}
        
    Returns:
        Comparison DataFrame
    """
    df = pd.DataFrame(models_dict).T
    return df