"""
Models Utilities
Factory functions to create configured estimators
"""

from sklearn.linear_model import (
    LinearRegression, Ridge, Lasso, ElasticNet,
    LogisticRegression
)
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from typing import Dict, Any


def get_linear_regression(
    model_type: str = 'ols',
    alpha: float = 1.0,
    l1_ratio: float = 0.5,
    **kwargs
):
    """
    Get linear regression model
    
    Args:
        model_type: 'ols', 'ridge', 'lasso', or 'elasticnet'
        alpha: Regularization strength
        l1_ratio: ElasticNet mixing parameter
        
    Returns:
        Regression estimator
    """
    if model_type == 'ols':
        return LinearRegression(**kwargs)
    elif model_type == 'ridge':
        return Ridge(alpha=alpha, **kwargs)
    elif model_type == 'lasso':
        return Lasso(alpha=alpha, max_iter=10000, **kwargs)
    elif model_type == 'elasticnet':
        return ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=10000, **kwargs)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")


def get_logistic_regression(
    penalty: str = 'l2',
    C: float = 1.0,
    solver: str = 'lbfgs',
    class_weight: Any = None,
    random_state: int = 42,
    **kwargs
):
    """
    Get logistic regression model
    
    Args:
        penalty: 'l1', 'l2', 'elasticnet', or 'none'
        C: Inverse regularization strength
        solver: Optimization algorithm
        class_weight: Class weights for imbalanced data
        random_state: Random seed
        
    Returns:
        Logistic regression classifier
    """
    if penalty == 'l1':
        solver = 'liblinear'  # l1 requires liblinear or saga
    elif penalty == 'elasticnet':
        solver = 'saga'  # elasticnet requires saga
    
    return LogisticRegression(
        penalty=penalty if penalty != 'none' else None,
        C=C,
        solver=solver,
        class_weight=class_weight,
        random_state=random_state,
        max_iter=10000,
        **kwargs
    )


def get_svm(
    task: str = 'classification',
    kernel: str = 'rbf',
    C: float = 1.0,
    gamma: str = 'scale',
    probability: bool = True,
    random_state: int = 42,
    **kwargs
):
    """
    Get SVM model
    
    Args:
        task: 'classification' or 'regression'
        kernel: 'linear', 'rbf', 'poly', or 'sigmoid'
        C: Regularization parameter
        gamma: Kernel coefficient
        probability: Enable probability estimates (classification only)
        random_state: Random seed
        
    Returns:
        SVM estimator
    """
    if task == 'classification':
        return SVC(
            kernel=kernel,
            C=C,
            gamma=gamma,
            probability=probability,
            random_state=random_state,
            **kwargs
        )
    elif task == 'regression':
        return SVR(
            kernel=kernel,
            C=C,
            gamma=gamma,
            **kwargs
        )
    else:
        raise ValueError(f"Unknown task: {task}")


def get_decision_tree(
    task: str = 'classification',
    criterion: str = 'gini',
    max_depth: int = None,
    min_samples_split: int = 2,
    min_samples_leaf: int = 1,
    random_state: int = 42,
    **kwargs
):
    """
    Get Decision Tree model
    
    Args:
        task: 'classification' or 'regression'
        criterion: Split criterion
        max_depth: Maximum tree depth
        min_samples_split: Minimum samples to split node
        min_samples_leaf: Minimum samples in leaf
        random_state: Random seed
        
    Returns:
        Decision tree estimator
    """
    if task == 'classification':
        return DecisionTreeClassifier(
            criterion=criterion,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
            **kwargs
        )
    elif task == 'regression':
        # Regression uses different criteria
        if criterion == 'gini':
            criterion = 'squared_error'
        elif criterion == 'entropy':
            criterion = 'squared_error'
        
        return DecisionTreeRegressor(
            criterion=criterion,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
            **kwargs
        )
    else:
        raise ValueError(f"Unknown task: {task}")


def get_random_forest(
    task: str = 'classification',
    n_estimators: int = 100,
    max_depth: int = None,
    max_features: str = 'sqrt',
    min_samples_leaf: int = 1,
    class_weight: Any = None,
    random_state: int = 42,
    **kwargs
):
    """
    Get Random Forest model
    
    Args:
        task: 'classification' or 'regression'
        n_estimators: Number of trees
        max_depth: Maximum tree depth
        max_features: Number of features for best split
        min_samples_leaf: Minimum samples in leaf
        class_weight: Class weights (classification only)
        random_state: Random seed
        
    Returns:
        Random forest estimator
    """
    if task == 'classification':
        return RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            max_features=max_features,
            min_samples_leaf=min_samples_leaf,
            class_weight=class_weight,
            random_state=random_state,
            n_jobs=-1,
            **kwargs
        )
    elif task == 'regression':
        return RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            max_features=max_features,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
            n_jobs=-1,
            **kwargs
        )
    else:
        raise ValueError(f"Unknown task: {task}")


def get_knn(
    task: str = 'classification',
    n_neighbors: int = 5,
    weights: str = 'uniform',
    metric: str = 'minkowski',
    p: int = 2,
    **kwargs
):
    """
    Get KNN model
    
    Args:
        task: 'classification' or 'regression'
        n_neighbors: Number of neighbors
        weights: 'uniform' or 'distance'
        metric: Distance metric
        p: Power parameter for Minkowski metric
        
    Returns:
        KNN estimator
    """
    if task == 'classification':
        return KNeighborsClassifier(
            n_neighbors=n_neighbors,
            weights=weights,
            metric=metric,
            p=p,
            **kwargs
        )
    elif task == 'regression':
        return KNeighborsRegressor(
            n_neighbors=n_neighbors,
            weights=weights,
            metric=metric,
            p=p,
            **kwargs
        )
    else:
        raise ValueError(f"Unknown task: {task}")


def get_naive_bayes(
    nb_type: str = 'gaussian',
    var_smoothing: float = 1e-9,
    alpha: float = 1.0,
    **kwargs
):
    """
    Get Naive Bayes model
    
    Args:
        nb_type: 'gaussian' or 'multinomial'
        var_smoothing: Gaussian smoothing parameter
        alpha: Multinomial smoothing parameter
        
    Returns:
        Naive Bayes classifier
    """
    if nb_type == 'gaussian':
        return GaussianNB(var_smoothing=var_smoothing, **kwargs)
    elif nb_type == 'multinomial':
        return MultinomialNB(alpha=alpha, **kwargs)
    else:
        raise ValueError(f"Unknown nb_type: {nb_type}")


def get_kmeans(
    n_clusters: int = 3,
    init: str = 'k-means++',
    n_init: int = 10,
    max_iter: int = 300,
    random_state: int = 42,
    **kwargs
):
    """
    Get K-Means clustering model
    
    Args:
        n_clusters: Number of clusters
        init: Initialization method
        n_init: Number of initializations
        max_iter: Maximum iterations
        random_state: Random seed
        
    Returns:
        K-Means estimator
    """
    return KMeans(
        n_clusters=n_clusters,
        init=init,
        n_init=n_init,
        max_iter=max_iter,
        random_state=random_state,
        **kwargs
    )


def get_pca(
    n_components: Any = None,
    random_state: int = 42,
    **kwargs
):
    """
    Get PCA model
    
    Args:
        n_components: Number of components or variance threshold
        random_state: Random seed
        
    Returns:
        PCA estimator
    """
    return PCA(
        n_components=n_components,
        random_state=random_state,
        **kwargs
    )