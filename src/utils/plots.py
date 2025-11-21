

def plot_confusion_matrix(y_true, y_pred, labels=None, title="Confusion Matrix"):
    """Create interactive confusion matrix heatmap"""
    cm = confusion_matrix(y_true, y_pred)
    
    if labels is None:
        labels = [f"Class {i}" for i in range(cm.shape[0])]
    
    # Normalize for percentages
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Create text annotations
    text = [[f"{cm[i,j]}<br>({cm_normalized[i,j]:.1%})" 
             for j in range(cm.shape[1])] for i in range(cm.shape[0])]
    
    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=labels,
        y=labels,
        text=text,
        texttemplate="%{text}",
        textfont={"size": 12},
        colorscale='Blues',
        showscale=True
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Predicted",
        yaxis_title="Actual",
        width=600,
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)
    st.caption("💡 Diagonal cells show correct predictions. Off-diagonal cells show misclassifications.")


def plot_roc_curve(y_true, y_pred_proba, title="ROC Curve"):
    """Plot ROC curve for binary classification"""
    from sklearn.metrics import roc_auc_score
    
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    auc_score = roc_auc_score(y_true, y_pred_proba)
    
    fig = go.Figure()
    
    # ROC curve
    fig.add_trace(go.Scatter(
        x=fpr, y=tpr,
        mode='lines',
        name=f'ROC (AUC = {auc_score:.3f})',
        line=dict(color='blue', width=2)
    ))
    
    # Random classifier line
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode='lines',
        name='Random Classifier',
        line=dict(color='red', dash='dash')
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
        width=600,
        height=500,
        showlegend=True
    )
    
    st.plotly_chart(fig, use_container_width=True)
    st.caption(f"💡 AUC = {auc_score:.3f}. Higher is better (1.0 = perfect, 0.5 = random).")


def plot_precision_recall_curve(y_true, y_pred_proba, title="Precision-Recall Curve"):
    """Plot PR curve for binary classification"""
    from sklearn.metrics import average_precision_score
    
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
    ap_score = average_precision_score(y_true, y_pred_proba)
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=recall, y=precision,
        mode='lines',
        name=f'PR (AP = {ap_score:.3f})',
        line=dict(color='green', width=2),
        fill='tozeroy'
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title='Recall',
        yaxis_title='Precision',
        width=600,
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)
    st.caption(f"💡 Average Precision = {ap_score:.3f}. Higher is better. Useful for imbalanced datasets.")


def plot_calibration_curve(y_true, y_pred_proba, n_bins=10, title="Calibration Curve"):
    """Plot calibration curve to assess probability estimates"""
    fraction_of_positives, mean_predicted_value = calibration_curve(
        y_true, y_pred_proba, n_bins=n_bins, strategy='uniform'
    )
    
    fig = go.Figure()
    
    # Calibration curve
    fig.add_trace(go.Scatter(
        x=mean_predicted_value,
        y=fraction_of_positives,
        mode='lines+markers',
        name='Model',
        line=dict(color='blue', width=2)
    ))
    
    # Perfect calibration
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode='lines',
        name='Perfect Calibration',
        line=dict(color='red', dash='dash')
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title='Mean Predicted Probability',
        yaxis_title='Fraction of Positives',
        width=600,
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)
    st.caption("💡 Points on diagonal = well-calibrated. Above = overconfident, Below = underconfident.")


def plot_residuals_vs_fitted(y_true, y_pred, title="Residuals vs Fitted"):
    """Plot residuals against fitted values for regression"""
    residuals = y_true - y_pred
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=y_pred,
        y=residuals,
        mode='markers',
        marker=dict(color='blue', size=5, opacity=0.6),
        name='Residuals'
    ))
    
    # Zero line
    fig.add_hline(y=0, line_dash="dash", line_color="red")
    
    fig.update_layout(
        title=title,
        xaxis_title='Fitted Values',
        yaxis_title='Residuals',
        width=700,
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)
    st.caption("💡 Look for random scatter around zero. Patterns suggest model issues.")


def plot_qq_plot(residuals, title="Q-Q Plot"):
    """Plot Q-Q plot to assess normality of residuals"""
    # Standardize residuals
    standardized = (residuals - np.mean(residuals)) / np.std(residuals)
    
    # Theoretical quantiles
    theoretical_quantiles = stats.norm.ppf(np.linspace(0.01, 0.99, len(standardized)))
    sample_quantiles = np.sort(standardized)
    
    fig = go.Figure()
    
    # Q-Q points
    fig.add_trace(go.Scatter(
        x=theoretical_quantiles,
        y=sample_quantiles,
        mode='markers',
        marker=dict(color='blue', size=5),
        name='Sample Quantiles'
    ))
    
    # Reference line
    fig.add_trace(go.Scatter(
        x=theoretical_quantiles,
        y=theoretical_quantiles,
        mode='lines',
        line=dict(color='red', dash='dash'),
        name='Normal Distribution'
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title='Theoretical Quantiles',
        yaxis_title='Sample Quantiles',
        width=600,
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)
    st.caption("💡 Points on line = normally distributed residuals. Deviations suggest non-normality.")


def plot_prediction_vs_actual(y_true, y_pred, title="Prediction vs Actual"):
    """Plot predictions against actual values"""
    fig = go.Figure()
    
    # Scatter plot
    fig.add_trace(go.Scatter(
        x=y_true,
        y=y_pred,
        mode='markers',
        marker=dict(color='blue', size=5, opacity=0.6),
        name='Predictions'
    ))
    
    # Perfect prediction line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    fig.add_trace(go.Scatter(
        x=[min_val, max_val],
        y=[min_val, max_val],
        mode='lines',
        line=dict(color='red', dash='dash'),
        name='Perfect Prediction'
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title='Actual Values',
        yaxis_title='Predicted Values',
        width=700,
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)
    st.caption("💡 Points on diagonal = perfect predictions. Spread indicates error.")


def plot_residuals_histogram(residuals, title="Residuals Distribution"):
    """Plot histogram of residuals"""
    fig = go.Figure()
    
    fig.add_trace(go.Histogram(
        x=residuals,
        nbinsx=30,
        marker_color='skyblue',
        name='Residuals'
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title='Residuals',
        yaxis_title='Frequency',
        width=700,
        height=400,
        showlegend=False
    )
    
    st.plotly_chart(fig, use_container_width=True)
    st.caption("💡 Should be approximately bell-shaped (normal) centered at zero.")


def plot_learning_curve(train_sizes, train_scores, val_scores, title="Learning Curve"):
    """Plot learning curve showing training and validation scores"""
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)
    
    fig = go.Figure()
    
    # Training scores
    fig.add_trace(go.Scatter(
        x=train_sizes, y=train_mean,
        mode='lines+markers',
        name='Training Score',
        line=dict(color='blue', width=2),
        marker=dict(size=8)
    ))
    
    fig.add_trace(go.Scatter(
        x=np.concatenate([train_sizes, train_sizes[::-1]]),
        y=np.concatenate([train_mean + train_std, (train_mean - train_std)[::-1]]),
        fill='toself',
        fillcolor='rgba(0,0,255,0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        showlegend=False,
        name='Train ±1 std'
    ))
    
    # Validation scores
    fig.add_trace(go.Scatter(
        x=train_sizes, y=val_mean,
        mode='lines+markers',
        name='Validation Score',
        line=dict(color='red', width=2),
        marker=dict(size=8)
    ))
    
    fig.add_trace(go.Scatter(
        x=np.concatenate([train_sizes, train_sizes[::-1]]),
        y=np.concatenate([val_mean + val_std, (val_mean - val_std)[::-1]]),
        fill='toself',
        fillcolor='rgba(255,0,0,0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        showlegend=False,
        name='Val ±1 std'
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title='Training Set Size',
        yaxis_title='Score',
        width=800,
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)
    st.caption("💡 Converging lines = good. Large gap = overfitting. Both low = underfitting.")


def plot_validation_curve(param_range, train_scores, val_scores, param_name, title="Validation Curve"):
    """Plot validation curve for hyperparameter tuning"""
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)
    
    fig = go.Figure()
    
    # Training scores
    fig.add_trace(go.Scatter(
        x=param_range, y=train_mean,
        mode='lines+markers',
        name='Training Score',
        line=dict(color='blue', width=2)
    ))
    
    # Validation scores
    fig.add_trace(go.Scatter(
        x=param_range, y=val_mean,
        mode='lines+markers',
        name='Validation Score',
        line=dict(color='red', width=2)
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title=param_name,
        yaxis_title='Score',
        width=800,
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)
    st.caption(f"💡 Shows how {param_name} affects model performance. Peak on validation = optimal value.")


def plot_feature_importance(feature_names, importances, title="Feature Importance", top_n=20):
    """Plot feature importance bar chart"""
    # Sort by importance
    indices = np.argsort(importances)[::-1][:top_n]
    sorted_features = [feature_names[i] for i in indices]
    sorted_importances = importances[indices]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        y=sorted_features[::-1],  # Reverse for better display
        x=sorted_importances[::-1],
        orientation='h',
        marker_color='steelblue'
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title='Importance',
        yaxis_title='Feature',
        width=800,
        height=max(400, top_n * 25),
        showlegend=False
    )
    
    st.plotly_chart(fig, use_container_width=True)
    st.caption(f"💡 Top {top_n} most important features for model predictions.")


def plot_correlation_heatmap(df, title="Correlation Heatmap"):
    """Plot correlation heatmap for numeric features"""
    corr = df.corr()
    
    fig = go.Figure(data=go.Heatmap(
        z=corr.values,
        x=corr.columns,
        y=corr.columns,
        colorscale='RdBu',
        zmid=0,
        text=corr.values,
        texttemplate='%{text:.2f}',
        textfont={"size": 8},
        showscale=True
    ))
    
    fig.update_layout(
        title=title,
        width=800,
        height=700
    )
    
    st.plotly_chart(fig, use_container_width=True)
    st.caption("💡 Red = positive correlation, Blue = negative correlation. Look for redundant features.")


def plot_scatter_matrix(df, target_col=None, sample_size=500):
    """Plot scatter matrix for numeric features"""
    # Sample if too large
    if len(df) > sample_size:
        df_plot = df.sample(n=sample_size, random_state=42)
    else:
        df_plot = df
    
    # Select numeric columns
    numeric_cols = df_plot.select_dtypes(include=[np.number]).columns.tolist()
    
    if target_col and target_col in numeric_cols:
        numeric_cols.remove(target_col)
    
    # Limit to first 6 features for readability
    numeric_cols = numeric_cols[:6]
    
    if target_col:
        color = df_plot[target_col]
        fig = px.scatter_matrix(
            df_plot,
            dimensions=numeric_cols,
            color=color,
            title="Feature Scatter Matrix"
        )
    else:
        fig = px.scatter_matrix(
            df_plot,
            dimensions=numeric_cols,
            title="Feature Scatter Matrix"
        )
    
    fig.update_layout(
        width=900,
        height=900
    )
    
    st.plotly_chart(fig, use_container_width=True)
    st.caption("💡 Visualize pairwise relationships between features.")


def plot_pca_explained_variance(explained_variance_ratio, title="PCA Explained Variance"):
    """Plot explained variance for PCA components"""
    n_components = len(explained_variance_ratio)
    cumsum = np.cumsum(explained_variance_ratio)
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Bar chart for individual variance
    fig.add_trace(
        go.Bar(
            x=list(range(1, n_components + 1)),
            y=explained_variance_ratio,
            name='Individual',
            marker_color='steelblue'
        ),
        secondary_y=False
    )
    
    # Line chart for cumulative variance
    fig.add_trace(
        go.Scatter(
            x=list(range(1, n_components + 1)),
            y=cumsum,
            name='Cumulative',
            mode='lines+markers',
            line=dict(color='red', width=2),
            marker=dict(size=8)
        ),
        secondary_y=True
    )
    
    fig.update_xaxes(title_text="Principal Component")
    fig.update_yaxes(title_text="Explained Variance Ratio", secondary_y=False)
    fig.update_yaxes(title_text="Cumulative Variance", secondary_y=True)
    
    fig.update_layout(
        title=title,
        width=800,
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)
    st.caption("💡 Bars show individual component variance. Line shows cumulative. Choose n to reach ~90% variance.")


def plot_pca_biplot(pca_data, loadings, pc1=0, pc2=1, feature_names=None, target=None):
    """Create PCA biplot with scores and loadings"""
    fig = go.Figure()
    
    # Scatter plot of scores
    if target is not None:
        for label in np.unique(target):
            mask = target == label
            fig.add_trace(go.Scatter(
                x=pca_data[mask, pc1],
                y=pca_data[mask, pc2],
                mode='markers',
                name=f'Class {label}',
                marker=dict(size=6, opacity=0.6)
            ))
    else:
        fig.add_trace(go.Scatter(
            x=pca_data[:, pc1],
            y=pca_data[:, pc2],
            mode='markers',
            marker=dict(size=6, opacity=0.6, color='blue'),
            name='Samples'
        ))
    
    # Add loading vectors
    if feature_names is not None:
        scale = 3  # Scale factor for visibility
        for i, feature in enumerate(feature_names[:10]):  # Limit to top 10
            fig.add_trace(go.Scatter(
                x=[0, loadings[i, pc1] * scale],
                y=[0, loadings[i, pc2] * scale],
                mode='lines+text',
                line=dict(color='red', width=1),
                text=['', feature],
                textposition='top center',
                showlegend=False
            ))
    
    fig.update_layout(
        title=f'PCA Biplot (PC{pc1+1} vs PC{pc2+1})',
        xaxis_title=f'PC{pc1+1}',
        yaxis_title=f'PC{pc2+1}',
        width=800,
        height=600
    )
    
    st.plotly_chart(fig, use_container_width=True)
    st.caption("💡 Points = samples, Red arrows = feature influence on components.")


def plot_kmeans_elbow(k_range, inertias, title="K-Means Elbow Plot"):
    """Plot elbow curve for K-Means"""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=k_range,
        y=inertias,
        mode='lines+markers',
        line=dict(color='blue', width=2),
        marker=dict(size=10)
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title='Number of Clusters (k)',
        yaxis_title='Inertia (Within-Cluster Sum of Squares)',
        width=700,
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)
    st.caption("💡 Look for 'elbow' where inertia decrease slows. That k often works well.")


def plot_kmeans_silhouette(k_range, silhouette_scores, title="Silhouette Score vs K"):
    """Plot silhouette scores for different k values"""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=k_range,
        y=silhouette_scores,
        mode='lines+markers',
        line=dict(color='green', width=2),
        marker=dict(size=10)
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title='Number of Clusters (k)',
        yaxis_title='Silhouette Score',
        width=700,
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)
    st.caption("💡 Higher score = better-defined clusters. Peak suggests optimal k.")


def plot_clusters_2d(X_2d, labels, centers_2d=None, title="Cluster Visualization"):
    """Plot clusters in 2D (e.g., after PCA)"""
    fig = go.Figure()
    
    for cluster in np.unique(labels):
        mask = labels == cluster
        fig.add_trace(go.Scatter(
            x=X_2d[mask, 0],
            y=X_2d[mask, 1],
            mode='markers',
            name=f'Cluster {cluster}',
            marker=dict(size=6, opacity=0.6)
        ))
    
    # Add cluster centers
    if centers_2d is not None:
        fig.add_trace(go.Scatter(
            x=centers_2d[:, 0],
            y=centers_2d[:, 1],
            mode='markers',
            marker=dict(
                size=15,
                color='black',
                symbol='x',
                line=dict(width=2, color='white')
            ),
            name='Centroids'
        ))
    
    fig.update_layout(
        title=title,
        xaxis_title='First Component',
        yaxis_title='Second Component',
        width=800,
        height=600
    )
    
    st.plotly_chart(fig, use_container_width=True)
    st.caption("💡 Different colors = different clusters. X marks = cluster centers.")


def plot_coefficients(coef, feature_names, title="Model Coefficients", top_n=20):
    """Plot model coefficients (for linear models)"""
    # Get absolute values for sorting
    abs_coef = np.abs(coef)
    indices = np.argsort(abs_coef)[::-1][:top_n]
    
    sorted_features = [feature_names[i] for i in indices]
    sorted_coef = coef[indices]
    
    colors = ['red' if c < 0 else 'green' for c in sorted_coef]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        y=sorted_features[::-1],
        x=sorted_coef[::-1],
        orientation='h',
        marker_color=colors[::-1]
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title='Coefficient Value',
        yaxis_title='Feature',
        width=800,
        height=max(400, top_n * 25),
        showlegend=False
    )
    
    st.plotly_chart(fig, use_container_width=True)
    st.caption("💡 Green = positive impact, Red = negative impact. Larger magnitude = stronger effect.")


def plot_decision_boundary_2d(X, y, model, h=0.02, title="Decision Boundary"):
    """Plot 2D decision boundary (requires 2 features)"""
    if X.shape[1] != 2:
        st.warning("Decision boundary visualization requires exactly 2 features.")
        return
    
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    fig = go.Figure()
    
    # Decision boundary
    fig.add_trace(go.Contour(
        x=xx[0],
        y=yy[:, 0],
        z=Z,
        colorscale='Viridis',
        opacity=0.3,
        showscale=False
    ))
    
    # Data points
    for label in np.unique(y):
        mask = y == label
        fig.add_trace(go.Scatter(
            x=X[mask, 0],
            y=X[mask, 1],
            mode='markers',
            name=f'Class {label}',
            marker=dict(size=8)
        ))
    
    fig.update_layout(
        title=title,
        xaxis_title='Feature 1',
        yaxis_title='Feature 2',
        width=700,
        height=600
    )
    
    st.plotly_chart(fig, use_container_width=True)
    st.caption("💡 Background shows decision regions. Points show actual data.")

