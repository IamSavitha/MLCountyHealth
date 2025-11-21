"""
Food Desert Project - Principal Component Analysis (PCA)
Course: DATA-245 Machine Learning
Group 3: Rishi Visweswar Boppana (PCA Lead)

This script performs PCA for dimensionality reduction and visualization
of the food desert and health outcomes relationships.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class FoodDesertPCA:
    """
    Principal Component Analysis for Food Desert Project
    """
    
    def __init__(self, data_path):
        self.data_path = data_path
        self.df = None
        self.pca = None
        self.scaler = None
        self.X_scaled = None
        self.principal_components = None
        self.feature_names = None
        
    def load_data(self):
        """Load cleaned dataset"""
        print("=" * 80)
        print("Loading Data for PCA Analysis")
        print("=" * 80)
        
        self.df = pd.read_csv(self.data_path)
        print(f"Dataset shape: {self.df.shape}")
        print(f"Counties: {len(self.df):,}")
        
        return self.df
    
    def prepare_features(self):
        """Select and prepare features for PCA"""
        print("\n" + "=" * 80)
        print("Selecting Features for PCA")
        print("=" * 80)
        
        # Select features for PCA
        self.feature_names = [
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
        
        # Check availability
        available_features = [f for f in self.feature_names if f in self.df.columns]
        
        print(f"Requested features: {len(self.feature_names)}")
        print(f"Available features: {len(available_features)}")
        
        # Extract feature matrix
        X = self.df[available_features].dropna()
        
        print(f"\nFeature matrix shape: {X.shape}")
        print(f"Samples after removing missing values: {len(X):,}")
        
        self.feature_names = available_features
        
        return X
    
    def standardize_features(self, X):
        """Standardize features (required for PCA)"""
        print("\n" + "=" * 80)
        print("Standardizing Features")
        print("=" * 80)
        
        self.scaler = StandardScaler()
        self.X_scaled = self.scaler.fit_transform(X)
        
        print("✓ Features standardized (mean=0, std=1)")
        print(f"Scaled data shape: {self.X_scaled.shape}")
        
        # Verify standardization
        means = self.X_scaled.mean(axis=0)
        stds = self.X_scaled.std(axis=0)
        print(f"Mean of scaled features: {means.mean():.6f} (should be ~0)")
        print(f"Std of scaled features: {stds.mean():.3f} (should be ~1)")
        
        return self.X_scaled
    
    def fit_pca(self, n_components=None):
        """Fit PCA model"""
        print("\n" + "=" * 80)
        print("Fitting PCA Model")
        print("=" * 80)
        
        if n_components is None:
            n_components = min(len(self.feature_names), self.X_scaled.shape[0])
        
        self.pca = PCA(n_components=n_components)
        self.principal_components = self.pca.fit_transform(self.X_scaled)
        
        print(f"Number of components: {n_components}")
        print(f"Principal components shape: {self.principal_components.shape}")
        
        return self.pca
    
    def analyze_variance(self):
        """Analyze variance explained by components"""
        print("\n" + "=" * 80)
        print("Variance Analysis")
        print("=" * 80)
        
        # Variance explained by each component
        var_explained = self.pca.explained_variance_ratio_
        cum_var_explained = np.cumsum(var_explained)
        
        print("\nVariance Explained by Each Component:")
        print("-" * 80)
        for i, (var, cum_var) in enumerate(zip(var_explained, cum_var_explained), 1):
            print(f"PC{i}: {var*100:6.2f}% (Cumulative: {cum_var*100:6.2f}%)")
        
        print(f"\n✓ First 2 components explain: {cum_var_explained[1]*100:.2f}% of variance")
        print(f"✓ First 3 components explain: {cum_var_explained[2]*100:.2f}% of variance")
        
        return var_explained, cum_var_explained
    
    def get_component_loadings(self):
        """Get feature loadings for each component"""
        print("\n" + "=" * 80)
        print("Component Loadings (Feature Contributions)")
        print("=" * 80)
        
        # Create loadings dataframe
        loadings_df = pd.DataFrame(
            self.pca.components_.T,
            columns=[f'PC{i+1}' for i in range(self.pca.n_components_)],
            index=self.feature_names
        )
        
        print("\nTop 3 features for each component:")
        print("-" * 80)
        
        for i in range(min(3, self.pca.n_components_)):
            pc_name = f'PC{i+1}'
            print(f"\n{pc_name}:")
            top_features = loadings_df[pc_name].abs().sort_values(ascending=False).head(3)
            for feature, loading in top_features.items():
                actual_loading = loadings_df.loc[feature, pc_name]
                print(f"  {feature:45s}: {actual_loading:7.3f}")
        
        return loadings_df
    
    def create_scree_plot(self):
        """Create scree plot showing variance explained"""
        var_explained = self.pca.explained_variance_ratio_
        cum_var_explained = np.cumsum(var_explained)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Individual variance
        ax1.bar(range(1, len(var_explained) + 1), var_explained * 100, 
                color='steelblue', alpha=0.8, edgecolor='black')
        ax1.set_xlabel('Principal Component', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Variance Explained (%)', fontsize=12, fontweight='bold')
        ax1.set_title('Scree Plot: Variance per Component', fontsize=14, fontweight='bold')
        ax1.grid(alpha=0.3)
        ax1.set_xticks(range(1, len(var_explained) + 1))
        
        # Cumulative variance
        ax2.plot(range(1, len(cum_var_explained) + 1), cum_var_explained * 100,
                marker='o', markersize=8, linewidth=2, color='darkred')
        ax2.axhline(y=80, color='green', linestyle='--', label='80% threshold')
        ax2.fill_between(range(1, len(cum_var_explained) + 1), 
                         cum_var_explained * 100, alpha=0.3, color='darkred')
        ax2.set_xlabel('Number of Components', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Cumulative Variance Explained (%)', fontsize=12, fontweight='bold')
        ax2.set_title('Cumulative Variance Explained', fontsize=14, fontweight='bold')
        ax2.legend()
        ax2.grid(alpha=0.3)
        ax2.set_xticks(range(1, len(cum_var_explained) + 1))
        
        plt.tight_layout()
        plt.savefig('/Users/savithavijayarangan/Desktop/ML group project/output/pca_scree_plot.png', dpi=300, bbox_inches='tight')
        print("\n✓ Scree plot saved to ./outputs/pca_scree_plot.png")
        
        return fig
    
    def create_biplot(self, pc1=0, pc2=1):
        """Create biplot showing samples and feature loadings"""
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Plot samples
        scatter = ax.scatter(
            self.principal_components[:, pc1],
            self.principal_components[:, pc2],
            c=self.df['Health_Risk_Score'].values[:len(self.principal_components)],
            cmap='RdYlGn_r',
            alpha=0.6,
            s=50,
            edgecolors='black',
            linewidth=0.5
        )
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Health Risk Score', fontsize=12, fontweight='bold')
        
        # Plot feature loadings as arrows
        loadings = self.pca.components_
        scale = 3.5  # Scale for visibility
        
        for i, feature in enumerate(self.feature_names):
            ax.arrow(0, 0, 
                    loadings[pc1, i] * scale, 
                    loadings[pc2, i] * scale,
                    head_width=0.15, head_length=0.15,
                    fc='darkblue', ec='darkblue', alpha=0.7, linewidth=2)
            ax.text(loadings[pc1, i] * scale * 1.15,
                   loadings[pc2, i] * scale * 1.15,
                   feature, fontsize=9, ha='center', va='center',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
        
        var1 = self.pca.explained_variance_ratio_[pc1] * 100
        var2 = self.pca.explained_variance_ratio_[pc2] * 100
        
        ax.set_xlabel(f'PC{pc1+1} ({var1:.1f}% variance)', fontsize=12, fontweight='bold')
        ax.set_ylabel(f'PC{pc2+1} ({var2:.1f}% variance)', fontsize=12, fontweight='bold')
        ax.set_title('PCA Biplot: Counties and Feature Loadings', 
                    fontsize=14, fontweight='bold', pad=20)
        ax.grid(alpha=0.3)
        ax.axhline(y=0, color='k', linewidth=0.5)
        ax.axvline(x=0, color='k', linewidth=0.5)
        
        plt.tight_layout()
        plt.savefig('./output/pca_biplot.png', dpi=300, bbox_inches='tight')
        print("✓ Biplot saved to ./output/pca_biplot.png")
        
        return fig
    
    def create_3d_scatter(self):
        """Create 3D scatter plot of first 3 components"""
        from mpl_toolkits.mplot3d import Axes3D
        
        fig = plt.figure(figsize=(12, 9))
        ax = fig.add_subplot(111, projection='3d')
        
        scatter = ax.scatter(
            self.principal_components[:, 0],
            self.principal_components[:, 1],
            self.principal_components[:, 2],
            c=self.df['Health_Risk_Score'].values[:len(self.principal_components)],
            cmap='RdYlGn_r',
            s=50,
            alpha=0.6,
            edgecolors='black',
            linewidth=0.5
        )
        
        var1 = self.pca.explained_variance_ratio_[0] * 100
        var2 = self.pca.explained_variance_ratio_[1] * 100
        var3 = self.pca.explained_variance_ratio_[2] * 100
        
        ax.set_xlabel(f'PC1 ({var1:.1f}%)', fontsize=11, fontweight='bold')
        ax.set_ylabel(f'PC2 ({var2:.1f}%)', fontsize=11, fontweight='bold')
        ax.set_zlabel(f'PC3 ({var3:.1f}%)', fontsize=11, fontweight='bold')
        ax.set_title('3D PCA Scatter: First Three Components', 
                    fontsize=14, fontweight='bold', pad=20)
        
        cbar = plt.colorbar(scatter, ax=ax, pad=0.1, shrink=0.8)
        cbar.set_label('Health Risk Score', fontsize=11, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('./output/pca_3d_scatter.png', dpi=300, bbox_inches='tight')
        print( "3D scatter plot saved to ./output/pca_3d_scatter.png")
        
        return fig
    
    def save_results(self):
        """Save PCA results"""
        print("\n" + "=" * 80)
        print("Saving PCA Results")
        print("=" * 80)
        
        # Create results dataframe with principal components
        results_df = self.df.iloc[:len(self.principal_components)].copy()
        
        for i in range(min(5, self.pca.n_components_)):
            results_df[f'PC{i+1}'] = self.principal_components[:, i]
        
        # Save
        output_file = './output/pca_results.csv'
        results_df.to_csv(output_file, index=False)
        print(f"PCA results saved to: {output_file}")
        
        # Save loadings
        loadings_df = pd.DataFrame(
            self.pca.components_.T,
            columns=[f'PC{i+1}' for i in range(self.pca.n_components_)],
            index=self.feature_names
        )
        loadings_file = './output/pca_loadings.csv'
        loadings_df.to_csv(loadings_file)
        print(f"Component loadings saved to: {loadings_file}")
        
        return results_df
    
    def run_complete_analysis(self):
        """Run complete PCA pipeline"""
        print("\n")
        print("╔" + "=" * 78 + "╗")
        print("║" + " " * 20 + "PCA ANALYSIS - FOOD DESERT PROJECT" + " " * 24 + "║")
        print("╚" + "=" * 78 + "╝")
        print("\n")
        
        # Pipeline
        self.load_data()
        X = self.prepare_features()
        self.standardize_features(X)
        self.fit_pca(n_components=min(11, len(self.feature_names)))
        
        var_explained, cum_var = self.analyze_variance()
        loadings_df = self.get_component_loadings()
        
        # Visualizations
        print("\n" + "=" * 80)
        print("Creating Visualizations")
        print("=" * 80)
        
        self.create_scree_plot()
        self.create_biplot(pc1=0, pc2=1)
        self.create_3d_scatter()
        
        # Save results
        results_df = self.save_results()
        
        print("\n" + "=" * 80)
        print("PCA ANALYSIS COMPLETE! ")
        print("=" * 80)
        print(f"\nKey Findings:")
        print(f"  • PC1 explains {var_explained[0]*100:.1f}% of variance")
        print(f"  • PC2 explains {var_explained[1]*100:.1f}% of variance")
        print(f"  • First 3 PCs explain {cum_var[2]*100:.1f}% of total variance")
        print(f"  • Total components: {self.pca.n_components_}")
        
        return results_df, loadings_df


# Main execution
if __name__ == "__main__":
    pca_analyzer = FoodDesertPCA(data_path='./output/cleaned_health_data.csv')
    results_df, loadings_df = pca_analyzer.run_complete_analysis()
    
    print("\n All PCA outputs saved to ./output/")
    print(" Ready for Streamlit dashboard!")