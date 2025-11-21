"""
Food Desert Project - Regression Analysis
Course: DATA-245 Machine Learning
Group 3: Jane Heng (Regression Lead)

This script performs comprehensive regression modeling to predict
obesity and diabetes rates from food access and socioeconomic factors.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.linear_model import LinearRegression, Ridge, Lasso, RidgeCV, LassoCV
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class FoodDesertRegression:
    """
    Comprehensive regression analysis for Food Desert Project
    """
    
    def __init__(self, data_path):
        self.data_path = data_path
        self.df = None
        self.models = {}
        self.results = {}
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.feature_names = None
        
    def load_data(self):
        """Load cleaned dataset"""
        print("=" * 80)
        print("Loading Data for Regression Analysis")
        print("=" * 80)
        
        self.df = pd.read_csv(self.data_path)
        print(f"Dataset shape: {self.df.shape}")
        print(f"Counties: {len(self.df):,}")
        
        return self.df
    
    def prepare_features(self, target='obesity'):
        """Prepare features and target for modeling"""
        print("\n" + "=" * 80)
        print(f"Preparing Features for {target.upper()} Prediction")
        print("=" * 80)
        
        # Define predictor features (using normalized versions)
        self.feature_names = [
            'Food_Access_Barrier_Index_normalized',
            'Socioeconomic_Vulnerability_Index_normalized',
            '% Completed High School_normalized',
            'Income Ratio_normalized',
            '% Rural_normalized',
            '% Uninsured_normalized',
            'Food Environment Index_normalized'
        ]
        
        # Check availability
        available_features = [f for f in self.feature_names if f in self.df.columns]
        print(f"\nRequested features: {len(self.feature_names)}")
        print(f"Available features: {len(available_features)}")
        
        # Define target
        if target == 'obesity':
            target_col = '% Adults with Obesity'
        elif target == 'diabetes':
            target_col = '% Adults with Diabetes'
        elif target == 'health_risk':
            target_col = 'Health_Risk_Score'
        else:
            raise ValueError(f"Unknown target: {target}")
        
        print(f"Target variable: {target_col}")
        
        # Extract features and target
        X = self.df[available_features].dropna()
        y = self.df.loc[X.index, target_col]
        
        # Remove any rows with missing target
        mask = ~y.isna()
        X = X[mask]
        y = y[mask]
        
        print(f"\nFinal dataset shape: {X.shape}")
        print(f"Samples: {len(X):,}")
        print(f"Features: {len(available_features)}")
        
        self.feature_names = available_features
        
        return X, y, target_col
    
    def split_data(self, X, y, test_size=0.2, random_state=42):
        """Split data into train and test sets"""
        print("\n" + "=" * 80)
        print("Splitting Data: Train/Test")
        print("=" * 80)
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        print(f"Training set: {len(self.X_train):,} samples ({(1-test_size)*100:.0f}%)")
        print(f"Test set: {len(self.X_test):,} samples ({test_size*100:.0f}%)")
        print(f"Features: {self.X_train.shape[1]}")
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def train_ols(self):
        """Train Ordinary Least Squares regression"""
        print("\n" + "=" * 80)
        print("Training OLS (Ordinary Least Squares) Regression")
        print("=" * 80)
        
        model = LinearRegression()
        model.fit(self.X_train, self.y_train)
        
        # Predictions
        y_train_pred = model.predict(self.X_train)
        y_test_pred = model.predict(self.X_test)
        
        # Metrics
        train_r2 = r2_score(self.y_train, y_train_pred)
        test_r2 = r2_score(self.y_test, y_test_pred)
        train_rmse = np.sqrt(mean_squared_error(self.y_train, y_train_pred))
        test_rmse = np.sqrt(mean_squared_error(self.y_test, y_test_pred))
        train_mae = mean_absolute_error(self.y_train, y_train_pred)
        test_mae = mean_absolute_error(self.y_test, y_test_pred)
        
        # Cross-validation
        cv_scores = cross_val_score(model, self.X_train, self.y_train, 
                                     cv=5, scoring='r2')
        
        print(f"\n✓ OLS Model Trained")
        print(f"  Train R²: {train_r2:.4f}")
        print(f"  Test R²:  {test_r2:.4f}")
        print(f"  Train RMSE: {train_rmse:.4f}")
        print(f"  Test RMSE:  {test_rmse:.4f}")
        print(f"  CV R² (5-fold): {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")
        
        # Store results
        self.models['OLS'] = model
        self.results['OLS'] = {
            'model': model,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'train_rmse': train_rmse,
            'test_rmse': test_rmse,
            'train_mae': train_mae,
            'test_mae': test_mae,
            'cv_scores': cv_scores,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'coefficients': model.coef_,
            'intercept': model.intercept_
        }
        
        return model
    
    def train_ridge(self, alphas=None):
        """Train Ridge regression with cross-validated alpha"""
        print("\n" + "=" * 80)
        print("Training Ridge Regression (L2 Regularization)")
        print("=" * 80)
        
        if alphas is None:
            alphas = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
        
        print(f"Testing {len(alphas)} alpha values: {alphas}")
        
        # Cross-validated Ridge
        model = RidgeCV(alphas=alphas, cv=5, scoring='r2')
        model.fit(self.X_train, self.y_train)
        
        print(f"\n✓ Best alpha selected: {model.alpha_}")
        
        # Predictions
        y_train_pred = model.predict(self.X_train)
        y_test_pred = model.predict(self.X_test)
        
        # Metrics
        train_r2 = r2_score(self.y_train, y_train_pred)
        test_r2 = r2_score(self.y_test, y_test_pred)
        train_rmse = np.sqrt(mean_squared_error(self.y_train, y_train_pred))
        test_rmse = np.sqrt(mean_squared_error(self.y_test, y_test_pred))
        train_mae = mean_absolute_error(self.y_train, y_train_pred)
        test_mae = mean_absolute_error(self.y_test, y_test_pred)
        
        # Cross-validation with best alpha
        cv_scores = cross_val_score(
            Ridge(alpha=model.alpha_), 
            self.X_train, self.y_train, 
            cv=5, scoring='r2'
        )
        
        print(f"  Train R²: {train_r2:.4f}")
        print(f"  Test R²:  {test_r2:.4f}")
        print(f"  Train RMSE: {train_rmse:.4f}")
        print(f"  Test RMSE:  {test_rmse:.4f}")
        print(f"  CV R² (5-fold): {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")
        
        # Store results
        self.models['Ridge'] = model
        self.results['Ridge'] = {
            'model': model,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'train_rmse': train_rmse,
            'test_rmse': test_rmse,
            'train_mae': train_mae,
            'test_mae': test_mae,
            'cv_scores': cv_scores,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'alpha': model.alpha_,
            'coefficients': model.coef_,
            'intercept': model.intercept_
        }
        
        return model
    
    def train_lasso(self, alphas=None):
        """Train Lasso regression with cross-validated alpha"""
        print("\n" + "=" * 80)
        print("Training Lasso Regression (L1 Regularization)")
        print("=" * 80)
        
        if alphas is None:
            alphas = [0.001, 0.01, 0.1, 1.0, 10.0]
        
        print(f"Testing {len(alphas)} alpha values: {alphas}")
        
        # Cross-validated Lasso
        model = LassoCV(alphas=alphas, cv=5, max_iter=10000)
        model.fit(self.X_train, self.y_train)
        
        print(f"\n✓ Best alpha selected: {model.alpha_}")
        
        # Predictions
        y_train_pred = model.predict(self.X_train)
        y_test_pred = model.predict(self.X_test)
        
        # Metrics
        train_r2 = r2_score(self.y_train, y_train_pred)
        test_r2 = r2_score(self.y_test, y_test_pred)
        train_rmse = np.sqrt(mean_squared_error(self.y_train, y_train_pred))
        test_rmse = np.sqrt(mean_squared_error(self.y_test, y_test_pred))
        train_mae = mean_absolute_error(self.y_train, y_train_pred)
        test_mae = mean_absolute_error(self.y_test, y_test_pred)
        
        # Cross-validation with best alpha
        cv_scores = cross_val_score(
            Lasso(alpha=model.alpha_, max_iter=10000), 
            self.X_train, self.y_train, 
            cv=5, scoring='r2'
        )
        
        # Feature selection
        n_features_selected = np.sum(model.coef_ != 0)
        selected_features = [f for f, c in zip(self.feature_names, model.coef_) if c != 0]
        
        print(f"  Train R²: {train_r2:.4f}")
        print(f"  Test R²:  {test_r2:.4f}")
        print(f"  Train RMSE: {train_rmse:.4f}")
        print(f"  Test RMSE:  {test_rmse:.4f}")
        print(f"  CV R² (5-fold): {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")
        print(f"  Features selected: {n_features_selected}/{len(self.feature_names)}")
        
        if len(selected_features) < len(self.feature_names):
            print(f"  Selected features: {selected_features}")
        
        # Store results
        self.models['Lasso'] = model
        self.results['Lasso'] = {
            'model': model,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'train_rmse': train_rmse,
            'test_rmse': test_rmse,
            'train_mae': train_mae,
            'test_mae': test_mae,
            'cv_scores': cv_scores,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'alpha': model.alpha_,
            'coefficients': model.coef_,
            'intercept': model.intercept_,
            'n_features_selected': n_features_selected,
            'selected_features': selected_features
        }
        
        return model
    
    def compare_models(self):
        """Compare all trained models"""
        print("\n" + "=" * 80)
        print("MODEL COMPARISON")
        print("=" * 80)
        
        comparison = []
        for name, results in self.results.items():
            comparison.append({
                'Model': name,
                'Train R²': results['train_r2'],
                'Test R²': results['test_r2'],
                'Train RMSE': results['train_rmse'],
                'Test RMSE': results['test_rmse'],
                'CV R² Mean': results['cv_mean'],
                'CV R² Std': results['cv_std'],
                'Overfit Gap': results['train_r2'] - results['test_r2']
            })
        
        comparison_df = pd.DataFrame(comparison).sort_values('Test R²', ascending=False)
        
        print("\n" + comparison_df.to_string(index=False))
        
        # Identify best model
        best_model = comparison_df.iloc[0]['Model']
        print(f"\n🏆 Best Model (by Test R²): {best_model}")
        print(f"   Test R²: {comparison_df.iloc[0]['Test R²']:.4f}")
        print(f"   Overfitting Gap: {comparison_df.iloc[0]['Overfit Gap']:.4f}")
        
        return comparison_df
    
    def analyze_coefficients(self, model_name='Ridge'):
        """Analyze and visualize coefficients"""
        print("\n" + "=" * 80)
        print(f"Coefficient Analysis: {model_name}")
        print("=" * 80)
        
        results = self.results[model_name]
        coefficients = results['coefficients']
        
        # Create coefficient dataframe
        coef_df = pd.DataFrame({
            'Feature': [f.replace('_normalized', '') for f in self.feature_names],
            'Coefficient': coefficients,
            'Abs_Coefficient': np.abs(coefficients)
        }).sort_values('Abs_Coefficient', ascending=False)
        
        print("\nCoefficients (sorted by absolute value):")
        print("-" * 80)
        for _, row in coef_df.iterrows():
            sign = '+' if row['Coefficient'] > 0 else ''
            print(f"  {row['Feature']:50s}: {sign}{row['Coefficient']:7.4f}")
        
        print(f"\nIntercept: {results['intercept']:.4f}")
        
        return coef_df
    
    def create_visualizations(self, target_name):
        """Create comprehensive visualizations"""
        print("\n" + "=" * 80)
        print("Creating Visualizations")
        print("=" * 80)
        
        # 1. Model comparison bar chart
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        models_list = list(self.results.keys())
        train_r2 = [self.results[m]['train_r2'] for m in models_list]
        test_r2 = [self.results[m]['test_r2'] for m in models_list]
        train_rmse = [self.results[m]['train_rmse'] for m in models_list]
        test_rmse = [self.results[m]['test_rmse'] for m in models_list]
        
        # R² comparison
        x = np.arange(len(models_list))
        width = 0.35
        axes[0, 0].bar(x - width/2, train_r2, width, label='Train', color='steelblue', alpha=0.8)
        axes[0, 0].bar(x + width/2, test_r2, width, label='Test', color='coral', alpha=0.8)
        axes[0, 0].set_xlabel('Model', fontweight='bold')
        axes[0, 0].set_ylabel('R² Score', fontweight='bold')
        axes[0, 0].set_title('R² Score Comparison', fontweight='bold', fontsize=14)
        axes[0, 0].set_xticks(x)
        axes[0, 0].set_xticklabels(models_list)
        axes[0, 0].legend()
        axes[0, 0].grid(alpha=0.3, axis='y')
        
        # RMSE comparison
        axes[0, 1].bar(x - width/2, train_rmse, width, label='Train', color='steelblue', alpha=0.8)
        axes[0, 1].bar(x + width/2, test_rmse, width, label='Test', color='coral', alpha=0.8)
        axes[0, 1].set_xlabel('Model', fontweight='bold')
        axes[0, 1].set_ylabel('RMSE', fontweight='bold')
        axes[0, 1].set_title('RMSE Comparison', fontweight='bold', fontsize=14)
        axes[0, 1].set_xticks(x)
        axes[0, 1].set_xticklabels(models_list)
        axes[0, 1].legend()
        axes[0, 1].grid(alpha=0.3, axis='y')
        
        # Coefficient comparison for Ridge
        ridge_coef = self.results['Ridge']['coefficients']
        coef_names = [f.replace('_normalized', '').replace('_', ' ') for f in self.feature_names]
        colors = ['green' if c > 0 else 'red' for c in ridge_coef]
        axes[1, 0].barh(range(len(ridge_coef)), ridge_coef, color=colors, alpha=0.7, edgecolor='black')
        axes[1, 0].set_yticks(range(len(ridge_coef)))
        axes[1, 0].set_yticklabels(coef_names, fontsize=9)
        axes[1, 0].set_xlabel('Coefficient Value', fontweight='bold')
        axes[1, 0].set_title('Ridge Coefficients', fontweight='bold', fontsize=14)
        axes[1, 0].axvline(x=0, color='black', linewidth=1)
        axes[1, 0].grid(alpha=0.3, axis='x')
        
        # Actual vs Predicted (Ridge)
        y_pred = self.results['Ridge']['model'].predict(self.X_test)
        axes[1, 1].scatter(self.y_test, y_pred, alpha=0.5, s=30, edgecolors='black', linewidth=0.5)
        
        # Perfect prediction line
        min_val = min(self.y_test.min(), y_pred.min())
        max_val = max(self.y_test.max(), y_pred.max())
        axes[1, 1].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
        
        axes[1, 1].set_xlabel(f'Actual {target_name}', fontweight='bold')
        axes[1, 1].set_ylabel(f'Predicted {target_name}', fontweight='bold')
        axes[1, 1].set_title(f'Actual vs Predicted (Ridge) - R²={self.results["Ridge"]["test_r2"]:.3f}', 
                            fontweight='bold', fontsize=14)
        axes[1, 1].legend()
        axes[1, 1].grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'./output/regression_{target_name.lower().replace(" ", "_")}.png', 
                   dpi=300, bbox_inches='tight')
        print(f"✓ Visualization saved: regression_{target_name.lower().replace(' ', '_')}.png")
        
        return fig
    
    def save_results(self, target_name):
        """Save regression results"""
        print("\n" + "=" * 80)
        print("Saving Results")
        print("=" * 80)
        
        # Save comparison table
        comparison = []
        for name, results in self.results.items():
            comparison.append({
                'Model': name,
                'Target': target_name,
                'Train_R2': results['train_r2'],
                'Test_R2': results['test_r2'],
                'Train_RMSE': results['train_rmse'],
                'Test_RMSE': results['test_rmse'],
                'Train_MAE': results['train_mae'],
                'Test_MAE': results['test_mae'],
                'CV_R2_Mean': results['cv_mean'],
                'CV_R2_Std': results['cv_std']
            })
        
        comparison_df = pd.DataFrame(comparison)
        output_file = f'./output/regression_results_{target_name.lower().replace(" ", "_")}.csv'
        comparison_df.to_csv(output_file, index=False)
        print(f"✓ Results saved: {output_file}")
        
        # Save coefficients
        best_model_name = comparison_df.loc[comparison_df['Test_R2'].idxmax(), 'Model']
        coef_df = pd.DataFrame({
            'Feature': [f.replace('_normalized', '') for f in self.feature_names],
            'Coefficient': self.results[best_model_name]['coefficients']
        })
        coef_df['Intercept'] = self.results[best_model_name]['intercept']
        
        coef_file = f'./output/regression_coefficients_{target_name.lower().replace(" ", "_")}.csv'
        coef_df.to_csv(coef_file, index=False)
        print(f"✓ Coefficients saved: {coef_file}")
        
        return comparison_df
    
    def run_complete_analysis(self, target='obesity'):
        """Run complete regression pipeline"""
        print("\n")
        print("╔" + "=" * 78 + "╗")
        print("║" + " " * 18 + "REGRESSION ANALYSIS - FOOD DESERT PROJECT" + " " * 19 + "║")
        print("╚" + "=" * 78 + "╝")
        print("\n")
        
        # Load and prepare
        self.load_data()
        X, y, target_name = self.prepare_features(target=target)
        self.split_data(X, y)
        
        # Train models
        self.train_ols()
        self.train_ridge()
        self.train_lasso()
        
        # Compare
        comparison_df = self.compare_models()
        
        # Analyze best model
        best_model = comparison_df.iloc[0]['Model']
        self.analyze_coefficients(model_name=best_model)
        
        # Visualize
        self.create_visualizations(target_name)
        
        # Save
        self.save_results(target_name)
        
        print("\n" + "=" * 80)
        print("✓✓✓ REGRESSION ANALYSIS COMPLETE! ✓✓✓")
        print("=" * 80)
        
        return self.results, comparison_df


# Main execution
if __name__ == "__main__":
    # Analyze Obesity
    print("\n" + "🔴" * 40)
    print("ANALYZING OBESITY RATES")
    print("🔴" * 40)
    
    analyzer_obesity = FoodDesertRegression(data_path='./output/cleaned_health_data.csv')
    results_obesity, comparison_obesity = analyzer_obesity.run_complete_analysis(target='obesity')
    
    # Analyze Diabetes
    print("\n\n" + "🔵" * 40)
    print("ANALYZING DIABETES RATES")
    print("🔵" * 40)
    
    analyzer_diabetes = FoodDesertRegression(data_path='./output/cleaned_health_data.csv')
    results_diabetes, comparison_diabetes = analyzer_diabetes.run_complete_analysis(target='diabetes')
    
    print("\n\n" + "=" * 80)
    print("ALL REGRESSION ANALYSES COMPLETE!")
    print("=" * 80)
    print("\n✓ Results saved to ./output/")
    print("✓ Ready for Streamlit dashboard!")