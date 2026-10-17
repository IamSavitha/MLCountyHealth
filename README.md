# Food Desert Effect on County-Level Health Outcomes

![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikitlearn&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-3F4F75?logo=plotly&logoColor=white)

Team machine-learning study of how food access shapes community health.

## Overview

This project investigates how food access inequality correlates with community-level health outcomes (obesity, diabetes) across 2,275 US counties.

## Project Structure

```
MLCountyHealth/
├── data/
│   ├── raw/                    # Original source data
│   ├── processed/              # Cleaned datasets
│   └── output/                 # Analysis results (CSV, PNG)
├── notebooks/                  # Jupyter notebooks for analysis
├── src/
│   ├── analysis/               # Core analysis scripts
│   └── utils/                  # Utility modules
├── dashboards/                 # Streamlit applications
├── docs/
│   ├── reports/                # PDF documentation
│   └── images/                 # Charts and diagrams
├── requirements.txt            # Python dependencies
└── README.md
```

## Installation

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Run Analysis Scripts

```bash
# Regression analysis
python src/analysis/Regression_analysis.py

# PCA analysis
python src/analysis/pcaanalysis.py
```

### Launch Main Dashboard

```bash
cd dashboards
streamlit run main_dashboard.py
```

The dashboard includes 8 pages:
1. Project Overview
2. Data Exploration
3. Regression Analysis (Linear, Ridge, Lasso)
4. Classification Models (Logistic, KNN, Naive Bayes, SVM, Decision Tree, Random Forest, Extra Trees)
5. Clustering Analysis (K-Means, Hierarchical)
6. PCA Analysis
7. Model Comparison
8. Key Insights & Conclusions

### Jupyter Notebooks

```bash
jupyter notebook notebooks/
```

Key notebooks:
- `Food_Desert_Data_Cleaning.ipynb` - Data preprocessing pipeline
- `Google_Colab_K_Means.ipynb` - Clustering analysis
- `Regression_Modeling_Diabetes_Obesity.ipynb` - Regression modeling

## ML Methods

- **Regression**: OLS, Ridge, Lasso
- **Clustering**: K-Means (5 clusters)
- **Dimensionality Reduction**: PCA

## Data

- **Source**: 2025 County Health Rankings
- **Size**: 2,275 counties across 48 states
- **Target Variables**: Adult obesity rate, diabetes rate
- **Features**: Food environment, socioeconomic factors, education, rurality

## Team

- Savitha Vijayarangan - Project Coordination
- Jane Heng - Regression Lead
- Rishi Visweswar Boppana - PCA Lead
- Kapil Reddy Sanikommu - Dashboard Lead
