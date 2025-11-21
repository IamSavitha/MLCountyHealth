# app.py
# Streamlit Policy Recommender for County Health Rankings (2025)

# Description: End-to-end app to (1) load County Health Rankings data, (2) fit a model
# to predict a chosen health outcome, (3) rank feature importances, and (4) suggest
# policy actions for top drivers.

import io
import re
import json
from typing import List, Tuple

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# ----------------------------
# Page & Style
# ----------------------------
st.set_page_config(
    page_title="County Health Policy Recommender",
    page_icon="🩺",
    layout="wide",
)

st.title("County Health Policy Recommender")
st.caption(
    "Identify the top predictors of poor health outcomes and generate targeted policy recommendations."
)

# ----------------------------
# Utilities
# ----------------------------

DEFAULT_DATA_PATH = "../data/processed/cleaned_health_data.csv"

# Known identifier-like columns to exclude from modeling by default
ID_LIKE = {
    "FIPS", "CountyFIPS", "STATEFIPS", "STATE", "ST", "STATEFP", "GEOID",
    "CountyFIPSCode", "County Code", "CountyCode", "CountyID", "County Id",
    "County", "County Name", "county", "county_name", "CountyName",
    "State", "state", "State Name", "state_name", "StateAbbr", "State Abbreviation"
}

LIKELY_TARGETS = [
    "Life Expectancy",
    "Premature death",
    "Years of Potential Life Lost",
    "% Adults with Obesity",
    "% Adult Obesity",
    "% Fair or Poor Health",
    "% Poor or Fair Health",
]

NUMERIC_DTYPES = ["int16", "int32", "int64", "float16", "float32", "float64"]

@st.cache_data(show_spinner=False)
def read_any_excel(uploaded_or_path) -> pd.DataFrame:
    """Read first sheet of an xlsx into DataFrame, trying header inference."""
    if uploaded_or_path is None:
        raise FileNotFoundError("No file provided and default not available.")
    try:
        if isinstance(uploaded_or_path, io.BytesIO):
            return pd.read_csv(uploaded_or_path)
        if isinstance(uploaded_or_path, (bytes, bytearray)):
            return pd.read_csv(io.BytesIO(uploaded_or_path))
        # str path
        return pd.read_csv(uploaded_or_path)
    except Exception as e:
        raise RuntimeError(f"Failed to read Excel: {e}")


def infer_numeric_columns(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if str(df[c].dtype) in NUMERIC_DTYPES]


def guess_target(df: pd.DataFrame) -> str:
    # prefer likely targets present
    for t in LIKELY_TARGETS:
        for col in df.columns:
            if t.lower() in str(col).lower():
                return col
    # fallback: the first numeric column
    numeric_cols = infer_numeric_columns(df)
    return numeric_cols[0] if numeric_cols else df.columns[0]


def default_features(df: pd.DataFrame, target: str) -> List[str]:
    numeric_cols = set(infer_numeric_columns(df))
    # remove target and obvious id-like cols
    cleaned = [
        c for c in numeric_cols
        if c != target and c not in ID_LIKE and not re.search(r"fips|geoid|code", str(c), re.I)
    ]
    return sorted(cleaned)


def split_train_test(df: pd.DataFrame, features: List[str], target: str, test_size: float, seed: int):
    X = df[features].copy()
    y = df[target].copy()
    # drop rows with NA in features or target
    mask = X.notna().all(axis=1) & y.notna()
    X = X.loc[mask]
    y = y.loc[mask]
    return train_test_split(X, y, test_size=test_size, random_state=seed)


def fit_model(model_name: str, X_train: pd.DataFrame, y_train: pd.Series):
    if model_name == "Linear Regression (Standardized)":
        model = Pipeline([
            ("scaler", StandardScaler()),
            ("linreg", LinearRegression()),
        ])
    elif model_name == "Random Forest (200 trees)":
        model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    else:
        raise ValueError("Unknown model option")

    model.fit(X_train, y_train)
    return model


def compute_metrics(y_true, y_pred) -> dict:
    return {
        "R²": r2_score(y_true, y_pred),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
    }


def get_feature_importance(model, X_valid: pd.DataFrame, y_valid: pd.Series, model_name: str) -> pd.DataFrame:
    # Try model-native importances first
    if model_name.startswith("Random Forest") and hasattr(model, "feature_importances_"):
        imp = pd.Series(model.feature_importances_, index=X_valid.columns)
        df_imp = imp.sort_values(ascending=False).rename("importance").reset_index()
        df_imp.columns = ["feature", "importance"]
        df_imp["method"] = "model"
        return df_imp

    # For linear regression pipeline or any model, compute permutation importance
    try:
        result = permutation_importance(model, X_valid, y_valid, n_repeats=15, random_state=42, n_jobs=-1)
        imp = pd.Series(result.importances_mean, index=X_valid.columns)
        df_imp = imp.sort_values(ascending=False).rename("importance").reset_index()
        df_imp.columns = ["feature", "importance"]
        df_imp["method"] = "permutation"
        return df_imp
    except Exception:
        # Fallback: zero importances
        return pd.DataFrame({"feature": list(X_valid.columns), "importance": 0.0, "method": "none"})


# Simple pattern -> policy playbook (expand as needed)
POLICY_PLAYBOOK = [
    (r"obesity|bmi|overweight|physical inactivity|exercise", "Fund community-based physical activity programs; expand safe parks and trails; support school PE and workplace wellness."),
    (r"smok", "Increase tobacco taxes; expand smoking cessation services; enforce smoke-free policies."),
    (r"alcohol|binge", "Limit alcohol outlet density; enforce responsible beverage service; expand screening and brief intervention."),
    (r"uninsured|insurance|coverage", "Expand Medicaid outreach; support enrollment navigators; incentivize employer-sponsored coverage."),
    (r"primary care|physician|doctor|pcp|clinic", "Recruit and retain primary care providers; fund FQHCs; deploy mobile clinics and telehealth."),
    (r"preventable hospital|readmission|er visit|hospital stay", "Invest in care coordination and transitional care; expand chronic disease management; improve access to primary care."),
    (r"education|hs graduation|bachelor|college|literacy", "Invest in early childhood education; increase high school completion programs; expand adult education and job training."),
    (r"poverty|income|unemployment|inequality|children in poverty", "Expand EITC/CTC uptake; job training; childcare support; affordable housing and food assistance."),
    (r"housing|crowding|severe housing|cost burden", "Increase affordable housing supply; rental assistance; healthy homes remediation; housing-first initiatives."),
    (r"air pollution|pm2.5|ozone|water violations|environment", "Tighten emissions controls; expand air monitoring; water system upgrades; green infrastructure."),
    (r"food|grocery|food environment|food desert|access to healthy food", "Support healthy corner stores; expand SNAP/WIC access; subsidize fresh food delivery; incentivize supermarkets in underserved areas."),
    (r"violence|crime|injury|homicide", "Community violence intervention; street outreach; trauma-informed care; environmental design for safety."),
    (r"commute|transport|transit|vehicle miles", "Improve public transit frequency; build complete streets; support employer commute programs."),
]


def policy_recs_from_features(top_features: List[Tuple[str, float]], k: int = 5) -> pd.DataFrame:
    rows = []
    for feat, score in top_features[:k]:
        rec = "General capacity-building in public health and data systems."
        for pattern, suggestion in POLICY_PLAYBOOK:
            if re.search(pattern, str(feat), re.I):
                rec = suggestion
                break
        rows.append({"Driver (feature)": feat, "Importance": round(float(score), 6), "Suggested policies": rec})
    return pd.DataFrame(rows)


# ----------------------------
# Sidebar: Data & Options
# ----------------------------
with st.sidebar:
    st.header("1) Load Data")
    uploaded = st.file_uploader("Upload the 2025 County Health Rankings .csv", type=["csv"])  

    df = None
    if uploaded is not None:
        df = read_any_excel(uploaded)
    else:
        # Try default path (works in this environment only)
        try:
            df = read_any_excel(DEFAULT_DATA_PATH)
            st.info("Loaded default dataset from /mnt/data (you can upload your own file).")
        except Exception:
            st.warning("Upload the dataset to continue.")

    if df is not None:
        st.success(f"Data loaded: {df.shape[0]:,} rows × {df.shape[1]:,} columns")
        # Optional state filter if available
        state_col = None
        for c in ["State", "state", "STATE", "State Name", "StateName"]:
            if c in df.columns:
                state_col = c
                break
        if state_col:
            states = ["(All)"] + sorted([s for s in df[state_col].dropna().unique()])
            chosen_state = st.selectbox("Filter by State (optional)", states)
            if chosen_state != "(All)":
                df = df[df[state_col] == chosen_state].copy()

    st.header("2) Modeling Options")
    model_name = st.selectbox(
        "Model",
        ["Linear Regression (Standardized)", "Random Forest (200 trees)"]
    )
    test_size = st.slider("Test size", 0.1, 0.4, 0.2, 0.05)
    seed = st.number_input("Random seed", min_value=0, max_value=10_000, value=42, step=1)


# ----------------------------
# Main App Body
# ----------------------------
if df is not None and len(df) > 2:
    st.header("Select Target & Features")

    # Guess target
    tgt_guess = guess_target(df)

    # Show column type summary
    numeric_cols = infer_numeric_columns(df)
    st.caption(f"Numeric columns detected: {len(numeric_cols)}")

    target = st.selectbox(
        "Target outcome (to predict)",
        options=[c for c in df.columns if c in numeric_cols or any(t.lower() in c.lower() for t in LIKELY_TARGETS)],
        index=max(0, [i for i, c in enumerate(df.columns) if c == tgt_guess][0]) if tgt_guess in df.columns else 0
    )

    feat_default = default_features(df, target)
    features = st.multiselect(
        "Predictor features",
        options=[c for c in numeric_cols if c != target],
        default=feat_default[:30]  # cap to first 30 for convenience
    )

    if len(features) < 2:
        st.warning("Please select at least two predictor features.")
        st.stop()

    # Train/validate
    X_train, X_test, y_train, y_test = split_train_test(df, features, target, test_size, seed)

    if len(X_train) < 20 or len(X_test) < 10:
        st.warning("Not enough rows after cleaning. Try removing filters or selecting different features.")
        st.stop()

    model = fit_model(model_name, X_train, y_train)

    # Predictions & metrics
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    metrics_train = compute_metrics(y_train, y_pred_train)
    metrics_test = compute_metrics(y_test, y_pred_test)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Train Metrics")
        st.metric("R²", f"{metrics_train['R²']:.3f}")
        st.metric("RMSE", f"{metrics_train['RMSE']:.3f}")
        st.metric("MAE", f"{metrics_train['MAE']:.3f}")
    with col2:
        st.subheader("Test Metrics")
        st.metric("R²", f"{metrics_test['R²']:.3f}")
        st.metric("RMSE", f"{metrics_test['RMSE']:.3f}")
        st.metric("MAE", f"{metrics_test['MAE']:.3f}")

    st.divider()
    st.subheader("Top Drivers (Feature Importance)")

    # Importance (model-native or permutation)
    imp_df = get_feature_importance(model, X_test, y_test, model_name)
    if imp_df.empty:
        st.info("Could not compute feature importance.")
    else:
        topk = st.slider("Show top K drivers", 3, min(20, len(imp_df)), 10)
        top_imp = imp_df.head(topk).copy()
        st.dataframe(top_imp, use_container_width=True)

        # Plot (Altair)
        try:
            import altair as alt
            chart = (
                alt.Chart(top_imp)
                .mark_bar()
                .encode(
                    x=alt.X("importance:Q", title="Importance"),
                    y=alt.Y("feature:N", sort='-x', title="Feature"),
                    tooltip=["feature", "importance"]
                )
                .properties(height=max(300, 20 * len(top_imp)))
            )
            st.altair_chart(chart, use_container_width=True)
        except Exception:
            pass

    st.divider()
    st.subheader("Policy Recommendations")

    if not imp_df.empty:
        # Prepare policy recommendations focusing on strongest drivers
        top_features = list(zip(imp_df["feature"].tolist(), imp_df["importance"].tolist()))
        recs_df = policy_recs_from_features(top_features, k=min(10, len(top_features)))
        st.dataframe(recs_df, use_container_width=True)

        # Export
        colA, colB = st.columns([1,1])
        with colA:
            csv = recs_df.to_csv(index=False).encode("utf-8")
            st.download_button("Download Recommendations (CSV)", csv, file_name="policy_recommendations.csv", mime="text/csv")
        with colB:
            report = {
                "target": target,
                "model": model_name,
                "test_metrics": metrics_test,
                "top_drivers": imp_df.head(10).to_dict(orient="records"),
                "policy_recommendations": recs_df.to_dict(orient="records"),
            }
            st.download_button(
                "Download JSON Report",
                data=json.dumps(report, indent=2).encode("utf-8"),
                file_name="policy_report.json",
                mime="application/json",
            )

    st.divider()
    st.subheader("County-Level Predictions (for targeting)")

    # Fit on full data (rows with complete features/target), then predict to rank counties
    mask = df[features + [target]].notna().all(axis=1)
    df_fit = df.loc[mask].copy()
    if len(df_fit) >= 30:
        X_full = df_fit[features]
        y_full = df_fit[target]
        full_model = fit_model(model_name, X_full, y_full)
        df_fit["Predicted"] = full_model.predict(X_full)
        df_fit["Residual (Actual-Pred)"] = df_fit[target] - df_fit["Predicted"]

        # Define risk direction: if higher target is worse or better
        worse_if_higher = bool(re.search(r"premature|ypll|poor|low birth|obesity|smok|crime|violence|pollution|pm2.5|uninsured|hospital", target, re.I))
        if worse_if_higher:
            df_fit["RiskScore"] = df_fit["Predicted"]
            rank_ascending = False
        else:
            # e.g., Life Expectancy: lower is worse, so invert
            df_fit["RiskScore"] = -df_fit["Predicted"]
            rank_ascending = False

        # Try to show county/state labels
        label_cols = [c for c in ["County", "County Name", "county", "CountyName"] if c in df_fit.columns]
        state_cols = [c for c in ["State", "state", "STATE", "State Name", "StateName"] if c in df_fit.columns]
        show_cols = (label_cols[:1] + state_cols[:1]) or []

        preview_cols = show_cols + [target, "Predicted", "Residual (Actual-Pred)", "RiskScore"]
        preview_cols = [c for c in preview_cols if c in df_fit.columns]

        top_n = st.slider("Show top N highest-need counties", 10, 200, 25, 5)
        ranked = df_fit.sort_values("RiskScore", ascending=rank_ascending).head(top_n)
        st.dataframe(ranked[preview_cols], use_container_width=True)

        st.caption("Interpretation: These counties rank worst by predicted outcome (or lowest predicted Life Expectancy). Use top drivers above to tailor policy actions.")
    else:
        st.info("Not enough complete rows to rank counties. Try fewer features or a different target.")

else:
    st.info("Upload data to begin.")

st.divider()
st.caption("Tip: Start with targets like 'Life Expectancy' or 'Premature death' and include socioeconomic, clinical care, health behaviors, and environment predictors for richer insights.")
