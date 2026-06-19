"""
================================================================================
MEDICAL INSURANCE COST PREDICTOR — Interactive Streamlit App
================================================================================
Interpretable vs High-Performance Risk Pricing: Linear Regression vs XGBoost

Run with:  streamlit run app/app.py
================================================================================
"""

import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

# ──────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Insurance Cost Predictor | Linear vs XGBoost",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────────────────
# CUSTOM CSS — visual polish
# ──────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #F7F9FA; }
    .stApp { font-family: 'Helvetica Neue', sans-serif; }

    .hero-banner {
        background: linear-gradient(135deg, #0D1B2A 0%, #1A6B72 100%);
        padding: 2rem 2.5rem;
        border-radius: 14px;
        color: white;
        margin-bottom: 1.5rem;
    }
    .hero-banner h1 { margin: 0; font-size: 2rem; font-weight: 700; }
    .hero-banner p { margin: 0.4rem 0 0 0; color: #B2DFDB; font-size: 0.95rem; }

    .metric-card {
        background: white;
        border-radius: 12px;
        padding: 1.2rem 1.4rem;
        box-shadow: 0 2px 10px rgba(0,0,0,0.06);
        border-left: 4px solid #27AE8F;
    }
    .metric-card h3 { margin: 0; font-size: 0.8rem; color: #777; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px;}
    .metric-card .value { font-size: 1.8rem; font-weight: 700; color: #0D1B2A; margin-top: 0.2rem;}

    .winner-badge {
        background: linear-gradient(135deg, #27AE8F, #1A6B72);
        color: white;
        padding: 0.3rem 0.9rem;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 700;
        display: inline-block;
    }

    .prediction-box {
        background: linear-gradient(135deg, #1A6B72 0%, #0D1B2A 100%);
        border-radius: 16px;
        padding: 2rem;
        color: white;
        text-align: center;
    }
    .prediction-box .amount { font-size: 3rem; font-weight: 800; margin: 0.5rem 0; }
    .prediction-box .label { font-size: 0.9rem; color: #B2DFDB; text-transform: uppercase; letter-spacing: 1px; }

    .insight-box {
        background: #FFF8E1;
        border: 1px solid #D4A017;
        border-radius: 10px;
        padding: 1rem 1.2rem;
        font-size: 0.9rem;
        color: #5D3A00;
    }
    section[data-testid="stSidebar"] {
        background-color: #0D1B2A;
    }
    section[data-testid="stSidebar"] * {
        color: #EAF4F4 !important;
    }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────
# LOAD MODELS & METADATA (cached)
# ──────────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    lr_model = joblib.load("models/linear_regression_model.joblib")
    xgb_model = joblib.load("models/xgboost_model.joblib")
    scaler = joblib.load("models/scaler.joblib")
    with open("models/feature_metadata.json") as f:
        meta = json.load(f)
    with open("outputs/final_metrics.json") as f:
        metrics = json.load(f)
    df_raw = pd.read_csv("data/insurance.csv")
    return lr_model, xgb_model, scaler, meta, metrics, df_raw

lr_model, xgb_model, scaler, meta, metrics, df_raw = load_artifacts()
feature_cols = meta["feature_cols"]

# ──────────────────────────────────────────────────────────────────────────
# FEATURE ENGINEERING FUNCTION (mirrors training pipeline exactly)
# ──────────────────────────────────────────────────────────────────────────
def engineer_features(age, sex, bmi, children, smoker, region):
    smoker_flag = 1 if smoker == "yes" else 0
    sex_flag = 1 if sex == "male" else 0
    smoker_x_bmi = smoker_flag * bmi

    if bmi < 18.5:
        bmi_cat = "underweight"
    elif bmi < 25:
        bmi_cat = "normal"
    elif bmi < 30:
        bmi_cat = "overweight"
    else:
        bmi_cat = "obese"

    if age <= 25:
        age_band = "18-25"
    elif age <= 35:
        age_band = "26-35"
    elif age <= 50:
        age_band = "36-50"
    else:
        age_band = "51+"

    row = {col: 0 for col in feature_cols}
    row["age"] = age
    row["bmi"] = bmi
    row["children"] = children
    row["smoker_flag"] = smoker_flag
    row["sex_flag"] = sex_flag
    row["smoker_x_bmi"] = smoker_x_bmi

    region_col = f"region_{region}"
    if region_col in row:
        row[region_col] = 1

    bmi_col = f"bmi_category_{bmi_cat}"
    if bmi_col in row:
        row[bmi_col] = 1

    age_col = f"age_band_{age_band}"
    if age_col in row:
        row[age_col] = 1

    return pd.DataFrame([row])[feature_cols], bmi_cat, age_band


# ──────────────────────────────────────────────────────────────────────────
# HERO BANNER
# ──────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-banner">
    <h1>🏥 Medical Insurance Cost Predictor</h1>
    <p>Interpretable (Linear Regression) vs High-Performance (XGBoost) Risk Pricing — A Comparative Study</p>
</div>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────
# SIDEBAR — INPUT CONTROLS
# ──────────────────────────────────────────────────────────────────────────
st.sidebar.markdown("## 👤 Applicant Profile")
st.sidebar.markdown("Adjust the sliders to simulate a policy applicant.")

age = st.sidebar.slider("Age", 18, 64, 30)
sex = st.sidebar.selectbox("Sex", ["male", "female"])
bmi = st.sidebar.slider("BMI (Body Mass Index)", 15.0, 53.0, 26.0, step=0.1)
children = st.sidebar.slider("Number of Children / Dependents", 0, 5, 0)
smoker = st.sidebar.radio("Smoker?", ["no", "yes"], horizontal=True)
region = st.sidebar.selectbox("Region", ["northeast", "northwest", "southeast", "southwest"])

st.sidebar.markdown("---")
model_choice = st.sidebar.radio(
    "🔍 Model to Use for Prediction",
    ["XGBoost (Recommended — Best Accuracy)", "Linear Regression (Interpretable)"],
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    f"""
    <div style="font-size:0.78rem; color:#9FC9C9; line-height:1.5;">
    <b>Dataset:</b> 1,338 US insurance policyholders<br>
    <b>Winning Model:</b> {metrics['winner']}<br>
    <b>Smoker cost multiplier:</b> {metrics['smoker_avg_charges']['multiplier']:.2f}x
    </div>
    """,
    unsafe_allow_html=True,
)

# ──────────────────────────────────────────────────────────────────────────
# RUN INFERENCE
# ──────────────────────────────────────────────────────────────────────────
X_input, bmi_cat, age_band = engineer_features(age, sex, bmi, children, smoker, region)

if "Linear" in model_choice:
    X_input_scaled = scaler.transform(X_input)
    prediction = lr_model.predict(X_input_scaled)[0]
    active_model = "Linear Regression"
else:
    prediction = xgb_model.predict(X_input)[0]
    active_model = "XGBoost"

prediction = max(prediction, 0)  # guard against negative predictions

# Also compute the other model's prediction for comparison
X_input_scaled = scaler.transform(X_input)
lr_pred_compare = max(lr_model.predict(X_input_scaled)[0], 0)
xgb_pred_compare = max(xgb_model.predict(X_input)[0], 0)

# ──────────────────────────────────────────────────────────────────────────
# MAIN LAYOUT — PREDICTION + CONTEXT
# ──────────────────────────────────────────────────────────────────────────
col_left, col_right = st.columns([1, 1.3])

with col_left:
    st.markdown(f"""
    <div class="prediction-box">
        <div class="label">Predicted Annual Charges ({active_model})</div>
        <div class="amount">${prediction:,.0f}</div>
        <div class="label">BMI Category: {bmi_cat.title()} &nbsp;|&nbsp; Age Band: {age_band}</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Side-by-side model comparison for this specific applicant
    comp_fig = go.Figure()
    comp_fig.add_trace(go.Bar(
        x=["Linear Regression", "XGBoost"],
        y=[lr_pred_compare, xgb_pred_compare],
        marker_color=["#1A6B72", "#D4A017"],
        text=[f"${lr_pred_compare:,.0f}", f"${xgb_pred_compare:,.0f}"],
        textposition="outside",
    ))
    comp_fig.update_layout(
        title="Prediction Comparison for This Applicant",
        yaxis_title="Predicted Charges ($)",
        height=320,
        margin=dict(t=50, b=20, l=20, r=20),
        plot_bgcolor="white",
    )
    st.plotly_chart(comp_fig, use_container_width=True)

    if smoker == "yes" and bmi >= 30:
        st.markdown("""
        <div class="insight-box">
        ⚠️ <b>High-Risk Interaction Detected:</b> This applicant is both a smoker and has an
        obese-range BMI. Our analysis shows this specific combination is the single strongest
        cost driver in the dataset — <b>far more than either factor alone</b>. This is captured
        explicitly via the engineered <code>smoker × bmi</code> feature.
        </div>
        """, unsafe_allow_html=True)
    elif smoker == "yes":
        st.markdown("""
        <div class="insight-box">
        🚬 <b>Smoker Status Detected:</b> Smoking status alone is associated with roughly a
        3.8x increase in average charges across this dataset.
        </div>
        """, unsafe_allow_html=True)

with col_right:
    st.markdown("#### 📊 Where This Applicant Falls in the Population")

    fig = px.scatter(
        df_raw, x="bmi", y="charges", color="smoker",
        color_discrete_map={"yes": "#D4A017", "no": "#27AE8F"},
        opacity=0.45,
        labels={"bmi": "BMI", "charges": "Charges ($)", "smoker": "Smoker"},
    )
    fig.add_trace(go.Scatter(
        x=[bmi], y=[prediction],
        mode="markers",
        marker=dict(size=18, color="#0D1B2A", symbol="star", line=dict(width=2, color="white")),
        name="This Applicant (Predicted)",
    ))
    fig.update_layout(
        height=420,
        plot_bgcolor="white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        margin=dict(t=30, b=20, l=20, r=20),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### 🧬 Feature Contribution Snapshot")
    feat_summary = pd.DataFrame({
        "Feature": ["Age", "BMI", "Smoker × BMI Interaction", "Children", "Region"],
        "Value": [age, f"{bmi:.1f}", f"{bmi if smoker=='yes' else 0:.1f}", children, region.title()],
    })
    st.table(feat_summary.set_index("Feature"))

st.markdown("---")

# ──────────────────────────────────────────────────────────────────────────
# MODEL PERFORMANCE TABS
# ──────────────────────────────────────────────────────────────────────────
st.markdown("## 📈 Model Performance & Comparison")

tab1, tab2, tab3, tab4 = st.tabs([
    "🏆 Model Comparison", "📉 Residual Analysis", "🔬 SHAP Explainability", "🗃️ Dataset Overview"
])

with tab1:
    c1, c2, c3, c4 = st.columns(4)
    lr_m = metrics["linear_regression"]
    xgb_m = metrics["xgboost"]

    with c1:
        st.markdown(f"""
        <div class="metric-card">
            <h3>Linear Regression — Test R²</h3>
            <div class="value">{lr_m['test_r2']:.3f}</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="metric-card">
            <h3>XGBoost — Test R²</h3>
            <div class="value">{xgb_m['test_r2']:.3f}</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class="metric-card">
            <h3>Linear Regression — RMSE</h3>
            <div class="value">${lr_m['test_rmse']:,.0f}</div>
        </div>""", unsafe_allow_html=True)
    with c4:
        st.markdown(f"""
        <div class="metric-card">
            <h3>XGBoost — RMSE</h3>
            <div class="value">${xgb_m['test_rmse']:,.0f}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown(f"<br><span class='winner-badge'>🏆 WINNER: {metrics['winner']}</span> — selected by lowest Test RMSE", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    metrics_compare_df = pd.DataFrame({
        "Metric": ["Test RMSE ($)", "Test MAE ($)", "Test R²", "5-Fold CV R² (mean)", "5-Fold CV R² (std)"],
        "Linear Regression": [
            f"${lr_m['test_rmse']:,.2f}", f"${lr_m['test_mae']:,.2f}",
            f"{lr_m['test_r2']:.4f}", f"{lr_m['cv_r2_mean']:.4f}", f"{lr_m['cv_r2_std']:.4f}"
        ],
        "XGBoost": [
            f"${xgb_m['test_rmse']:,.2f}", f"${xgb_m['test_mae']:,.2f}",
            f"{xgb_m['test_r2']:.4f}", f"{xgb_m['cv_r2_mean']:.4f}", f"{xgb_m['cv_r2_std']:.4f}"
        ],
    })
    st.table(metrics_compare_df.set_index("Metric"))

    st.markdown("""
    <div class="insight-box">
    💡 <b>Why XGBoost wins, but by a narrower margin than typical:</b> Engineering the
    <code>smoker × bmi</code> interaction term gave Linear Regression access to the same
    non-linear relationship XGBoost discovers automatically through tree splits. This is why
    the gap between the two models here is smaller than in a naive feature setup — it
    demonstrates that <i>some</i> of XGBoost's advantage comes from a learnable, explainable
    interaction rather than an unexplainable black-box pattern.
    </div>
    """, unsafe_allow_html=True)

with tab2:
    st.image("outputs/residual_analysis.png", use_container_width=True)
    st.markdown("""
    <div class="insight-box">
    📌 <b>Reading this chart:</b> Linear Regression's residual plot shows a wider spread and
    a faint curve/funnel pattern — visual evidence of non-linearity it can't fully capture.
    XGBoost's residuals cluster tighter around zero, particularly for high-cost cases.
    </div>
    """, unsafe_allow_html=True)

with tab3:
    st.image("outputs/shap_summary.png", use_container_width=True)
    st.markdown("""
    <div class="insight-box">
    📌 <b>Reading this chart:</b> Each dot is one applicant in the test set. Pink = high feature
    value, blue = low. <code>smoker_x_bmi</code> dominates: high values (smokers with high BMI,
    pink, far right) push predicted cost dramatically upward, while low values (non-smokers,
    blue, clustered left) have minimal impact — exactly confirming the central hypothesis of
    this project.
    </div>
    """, unsafe_allow_html=True)

with tab4:
    st.markdown("#### Sample of Raw Training Data")
    st.dataframe(df_raw.head(10), use_container_width=True)
    st.image("outputs/eda_overview.png", use_container_width=True)

# ──────────────────────────────────────────────────────────────────────────
# FOOTER
# ──────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style="text-align:center; color:#888; font-size:0.8rem; padding: 1rem 0;">
Medical Insurance Cost Prediction — Term Project | Built with Streamlit, Scikit-learn, XGBoost & SHAP<br>
Dataset: Medical Cost Personal Dataset (1,338 records) | For academic demonstration purposes only — not actuarial advice.
</div>
""", unsafe_allow_html=True)
