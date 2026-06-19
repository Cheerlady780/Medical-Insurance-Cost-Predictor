"""
================================================================================
 MEDICAL INSURANCE COST PREDICTION
 Comparative Study: Linear Regression (Interpretable) vs XGBoost (High-Performance)
================================================================================

This script performs the full pipeline:
 1. Load & explore data
 2. Feature engineering (age bands, BMI categories, smoker x bmi interaction)
 3. Train Linear Regression (baseline, interpretable)
 4. Train XGBoost (challenger, high-performance) with hyperparameter tuning
 5. Evaluate both with RMSE, MAE, R², 5-fold CV, residual analysis, segmented error
 6. SHAP explainability on the winning model
 7. Save trained models, preprocessing pipeline, and all metrics/plots to disk
================================================================================
"""

import json
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor
import shap

warnings.filterwarnings("ignore")
sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 120

RANDOM_STATE = 42
OUT_DIR = "outputs"
MODEL_DIR = "models"

# ──────────────────────────────────────────────────────────────────────────
# 1. LOAD DATA
# ──────────────────────────────────────────────────────────────────────────
print("=" * 70)
print("STEP 1: Loading data")
print("=" * 70)

df = pd.read_csv("data/insurance.csv")
print(f"Loaded {len(df)} rows, {df.shape[1]} columns")
print(df.head())
print("\nMissing values per column:\n", df.isnull().sum())
print("\nTarget (charges) summary:\n", df["charges"].describe())

# ──────────────────────────────────────────────────────────────────────────
# 2. EDA — save key plots
# ──────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("STEP 2: Exploratory Data Analysis")
print("=" * 70)

fig, axes = plt.subplots(2, 2, figsize=(13, 10))

# Distribution of charges
sns.histplot(df["charges"], kde=True, ax=axes[0, 0], color="#1A6B72")
axes[0, 0].set_title("Distribution of Medical Charges (Right-Skewed)")
axes[0, 0].set_xlabel("Charges ($)")

# Charges by smoker status — THE key chart
sns.boxplot(data=df, x="smoker", y="charges", ax=axes[0, 1], palette=["#27AE8F", "#D4A017"])
axes[0, 1].set_title("Charges by Smoking Status (Primary Cost Driver)")
axes[0, 1].set_xlabel("Smoker")
axes[0, 1].set_ylabel("Charges ($)")

# BMI vs charges, colored by smoker — shows the interaction visually
sns.scatterplot(data=df, x="bmi", y="charges", hue="smoker", ax=axes[1, 0],
                 palette={"yes": "#D4A017", "no": "#27AE8F"}, alpha=0.6)
axes[1, 0].set_title("BMI vs Charges, Split by Smoker Status\n(Visualises the Smoker x BMI Interaction)")
axes[1, 0].set_xlabel("BMI")
axes[1, 0].set_ylabel("Charges ($)")

# Correlation heatmap (numeric only)
corr = df[["age", "bmi", "children", "charges"]].corr()
sns.heatmap(corr, annot=True, cmap="YlGnBu", ax=axes[1, 1], fmt=".2f")
axes[1, 1].set_title("Correlation Heatmap (Numeric Features)")

plt.tight_layout()
plt.savefig(f"{OUT_DIR}/eda_overview.png", bbox_inches="tight")
plt.close()
print(f"Saved EDA overview -> {OUT_DIR}/eda_overview.png")

# Quantify the smoker effect explicitly
smoker_avg = df.groupby("smoker")["charges"].mean()
print(f"\nAverage charges — Non-smoker: ${smoker_avg['no']:,.2f}")
print(f"Average charges — Smoker:     ${smoker_avg['yes']:,.2f}")
print(f"Smoker premium multiplier:    {smoker_avg['yes'] / smoker_avg['no']:.2f}x")

# ──────────────────────────────────────────────────────────────────────────
# 3. FEATURE ENGINEERING
# ──────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("STEP 3: Feature Engineering")
print("=" * 70)

data = df.copy()

# Binary encode smoker and sex
data["smoker_flag"] = (data["smoker"] == "yes").astype(int)
data["sex_flag"] = (data["sex"] == "male").astype(int)

# BMI category (WHO standard cutoffs)
def bmi_category(bmi):
    if bmi < 18.5:
        return "underweight"
    elif bmi < 25:
        return "normal"
    elif bmi < 30:
        return "overweight"
    else:
        return "obese"

data["bmi_category"] = data["bmi"].apply(bmi_category)

# Age band (actuarial-style banding)
def age_band(age):
    if age <= 25:
        return "18-25"
    elif age <= 35:
        return "26-35"
    elif age <= 50:
        return "36-50"
    else:
        return "51+"

data["age_band"] = data["age"].apply(age_band)

# THE key engineered feature: smoker x bmi interaction
data["smoker_x_bmi"] = data["smoker_flag"] * data["bmi"]

# One-hot encode categorical columns
data_encoded = pd.get_dummies(
    data,
    columns=["region", "bmi_category", "age_band"],
    drop_first=True
)

# Final feature set
feature_cols = [c for c in data_encoded.columns if c not in
                ["charges", "sex", "smoker"]]
X = data_encoded[feature_cols]
y = data_encoded["charges"]

print(f"Final feature set ({len(feature_cols)} features):")
for c in feature_cols:
    print(f"  - {c}")

# Save the exact feature list + bmi/age band functions logic for inference later
feature_metadata = {
    "feature_cols": feature_cols,
    "bmi_bins": [0, 18.5, 25, 30, 100],
    "bmi_labels": ["underweight", "normal", "overweight", "obese"],
    "age_bins": [0, 25, 35, 50, 120],
    "age_labels": ["18-25", "26-35", "36-50", "51+"],
}
with open(f"{MODEL_DIR}/feature_metadata.json", "w") as f:
    json.dump(feature_metadata, f, indent=2)

# ──────────────────────────────────────────────────────────────────────────
# 4. TRAIN / TEST SPLIT
# ──────────────────────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE
)
print(f"\nTrain size: {len(X_train)} | Test size: {len(X_test)}")

# ──────────────────────────────────────────────────────────────────────────
# 5. MODEL 1 — LINEAR REGRESSION (interpretable baseline)
# ──────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("STEP 5: Training Linear Regression (Interpretable Baseline)")
print("=" * 70)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

lr_model = LinearRegression()
lr_model.fit(X_train_scaled, y_train)
lr_pred = lr_model.predict(X_test_scaled)

lr_rmse = np.sqrt(mean_squared_error(y_test, lr_pred))
lr_mae = mean_absolute_error(y_test, lr_pred)
lr_r2 = r2_score(y_test, lr_pred)

# 5-fold cross-validation for robustness
kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
lr_cv_scores = cross_val_score(lr_model, scaler.fit_transform(X), y, cv=kf, scoring="r2")

print(f"Linear Regression — Test RMSE: ${lr_rmse:,.2f}")
print(f"Linear Regression — Test MAE:  ${lr_mae:,.2f}")
print(f"Linear Regression — Test R²:   {lr_r2:.4f}")
print(f"Linear Regression — 5-Fold CV R² (mean ± std): {lr_cv_scores.mean():.4f} ± {lr_cv_scores.std():.4f}")

# Coefficient interpretation
coef_df = pd.DataFrame({
    "feature": feature_cols,
    "coefficient": lr_model.coef_
}).sort_values("coefficient", key=abs, ascending=False)
print("\nTop Linear Regression coefficients (standardised):")
print(coef_df.head(8).to_string(index=False))

# ──────────────────────────────────────────────────────────────────────────
# 6. MODEL 2 — XGBOOST (high-performance challenger) with tuning
# ──────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("STEP 6: Training XGBoost (High-Performance Challenger)")
print("=" * 70)

param_grid = {
    "n_estimators": [100, 200, 300],
    "max_depth": [2, 3, 4],
    "learning_rate": [0.03, 0.05, 0.1],
    "subsample": [0.8, 1.0],
}

xgb_base = XGBRegressor(random_state=RANDOM_STATE, objective="reg:squarederror")
grid_search = GridSearchCV(
    xgb_base, param_grid, cv=5, scoring="neg_root_mean_squared_error",
    n_jobs=-1, verbose=0
)
grid_search.fit(X_train, y_train)

xgb_model = grid_search.best_estimator_
print(f"Best XGBoost params: {grid_search.best_params_}")

xgb_pred = xgb_model.predict(X_test)

xgb_rmse = np.sqrt(mean_squared_error(y_test, xgb_pred))
xgb_mae = mean_absolute_error(y_test, xgb_pred)
xgb_r2 = r2_score(y_test, xgb_pred)

xgb_cv_scores = cross_val_score(xgb_model, X, y, cv=kf, scoring="r2")

print(f"XGBoost — Test RMSE: ${xgb_rmse:,.2f}")
print(f"XGBoost — Test MAE:  ${xgb_mae:,.2f}")
print(f"XGBoost — Test R²:   {xgb_r2:.4f}")
print(f"XGBoost — 5-Fold CV R² (mean ± std): {xgb_cv_scores.mean():.4f} ± {xgb_cv_scores.std():.4f}")

# ──────────────────────────────────────────────────────────────────────────
# 7. MODEL COMPARISON
# ──────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("STEP 7: Model Comparison")
print("=" * 70)

comparison = pd.DataFrame({
    "Model": ["Linear Regression", "XGBoost"],
    "Test RMSE ($)": [lr_rmse, xgb_rmse],
    "Test MAE ($)": [lr_mae, xgb_mae],
    "Test R²": [lr_r2, xgb_r2],
    "CV R² (mean)": [lr_cv_scores.mean(), xgb_cv_scores.mean()],
    "CV R² (std)": [lr_cv_scores.std(), xgb_cv_scores.std()],
})
print(comparison.to_string(index=False))
comparison.to_csv(f"{OUT_DIR}/model_comparison.csv", index=False)

winner = "XGBoost" if xgb_rmse < lr_rmse else "Linear Regression"
print(f"\n>>> WINNER (by Test RMSE): {winner}")

# ──────────────────────────────────────────────────────────────────────────
# 8. SEGMENTED ERROR ANALYSIS (smoker vs non-smoker)
# ──────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("STEP 8: Segmented Error Analysis (Smoker vs Non-Smoker)")
print("=" * 70)

test_idx = X_test.index
smoker_test = data.loc[test_idx, "smoker"]

seg_results = []
for label, pred, name in [(lr_pred, lr_pred, "Linear Regression"), (xgb_pred, xgb_pred, "XGBoost")]:
    for grp in ["yes", "no"]:
        mask = (smoker_test == grp).values
        rmse_grp = np.sqrt(mean_squared_error(y_test[mask], pred[mask]))
        seg_results.append({"Model": name, "Smoker": grp, "RMSE ($)": rmse_grp, "N": mask.sum()})

seg_df = pd.DataFrame(seg_results)
print(seg_df.to_string(index=False))
seg_df.to_csv(f"{OUT_DIR}/segmented_error.csv", index=False)

# ──────────────────────────────────────────────────────────────────────────
# 9. RESIDUAL PLOTS — visual proof of non-linearity
# ──────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("STEP 9: Residual Analysis")
print("=" * 70)

fig, axes = plt.subplots(2, 2, figsize=(13, 10))

# Linear Regression: predicted vs actual
axes[0, 0].scatter(y_test, lr_pred, alpha=0.5, color="#1A6B72")
axes[0, 0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--")
axes[0, 0].set_title("Linear Regression: Predicted vs Actual")
axes[0, 0].set_xlabel("Actual Charges ($)")
axes[0, 0].set_ylabel("Predicted Charges ($)")

# Linear Regression: residuals
lr_resid = y_test - lr_pred
axes[0, 1].scatter(lr_pred, lr_resid, alpha=0.5, color="#1A6B72")
axes[0, 1].axhline(0, color="r", linestyle="--")
axes[0, 1].set_title("Linear Regression: Residual Plot\n(Funnel/curve shape = non-linearity present)")
axes[0, 1].set_xlabel("Predicted Charges ($)")
axes[0, 1].set_ylabel("Residual ($)")

# XGBoost: predicted vs actual
axes[1, 0].scatter(y_test, xgb_pred, alpha=0.5, color="#D4A017")
axes[1, 0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--")
axes[1, 0].set_title("XGBoost: Predicted vs Actual")
axes[1, 0].set_xlabel("Actual Charges ($)")
axes[1, 0].set_ylabel("Predicted Charges ($)")

# XGBoost: residuals
xgb_resid = y_test - xgb_pred
axes[1, 1].scatter(xgb_pred, xgb_resid, alpha=0.5, color="#D4A017")
axes[1, 1].axhline(0, color="r", linestyle="--")
axes[1, 1].set_title("XGBoost: Residual Plot")
axes[1, 1].set_xlabel("Predicted Charges ($)")
axes[1, 1].set_ylabel("Residual ($)")

plt.tight_layout()
plt.savefig(f"{OUT_DIR}/residual_analysis.png", bbox_inches="tight")
plt.close()
print(f"Saved residual analysis -> {OUT_DIR}/residual_analysis.png")

# ──────────────────────────────────────────────────────────────────────────
# 10. SHAP EXPLAINABILITY (on the winning / more complex model)
# ──────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("STEP 10: SHAP Explainability (XGBoost)")
print("=" * 70)

explainer = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(X_test)

plt.figure(figsize=(10, 7))
shap.summary_plot(shap_values, X_test, show=False)
plt.title("SHAP Feature Importance — XGBoost", fontsize=13)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/shap_summary.png", bbox_inches="tight", dpi=120)
plt.close()
print(f"Saved SHAP summary plot -> {OUT_DIR}/shap_summary.png")

# Mean absolute SHAP value per feature (global importance table)
mean_abs_shap = np.abs(shap_values).mean(axis=0)
shap_importance = pd.DataFrame({
    "feature": feature_cols,
    "mean_abs_shap": mean_abs_shap
}).sort_values("mean_abs_shap", ascending=False)
print("\nGlobal SHAP feature importance:")
print(shap_importance.head(8).to_string(index=False))
shap_importance.to_csv(f"{OUT_DIR}/shap_importance.csv", index=False)

# ──────────────────────────────────────────────────────────────────────────
# 11. SAVE EVERYTHING NEEDED FOR INFERENCE
# ──────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("STEP 11: Saving Models & Artifacts")
print("=" * 70)

joblib.dump(lr_model, f"{MODEL_DIR}/linear_regression_model.joblib")
joblib.dump(xgb_model, f"{MODEL_DIR}/xgboost_model.joblib")
joblib.dump(scaler, f"{MODEL_DIR}/scaler.joblib")

final_metrics = {
    "linear_regression": {
        "test_rmse": float(lr_rmse),
        "test_mae": float(lr_mae),
        "test_r2": float(lr_r2),
        "cv_r2_mean": float(lr_cv_scores.mean()),
        "cv_r2_std": float(lr_cv_scores.std()),
    },
    "xgboost": {
        "test_rmse": float(xgb_rmse),
        "test_mae": float(xgb_mae),
        "test_r2": float(xgb_r2),
        "cv_r2_mean": float(xgb_cv_scores.mean()),
        "cv_r2_std": float(xgb_cv_scores.std()),
        "best_params": grid_search.best_params_,
    },
    "winner": winner,
    "smoker_avg_charges": {
        "non_smoker": float(smoker_avg["no"]),
        "smoker": float(smoker_avg["yes"]),
        "multiplier": float(smoker_avg["yes"] / smoker_avg["no"]),
    },
}
with open(f"{OUT_DIR}/final_metrics.json", "w") as f:
    json.dump(final_metrics, f, indent=2)

print("\nAll models and artifacts saved successfully:")
print(f"  - {MODEL_DIR}/linear_regression_model.joblib")
print(f"  - {MODEL_DIR}/xgboost_model.joblib")
print(f"  - {MODEL_DIR}/scaler.joblib")
print(f"  - {MODEL_DIR}/feature_metadata.json")
print(f"  - {OUT_DIR}/final_metrics.json")
print(f"  - {OUT_DIR}/model_comparison.csv")
print(f"  - {OUT_DIR}/segmented_error.csv")
print(f"  - {OUT_DIR}/eda_overview.png")
print(f"  - {OUT_DIR}/residual_analysis.png")
print(f"  - {OUT_DIR}/shap_summary.png")

print("\n" + "=" * 70)
print(f"PIPELINE COMPLETE. WINNING MODEL: {winner}")
print("=" * 70)
