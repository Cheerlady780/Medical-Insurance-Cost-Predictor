# 🏥 Medical Insurance Cost Prediction
### Interpretable vs. High-Performance Risk Pricing: A Comparative Study of Linear Regression and XGBoost

---

## 1. Project Overview

Insurance companies must predict how much a policyholder will cost in medical
claims *before* issuing a policy. This is the core problem behind health
insurance pricing, and it sits at the center of a real industry tension:

- **Interpretable models** (Linear Regression / GLMs) are the historical
  industry standard because regulators and customers can be shown exactly
  *why* a price was set. This matters for compliance with consumer protection
  and anti-discrimination law.
- **High-performance models** (gradient boosting, neural nets) used by modern
  insurtechs (Lemonade, Root, etc.) are more accurate because they can capture
  non-linear interactions — but they are harder to explain.

**This project investigates: how much accuracy do you actually give up by
insisting on an interpretable model — and can smart feature engineering close
that gap?**

To answer this, we built and compared two models on the same dataset:

| Model | Role |
|---|---|
| **Linear Regression** | Represents the interpretable, regulator-friendly status quo |
| **XGBoost** | Represents the modern, high-performance challenger |

The key engineered feature — `smoker × bmi` (an interaction term) — is what
makes this comparison meaningful rather than a generic tutorial exercise. See
Section 5 for why.

---

## 2. Dataset

**Source:** Medical Cost Personal Dataset (a widely used open dataset for
insurance cost modelling; originally distributed alongside *Machine Learning
with R* by Brett Lantz, mirrored publicly on GitHub).

- **Rows:** 1,338 individual policyholder records
- **Columns:** 7 (6 features + 1 target)

| Column | Type | Description |
|---|---|---|
| `age` | numeric | Age of primary beneficiary |
| `sex` | categorical | male / female |
| `bmi` | numeric | Body Mass Index |
| `children` | numeric | Number of dependents covered |
| `smoker` | categorical | yes / no |
| `region` | categorical | US region: northeast, northwest, southeast, southwest |
| `charges` | numeric | **Target** — individual medical costs billed by insurance ($) |

No missing values. No data cleaning required beyond the engineering described
below.

---

## 3. Methodology / Workflow

1. **Exploratory Data Analysis** — distribution of charges, smoker vs.
   non-smoker cost comparison, BMI-vs-charges scatter split by smoker status,
   correlation heatmap.
2. **Feature Engineering**
   - `smoker_flag`, `sex_flag` — binary encodings
   - `bmi_category` — WHO-standard bands (underweight / normal / overweight / obese)
   - `age_band` — actuarial-style age banding (18-25 / 26-35 / 36-50 / 51+)
   - **`smoker_x_bmi`** — the interaction term (see Section 5)
   - One-hot encoding for `region`, `bmi_category`, `age_band`
3. **Train/Test Split** — 80/20, fixed random seed for reproducibility
4. **Model 1: Linear Regression** — trained on standardized features
5. **Model 2: XGBoost** — hyperparameter-tuned via 5-fold `GridSearchCV` over
   `n_estimators`, `max_depth`, `learning_rate`, `subsample`
6. **Evaluation**
   - Test set: RMSE, MAE, R²
   - 5-fold cross-validation R² (mean ± std) for robustness beyond a single split
   - **Residual analysis** — visual diagnostic for non-linearity
   - **Segmented error analysis** — RMSE reported separately for smokers vs.
     non-smokers, since a single blended metric can hide where a model
     actually struggles
7. **Explainability** — SHAP applied to the XGBoost model to recover
   interpretability that is normally Linear Regression's only advantage
8. **Model Selection** — lower Test RMSE wins
9. **Artifact Saving** — both trained models, the scaler, and all metrics/plots
   saved to disk for reuse in the Streamlit app and `predict.py`

All of this is implemented in [`train_model.py`](train_model.py), fully
commented section-by-section in the order above.

---

## 4. Results

*(Figures below are from the reference run included in this package —
`outputs/final_metrics.json` has the exact numbers; your own re-run with
`python train_model.py` will reproduce the same results since the random
seed is fixed.)*

| Metric | Linear Regression | XGBoost |
|---|---|---|
| Test RMSE ($) | $4,561.88 | **$4,351.43** |
| Test MAE ($) | $2,749.35 | **$2,510.05** |
| Test R² | 0.8660 | **0.8780** |
| 5-Fold CV R² (mean ± std) | 0.8416 ± 0.0343 | **0.8564 ± 0.0320** |

**🏆 Winner: XGBoost** — by lowest Test RMSE.

**Segmented error (RMSE by smoker status):**

| Model | Smokers | Non-Smokers |
|---|---|---|
| Linear Regression | $5,050.96 | $4,429.95 |
| XGBoost | $4,665.75 | $4,268.46 |

Both models find smokers harder to predict accurately than non-smokers — a
genuine, honestly-reported limitation (see Section 7).

**Smoker cost multiplier:** Smokers cost **3.80×** more on average than
non-smokers in this dataset ($32,050 vs. $8,434 average charges).

---

## 5. The Central Finding: Why `smoker × bmi` Matters

If you give a Linear Regression model `smoker` and `bmi` as two *separate*
columns, it is mathematically forced to treat their effects as independent
and additive: being a smoker adds a fixed dollar amount, and having a higher
BMI adds a fixed (different) dollar amount, regardless of one another.

That is not how health risk actually works. A smoker with obesity is not
simply "risky + risky" — the two conditions compound (cardiovascular strain,
reduced lung capacity worsening weight-related complications, etc.), so the
real cost curve jumps multiplicatively for that specific combination.

XGBoost finds this automatically through tree splitting (it can ask "is this
person a smoker AND is their BMI > 30?" as a single decision path). Linear
Regression cannot represent this unless the interaction is handed to it
explicitly — which is exactly what the `smoker_x_bmi` feature does.

**This is confirmed empirically in this project**, not just asserted:

- `smoker_x_bmi` is the **#1 ranked feature by SHAP importance** in the
  XGBoost model (see `outputs/shap_summary.png`)
- It is also the **largest-magnitude coefficient** in the Linear Regression
  model (see console output of `train_model.py`, Step 5)
- Adding this single engineered feature is a meaningful part of why the
  accuracy gap between the "simple" and "complex" model is narrower here than
  it would be with a naive feature set — directly supporting the project's
  central thesis that **some of a black-box model's advantage is actually a
  learnable, explainable interaction, not an unexplainable pattern.**

---

## 6. Tech Stack

| Purpose | Library |
|---|---|
| Data handling | pandas, numpy |
| Modelling | scikit-learn (Linear Regression, preprocessing, CV), xgboost |
| Explainability | shap |
| Visualization (offline plots) | matplotlib, seaborn |
| Visualization (interactive app) | plotly |
| Model persistence | joblib |
| Interactive app | streamlit |

No GPU required. No internet connection required after initial setup (other
than installing packages). Trains in well under a minute on a standard laptop CPU.

---

## 7. Limitations & Honest Caveats

- **Small sample size** (1,338 rows) limits how much complexity can be
  justified — this is why model comparison was deliberately kept to two
  models rather than an ensemble of many.
- **US-only data, 4 broad regions** — no granular geographic cost-of-living
  data (e.g., zip-code level), which real insurers do use.
- **No pre-existing conditions, no family medical history, no income data** —
  all known real-world pricing factors absent from this dataset by design.
- **Higher error for smokers specifically** — both models are less accurate
  for this subgroup (see Section 4), likely because smoker-related costs have
  higher inherent variance (some smokers develop serious illness, many don't,
  and the dataset can't distinguish why).
- **Not actuarial advice** — this is an academic/portfolio demonstration, not
  a production pricing model, and should not be used to set real insurance
  premiums.

---

## 8. Folder Structure

```
insurance_project/
├── README.md                          <- you are here
├── requirements.txt                   <- exact dependency versions
├── check_setup.py                     <- run this if anything seems missing
├── train_model.py                     <- full training pipeline (run this first)
├── predict.py                         <- standalone CLI inference script
├── data/
│   └── insurance.csv                  <- the dataset
├── models/                             <- created by train_model.py
│   ├── linear_regression_model.joblib
│   ├── xgboost_model.joblib
│   ├── scaler.joblib
│   └── feature_metadata.json
├── outputs/                            <- created by train_model.py
│   ├── eda_overview.png
│   ├── residual_analysis.png
│   ├── shap_summary.png
│   ├── shap_importance.csv
│   ├── model_comparison.csv
│   ├── segmented_error.csv
│   ├── final_metrics.json
│   └── app_screenshot_*.png            <- reference screenshots of the app
└── app/
    └── app.py                          <- Streamlit interactive app
```

---

## 9. Installation & Setup

### Prerequisites
- Python 3.9 – 3.12 installed
- ~200 MB free disk space
- No GPU needed

### Step-by-step

**1. Extract this folder** and open a terminal inside it.

**2. (Strongly recommended) Create a virtual environment:**

```bash
python3 -m venv venv

# Activate it:
# On macOS/Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate
```

**3. Install dependencies — use this exact pattern, not plain `pip install`:**

```bash
python -m pip install -r requirements.txt
```

> ⚠️ **Why `python -m pip` instead of just `pip`:** On many systems (especially
> Windows and Mac with multiple Python versions installed), the `pip` command
> on its own can silently point to a *different* Python installation than the
> `python` command you use to actually run the scripts. This causes the
> single most common error people hit with any Python project:
> `ModuleNotFoundError: No module named 'matplotlib'` (or any other package)
> **even though the install appeared to succeed.** Using `python -m pip`
> guarantees the packages are installed into the *same* Python that will run
> the code.

This installs pandas, numpy, scikit-learn, xgboost, shap, matplotlib, seaborn,
joblib, streamlit, and plotly. Takes roughly 1–3 minutes depending on your
connection.

**4. Verify the install before doing anything else:**

```bash
python check_setup.py
```

This prints exactly which Python interpreter is active and checks every
required package against it. If anything is missing, it tells you the exact
fix command for your machine. **Do not skip this step** — it takes 2 seconds
and prevents the most common setup issue from wasting your time later.

---

## 10. How to Run

### A. Train the models (do this first)

```bash
python train_model.py
```

This will:
- Load and explore `data/insurance.csv`
- Engineer all features described in Section 3
- Train and tune both models
- Print a full console report (EDA stats, coefficients, metrics, SHAP importances)
- Save everything needed for inference into `models/` and `outputs/`

Expect this to take **under one minute** on a normal laptop CPU. You will see
a step-by-step printed log exactly matching Section 3 of this README.

### B. Launch the interactive Streamlit app

```bash
python -m streamlit run app/app.py
```

(Plain `streamlit run app/app.py` also works *if* the `streamlit` command on
your system resolves to the same Python you installed packages into — using
`python -m streamlit` sidesteps that ambiguity entirely, same reasoning as
Section 9.)

This opens a browser tab (usually `http://localhost:8501`) with the full
interactive dashboard described in Section 11 below.

> ⚠️ You must run `train_model.py` at least once before launching the app —
> the app loads the saved model files from `models/` and `outputs/`.

### C. (Optional) Run a single prediction from the command line

```bash
python predict.py --age 45 --sex male --bmi 32 --children 2 --smoker yes --region southeast
```

Add `--model linear` to use Linear Regression instead of XGBoost (XGBoost is
the default since it's the winning model). Example output:

```
============================================================
 PREDICTION RESULT — XGBoost
============================================================
 Age:              45
 Sex:              male
 BMI:              32.0  (obese)
 Age Band:         36-50
 Children:         2
 Smoker:           yes
 Region:           southeast
------------------------------------------------------------
 PREDICTED ANNUAL CHARGES:  $42,975.09
============================================================
```

---

## 11. The Streamlit App — What It Does

The app (`app/app.py`) is a single-page interactive dashboard:

**Sidebar (left):** Adjustable controls for a hypothetical policy applicant —
age, sex, BMI, number of children, smoker status, region — plus a toggle to
choose which model generates the live prediction.

**Main panel:**
- **Prediction card** — large, prominent display of the predicted annual
  charges for the current applicant profile, with their BMI category and age
  band shown for context.
- **Side-by-side model comparison chart** — shows what *both* models would
  predict for the exact same applicant, so you can see the gap (or lack of
  one) directly.
- **Smart contextual insight box** — automatically appears when the selected
  applicant is a smoker with a high BMI, explaining *why* this combination
  drives cost so much higher (ties directly back to Section 5).
- **Population scatter plot** — places the current applicant as a star marker
  on top of the full dataset's BMI-vs-charges distribution, color-coded by
  smoker status, so you can see exactly where they fall relative to everyone else.
- **Feature contribution snapshot** — a small table showing the exact
  engineered feature values feeding the model for this prediction.

**Below the fold — four tabs:**
1. **Model Comparison** — metric cards (R², RMSE for both models), full
   comparison table, and the winner badge with an explanation of *why*.
2. **Residual Analysis** — the saved residual plots with a guided explanation
   of how to read them.
3. **SHAP Explainability** — the SHAP summary plot with a guided explanation
   of which features drive predictions and why.
4. **Dataset Overview** — a raw data sample and the full EDA chart panel.

Reference screenshots of all of this are included in `outputs/app_screenshot_*.png`.

---

## 12. Suggested Report Structure Mapping

If you're writing this up against a formal report brief, here's how the
project maps to common requirements:

| Report Section | Where to find it |
|---|---|
| Problem statement & objectives | Section 1 of this README |
| Dataset description & justification | Section 2 |
| Data preprocessing & feature engineering | Section 3, step 2 |
| Methodology & model selection | Section 3, steps 3–6 |
| Experimental design | `train_model.py` Steps 4–8 (train/test split, CV, grid search) |
| Results & evaluation | Section 4, `outputs/final_metrics.json`, `outputs/model_comparison.csv` |
| Comparative analysis & discussion | Section 5 (the smoker × bmi finding) |
| Limitations, biases, lessons learned | Section 7 |
| Reproducibility / software engineering practice | Section 9–10, fixed random seeds throughout |

---

## 13. Troubleshooting

**"ModuleNotFoundError: No module named 'matplotlib'" (or any other package),
even after `pip install -r requirements.txt` appeared to succeed:**

This means `pip` and `python` point to two different Python installations on
your machine — extremely common on Windows and on Macs with both a
system Python and a separately installed one. Fix:

```bash
python -m pip install -r requirements.txt
python check_setup.py
```

`check_setup.py` will print the exact Python path in use and confirm every
package is actually visible to it. If it still reports something missing,
run the exact command it prints — it's generated using `sys.executable`, so
it is guaranteed to target the correct interpreter.

**"streamlit: command not found":**
Use `python -m streamlit run app/app.py` instead of bare `streamlit run ...`
— same root cause as above.

**App loads but shows a file-not-found error for the models:**
You need to run `python train_model.py` at least once before launching the
app — it generates the `models/` and `outputs/` files the app depends on.
(This package ships with those files pre-generated, so this should only
happen if they were deleted.)

**Still stuck:**
Run `python check_setup.py` and `python --version` and check the printed
Python path matches where you ran `pip install`. Virtually all setup issues
trace back to this single mismatch.

---

## 14. Credits & Disclaimer

Built as an academic/portfolio term project. Dataset is a widely used public
teaching dataset for insurance cost modelling. This tool and its predictions
are for educational demonstration only and must not be used for real
insurance underwriting, pricing, or any actuarial decision-making.
