"""
================================================================================
STANDALONE INFERENCE SCRIPT
================================================================================
Use this to test the trained models from the command line without launching
the Streamlit app. Useful for quick checks, debugging, or integrating the
model into another pipeline.

Usage:
    python predict.py --age 35 --sex male --bmi 31.5 --children 2 --smoker yes --region southeast
================================================================================
"""

import argparse
import json
import joblib
import pandas as pd


def engineer_features(age, sex, bmi, children, smoker, region, feature_cols):
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

    for col, val in [(f"region_{region}", 1), (f"bmi_category_{bmi_cat}", 1), (f"age_band_{age_band}", 1)]:
        if col in row:
            row[col] = val

    return pd.DataFrame([row])[feature_cols], bmi_cat, age_band


def main():
    parser = argparse.ArgumentParser(description="Predict medical insurance charges.")
    parser.add_argument("--age", type=int, required=True)
    parser.add_argument("--sex", choices=["male", "female"], required=True)
    parser.add_argument("--bmi", type=float, required=True)
    parser.add_argument("--children", type=int, default=0)
    parser.add_argument("--smoker", choices=["yes", "no"], required=True)
    parser.add_argument("--region", choices=["northeast", "northwest", "southeast", "southwest"], required=True)
    parser.add_argument("--model", choices=["xgboost", "linear"], default="xgboost",
                         help="Which model to use for prediction (default: xgboost, the winning model)")
    args = parser.parse_args()

    lr_model = joblib.load("models/linear_regression_model.joblib")
    xgb_model = joblib.load("models/xgboost_model.joblib")
    scaler = joblib.load("models/scaler.joblib")
    with open("models/feature_metadata.json") as f:
        meta = json.load(f)
    feature_cols = meta["feature_cols"]

    X, bmi_cat, age_band = engineer_features(
        args.age, args.sex, args.bmi, args.children, args.smoker, args.region, feature_cols
    )

    if args.model == "linear":
        X_scaled = scaler.transform(X)
        pred = lr_model.predict(X_scaled)[0]
        model_name = "Linear Regression"
    else:
        pred = xgb_model.predict(X)[0]
        model_name = "XGBoost"

    pred = max(pred, 0)

    print("=" * 60)
    print(f" PREDICTION RESULT — {model_name}")
    print("=" * 60)
    print(f" Age:              {args.age}")
    print(f" Sex:              {args.sex}")
    print(f" BMI:              {args.bmi}  ({bmi_cat})")
    print(f" Age Band:         {age_band}")
    print(f" Children:         {args.children}")
    print(f" Smoker:           {args.smoker}")
    print(f" Region:           {args.region}")
    print("-" * 60)
    print(f" PREDICTED ANNUAL CHARGES:  ${pred:,.2f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
