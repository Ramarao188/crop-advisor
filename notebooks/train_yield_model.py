"""
Crop Yield Prediction Model Training
--------------------------------------
Predicts crop yield (tons/hectare) given soil, weather, region,
irrigation, fertilizer and pesticide usage. Regression task.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import joblib
import json

DATA_PATH = "../data/crop-yield.csv"
MODEL_DIR = "../models"
TARGET = "Crop_Yield_ton_per_hectare"


def main():
    print("Loading data...")
    df = pd.read_csv(DATA_PATH)

    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X.select_dtypes(include=["object", "str"]).columns.tolist()

    print(f"Dataset shape: {X.shape}")
    print(f"Numeric features: {numeric_cols}")
    print(f"Categorical features: {categorical_cols}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), numeric_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
    ])

    model = xgb.XGBRegressor(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
        n_jobs=-1,
    )

    pipe = Pipeline([("preprocessor", preprocessor), ("model", model)])

    print("\nTraining XGBoost regressor...")
    pipe.fit(X_train, y_train)

    preds = pipe.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)

    print(f"\nTest MAE: {mae:.3f} tons/hectare")
    print(f"Test RMSE: {rmse:.3f} tons/hectare")
    print(f"Test R2: {r2:.4f}")

    cv_scores = cross_val_score(pipe, X, y, cv=5, scoring="r2", n_jobs=-1)
    print(f"5-Fold CV R2: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

    # Feature importance (numeric + one-hot expanded names)
    ohe = pipe.named_steps["preprocessor"].named_transformers_["cat"]
    cat_feature_names = ohe.get_feature_names_out(categorical_cols).tolist()
    all_feature_names = numeric_cols + cat_feature_names
    importances = pipe.named_steps["model"].feature_importances_
    importance_dict = dict(zip(all_feature_names, importances.tolist()))
    importance_dict = dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))

    print("\nTop 10 Feature Importances:")
    for feat, imp in list(importance_dict.items())[:10]:
        print(f"  {feat}: {imp:.4f}")

    # Save artifacts
    joblib.dump(pipe, f"{MODEL_DIR}/yield_prediction_pipeline.joblib")

    metrics = {
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "cv_r2_mean": cv_scores.mean(),
        "cv_r2_std": cv_scores.std(),
        "feature_importance_top10": dict(list(importance_dict.items())[:10]),
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "categorical_options": {col: sorted(df[col].unique().tolist()) for col in categorical_cols},
    }
    with open(f"{MODEL_DIR}/yield_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print("\nModel and metrics saved to", MODEL_DIR)


if __name__ == "__main__":
    main()
