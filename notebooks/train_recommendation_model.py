"""
Crop Recommendation Model Training
------------------------------------
Predicts the best crop to grow given soil nutrients (N, P, K),
pH, rainfall, and temperature. Multi-class classification (40 crops).
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import xgboost as xgb
import joblib
import json

DATA_PATH = "../data/Train_Dataset.csv"
MODEL_DIR = "../models"

def main():
    print("Loading data...")
    df = pd.read_csv(DATA_PATH)
    df = df.drop(columns=["Unnamed: 0"])

    feature_cols = ["N", "P", "K", "pH", "rainfall", "temperature"]
    X = df[feature_cols]
    y = df["Crop"]

    print(f"Dataset shape: {X.shape}, Classes: {y.nunique()}")

    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.2, random_state=42, stratify=y_enc
    )

    print("\nTraining XGBoost classifier...")
    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=8,
        learning_rate=0.1,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
        eval_metric="mlogloss",
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds, average="weighted")

    print(f"\nTest Accuracy: {acc:.4f}")
    print(f"Test Weighted F1: {f1:.4f}")

    # 5-fold cross-validation for a more robust estimate
    cv_scores = cross_val_score(model, X, y_enc, cv=5, scoring="accuracy", n_jobs=-1)
    print(f"5-Fold CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

    print("\nClassification Report (top classes):")
    report = classification_report(y_test, preds, target_names=le.classes_, output_dict=True)
    print(classification_report(y_test, preds, target_names=le.classes_))

    # Feature importance
    importance = dict(zip(feature_cols, model.feature_importances_.tolist()))
    importance = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))
    print("\nFeature Importance:")
    for feat, imp in importance.items():
        print(f"  {feat}: {imp:.4f}")

    # Save model artifacts
    joblib.dump(model, f"{MODEL_DIR}/crop_recommendation_model.joblib")
    joblib.dump(le, f"{MODEL_DIR}/crop_label_encoder.joblib")

    metrics = {
        "accuracy": acc,
        "weighted_f1": f1,
        "cv_accuracy_mean": cv_scores.mean(),
        "cv_accuracy_std": cv_scores.std(),
        "feature_importance": importance,
        "n_classes": int(y.nunique()),
        "feature_cols": feature_cols,
    }
    with open(f"{MODEL_DIR}/recommendation_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print("\nModel and metrics saved to", MODEL_DIR)


if __name__ == "__main__":
    main()
