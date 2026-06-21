"""
AI Crop Advisor — Flask Backend
---------------------------------
Serves two models:
  1. Crop Recommendation (classification) - given soil/weather -> top crops
  2. Yield Prediction (regression) - given crop + conditions -> expected yield
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import json
import numpy as np
import pandas as pd
import os

app = Flask(__name__)
CORS(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "..", "models")

# Load models once at startup
rec_model = joblib.load(os.path.join(MODEL_DIR, "crop_recommendation_model.joblib"))
rec_label_encoder = joblib.load(os.path.join(MODEL_DIR, "crop_label_encoder.joblib"))
yield_pipeline = joblib.load(os.path.join(MODEL_DIR, "yield_prediction_pipeline.joblib"))

with open(os.path.join(MODEL_DIR, "recommendation_metrics.json")) as f:
    rec_metrics = json.load(f)

with open(os.path.join(MODEL_DIR, "yield_metrics.json")) as f:
    yield_metrics = json.load(f)

YIELD_SUPPORTED_CROPS = set(yield_metrics["categorical_options"]["Crop_Type"])
REC_FEATURE_COLS = rec_metrics["feature_cols"]  # ["N","P","K","pH","rainfall","temperature"]


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/api/metadata", methods=["GET"])
def metadata():
    """Returns dropdown options + model performance stats for the frontend."""
    return jsonify({
        "recommendation": {
            "accuracy": rec_metrics["accuracy"],
            "weighted_f1": rec_metrics["weighted_f1"],
            "cv_accuracy_mean": rec_metrics["cv_accuracy_mean"],
            "n_classes": rec_metrics["n_classes"],
            "feature_importance": rec_metrics["feature_importance"],
        },
        "yield": {
            "mae": yield_metrics["mae"],
            "rmse": yield_metrics["rmse"],
            "r2": yield_metrics["r2"],
            "cv_r2_mean": yield_metrics["cv_r2_mean"],
            "categorical_options": yield_metrics["categorical_options"],
            "feature_importance_top10": yield_metrics["feature_importance_top10"],
        },
        "yield_supported_crops": sorted(list(YIELD_SUPPORTED_CROPS)),
    })


@app.route("/api/recommend", methods=["POST"])
def recommend():
    """
    Input JSON: { N, P, K, pH, rainfall, temperature }
    Output: top 3 recommended crops with confidence scores
    """
    data = request.get_json()
    try:
        row = {col: float(data[col]) for col in REC_FEATURE_COLS}
    except (KeyError, ValueError, TypeError) as e:
        return jsonify({"error": f"Invalid or missing input: {e}"}), 400

    X = pd.DataFrame([row], columns=REC_FEATURE_COLS)
    probs = rec_model.predict_proba(X)[0]

    top_idx = np.argsort(probs)[::-1][:3]
    top_crops = [
        {
            "crop": rec_label_encoder.inverse_transform([idx])[0],
            "confidence": round(float(probs[idx]) * 100, 2),
        }
        for idx in top_idx
    ]

    for item in top_crops:
        item["yield_model_available"] = item["crop"].capitalize() in YIELD_SUPPORTED_CROPS or item["crop"] in YIELD_SUPPORTED_CROPS

    return jsonify({"recommendations": top_crops, "input": row})


@app.route("/api/predict-yield", methods=["POST"])
def predict_yield():
    """
    Input JSON: {
      N, P, K, Soil_pH, Soil_Moisture, Soil_Type, Organic_Carbon,
      Temperature, Humidity, Rainfall, Sunlight_Hours, Wind_Speed,
      Region, Altitude, Season, Crop_Type, Irrigation_Type,
      Fertilizer_Used, Pesticide_Used
    }
    Output: predicted yield (tons/hectare) + risk flag
    """
    data = request.get_json()

    numeric_cols = yield_metrics["numeric_cols"]
    categorical_cols = yield_metrics["categorical_cols"]
    all_cols = numeric_cols + categorical_cols

    try:
        row = {}
        for col in numeric_cols:
            row[col] = float(data[col])
        for col in categorical_cols:
            row[col] = str(data[col])
    except (KeyError, ValueError, TypeError) as e:
        return jsonify({"error": f"Invalid or missing input: {e}"}), 400

    if row["Crop_Type"] not in YIELD_SUPPORTED_CROPS:
        return jsonify({
            "error": f"Yield model currently supports: {sorted(YIELD_SUPPORTED_CROPS)}"
        }), 400

    X = pd.DataFrame([row], columns=all_cols)
    pred = yield_pipeline.predict(X)[0]
    pred = max(0, float(pred))

    # simple risk heuristic based on percentile bands seen during training
    if pred < 8:
        risk = "High risk — low expected yield"
    elif pred < 20:
        risk = "Moderate — average expected yield"
    else:
        risk = "Low risk — strong expected yield"

    return jsonify({
        "predicted_yield_tons_per_hectare": round(pred, 2),
        "risk_assessment": risk,
        "input": row,
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host="0.0.0.0", port=port)
