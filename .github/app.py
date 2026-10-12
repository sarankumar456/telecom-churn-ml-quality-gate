from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, jsonify, request

app = Flask(__name__)

MODEL_PATH = Path("telecom_churn_model.pkl")
FEATURES = [
    "tenure",
    "monthly_charges",
    "contract",
    "tech_support",
    "internet_service",
    "payment_method",
]


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "telecom_churn_model.pkl was not found. Run train_model.py first."
        )
    return joblib.load(MODEL_PATH)


@app.get("/")
def health_check():
    return jsonify({
        "status": "ok",
        "service": "telecom-churn-prediction",
    })


@app.post("/predict")
def predict():
    data = request.get_json(silent=True)

    if not isinstance(data, dict):
        return jsonify({"error": "JSON request body is required"}), 400

    missing_fields = [feature for feature in FEATURES if feature not in data]
    if missing_fields:
        return jsonify({
            "error": "Missing required fields",
            "missing_fields": missing_fields,
        }), 400

    try:
        sample = pd.DataFrame([{feature: data[feature] for feature in FEATURES}])
        model = load_model()
        prediction_code = int(model.predict(sample)[0])
    except (TypeError, ValueError) as error:
        return jsonify({"error": "Invalid input values", "details": str(error)}), 400

    prediction = "CHURN" if prediction_code == 1 else "NO_CHURN"
    return jsonify({
        "prediction": prediction,
        "prediction_code": prediction_code,
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
