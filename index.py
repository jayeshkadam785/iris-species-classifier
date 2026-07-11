import os
import joblib
import pandas as pd
from flask import Flask, request, jsonify

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model = joblib.load(os.path.join(BASE_DIR, "model.pkl"))
label_encoder = joblib.load(os.path.join(BASE_DIR, "label_encoder.pkl"))


@app.route("/api/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        sepal_length = float(data["sepal_length"])
        sepal_width = float(data["sepal_width"])
        petal_length = float(data["petal_length"])
        petal_width = float(data["petal_width"])

        features = pd.DataFrame(
            [[sepal_length, sepal_width, petal_length, petal_width]],
            columns=["sepal_length", "sepal_width", "petal_length", "petal_width"]
        )
        pred = model.predict(features)[0]
        probs = model.predict_proba(features)[0]
        species = label_encoder.inverse_transform([pred])[0]

        confidence = {
            label_encoder.classes_[i]: round(float(probs[i]) * 100, 2)
            for i in range(len(probs))
        }

        return jsonify({
            "species": species,
            "confidence": confidence
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})
