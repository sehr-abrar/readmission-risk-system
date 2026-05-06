import json
import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__)
CORS(app)

model = joblib.load(os.path.join(BASE_DIR, "model_pipeline.pkl"))

with open(os.path.join(BASE_DIR, "feature_importance.json")) as f:
    feature_importance = json.load(f)


@app.route("/")
def home():
    return "backend is running"


@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    try:
        input_df = pd.DataFrame([data])
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1]
        return jsonify({
            "prediction": int(prediction),
            "probability": round(float(probability), 4),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/feature-importance", methods=["GET"])
def get_feature_importance():
    return jsonify(feature_importance)


if __name__ == "__main__":
    app.run(debug=True)
