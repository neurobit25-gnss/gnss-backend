from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import tensorflow as tf
import os

app = Flask(__name__)
CORS(app)

# ✅ Load model once
MODEL_PATH = "fixed_model.h5"
model = tf.keras.models.load_model(MODEL_PATH)

@app.route("/")
def home():
    return jsonify({"message": "GNSS Error Prediction Backend is Live 🚀"})

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Check if file exists
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files["file"]
        filename = file.filename.lower()

        # ✅ Read the uploaded file
        if filename.endswith(".csv"):
            df = pd.read_csv(file)
        elif filename.endswith((".xlsx", ".xls")):
            df = pd.read_excel(file)
        else:
            return jsonify({"error": "Unsupported file format. Upload CSV or Excel only."}), 400

        # ✅ Drop non-numeric columns (like timestamps)
        df_numeric = df.select_dtypes(include=[np.number])

        if df_numeric.empty:
            return jsonify({"error": "No numeric data found in file."}), 400

        # ✅ Ensure input fits model shape
        expected_features = model.input_shape[-1]  # typically 13
        actual_features = df_numeric.shape[1]

        # Auto-adjust columns
        if actual_features < expected_features:
            # Pad with zeros if fewer features
            missing = expected_features - actual_features
            df_numeric = np.pad(df_numeric.values, ((0,0),(0,missing)), mode='constant')
        elif actual_features > expected_features:
            # Truncate extra columns
            df_numeric = df_numeric.iloc[:, :expected_features].values
        else:
            df_numeric = df_numeric.values

        # ✅ Reshape to match LSTM input: (batch_size, timesteps, features)
        timesteps = model.input_shape[1] or 1
        total_points = df_numeric.shape[0]

        # Handle too-short sequences
        if total_points < timesteps:
            padding = np.zeros((timesteps - total_points, expected_features))
            df_numeric = np.vstack([df_numeric, padding])

        # Cut to exact multiple of timesteps
        df_numeric = df_numeric[:timesteps, :].reshape(1, timesteps, expected_features)

        # ✅ Predict
        prediction = model.predict(df_numeric)
        return jsonify({"prediction": prediction.tolist()})

    except Exception as e:
        print("❌ Error:", str(e))
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
