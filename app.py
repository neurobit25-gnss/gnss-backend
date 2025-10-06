from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import tensorflow as tf
import os

# Initialize Flask app
app = Flask(__name__)

# ✅ Enable CORS globally
CORS(app, resources={r"/*": {"origins": "*"}})

# ✅ Load model safely
MODEL_PATH = "fixed_model.h5"
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print("✅ Model loaded successfully.")
except Exception as e:
    print("❌ Error loading model:", e)
    model = None


@app.route("/")
def home():
    return jsonify({"message": "GNSS Error Prediction Backend is Live 🚀"})


@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded on server"}), 500

    try:
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files["file"]
        filename = file.filename.lower()

        # ✅ Read the uploaded dataset
        if filename.endswith(".csv"):
            df = pd.read_csv(file)
        elif filename.endswith((".xlsx", ".xls")):
            df = pd.read_excel(file)
        else:
            return jsonify({"error": "Unsupported file format. Please upload CSV or Excel."}), 400

        # ✅ Select numeric columns only
        df_numeric = df.select_dtypes(include=[np.number])
        if df_numeric.empty:
            return jsonify({"error": "No numeric columns found in file."}), 400

        # ✅ Get model’s expected input shape
        expected_features = model.input_shape[-1]
        timesteps = model.input_shape[1] or 1

        # ✅ Fix feature count mismatch
        actual_features = df_numeric.shape[1]
        if actual_features < expected_features:
            missing = expected_features - actual_features
            df_numeric = np.pad(df_numeric.values, ((0, 0), (0, missing)), mode='constant')
        elif actual_features > expected_features:
            df_numeric = df_numeric.iloc[:, :expected_features].values
        else:
            df_numeric = df_numeric.values

        # ✅ Fix time step mismatch
        total_points = df_numeric.shape[0]
        if total_points < timesteps:
            padding = np.zeros((timesteps - total_points, expected_features))
            df_numeric = np.vstack([df_numeric, padding])
        else:
            df_numeric = df_numeric[:timesteps, :]

        # ✅ Reshape for model input
        df_input = df_numeric.reshape(1, timesteps, expected_features)

        # ✅ Predict
        preds = model.predict(df_input)
        return jsonify({"predictions": preds.tolist()})

    except Exception as e:
        print("❌ Backend error:", e)
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
