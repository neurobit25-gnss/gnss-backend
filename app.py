from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import pandas as pd
import io

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# ✅ Load your trained model
MODEL_PATH = "fixed_model.h5"
model = load_model(MODEL_PATH, compile=False)

@app.route('/')
def home():
    return jsonify({"message": "GNSS Error Prediction Backend is Live 🚀"})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files['file']
        filename = file.filename.lower()

        # ✅ Handle different file formats
        if filename.endswith('.csv'):
            df = pd.read_csv(file)
        elif filename.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(file)
        else:
            return jsonify({"error": "Unsupported file format. Upload CSV or Excel files only."}), 400

        # ✅ Basic validation
        if df.shape[1] < 5:
            return jsonify({"error": f"Expected at least 5 features, got {df.shape[1]}"}), 400

        # ✅ Dynamically reshape based on available features
        timesteps = 48
        num_features = df.shape[1]
        total_values = df.shape[0] * df.shape[1]

        # Ensure enough data for 48 timesteps
        if total_values < timesteps * num_features:
            return jsonify({"error": f"Not enough data to reshape into (1, {timesteps}, {num_features})"}), 400

        reshaped_data = df.values[:timesteps * num_features].reshape((1, timesteps, num_features))

        preds = model.predict(reshaped_data)
        return jsonify({"prediction": preds.tolist()})

    except Exception as e:
        print("❌ Backend Error:", e)
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
