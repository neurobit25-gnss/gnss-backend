from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import pandas as pd
import os

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# ✅ Load the trained model
MODEL_PATH = "fixed_model.h5"
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found at path: {MODEL_PATH}")
model = load_model(MODEL_PATH, compile=False)

@app.route('/')
def home():
    return jsonify({"message": "GNSS Error Prediction Backend is Live 🚀"})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # ✅ Ensure file exists
        if 'file' not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files['file']
        filename = file.filename.lower()

        # ✅ Read the file
        if filename.endswith('.csv'):
            df = pd.read_csv(file)
        elif filename.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(file)
        else:
            return jsonify({"error": "Unsupported file type. Please upload a CSV or Excel file."}), 400

        if df.empty:
            return jsonify({"error": "Uploaded file is empty."}), 400

        # ✅ Model expects 5 features
        expected_features = 5
        timesteps = 48

        # ✅ Validate column count
        if df.shape[1] != expected_features:
            return jsonify({
                "error": f"Expected {expected_features} features, got {df.shape[1]}"
            }), 400

        num_rows = df.shape[0]
        data = df.values

        # ✅ Automatically handle shorter or longer sequences
        if num_rows < timesteps:
            # Pad with zeros if not enough data
            padding = np.zeros((timesteps - num_rows, expected_features))
            data = np.vstack((padding, data))
        elif num_rows > timesteps:
            # Use the last 48 rows if too long
            data = data[-timesteps:, :]

        # ✅ Reshape to match LSTM input
        data = data.reshape((1, timesteps, expected_features))

        # ✅ Make prediction
        preds = model.predict(data)
        preds_list = preds.tolist()

        return jsonify({
            "message": "Prediction successful 🎯",
            "shape_used": [1, timesteps, expected_features],
            "predictions": preds_list
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
