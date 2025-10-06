from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import pandas as pd

app = Flask(__name__)

# ✅ Allow CORS for all routes and origins (important for frontend connection)
CORS(app, resources={r"/*": {"origins": "*"}})

# ✅ Load the updated model
MODEL_PATH = "fixed_model.h5"
model = load_model(MODEL_PATH, compile=False)

@app.route('/')
def home():
    return jsonify({"message": "GNSS Error Prediction Backend is Live 🚀"})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Check if file is uploaded
        if 'file' not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files['file']

        # ✅ Read CSV safely
        try:
            df = pd.read_csv(file)
        except Exception:
            return jsonify({"error": "Invalid file format. Please upload a valid CSV file."}), 400

        # ✅ Basic shape check before prediction
        if df.shape[1] != 13:
            return jsonify({"error": f"Expected 13 features, got {df.shape[1]}"}), 400

        # ✅ Assuming dataset has 48 timesteps and 13 features
        try:
            data = df.values.reshape((1, 48, 13))
        except Exception:
            return jsonify({"error": "Data shape is invalid for model input."}), 400

        preds = model.predict(data)
        return jsonify({"predictions": preds.tolist()})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
