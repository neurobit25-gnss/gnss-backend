from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import pandas as pd
import os

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# ✅ Load model safely
MODEL_PATH = "fixed_model.h5"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"❌ Model file not found: {MODEL_PATH}")

try:
    model = load_model(MODEL_PATH, compile=False)
    print("✅ Model loaded successfully!")
except Exception as e:
    print("❌ Error loading model:", e)
    raise e


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

        # ✅ Support both CSV and Excel formats
        if filename.endswith('.csv'):
            df = pd.read_csv(file)
        elif filename.endswith(('.xls', '.xlsx')):
            df = pd.read_excel(file)
        else:
            return jsonify({"error": "Invalid file format. Upload CSV or Excel only."}), 400

        print(f"✅ File loaded successfully: {filename}")
        print(f"📊 Data shape: {df.shape}")

        # ✅ Check number of columns (expected 13)
        if df.shape[1] != 13:
            return jsonify({"error": f"Expected 13 features, got {df.shape[1]}"}), 400

        # ✅ Check number of rows
        if df.shape[0] < 48:
            return jsonify({"error": f"Need at least 48 rows for prediction, got {df.shape[0]}"}), 400

        # ✅ Prepare last 48 timesteps for prediction
        data = df.tail(48).values.reshape((1, 48, 13))
        print("✅ Data reshaped successfully:", data.shape)

        preds = model.predict(data)
        print("✅ Prediction completed!")

        return jsonify({"predictions": preds.tolist()})

    except Exception as e:
        print("❌ Error during prediction:", e)
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    print("🚀 Starting Flask app...")
    app.run(host='0.0.0.0', port=10000)
