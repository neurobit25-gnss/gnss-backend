from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
from tensorflow.keras.models import load_model
import pandas as pd
import numpy as np
import io

app = Flask(__name__)

# ✅ Allow Vercel frontend access (add localhost for safety)
CORS(app, resources={r"/*": {"origins": ["https://gnss-xi.vercel.app", "http://localhost:5173"]}})

# ✅ Load model once
MODEL_PATH = "fixed_model.h5"
model = load_model(MODEL_PATH, compile=False)

@app.route('/')
def home():
    return jsonify({"message": "GNSS Error Prediction Backend is Live 🚀"})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # ✅ Check file
        if 'file' not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files['file']
        filename = file.filename.lower()

        # ✅ Read CSV or Excel
        if filename.endswith('.csv'):
            df = pd.read_csv(file)
        elif filename.endswith(('.xls', '.xlsx')):
            df = pd.read_excel(file)
        else:
            return jsonify({"error": "Unsupported file format. Please upload a CSV or Excel file."}), 400

        # ✅ Validate input columns
        if df.shape[1] not in [5, 13]:
            return jsonify({"error": f"Invalid input shape. Expected 5 or 13 columns, got {df.shape[1]}"}), 400

        # ✅ Reshape and predict
        time_steps = df.shape[0] // 48
        data = df.values[:time_steps * 48].reshape((time_steps, 48, df.shape[1]))
        preds = model.predict(data)

        return jsonify({"prediction": preds.tolist()})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
