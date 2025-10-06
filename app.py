from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import pandas as pd
import io

app = Flask(__name__)

# ✅ Enable CORS (allow frontend requests)
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
        # ✅ Check if a file is uploaded
        if 'file' not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files['file']
        filename = file.filename.lower()

        # ✅ Read CSV or Excel file
        try:
            if filename.endswith('.csv'):
                df = pd.read_csv(file)
            elif filename.endswith(('.xlsx', '.xls')):
                df = pd.read_excel(file)
            else:
                return jsonify({"error": "Unsupported file format. Upload .csv or .xlsx"}), 400
        except Exception as e:
            return jsonify({"error": f"File read error: {str(e)}"}), 400

        # ✅ Define model input configuration
        expected_features = 5  # Number of columns in your file
        timesteps = 48         # Time steps expected by model

        # ✅ Validate shape
        if df.shape[1] != expected_features:
            return jsonify({"error": f"Expected {expected_features} features, got {df.shape[1]}"}), 400

        # ✅ Convert data to numpy and reshape
        try:
            data = df.values.reshape((1, timesteps, expected_features))
        except Exception as e:
            return jsonify({"error": f"Reshape failed: {str(e)}"}), 400

        # ✅ Get model predictions
        preds = model.predict(data)

        return jsonify({"predictions": preds.tolist()})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    # ✅ Run the app
    app.run(host='0.0.0.0', port=10000)
