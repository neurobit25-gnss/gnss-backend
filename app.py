from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import pandas as pd
import os

# ----------------------------------------------------
# 🌐 Flask App Initialization
# ----------------------------------------------------
app = Flask(__name__)
CORS(app)

# ----------------------------------------------------
# 🧠 Load Your Trained Model
# ----------------------------------------------------
MODEL_PATH = "fixed_model.h5"
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError("❌ Model file 'fixed_model.h5' not found in directory!")

model = load_model(MODEL_PATH, compile=False)
print("✅ Model loaded successfully!")

# ----------------------------------------------------
# 🏠 Home Route (Health Check)
# ----------------------------------------------------
@app.route('/')
def home():
    return jsonify({"message": "GNSS Error Prediction Backend is Live 🚀"})

# ----------------------------------------------------
# 🔮 Prediction Route
# ----------------------------------------------------
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # ✅ Check file upload
        if 'file' not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files['file']
        filename = file.filename.lower()

        # ✅ Read the file based on its extension
        if filename.endswith('.csv'):
            df = pd.read_csv(file)
        elif filename.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(file)
        else:
            return jsonify({"error": "Unsupported file format. Please upload CSV or Excel."}), 400

        # ✅ Drop non-numeric columns like utc_time if present
        if 'utc_time' in df.columns:
            df = df.drop(columns=['utc_time'])

        # ✅ Select only numeric data and handle NaNs
        df = df.select_dtypes(include=[np.number]).fillna(0)

        # ----------------------------------------------------
        # 🧩 Check dataset shape consistency
        # ----------------------------------------------------
        expected_features = 5   # Columns: x_error, y_error, z_error, satclockerror, etc.
        timesteps = 48          # Your LSTM input sequence length

        if df.shape[1] != expected_features:
            return jsonify({"error": f"Expected {expected_features} features, got {df.shape[1]}"}), 400

        # ✅ Ensure exactly 48 timesteps by slicing or padding
        data = df.values[-timesteps:]  # Take last 48 rows
        if data.shape[0] < timesteps:
            padding = np.zeros((timesteps - data.shape[0], data.shape[1]))
            data = np.vstack((padding, data))

        # ✅ Reshape for LSTM input (1, 48, 5)
        x_input = np.expand_dims(data, axis=0)

        # ----------------------------------------------------
        # 🧠 Make Prediction
        # ----------------------------------------------------
        prediction = model.predict(x_input)
        prediction_list = prediction.tolist()

        return jsonify({
            "message": "Prediction successful!",
            "input_shape": list(x_input.shape),
            "prediction": prediction_list
        })

    except Exception as e:
        print("❌ Error during prediction:", str(e))
        return jsonify({"error": str(e)}), 500

# ----------------------------------------------------
# 🚀 Main Entry Point
# ----------------------------------------------------
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
