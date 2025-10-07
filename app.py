from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import pandas as pd
import os

# ---------------------------------------------------------------------
# ✅ Flask Setup
# ---------------------------------------------------------------------
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# ---------------------------------------------------------------------
# ✅ Model Loading
# ---------------------------------------------------------------------
MODEL_PATH = "fixed_model.h5"

try:
    model = load_model(MODEL_PATH, compile=False)
    print("✅ Model loaded successfully.")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None


# ---------------------------------------------------------------------
# ✅ Health Check Route
# ---------------------------------------------------------------------
@app.route('/')
def home():
    return jsonify({"message": "✅ GNSS Error Prediction Backend is Live"}), 200


# ---------------------------------------------------------------------
# ✅ Prediction Route
# ---------------------------------------------------------------------
@app.route('/predict', methods=['POST'])
def predict():
    try:
        if model is None:
            return jsonify({"error": "Model not loaded on server"}), 500

        # ✅ Check file
        if 'file' not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files['file']
        filename = file.filename.lower()

        # ✅ Handle CSV / Excel
        if filename.endswith('.csv'):
            df = pd.read_csv(file)
        elif filename.endswith(('.xls', '.xlsx')):
            df = pd.read_excel(file)
        else:
            return jsonify({"error": "Unsupported file format. Please upload CSV or Excel."}), 400

        # ✅ Keep only numeric columns (ignore utc_time)
        numeric_df = df.select_dtypes(include=[np.number])

        if numeric_df.empty:
            return jsonify({"error": "No numeric columns found in uploaded file."}), 400

        # ✅ Expecting 5 features (x_error, y_error, z_error, clock error, etc.)
        if numeric_df.shape[1] != 5:
            return jsonify({"error": f"Expected 5 numeric columns, got {numeric_df.shape[1]}"}), 400

        # ✅ Convert to numpy array
        data = numeric_df.values.astype(float)

        # ✅ Limit or pad to 48 timesteps for LSTM
        time_steps = min(48, len(data))
        data = data[:time_steps].reshape((1, time_steps, 5))

        # ✅ Run Prediction
        preds = model.predict(data)
        preds_list = preds.flatten().tolist()

        # ✅ Response
        return jsonify({
            "status": "success",
            "message": "Prediction completed successfully.",
            "prediction": preds_list[:10]  # Send first 10 predicted values for display
        }), 200

    except Exception as e:
        print("❌ Error during prediction:", e)
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500


# ---------------------------------------------------------------------
# ✅ Server Entry Point
# ---------------------------------------------------------------------
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)
