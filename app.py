from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
from tensorflow.keras.models import load_model
import pandas as pd
import numpy as np
import io

app = Flask(__name__)

# ✅ Allow both local and Vercel frontend access
CORS(app, resources={r"/*": {"origins": ["https://gnss-xi.vercel.app", "http://localhost:5173"]}})

# ✅ Load the model once at startup
MODEL_PATH = "fixed_model.h5"
try:
    model = load_model(MODEL_PATH, compile=False)
    print("✅ Model loaded successfully.")
except Exception as e:
    print("❌ Model loading failed:", e)
    model = None


@app.route('/')
def home():
    return jsonify({"message": "✅ GNSS Error Prediction Backend is Live"}), 200


@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files['file']
        filename = file.filename.lower()

        # ✅ Read CSV / Excel files
        if filename.endswith('.csv'):
            df = pd.read_csv(file)
        elif filename.endswith(('.xls', '.xlsx')):
            df = pd.read_excel(file)
        else:
            return jsonify({"error": "Unsupported file format. Please upload a CSV or Excel file."}), 400

        # ✅ Validate that model is loaded
        if model is None:
            return jsonify({"error": "Model failed to load on the server."}), 500

        # ✅ Clean up NaN values
        df = df.dropna()
        if df.empty:
            return jsonify({"error": "Uploaded file is empty or invalid."}), 400

        # ✅ Ensure correct shape
        n_features = df.shape[1]
        if n_features not in [5, 13]:
            return jsonify({"error": f"Invalid number of columns: {n_features}. Expected 5 or 13."}), 400

        # ✅ Convert and reshape safely
        data = df.values.astype(float)
        time_steps = min(48, len(data))  # Trim or use first 48 samples
        data = data[:time_steps].reshape((1, time_steps, n_features))

        # ✅ Run prediction
        preds = model.predict(data)
        preds_list = preds.flatten().tolist()

        return jsonify({
            "status": "success",
            "message": "Prediction completed successfully.",
            "prediction": preds_list[:10]  # return first 10 values
        }), 200

    except Exception as e:
        print("❌ Error during prediction:", e)
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
