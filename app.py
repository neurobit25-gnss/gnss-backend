from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import pandas as pd
import traceback
import os

app = Flask(__name__)

# ✅ Allow all frontend origins
CORS(app, resources={r"/*": {"origins": "*"}})

# ✅ Load model safely
MODEL_PATH = "fixed_model.h5"
try:
    model = load_model(MODEL_PATH, compile=False)
    print("✅ Model loaded successfully.")
except Exception as e:
    print(f"❌ Error loading model: {e}")

@app.route('/')
def home():
    return jsonify({"message": "GNSS Error Prediction Backend is Live 🚀"})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        print("\n🚀 Received request for /predict")

        # ✅ Step 1: File validation
        if 'file' not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files['file']
        filename = file.filename.lower()
        print(f"📂 Received file: {filename}")

        # ✅ Step 2: Load file dynamically (Excel or CSV)
        try:
            if filename.endswith('.csv'):
                df = pd.read_csv(file)
            elif filename.endswith('.xlsx') or filename.endswith('.xls'):
                df = pd.read_excel(file)
            else:
                return jsonify({"error": "Unsupported file format. Please upload CSV or Excel."}), 400
            print(f"✅ File loaded successfully. Shape: {df.shape}")
        except Exception as e:
            print(f"❌ File reading error: {e}")
            return jsonify({"error": "Failed to read file. Ensure it’s a valid CSV/XLSX format."}), 400

        # ✅ Step 3: Validate columns
        if df.shape[1] != 13:
            return jsonify({"error": f"Expected 13 features, got {df.shape[1]}"}), 400

        # ✅ Step 4: Ensure proper number of timesteps
        timesteps = 48
        rows = df.shape[0]
        print(f"📊 Total rows detected: {rows}")

        if rows > timesteps:
            df = df.iloc[:timesteps]
        elif rows < timesteps:
            pad_rows = timesteps - rows
            pad_df = pd.DataFrame(np.zeros((pad_rows, df.shape[1])), columns=df.columns)
            df = pd.concat([df, pad_df], ignore_index=True)

        # ✅ Step 5: Prepare data for prediction
        data = df.values.reshape((1, timesteps, 13))
        print(f"✅ Data reshaped for model input: {data.shape}")

        # ✅ Step 6: Run prediction
        preds = model.predict(data)
        print("✅ Model prediction completed")

        return jsonify({"predictions": preds.tolist()})

    except Exception as e:
        print("❌ Unexpected error:")
        print(traceback.format_exc())
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 10000)))
