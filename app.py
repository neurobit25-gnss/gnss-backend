from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import pandas as pd
import traceback

app = Flask(__name__)

# ✅ Allow CORS for all routes and origins (so frontend can call backend)
CORS(app, resources={r"/*": {"origins": "*"}})

# ✅ Load the trained model
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

        # ✅ Step 1: Check if file is uploaded
        if 'file' not in request.files:
            print("❌ No file found in request")
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files['file']
        print(f"📂 Received file: {file.filename}")

        # ✅ Step 2: Try reading the file as CSV
        try:
            df = pd.read_csv(file)
            print(f"✅ CSV loaded successfully. Shape: {df.shape}")
        except Exception as e:
            print(f"❌ Error reading CSV: {e}")
            return jsonify({"error": "Invalid file format. Please upload a valid CSV file."}), 400

        # ✅ Step 3: Validate column count
        if df.shape[1] != 13:
            print(f"❌ Invalid number of columns: {df.shape[1]}")
            return jsonify({"error": f"Expected 13 features, got {df.shape[1]}"}), 400

        # ✅ Step 4: Reshape safely for model
        rows = df.shape[0]
        print(f"📊 Total rows detected: {rows}")

        timesteps = 48  # Expected timesteps for LSTM model
