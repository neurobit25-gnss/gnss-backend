from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import pandas as pd

app = Flask(__name__)
CORS(app)

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
        df = pd.read_csv(file)

        # ✅ Assuming your dataset has 48 timesteps and 13 features
        data = df.values.reshape((1, 48, 13))
        preds = model.predict(data)

        return jsonify({"predictions": preds.tolist()})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
