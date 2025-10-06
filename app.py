from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# Load the trained model
MODEL_PATH = "lstm_model.h5"
model = load_model(MODEL_PATH, compile=False)
model.compile(optimizer='adam', loss='mse')

@app.route('/')
def home():
    return jsonify({"message": "GNSS Backend is Live 🚀"})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400

        file = request.files['file']
        if file.filename.endswith('.csv'):
            df = pd.read_csv(file)
        elif file.filename.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(file)
        else:
            return jsonify({'error': 'Unsupported file format'}), 400

        # Clean and preprocess data
        df.columns = [c.strip().lower().replace(' ', '_').replace('(m)', '') for c in df.columns]
        print("Cleaned columns:", df.columns.tolist())

        # Check if required columns exist
        required_cols = ['x_error', 'y_error', 'z_error', 'satclockerror']
        for col in required_cols:
            if col not in df.columns:
                return jsonify({'error': f'Missing column: {col}'}), 400

        # Prepare data for model
        X = df[required_cols].values
        X = np.expand_dims(X, axis=0)  # shape (1, n_samples, 4)

        # Predict
        y_pred = model.predict(X).tolist()

        return jsonify({'prediction': y_pred})

    except Exception as e:
        print("Error:", e)
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    from flask_cors import CORS
    CORS(app)
    app.run(host='0.0.0.0', port=10000)

