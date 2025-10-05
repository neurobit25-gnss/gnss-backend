from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model

app = Flask(__name__)

# ✅ Allow frontend (Vercel) to connect
CORS(app, resources={r"/*": {"origins": "*"}})

# ✅ Load model
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
        filename = file.filename.lower()

        # ✅ Handle both CSV and Excel
        if filename.endswith('.csv'):
            df = pd.read_csv(file)
        elif filename.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(file)
        else:
            return jsonify({'error': 'Unsupported file format'}), 400

        # ✅ Clean column names
        df.columns = [c.strip().lower().replace(' ', '_').replace('(m)', '') for c in df.columns]
        print("Cleaned columns:", df.columns.tolist())

        if 'x_error' not in df.columns:
            return jsonify({'error': "Column 'x_error' missing in file"}), 400

        # ✅ Predict using model
        X = np.expand_dims(df['x_error'].values, axis=0)
        y_pred = model.predict(X).tolist()

        return jsonify({'prediction': y_pred})

    except Exception as e:
        print("Error:", e)
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
