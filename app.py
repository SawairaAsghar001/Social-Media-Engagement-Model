import os
import numpy as np
import pickle
import xgboost as xgb
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Paths to the saved models
MODEL_DIR = r"C:\Users\Lenovo\PycharmProjects\FlaskProject\Social Media Engament"
lstm_model_path = os.path.join(MODEL_DIR, "model_lstm.h5")
xgboost_model_path = os.path.join(MODEL_DIR, "xgboost_model.pkl")
cnn_model_path = os.path.join(MODEL_DIR, "model_cnn.h5")
scaler_path = os.path.join(MODEL_DIR, "scaler.pkl")

# Initialize models
lstm_model = xgboost_model = cnn_model = scaler = None


def load_models():
    global lstm_model, xgboost_model, cnn_model, scaler

    try:
        lstm_model = load_model(lstm_model_path)
        print("✅ LSTM model loaded successfully")
    except Exception as e:
        print(f"❌ Error loading LSTM model: {e}")

    try:
        with open(xgboost_model_path, "rb") as file:
            xgboost_model = pickle.load(file)
        print("✅ XGBoost model loaded successfully")
    except Exception as e:
        print(f"❌ Error loading XGBoost model: {e}")

    try:
        cnn_model = load_model(cnn_model_path)
        print("✅ CNN model loaded successfully")
    except Exception as e:
        print(f"❌ Error loading CNN model: {e}")

    try:
        with open(scaler_path, "rb") as file:
            scaler = pickle.load(file)
        print("✅ Scaler loaded successfully")
    except Exception as e:
        print(f"❌ Error loading scaler: {e}")


load_models()


def preprocess_input(data):
    """Preprocess input data by scaling and reshaping."""
    global scaler

    if scaler is None:
        return None, "Scaler not loaded"

    features = np.array(data.get('features', []))
    if features.size == 0:
        return None, "No input features provided"

    features = features.reshape(1, -1)
    scaled_features = scaler.transform(features)
    return scaled_features, None


@app.route('/')
def home():
    return jsonify({"message": "Welcome to the Flask API!"})


@app.route('/favicon.ico')
def favicon():
    return '', 204


@app.route('/predict/xgb', methods=['POST'])
def predict_xgb():
    if xgboost_model is None:
        return jsonify({'error': 'XGBoost model not loaded'}), 500

    data = request.get_json()
    scaled_features, error = preprocess_input(data)
    if error:
        return jsonify({'error': error}), 400

    prediction = xgboost_model.predict(scaled_features)
    return jsonify({'prediction': prediction.tolist()})


if __name__ == "__main__":
    app.run(debug=True)
