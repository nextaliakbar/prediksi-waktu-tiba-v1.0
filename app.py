from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd
import os
import xgboost as xgb

app = Flask(__name__)

# Load model dengan backward compatibility
try:
    # Method 1: Coba load langsung
    model = joblib.load('model_xgb.pkl')
    
    # Force CPU-only parameters
    if hasattr(model, 'set_param'):
        model.set_param({
            'predictor': 'cpu_predictor',
            'gpu_id': -1
        })
    
    print("Model loaded successfully with joblib")
    
except Exception as e1:
    print(f"Joblib load failed: {str(e1)}")
    try:
        # Method 2: Load sebagai XGBoost Booster
        model = xgb.Booster()
        model.load_model('model_xgb.pkl')
        model.set_param({'predictor': 'cpu_predictor'})
        print("Model loaded successfully with XGBoost Booster")
        
    except Exception as e2:
        print(f"XGBoost load failed: {str(e2)}")
        model = None

@app.route('/')
def home():
    return jsonify({
        'message': 'ML API is running!',
        'model_loaded': model is not None,
        'xgboost_version': xgb.__version__,
        'model_type': str(type(model)) if model else 'None'
    })

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    data = request.get_json()

    if not data:
        return jsonify({'error': 'No input data provided'}), 400

    try:
        input_data = pd.DataFrame([data])
        
        # Handle prediction berdasarkan tipe model
        if hasattr(model, 'predict'):
            # Sklearn-style model
            prediction = model.predict(input_data)[0]
        else:
            # XGBoost Booster
            dmatrix = xgb.DMatrix(input_data)
            prediction = model.predict(dmatrix)[0]

        return jsonify({'predict': round(float(prediction), 2)})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)