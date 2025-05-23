from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd
import os
import xgboost as xgb

app = Flask(__name__)

# Load model dengan fix untuk gpu_id issue
def load_model_safe():
    try:
        # Load model
        model = joblib.load('model_xgb.pkl')
        
        # Check jika model memiliki get_booster method (XGBRegressor/XGBClassifier)
        if hasattr(model, 'get_booster'):
            booster = model.get_booster()
            # Set parameter untuk CPU-only
            booster.set_param({'predictor': 'cpu_predictor'})
            
        # Jika model langsung XGBoost Booster
        elif hasattr(model, 'set_param'):
            model.set_param({'predictor': 'cpu_predictor'})
            
        print("Model loaded and configured for CPU")
        return model
        
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return None

model = load_model_safe()

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
        
        # Prediction dengan handling berbagai tipe model
        if hasattr(model, 'predict'):
            # Untuk XGBRegressor/XGBClassifier atau sklearn-style
            try:
                prediction = model.predict(input_data)[0]
            except Exception as pred_err:
                # Jika masih ada masalah dengan predict, coba akses booster langsung
                if hasattr(model, 'get_booster'):
                    booster = model.get_booster()
                    dmatrix = xgb.DMatrix(input_data)
                    prediction = booster.predict(dmatrix)[0]
                else:
                    raise pred_err
        else:
            # Untuk XGBoost Booster langsung
            dmatrix = xgb.DMatrix(input_data)
            prediction = model.predict(dmatrix)[0]

        return jsonify({'predict': round(float(prediction), 2)})
    
    except Exception as e:
        return jsonify({'error': f'Prediction error: {str(e)}'}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)