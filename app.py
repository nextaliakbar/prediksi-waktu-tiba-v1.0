from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

model = joblib.load('model_xgb.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    try:
        input_data = pd.DataFrame([data])

        prediction = model.predict(input_data)[0]

        return jsonify({'predict' : round(float(prediction), 2)})
    
    except Exception as e:
        return jsonify({'error' : str(e)})
    

if __name__ == '__main__':
    app.run(debug=True)