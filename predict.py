import pickle

from flask import Flask
from flask import request
from flask import jsonify
from sympy import Order


model_file = 'model_C=1.0.bin'

with open(model_file, 'rb') as f_in:
    dv, model = pickle.load(f_in)

app = Flask('profit-predictor')

@app.route('/predict', methods=['POST'])
def predict():
    # json = Python dictionary
    order = request.get_json()

    X = dv.transform([order])
    y_pred = model.predict_proba(X)[0, 1]
    profit = y_pred >= 0.5

    result = {
        'profit_probability': float(y_pred),
        'profit': bool(profit)
    }

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)