from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib

app = Flask(__name__)
CORS(app)

# Load all three models
sales_model = joblib.load('sales_model.pkl')
customer_model = joblib.load('customer_model.pkl')
stock_model = joblib.load('stock_model.pkl')

# Predict total sales
@app.route('/predict', methods=['POST'])
def predict_sales():
    data = request.get_json()
    print("Received for sales:", data)
    X = [[data['day_of_week'], data['quantity']]]
    prediction = sales_model.predict(X)[0]
    return jsonify({'prediction': round(prediction, 2)})

# Predict customer purchasing behavior
@app.route('/predict-customer', methods=['POST'])
def predict_customer():
    data = request.get_json()
    print("Received for customer:", data)
    X = [[data['hour'], data['day_of_week'], data['total_items']]]
    prediction = customer_model.predict(X)[0]
    return jsonify({'segment': prediction})

# Predict stock demand
@app.route('/predict-stock', methods=['POST'])
def predict_stock():
    data = request.get_json()
    print("Received for stock:", data)
    X = [[data['product_id'], data['day_of_week']]]
    prediction = stock_model.predict(X)[0]
    return jsonify({'demand_prediction': round(prediction, 2)})

# Predict recommended buy
@app.route('/predict-recommended-buy', methods=['POST'])
def predict_recommended_buy():
    data = request.get_json()
    print("📥 Received data:", data)

    X = [[data['current_stock'], data['sales_30_days']]]
    prediction = model.predict(X)[0]

    print("📤 Prediction result:", prediction)
    return jsonify({'recommended_buy': int(round(prediction))})



if __name__ == '__main__':
    app.run(port=5000)
