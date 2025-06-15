import pandas as pd
import ast
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import joblib

# Load sales data
df = pd.read_csv('sales_data.csv')

# 🧹 Drop empty rows or missing 'amount' or 'timestamp'
df = df.dropna(subset=['amount', 'timestamp'])

# 🧠 Convert timestamp to day of week
df['day_of_week'] = pd.to_datetime(df['timestamp']).dt.dayofweek

# 🔁 Extract total quantity from 'products' string
def extract_total_quantity(products_str):
    try:
        # Some rows have empty products
        if pd.isna(products_str):
            return 0
        # Convert string to list of dicts
        products = ast.literal_eval(products_str)
        return sum(item.get('quantity', 0) for item in products)
    except Exception as e:
        print("Error parsing products:", products_str)
        return 0

df['quantity'] = df['products'].apply(extract_total_quantity)

# 🧪 Print to verify
print(df[['amount', 'timestamp', 'day_of_week', 'quantity']].head())

# 🎯 Feature matrix X and target y
X = df[['day_of_week', 'quantity']]
y = df['amount']

# 🧪 Split data into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# 🤖 Train model
model = LinearRegression()
model.fit(X_train, y_train)

# 💾 Save trained model
joblib.dump(model, 'sales_model.pkl')

print("✅ Model trained and saved as sales_model.pkl")
