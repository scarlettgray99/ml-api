# train_stock_model.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib

# Load data
df = pd.read_csv('sales_data.csv')

# Preprocess
df['timestamp'] = pd.to_datetime(df['timestamp'])
df['day_of_week'] = df['timestamp'].dt.dayofweek

# Flatten product sales
product_sales = []
for _, row in df.iterrows():
    if pd.notnull(row['products']):
        for p in eval(row['products']):
            product_sales.append({
                'product_name': p['name'],
                'quantity': p['quantity'],
                'day_of_week': row['timestamp'].dayofweek
            })

sales_df = pd.DataFrame(product_sales)

# Encode product names
sales_df['product_id'] = sales_df['product_name'].astype('category').cat.codes

# Features and target
X = sales_df[['product_id', 'day_of_week']]
y = sales_df['quantity']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Save model
joblib.dump(model, 'stock_model.pkl')
print("✅ Stock forecast model saved as stock_model.pkl")
