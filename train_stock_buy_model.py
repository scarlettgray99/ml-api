import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib
from collections import defaultdict
import ast

# Load CSV
df = pd.read_csv('sales_data.csv')

# Filter only 'Completed' sales with valid product entries
df = df[df['status'] == 'Completed']
df = df[df['products'].notna()]

# Dictionary to accumulate sales per product
product_sales = defaultdict(lambda: {'sales_30_days': 0, 'transactions': 0})

for _, row in df.iterrows():
    try:
        products = ast.literal_eval(row['products'])
        if not isinstance(products, list):  # Skip if not a list
            continue

        for p in products:
            if 'name' in p and 'quantity' in p:
                name = p['name']
                qty = int(p['quantity'])
                product_sales[name]['sales_30_days'] += qty
                product_sales[name]['transactions'] += 1
    except Exception as e:
        print(f"❌ Error parsing row: {e}")
        continue

# Mock current stock values
stock_data = {
    name: 20 for name in product_sales.keys()  # In real case, get actual stock
}

# Build training data
data = []
for name, stats in product_sales.items():
    current_stock = stock_data.get(name, 10)
    sales_30_days = stats['sales_30_days']
    recommended_buy = max(0, sales_30_days - current_stock)

    data.append([name, current_stock, sales_30_days, recommended_buy])

df_model = pd.DataFrame(data, columns=['product', 'current_stock', 'sales_30_days', 'recommended_buy'])

# Drop products with zero sales to avoid training on irrelevant data
df_model = df_model[df_model['sales_30_days'] > 0]

# Prepare features and target
X = df_model[['current_stock', 'sales_30_days']]
y = df_model['recommended_buy']

# Train model
model = LinearRegression()
model.fit(X, y)

# Save the trained model
joblib.dump(model, 'stock_buy_model.pkl')
print("✅ Model trained and saved as stock_buy_model.pkl")
