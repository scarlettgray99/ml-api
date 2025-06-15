# train_customer_model.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load data
df = pd.read_csv('sales_data.csv')

# Extract features
df['timestamp'] = pd.to_datetime(df['timestamp'])
df['hour'] = df['timestamp'].dt.hour
df['day_of_week'] = df['timestamp'].dt.dayofweek
df['total_quantity'] = df['products'].apply(lambda p: eval(p) if pd.notnull(p) else []).apply(lambda plist: sum(item['quantity'] for item in plist) if plist else 0)

# Example label: classify light vs heavy shopper
df['label'] = df['total_quantity'].apply(lambda q: 'heavy' if q >= 3 else 'light')

# Features and target
X = df[['hour', 'day_of_week', 'total_quantity']]
y = df['label']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save model
joblib.dump(model, 'customer_model.pkl')
print("✅ Customer behavior model saved as customer_model.pkl")
