import firebase_admin
from firebase_admin import credentials, db
import pandas as pd

# 🔄 CHANGE THIS: Path to your downloaded Firebase Admin SDK JSON
cred = credentials.Certificate('firebase-adminsdk.json')

# 🔄 CHANGE THIS: Your Firebase Realtime Database URL
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://fir-d66b2-default-rtdb.asia-southeast1.firebasedatabase.app/'  # example: https://pos-retail.firebaseio.com/
})

# 🔄 CHANGE THIS: Firebase path where sales are stored
ref = db.reference('/sales')  # Adjust if your path is different, e.g., '/transactions' or '/salesData'

# Fetch and convert to list of dicts
data = ref.get()
sales = []

for key, val in data.items():
    sales.append(val)

# Convert to DataFrame
df = pd.DataFrame(sales)

# 🔄 OPTIONAL: Save CSV for training
df.to_csv('sales_data.csv', index=False)

print("Export complete. Data saved to sales_data.csv.")

