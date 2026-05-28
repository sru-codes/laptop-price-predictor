import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# 1. Create a sample dataset
data = {
    'Brand': ['Dell', 'HP', 'Apple', 'Lenovo', 'Asus', 'Dell', 'HP', 'Apple', 'Lenovo', 'Asus', 'Dell', 'HP', 'Apple', 'Lenovo', 'Asus'],
    'RAM': [8, 16, 8, 16, 32, 4, 8, 16, 8, 16, 32, 8, 16, 4, 8],
    'Storage': [256, 512, 256, 512, 1024, 128, 256, 512, 256, 512, 1024, 256, 512, 128, 256],
    'Processor_Speed': [2.5, 3.2, 3.1, 2.8, 3.5, 1.8, 2.4, 3.2, 2.6, 3.0, 3.8, 2.5, 3.4, 1.6, 2.2],
    'Price': [50000, 75000, 120000, 65000, 95000, 30000, 45000, 130000, 55000, 80000, 110000, 52000, 125000, 28000, 42000]
}

df = pd.DataFrame(data)

# 2. Preprocessing
# One-Hot Encoding for 'Brand'
df = pd.get_dummies(df, columns=['Brand'], drop_first=True)

# 3. Split data
X = df.drop('Price', axis=1)
y = df['Price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Train model
model = LinearRegression()
model.fit(X_train, y_train)

# 5. Evaluate
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")

# 6. Save model and feature list
joblib.dump(model, 'laptop_price_model.pkl')
joblib.dump(X.columns.tolist(), 'features.pkl')

print("Model saved successfully!")
