import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import joblib

# ✅ Load dataset
data = pd.read_csv("data/crop_yield.csv")
print("Columns in your dataset:", data.columns)

# ✅ Select correct feature columns
X = data[["Rainfall(mm)", "Temperature(C)", "Fertilizer(kg/ha)"]]
y = data["Yield(ton/ha)"]

# ✅ Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ✅ Train Model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# ✅ Save Model
os.makedirs("model", exist_ok=True)
joblib.dump(model, "model/yield_model.pkl")
print("✅ Model trained and saved successfully!")
