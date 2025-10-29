import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
import joblib
import os

data = pd.read_csv("data/crop_yield.csv")

X = data[["Rainfall", "Temperature", "Humidity"]]
y = data["Yield"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("R2 Score:", r2_score(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))

os.makedirs("model", exist_ok=True)
joblib.dump(model, "model/yield_model.pkl")
print("✅ Model saved in model/yield_model.pkl")
