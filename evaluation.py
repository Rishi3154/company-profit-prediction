
import matplotlib.pyplot as plt
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score

data = pd.read_csv("https://drive.google.com/uc?id=1Z7RKmScBO7n9vcDIG3Xeo853Ics4QFaF")
X = data[["R&D Spend", "Administration", "Marketing Spend"]]
y = data["Profit"]

scaler = joblib.load("models/scaler.pkl")
model = joblib.load("models/best_model.pkl")

X_scaled = scaler.transform(X)
y_pred = model.predict(X_scaled)

residuals = y - y_pred

plt.figure(figsize=(6,4))
plt.scatter(y_pred, residuals, c="red", alpha=0.6)
plt.axhline(0, color='white', linestyle='--')
plt.xlabel("Predicted Profit")
plt.ylabel("Residuals")
plt.title("Outlier Detection: Residuals Plot")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("static/outliers.png")
