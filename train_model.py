# train_model.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import os

# Load dataset
url = "https://drive.google.com/uc?id=1Z7RKmScBO7n9vcDIG3Xeo853Ics4QFaF"
data = pd.read_csv(url)

X = data[["R&D Spend", "Administration", "Marketing Spend"]]
y = data["Profit"]

# Scale features
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
joblib.dump(scaler, "models/scaler.pkl")

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.25, random_state=42)

# Hyperparameter tuning
knn_params = {"n_neighbors": [3, 4, 5, 6]}
extra_params = {"n_estimators": [50, 100, 150]}

knn = GridSearchCV(KNeighborsRegressor(), knn_params, cv=5)
extra = GridSearchCV(ExtraTreesRegressor(random_state=0), extra_params, cv=5)

models = {
    "Linear Regression": LinearRegression(),
    "Polynomial Regression": make_pipeline(PolynomialFeatures(2), LinearRegression()),
    "KNN": knn,
    "Extra Trees": extra
}

best_score = -np.inf
best_model = None
best_name = ""
results = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    r2 = r2_score(y_test, preds)
    results[name] = {
        "MAE": mean_absolute_error(y_test, preds),
        "MSE": mean_squared_error(y_test, preds),
        "RMSE": np.sqrt(mean_squared_error(y_test, preds)),
        "R2": r2
    }
    if r2 > best_score:
        best_score = r2
        best_model = model
        best_name = name

# Save best model
joblib.dump(best_model, "models/best_model.pkl")

# Feature importance plot
if hasattr(best_model, 'best_estimator_'):
    model_to_use = best_model.best_estimator_
else:
    model_to_use = best_model

if hasattr(model_to_use, "feature_importances_"):
    importances = model_to_use.feature_importances_
    plt.figure(figsize=(6,4))
    plt.bar(["R&D", "Admin", "Marketing"], importances, color="#00ffcc")
    plt.title("Feature Importance")
    plt.tight_layout()
    plt.savefig("static/feature_importance.png")
