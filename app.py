# app.py

from flask import Flask, render_template, request, send_file
import joblib
import numpy as np
import pandas as pd
import os

app = Flask(__name__)

model = joblib.load("models/best_model.pkl")
scaler = joblib.load("models/scaler.pkl")

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    error = None

    if request.method == "POST":
        try:
            rd = float(request.form["rd"])
            admin = float(request.form["admin"])
            market = float(request.form["market"])
            features = np.array([[rd, admin, market]])
            scaled = scaler.transform(features)
            prediction = round(model.predict(scaled)[0], 2)
        except Exception as e:
            error = str(e)

    return render_template("index.html", prediction=prediction, error=error)

@app.route("/upload", methods=["POST"])
def upload():
    file = request.files["csv"]
    if file:
        df = pd.read_csv(file)
        features = df[["R&D Spend", "Administration", "Marketing Spend"]]
        scaled = scaler.transform(features)
        df["Predicted Profit"] = model.predict(scaled)
        output_path = "static/predicted_output.csv"
        df.to_csv(output_path, index=False)
        return send_file(output_path, as_attachment=True)
    return "Upload failed"

if __name__ == "__main__":
    app.run(debug=True)
