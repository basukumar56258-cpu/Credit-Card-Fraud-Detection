import os
import joblib
import pandas as pd
from flask import Flask, render_template, request

BASE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE, "models", "fraud_detection_model.joblib")
DATA_PATH = os.path.join(BASE, "data", "raw", "credit_card_transactions.csv")

app = Flask(__name__)
model = joblib.load(MODEL_PATH) if os.path.exists(MODEL_PATH) else None
df = pd.read_csv(DATA_PATH)

def stats():
    fraud = int((df["Fraud"] == "Yes").sum())
    return {
        "transactions": len(df),
        "fraud": fraud,
        "rate": round(fraud / len(df) * 100, 2),
        "avg_amount": round(df["Amount"].mean(), 2)
    }

@app.route("/")
def home():
    return render_template("index.html", stats=stats(), prediction=None)

@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return render_template("index.html", stats=stats(),
            prediction="Model not found. Run: python src/train_model.py")

    data = {
        "Amount": float(request.form["Amount"]),
        "Hour": int(request.form["Hour"]),
        "TimeSinceLastTransaction": int(request.form["TimeSinceLastTransaction"]),
        "V1": float(request.form["V1"]),
        "V2": float(request.form["V2"]),
        "V3": float(request.form["V3"]),
        "V4": float(request.form["V4"]),
        "V5": float(request.form["V5"]),
        "MerchantRiskScore": int(request.form["MerchantRiskScore"]),
        "DeviceTransactions": int(request.form["DeviceTransactions"])
    }
    x = pd.DataFrame([data])
    risk = float(model.predict_proba(x)[0][1]) * 100
    result = "HIGH FRAUD RISK" if risk >= 50 else "LOW FRAUD RISK"
    message = f"{result} — estimated fraud probability: {risk:.1f}%"
    return render_template("index.html", stats=stats(), prediction=message)

if __name__ == "__main__":
    app.run(debug=True)
