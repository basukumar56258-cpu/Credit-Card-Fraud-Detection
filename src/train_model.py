import os, json, joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data", "raw", "credit_card_transactions.csv")
MODEL_DIR = os.path.join(ROOT, "models")
REPORT_DIR = os.path.join(ROOT, "reports")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

df = pd.read_csv(DATA)
target = "Fraud"
X = df.drop(columns=[target, "TransactionID"])
y = (df[target] == "Yes").astype(int)

numeric = X.columns.tolist()
preprocessor = ColumnTransformer([
    ("num", Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]), numeric)
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

candidates = {
    "Logistic Regression": LogisticRegression(max_iter=2000, class_weight="balanced"),
    "Random Forest": RandomForestClassifier(
        n_estimators=300, max_depth=12, random_state=42, class_weight="balanced_subsample", n_jobs=-1
    )
}

results = {}
best_name, best_pipe, best_f1 = None, None, -1

for name, estimator in candidates.items():
    pipe = Pipeline([("preprocessor", preprocessor), ("model", estimator)])
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    proba = pipe.predict_proba(X_test)[:,1]
    metrics = {
        "accuracy": round(accuracy_score(y_test, pred), 4),
        "precision": round(precision_score(y_test, pred, zero_division=0), 4),
        "recall": round(recall_score(y_test, pred, zero_division=0), 4),
        "f1": round(f1_score(y_test, pred, zero_division=0), 4),
        "roc_auc": round(roc_auc_score(y_test, proba), 4)
    }
    results[name] = metrics
    if metrics["f1"] > best_f1:
        best_f1 = metrics["f1"]
        best_name, best_pipe = name, pipe

joblib.dump(best_pipe, os.path.join(MODEL_DIR, "fraud_detection_model.joblib"))

pred = best_pipe.predict(X_test)
proba = best_pipe.predict_proba(X_test)[:,1]
cm = confusion_matrix(y_test, pred)

plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
            xticklabels=["Legitimate","Fraud"], yticklabels=["Legitimate","Fraud"])
plt.title(f"Confusion Matrix - {best_name}")
plt.xlabel("Predicted"); plt.ylabel("Actual"); plt.tight_layout()
plt.savefig(os.path.join(REPORT_DIR, "confusion_matrix.png"), dpi=160)
plt.close()

fpr, tpr, _ = roc_curve(y_test, proba)
auc = roc_auc_score(y_test, proba)
plt.figure(figsize=(7,5))
plt.plot(fpr, tpr, label=f"ROC-AUC = {auc:.3f}")
plt.plot([0,1],[0,1],"--")
plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
plt.title(f"ROC Curve - {best_name}")
plt.legend(); plt.tight_layout()
plt.savefig(os.path.join(REPORT_DIR, "roc_curve.png"), dpi=160)
plt.close()

summary = {
    "total_transactions": int(len(df)),
    "fraud_transactions": int(y.sum()),
    "fraud_rate_percent": round(float(y.mean()*100), 2),
    "average_amount": round(float(df["Amount"].mean()), 2),
    "best_model": best_name,
    "model_metrics": results
}
with open(os.path.join(REPORT_DIR, "model_metrics.json"), "w") as f:
    json.dump(summary, f, indent=2)

print("Best model:", best_name)
print(json.dumps(results, indent=2))
print("Saved:", os.path.join(MODEL_DIR, "fraud_detection_model.joblib"))
