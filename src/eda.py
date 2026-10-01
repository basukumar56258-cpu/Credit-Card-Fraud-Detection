import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
df = pd.read_csv(os.path.join(ROOT, "data", "raw", "credit_card_transactions.csv"))
REPORT = os.path.join(ROOT, "reports")
os.makedirs(REPORT, exist_ok=True)

plt.figure(figsize=(6,4))
sns.countplot(data=df, x="Fraud")
plt.title("Fraud vs Legitimate Transactions")
plt.tight_layout()
plt.savefig(os.path.join(REPORT, "fraud_distribution.png"), dpi=160)
plt.close()

plt.figure(figsize=(8,5))
sns.boxplot(data=df, x="Fraud", y="Amount")
plt.title("Transaction Amount by Fraud Status")
plt.tight_layout()
plt.savefig(os.path.join(REPORT, "amount_by_fraud.png"), dpi=160)
plt.close()

plt.figure(figsize=(9,5))
sns.histplot(data=df, x="Hour", hue="Fraud", bins=24, multiple="stack")
plt.title("Transactions by Hour")
plt.tight_layout()
plt.savefig(os.path.join(REPORT, "transactions_by_hour.png"), dpi=160)
plt.close()

print("EDA reports generated.")
