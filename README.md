# Credit Card Fraud Detection

A professional end-to-end Data Science and Machine Learning portfolio project for detecting potentially fraudulent credit card transactions.

## What is included
- Synthetic, imbalanced transaction dataset (5,000 transactions)
- Data validation and preprocessing
- Exploratory Data Analysis
- Fraud-rate and transaction analytics
- Logistic Regression baseline
- Random Forest model comparison
- Class-imbalance handling with class weights
- Precision, Recall, F1 and ROC-AUC evaluation
- Confusion matrix and ROC curve reports
- Saved production ML pipeline
- Flask web dashboard
- Real-time transaction fraud-risk prediction form
- Professional VS Code-ready project structure

## Tech Stack
Python, Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn, Flask, Joblib

## Run in VS Code

### 1. Create environment
```bash
python -m venv .venv
```

Windows:
```bash
.venv\Scripts\activate
```

### 2. Install packages
```bash
pip install -r requirements.txt
```

### 3. Train model
```bash
python src/train_model.py
```

### 4. Start web application
```bash
python app.py
```

Open:
http://127.0.0.1:5000

## Project Workflow
Data -> Cleaning -> EDA -> Feature Processing -> Model Training -> Evaluation -> Saved Pipeline -> Flask Prediction API/UI

## Dataset
The included dataset is synthetic and created for educational/portfolio use. Real fraud systems require validated transaction data, stronger feature engineering, continuous monitoring, cost-sensitive thresholds, privacy controls and human/risk-team oversight.

## Important Metrics
Fraud detection should not be judged by accuracy alone because fraud is usually a minority class. This project emphasizes Precision, Recall, F1 and ROC-AUC.
