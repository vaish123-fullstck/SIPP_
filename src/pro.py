import pandas as pd
import joblib
from sklearn.linear_model import LinearRegression
import os

# === Paths ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "../data/synthetic_dataset_with_districts.csv")
MODEL_PATH = os.path.join(BASE_DIR, "../models/impact_model.pkl")

# === Load Dataset ===
try:
    df = pd.read_csv(DATA_PATH)

    # Features used for training
    feature_cols = ['Budget', 'Target_Audience', 'Location', 'Sustainability_Factors']
    target_col = 'Impact_Score'

    X = df[feature_cols]
    y = df[target_col]

    # === Train Model ===
    model = LinearRegression()
    model.fit(X, y)

    # === Save Model ===
    joblib.dump(model, MODEL_PATH)
    print(f"✅ Model trained and saved to: {MODEL_PATH}")

except Exception as e:
    print(f"❌ Error: {e}")
