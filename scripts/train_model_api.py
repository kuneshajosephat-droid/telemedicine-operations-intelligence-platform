
# ===============================
# TELEHEALTH MODEL TRAINING SCRIPT
# ===============================

import pandas as pd
import pickle
import os

from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# -----------------------------
# 1. LOAD DATA (FIXED PATH)
# -----------------------------
df = pd.read_csv("../data/telehealth_data.csv")

# Clean column names
df.columns = df.columns.str.lower().str.strip()

# -----------------------------
# 2. CLEAN DATA
# -----------------------------
df = df[df["quarter"] == "overall"].copy()

# Ensure rate is in %
if df["telehealth_rate"].max() <= 1:
    df["telehealth_rate"] = df["telehealth_rate"] * 100

# Drop missing
df = df.dropna(subset=[
    "year", "region",
    "total_enrollment",
    "telehealth_users",
    "telehealth_rate"
])

# -----------------------------
# 3. FEATURES
# -----------------------------
features = ["year", "region", "total_enrollment", "telehealth_users"]
target = "telehealth_rate"

X = df[features]
y = df[target]

# -----------------------------
# 4. PIPELINE (IMPORTANT)
# -----------------------------
preprocessor = ColumnTransformer(
    transformers=[
        ("region_encoder", OneHotEncoder(handle_unknown="ignore"), ["region"])
    ],
    remainder="passthrough"
)

model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(
        n_estimators=200,
        random_state=42,
        max_depth=5
    ))
])

# -----------------------------
# 5. TRAIN
# -----------------------------
model.fit(X, y)

# -----------------------------
# 6. SAVE MODEL (CORRECT PATH)
# -----------------------------
os.makedirs("../models", exist_ok=True)

model_path = "../models/telehealth_model.pkl"

with open(model_path, "wb") as f:
    pickle.dump(model, f)

print("✅ Model saved at:", model_path)
