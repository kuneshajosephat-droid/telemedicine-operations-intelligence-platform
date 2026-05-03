import pickle
import pandas as pd

print("Loading model...")

with open("models/telehealth_model.pkl", "rb") as f:
    model = pickle.load(f)

print("Model loaded. Running prediction...")

samples = pd.DataFrame([
    {"year": 2025, "region": "California", "total_enrollment": 500000, "telehealth_users": 120000},
    {"year": 2025, "region": "Texas", "total_enrollment": 400000, "telehealth_users": 90000},
    {"year": 2025, "region": "New York", "total_enrollment": 600000, "telehealth_users": 150000}
])

print("Input samples:")
print(samples)

predictions = model.predict(samples)
print("Predictions:", predictions)

