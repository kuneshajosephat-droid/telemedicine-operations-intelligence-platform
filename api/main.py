
from fastapi import FastAPI, Query
import pandas as pd
import pickle

# -----------------------------
# INITIALIZE API
# -----------------------------
app = FastAPI(title="Telehealth Forecast API")

# -----------------------------
# LOAD TRAINED MODEL
# -----------------------------
with open("models/telehealth_model.pkl", "rb") as f:
    model = pickle.load(f)

# -----------------------------
# HOME ROUTE
# -----------------------------
@app.get("/")
def home():
    return {"message": "Telehealth Forecast API is running"}

# -----------------------------
# PREDICTION ROUTE
# -----------------------------
@app.get("/predict")
def predict(
    year: int = Query(...),
    region: str = Query(...),
    total_enrollment: float = Query(...),
    telehealth_users: float = Query(...)
):
    try:
        # Calculate current rate
        calculated_input_rate = (telehealth_users / total_enrollment) * 100

        # Prepare input
        input_data = pd.DataFrame([{
            "year": year,
            "region": region,
            "total_enrollment": total_enrollment,
            "telehealth_users": telehealth_users
        }])

        # Predict
        prediction = model.predict(input_data)[0]

        return {
            "year": year,
            "region": region,
            "total_enrollment": total_enrollment,
            "telehealth_users": telehealth_users,
            "calculated_input_rate": round(calculated_input_rate, 2),
            "forecasted_telehealth_rate": round(float(prediction), 2)
        }

    except Exception as e:
        return {"error": str(e)}
