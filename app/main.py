
from fastapi import FastAPI
import joblib
import pandas as pd
from pathlib import Path

from app.schemas import HeartDiseaseInput

app = FastAPI(
    title="Heart Disease Prediction API",
    description="Predict heart disease using Machine Learning",
    version="1.0"
)

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "model" / "heart_model.joblib"

model = joblib.load(MODEL_PATH)

FEATURES = [
    "age", "sex", "cp", "trestbps", "chol",
    "fbs", "restecg", "thalach", "exang",
    "oldpeak", "slope", "ca", "thal"
]

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.get("/info")
def info():
    return {
        "model_type": "Logistic Regression",
        "features": FEATURES
    }

@app.post("/predict")
def predict(data: HeartDiseaseInput):

    input_data = pd.DataFrame(
        [data.model_dump()],
        columns=FEATURES
    )

    prediction = model.predict(input_data)[0]

    return {
        "heart_disease": bool(prediction == 1)
    }
