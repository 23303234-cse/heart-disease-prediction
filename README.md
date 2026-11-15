# Heart Disease Prediction Using Machine Learning

## Project Overview
This project is a Machine Learning-based Heart Disease Prediction API developed using Python, Scikit-learn, and FastAPI.

The system predicts whether a person may have heart disease based on medical attributes provided by the user.

## Project Features
- Heart Disease Prediction
- Logistic Regression Machine Learning Model
- FastAPI REST API
- Model Serialization using Joblib
- Docker Containerization
- Interactive API Documentation using Swagger UI

## Technologies Used
- Python 3.11
- Pandas
- NumPy
- Scikit-learn
- FastAPI
- Uvicorn
- Joblib
- Docker

## Dataset Information
- Dataset: Heart Disease Dataset
- Total Records: 1025
- Total Columns: 14
- Target Variable: `target`

## Model Performance

| Metric | Result |
|---|---:|
| Model | Logistic Regression |
| Accuracy | 80.98% |

## Project Structure

```text
heart-disease-prediction/
│
├── app/
│   ├── main.py
│   └── schemas.py
│
├── data/
│   └── heart.csv
│
├── model/
│   └── heart_model.joblib
│
├── .dockerignore
├── .gitignore
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── train.py
```

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Check API health |
| GET | `/info` | Get model information |
| POST | `/predict` | Predict heart disease |

## Run Locally

Install dependencies:

```bash
pip install -r requirements.txt
```

Train the model:

```bash
python train.py
```

Run the API:

```bash
uvicorn app.main:app --reload
```

Open Swagger documentation:

```text
http://127.0.0.1:8000/docs
```

## Run with Docker

Build Docker image:

```bash
docker compose build
```

Start the container:

```bash
docker compose up -d
```

Check running containers:

```bash
docker compose ps
```

Access API documentation:

```text
http://localhost:8000/docs
```

## Disclaimer
This project is developed for educational and research purposes only. It is not intended to replace professional medical diagnosis or advice.
