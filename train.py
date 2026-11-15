
import pandas as pd
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# 1. Load Dataset
df = pd.read_csv("data/heart.csv")

print("Dataset Loaded Successfully!")
print("Dataset Shape:", df.shape)
print(df.head())

# 2. Define Features
features = [
    "age", "sex", "cp", "trestbps", "chol",
    "fbs", "restecg", "thalach", "exang",
    "oldpeak", "slope", "ca", "thal"
]

# 3. Check Required Columns
required_columns = features + ["target"]

missing = [col for col in required_columns if col not in df.columns]

if missing:
    raise ValueError(f"Missing columns: {missing}")

# 4. Prepare X and y
X = df[features]
y = df["target"]

# 5. Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 6. Create Model
model = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(max_iter=1000))
])

# 7. Train Model
model.fit(X_train, y_train)

print("\nModel Training Completed!")

# 8. Evaluate Model
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:", round(accuracy * 100, 2), "%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# 9. Save Model
os.makedirs("model", exist_ok=True)

joblib.dump(model, "model/heart_model.joblib")

print("\nModel Saved Successfully!")
print("Location: model/heart_model.joblib")
