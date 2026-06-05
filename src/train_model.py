from pathlib import Path

import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# --------------------------------------------------
# 1. Set project paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
MLRUNS_DIR = PROJECT_ROOT / "mlruns"

# Force MLflow to always save runs in the main project mlruns folder
mlflow.set_tracking_uri(f"file:///{MLRUNS_DIR.as_posix()}")
mlflow.set_experiment("Customer Churn Prediction")


# --------------------------------------------------
# 2. Load dataset
# --------------------------------------------------

df = pd.read_csv(DATA_PATH)

print("Dataset loaded successfully.")
print("Dataset shape:", df.shape)


# --------------------------------------------------
# 3. Clean dataset
# --------------------------------------------------

# Remove customerID because it is not useful for prediction
df = df.drop("customerID", axis=1)

# Convert TotalCharges to numeric
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

# Fill missing TotalCharges values with the median
df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

# Convert target column: Churn
df["Churn"] = df["Churn"].map({"No": 0, "Yes": 1})


# --------------------------------------------------
# 4. Prepare features and target
# --------------------------------------------------

X = df.drop("Churn", axis=1)
y = df["Churn"]

# One-hot encode categorical columns
X = pd.get_dummies(X, drop_first=True)


# --------------------------------------------------
# 5. Train/test split
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# --------------------------------------------------
# 6. Create model pipeline
# --------------------------------------------------

model = Pipeline(
    steps=[
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(max_iter=2000))
    ]
)


# --------------------------------------------------
# 7. Train model and log experiment with MLflow
# --------------------------------------------------

with mlflow.start_run(run_name="Logistic Regression Baseline"):

    # Train model
    model.fit(X_train, y_train)

    # Predict
    predictions = model.predict(X_test)

    # Calculate metrics
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)

    # Print results
    print("\nModel Results:")
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)

    # Log parameters
    mlflow.log_param("model_type", "Logistic Regression")
    mlflow.log_param("max_iter", 2000)
    mlflow.log_param("test_size", 0.2)
    mlflow.log_param("random_state", 42)
    mlflow.log_param("preprocessing", "StandardScaler + OneHotEncoding")

    # Log metrics
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)

    # Log model
    mlflow.sklearn.log_model(
        sk_model=model,
        name="logistic_regression_model"
    )

print("\nExperiment completed successfully.")
print("MLflow tracking folder:", MLRUNS_DIR)