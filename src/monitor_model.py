from pathlib import Path

import pandas as pd
import mlflow
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# --------------------------------------------------
# 1. Set project paths and MLflow tracking
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
MLRUNS_DIR = PROJECT_ROOT / "mlruns"

mlflow.set_tracking_uri(f"file:///{MLRUNS_DIR.as_posix()}")
mlflow.set_experiment("Customer Churn Prediction")


# --------------------------------------------------
# 2. Load production model
# --------------------------------------------------

MODEL_NAME = "CustomerChurnRandomForest"
MODEL_ALIAS = "production"

model_uri = f"models:/{MODEL_NAME}@{MODEL_ALIAS}"

print("Loading production model from:", model_uri)

model = mlflow.pyfunc.load_model(model_uri)

print("Production model loaded successfully.")


# --------------------------------------------------
# 3. Load dataset and simulate new incoming data
# --------------------------------------------------

df = pd.read_csv(DATA_PATH)

# Simulate new incoming data using last 500 customers
new_data = df.tail(500).copy()

print("New incoming data shape:", new_data.shape)


# --------------------------------------------------
# 4. Prepare actual labels
# --------------------------------------------------

y_true = new_data["Churn"].map({"No": 0, "Yes": 1})

X_new = new_data.drop("Churn", axis=1)


# --------------------------------------------------
# 5. Apply same preprocessing as training
# --------------------------------------------------

# Drop customerID
X_new = X_new.drop("customerID", axis=1)

# Convert TotalCharges to numeric
X_new["TotalCharges"] = pd.to_numeric(X_new["TotalCharges"], errors="coerce")
X_new["TotalCharges"] = X_new["TotalCharges"].fillna(0)

# One-hot encode incoming data
X_new = pd.get_dummies(X_new, drop_first=True)


# --------------------------------------------------
# 6. Recreate training columns
# --------------------------------------------------

training_df = df.drop("customerID", axis=1)

training_df["TotalCharges"] = pd.to_numeric(
    training_df["TotalCharges"],
    errors="coerce"
)

training_df["TotalCharges"] = training_df["TotalCharges"].fillna(
    training_df["TotalCharges"].median()
)

training_df["Churn"] = training_df["Churn"].map({"No": 0, "Yes": 1})

X_training = training_df.drop("Churn", axis=1)
X_training = pd.get_dummies(X_training, drop_first=True)

# Add missing columns
for col in X_training.columns:
    if col not in X_new.columns:
        X_new[col] = 0

# Reorder columns
X_new = X_new[X_training.columns]


# --------------------------------------------------
# 7. Make predictions
# --------------------------------------------------

predictions = model.predict(X_new)


# --------------------------------------------------
# 8. Calculate monitoring metrics
# --------------------------------------------------

accuracy = accuracy_score(y_true, predictions)
precision = precision_score(y_true, predictions)
recall = recall_score(y_true, predictions)
f1 = f1_score(y_true, predictions)

predicted_churn_rate = predictions.mean()
actual_churn_rate = y_true.mean()

print("\nMonitoring Results:")
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
print("Predicted Churn Rate:", predicted_churn_rate)
print("Actual Churn Rate:", actual_churn_rate)


# --------------------------------------------------
# 9. Log monitoring run to MLflow
# --------------------------------------------------

with mlflow.start_run(run_name="Production Model Monitoring"):

    mlflow.log_param("monitoring_data", "last_500_customers")
    mlflow.log_param("model_name", MODEL_NAME)
    mlflow.log_param("model_alias", MODEL_ALIAS)

    mlflow.log_metric("monitoring_accuracy", accuracy)
    mlflow.log_metric("monitoring_precision", precision)
    mlflow.log_metric("monitoring_recall", recall)
    mlflow.log_metric("monitoring_f1_score", f1)
    mlflow.log_metric("predicted_churn_rate", predicted_churn_rate)
    mlflow.log_metric("actual_churn_rate", actual_churn_rate)

print("\nMonitoring completed and logged to MLflow.")