from pathlib import Path

import pandas as pd
import mlflow


# --------------------------------------------------
# 1. Set project paths and MLflow tracking
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
MLRUNS_DIR = PROJECT_ROOT / "mlruns"

mlflow.set_tracking_uri(f"file:///{MLRUNS_DIR.as_posix()}")


# --------------------------------------------------
# 2. Load production model from MLflow Model Registry
# --------------------------------------------------

MODEL_NAME = "CustomerChurnRandomForest"
MODEL_ALIAS = "production"

model_uri = f"models:/{MODEL_NAME}@{MODEL_ALIAS}"

print("Loading model from:", model_uri)

model = mlflow.pyfunc.load_model(model_uri)

print("Production model loaded successfully.")


# --------------------------------------------------
# 3. Load dataset and prepare one sample customer
# --------------------------------------------------

df = pd.read_csv(DATA_PATH)

# Keep one customer for testing
sample = df.drop("Churn", axis=1).iloc[[0]]

print("\nOriginal sample customer:")
print(sample)


# --------------------------------------------------
# 4. Apply the same preprocessing used during training
# --------------------------------------------------

# Drop customerID
sample = sample.drop("customerID", axis=1)

# Convert TotalCharges to numeric
sample["TotalCharges"] = pd.to_numeric(sample["TotalCharges"], errors="coerce")
sample["TotalCharges"] = sample["TotalCharges"].fillna(0)

# One-hot encode sample
sample = pd.get_dummies(sample, drop_first=True)


# --------------------------------------------------
# 5. Match training columns
# --------------------------------------------------

# Recreate training feature columns from full dataset
training_df = df.drop("customerID", axis=1)
training_df["TotalCharges"] = pd.to_numeric(training_df["TotalCharges"], errors="coerce")
training_df["TotalCharges"] = training_df["TotalCharges"].fillna(training_df["TotalCharges"].median())
training_df["Churn"] = training_df["Churn"].map({"No": 0, "Yes": 1})

X_training = training_df.drop("Churn", axis=1)
X_training = pd.get_dummies(X_training, drop_first=True)

# Add missing columns to sample
for col in X_training.columns:
    if col not in sample.columns:
        sample[col] = 0

# Reorder columns to match training
sample = sample[X_training.columns]


# --------------------------------------------------
# 6. Make prediction
# --------------------------------------------------

prediction = model.predict(sample)

print("\nPrediction result:", prediction)

if prediction[0] == 1:
    print("The model predicts: This customer is likely to CHURN.")
else:
    print("The model predicts: This customer is likely to STAY.")