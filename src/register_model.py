from pathlib import Path

import mlflow
from mlflow.tracking import MlflowClient


# --------------------------------------------------
# 1. Set project paths and MLflow tracking
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MLRUNS_DIR = PROJECT_ROOT / "mlruns"

mlflow.set_tracking_uri(f"file:///{MLRUNS_DIR.as_posix()}")

EXPERIMENT_NAME = "Customer Churn Prediction"
REGISTERED_MODEL_NAME = "CustomerChurnRandomForest"


# --------------------------------------------------
# 2. Connect to MLflow
# --------------------------------------------------

client = MlflowClient()

experiment = client.get_experiment_by_name(EXPERIMENT_NAME)

if experiment is None:
    raise Exception(f"Experiment '{EXPERIMENT_NAME}' not found.")


# --------------------------------------------------
# 3. Find the best Random Forest run by F1 score
# --------------------------------------------------

runs = client.search_runs(
    experiment_ids=[experiment.experiment_id],
    filter_string="params.model_type = 'Random Forest'",
    order_by=["metrics.f1_score DESC"],
    max_results=1
)

if len(runs) == 0:
    raise Exception("No Random Forest runs found.")

best_run = runs[0]
best_run_id = best_run.info.run_id
best_f1 = best_run.data.metrics["f1_score"]

print("Best Random Forest Run ID:", best_run_id)
print("Best F1 Score:", best_f1)


# --------------------------------------------------
# 4. Register the best model
# --------------------------------------------------

model_uri = f"runs:/{best_run_id}/random_forest_model"

registered_model = mlflow.register_model(
    model_uri=model_uri,
    name=REGISTERED_MODEL_NAME
)

print("\nModel registered successfully.")
print("Registered Model Name:", REGISTERED_MODEL_NAME)
print("Model Version:", registered_model.version)


# --------------------------------------------------
# 5. Add simple registered model description
# --------------------------------------------------

client.update_registered_model(
    name=REGISTERED_MODEL_NAME,
    description="Random Forest model for customer churn prediction using the Telco Customer Churn dataset."
)

print("\nRegistered model description added successfully.")
print("Done.")