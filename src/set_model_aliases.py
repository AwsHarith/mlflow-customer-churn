from pathlib import Path

import mlflow
from mlflow.tracking import MlflowClient


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MLRUNS_DIR = PROJECT_ROOT / "mlruns"

mlflow.set_tracking_uri(f"file:///{MLRUNS_DIR.as_posix()}")

client = MlflowClient()

MODEL_NAME = "CustomerChurnRandomForest"

# Set version 1 as staging
client.set_registered_model_alias(
    name=MODEL_NAME,
    alias="staging",
    version="1"
)

# Set version 2 as production
client.set_registered_model_alias(
    name=MODEL_NAME,
    alias="production",
    version="2"
)

print("Model aliases updated successfully.")
print("Version 1 -> staging")
print("Version 2 -> production")