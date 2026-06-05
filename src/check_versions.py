from pathlib import Path

import mlflow
from mlflow.tracking import MlflowClient


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MLRUNS_DIR = PROJECT_ROOT / "mlruns"

mlflow.set_tracking_uri(f"file:///{MLRUNS_DIR.as_posix()}")

client = MlflowClient()

versions = client.search_model_versions("name='CustomerChurnRandomForest'")

print("Model versions found:")

for version in versions:
    print(
        "Name:", version.name,
        "| Version:", version.version,
        "| Stage:", version.current_stage,
        "| Aliases:", version.aliases,
    )