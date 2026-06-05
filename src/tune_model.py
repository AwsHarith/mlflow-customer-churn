from pathlib import Path

import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# --------------------------------------------------
# 1. Set project paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
MLRUNS_DIR = PROJECT_ROOT / "mlruns"

mlflow.set_tracking_uri(f"file:///{MLRUNS_DIR.as_posix()}")
mlflow.set_experiment("Customer Churn Prediction")


# --------------------------------------------------
# 2. Load and clean dataset
# --------------------------------------------------

df = pd.read_csv(DATA_PATH)

df = df.drop("customerID", axis=1)

df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

df["Churn"] = df["Churn"].map({"No": 0, "Yes": 1})


# --------------------------------------------------
# 3. Prepare features and target
# --------------------------------------------------

X = df.drop("Churn", axis=1)
y = df["Churn"]

X = pd.get_dummies(X, drop_first=True)


# --------------------------------------------------
# 4. Train/test split
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# --------------------------------------------------
# 5. Hyperparameter tuning values
# --------------------------------------------------

n_estimators_options = [50, 100, 200]
max_depth_options = [5, 10, None]
min_samples_split_options = [2, 5]


best_f1 = 0
best_params = None


# --------------------------------------------------
# 6. Train multiple Random Forest models
# --------------------------------------------------

for n_estimators in n_estimators_options:
    for max_depth in max_depth_options:
        for min_samples_split in min_samples_split_options:

            run_name = (
                f"RF_n{n_estimators}_depth{max_depth}_split{min_samples_split}"
            )

            with mlflow.start_run(run_name=run_name):

                model = RandomForestClassifier(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    min_samples_split=min_samples_split,
                    random_state=42
                )

                model.fit(X_train, y_train)

                predictions = model.predict(X_test)

                accuracy = accuracy_score(y_test, predictions)
                precision = precision_score(y_test, predictions)
                recall = recall_score(y_test, predictions)
                f1 = f1_score(y_test, predictions)

                print("\nRun:", run_name)
                print("Accuracy:", accuracy)
                print("Precision:", precision)
                print("Recall:", recall)
                print("F1 Score:", f1)

                # Log parameters
                mlflow.log_param("model_type", "Random Forest")
                mlflow.log_param("n_estimators", n_estimators)
                mlflow.log_param("max_depth", max_depth)
                mlflow.log_param("min_samples_split", min_samples_split)
                mlflow.log_param("test_size", 0.2)
                mlflow.log_param("random_state", 42)

                # Log metrics
                mlflow.log_metric("accuracy", accuracy)
                mlflow.log_metric("precision", precision)
                mlflow.log_metric("recall", recall)
                mlflow.log_metric("f1_score", f1)

                # Log model
                mlflow.sklearn.log_model(
                    sk_model=model,
                    name="random_forest_model"
                )

                if f1 > best_f1:
                    best_f1 = f1
                    best_params = {
                        "n_estimators": n_estimators,
                        "max_depth": max_depth,
                        "min_samples_split": min_samples_split
                    }


print("\nHyperparameter tuning completed successfully.")
print("Best F1 Score:", best_f1)
print("Best Parameters:", best_params)