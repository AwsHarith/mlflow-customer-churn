# Customer Churn Prediction Lifecycle Management System Using MLflow

## Project Overview

This project was developed for the AIN-3009 MLOps Term Project. The goal of the project is to build an end-to-end machine learning lifecycle management system using MLflow.

The project focuses on predicting customer churn for a telecommunications company. Customer churn means that a customer stops using the company's service. The machine learning model uses customer information such as contract type, tenure, internet service, payment method, monthly charges, and total charges to predict whether a customer is likely to churn or stay.

MLflow is used to manage the full machine learning lifecycle, including experiment tracking, model training, hyperparameter tuning, model registry, model versioning, model deployment, and performance monitoring.

## Dataset

The dataset used in this project is the Telco Customer Churn dataset.

Each row represents one customer. The target column is:

Churn

The values are:

Yes = customer left the company  
No = customer stayed with the company

Main features include:

- gender
- SeniorCitizen
- Partner
- Dependents
- tenure
- PhoneService
- InternetService
- Contract
- PaymentMethod
- MonthlyCharges
- TotalCharges

## Project Structure

mlflow-customer-churn/
│
├── data/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
│
├── src/
│   ├── load_data.py
│   ├── train_model.py
│   ├── tune_model.py
│   ├── register_model.py
│   ├── check_versions.py
│   ├── set_model_aliases.py
│   ├── test_production_model.py
│   └── monitor_model.py
│
├── notebooks/
├── reports/
├── presentation/
├── requirements.txt
└── README.md

## Tools and Libraries

This project uses:

- Python 3.9
- MLflow
- Pandas
- NumPy
- Scikit-learn
- Matplotlib

## Installation

First, create and activate a Python virtual environment.

Then install the required packages:

pip install -r requirements.txt

If installing manually:

pip install mlflow pandas numpy scikit-learn matplotlib

## 1. Load Dataset

The dataset is loaded using:

python src/load_data.py

This verifies that the CSV file can be read correctly.

## 2. Train Baseline Model

The baseline model is trained using Logistic Regression:

python src/train_model.py

This script performs:

- Data loading
- Data cleaning
- Encoding categorical variables
- Train/test split
- Logistic Regression training
- MLflow parameter logging
- MLflow metric logging
- MLflow model logging

The logged metrics include:

- Accuracy
- Precision
- Recall
- F1 Score

## 3. Hyperparameter Tuning

Random Forest hyperparameter tuning is performed using:

python src/tune_model.py

The following parameters are tested:

n_estimators = 50, 100, 200  
max_depth = 5, 10, None  
min_samples_split = 2, 5

This creates 18 Random Forest runs in MLflow.

The best model was selected based on the highest F1 score.

Best model parameters:

n_estimators = 100  
max_depth = 10  
min_samples_split = 2

Best performance:

Accuracy = 0.8069  
Precision = 0.6711  
Recall = 0.5347  
F1 Score = 0.5952

## 4. Start MLflow UI

To view experiments, run:

mlflow ui --backend-store-uri .\mlruns

Then open:

http://127.0.0.1:5000

The MLflow UI shows:

- Experiment runs
- Parameters
- Metrics
- Logged models
- Model registry

## 5. Register Best Model

The best Random Forest model is registered using:

python src/register_model.py

The registered model name is:

CustomerChurnRandomForest

This creates model versions in the MLflow Model Registry.

## 6. Check Model Versions

Model versions can be checked using:

python src/check_versions.py

Example versions:

Version 1  
Version 2

## 7. Set Staging and Production Aliases

The model lifecycle is managed using aliases:

python src/set_model_aliases.py

The aliases are:

Version 1 → staging  
Version 2 → production

This demonstrates model lifecycle management using MLflow Model Registry.

## 8. Test Production Model

The production model can be loaded and tested using:

python src/test_production_model.py

The script loads the model using:

models:/CustomerChurnRandomForest@production

It then passes a sample customer to the model and returns a churn prediction.

Example output:

Prediction result: [0]  
The model predicts: This customer is likely to STAY.

## 9. Deploy Model as an API

The production model can be served as a local API using MLflow:

mlflow models serve -m "models:/CustomerChurnRandomForest@production" -p 5001 --no-conda

The API runs at:

http://127.0.0.1:5001

A prediction request can be sent to:

http://127.0.0.1:5001/invocations

Example API result:

predictions  
-----------  
{1}

This means the model predicts that the customer is likely to churn.

## 10. Monitor Production Model

Performance monitoring is simulated using the last 500 customers from the dataset:

python src/monitor_model.py

The monitoring script logs new performance metrics to MLflow.

Monitoring metrics:

Accuracy = 0.858  
Precision = 0.800  
Recall = 0.642  
F1 Score = 0.713  
Predicted Churn Rate = 0.22  
Actual Churn Rate = 0.274

This demonstrates how a deployed model can be monitored over time.

## Results Summary

The project successfully demonstrates an end-to-end machine learning lifecycle using MLflow.

Completed lifecycle steps:

- Dataset selection
- Data preprocessing
- Baseline model training
- Experiment tracking
- Hyperparameter tuning
- Best model selection
- Model registry
- Model versioning
- Staging and production aliases
- Model deployment as an API
- Production model testing
- Performance monitoring

## Conclusion

This project shows how MLflow can be used to manage the complete lifecycle of a machine learning model. By using the Telco Customer Churn dataset, the project demonstrates how models can be trained, compared, registered, deployed, and monitored in a structured MLOps workflow.

The final production model is a Random Forest classifier selected based on the highest F1 score during hyperparameter tuning. The model was registered in MLflow, promoted using aliases, deployed as a local API, and monitored using simulated incoming customer data.
