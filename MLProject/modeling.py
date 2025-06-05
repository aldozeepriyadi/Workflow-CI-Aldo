import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import mlflow
import mlflow.sklearn
from joblib import dump
import argparse
import os

# Parsing argument dari command line
parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, default="preprocessed_dataset.csv")
args = parser.parse_args()

# === Load data ===
data = pd.read_csv(args.data_path)
X = data.drop("Personality", axis=1)
y = data["Personality"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# === Mulai MLflow run ===
with mlflow.start_run(run_name="RandomForest_Default") as run:

    # Log the run_id for reference
    run_id = run.info.run_id
    print(f"Started MLflow run with run_id: {run_id}")

    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="macro")
    rec = recall_score(y_test, y_pred, average="macro")
    f1 = f1_score(y_test, y_pred, average="macro")

    # Save model
    dump(model, "best_model_default.pkl")

    # Log model to MLflow
    mlflow.sklearn.log_model(model, "model_default")

    # Log metrics
    mlflow.log_metrics({
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1
    })

    # Ensure run_id is being logged
    mlflow.log_param("run_id", run_id)  # Log the run_id as a parameter
    print(f"MLflow run completed with run_id: {run_id}")
