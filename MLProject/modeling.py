import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import mlflow
import mlflow.sklearn
from joblib import dump
import os
from dotenv import load_dotenv

# Load environment variable dari .env
load_dotenv()

# Set DagsHub Tracking URI
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))

# Set credentials ke environment variable agar dikenali mlflow
os.environ["MLFLOW_TRACKING_USERNAME"] = os.getenv("MLFLOW_TRACKING_USERNAME")
os.environ["MLFLOW_TRACKING_PASSWORD"] = os.getenv("MLFLOW_TRACKING_PASSWORD")

# Gunakan nama eksperimen yang baru atau bersihkan sebelumnya
experiment_name = "Eksperimen_Kriteria3_Advanced"
mlflow.set_experiment(experiment_name)

# === Load data ===
data = pd.read_csv("preprocessing/preprocessed_dataset.csv")
X = data.drop("Personality", axis=1)
y = data["Personality"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# === Mulai MLflow run ===
with mlflow.start_run(run_name="RandomForest_Default"):
    mlflow.autolog()

    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="macro")
    rec = recall_score(y_test, y_pred, average="macro")
    f1 = f1_score(y_test, y_pred, average="macro")

    # Simpan model secara relatif
    dump(model, "best_model_default.pkl")

    # Log model ke DagsHub
    mlflow.sklearn.log_model(model, "model_default")

    # Log metrics
    mlflow.log_metrics({
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1
    })

    # Membuat Docker Image dari Model MLflow
    model_uri = f"runs:/{mlflow.active_run().info.run_id}/model_default"
    mlflow.models.build_docker(model_uri, "personality-model")

    # Menyimpan model ke Docker Hub
    docker_image_tag = f"{os.getenv('DOCKERHUB_USERNAME')}/personality-model:{mlflow.active_run().info.run_id}"
    mlflow.log_artifact("best_model_default.pkl", artifact_path="model_artifacts")

    print(f"Model saved as Docker image: {docker_image_tag}")

    # Menyimpan ke Docker Hub
    os.system(f"docker build -t {docker_image_tag} .")
    os.system(f"docker login -u {os.getenv('DOCKERHUB_USERNAME')} -p {os.getenv('DOCKERHUB_TOKEN')}")
    os.system(f"docker push {docker_image_tag}")
    
    print(f"Docker image pushed successfully: {docker_image_tag}")
