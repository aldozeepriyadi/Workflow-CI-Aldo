import pandas as pd
import mlflow
from mlflow.models.signature import infer_signature
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
import os
import warnings


class MLModel:
    def __init__(self):
        self.model = RandomForestClassifier()

    def load_data(self, data_path):
        """Load preprocessed data from a CSV file."""
        df = pd.read_csv(data_path)
        return df

    def split_X_y(self, df):
        """Split the DataFrame into features and target variable."""
        X = df.drop(columns=['Personality'])
        y = df['Personality']
        return X, y

    def train(self, X_train, y_train):
        """Train the model."""
        self.model.fit(X_train, y_train)
        return self.model

    def evaluate(self, X_test, y_test):
        """Evaluate the model."""
        y_pred = self.model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average="macro")
        rec = recall_score(y_test, y_pred, average="macro")
        f1 = f1_score(y_test, y_pred, average="macro")
        return acc, prec, rec, f1

    def save_model(self, path):
        joblib.dump(self.model, path)

    def load_model(self, path):
        self.model = joblib.load(path)

if __name__ == '__main__':
    warnings.filterwarnings("ignore")

    # train and test set paths
    data_path = os.path.join(os.path.dirname(__file__), "preprocessing/preprocessed_dataset.csv")

    # model initialization
    model = MLModel()

    # load data
    df = model.load_data(data_path)

    # split features and target variable
    X, y = model.split_X_y(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Check for existing MLflow run, or start a new one
    if mlflow.active_run() is None:
        mlflow_run = mlflow.start_run()
    else:
        mlflow_run = mlflow.active_run()

    with mlflow_run:
        # Log parameters
        mlflow.log_param("model_type", "RandomForestClassifier")

        # Train and evaluate
        model.train(X_train, y_train)
        acc, prec, rec, f1 = model.evaluate(X_test, y_test)
        print(f"Model accuracy: {acc}")
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall", rec)
        mlflow.log_metric("f1_score", f1)

        # Predict once to infer signature
        predictions = model.model.predict(X_test)
        signature = infer_signature(X_test, predictions)

        # Log model artifact (in MLflow format)
        mlflow.sklearn.log_model(model.model, artifact_path="model", input_example=X_train[0:5], signature=signature)

        # Save local model
        model.save_model("RandomForest_v3.joblib")

        # Log run_id to a text file
        run_id = mlflow_run.info.run_id
        with open("run_id.txt", "w") as f:
            f.write(run_id)  # Save the run_id to the file
        print(f"MLflow run completed with run_id: {run_id}")

    # End run if we started it
    active_run = mlflow.active_run()
    if mlflow_run and (active_run is None or mlflow_run.info.run_id != active_run.info.run_id):
        mlflow.end_run()
