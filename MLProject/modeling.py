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
        df = pd.read_csv(data_path)
        return df

    def split_X_y(self, df):
        X = df.drop(columns=['Personality'])
        y = df['Personality']
        return X, y

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)
        return self.model

    def evaluate(self, X_test, y_test):
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

    data_path = os.path.join(os.path.dirname(__file__), "preprocessing/preprocessed_dataset.csv")
    model = MLModel()
    df = model.load_data(data_path)
    X, y = model.split_X_y(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    with mlflow.start_run() as run:
        mlflow.log_param("model_type", "RandomForestClassifier")

        model.train(X_train, y_train)
        acc, prec, rec, f1 = model.evaluate(X_test, y_test)
        print(f"Model accuracy: {acc}")

        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall", rec)
        mlflow.log_metric("f1_score", f1)

        predictions = model.model.predict(X_test)
        signature = infer_signature(X_test, predictions)

        mlflow.sklearn.log_model(model.model, artifact_path="model", input_example=X_train[0:5], signature=signature)

        model.save_model("RandomForest_v3.joblib")

        run_id = run.info.run_id
        with open("run_id.txt", "w") as f:
            f.write(run_id)
        print(f"MLflow run completed with run_id: {run_id}")
