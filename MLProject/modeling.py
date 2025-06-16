import pandas as pd
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
import os
import warnings
import argparse

class MLModel:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)

    def load_data(self, data_path):
        print(f"Loading data from: {data_path}")
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data file not found: {data_path}")
        df = pd.read_csv(data_path)
        print(f"Data loaded successfully. Shape: {df.shape}")
        return df

    def split_X_y(self, df):
        if 'Personality' not in df.columns:
            raise ValueError("Column 'Personality' not found in dataset")
        X = df.drop(columns=['Personality'])
        y = df['Personality']
        print(f"Features shape: {X.shape}, Target shape: {y.shape}")
        return X, y

    def train(self, X_train, y_train):
        print("Training model...")
        self.model.fit(X_train, y_train)
        print("Model training completed")
        return self.model

    def evaluate(self, X_test, y_test):
        print("Evaluating model...")
        y_pred = self.model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average="macro")
        rec = recall_score(y_test, y_pred, average="macro")
        f1 = f1_score(y_test, y_pred, average="macro")
        print(f"Evaluation completed - Accuracy: {acc:.4f}")
        return acc, prec, rec, f1, y_pred

    def save_model(self, path):
        joblib.dump(self.model, path)
        print(f"Model saved to: {path}")

    def load_model(self, path):
        self.model = joblib.load(path)


def main(data_path=None):
    warnings.filterwarnings("ignore")
    
    # Set MLflow tracking URI
    mlflow.set_tracking_uri("file:./mlruns")
    
    # Determine data path
    if data_path is None:
        data_path = os.path.join(os.path.dirname(__file__), "preprocessing/preprocessed_dataset.csv")
    
    print(f"Starting ML pipeline with data: {data_path}")
    
    # Initialize model
    ml_model = MLModel()
    
    try:
        # Load and prepare data
        df = ml_model.load_data(data_path)
        X, y = ml_model.split_X_y(df)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Start MLflow run
        with mlflow.start_run() as run:
            print(f"MLflow run started with ID: {run.info.run_id}")
            
            # Log parameters
            mlflow.log_param("model_type", "RandomForestClassifier")
            mlflow.log_param("n_estimators", 100)
            mlflow.log_param("random_state", 42)
            mlflow.log_param("test_size", 0.2)
            mlflow.log_param("n_features", X.shape[1])
            mlflow.log_param("n_samples", X.shape[0])
            
            # Train model
            ml_model.train(X_train, y_train)
            
            # Evaluate model
            acc, prec, rec, f1, y_pred = ml_model.evaluate(X_test, y_test)
            
            # Log metrics
            mlflow.log_metric("accuracy", acc)
            mlflow.log_metric("precision", prec)
            mlflow.log_metric("recall", rec)
            mlflow.log_metric("f1_score", f1)
            
            # Create model signature
            signature = infer_signature(X_test, y_pred)
            
            # CRITICAL: Log model with proper artifact path
            mlflow.sklearn.log_model(
                sk_model=ml_model.model,
                artifact_path="model",  # This must be "model" for Docker build
                input_example=X_train.iloc[:5],  # Use iloc for proper indexing
                signature=signature,
                registered_model_name="PersonalityModel"
            )
            
            # Save additional artifacts
            model_filename = "RandomForest_v3.joblib"
            ml_model.save_model(model_filename)
            mlflow.log_artifact(model_filename)
            
            # Create and log evaluation report
            from sklearn.metrics import classification_report
            report = classification_report(y_test, y_pred)
            report_filename = "classification_report.txt"
            with open(report_filename, "w") as f:
                f.write(f"Model Performance Report\n")
                f.write(f"========================\n\n")
                f.write(f"Accuracy: {acc:.4f}\n")
                f.write(f"Precision: {prec:.4f}\n")
                f.write(f"Recall: {rec:.4f}\n")
                f.write(f"F1-Score: {f1:.4f}\n\n")
                f.write("Detailed Classification Report:\n")
                f.write(report)
            mlflow.log_artifact(report_filename)
            
            # Save run_id for GitHub Actions
            run_id = run.info.run_id
            with open("run_id.txt", "w") as f:
                f.write(run_id)
            
            print(f"\n=== Training Summary ===")
            print(f"MLflow run ID: {run_id}")
            print(f"Model accuracy: {acc:.4f}")
            print(f"Model precision: {prec:.4f}")
            print(f"Model recall: {rec:.4f}")
            print(f"Model f1-score: {f1:.4f}")
            
            # Verify model artifacts
            experiment_id = run.info.experiment_id
            artifacts_path = f"mlruns/{experiment_id}/{run_id}/artifacts/model"
            if os.path.exists(artifacts_path):
                print(f"✓ Model artifacts verified at: {artifacts_path}")
                artifacts = os.listdir(artifacts_path)
                print(f"✓ Artifact files: {artifacts}")
                
                # Check for critical files
                required_files = ['MLmodel', 'model.pkl']
                for req_file in required_files:
                    if req_file in artifacts:
                        print(f"✓ {req_file} found")
                    else:
                        print(f"✗ {req_file} missing!")
            else:
                print(f"✗ Model artifacts not found at: {artifacts_path}")
            
            print("=== Pipeline Completed Successfully ===")
            
    except Exception as e:
        print(f"Error in ML pipeline: {str(e)}")
        raise e
    finally:
        # Clean up temporary files
        temp_files = ["RandomForest_v3.joblib", "classification_report.txt"]
        for temp_file in temp_files:
            if os.path.exists(temp_file):
                os.remove(temp_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train ML Model')
    parser.add_argument('--data_path', type=str, help='Path to the dataset')
    args = parser.parse_args()
    
    main(data_path=args.data_path)