import logging
import yaml
import joblib
import dagshub
import mlflow
import mlflow.sklearn
import json
import numpy as np

from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from lightgbm import LGBMClassifier
from mlflow.models import infer_signature


# Initialize DagsHub MLflow

dagshub.init(
    repo_owner="Aryanupadhyay23",
    repo_name="Twitter-Sentiment-Analysis-MLOps",
    mlflow=True
)

mlflow.set_tracking_uri(
    "https://dagshub.com/Aryanupadhyay23/Twitter-Sentiment-Analysis-MLOps.mlflow"
)


# Configure logging

logger = logging.getLogger("model_training")
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

console_handler.setFormatter(formatter)
logger.addHandler(console_handler)


# Load YAML parameters

def load_params(path: Path):
    with path.open("r") as f:
        return yaml.safe_load(f)


# Flatten nested dictionary for MLflow logging

def flatten_dict(d, parent_key="", sep="."):
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


# Main training function

def main():
    try:
        project_root = Path(__file__).resolve().parents[2]

        params = load_params(project_root / "params.yaml")
        config = params["model_training"]
        lgbm_params = config["lightgbm"]

        data_dir = project_root / "data" / "processed"
        model_path = project_root / config["save_model_path"]
        model_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info("Loading processed data...")

        X_train = joblib.load(data_dir / "X_train.pkl")
        X_test = joblib.load(data_dir / "X_test.pkl")
        y_train = joblib.load(data_dir / "y_train.pkl")
        y_test = joblib.load(data_dir / "y_test.pkl")

        # Convert features to float32
        X_train = X_train.astype("float32")
        X_test = X_test.astype("float32")

        label_encoder = joblib.load(
            project_root / "models" / "artifacts" / "label_encoder.pkl"
        )

        num_classes = len(label_encoder.classes_)

        # Initialize LightGBM model
        model = LGBMClassifier(
            objective=lgbm_params["objective"],
            num_class=num_classes,
            n_estimators=lgbm_params["n_estimators"],
            learning_rate=lgbm_params["learning_rate"],
            max_depth=lgbm_params["max_depth"],
            num_leaves=lgbm_params["num_leaves"],
            min_child_samples=lgbm_params["min_child_samples"],
            subsample=lgbm_params["subsample"],
            colsample_bytree=lgbm_params["colsample_bytree"],
            class_weight=lgbm_params["class_weight"],
            random_state=config["random_state"],
            n_jobs=-1,
            verbosity=lgbm_params["verbosity"]
        )

        mlflow.set_experiment(config["experiment_name"])

        with mlflow.start_run(run_name="lgbm_training") as run:

            logger.info("Training LightGBM model...")
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)

            # Compute metrics
            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average="weighted")
            precision = precision_score(y_test, y_pred, average="weighted")
            recall = recall_score(y_test, y_pred, average="weighted")

            logger.info(f"Accuracy: {acc}")
            logger.info(f"F1 Score: {f1}")

            # Log parameters
            mlflow.log_params(flatten_dict(lgbm_params))

            # Log metadata
            mlflow.set_tags({
                "model": "LightGBM",
                "task": "multiclass_classification",
                "dataset": "twitter_sentiment",
                "num_classes": num_classes
            })

            # Log metrics
            mlflow.log_metrics({
                "accuracy": acc,
                "f1_score": f1,
                "precision": precision,
                "recall": recall
            })

            # Save model locally
            joblib.dump(model, model_path, compress=3)

            # Build signature
            sample_input = X_train[:5]
            signature = infer_signature(sample_input, model.predict(sample_input))

            model_info = mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path="model",
                signature=signature,
                input_example=sample_input
            )

            run_id = run.info.run_id
            model_uri = model_info.model_uri  # runs:/<run_id>/model

            logger.info(f"Run ID: {run_id}")
            logger.info(f"Model URI: {model_uri}")

            # Save experiment metadata — model_registry.py reads model_uri from here
            experiment_data = {
                "run_id": run_id,
                "model_uri": model_uri,
                "model_path": str(model_path),
                "metrics": {
                    "accuracy": acc,
                    "f1_score": f1
                }
            }

            experiment_path = project_root / "reports" / "experiment.json"
            experiment_path.parent.mkdir(parents=True, exist_ok=True)

            with open(experiment_path, "w") as f:
                json.dump(experiment_data, f, indent=4)

            mlflow.log_artifact(str(experiment_path))

            logger.info("Model training completed successfully.")

    except Exception as e:
        logger.error("Model training failed: %s", e)
        raise


if __name__ == "__main__":
    main()