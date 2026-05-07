import logging
import yaml
import json
import os
import dagshub
import mlflow

from pathlib import Path


# Token-based authentication

dagshub.auth.add_app_token(
    os.getenv("MLFLOW_TRACKING_PASSWORD")
)

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

logger = logging.getLogger("model_registry")
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

console_handler.setFormatter(formatter)
logger.addHandler(console_handler)


def main():
    try:
        project_root = Path(__file__).resolve().parents[2]

        # Load experiment metadata written by model_training.py
        experiment_path = project_root / "reports" / "experiment.json"

        with open(experiment_path, "r") as f:
            experiment_data = json.load(f)

        run_id = experiment_data["run_id"]
        model_path = experiment_data["model_path"]

        # Use model_uri from experiment.json if present, else construct it
        model_uri = experiment_data.get(
            "model_uri", f"runs:/{run_id}/model"
        )

        logger.info(f"Run ID: {run_id}")
        logger.info(f"Model Path: {model_path}")
        logger.info(f"Model URI: {model_uri}")

        # Register model
        logger.info("Registering model...")

        registered_model = mlflow.register_model(
            model_uri=model_uri,
            name="sentiment-classification-model"
        )

        logger.info(
            f"Model registered: version {registered_model.version}"
        )

        # Transition to Staging
        client = mlflow.tracking.MlflowClient()

        client.set_registered_model_alias(
            name="sentiment-classification-model",
            alias="staging",
            version=registered_model.version
        )

        logger.info(
            f"Model version {registered_model.version} aliased as 'staging'."
        )

        logger.info("Model registry completed successfully.")

    except Exception as e:
        logger.error("Model registry failed: %s", e)
        raise


if __name__ == "__main__":
    main()