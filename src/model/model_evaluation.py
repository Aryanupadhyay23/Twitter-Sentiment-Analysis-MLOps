import logging
import yaml
import joblib
import dagshub
import mlflow
import mlflow.sklearn

import json
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
    confusion_matrix
)


# DAGSHUB + MLFLOW SETUP

dagshub.init(
    repo_owner="Aryanupadhyay23",
    repo_name="Twitter-Sentiment-Analysis-MLOps",
    mlflow=True
)

mlflow.set_tracking_uri(
    "https://dagshub.com/Aryanupadhyay23/Twitter-Sentiment-Analysis-MLOps.mlflow"
)


# LOGGING

logger = logging.getLogger("model_evaluation")
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

console_handler.setFormatter(formatter)
logger.addHandler(console_handler)


# LOAD PARAMS

def load_params(path: Path):
    with path.open("r") as f:
        return yaml.safe_load(f)


# MAIN

def main():
    try:
        project_root = Path(__file__).resolve().parents[2]

        params = load_params(project_root / "params.yaml")
        config = params["model_training"]

        data_dir = project_root / "data" / "processed"
        model_path = project_root / config["save_model_path"]
        reports_dir = project_root / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Loading data and model...")

        X_test = joblib.load(data_dir / "X_test.pkl")
        y_test = joblib.load(data_dir / "y_test.pkl")

        model = joblib.load(model_path)

        label_encoder = joblib.load(
            project_root / "models" / "artifacts" / "label_encoder.pkl"
        )

        class_names = list(label_encoder.classes_)

        # FIX dtype
        X_test = X_test.astype("float32")

        mlflow.set_experiment(config["experiment_name"])

        with mlflow.start_run():

            logger.info("Running evaluation...")

            y_pred = model.predict(X_test)

            # Core metrics
            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average="weighted")
            precision = precision_score(y_test, y_pred, average="weighted")
            recall = recall_score(y_test, y_pred, average="weighted")

            logger.info(f"Accuracy: {acc}")
            logger.info(f"F1 Score: {f1}")
            logger.info(f"Precision: {precision}")
            logger.info(f"Recall: {recall}")

            # Classification report
            report = classification_report(
                y_test,
                y_pred,
                target_names=class_names,
                output_dict=True
            )

            report_path = reports_dir / "classification_report.json"
            with open(report_path, "w") as f:
                json.dump(report, f, indent=4)

            # Confusion matrix
            cm = confusion_matrix(y_test, y_pred)

            plt.figure(figsize=(8, 6))
            sns.heatmap(
                cm,
                annot=True,
                fmt="d",
                cmap="Blues",
                xticklabels=class_names,
                yticklabels=class_names
            )
            plt.xlabel("Predicted")
            plt.ylabel("Actual")
            plt.title("Confusion Matrix")

            cm_path = reports_dir / "confusion_matrix.png"
            plt.savefig(cm_path)
            plt.close()

            # MLflow logging
            mlflow.log_metric("accuracy", acc)
            mlflow.log_metric("f1_score", f1)
            mlflow.log_metric("precision", precision)
            mlflow.log_metric("recall", recall)

            mlflow.log_artifact(str(report_path))
            mlflow.log_artifact(str(cm_path))

            logger.info("Model evaluation completed successfully.")

    except Exception as e:
        logger.error("Model evaluation failed: %s", e)
        raise


if __name__ == "__main__":
    main()