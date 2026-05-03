import logging
import yaml
import joblib
import dagshub
import mlflow
import mlflow.sklearn

from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score
from lightgbm import LGBMClassifier


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

logger = logging.getLogger("model_training")
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
        lgbm_params = config["lightgbm"]

        data_dir = project_root / "data" / "processed"
        model_path = project_root / config["save_model_path"]
        model_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info("Loading processed data...")

        X_train = joblib.load(data_dir / "X_train.pkl")
        X_test = joblib.load(data_dir / "X_test.pkl")
        y_train = joblib.load(data_dir / "y_train.pkl")
        y_test = joblib.load(data_dir / "y_test.pkl")

        # FIX: LightGBM needs float
        X_train = X_train.astype("float32")
        X_test = X_test.astype("float32")

        label_encoder = joblib.load(
            project_root / "models" / "artifacts" / "label_encoder.pkl"
        )

        num_classes = len(label_encoder.classes_)

        # SAME LightGBM config (unchanged)
        model = LGBMClassifier(
            objective="multiclass",
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
            verbosity=-1
        )

        mlflow.set_experiment(config["experiment_name"])

        with mlflow.start_run():

            logger.info("Training LightGBM model...")
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)

            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average="weighted")

            logger.info(f"Accuracy: {acc}")
            logger.info(f"F1 Score: {f1}")

            # Log SAME params
            mlflow.log_params(lgbm_params)

            # Log SAME metrics
            mlflow.log_metric("accuracy", acc)
            mlflow.log_metric("f1_score", f1)

            # Save model (compressed)
            joblib.dump(model, model_path, compress=3)

            # Log model
            mlflow.sklearn.log_model(model, "model")

            logger.info("Model training completed successfully.")

    except Exception as e:
        logger.error("Model training failed: %s", e)
        raise


if __name__ == "__main__":
    main()