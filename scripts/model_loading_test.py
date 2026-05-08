import os

import dagshub
import mlflow
import mlflow.pyfunc
import pytest

from mlflow.tracking import MlflowClient


# DagsHub Authentication

dagshub.auth.add_app_token(
    token=os.getenv("DAGSHUB_TOKEN")
)


# Initialize DagsHub Repository

dagshub.init(
    repo_owner="Aryanupadhyay23",
    repo_name="Twitter-Sentiment-Analysis-MLOps",
    mlflow=True
)


# MLflow Tracking URI

mlflow.set_tracking_uri(
    "https://dagshub.com/Aryanupadhyay23/Twitter-Sentiment-Analysis-MLOps.mlflow"
)


# Test Model Loading

@pytest.mark.parametrize(
    "model_name, alias",
    [
        ("sentiment-classification-model", "staging"),
    ],
)
def test_load_latest_model(model_name, alias):
    """
    Test loading model from MLflow Model Registry
    using model alias.
    """

    try:
        # Model URI using alias
        model_uri = f"models:/{model_name}@{alias}"

        # Load model
        model = mlflow.pyfunc.load_model(model_uri)

        # Validation
        assert model is not None, "Failed to load model"

        print(
            f"\nSuccessfully loaded model:"
            f"\nModel Name : {model_name}"
            f"\nAlias      : {alias}\n"
        )

    except Exception as e:
        pytest.fail(
            f"Model loading failed.\n"
            f"Model : {model_name}\n"
            f"Alias : {alias}\n"
            f"Error : {e}"
        )