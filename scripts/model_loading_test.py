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


# Test Configuration

@pytest.mark.parametrize(
    "model_name, stage",
    [
        ("sentiment-classification-model", "staging"),
    ],
)
def test_load_latest_staging_model(model_name, stage):
    """
    Test whether the latest staging model
    can be successfully loaded from
    DagsHub MLflow Model Registry.
    """

    client = MlflowClient()

    # Fetch latest model version from registry
    latest_versions = client.get_latest_versions(
        name=model_name,
        stages=[stage]
    )

    assert len(latest_versions) > 0, (
        f"No model found in '{stage}' stage "
        f"for model '{model_name}'"
    )

    latest_version = latest_versions[0].version

    try:
        # Model URI
        model_uri = f"models:/{model_name}/{latest_version}"

        # Load model
        model = mlflow.pyfunc.load_model(model_uri)

        # Validate loading
        assert model is not None, "Model loading failed"

        print(
            f"\nSuccessfully loaded model:"
            f"\nModel Name : {model_name}"
            f"\nStage      : {stage}"
            f"\nVersion    : {latest_version}\n"
        )

    except Exception as e:
        pytest.fail(
            f"Failed to load model "
            f"'{model_name}' from stage "
            f"'{stage}'. Error: {e}"
        )