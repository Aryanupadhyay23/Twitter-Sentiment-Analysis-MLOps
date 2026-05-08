import os

import dagshub
import mlflow

from mlflow.tracking import MlflowClient


# DagsHub authentication

dagshub.auth.add_app_token(
    token=os.getenv("DAGSHUB_TOKEN")
)


# Initialize DagsHub repository

dagshub.init(
    repo_owner="Aryanupadhyay23",
    repo_name="Twitter-Sentiment-Analysis-MLOps",
    mlflow=True
)


# MLflow tracking URI

mlflow.set_tracking_uri(
    "https://dagshub.com/Aryanupadhyay23/Twitter-Sentiment-Analysis-MLOps.mlflow"
)


# Promote staging model to production

def promote_model():

    client = MlflowClient()

    model_name = "sentiment-classification-model"

    staging_alias = "staging"

    production_alias = "production"

    try:

        # Get staging model version

        staging_model = client.get_model_version_by_alias(
            name=model_name,
            alias=staging_alias
        )

        staging_version = staging_model.version

        # Remove current production alias if exists

        try:

            current_production = client.get_model_version_by_alias(
                name=model_name,
                alias=production_alias
            )

            client.delete_registered_model_alias(
                name=model_name,
                alias=production_alias
            )

            print(
                f"Removed production alias "
                f"from version {current_production.version}"
            )

        except Exception:

            print("No existing production model found")

        # Set production alias to staging model

        client.set_registered_model_alias(
            name=model_name,
            alias=production_alias,
            version=staging_version
        )

        print(
            f"\nModel promoted successfully"
            f"\nModel Name : {model_name}"
            f"\nVersion    : {staging_version}"
            f"\nAlias      : production\n"
        )

    except Exception as e:

        print(
            f"\nModel promotion failed"
            f"\nError : {e}\n"
        )


if __name__ == "__main__":

    promote_model()