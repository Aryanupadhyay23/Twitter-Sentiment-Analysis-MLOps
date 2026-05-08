import os
import pickle

import dagshub
import mlflow
import mlflow.pyfunc
import pandas as pd
import pytest


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

MODEL_NAME = "sentiment-classification-model"

MODEL_ALIAS = "staging"

VECTORIZER_PATH = "models/artifacts/vectorizer.pkl"


# Prediction Pipeline Test

def test_model_prediction_pipeline():
    """
    Test full prediction pipeline:
    vectorizer + model inference.
    """

    try:
        # Load model
        model_uri = f"models:/{MODEL_NAME}@{MODEL_ALIAS}"

        model = mlflow.pyfunc.load_model(model_uri)

        # Load vectorizer
        with open(VECTORIZER_PATH, "rb") as file:
            vectorizer = pickle.load(file)

        # Dummy input text
        input_text = [
            "This movie was absolutely amazing"
        ]

        # Vectorize input
        transformed_input = vectorizer.transform(input_text)

        # Convert sparse matrix to dataframe
        input_df = pd.DataFrame(
            transformed_input.toarray(),
            columns=vectorizer.get_feature_names_out()
        )

        # Predict
        prediction = model.predict(input_df)

        # Assertions
        assert prediction is not None, (
            "Prediction failed"
        )

        assert len(prediction) == len(input_text), (
            "Prediction output size mismatch"
        )

        assert input_df.shape[1] == len(
            vectorizer.get_feature_names_out()
        ), "Feature mismatch detected"

        print(
            f"\nPrediction pipeline test passed."
            f"\nInput      : {input_text[0]}"
            f"\nPrediction : {prediction[0]}\n"
        )

    except Exception as e:
        pytest.fail(
            f"Prediction pipeline test failed.\n"
            f"Error : {e}"
        )