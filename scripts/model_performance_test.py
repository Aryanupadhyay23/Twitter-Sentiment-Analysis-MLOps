import os
import pickle

import dagshub
import mlflow
import mlflow.pyfunc
import pandas as pd
import pytest

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)


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

LABEL_ENCODER_PATH = "models/artifacts/label_encoder.pkl"

TEST_DATA_PATH = "data/interim/cleaned_data.csv"


# Performance Thresholds

EXPECTED_ACCURACY = 0.75
EXPECTED_PRECISION = 0.75
EXPECTED_RECALL = 0.75
EXPECTED_F1_SCORE = 0.75


# Model Performance Test

def test_model_performance():
    """
    Test model performance on holdout dataset.
    """

    try:
        # Load Model

        model_uri = f"models:/{MODEL_NAME}@{MODEL_ALIAS}"

        model = mlflow.pyfunc.load_model(model_uri)

        # Load Vectorizer

        with open(VECTORIZER_PATH, "rb") as file:
            vectorizer = pickle.load(file)

        # Load Label Encoder

        with open(LABEL_ENCODER_PATH, "rb") as file:
            label_encoder = pickle.load(file)

        # Load Dataset

        df = pd.read_csv(TEST_DATA_PATH)

        # Features

        X_test_raw = df["text"].fillna("")

        # Encode Labels

        y_test = label_encoder.transform(
            df["sentiment"]
        )

        # Vectorize Input

        X_test_vectorized = vectorizer.transform(
            X_test_raw
        )

        # Convert to DataFrame

        X_test_df = pd.DataFrame(
            X_test_vectorized.toarray(),
            columns=vectorizer.get_feature_names_out()
        )

        # Predictions

        y_pred = model.predict(X_test_df)

        # Metrics

        accuracy = accuracy_score(
            y_test,
            y_pred
        )

        precision = precision_score(
            y_test,
            y_pred,
            average="weighted",
            zero_division=1
        )

        recall = recall_score(
            y_test,
            y_pred,
            average="weighted",
            zero_division=1
        )

        f1 = f1_score(
            y_test,
            y_pred,
            average="weighted",
            zero_division=1
        )

        # Assertions

        assert accuracy >= EXPECTED_ACCURACY, (
            f"Accuracy below threshold: {accuracy:.4f}"
        )

        assert precision >= EXPECTED_PRECISION, (
            f"Precision below threshold: {precision:.4f}"
        )

        assert recall >= EXPECTED_RECALL, (
            f"Recall below threshold: {recall:.4f}"
        )

        assert f1 >= EXPECTED_F1_SCORE, (
            f"F1 Score below threshold: {f1:.4f}"
        )

        # Success Output

        print(
            f"\nModel Performance Test Passed"
            f"\nAccuracy  : {accuracy:.4f}"
            f"\nPrecision : {precision:.4f}"
            f"\nRecall    : {recall:.4f}"
            f"\nF1 Score  : {f1:.4f}\n"
        )

    except Exception as e:
        pytest.fail(
            f"Model performance test failed.\n"
            f"Error : {e}"
        )