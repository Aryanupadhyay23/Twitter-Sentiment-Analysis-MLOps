import requests
import pytest


# Base URL

BASE_URL = "http://localhost:5000"


# Health endpoint test

def test_health_endpoint():

    response = requests.get(
        f"{BASE_URL}/health"
    )

    assert response.status_code == 200

    response_json = response.json()

    assert response_json["status"] == "ok"


# Predict endpoint test

def test_predict_endpoint():

    data = {
        "comments": [
            "This movie was amazing",
            "Worst product ever",
            "Average experience"
        ]
    }

    response = requests.post(
        f"{BASE_URL}/predict",
        json=data
    )

    assert response.status_code == 200

    response_json = response.json()

    assert "predictions" in response_json

    assert isinstance(
        response_json["predictions"],
        list
    )

    assert len(
        response_json["predictions"]
    ) == len(data["comments"])


# Single comment prediction test

def test_single_comment_prediction():

    data = {
        "comments": "This product is fantastic"
    }

    response = requests.post(
        f"{BASE_URL}/predict",
        json=data
    )

    assert response.status_code == 200

    response_json = response.json()

    assert "predictions" in response_json

    assert isinstance(
        response_json["predictions"],
        list
    )

    assert len(
        response_json["predictions"]
    ) == 1


# Invalid request test

def test_predict_without_comments():

    data = {}

    response = requests.post(
        f"{BASE_URL}/predict",
        json=data
    )

    assert response.status_code == 400

    response_json = response.json()

    assert "error" in response_json


# Invalid datatype test

def test_invalid_comment_datatype():

    data = {
        "comments": 12345
    }

    response = requests.post(
        f"{BASE_URL}/predict",
        json=data
    )

    assert response.status_code == 400

    response_json = response.json()

    assert "error" in response_json


# Empty list test

def test_empty_comment_list():

    data = {
        "comments": []
    }

    response = requests.post(
        f"{BASE_URL}/predict",
        json=data
    )

    assert response.status_code == 400

    response_json = response.json()

    assert "error" in response_json