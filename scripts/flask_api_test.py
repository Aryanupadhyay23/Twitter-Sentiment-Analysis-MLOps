import pytest

from flask_app.app import app


# Flask test client

@pytest.fixture
def client():

    app.testing = True

    with app.test_client() as client:

        yield client


# Health endpoint test

def test_health_endpoint(client):

    response = client.get(
        "/health"
    )

    assert response.status_code == 200

    response_json = response.get_json()

    assert response_json["status"] == "ok"


# Predict endpoint test

def test_predict_endpoint(client):

    data = {
        "comments": [
            "This movie was amazing",
            "Worst product ever",
            "Average experience"
        ]
    }

    response = client.post(
        "/predict",
        json=data
    )

    assert response.status_code == 200

    response_json = response.get_json()

    assert "predictions" in response_json

    assert isinstance(
        response_json["predictions"],
        list
    )

    assert len(
        response_json["predictions"]
    ) == len(data["comments"])


# Single comment prediction test

def test_single_comment_prediction(client):

    data = {
        "comments": "This product is fantastic"
    }

    response = client.post(
        "/predict",
        json=data
    )

    assert response.status_code == 200

    response_json = response.get_json()

    assert "predictions" in response_json

    assert isinstance(
        response_json["predictions"],
        list
    )

    assert len(
        response_json["predictions"]
    ) == 1


# Invalid request test

def test_predict_without_comments(client):

    data = {}

    response = client.post(
        "/predict",
        json=data
    )

    assert response.status_code == 400

    response_json = response.get_json()

    assert "error" in response_json


# Invalid datatype test

def test_invalid_comment_datatype(client):

    data = {
        "comments": 12345
    }

    response = client.post(
        "/predict",
        json=data
    )

    assert response.status_code == 400

    response_json = response.get_json()

    assert "error" in response_json


# Empty list test

def test_empty_comment_list(client):

    data = {
        "comments": []
    }

    response = client.post(
        "/predict",
        json=data
    )

    assert response.status_code == 400

    response_json = response.get_json()

    assert "error" in response_json