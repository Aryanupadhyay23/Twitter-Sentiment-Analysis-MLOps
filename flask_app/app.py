import re
import os

import dagshub
import joblib
import mlflow
import lightgbm

from flask import Flask, request, jsonify
from flask_cors import CORS
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


# Token-based authentication

dagshub.auth.add_app_token(
    os.getenv("DAGSHUB_TOKEN")
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


# Flask app

app = Flask(__name__)
CORS(app)


# Load preprocessing resources

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()


# Text preprocessing function

def preprocess_comment(comment: str) -> str:

    comment = re.sub(r"http\S+", "", comment)
    comment = re.sub(r"@\w+", "", comment)
    comment = re.sub(r"#\w+", "", comment)
    comment = re.sub(r"[^\w\s]", "", comment)

    comment = comment.lower()

    tokens = comment.split()

    tokens = [
        lemmatizer.lemmatize(word)
        for word in tokens
        if word not in stop_words
    ]

    return " ".join(tokens)


# Load model and artifacts

print("Loading model from MLflow registry...")

model = mlflow.sklearn.load_model(
    "models:/sentiment-classification-model@staging"
)

vectorizer = joblib.load(
    "models/artifacts/vectorizer.pkl"
)

label_encoder = joblib.load(
    "models/artifacts/label_encoder.pkl"
)

print("Model and artifacts loaded successfully!")


@app.route("/")
def home():

    return jsonify({
        "message": "Welcome to the Flask API"
    })


# Prediction endpoint

@app.route("/predict", methods=["POST"])
def predict():

    try:

        data = request.get_json()

        # Validate request body

        if not data or "comments" not in data:

            return jsonify({
                "error": "No comments provided"
            }), 400

        comments = data["comments"]

        # Convert single string to list

        if isinstance(comments, str):

            comments = [comments]

        # Validate datatype

        if not isinstance(comments, list):

            return jsonify({
                "error": "comments must be a list"
            }), 400

        # Validate empty list

        if len(comments) == 0:

            return jsonify({
                "error": "comments list is empty"
            }), 400

        # Validate comment values

        if not all(
            isinstance(comment, str)
            for comment in comments
        ):

            return jsonify({
                "error": "all comments must be strings"
            }), 400

        # Preprocess comments

        processed_comments = [
            preprocess_comment(comment)
            for comment in comments
        ]

        # Vectorize

        X = vectorizer.transform(
            processed_comments
        )

        # Predict

        predictions = model.predict(X)

        sentiments = label_encoder.inverse_transform(
            predictions
        )

        # Return response

        return jsonify({
            "predictions": sentiments.tolist()
        })

    except Exception as e:

        return jsonify({
            "error": str(e)
        }), 500


# Health endpoint

@app.route("/health", methods=["GET"])
def health():

    return jsonify({
        "status": "ok"
    })


# Run app

if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )