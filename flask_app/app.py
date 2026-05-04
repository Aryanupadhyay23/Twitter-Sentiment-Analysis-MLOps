import mlflow
import dagshub
from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# DagsHub + MLflow setup
dagshub.init(
    repo_owner="Aryanupadhyay23",
    repo_name="Twitter-Sentiment-Analysis-MLOps",
    mlflow=True
)

mlflow.set_tracking_uri(
    "https://dagshub.com/Aryanupadhyay23/Twitter-Sentiment-Analysis-MLOps.mlflow"
)

app = Flask(__name__)
CORS(app)

# Load global resources once at startup
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def preprocess_comment(comment: str) -> str:
    comment = re.sub(r"http\S+", "", comment)       # Remove URLs
    comment = re.sub(r"@\w+", "", comment)           # Remove mentions
    comment = re.sub(r"#\w+", "", comment)           # Remove hashtags
    comment = re.sub(r"[^\w\s]", "", comment)        # Remove punctuation
    comment = comment.lower()
    tokens = comment.split()
    tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words]
    return " ".join(tokens)

# Load model using alias syntax (MLflow 3.x compatible)
print("Loading model from MLflow registry...")
model = mlflow.sklearn.load_model(
    "models:/sentiment-classification-model@staging"
)

vectorizer = joblib.load("models/artifacts/vectorizer.pkl")
label_encoder = joblib.load("models/artifacts/label_encoder.pkl")

print("Model and artifacts loaded successfully!")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        if not data or "comment" not in data:
            return jsonify({"error": "No comment provided"}), 400

        comments = data["comment"]

        # Normalize to list (accept both single string and list)
        if isinstance(comments, str):
            comments = [comments]

        if not isinstance(comments, list):
            return jsonify({"error": "comment must be a string or list of strings"}), 400

        if not all(isinstance(c, str) for c in comments):
            return jsonify({"error": "All comments must be strings"}), 400

        # Preprocess each comment
        processed = [preprocess_comment(c) for c in comments]

        # Vectorize
        X = vectorizer.transform(processed)

        # Predict
        predictions = model.predict(X)
        sentiments = label_encoder.inverse_transform(predictions)

        return jsonify({
            "inputs": comments,
            "predictions": sentiments.tolist()
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)