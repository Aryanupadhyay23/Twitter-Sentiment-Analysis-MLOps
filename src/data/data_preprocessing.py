import logging
import yaml
import joblib
import pandas as pd

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder


# Logging

logger = logging.getLogger("data_preprocessing")
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

console_handler.setFormatter(formatter)
logger.addHandler(console_handler)


# Load params

def load_params(path: Path) -> dict:
    with path.open("r") as f:
        return yaml.safe_load(f)


# Main Preprocessing

def main():
    try:
        project_root = Path(__file__).resolve().parents[2]

        interim_path = project_root / "data" / "interim" / "cleaned_data.csv"
        
        # Separate directories
        data_dir = project_root / "data" / "processed"
        model_dir = project_root / "models" / "artifacts"

        data_dir.mkdir(parents=True, exist_ok=True)
        model_dir.mkdir(parents=True, exist_ok=True)

        params = load_params(project_root / "params.yaml")

        text_column = params["data_preprocessing"]["text_column"]
        label_column = params["data_preprocessing"]["label_column"]
        test_size = params["data_preprocessing"]["test_size"]
        max_features = params["data_preprocessing"]["max_features"]
        ngram_range = tuple(params["data_preprocessing"]["ngram_range"])

        logger.info("Loading cleaned dataset...")
        df = pd.read_csv(interim_path)

        # FIX: handle NaN and empty text
        df[text_column] = df[text_column].fillna("")
        df = df[df[text_column].str.strip() != ""]
        df[text_column] = df[text_column].astype(str)

        # Label Encoding
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(df[label_column])

        # Train-Test Split
        X_train_text, X_test_text, y_train, y_test = train_test_split(
            df[text_column],
            y,
            test_size=test_size,
            random_state=42,
            stratify=y
        )

        # BoW Vectorization
        vectorizer = CountVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            dtype="float32"
        )

        logger.info("Applying BoW vectorization...")
        X_train = vectorizer.fit_transform(X_train_text)
        X_test = vectorizer.transform(X_test_text)

        # Save Artifacts
        logger.info("Saving processed artifacts...")

        # DATA artifacts
        joblib.dump(X_train, data_dir / "X_train.pkl", compress=3)
        joblib.dump(X_test, data_dir / "X_test.pkl", compress=3)
        joblib.dump(y_train, data_dir / "y_train.pkl", compress=3)
        joblib.dump(y_test, data_dir / "y_test.pkl", compress=3)

        # MODEL artifacts
        joblib.dump(vectorizer, model_dir / "vectorizer.pkl")
        joblib.dump(label_encoder, model_dir / "label_encoder.pkl")

        logger.info("Data preprocessing completed successfully.")

    except Exception as e:
        logger.error("Data preprocessing failed: %s", e)
        raise


if __name__ == "__main__":
    main()