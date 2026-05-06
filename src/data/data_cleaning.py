import re
import string
import emoji
import yaml
import logging
import pandas as pd
import nltk

from pathlib import Path
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag


# Logging


logger = logging.getLogger("data_cleaning")
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

console_handler.setFormatter(formatter)
logger.addHandler(console_handler)


# NLTK Setup (run once manually ideally)

"""
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('wordnet')
nltk.download('omw-1.4')
"""

lemmatizer = WordNetLemmatizer()

STOP_WORDS = set(stopwords.words("english")) - {
    "not", "no", "but", "however", "yet"
}

URL_PATTERN = re.compile(r"http\S+|www\S+")
MENTION_PATTERN = re.compile(r"@\w+")
HASHTAG_PATTERN = re.compile(r"#")
REPEAT_PATTERN = re.compile(r"(.)\1{2,}")
RT_PATTERN = re.compile(r"\brt\b")
SPECIAL_CHAR_PATTERN = re.compile(r"[^a-zA-Z0-9\s!?.,']")
MULTISPACE_PATTERN = re.compile(r"\s+")

PUNCT_TABLE = str.maketrans("", "", string.punctuation)


# Helper Functions

def load_params(path: Path) -> dict:
    with path.open("r") as f:
        return yaml.safe_load(f)


def get_wordnet_pos(tag):
    if tag.startswith("J"):
        return wordnet.ADJ
    elif tag.startswith("V"):
        return wordnet.VERB
    elif tag.startswith("N"):
        return wordnet.NOUN
    elif tag.startswith("R"):
        return wordnet.ADV
    return wordnet.NOUN


def vectorized_clean_text(series):
    series = series.str.lower()
    series = series.str.replace(URL_PATTERN, "", regex=True)
    series = series.str.replace(MENTION_PATTERN, "", regex=True)
    series = series.str.replace(HASHTAG_PATTERN, "", regex=True)
    series = series.apply(emoji.demojize)
    series = series.str.replace(REPEAT_PATTERN, r"\1\1", regex=True)
    series = series.str.replace(RT_PATTERN, "", regex=True)
    series = series.str.replace(SPECIAL_CHAR_PATTERN, "", regex=True)
    series = series.str.replace(MULTISPACE_PATTERN, " ", regex=True)
    series = series.str.strip()
    return series


def remove_punctuation_series(series):
    return series.str.translate(PUNCT_TABLE)


def remove_stopwords_series(series):
    return series.apply(
        lambda text: " ".join(
            word for word in text.split()
            if word not in STOP_WORDS
        )
    )


def lemmatize_series(series):
    def lemmatize_text(text):
        words = text.split()
        tagged_words = pos_tag(words)
        lemmas = [
            lemmatizer.lemmatize(word, get_wordnet_pos(tag))
            for word, tag in tagged_words
        ]
        return " ".join(lemmas)

    return series.apply(lemmatize_text)


# Main Cleaning Stage


def main():
    try:
        project_root = Path(__file__).resolve().parents[2]

        raw_path = project_root / "data" / "raw" / "twitter_training.csv"
        interim_dir = project_root / "data" / "interim"
        interim_dir.mkdir(parents=True, exist_ok=True)

        df = pd.read_csv(raw_path)

        # Structural cleaning
        df = df[df["sentiment"] != "Irrelevant"]
        df = df.dropna()
        df = df.drop_duplicates(subset="text")

        df["sentiment"] = df["sentiment"].str.lower()

        # Text cleaning pipeline
        df["text"] = vectorized_clean_text(df["text"])
        df = df[df["text"] != ""]

        df["text"] = remove_punctuation_series(df["text"])
        df["text"] = remove_stopwords_series(df["text"])
        df["text"] = lemmatize_series(df["text"])

        # Save cleaned dataset
        output_path = interim_dir / "cleaned_data.csv"
        df.to_csv(output_path, index=False)

        logger.info("Data cleaning completed successfully.")

    except Exception as e:
        logger.error("Data cleaning failed: %s", e)
        raise


if __name__ == "__main__":
    main()