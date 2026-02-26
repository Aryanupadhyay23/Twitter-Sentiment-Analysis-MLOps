import pandas as pd
import json
import yaml
import logging
from pathlib import Path


# Logging Configuration


logger = logging.getLogger("data_validation")
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

console_handler.setFormatter(formatter)
logger.addHandler(console_handler)


# Load parameters


def load_params(params_path: Path) -> dict:
    with params_path.open("r") as f:
        return yaml.safe_load(f)


# Validation Functions

def validate_schema(df: pd.DataFrame, required_columns: list) -> list:
    return [col for col in required_columns if col not in df.columns]


def validate_missing_values(df: pd.DataFrame) -> dict:
    return df.isnull().mean().to_dict()


def validate_duplicates(df: pd.DataFrame) -> int:
    return int(df.duplicated().sum())


def validate_class_distribution(df: pd.DataFrame, label_column: str) -> dict:
    return df[label_column].value_counts(normalize=True).to_dict()


def validate_text_statistics(df: pd.DataFrame, text_column: str) -> dict:
    # Replace NaN with empty string
    text_series = df[text_column].fillna("").astype(str)

    text_lengths = text_series.apply(len)

    return {
        "min_length": int(text_lengths.min()),
        "max_length": int(text_lengths.max()),
        "mean_length": float(text_lengths.mean()),
        "empty_text_count": int((text_series.str.strip() == "").sum())
    }


# Main Validation Pipeline

def main():
    try:
        project_root = Path(__file__).resolve().parents[2]
        raw_data_path = project_root / "data" / "raw" / "twitter_training.csv"
        report_path = project_root / "reports" / "validation_report.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        params_path = project_root / "params.yaml"

        params = load_params(params_path)

        required_columns = params["data_validation"]["required_columns"]
        label_column = params["data_validation"]["label_column"]
        text_column = params["data_validation"]["text_column"]
        missing_threshold = params["data_validation"]["missing_threshold"]

        df = pd.read_csv(raw_data_path)

        report = {}

        # Schema validation
        missing_columns = validate_schema(df, required_columns)
        report["missing_columns"] = missing_columns

        # Missing value validation
        missing_percentages = validate_missing_values(df)
        report["missing_percentages"] = missing_percentages

        # Duplicate validation
        duplicate_count = validate_duplicates(df)
        report["duplicate_rows"] = duplicate_count

        # Class distribution
        class_distribution = validate_class_distribution(df, label_column)
        report["class_distribution"] = class_distribution

        # Text statistics
        text_stats = validate_text_statistics(df, text_column)
        report["text_statistics"] = text_stats

        # Save report
        with report_path.open("w") as f:
            json.dump(report, f, indent=4)

        # Hard Failure Conditions
  
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        for col, pct in missing_percentages.items():
            if pct > missing_threshold:
                raise ValueError(
                    f"Column '{col}' exceeds missing value threshold ({pct:.2%})"
                )

        logger.info("Data validation completed successfully.")

    except Exception as e:
        logger.error("Data validation failed: %s", e)
        raise


if __name__ == "__main__":
    main()