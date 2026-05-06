import pandas as pd
import yaml
import logging
import boto3
from pathlib import Path


# Logging configuration

logger = logging.getLogger("data_ingestion")
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

file_handler = logging.FileHandler("errors.log")
file_handler.setLevel(logging.ERROR)

formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


# Load parameters

def load_params(params_path: Path) -> dict:
    """Load parameters from YAML."""
    try:
        with params_path.open("r") as file:
            params = yaml.safe_load(file)
        logger.debug("Parameters loaded from %s", params_path)
        return params
    except Exception as e:
        logger.error("Error loading params.yaml: %s", e)
        raise

# Download data from S3

def download_from_s3(bucket_name: str, file_key: str, save_path: Path) -> None:
    """Download file from S3 and save locally."""
    try:
        s3 = boto3.client("s3")

        save_path.parent.mkdir(parents=True, exist_ok=True)

        s3.download_file(bucket_name, file_key, str(save_path))

        logger.debug(
            "File downloaded from s3://%s/%s to %s",
            bucket_name,
            file_key,
            save_path,
        )

    except Exception as e:
        logger.error("Error downloading file from S3: %s", e)
        raise


# Main

def main():
    try:
        # Define project root dynamically
        current_file = Path(__file__).resolve()
        project_root = current_file.parents[2]

        params_path = project_root / "params.yaml"
        data_dir = project_root / "data" / "raw"

        # Load parameters (optional for bucket/key config)
        params = load_params(params_path)

        bucket_name = params["data_ingestion"]["bucket_name"]
        file_key = params["data_ingestion"]["file_key"]

        # Final save location
        save_path = data_dir / "twitter_training.csv"

        # Download raw data
        download_from_s3(bucket_name, file_key, save_path)

        logger.info("Raw data ingestion completed successfully.")

    except Exception as e:
        logger.error("Data ingestion pipeline failed: %s", e)
        print(f"Error: {e}")


if __name__ == "__main__":
    main()